import os
import time
import argparse
import json
import shutil

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

import math
import numpy as np

from data import TextMelDataset, TextMelCollate
from model import Tacotron2
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger

import warnings
warnings.filterwarnings('ignore')


def prepare_dataloaders(
        training_files,
        validation_files,
        n_frames_per_step,
        n_gpus
    ):
    # Get data, data loaders and collate function ready
    trainset = TextMelDataset(training_files, config)
    valset = TextMelDataset(validation_files, config)
    collate_fn = TextMelCollate(n_frames_per_step)

    train_sampler = DistributedSampler(trainset) \
        if n_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=False,
        sampler=train_sampler,
        batch_size=config['batch_size'],
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(
        output_directory,
        log_directory,
        rank
    ):
    if rank == 0:
        log_path = os.path.join(output_directory, log_directory)
        try:
            shutil.rmtree(log_path)
        except:
            pass
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o777)
        logger = Tacotron2Logger(log_path)
    else:
        logger = None
    return logger


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])

    #optimizer.load_state_dict(checkpoint_dict['optimizer'])
    #iteration = checkpoint_dict['iteration']
    return model # , optimizer, iteration


def save_checkpoint(model, optimizer, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, rank):
    """Handles all the validation scoring and printing"""
    
    model.eval()
    
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if n_gpus > 1 else None
        val_loader = DataLoader(
            valset,
            sampler=val_sampler,
            num_workers=1,        
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn
        )

        val_loss = 0.0

        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = reduce_tensor(loss.data, n_gpus).item() \
                if n_gpus > 1 else loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)
        
    model.train()

    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)
        
    return val_loss


def train(n_gpus, rank, group_name):
    if n_gpus > 1:
        if rank == 0: print('Synchronizing distributed flow...')
        init_distributed(rank, n_gpus, group_name, config['dist_config'])

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    if rank == 0: print('Initializing model, optimizer and loss...')
    model = Tacotron2(config).cuda()
    criterion = Tacotron2Loss()
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=config['weight_decay']
    )
    if config['fp16_run']:
        if rank == 0: print('Using FP16...')
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if rank == 0: print('Preparing dirs, data loaders and logger...')
    logger = prepare_directories_and_logger(
        config['output_directory'], config['log_directory'], rank
    )
    train_loader, valset, collate_fn = prepare_dataloaders(
        config['training_files'],
        config['validation_files'],
        config['n_frames_per_step'],
        n_gpus
    )

    iteration = 0
    epoch_offset = 0
    if not config['warm_up_checkpoint'] is None:
        if rank == 0: print('Loading checkpoint from {}...'.format(config['warm_up_checkpoint']))

        model = load_checkpoint(
            config['warm_up_checkpoint'], model, optimizer
        )

        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
    
    model.compress_factorize(config=config['compress_config'])
    model.train()

    # Main training loop
    for epoch in range(epoch_offset, config['epochs']):
        print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if n_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if config['fp16_run']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if iteration % config['iters_per_grad_acc'] == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['grad_clip_thresh'])

                optimizer.step()
                model.zero_grad()

                if rank == 0:
                    duration = time.perf_counter() - start
                    print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration))
                    logger.log_training(
                        reduced_loss, grad_norm, learning_rate, duration, iteration)

            if iteration % config['iters_per_validation'] == 0:
                validate(model, criterion, valset, iteration,
                    config['batch_size'], n_gpus, collate_fn, logger, rank)

            if iteration % config['iters_per_checkpoint'] == 0:
                if rank == 0:
                    checkpoint_path = os.path.join(
                        config['output_directory'], "checkpoint_{}".format(iteration)
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        iteration,
                        checkpoint_path
                    )

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--config', type=str,
                        required=False, help='configuration JSON file')
    args = parser.parse_args()
    
    global config
    config = json.load(open(args.config))
    
    torch.backends.cudnn.enabled = config['cudnn_enabled']
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']

    train(args.n_gpus, args.rank, args.group_name)
