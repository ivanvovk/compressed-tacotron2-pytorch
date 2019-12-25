import torch
import torch.utils.data

import random
import numpy as np

import sys
sys.path.insert(0, '../')

import layers
from audio.mel import MelTransformer
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from FLAGS import *


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads (audio,text) pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(
            self,
            audiopaths_and_text,
            config,
            shuffle=True
        ):
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, config['sort_by_length'])
        self.text_cleaners = config['text_cleaners']
        self.sampling_rate = config['sampling_rate']
        self.mel_transform_fn = MelTransformer(
            config['filter_length'], config['hop_length'], config['win_length'],
            config['n_mel_channels'], config['sampling_rate'], config['mel_fmin'],
            config['mel_fmax']
        )
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        audio = load_wav_to_torch(filename, self.sampling_rate)
        mel = self.mel_transform_fn.transform(audio.view(1, -1))
        return mel

    def get_text(self, text):
        text_norm = torch.LongTensor(text_to_sequence(text, [self.text_cleaners]))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate(object):
    """
    Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        # PAD TEXT
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # PAD MEL with extra padding for EOS
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
