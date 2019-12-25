import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../tacotron2')

import random

import torch
import torch.utils.data

from utils import load_filepaths_and_text
from text import text_to_sequence
from module import ConvModule


class TextDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            audiopaths_and_texts,
            config,
            shuffle=True
        ):
        self.audiopaths_and_texts = load_filepaths_and_text(
            audiopaths_and_texts, config['sort_by_length'])
        self.text_cleaners = config['text_cleaners']
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_texts)
        self.gt_module = ConvModule(config)
        self.gt_module.load_state_dict(torch.load('conv_module.pt',
                                                  map_location=lambda storage, loc: storage))
        _ = self.gt_module.cpu().eval()
        
    def get_text(self, text):
        text_norm = torch.LongTensor(text_to_sequence(text, [self.text_cleaners]))
        return text_norm

    def __getitem__(self, index):
        audiopath_and_text = self.audiopaths_and_texts[index]
        _, text = audiopath_and_text[0], audiopath_and_text[1]
        
        text = self.get_text(text)
        with torch.no_grad():
            return (text, self.gt_module(text[None]).squeeze(0))

    def __len__(self):
        return len(self.audiopaths_and_texts)
    
    
class TextCollate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        max_input_len = max(map(len, [sample[0] for sample in batch]))

        # PAD TEXT
        text_padded = torch.zeros(len(batch), max_input_len)
        for i, text in enumerate([sample[0] for sample in batch]):
            text_padded[i, :text.size(0)] = text
        
        # PAD CONVOLUTIONS OUTPUTS
        targets_padded = torch.zeros(len(batch), 512, max_input_len)
        for i, target in enumerate([sample[1] for sample in batch]):
            targets_padded[i, :target.size(0), :target.size(1)]
            
        return (text_padded, targets_padded)
