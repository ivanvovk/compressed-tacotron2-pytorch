import torch
import torch.utils.data

import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../tacotron2')

from utils import load_filepaths_and_text
from text import text_to_sequence
from module import ConvModule


class TextDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            audiopaths_and_text,
            config,
            shuffle=True
        ):
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, config['sort_by_length'])
        self.text_cleaners = config['text_cleaners']
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)
        self.gt_module = ConvModule(config)
        self.gt_module.load_state_dict(torch.load('conv_module.pt',
                                                  map_location=lambda storage, loc: storage))
        _ = self.gt_module.cpu().eval()
        
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        _, text = audiopath_and_text[0], audiopath_and_text[1]
        
        text = self.get_text(text)
        with torch.no_grad():
            return (text, self.gt_module(text))

    def __len__(self):
        return len(self.audiopaths_and_text)
    
    
class TextCollate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        max_input_len = max(map(len, [sample[0] for sample in batch]))

        # PAD TEXT
        text_padded = torch.LongTensor(len(batch), max_input_len).zero_()
        for i, text in enumerate([sample[0] for sample in batch]):
            text_padded[i, :text.size(0)] = text
        
        # PAD CONVOLUTIONS OUTPUTS
        targets_padded = torch.FloatTensor(len(batch), 512, max_input_len).zeros_()
        for i in range(len(idx_sorted_decreasing)):
            target = batch[ids_sorted_decreasing[i]][1]
            convs_outputs_padded[i, :, :]
            
        return text_padded, targets_padded
