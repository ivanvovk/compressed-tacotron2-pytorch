import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../tacotron2')

import torch
import torch.nn.functional as F

from math import sqrt
from tacotron2.model import Encoder
from tacotron2.layers import ConvNorm


class ConvModule(torch.nn.Module):
    def __init__(self, config):
        super(ConvModule, self).__init__()
        
        self.embedding = torch.nn.Embedding(
            config["n_symbols"],
            config["symbols_embedding_dim"]
        )
        std = sqrt(2.0 / (config["n_symbols"] + config["symbols_embedding_dim"]))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        convolutions = []
        for _ in range(config["encoder_n_convolutions"]):
            conv_layer = torch.nn.Sequential(
                ConvNorm(config["encoder_embedding_dim"],
                         config["encoder_embedding_dim"],
                         kernel_size=config["encoder_kernel_size"], stride=1,
                         padding=int((config["encoder_kernel_size"] - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                torch.nn.BatchNorm1d(config["encoder_embedding_dim"]))
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)
        
    def forward(self, x):
        outputs = self.embedding(x).transpose(1, 2)
        for conv in self.convolutions:
            outputs = F.dropout(F.relu(conv(outputs)), 0.5, self.training)
        return outputs
    
    def inference(self, x):
        outputs = self.embedding(x).transpose(1, 2)
        for conv in self.convolutions:
            outputs = F.dropout(F.relu(conv(outputs)), 0.5, self.training)
        return outputs
    
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
