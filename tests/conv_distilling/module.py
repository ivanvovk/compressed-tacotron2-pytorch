import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../tacotron2')

import torch
import torch.nn.functional as F
from tacotron2.model import Encoder


class ConvModule(torch.nn.Module):
    def __init__(self, config):
        super(ConvModule, self).__init__()
        
        self.embedding = torch.nn.Embedding(
            config["n_symbols"],
            config["symbols_embedding_dim"]
        )
        self.convolutions = Encoder(config).convolutions
        
    def forward(self, x):
        outputs = self.embedding(x).transpose(1, 2)
        for conv in self.convolutions:
            outputs = F.dropout(F.relu(conv(outputs)), 0.5, self.training)
        return outputs
    
    def infenrece(self, x):
        outputs = self.embedding(x).transpose(1, 2)
        for conv in self.convolutions:
            outputs = F.dropout(F.relu(conv(outputs)), 0.5, self.training)
        return outputs
    
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
