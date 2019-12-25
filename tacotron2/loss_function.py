import torch
import numpy as np


class Tacotron2Loss(torch.nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mel_loss = None
        self.gate_loss = None

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        self.mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
            torch.nn.MSELoss()(mel_out_postnet, mel_target)
        self.gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return self.mel_loss + self.gate_loss
