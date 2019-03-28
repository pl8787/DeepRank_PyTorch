from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self, config, device=None):
        super(RankNet, self).__init__()
        self.config = config
        # S: sentence, L: list, LL: list of list
        self.input_type = 'S'
        self.device = device

    def forward(self, q_data, d_data, q_len, d_len):
        return q_data, d_data, q_len, d_len

    def pair_loss(self, x, y):
        x = x.view(-1)
        pos = x[::2]
        neg = x[1::2]
        loss = torch.mean(torch.max(1.0 + neg - pos,
            torch.tensor(0.0).to(self.device)))
        return loss
