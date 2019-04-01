from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import rank_module


class DeepRankNet(rank_module.RankNet):
    def __init__(self):
        super().__init__()

    def forward(self, q_data, d_data, q_len ,d_len):
        return F.log_softmax(x, dim=1)
