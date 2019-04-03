from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # S: sentence, L: list, LL: list of list
        self.output_type = 'S'

    def forward(self, q_data, d_data, q_len, d_len, q_id, d_id):
        return q_data, d_data, q_len, d_len

    def loss(self, reward):
        return None
