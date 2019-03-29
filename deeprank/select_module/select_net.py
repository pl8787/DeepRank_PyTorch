from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, q_data, d_data, q_len, d_len):
        return q_data, d_data, q_len, d_len
