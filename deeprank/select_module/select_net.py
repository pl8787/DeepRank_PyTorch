from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectNet(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device

    def _to_tensor(self, x):
        if type(x) is np.ndarray:
            return torch.from_numpy(np.int64(x))
        return x

    def forward(self, q_data, d_data, q_len, d_len):
        return q_data, d_data, q_len, d_len
