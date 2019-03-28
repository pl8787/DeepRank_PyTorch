from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class IdentityNet(select_module.SelectNet):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.register_buffer('q_limit', torch.tensor(self.config['q_limit']))
        self.register_buffer('d_limit', torch.tensor(self.config['d_limit']))

    def forward(self, q_data, d_data, q_len, d_len):
        q_data, d_data, q_len, d_len = map(self._to_tensor,
            [q_data, d_data, q_len, d_len])
        q_data = q_data[:, :self.q_limit.item()]
        d_data = d_data[:, :self.d_limit.item()]
        q_len = torch.min(q_len, self.q_limit)
        d_len = torch.min(d_len, self.d_limit)
        return q_data, d_data, q_len, d_len
