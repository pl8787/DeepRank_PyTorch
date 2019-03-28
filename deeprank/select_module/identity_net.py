from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class IdentityNet(select_module.SelectNet):
    def __init__(self, config, device=None):
        super().__init__(config, device)

    def forward(self, q_data, d_data, q_len, d_len):
        q_data, d_data, q_len, d_len = map(self._to_tensor,
            [q_data, d_data, q_len, d_len])
        q_data = q_data[:, :self.config['q_limit']]
        d_data = d_data[:, :self.config['d_limit']]
        q_len = torch.min(q_len, torch.tensor(self.config['q_limit']).to(self.device))
        d_len = torch.min(d_len, torch.tensor(self.config['d_limit']).to(self.device))
        return q_data, d_data, q_len, d_len
