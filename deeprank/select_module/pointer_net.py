from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class PointerNet(select_module.SelectNet):
    def __init__(self, config):
        super().__init__(config)
        self.output_type = 'LL'

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=0
        )

        self.embedding.weight.requires_grad = config['finetune_embed']

        self.avg_pool_layer = nn.AvgPool1d(kernel_size=, stride=)



    def forward(self, q_data, d_data, q_len, d_len):

