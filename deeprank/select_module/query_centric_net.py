from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class QueryCentricNet(select_module.SelectNet):
    def __init__(self, config, embedding):
        super().__init__(config)

        self.max_match = self.config['max_match']
        self.win_size = self.config['win_size']

        if type(embedding) is np.ndarray:
            embedding = torch.from_mumpy(embedding)
        self.embedding = nn.Embedding.from_pretrained(embedding)

        self.fc_q = nn.Linear(
            self.embedding.embedding_dim, self.config['dim_q'])
        self.fc_d = nn.Linear(
            self.embedding.embedding_dim, self.config['dim_d'])

    def get_win(self, d, p):
        start = p - self.win_size
        stop = p + self.win_size + 1
        return d[start: stop]

    def get_fuse(self, q_embed, win_embed, q_compress, win_compress):
        q_compress_copy_shape = (
            q_compress.shape[0], win_compress.shape[0], self.config['dim_q'])
        win_compress_copy_shape = (
            q_compress.shape[0], win_compress.shape[0], self.config['dim_d'])

        q_compress_copy = q_compress.unsqueeze(1).expand(q_compress_copy_shape)
        win_compress_copy = win_compress.unsqueeze(0).expand(
            win_compress_copy_shape)
        interact = q_embed.matmul(win_embed.t())[:, :, None]

        return torch.cat(
            [q_compress_copy, win_compress_copy, interact], dim=2)

    def forward(self, q, d):
        q = torch.unique(q)
        q_embed = self.embedding(q)
        q_compress = self.fc_q(q_embed)

        d_pad = F.pad(d, (self.win_size, self.win_size), value=-1)
        snippets = defaultdict(list)
        for qw in q:
            for p, dw in enumerate(d_pad):
                if dw.item() == qw.item():
                    win = self.get_win(d_pad, p)
                    win_embed = self.embedding(win)
                    win_compress = self.fc_d(win_embed)

                    snippets[qw.item()].append(
                        self.get_fuse(
                            q_embed, win_embed, q_compress, win_compress))

                    if len(snippets[qw.item()]) > self.max_match:
                        break

        return snippets
