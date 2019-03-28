from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _to_tensor(self, x):
        if type(x) is np.ndarray:
            return torch.from_numpy(np.int64(x))
        return x

    def forward(self, q_data, d_data, q_len, d_len):
        return q_data, d_data, q_len, d_len


class IdentityNet(SelectNet):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, q_data, d_data, q_len, d_len):
        q_data, d_data, q_len, d_len = map(self._to_tensor,
            [q_data, d_data, q_len, d_len])
        q_data = q_data[:, :self.config['q_limit']]
        d_data = d_data[:, :self.config['d_limit']]
        q_len = torch.min(q_len, torch.tensor(self.config['q_limit']))
        d_len = torch.min(d_len, torch.tensor(self.config['d_limit']))
        return q_data, d_data, q_len, d_len


class QueryCentricNet(SelectNet):
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


class PointerNet(SelectNet):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

