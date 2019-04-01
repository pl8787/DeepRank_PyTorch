from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class QueryCentricNet(select_module.SelectNet):
    def __init__(self, config):
        super().__init__(config)
        self.output_type = 'LL'

        self.max_match = self.config['max_match']
        self.win_size = self.config['win_size']

        self.q_size = self.config['q_limit']
        self.d_size = self.max_match

    def get_win(self, d, p):
        start = p - self.win_size
        stop = p + self.win_size + 1
        return d[start: stop]

    def process_item(self, q_item, d_item, q_item_len, d_item_len):

        # padding with -1, embedding will use the last one
        d_pad = F.pad(d_item, (self.win_size, self.win_size), value=-1)

        snippet = torch.ones(
            [self.q_size, self.d_size, 2*self.win_size+1]).to(d_item)
        snippet_len = torch.zeros(self.q_size).to(d_item)
        for d_p in range(d_item_len):
            dw = d_pad[d_p]
            win = None
            for q_p in range(q_item_len):
                if snippet_len[q_p] >= self.max_match:
                    continue
                qw = q_item[q_p]
                if dw.item() == qw.item():
                    win =  win if win is not None else self.get_win(d_pad, d_p)
                    snippet[q_p, snippet_len[q_p]] = win
                    snippet_len[q_p] += 1

        return snippet, snippet_len

    def forward(self, q_data, d_data, q_len, d_len):

        n_item = q_data.shape[0]
        snippets = []
        snippets_len = []
        for i in range(n_item):
            snippet, snippet_len = self.process_item(
                q_data[i], d_data[i], q_len[i].item(), d_len[i].item())
            snippets.append(snippet.unsqueeze(dim=0))
            snippets_len.append(snippet_len.unsqueeze(dim=0))
        snippets = torch.cat(snippets, dim=0)
        snippets_len = torch.cat(snippets_len, dim=0)
        return q_data, snippets, q_len, snippets_len
