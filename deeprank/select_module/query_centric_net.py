from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class QueryCentricNet(select_module.SelectNet):
    def __init__(self, config):
        super().__init__(config)

        self.max_match = self.config['max_match']
        self.win_size = self.config['win_size']

        self.q_size = self.config['q_limit']
        self.d_size = self.max_match

        # key (doc_id, q_item)
        self.cache = {}

    def get_win(self, d, p):
        start = p - self.win_size
        stop = p + self.win_size + 1
        return d[start: stop]

    def process_item(self, q_item, d_item, q_item_len, d_item_len):

        # padding with -1, embedding will use the last one
        d_pad = F.pad(d_item, (self.win_size, self.win_size), value=-1)

        snippet = []
        snippet_len = [0] * q_item_len
        for d_p in range(d_item_len):
            dw = d_pad[d_p]
            win = None
            for q_p in range(q_item_len):
                if snippet_len[q_p] >= self.max_match:
                    continue
                qw = q_item[q_p]
                if dw.item() == qw.item():
                    win =  win if win is not None else self.get_win(d_pad, d_p)
                    snippet.append(win.unsqueeze(dim=0))
                    snippet_len[q_p] += 1
        snippet = torch.cat(snippet, dim=0) if len(snippet) > 0 else None

        return snippet, snippet_len


    def forward_cache(self, q_data, d_data, q_len, d_len, qid_list, did_list):

        n_item = q_data.shape[0]
        snippets = []
        snippets_len = []

        for i in range(n_item):
            key = (qid_list[i], did_list[i])
            if key not in self.cache:
                self.cache[key] = self.process_item(
                    q_data[i], d_data[i], q_len[i].item(), d_len[i].item())
            snippet, snippet_len = self.cache[key]

            if snippet is not None:
                snippets.append(snippet)
            snippets_len.append(snippet_len)

        return q_data, snippets, q_len, snippets_len

    def forward_normal(self, q_data, d_data, q_len, d_len):

        n_item = q_data.shape[0]
        snippets = []
        snippets_len = []
        for i in range(n_item):
            snippet, snippet_len = self.process_item(
                q_data[i], d_data[i], q_len[i].item(), d_len[i].item())
            if snippet is not None:
                snippets.append(snippet)
            snippets_len.append(snippet_len)
        return q_data, snippets, q_len, snippets_len


    def forward(self, *args):
        return self.forward_cache(*args)
