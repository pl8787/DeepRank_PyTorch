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

        self.pad_value = self.config['pad_value']
        self.win_size = self.config['win_size']

        self.max_match = self.config['max_match']

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=self.pad_value
        )

        self.embedding.weight.requires_grad = self.config['finetune_embed']

        self.d_avg_pool = nn.AvgPool1d(
            kernel_size=self.win_size*2+1,
            stride=self.win_size
        )

    def get_win(self, d, p):
        start = p - self.win_size
        stop = p + self.win_size + 1
        return d[start: stop]

    def process_item(self, d_item, d_p_list):
        snippet = []
        for d_p in d_p_list.tolist():
            win = self.get_win(d_item, d_p * self.win_size + self.win_size)
            snippet.append(win.unsqueeze(dim=0))
        snippet = torch.cat(snippet, dim=0)
        return snippet

    def forward(self, q_data, d_data, q_len, d_len, q_id, d_id):
        # B x Q x E  &  B x D x E
        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)

        INF = -1e6
        # B x D
        mask_d = torch.arange(self.config['d_limit'])[None, :] < d_len[:, None]
        embed_d = embed_d + ((~mask_d).type(torch.float32) * INF)[:,:,None]
        
        # B x E x Q  &  B x E x D
        embed_q_r = embed_q.permute(0, 2, 1)
        embed_d_r = embed_d.permute(0, 2, 1)

        # B x E
        vec_q = torch.sum(embed_q_r, dim=2, keepdim=False) / q_len.unsqueeze(dim=1).type(torch.float32)
        
        # B x E x D/2
        vec_d_list = self.d_avg_pool(embed_d_r)

        # B x D/2
        logit_val = torch.einsum('ix,ixk->ik', vec_q, vec_d_list)

        top_k_val, top_k_idx = torch.topk(logit_val, k=self.max_match, dim=1)
        #prob_val = F.softmax(logit_val, dim=1)

        d_snippet = []
        d_snippet_len = [[self.max_match] for i in range(len(top_k_idx))]
        for i in range(len(top_k_idx)):
            d_snippet.append(self.process_item(d_data[i], top_k_idx[i]))
        
        return q_data, d_snippet, q_len, d_snippet_len
