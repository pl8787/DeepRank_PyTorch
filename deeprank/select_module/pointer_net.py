from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class PointerNet(select_module.SelectNet):
    def __init__(self, config, out_device=None):
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

        self.q_conv = nn.Conv2d(
            1,
            1,
            kernel_size = [self.config['q_rep_kernel'], 1],
            padding = [self.config['q_rep_kernel'] // 2, 0],
            bias = False
        )

        self.d_conv = nn.Conv2d(
            1,
            1,
            kernel_size = [self.config['d_rep_kernel'], 1],
            stride = [self.config['d_rep_kernel'] // 2, 1],
            bias = False
        )

        self.out_device = out_device

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

        # B x 1 x Q x E  &  B x 1 x D x E
        embed_q_r = embed_q.unsqueeze(dim=1)
        embed_d_r = embed_d.unsqueeze(dim=1)

        # B x 1 x Q x E
        vec_q = self.q_conv(embed_q_r)

        # B x 1 x D/2 x E
        vec_d = self.d_conv(embed_d_r)

        # B x Q x E  &  B x D/2 x E
        vec_q_r = vec_q.squeeze(dim=1)
        vec_d_r = vec_d.squeeze(dim=1)

        # B x E
        vec_q_agg = torch.sum(vec_q_r, dim=1, keepdim=False) / q_len.unsqueeze(dim=1).type(torch.float32)
        
        # B x D/2
        logit_val = torch.einsum('ix,ikx->ik', vec_q_agg, vec_d_r)

        # B x K
        prob_val = F.softmax(logit_val, dim=1)
        self.top_k_val, self.top_k_idx = \
            torch.topk(prob_val, k=self.max_match, dim=1)

        snippets = []
        snippets_len = [[self.max_match] * q_len[i].item() for i in range(len(q_len))]
        for i in range(len(q_len)):
            snippets.append(self.process_item(d_data[i], self.top_k_idx[i]))
        
        if self.out_device:
            q_data = q_data.to(self.out_device)
            snippets = [snippet.to(self.out_device) for snippet in snippets]

        return q_data, snippets, q_len, snippets_len

    def loss(self, reward):
        reward = reward.to(self.top_k_val.device)

        # B
        sum_log_prob = -torch.log(self.top_k_val).sum(dim=1)
        # B/2
        pos_sum_log_prob = sum_log_prob[::2]
        neg_sum_log_prob = sum_log_prob[1::2]

        exp_reward = reward * (pos_sum_log_prob + neg_sum_log_prob)

        loss = exp_reward.mean()
        return loss
