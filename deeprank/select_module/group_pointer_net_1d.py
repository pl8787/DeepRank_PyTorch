from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import select_module


class GroupPointerNet(select_module.SelectNet):
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

        self.q_conv = nn.Conv1d(
            self.config['embed_dim'],
            self.config['q_rep_dim'],
            kernel_size = self.config['q_rep_kernel'],
            padding = self.config['q_rep_kernel'] // 2,
            bias = False
        )

        self.d_conv = nn.Conv1d(
            self.config['embed_dim'],
            self.config['d_rep_dim'],
            kernel_size = self.config['d_rep_kernel'],
            stride = self.config['d_rep_kernel'] // 2,
            bias = False
        )

        #torch.nn.init.constant_(self.q_conv.weight, 0.2)
        #torch.nn.init.constant_(self.d_conv.weight, 0.1)

        torch.nn.init.uniform_(self.q_conv.weight, a=0.0, b=0.5)
        torch.nn.init.uniform_(self.d_conv.weight, a=0.0, b=0.5)

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

        # B x E x Q  &  B x E x D
        embed_q_r = embed_q.permute(0, 2, 1)
        embed_d_r = embed_d.permute(0, 2, 1)

        # B x E' x Q
        vec_q = self.q_conv(embed_q_r)

        # B x E' x D/2
        vec_d = self.d_conv(embed_d_r)

        # B x Q x D/2
        logit_val = torch.einsum('ixj,ixk->ijk', vec_q, vec_d)

        # B x Q x K
        prob_val = F.softmax(logit_val, dim=2)
        self.top_k_val, self.top_k_idx = \
            torch.topk(prob_val, k=self.max_match, dim=2)

        positions = []
        positions_val = []

        snippets = []
        snippets_len = [[self.max_match] * q_len[i].item() for i in range(len(q_len))]
        for i in range(len(q_len)):
            snippet = []
            q_len_i = q_len[i].item()
            positions.append(self.top_k_idx[i][:q_len_i])
            positions_val.append(self.top_k_val[i][:q_len_i])
            for j in range(q_len_i):
                snippet.append(self.process_item(d_data[i], self.top_k_idx[i][j]))
            snippets.append(torch.cat(snippet, dim=0))
        
        if self.out_device:
            q_data = q_data.to(self.out_device)
            snippets = [snippet.to(self.out_device) for snippet in snippets]
            positions = [
                position.view(-1).to(torch.float32).to(self.out_device) for position in positions]
            positions_val = [
                position_val.view(-1) for position_val in positions_val]

        self.positions_val = positions_val
        return q_data, snippets, q_len, snippets_len, positions

    def loss(self, reward):
        reward = reward.to(self.positions_val[0].device)

        log_sum_func = lambda x: torch.log(x).sum(dim=0)
        # B
        sum_log_prob = torch.stack(list(map(log_sum_func, self.positions_val)), dim=0)
        # B/2
        pos_sum_log_prob = sum_log_prob[::2]
        neg_sum_log_prob = sum_log_prob[1::2]

        exp_reward = reward * (pos_sum_log_prob + neg_sum_log_prob)

        loss = exp_reward.mean()
        return loss
