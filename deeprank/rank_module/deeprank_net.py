from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import rank_module


class DeepRankNet(rank_module.RankNet):
    def __init__(self, config):
        super().__init__(config)
        self.input_type = 'LL'
        self.qw_embedding = nn.Embedding(
            config['vocab_size'],
            config['dim_weight'],
            padding_idx=config['pad_value']
        )

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=config['pad_value']
        )

        self.embedding.weight.requires_grad = config['finetune_embed']

        c_reduce_in, c_reduce_out = config['c_reduce']
        self.q_reduce = nn.Conv2d(
            c_reduce_in, c_reduce_out, config['k_reduce'],
            stride=config['s_reduce'], padding=config['p_reduce'])
        self.d_reduce = nn.Conv2d(
            c_reduce_in, c_reduce_out, config['k_reduce'],
            stride=config['s_reduce'], padding=config['p_reduce'])

        self.win = 2 * config['win_size'] + 1

    def forward(self, q_data, d_data, q_len ,d_len):
        n_q = q_data.shape[1]

        # B x Q -> B x Q x E -> B x 1 x Q x E
        q = self.embedding(q_data).unsqueeze(dim=1)
        # B x 1 x Q x E -> B x 1 x Q x 1 -> B x 1 x Q x W
        qr = self.q_reduce(q).expand(-1, -1, -1, self.win)

        # B x Q -> B x Q x 1
        qw = self.qw_embedding(q_data)

        n_batch = q_data.shape[0]
        for i in range(n_batch):
            n_match = d_data[i].shape[0]
            if n_match == 0:
                continue

            # M x W -> M x W x E -> M x 1 x W x E
            d_item = self.embedding(d_data[i]).unsqueeze(dim=1)
            # M x 1 x W x E -> M x 1 x W x 1 -> M x 1 x W x Q -> M x 1 x Q x W
            dr_item = self.d_reduce(
                d_item).expand(-1, -1, -1, n_q).permute(0, 1, 3, 2)

            # 1 x Q x E -> M x 1 x Q x E
            q_item = q[i].unsqueeze(dim=0).expand(n_match, -1, -1, -1)
            # 1 x Q x W -> M x 1 x Q x W
            qr_item = qr[i].unsqueeze(dim=0).expand(n_match, -1, -1, -1)

            # M x 1 x W x E -> M x 1 x E x W
            d_item = d_item.permute(0, 1, 3, 2)
            # M x 1 x Q x E, M x 1 x E x W -> M x 1 x Q x W
            inter_item = torch.einsum('miqe,miew->miqw', q_item, d_item)

            # M x 3 x Q x W
            input_tensor = torch.cat([qr_item, dr_item, inter_item], dim=1)

        return input_tensor
