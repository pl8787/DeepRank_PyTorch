from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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

        c_en_conv_in = c_reduce_in * 2 + 1
        c_en_conv_out = config['c_en_conv_out']
        self.encode = nn.Sequential(
            nn.Conv2d(
                c_en_conv_in, c_en_conv_out, config['k_en_conv'],
                stride=config['s_en_conv'], padding=config['p_en_conv']),
            nn.AdaptiveMaxPool2d(config['en_pool_out']),
            nn.LeakyReLU(config['en_leaky']))

        c_pos = 1
        self.c_gru_in = c_en_conv_out + c_pos
        self.c_gru_hidden = config['dim_gru_hidden']
        self.gru = nn.GRU(
            self.c_gru_in, self.c_gru_hidden, bidirectional=True)
        gru_pool_out = 1
        self.pool = nn.AdaptiveMaxPool1d(gru_pool_out)

        self.fc = nn.Linear(self.c_gru_hidden*2, 1)

    def group_match_by_q(self, conv_out, d_item_len):
        # list of QM x 5 with length Q
        raw_group = conv_out.split(d_item_len)
        # MaxQM x Q x 5
        group = pad_sequence(raw_group)
        return group


    def forward(self, q_data, d_data, q_len ,d_len, d_pos):
        n_q = q_data.shape[1]

        # B x Q -> B x Q x E -> B x 1 x Q x E
        q = self.embedding(q_data).unsqueeze(dim=1)
        # B x 1 x Q x E -> B x 1 x Q x 1 -> B x 1 x Q x W
        qr = self.q_reduce(q).expand(-1, -1, -1, self.win)

        # B x Q -> B x Q x 1 -> B x Q
        qw = self.qw_embedding(q_data).squeeze(2)

        n_batch = q_data.shape[0]
        out = []
        for i in range(n_batch):
            n_match = d_data[i].shape[0]
            if n_match == 0:
                out.append(torch.zeros(1, self.c_gru_hidden*2).to(qw.device))
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

            # M x 4 x 1 x 1 -> M x 4
            o = self.encode(input_tensor).squeeze(2).squeeze(2)
            # M x 1
            pos = d_pos[i][:, None]
            # M x 5
            o = torch.cat([o, 1.0/(pos+1.0)], dim=1)
            # MaxQM x Q x 5
            o = self.group_match_by_q(o, d_len[i])
            # MaxQM x Q x 6 -> Q x 6 x MaxQM
            o = self.gru(o)[0].permute([1, 2, 0])
            # Q x 6 x 1 -> Q x 6
            o = self.pool(o).squeeze(2)
            # 1 x 6
            o = qw[i][:o.shape[0]].matmul(o).unsqueeze(0)
            out.append(o)

        # B x 6
        out = torch.cat(out, dim=0)
        # B x 6 -> B x 1
        out = self.fc(out)

        return out
