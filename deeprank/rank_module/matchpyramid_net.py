from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprank import rank_module


class MatchPyramidNet(rank_module.RankNet):
    def __init__(self, config):
        super().__init__(config)
        self.input_type = 'S'

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=config['pad_value']
        )

        self.embedding.weight.requires_grad = config['finetune_embed']

        cin = config['simmat_channel']

        self.conv_layers = []
        for cout, h, w in config['conv_params']:
            self.conv_layers.append(nn.Conv2d(cin, cout, [h, w], padding=1))
            cin = cout
        self.conv_sequential = nn.Sequential(*self.conv_layers)

        self.dpool_layer = nn.AdaptiveMaxPool2d(config['dpool_size'])

        hin = cin * config['dpool_size'][0] * config['dpool_size'][1]

        self.fc_layers = []
        for hout in config['fc_params']:
            self.fc_layers.append(nn.Linear(hin, hout))
            hin = hout

        self.fc_sequential = nn.Sequential(*self.fc_layers)

        self.out_layer = nn.Linear(hin, 1)

        self._dpool_cache = {}

    def forward(self, q_data, d_data, q_len, d_len, mode=0):
        if mode == 0:
            return self.forward_v0(q_data, d_data, q_len, d_len)
        elif mode == 1:
            return self.forward_v1(q_data, d_data, q_len, d_len)
        elif mode == 2:
            return self.forward_v2(q_data, d_data, q_len, d_len)
        elif mode == 3:
            return self.forward_v3(q_data, d_data, q_len, d_len)

    def forward_v0(self, q_data, d_data, q_len, d_len):
        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = simmat.view(-1, 1, q_data.shape[1], d_data.shape[1])
        o = F.relu(o)

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        dpool_index = self._dynamic_pooling_index(
            q_len,
            d_len,
            o.shape[2],
            o.shape[3])

        out_channel = self.config['conv_params'][-1][0]
        dpool_ret = []
        for i in range(len(dpool_index)):
            for j in range(out_channel):
                dpool_ret.append(dpool_index[i] + \
                    j * self.config['q_limit'] * self.config['d_limit'] + \
                    i * out_channel * self.config['q_limit'] * self.config['d_limit'])

        dpool_reidx = torch.stack(dpool_ret).to(q_data.device)

        o = o.take(dpool_reidx).view(-1, out_channel, 
            self.config['q_limit'], self.config['d_limit'])

        o = self.dpool_layer(o)

        o = o.view(-1,
            out_channel * self.config['dpool_size'][0] * 
                self.config['dpool_size'][1])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))

        o = self.out_layer(o)
        return o

    def forward_v1(self, q_data, d_data, q_len, d_len):
        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q_data.shape[1], d_data.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        dpool_index = self._dynamic_pooling_index(
            q_len,
            d_len,
            o.shape[2],
            o.shape[3])

        dpool_ret = []
        for i in range(len(dpool_index)):
            for j in range(8):
                dpool_ret.append(o[i][j].take(dpool_index[i]))

        o = torch.stack(dpool_ret)
        o = o.view(-1, 8, self.config['q_limit'], self.config['d_limit'])

        o = self.dpool_layer(o)

        o = o.view(-1,
            self.config['conv_params'][-1][0] * self.config['dpool_size'][0] * 
                self.config['dpool_size'][1])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))

        o = self.out_layer(o)
        return o

    def forward_v2(self, q_data, d_data, q_len, d_len):
        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q_data.shape[1], d_data.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        dpool_ret = []
        for i in range(len(o)):
            dpool_ret.append(
                self.dpool_layer(o[i:i+1, :, :q_len[i], :d_len[i]]))

        o = torch.cat(dpool_ret, 0)
        o = self.dpool_layer(o)

        o = o.view(-1,
            self.config['conv_params'][-1][0] * self.config['dpool_size'][0] * 
                self.config['dpool_size'][1])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))

        o = self.out_layer(o)
        return o

    def forward_v3(self, q_data, d_data, q_len, d_len):
        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q_data.shape[1], d_data.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        o = self.dpool_layer(o)

        o = o.view(-1,
            self.config['conv_params'][-1][0] * self.config['dpool_size'][0] * 
                self.config['dpool_size'][1])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))

        o = self.out_layer(o)
        return o

    def _dynamic_pooling_index(self,
                               length_left,
                               length_right,
                               fixed_length_left,
                               fixed_length_right,
                               compress_ratio_left=1,
                               compress_ratio_right=1):


        def _dpool_index(one_length_left,
                         one_length_right,
                         fixed_length_left,
                         fixed_length_right):
            key = (one_length_left.item(), one_length_right.item())
            if key in self._dpool_cache:
                return self._dpool_cache[key]
            one_length_left_ = one_length_left.to(torch.float32)
            one_length_right_ = one_length_right.to(torch.float32)
            fixed_length_left_ = fixed_length_left
            fixed_length_right_ = fixed_length_right
            #fixed_length_left_ = fixed_length_left.to(torch.float32)
            #fixed_length_right_ = fixed_length_right.to(torch.float32)
            if one_length_left == 0:
                stride_left = fixed_length_left_
            else:
                stride_left = 1.0 * fixed_length_left_ / one_length_left_

            if one_length_right == 0:
                stride_right = fixed_length_right_
            else:
                stride_right = 1.0 * fixed_length_right_ / one_length_right_

            one_idx_left = torch.tensor([
                int(i / stride_left)
                for i in range(fixed_length_left)], dtype=torch.int64)
            one_idx_right = torch.tensor([
                int(i / stride_right)
                for i in range(fixed_length_right)], dtype=torch.int64)
            mesh1, mesh2 = torch.meshgrid(one_idx_left, one_idx_right)
            index_one = (mesh1 * fixed_length_right + mesh2).view(-1)
            '''
            print(stride_left, stride_right)
            print(one_length_left, one_length_right)
            print(one_idx_left)
            print(one_idx_right)
            print(mesh1)
            print(mesh2)
            print(index_one)
            input()
            '''
            self._dpool_cache[key] = index_one
            return index_one

        index = []
        dpool_bias_left = dpool_bias_right = 0
        if fixed_length_left % compress_ratio_left != 0:
            dpool_bias_left = 1
        if fixed_length_right % compress_ratio_right != 0:
            dpool_bias_right = 1
        cur_fixed_length_left = fixed_length_left // compress_ratio_left \
            + dpool_bias_left
        cur_fixed_length_right = fixed_length_right // compress_ratio_right \
            + dpool_bias_right
        for i in range(len(length_left)):
            index.append(_dpool_index(length_left[i] // compress_ratio_left,
                                      length_right[i] // compress_ratio_right,
                                      cur_fixed_length_left,
                                      cur_fixed_length_right))
        return index

