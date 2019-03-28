from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self, config):
        super(RankNet, self).__init__()
        self.config = config
        # S: sentence, L: list, LL: list of list
        self.input_type = 'S'

    def forward(self, q_data, d_data, q_len, d_len):
        return q_data, d_data, q_len, d_len

    def pair_loss(self, x, y):
        x = x.view(-1)
        pos = x[::2]
        neg = x[1::2]
        loss = torch.mean(torch.max(1.0 + neg - pos, torch.tensor(0.0)))
        return loss


class MatchPyramidNet(RankNet):
    def __init__(self, config):
        super(MatchPyramidNet, self).__init__(config)
        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=0
        )

        self.embedding.weight.requires_grad = config['finetune_embed']

        cin = config['simmat_channel']

        self.conv_layers = []
        for cout, h, w in config['conv_params']:
            self.conv_layers.append(nn.Conv2d(cin, cout, [h, w], padding=1))
            cin = cout

        self.dpool_layer = nn.AdaptiveMaxPool2d(config['dpool_size'])

        hin = cin * config['dpool_size'][0] * config['dpool_size'][1]

        self.fc_layers = []
        for hout in config['fc_params']:
            self.fc_layers.append(nn.Linear(hin, hout))
            hin = hout

        self.out_layer = nn.Linear(hin, 1)

    def forward(self, q_data, d_data, q_len, d_len, mode=0):
        if mode == 0:
            return self.forward_v0(q_data, d_data, q_len, d_len)
        elif mode == 1:
            return self.forward_v1(q_data, d_data, q_len, d_len)
        elif mode == 2:
            return self.forward_v2(q_data, d_data, q_len, d_len)

    def forward_v0(self, q_data, d_data, q_len, d_len):
        dpool_index = self._dynamic_pooling_index(
            q_len,
            d_len,
            self.config['q_limit'],
            self.config['d_limit'])

        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q_data.shape[1], d_data.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        dpool_ret = []
        for i in range(len(dpool_index)):
            dpool_ret.append(dpool_index[i] + \
                i * self.config['q_limit'] * self.config['d_limit'])
        dpool_reidx = torch.stack(dpool_ret)
        #print(d_len[0])
        #print(o[0][0][0].shape)
        #print(o[0][0][0])
        o = o.take(dpool_reidx).view(-1, 1, 
            self.config['q_limit'], self.config['d_limit'])
        #print(o[0][0][0].shape)
        #print(o[0][0][0])
        #input()
        o = self.dpool_layer(o)

        o = o.view(-1,
            self.config['conv_params'][-1][0] * self.config['dpool_size'][0] * 
                self.config['dpool_size'][1])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))

        o = self.out_layer(o)
        return o

    def forward_v1(self, q_data, d_data, q_len, d_len):
        dpool_index = self._dynamic_pooling_index(
            q_len,
            d_len,
            self.config['q_limit'],
            self.config['d_limit'])

        embed_q = self.embedding(q_data)
        embed_d = self.embedding(d_data)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q_data.shape[1], d_data.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        #print(d_len[0])
        #print(o[0][0][0])

        dpool_ret = []
        for i in range(len(dpool_index)):
            dpool_ret.append(o[i][0].take(dpool_index[i]))

        o = torch.stack(dpool_ret)
        o = o.view(-1, 1, self.config['q_limit'], self.config['d_limit'])

        #print(o[0][0][0])
        #input()

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
            dpool_ret.append(self.dpool_layer(o[i:i+1, :, :q_len[i], :d_len[i]]))

        o = torch.cat(dpool_ret, 0)
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
            one_length_left_ = one_length_left.to(torch.float32)
            one_length_right_ = one_length_right.to(torch.float32)
            if one_length_left == 0:
                stride_left = fixed_length_left
            else:
                stride_left = 1.0 * fixed_length_left / one_length_left_
    
            if one_length_right == 0:
                stride_right = fixed_length_right
            else:
                stride_right = 1.0 * fixed_length_right / one_length_right_
    
            one_idx_left = torch.tensor([int(i / stride_left)
                            for i in range(fixed_length_left)], dtype=torch.int64)
            one_idx_right = torch.tensor([int(i / stride_right)
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


class DeepRankNet(RankNet):
    def __init__(self):
        super(DeepRankNet, self).__init__()
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
