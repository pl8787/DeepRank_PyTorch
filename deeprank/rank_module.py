from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class RankNet(nn.Module):
    def __init__(self, config):
        super(RankNet, self).__init__()
        self.config = config

    def forward(self, x):
        return x
    
class MatchPyramidNet(RankNet):
    def __init__(self, config):
        super(MatchPyramidNet, self).__init__(config)
        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=0
        )

        cin = config['simmat_channel']       

        self.conv_layers = []
        for cout, h, w in range(config['conv_params']):
            self.conv_layers.append(nn.Conv2d(cin, cout, [h, w]))
            cin = cout

        self.dpool_layer = nn.AdaptiveMaxPool2d(
            [config['dpool_h'], config['dpool_w']])

        hin = cin * config['dpool_h'] * config['dpool_w']

        self.fc_layers = []
        for hout in range(config['fc_params']):
            self.fc_layers.append(nn.Linear(hin, hout))
            hin = hout

        self.out_layer = nn.Linear(hin, 1)

    def forward(self, x):
        q = x[0]
        d = x[1]
        embed_q = self.embedding(q)
        embed_d = self.embedding(d)
        simmat = torch.einsum('ixk,iyk->ixy', embed_q, embed_d)
        o = F.relu(simmat.view(-1, 1, q.shape[1], d.shape[1]))

        for conv_op in self.conv_layers:
            o = F.relu(conv_op(o))

        o = self.dpool_layer(o)

        o = o.view(-1, config['fc_params'][0])

        for fc_op in self.fc_layers:
            o = F.relu(fc_op(o))
        
        o = self.out_layer(o)
        return o

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
