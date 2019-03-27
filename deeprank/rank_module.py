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
        self.conv_layers = []
        for cin, cout, h, w in range(config['conv_params']):
            self.conv_layers.append(nn.Conv2d(cin, cout, h, w))
        for hidden in range(config['fc_params']):
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
