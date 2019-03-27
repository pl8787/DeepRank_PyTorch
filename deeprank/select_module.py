from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SelectNet(nn.Module):
    def __init__(self, config):
        super(SelectNet, self).__init__()
        self.config = config

    def forward(self, x):
        return x

class IdentityNet(SelectNet):
    def __init__(self, config):
        super(IdentityNet, self).__init__(config)
        
    def forward(self, x):
        q = x[0][:self.config['q_limit']]
        d = x[1][:self.config['d_limit']]
        return q, d

class QueryCentricNet(SelectNet):
    def __init__(self, config):
        super(QueryCentricNet, self).__init__(config)
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
    
class PointerNet(SelectNet):
    def __init__(self, config):
        super(PointerNet, self).__init__(config)
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
    
