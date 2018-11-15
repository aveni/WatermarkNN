'''Lenet in PyTorch.

Implemented from https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNetModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(LeNetModel, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 100)
        self.fc2   = nn.Linear(100, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def LeNet(num_classes=10, num_channels=3):
    return LeNetModel(num_classes, num_channels)

def test():
    net = LeNet(num_classes=10, num_channels=1)
    y = net(Variable(torch.randn(1, 1, 32, 32)))
    print(y.size())

# test()
