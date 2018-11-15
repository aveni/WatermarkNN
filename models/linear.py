'''Linear Model in PyTorch.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LinearModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(num_channels*32*32, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


def Linear(num_classes=10, num_channels=3):
    return LinearModel(num_classes, num_channels)

def test():
    net = Linear(num_classes=10, num_channels=3)
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())

# test()
