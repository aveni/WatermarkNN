from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import ResNet18
from trainer import test, train_steal


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 100
resume = False
parent_path = './checkpoint/mnist_parent.t7'
load_path = './checkpoint/mnist_child.t7'
save_dir = './checkpoint'
save_model = 'mnist_child.t7'
lr = 0.01
lradj = 1
max_epochs = 3
runname = 'steal'


LOG_DIR = './log'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log_' + str(runname) + '.txt')

# confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')
# # save configuration parameters
# with open(confgfile, 'w') as f:
#     for arg in vars(args):
#         f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainloader, testloader, n_classes, n_channels = getdataloader(
    'mnist', './data', './data', batch_size)


# create the model
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(load_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = ResNet18(num_classes=n_classes, num_channels=n_channels)

# Load parent
assert os.path.exists(parent_path), 'Error: no parent checkpoint found!'
parent_checkpoint = torch.load(parent_path)
parent = parent_checkpoint['net']

parent = parent.to(device)
net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    parent = torch.nn.DataParallel(parent, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


test_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# start training
for epoch in range(start_epoch, start_epoch + max_epochs):
    # adjust learning rate
    adjust_learning_rate(lr, optimizer, epoch, lradj)

    train_steal(epoch, net, parent, optimizer, logfile,
          trainloader, device)

    print("Test acc:")
    acc = test(net, test_criterion, logfile, testloader, device)

    print('Saving..')
    state = {
        'net': net.module if device is 'cuda' else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, save_model))
