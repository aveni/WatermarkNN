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
from models import ResNet18, LeNet, Linear
from trainer import test, train_steal


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 100
resume = False

# np.random.seed(2)
lr = 0.1
lradj = 30
max_epochs = 90
runname = 'steal'
model = ResNet18
n_train = 1000
grad_query = False
dataset='cifar10'
save_dir = './checkpoint'
parent_name = 'resnet-cifar'
parent_path = save_dir + '/' + parent_name + '.t7'

for t in range(5):
    save_model = parent_name + '-%s-%d-child-%d.t7' % ('grad' if grad_query else 'mem', n_train, t+1)
    load_path = save_dir + '/' + save_model

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
        dataset, './data', './data', batch_size, n_train=n_train)


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
        net = model(num_classes=n_classes, num_channels=n_channels)

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
              trainloader, device, grad_query=grad_query)

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
