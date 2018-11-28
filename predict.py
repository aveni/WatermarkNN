from __future__ import print_function

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from helpers.loaders import *
from helpers.utils import progress_bar

parser = argparse.ArgumentParser(description='Test models on CIFAR-10 and watermark sets.')
parser.add_argument('--model_path', default='checkpoint/teacher-cifar100-2.t7', help='the model path')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
# parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--testwm', action='store_true', help='test the wm set or cifar10 dataset.')
parser.add_argument('--db_path', default='./data', help='the path to the root folder of the test data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [mnist cifar10]')
parser.add_argument('--children', action='store_true')
parser.add_argument('--train', action='store_true')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100
mnist = False
if args.dataset == 'mnist':
    mnist = True

# Data
if args.testwm:
    print('Loading watermark images')
    loader = getwmloader(args.wm_path, batch_size, mnist=mnist, shuffle=False)
else:
    train_loader, test_loader, _, _ = getdataloader(args.dataset, args.db_path, args.db_path, batch_size)
    if args.train:
        loader = train_loader
    else:
        loader = test_loader

def test_path(path):
    assert os.path.exists(path), 'Error: no checkpoint found!'
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

    #     print ("targets", targets)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
    #     print ("outputs", predicted)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

#        if args.testwm:
#            print (np.where(predicted.eq(targets.data)))

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

if args.children:
    for n in range(5):
        child_path = args.model_path+'-%d.t7'%(n+1)
        test_path(child_path)
else:
    test_path(args.model_path)
