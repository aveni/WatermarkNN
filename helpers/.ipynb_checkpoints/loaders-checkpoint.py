import os

import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from torch.utils.data.sampler import SubsetRandomSampler

def _getdatatransformsdb(datatype):
    transform_train, transform_test = None, None
    if datatype.lower() == CIFAR10 or datatype.lower() == CIFAR100:
        # Data preperation
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform_train, transform_test


def getdataloader(datatype, train_db_path, test_db_path, batch_size, n_train=None):
    # get transformations
    transform_train, transform_test = _getdatatransformsdb(datatype=datatype)
    n_classes = 0
    n_channels = 0

    # Data loaders
    if datatype.lower() == CIFAR10:
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
        n_channels = 3
    elif datatype.lower() == CIFAR100:
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
        n_channels = 3

    elif datatype.lower() == MNIST:
        print("Using MNIST dataset.")
        trainset = torchvision.datasets.MNIST(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.MNIST(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
        n_channels = 1
    else:
        print("Dataset is not supported.")
        return None, None, None

    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=4)
    if n_train != None:
#         N = len(trainset)
#         subset_ix = np.random.permutation(range(N))[:n_train]
        labels = [label.item() for _, label in trainset]
        subset_ix = np.hstack([np.random.choice(np.where(labels == l)[0], n_train//10, replace=False) for l in np.unique(labels)])
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=batch_size,
                                                  num_workers=4, 
                                                  shuffle=True,
                                                  sampler=SubsetRandomSampler(subset_ix))
                
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader, n_classes, n_channels


def _getdatatransformswm():
    transform_wm = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_wm

def _getdatatransformswmmnist():
    transform_wm = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.4883,), (0.2981,)),
    ])
    return transform_wm


def getwmloader(wm_path, batch_size, mnist=False):
    transform_wm = _getdatatransformswmmnist() if mnist else _getdatatransformswm()
    # load watermark images
    wmset = datasets.ImageFolder(
        wm_path,
        transform_wm,
        lambda x: int(x)-1
    )

    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    return wmloader
