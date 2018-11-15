import numpy as np
import torch
from torch.autograd import Variable

from helpers.utils import progress_bar


def train_steal(epoch, net, parent, optimizer, logfile, loader, device, grad_query=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    parent.eval()
    train_loss = 0
    progress_mem_loss = 0
    progress_grad_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    pseudo_label_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss(size_average=False)

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs = inputs.to(device)
        inputs.requires_grad = True
        targets = targets.to(device)
        batch_size, n_channels, ny, nx = inputs.shape
        n_classes, = targets.shape


        # Parent Computations
        parent_gradients = []

        target_logits = parent(inputs)
        _, pseudo_labels = torch.max(target_logits.data, 1)

        for l in range(10):
            out_l = target_logits[:,l].sum()
            l_gradient = torch.cat(torch.autograd.grad(out_l, inputs, create_graph=True))
            parent_gradients.append(l_gradient)

        parent_gradients = torch.stack(parent_gradients, dim=1)



        # Child Computations
        child_gradients = []

        outputs = net(inputs)

        for l in range(10):
            out_l = outputs[:,l].sum()
            l_gradient = torch.cat(torch.autograd.grad(out_l, inputs, create_graph=True))
            child_gradients.append(l_gradient)

        child_gradients = torch.stack(child_gradients, dim=1)


        membership_loss = pseudo_label_criterion(outputs, pseudo_labels)
        gradient_loss = mse_criterion(parent_gradients, child_gradients) / batch_size
        lam = 1 if grad_query else 0
        loss = membership_loss + lam*gradient_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_mem_loss += membership_loss.item()
        progress_grad_loss += gradient_loss.item()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx,
                     len(loader),
                     'Gradient Loss: %.3f | Membership Loss: %.3f | True Train Acc: %.3f%% (%d/%d)'
                     % (progress_grad_loss / (batch_idx + 1), progress_mem_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | True Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



# Train function


def train(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# train function in a teacher-student fashion
def train_teacher(epoch, net, criterion, optimizer, use_cuda, logfile, loader, wmloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            if use_cuda:
                wminput, wmtarget = wminput.cuda(), wmtarget.cuda()
            wminputs.append(wminput)
            wmtargets.append(wmtarget)
        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if wmloader:
            # add wmimages and targets
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # return the acc.
    return 100. * correct / total
