'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

import os
import argparse

from models import *
from torch.autograd import Variable

import shutil
from time import time
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='start learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--decrease_from', default=100, type=int, help='Epoch to decrease from')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', default='ResNet18', help='Model')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('{}/best_model'.format(args.log_dir))
    net = checkpoint['net']
    best_acc = checkpoint['test_accuracy']
    start_epoch = checkpoint['epoch']

    # TODO: Add optimizer inintialization
    raise NotImplementedError('Add optimizer initialization!!!')
else:
    print('==> Building model..')
    net = class_for_name('models', args.model)()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)

with open('{}/log'.format(args.log_dir), 'w') as f:
    f.write('epoch,loss,train_acc,test_acc\n')


def save_checkpoint(state, is_best):
    torch.save(state, '{}/model'.format(args.log_dir))
    if is_best:
        shutil.copyfile('{}/model'.format(args.log_dir), '{}/best_model'.format(args.log_dir))


def logit2acc(outputs, targets):
    if isinstance(outputs, np.ndarray):
        return np.mean(outputs.argmax(axis=1) == targets.data.cpu().numpy())
    else:
        return np.mean(outputs.data.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch, k):
    return max(0, (k * np.minimum(2. - 1.0 * epoch / 100., 1.)))


prev_test_acc = 0

for epoch in range(200):  # loop over the dataset multiple times
    t0 = time()
    adjust_learning_rate(optimizer, lr_linear(epoch, 1e-3))
    net.train()
    training_loss = 0
    accs = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # wrap data in Variable and put them on GPU
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        accs.append(logit2acc(outputs, labels))  # probably a bad way to calculate accuracy
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.cpu().data.numpy()[0]

    train_acc = np.mean(accs)
    accs = []
    net.eval()

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        outputs = net(inputs)
        accs.append(logit2acc(outputs, labels))  # probably a bad way to calculate accuracy


    print(' -- Epoch %d time: %.4f loss: %.4f training acc: %.4f, validation accuracy: %.4f --' %
          (epoch, time() - t0, training_loss, train_acc, np.mean(accs)))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'test_accuracy': np.mean(accs),
        'optimizer': optimizer.state_dict(),
        'net': net.mdule if use_cuda else net,
    }, prev_test_acc < np.mean(accs))

    with open('{}/log'.format(args.log_dir), 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, training_loss, train_acc, np.mean(accs)))

    prev_test_acc = np.mean(accs)

print('Finished Training')
