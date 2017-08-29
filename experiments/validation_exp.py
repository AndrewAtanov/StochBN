from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
from itertools import product


import os
import argparse

import sys
sys.path.append('/home/andrew/StochBN/StochBN/')

from models import *
from torch.autograd import Variable

import shutil
from time import time
import importlib
from utils import AccCounter, uniquify, load_model, set_MyBN_strategy, set_collect
from time import time


def cifar_accuracy(net, mode='test', random_strategy=False, tries=20):
    counter = AccCounter()
    loader = testloader if mode == 'test' else trainloader
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs).cpu().data.numpy()
        k = 1
        while random_strategy and k < tries:
            outputs += net(inputs).cpu().data.numpy()
            k += 1
        counter.add(outputs, labels.data.cpu().numpy())
    return counter.acc()

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.random.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', help='Model checkpoint filename')
parser.add_argument('--log_dir', default='./', help='Directory for logs')
parser.add_argument('--strategies', '-s', default=['vanilla', 'mean', 'random'],
                    nargs='+', type=str, help='Strategies for BN layer.')
parser.add_argument('--tries', '-t', nargs='+', type=int,
                    help='Number of tries for random strategy')
args = parser.parse_args()

print('==> Load model...')
net = load_model(args.model, print_info=True)

print('==> Load data...')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False,
                                         num_workers=2)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

net.train()
set_collect(net)
print('Train acc', cifar_accuracy(net, 'train'))


set_collect(net, mode=False)

log_fn = uniquify('{}/validation_exp'.format(args.log_dir))
cell_len = max(8, max(map(len, args.strategies)))

print('==> Start experiment')

with open(log_fn, 'w') as f:
    f.write('mean,var,acc\n')


for ms, vs in product(args.strategies, repeat=2):
    net.eval()
    set_MyBN_strategy(net, mean_strategy=ms, var_strategy=vs)

    rs = (ms == 'random') or (vs == 'random')
    if rs:
        results = []
        for t in args.tries:
            results.append(cifar_accuracy(net, random_strategy=rs, tries=t))
            with open(log_fn.format(ms, vs), 'a') as f:
                f.write('{},{},{}\n'.format(ms, vs, results[-1]))
            print('Mean strategy -- {}, var strategy -- {}, '
                  'acc {} ({} tries)'.format(ms, vs, results[-1], t))

    else:
        acc = cifar_accuracy(net)
        with open(log_fn.format(ms, vs), 'a') as f:
            f.write('{},{},{}\n'.format(ms, vs, acc))
        print('Mean strategy -- {}, var strategy -- {}, '
              'acc {}'.format(ms, vs, acc))
