from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

import os
import argparse

import sys
sys.path.append('/home/andrew/StochBN/StochBN/')

from models import *
from torch.autograd import Variable

import shutil
from time import time
import importlib
from utils import AccCounter
from time import time


torch.cuda.random.manual_seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', help='Trained model')
parser.add_argument('--n_inferences', '-n', type=int, nargs='+', help='List of number of batches for one object')
parser.add_argument('--data', '-d', default='test', help='Either \'train\' or \'test\' -- data to fill batch')
parser.add_argument('--log_dir', help='Directory for logging')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print('==> Read model ...')
chekpoint = torch.load(args.model)

print('-------------- WARNNIG!!! ResNet34 ONLY! -----------------')
net = ResNet34() #chekpoint['net']

if use_cuda:
    net = DataParallel(net, device_ids=range(torch.cuda.device_count()))

net.load_state_dict(chekpoint['state_dict'])

print('==> Preparing data ...')

dataset = torchvision.datasets.CIFAR10(root='./data', train=args.data == 'train',
                                       download=True)

data = dataset.train_data if args.data == 'train' else dataset.test_data
data = data.astype(float).transpose((0, 3, 1, 2))
labels = np.array(dataset.train_labels if args.data == 'train' else dataset.test_labels)

BS = 200

with open('{}/batch_avg_{}_logs'.format(args.log_dir, args.data), 'w') as f:
    f.write('n_infer,acc\n')

print('==> Start experiment')

net.train()

for n_infer in args.n_inferences:
    logits = np.zeros([data.shape[0], 10])
    start_time = time()
    
    for _ in range(n_infer):
        perm = np.random.permutation(np.arange(data.shape[0]))
        for i in range(0, len(perm), BS):
            idxs = perm[i: i + BS]
            inputs = Variable(torch.Tensor(data[idxs] / 255.).cuda(async=True))
            outputs = net(inputs)

            logits[idxs] += outputs.cpu().data.numpy()

    counter = AccCounter()
    counter.add(logits, labels)
    acc = counter.acc()

    print('{} inferences, accuracy {}, time {:.3f} sec'.format(n_infer, acc, time() - start_time))

    with open('{}/batch_avg_{}_logs'.format(args.log_dir, args.data), 'a') as f:
        f.write('{},{}'.format(n_infer, acc))
