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
from utils import AccCounter, uniquify, load_model
from time import time


torch.cuda.random.manual_seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', help='Trained model')
parser.add_argument('--n_inferences', '-n', type=int, nargs='+', help='List of number of batches for one object')
parser.add_argument('--data', '-d', default='test', help='Either \'train\' or \'test\' -- data to fill batch')
parser.add_argument('--acc', default='test', help='Either \'train\' or \'test\' -- to calculate accuracy')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--bs', type=int, nargs='+', default=[200], help='Batch size')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print('==> Read model ...')

net = load_model(args.model, print_info=True)

print('==> Preparing data ...')

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True)
test_data = testset.test_data.astype(float).transpose((0, 3, 1, 2))
test_labels = np.array(testset.test_labels)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)
train_data = trainset.train_data.astype(float).transpose((0, 3, 1, 2))
train_labels = trainset.train_labels


filename = uniquify('{}/batch_avg_{}_logs_{}_acc'.format(args.log_dir,
                                                         args.data,
                                                         args.acc))

with open(filename, 'w') as f:
    f.write('n_infer,batch_size,acc\n')

print('==> Start experiment')

net.train()

acc_data = test_data if args.acc == 'test' else train_data
acc_labels = test_labels if args.acc == 'test' else train_labels

for n_infer, BS in product(args.n_inferences, args.bs):
    start_time = time()
    logits = np.zeros([acc_data.shape[0], 10])

    if args.data == 'test' or args.acc == 'train':
        for _ in range(n_infer):
            perm = np.random.permutation(np.arange(acc_data.shape[0]))
            for i in range(0, len(perm), BS):
                idxs = perm[i: i + BS]
                inputs = Variable(torch.Tensor(acc_data[idxs] / 255.).cuda(async=True))
                outputs = net(inputs)

                logits[idxs] += outputs.cpu().data.numpy()
    else:
        for i, x in enumerate(test_data):
            for _ in range(n_infer):
                batch = train_data[np.random.choice(train_data.shape[0], BS, replace=False)]
                batch[0] = x
                inputs = Variable(torch.Tensor(batch / 255.).cuda(async=True))
                outputs = net(inputs)

                logits[i] += outputs.cpu().data.numpy()[0]


    counter = AccCounter()
    counter.add(logits, acc_labels)
    acc = counter.acc()

    print('{} inferences, batch size {} -- accuracy {}; time {:.3f} sec'.format(n_infer, BS, acc, time() - start_time))

    with open(filename, 'a') as f:
        f.write('{},{},{}\n'.format(n_infer, BS, acc))
