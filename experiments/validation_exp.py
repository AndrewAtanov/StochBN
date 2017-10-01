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
from utils import *
from time import time


def cifar_accuracy(net, mode='test', random_strategy=False, tries=20):
    counter = AccCounter()
    loader = testloader if mode == 'test' else trainloader
    for i, (inputs, labels) in enumerate(loader):
        ens = Ensemble()

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # outputs = net(inputs).cpu().data.numpy()
        ens.add_estimator(net(inputs).cpu().data.numpy())
        k = 1
        while random_strategy and k < tries:
            # outputs += net(inputs).cpu().data.numpy()
            ens.add_estimator(net(inputs).cpu().data.numpy())
            k += 1

        counter.add(ens.get_proba(), labels.data.cpu().numpy())
    return counter.acc()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', help='Model checkpoint filename')
parser.add_argument('--log_dir', default='./', help='Directory for logs')
parser.add_argument('--strategies', '-s', default=['vanilla', 'mean', 'random'],
                    nargs='+', type=str, help='Strategies for BN layer.')
parser.add_argument('--tries', '-t', nargs='+', type=int,
                    help='Number of tries for random strategy')
parser.add_argument('--train_passes', default=[1], nargs='+', type=int)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
args.script = os.path.basename(__file__)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.random.manual_seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


print('==> Load model...')
net = load_model(args.model, print_info=True)

print('==> Load data...')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_train if args.augmentation else transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False,
                                         num_workers=2)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                          shuffle=True, num_workers=2)


net.eval()
# print('Vanilla train accuracy', cifar_accuracy(net, 'train'))

log_fn = uniquify('{}/validation_exp'.format(args.log_dir))

print('==> Start experiment')

with open(log_fn, 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    f.write('mean,var,tries,data_passes,acc\n')

passes_done = 0
for n_passes in args.train_passes:
    print('--   ', n_passes, '   --')

    net.train()
    set_collect(net)
    while passes_done < n_passes:
        cifar_accuracy(net, 'train')
        passes_done += 1
    set_collect(net, mode=False)

    for ms, vs in product(args.strategies, repeat=2):
        net.eval()
        set_MyBN_strategy(net, mean_strategy=ms, var_strategy=vs)

        rs = (ms == 'random') or (vs == 'random')
        if rs:
            results = []
            for t in args.tries:
                results.append(cifar_accuracy(net, random_strategy=rs, tries=t))
                with open(log_fn.format(ms, vs), 'a') as f:
                    f.write('{},{},{},{},{}\n'.format(ms, vs, t, n_passes,
                                                      results[-1]))
                print('Mean strategy -- {}, var strategy -- {}, '
                      'acc {} ({} tries)'.format(ms, vs, results[-1], t))

        else:
            acc = cifar_accuracy(net)
            with open(log_fn.format(ms, vs), 'a') as f:
                f.write('{},{},1,{},{}\n'.format(ms, vs, n_passes, acc))
            print('Mean strategy -- {}, var strategy -- {}, '
                  'acc {}'.format(ms, vs, acc))
