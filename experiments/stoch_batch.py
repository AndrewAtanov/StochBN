from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from itertools import product

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
from utils import *
from time import time


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', '-m', help='Trained model')
parser.add_argument('--batch_sizes', '-n', type=int, nargs='+', help='List of batch sizes to sample')
parser.add_argument('--n_infer', type=int, nargs='+', help='List of numbers of inferences')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--first', action='store_true', help='Sample only on first BN layer.')
parser.add_argument('--augmentation', '-a', action='store_true', help='Add augmentation to data')
args = parser.parse_args()

torch.cuda.random.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print('=> Building model...')
net = load_model(args.model, print_info=True)

print('=> Loading data...')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transform_train if args.augmentation else transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


filename = uniquify(os.path.join(args.log_dir, 'stoch_batch'))
with open(filename, 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    f.write('n_infer,batch_size,accuracy\n')

print('=> Start experiment')

k = 0
for m in net.modules():
    if isinstance(m, MyBatchNorm2d):
        m.__n = k
        k += 1

net.eval()
for n_infer, bs in product(args.n_infer, args.batch_sizes):
    if not args.first:
        set_StochBN_test_mode(net, 'sample-reduce-batch-{}'.format(bs))
    else:
        for m in net.modules():
            if isinstance(m, MyBatchNorm2d):
                m.test_mode = 'sample-{}-batch-{}'.format('pass' if m.__n == 0 else 'reduce', bs)

    counter = AccCounter()
    start = time()
    for i, (inputs, labels) in enumerate(testloader):
        ens = Ensemble()
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        for _ in range(n_infer):
            ens.add_estimator(net(inputs).data.cpu().numpy())
        counter.add(ens.get_proba(), labels.data.cpu().numpy())

    with open(filename, 'a') as f:
        f.write('{},{},{}\n'.format(n_infer, bs, counter.acc()))

    print('== batch size {} -- accuracy {:.5f} ; time {:.2f} =='.format(bs, counter.acc(), time() - start))
