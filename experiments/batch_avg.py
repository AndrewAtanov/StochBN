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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', help='Trained model')
parser.add_argument('--n_inferences', '-n', type=int, nargs='+', help='List of number of batches for one object')
parser.add_argument('--data', '-d', default='test', help='Either \'train\' or \'test\' -- data to fill batch')
parser.add_argument('--acc', default='test', help='Either \'train\' or \'test\' -- to calculate accuracy')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--bs', type=int, nargs='+', default=[200], help='Batch size')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--permute', dest='permute', action='store_true')
parser.add_argument('--no-permute', dest='permute', action='store_false')
parser.set_defaults(permute=True)
args = parser.parse_args()
args.script = os.path.basename(__file__)

torch.cuda.random.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


use_cuda = torch.cuda.is_available()

print('==> Read model ...')

net = load_model(args.model, print_info=True)

print('==> Preparing data ...')

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)
test_data = testset.test_data.astype(float).transpose((0, 3, 1, 2))
test_labels = np.array(testset.test_labels)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=testset.test_data.shape[0],
                                         shuffle=False, num_workers=2)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)
train_data = trainset.train_data.astype(float).transpose((0, 3, 1, 2))
train_labels = trainset.train_labels


filename = uniquify(os.path.join(args.log_dir, 'batch_avg'))

with open(filename, 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    f.write('n_infer,batch_size,acc\n')

print('==> Start experiment')

net.train()

acc_data = test_data if args.acc == 'test' else train_data
acc_labels = test_labels if args.acc == 'test' else train_labels

for n_infer, BS in product(args.n_inferences, args.bs):
    start_time = time()
    counter = AccCounter()

    if args.data == 'test' or args.acc == 'train':
        ens = Ensemble()
        logits = np.zeros([acc_data.shape[0], 10])
        for _ in range(n_infer):
            if args.augmentation:
                acc_data = np.array(list(map(lambda x: transform_train(x).numpy(),
                                             testset.test_data if args.acc == 'test' else trainset.train_data)))
            else:
                acc_data = np.array(list(map(lambda x: transform_test(x).numpy(),
                                             testset.test_data if args.acc == 'test' else trainset.train_data)))

            if args.permute:
                perm = np.random.permutation(np.arange(acc_data.shape[0]))
            else:
                perm = np.arange(acc_data.shape[0])

            for i in range(0, len(perm), BS):
                idxs = perm[i: i + BS]
                inputs = Variable(torch.Tensor(acc_data[idxs]).cuda(async=True))
                outputs = net(inputs)
                logits[idxs] += outputs.cpu().data.numpy()

        ens.add_estimator(logits)

        counter = AccCounter()
        counter.add(ens.get_proba(), acc_labels)
    else:
        train_data = trainset.train_data
        for i, [x, y] in enumerate(zip(testset.test_data, testset.test_labels)):
            batch = train_data[np.random.choice(train_data.shape[0], BS, replace=False)]
            ens = Ensemble()
            for _ in range(n_infer):
                if args.permute:
                    batch = train_data[np.random.choice(train_data.shape[0], BS, replace=False)]

                if args.augmentation:
                    x = transform_train(x)
                    batch = np.array(list(map(lambda x: transform_train(x).numpy(), batch)))
                else:
                    x = transform_test(x)
                    batch = np.array(list(map(lambda x: transform_test(x).numpy(), batch)))

                logits = np.zeros([acc_data.shape[0], 10])
                batch[0] = x.numpy()
                inputs = Variable(torch.Tensor(batch).cuda(async=True))
                outputs = net(inputs)

                ens.add_estimator(outputs.cpu().data.numpy()[0][np.newaxis])

            counter.add(ens.get_proba(), [y])

    acc = counter.acc()
    print('{} inferences, batch size {} -- accuracy {}; time {:.3f} sec'.format(n_infer, BS, acc,
                                                                                time() - start_time))

    with open(filename, 'a') as f:
        f.write('{},{},{}\n'.format(n_infer, BS, acc))
