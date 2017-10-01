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

from utils import *
from time import time
from scipy.stats import entropy
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', '-m', help='Trained model')
parser.add_argument('--batch_sizes', '-n', type=int, nargs='+', default=[200], help='List of batch sizes to sample')
parser.add_argument('--n_infer', type=int, nargs='+', default=[1], help='List of numbers of inferences')
parser.add_argument('--log_dir', default='/tmp/', help='Directory for logging')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--first', action='store_true', help='Sample only on first BN layer.')
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--eval', action='store_true')
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

net = load_model(args.model, print_info=True, n_classes=5)

print('==> Preparing data..')

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

NOCIFAR5_CLASSES = [0, 1, 2, 3, 4]
testset = CIFAR(root='./data', train=False, download=True, transform=transform_test, classes=NOCIFAR5_CLASSES)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)


filename = uniquify(os.path.join(args.log_dir, 'uncertainty'))

logs = pd.DataFrame(columns=["n_infer", "batch_size", "acc", "entropy"])

print('==> Start experiment')

net.train()
if args.eval:
    net.eval()

for k, (n_infer, BS) in enumerate(product(args.n_infer, args.batch_sizes)):
    start_time = time()
    ens = Ensemble()

    logits = np.zeros([testset.test_data.shape[0], len(NOCIFAR5_CLASSES)])
    for _ in range(n_infer):
        if args.augmentation:
            acc_data = np.array(list(map(lambda x: transform_train(x).numpy(), testset.test_data)))
        else:
            acc_data = np.array(list(map(lambda x: transform_test(x).numpy(), testset.test_data)))

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
    counter.add(ens.get_proba(), testset.test_labels)
    acc = counter.acc()

    print('{} inferences, batch size {} -- accuracy {}; time {:.3f} sec'.format(n_infer, BS, acc, time() - start_time))

    logs.loc[k] = [n_infer, BS, acc, entropy(ens.get_proba().T).tolist()]


with open(filename, 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    logs.to_csv(f)
