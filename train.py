'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

import os
import argparse

from models import *
from models.stochbn import _MyBatchNorm
from torch.autograd import Variable

import shutil
from time import time
import importlib


torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='start learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--decrease_from', default=100, type=int, help='Epoch to decrease lr linear to 0 from')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--sample_stats_from', type=int, default=None)
parser.add_argument('--start_tmode', type=int, default=None)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--model', '-m', default='ResNet18', help='Model')
parser.add_argument('--k', '-k', default=1, type=float, help='Model size for VGG')
parser.add_argument('--decay', default=None, type=float,
                    help='Decay rate')
parser.add_argument('--finetune_bn', action='store_true')
parser.add_argument('--var_strategy', default='vanilla')
parser.add_argument('--mean_strategy', default='vanilla')
args = parser.parse_args()
args.script = os.path.basename(__file__)

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transform_train if args.augmentation else transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.model), 'Error: no checkpoint found!'
    checkpoint = torch.load('{}'.format(args.model))
    net = load_model(args.model)
    best_acc = checkpoint['test_accuracy']
    print('Loaded model test accuracy: {}'.format(best_acc))
else:
    print('==> Building model..')
    net = get_model(**vars(args))


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))


if args.finetune_bn:
    params = []
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            for p in m.parameters():
                params.append(p)
else:
    params = net.parameters()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(params, lr=args.lr)

# TODO: manage optimizer state
if args.resume and not args.finetune_bn:
    # TODO: why it works?
    optimizer.load_state_dict(checkpoint['optimizer'])


with open('{}/log'.format(args.log_dir), 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    f.write('epoch,loss,train_acc,test_acc\n')


def save_checkpoint(state, is_best):
    torch.save(state, '{}/model'.format(args.log_dir))
    if is_best:
        shutil.copyfile('{}/model'.format(args.log_dir), '{}/best_model'.format(args.log_dir))

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    return max(0, (args.lr * np.minimum((args.decrease_from - epoch) * 1. / (args.epochs - args.decrease_from) + 1, 1.)))


def lr_exponential(epoch):
    return args.lr * (args.decay ** np.maximum(0, epoch - args.decrease_from))


prev_test_acc = 0
counter = AccCounter()

for epoch in range(args.epochs):
    counter.flush()

    t0 = time()
    lr = lr_exponential(epoch) if args.decay else lr_linear(epoch)
    adjust_learning_rate(optimizer, lr)

    net.train()
    if args.training_mode == 'vanilla':
        set_StochBN_train_mode(net, 'vanilla')
    elif args.finetune_bn:
        net.eval()
    elif args.training_mode == 'sample-stats':
        if args.sample_stats_from > (epoch + 1):
            set_StochBN_train_mode(net, 'collect-stats')
        else:
            set_StochBN_train_mode(net, 'sample-stats')
    elif args.training_mode == 'running-stats':
        set_StochBN_train_mode(net, 'running-stats')

    training_loss = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # wrap data in Variable and put them on GPU
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        counter.add(outputs.data.cpu().numpy(), labels.data.cpu().numpy())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.cpu().data.numpy()[0]

    train_acc = counter.acc()
    counter.flush()
    net.eval()

    for _, (inputs, labels) in enumerate(testloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        outputs = net(inputs)
        counter.add(outputs.data.cpu().numpy(), labels.data.cpu().numpy())

    print(' -- Epoch %d time: %.4f loss: %.4f training acc: %.4f, validation accuracy: %.4f ; lr %.6f --' %
          (epoch, time() - t0, training_loss, train_acc, counter.acc(), lr))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.module.state_dict() if use_cuda else net.state_dict(),
        'test_accuracy': counter.acc(),
        'optimizer': optimizer.state_dict(),
        'net': net.module if use_cuda else net,
        'name': args.model,
        # TODO: not flexible
        'model_args': {'k': args.k},
        'script_args': vars(args)
    }, prev_test_acc < counter.acc())

    with open('{}/log'.format(args.log_dir), 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, training_loss, train_acc, counter.acc()))

    prev_test_acc = counter.acc()

print('Finish Training')
