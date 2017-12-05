'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *

import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

import os
import argparse

from models import *

from models.stochbn import _MyBatchNorm
from torch.autograd import Variable

import shutil, pickle
from time import time
import importlib
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Net training')
parser.add_argument('--lr', default=0.001, type=float, help='start learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--decrease_from', default=100, type=int, help='Epoch to decrease lr linear to 0 from')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--sample_stats_from', type=int, default=1)
parser.add_argument('--start_tmode', type=int, default=None)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--model', '-m', default='ResNet18', help='Model')
parser.add_argument('--k', '-k', default=1, type=float, help='Model size for VGG')
parser.add_argument('--dropout', type=float, nargs='+', default=None)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--decay', default=None, type=float, help='Decay rate')
parser.add_argument('--finetune_bn', action='store_true')
parser.add_argument('--log_grads', action='store_true')
parser.add_argument('--log_params', action='store_true')
parser.add_argument('--log_snr', action='store_true')
parser.add_argument('--noiid', action='store_true')
parser.add_argument('--learn_bn_stats', action='store_true')
parser.add_argument('--var_strategy', default='vanilla')
parser.add_argument('--mean_strategy', default='vanilla')
parser.add_argument('--sample_policy', default='one')
parser.add_argument('--update_policy', default='after')
parser.add_argument('--bn_mode', default='vanilla')
parser.add_argument('--sample_w_init', type=float, default=0.)
parser.add_argument('--sample_w_iter', type=int, default=None)
parser.add_argument('--data', default='cifar')
parser.add_argument('--log_n_epochs', default=10, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--b1', default=0.9, type=float)
parser.add_argument('--b2', default=0.999, type=float)
args = parser.parse_args()
args.script = os.path.basename(__file__)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# tensorbord writer
writer = SummaryWriter(args.log_dir)
NCLASSES = 10

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    MyPad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.data == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train if args.augmentation else transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.data == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                          transform=transform_train if args.augmentation else transform_test)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    NTEST = testset.test_data.shape[0]
    NTRAIN = trainset.train_data.shape[0]
elif args.data == 'cifar5':
    NCLASSES = 5
    CIFAR5_CLASSES = [0, 1, 2, 3, 4]
    trainset = CIFAR(root='./data', train=True, download=True,
                     transform=transform_train if args.augmentation else transform_test, classes=CIFAR5_CLASSES)
    testset = CIFAR(root='./data', train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    NTEST = testset.test_data.shape[0]
    NTRAIN = trainset.train_data.shape[0]
else:
    raise NotImplementedError

if args.noiid:
    if args.data != 'cifar':
        raise NotImplementedError
    noiidsampler = CIFARNoIIDSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, sampler=noiidsampler, num_workers=2)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

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
    net = get_model(n_classes=NCLASSES, **vars(args))


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
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay,
                       betas=(args.b1, args.b2))

# TODO: manage optimizer state
if args.resume and not args.finetune_bn:
    # TODO: why it works?
    optimizer.load_state_dict(checkpoint['optimizer'])


with open('{}/log'.format(args.log_dir), 'w') as f:
    f.write('#{}\n'.format(make_description(args)))
    f.write('epoch,loss,train_acc,test_acc\n')


def save_checkpoint(state, is_best, epoch=None):
    if epoch:
        torch.save(state, '{}/model-{}'.format(args.log_dir, epoch))
        if is_best:
            shutil.copyfile('{}/model-{}'.format(args.log_dir, epoch), '{}/best_model'.format(args.log_dir))
    else:
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


def sample_weight_linear(epoch):
    return float(1. - max(0, ((1. - args.sample_w_init) * np.minimum((args.sample_stats_from - epoch) * 1. / args.sample_w_iter + 1, 1.))))


best_test_acc = 0
counter = AccCounter()

set_bn_mode(net, args.bn_mode,
            update_policy=args.update_policy,
            sample_policy=args.sample_policy)

model_args = vars(args)
model_args['n_classes'] = NCLASSES

print(net)

for epoch in range(args.epochs):
    set_sample_policy(net, sample_policy=args.sample_policy)
    counter.flush()

    t0 = time()
    lr = lr_exponential(epoch) if args.decay else lr_linear(epoch)
    adjust_learning_rate(optimizer, lr)

    net.train()
    mean_strategy = args.mean_strategy
    var_strategy = args.var_strategy

    sample_w = sample_weight_linear(epoch + 1) if args.sample_w_iter else args.sample_w_init
    set_bn_sample_weight(net, sample_w)

    if args.sample_stats_from > (epoch + 1):
        if mean_strategy == 'sample':
            mean_strategy = 'batch'
        if var_strategy == 'sample':
            var_strategy = 'batch'

    set_MyBN_strategy(net, mean_strategy=mean_strategy, var_strategy=var_strategy)

    training_loss = 0
    grad_norms = {}
    for name, param in net.named_parameters():
        grad_norms[name] = []

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
        training_loss += loss.cpu().data.numpy()[0] * float(inputs.size(0))

        if args.log_grads:
            for name, param in net.named_parameters():
                norm = float((to_np(param.grad).reshape((-1,))**2).mean()**0.5)
                grad_norms[name].append(norm)

    if args.log_params:
        for name, param in net.named_parameters():
            writer.add_histogram('{}/values'.format(name), param.data.cpu().numpy().flatten(),
                                 bins='auto', global_step=epoch + 1)

    if args.log_grads:
        for name, norm in grad_norms.items():
            writer.add_histogram('{}/grad-norm'.format(name), np.array(norm), bins='auto', global_step=epoch + 1)

    train_acc = counter.acc()
    counter.flush()
    test_loss = 0

    net.eval()
    set_MyBN_strategy(net, var_strategy='running', mean_strategy='running')

    for _, (inputs, labels) in enumerate(testloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += to_np(loss) * float(inputs.size(0))
        counter.add(outputs.data.cpu().numpy(), labels.data.cpu().numpy())

    writer.add_scalar('sample_weight', sample_w, epoch + 1)
    writer.add_scalar('learning_rate', lr, epoch + 1)

    print(' -- Epoch %d | time: %.4f | loss: %.4f | training acc: %.4f validation accuracy: %.4f | lr %.6f | sample weight %.2f --' %
          (epoch, time() - t0, training_loss, train_acc, counter.acc(), lr, sample_w))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.module.state_dict() if use_cuda else net.state_dict(),
        'test_accuracy': counter.acc(),
        'optimizer': optimizer.state_dict(),
        # 'net': net.module if use_cuda else net,
        'name': args.model,
        # TODO: not flexible
        'model_args': model_args,
        'script_args': vars(args)
    }, best_test_acc < counter.acc())

    writer.add_scalar('log-loss/train', float(np.log(training_loss)) / float(NTRAIN), global_step=epoch)
    writer.add_scalar('log-loss/test', float(np.log(test_loss)) / float(NTEST), global_step=epoch)

    writer.add_scalar('accuracy/train', float(train_acc), global_step=epoch)
    writer.add_scalar('accuracy/test', float(counter.acc()), global_step=epoch)

    writer.export_scalars_to_json(os.path.join(args.log_dir, 'all_scalars.json'))

    with open('{}/log'.format(args.log_dir), 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, training_loss, train_acc, counter.acc()))

    if best_test_acc < counter.acc():
        best_test_acc = counter.acc()

    if ((epoch + 1) % args.log_n_epochs == 0) or (epoch == 0):
        big_log = {}

        if args.bn_mode == 'StochBN':
            # Log SNR of approximated distribution in SBN
            for i, bn in enumerate([m for m in net.modules() if isinstance(m, _MyBatchNorm)]):
                eps = 1e-6
                mean_snr = np.abs(bn.running_mean_mean.cpu().numpy()) / np.sqrt(bn.running_mean_var.cpu().numpy() + eps)
                var_snr = 1 / np.sqrt(np.exp(bn.running_logvar_var.cpu().numpy()) - 1 + eps)
                writer.add_histogram('BN-{}/mean_snr'.format(i + 1), mean_snr, epoch + 1, bins='auto')
                writer.add_histogram('BN-{}/var_snr'.format(i + 1), var_snr, epoch + 1, bins='auto')

                big_log['BN-{}/mean_mean'.format(i + 1)] = bn.running_mean_mean.cpu().numpy()
                big_log['BN-{}/mean_var'.format(i + 1)] = bn.running_mean_var.cpu().numpy()

                big_log['BN-{}/logvar_mean'.format(i + 1)] = bn.running_logvar_mean.cpu().numpy()
                big_log['BN-{}/logvar_var'.format(i + 1)] = bn.running_logvar_var.cpu().numpy()

            # ---------------
            # Calculate ensemble accuracy and loss on TEST set

            set_MyBN_strategy(net, var_strategy='sample', mean_strategy='sample')
            set_sample_policy(net, sample_policy='one')
            ens_proba, gt_labels = predict_proba(testloader, net, ensembles=30, n_classes=NCLASSES)
            ens_acc = np.mean(np.argmax(ens_proba, axis=1) == gt_labels)
            writer.add_scalar('accuracy/ensemble/test', float(ens_acc), global_step=epoch)
            big_log['ens_proba'] = ens_proba
            big_log['ens_labels'] = gt_labels

            ens_loss = -np.log(ens_proba[np.arange(ens_proba.shape[0]), gt_labels])
            writer.add_scalar('log-loss/ensemble/test',
                              float(np.log(np.mean(ens_loss))), global_step=epoch)

            correct = (np.argmax(ens_proba, axis=1) == gt_labels)
            writer.add_scalar('log-loss/ensemble/test/correct',
                              float(np.log(np.mean(ens_loss[correct]))), global_step=epoch)
            writer.add_scalar('log-loss/ensemble/test/incorrect',
                              float(np.log(np.mean(ens_loss[~correct]))), global_step=epoch)

            ens_entropy = entropy(ens_proba)
            writer.add_histogram('entropy/ensemble/test/correct', ens_entropy[correct], epoch + 1, bins='auto')
            writer.add_histogram('entropy/ensemble/test/incorrect', ens_entropy[~correct], epoch + 1, bins='auto')

            # ------------
            # Calculate ensemble accuracy and loss on TRAIN set

            set_MyBN_strategy(net, var_strategy='sample', mean_strategy='sample')
            set_sample_policy(net, sample_policy='one')
            ens_proba, gt_labels = predict_proba(trainloader, net, ensembles=30, n_classes=NCLASSES)
            ens_acc = np.mean(np.argmax(ens_proba, axis=1) == gt_labels)
            writer.add_scalar('accuracy/ensemble/train', float(ens_acc), global_step=epoch)
            big_log['ens_proba_train'] = ens_proba
            big_log['ens_labels_train'] = gt_labels

            ens_loss = -np.log(ens_proba[np.arange(ens_proba.shape[0]), gt_labels])
            writer.add_scalar('log-loss/ensemble/train', float(np.log(np.mean(ens_loss))),
                              global_step=epoch)

            correct = (np.argmax(ens_proba, axis=1) == gt_labels)
            writer.add_scalar('log-loss/ensemble/train/correct',
                              float(np.log(np.mean(ens_loss[correct]))), global_step=epoch)
            writer.add_scalar('log-loss/ensemble/train/incorrect',
                              float(np.log(np.mean(ens_loss[~correct]))), global_step=epoch)

            ens_entropy = entropy(ens_proba)
            writer.add_histogram('entropy/ensemble/train/correct', ens_entropy[correct], epoch + 1, bins='auto')
            writer.add_histogram('entropy/ensemble/train/incorrect', ens_entropy[~correct], epoch + 1, bins='auto')

        # -------------
        # Calculate eval mode accuracy and loss on TEST set

        set_MyBN_strategy(net, var_strategy='running', mean_strategy='running')
        set_sample_policy(net, sample_policy='one')
        eval_proba, gt_labels = predict_proba(testloader, net, ensembles=1, n_classes=NCLASSES)
        big_log['eval_proba'] = eval_proba
        big_log['eval_labels'] = gt_labels

        eval_loss = -np.log(eval_proba[np.arange(eval_proba.shape[0]), gt_labels])
        writer.add_scalar('log-loss/eval/test',
                          float(np.log(np.mean(eval_loss))), global_step=epoch)

        correct = (np.argmax(eval_proba, axis=1) == gt_labels)
        writer.add_scalar('log-loss/eval/test/correct',
                          float(np.log(np.mean(eval_loss[correct]))), global_step=epoch)
        writer.add_scalar('log-loss/eval/test/incorrect',
                          float(np.log(np.mean(eval_loss[~correct]))), global_step=epoch)

        eval_entropy = entropy(eval_proba)
        correct = (np.argmax(eval_proba, axis=1) == gt_labels)
        writer.add_histogram('entropy/eval/test/correct', eval_entropy[correct], epoch + 1, bins='auto')
        writer.add_histogram('entropy/eval/test/incorrect', eval_entropy[~correct], epoch + 1, bins='auto')

        # -------------
        # Calculate eval mode accuracy and loss on TRAIN set

        set_MyBN_strategy(net, var_strategy='running', mean_strategy='running')
        set_sample_policy(net, sample_policy='one')
        eval_proba, gt_labels = predict_proba(trainloader, net, ensembles=1, n_classes=NCLASSES)
        big_log['eval_proba_train'] = eval_proba
        big_log['eval_labels_train'] = gt_labels

        eval_loss = -np.log(eval_proba[np.arange(eval_proba.shape[0]), gt_labels])
        writer.add_scalar('log-loss/eval/train',
                          float(np.log(np.mean(eval_loss))), global_step=epoch)

        correct = (np.argmax(eval_proba, axis=1) == gt_labels)
        writer.add_scalar('log-loss/eval/train/correct', float(np.log(np.mean(eval_loss[correct]))), global_step=epoch)
        writer.add_scalar('log-loss/eval/train/incorrect',
                          float(np.log(np.mean(eval_loss[~correct]))), global_step=epoch)

        eval_entropy = entropy(eval_proba)
        correct = (np.argmax(eval_proba, axis=1) == gt_labels)
        writer.add_histogram('entropy/eval/train/correct', eval_entropy[correct], epoch + 1, bins='auto')
        writer.add_histogram('entropy/eval/train/incorrect', eval_entropy[~correct], epoch + 1, bins='auto')

        torch.save(big_log, os.path.join(args.log_dir, 'big_log-{}'.format(epoch + 1)))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.module.state_dict() if use_cuda else net.state_dict(),
            'test_accuracy': counter.acc(),
            'optimizer': optimizer.state_dict(),
            # 'net': net.module if use_cuda else net,
            'name': args.model,
            # TODO: not flexible
            'model_args': model_args,
            'script_args': vars(args)
        }, best_test_acc < counter.acc(), epoch + 1)

print('Finish Training')
