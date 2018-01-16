import argparse
import utils
import torch
from tqdm import tqdm
from torch.autograd import Variable
import os
from models.stochbn import _MyBatchNorm


parser = argparse.ArgumentParser(description='Net training')
parser.add_argument('--model')
parser.add_argument('--new_model', default=None)
parser.add_argument('--epochs', type=int)
parser.add_argument('--data')
parser.add_argument('--sample_policy', default='one')
parser.add_argument('--stats_strategy', default='sample')
parser.add_argument('--bs', default='auto')
parser.add_argument('--augmentation', dest='augmentation', action='store_true')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--n_sbn', default=None, type=int)
parser.add_argument('--stats_momentum', default=0.1, type=float)
args = parser.parse_args()

net = utils.load_model(args.model)

ckpt = torch.load(args.model)
train_bs = ckpt['script_args']['bs'] if args.bs == 'auto' else int(args.bs)

trainloader, _ = utils.get_dataloader(data=args.data, train_bs=train_bs, augmentation=args.augmentation)

utils.set_bn_mode(net, 'StochBN', sample_policy=args.sample_policy)
net.train()
utils.set_bn_params(net, stats_momentum=args.stats_momentum)
utils.set_MyBN_strategy(net, mean_strategy=args.stats_strategy, var_strategy=args.stats_strategy)

if args.n_sbn:
    bns = [m for m in net.modules() if isinstance(m, _MyBatchNorm)]
    for bn in bns[:len(bns) - args.n_sbn]:
        bn.mean_strategy = 'batch'
        bn.var_strategy = 'batch'

for _ in tqdm(range(args.epochs)):
    for x, _ in trainloader:
        net(Variable(x.cuda()))

ckpt['state_dict'] = net.module.state_dict()
ckpt['tune_epoch'] = args.epochs

fname = args.new_model
if fname is None:
    fname = 'tuned_' + os.path.basename(args.model)

if args.bs != 'auto':
    fname += '_{}'.format(args.bs)

fname = utils.uniquify(os.path.join(os.path.dirname(args.model), fname), sep='-')
torch.save(ckpt, fname)
