import torch
import argparse
import utils
from models.stochbn import _MyBatchNorm
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--tuned_model', default=None)
parser.add_argument('--kn_data', default='cifar5')
parser.add_argument('--unk_data', default='cifar5-rest')
parser.add_argument('--log_file', default='eval_data')
parser.add_argument('--bs', default=None, type=int, nargs='+')
parser.add_argument('--test_bs', default=500, type=int)
parser.add_argument('--n_classes', default=5, type=int)
parser.add_argument('--bn_types', nargs='+', default=['BN', 'uncorr_bn', 'HBN', 'HBN-T'])
parser.add_argument('--data_root', default='/home/andrew/StochBN/data')
parser.add_argument('--correction', action='store_true')
parser.add_argument('--correction_bs', default=None, type=int)
parser.add_argument('--seed', default=42, type=int)
# parser.add_argument('--model_types', nargs='+', default=['eval', 'ensemble'])
args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.model:
    ckpt = torch.load(args.model)
    log_file = utils.uniquify(os.path.join(os.path.dirname(args.model), args.log_file), sep='-')
else:
    ckpt = torch.load(args.tuned_model)
    log_file = utils.uniquify(os.path.join(os.path.dirname(args.tuned_model), args.log_file), sep='-')

trainloader_ukn, testloader_ukn = utils.get_dataloader(data=args.unk_data, test_bs=args.test_bs, drop_last_train=True,
                                                       data_root=args.data_root)

cifar5_train, cifar5_test = utils.get_dataloader(data=args.kn_data, test_bs=args.test_bs, data_root=args.data_root)
cifar5rest_train, cifar5rest_test = utils.get_dataloader(data=args.unk_data, test_bs=args.test_bs,
                                                         data_root=args.data_root)

model_bs = ckpt['script_args']['bs']
if args.bs is None:
    args.bs = [model_bs]

eval_data = {}

for model in args.bn_types:
    eval_data[model] = {}
    eval_data[model][model_bs] = {}

for eval_bs in args.bs:
    if 'uncorr_bn' in args.bn_types and args.model is not None:
        net = utils.load_model(args.model)
        net.eval()
        utils.set_do_to_train(net)
        utils.set_bn_mode(net, 'uncorr')
        bns = [m for m in net.modules() if isinstance(m, _MyBatchNorm)]
        utils.set_bn_params(net, bs=eval_bs)
        nbnlayers = len(bns)
        trainloader_kn, testloader_kn = utils.get_dataloader(data=args.kn_data, test_bs=args.test_bs,
                                                             train_bs=nbnlayers * eval_bs, drop_last_train=True,
                                                             data_root=args.data_root)

        eval_data['uncorr_bn'][model_bs][eval_bs] = {}

        res = utils.bn_ensemble(net, testloader_ukn, trainloader_kn, n_ensembles=50, return_logits=True)
        eval_data['uncorr_bn'][model_bs][eval_bs]['unknown'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

        res = utils.bn_ensemble(net, testloader_kn, trainloader_kn, n_ensembles=50, return_logits=True)
        eval_data['uncorr_bn'][model_bs][eval_bs]['known'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

    if 'BN' in args.bn_types and args.model is not None:
        net = utils.load_model(args.model)
        net.eval()
        utils.set_do_to_train(net)
        utils.set_bn_mode(net, 'uncorr')
        bns = [m for m in net.modules() if isinstance(m, _MyBatchNorm)]
        nbnlayers = len(bns)
        utils.set_bn_params(net, bs=eval_bs, uncorr_type='one-batch')
        trainloader_kn, testloader_kn = utils.get_dataloader(data=args.kn_data, test_bs=args.test_bs,
                                                             train_bs=eval_bs, drop_last_train=True,
                                                             data_root=args.data_root)

        res = utils.bn_ensemble(net, testloader_ukn, trainloader_kn, n_ensembles=50,
                                return_logits=True, vanilla=True)

        eval_data['BN'][model_bs][eval_bs] = {}
        eval_data['BN'][model_bs][eval_bs]['unknown'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

        res = utils.bn_ensemble(net, testloader_kn, trainloader_kn, n_ensembles=50, return_logits=True, vanilla=True)
        eval_data['BN'][model_bs][eval_bs]['known'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

if 'BN' in args.bn_types and args.model is not None:
    net = utils.load_model(args.model)
    utils.set_bn_mode(net, 'StochBN')
    utils.set_MyBN_strategy(net, mean_strategy='running', var_strategy='running')
    net.eval()
    utils.set_do_to_train(net)

    trainloader_kn, testloader_kn = utils.get_dataloader(data=args.kn_data, test_bs=args.test_bs, drop_last_train=True,
                                                         data_root=args.data_root)

    res = utils.predict_proba(cifar5rest_test, net, n_classes=args.n_classes, return_logits=True, ensembles=1)
    eval_data['BN'][model_bs][model_bs]['unknown'].update({
        'eval/proba': res[0],
        'eval/logits': res[2],
        'eval/labels': res[1]
    })

    res = utils.predict_proba(cifar5_test, net, n_classes=args.n_classes, return_logits=True, ensembles=1)
    eval_data['BN'][model_bs][model_bs]['known'].update({
        'eval/proba': res[0],
        'eval/logits': res[2],
        'eval/labels': res[1]
    })

for bn_type in ['HBN', 'HBN-T']:
    if bn_type == 'HBN-T' and args.tuned_model is None:
        continue
    if bn_type == 'HBN' and args.model is None:
        continue
    if bn_type in args.bn_types:
        if bn_type == 'HBN':
            hbn_net = utils.load_model(args.model)
        else:
            hbn_net = utils.load_model(args.tuned_model)
        hbn_net.eval()
        utils.set_bn_mode(hbn_net, 'StochBN')
        utils.set_MyBN_strategy(hbn_net, mean_strategy='sample', var_strategy='sample')
        utils.set_bn_params(hbn_net, bs=args.correction_bs, correction=args.correction)
        hbn_net.eval()
        utils.set_do_to_train(hbn_net)

        res = utils.predict_proba(cifar5rest_test, hbn_net, n_classes=args.n_classes, return_logits=True, ensembles=50)

        eval_data[bn_type][model_bs][model_bs] = {}

        eval_data[bn_type][model_bs][model_bs]['unknown'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

        res = utils.predict_proba(cifar5_test, hbn_net, n_classes=args.n_classes, return_logits=True, ensembles=50)
        eval_data[bn_type][model_bs][model_bs]['known'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'ensemble/labels': res[1]
        }

        hbn_net.eval()
        utils.set_MyBN_strategy(hbn_net, mean_strategy='running', var_strategy='running')
        utils.set_do_to_train(hbn_net)

        res = utils.predict_proba(cifar5rest_test, hbn_net, n_classes=args.n_classes, return_logits=True, ensembles=50)
        eval_data[bn_type][model_bs][model_bs]['unknown'].update({
            'eval/proba': res[0],
            'eval/logits': res[2],
            'eval/labels': res[1]
        })

        res = utils.predict_proba(cifar5_test, hbn_net, n_classes=args.n_classes, return_logits=True, ensembles=50)
        eval_data[bn_type][model_bs][model_bs]['known'].update({
            'eval/proba': res[0],
            'eval/logits': res[2],
            'eval/labels': res[1]
        })

torch.save(eval_data, log_file)
