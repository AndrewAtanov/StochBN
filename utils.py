import numpy as np
from models.stochbn import _MyBatchNorm
import tempfile
import itertools as IT
import os
from torch.nn.parallel import DataParallel
import torch
from models import *
import importlib


def uniquify(path, sep = ''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


class Ensemble:
    def __init__(self):
        self.__n_estimators = 0
        self.cum_proba = 0

    def add_estimator(self, logits):
        l = np.exp(logits - logits.max(1)[:,np.newaxis])
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            self.cum_proba += l / l.sum(1)[:,np.newaxis]
            assert not np.isnan(self.cum_proba).any(), 'NaNs while computing softmax'
        except:
            print(' Save logits to dubug.npy ')
            np.save('dubug.npy', logits)
            raise
        self.__n_estimators += 1

    def get_proba(self):
        return self.cum_proba / self.__n_estimators



class AccCounter:
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        self.__n_objects = 0
        self.__sum = 0


def set_collect(net, mode=True):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.collect = mode


def set_MyBN_strategy(net, mean_strategy='vanilla', var_strategy='vanilla'):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.mean_strategy = mean_strategy
            m.var_strategy = var_strategy


def set_StochBN_train_mode(net, mode):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.train_mode = mode


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def set_StochBN_test_mode(net, mode):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.test_mode = mode


def make_description(args):
    return '{}'.format(vars(args))


def load_model(filename, print_info=False):
    use_cuda = torch.cuda.is_available()
    chekpoint = torch.load(filename)
    net = class_for_name('models', chekpoint['name'])()
    if use_cuda:
        net = DataParallel(net, device_ids=range(torch.cuda.device_count()))

    net.load_state_dict(chekpoint['state_dict'])

    if print_info:
        print('Net validation accuracy = {}'.format(chekpoint['test_accuracy']))

    return net
