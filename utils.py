import numpy as np
from models.stochbn import _MyBatchNorm


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
