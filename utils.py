import numpy as np
from models.stochbn import _MyBatchNorm
import tempfile
import itertools as IT
import os
from torch.nn.parallel import DataParallel
import torch
from models import *
import importlib
import sys
import pickle
import torchvision
import tensorboardX
from torchvision import transforms
import PIL


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
        l = np.exp(logits - logits.max(1)[:, np.newaxis])
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            self.cum_proba += l / l.sum(1)[:, np.newaxis]
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


def get_model(model='ResNet18', k=1., **kwargs):
    if 'ResNet' in model:
        return class_for_name('models', model)()
    elif 'VGG' in model:
        return VGG(vgg_name=model, k=k)
    elif 'LeNet' in model:
        return LeNet()
    elif 'FC' in model:
        return FC()
    else:
        raise NotImplementedError('unknown {} model'.format(model))


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


def manage_state(net, ckpt_state):
    net_state = net.state_dict()
    for name, _ in net_state.items():
        if name in ckpt_state:
            net_state[name] = ckpt_state[name]
        # TODO: rewrite previous checkpoints and delete this
        elif 'module.{}'.format(name) in ckpt_state:
            net_state[name] = ckpt_state['module.{}'.format(name)]
    return net_state


def load_model(filename, print_info=False):
    use_cuda = torch.cuda.is_available()
    chekpoint = torch.load(filename)
    # TODO: add net kwargs
    net = get_model(chekpoint['name'], **chekpoint.get('model_args', {}))
    net.load_state_dict(manage_state(net, chekpoint['state_dict']))
    if use_cuda:
        net = DataParallel(net, device_ids=range(torch.cuda.device_count()))

    if print_info:
        print('Net validation accuracy = {}'.format(chekpoint['test_accuracy']))

    return net


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def load_optim(filename, print_info=False, n_classes=10):
    use_cuda = torch.cuda.is_available()
    chekpoint = torch.load(filename)
    # TODO: add net kwargs
    net = get_model(chekpoint['name'])
    if use_cuda:
        net = DataParallel(net, device_ids=range(torch.cuda.device_count()))

    net.load_state_dict(manage_state(net, chekpoint['state_dict']))

    if print_info:
        print('Net validation accuracy = {}'.format(chekpoint['test_accuracy']))

    return net


def set_bn_mode(net, mode):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.set_mode(mode)


def set_bn_sample_weight(net, val):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.sample_weight = val


def to_np(x):
    return x.data.cpu().numpy()


def log_params_info(net, writer, step):
    for name, param in net.named_parameters():
        try:
            writer.add_histogram('{}/grad'.format(name), to_np(param.grad), step, bins='auto')
        except:
            print('---- ', name)


class CIFAR(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with several classes.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, classes=None):

        if classes is None:
            classes = np.arange(10)

        self.classes = classes[:]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            mask = np.isin(self.train_labels, classes)
            self.train_labels = [classes.index(l) for l, cond in zip(self.train_labels, mask) if cond]

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))[mask]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            mask = np.isin(self.test_labels, classes)
            self.test_labels = [classes.index(l) for l, cond in zip(self.test_labels, mask) if cond]

            self.test_data = self.test_data.reshape((10000, 3, 32, 32))[mask]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

