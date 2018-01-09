import numpy as np
import tempfile
import itertools as IT
import os
from torch.nn.parallel import DataParallel
import torch
from models import *
from models.stochbn import _MyBatchNorm
import importlib
import sys
import pickle
import torchvision
from torchvision import transforms
import PIL


def uniquify(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


class Ensemble:
    """
    Ensemble for classification. Take logits and average probabilities using softmax.
    """
    def __init__(self, save_logits=False):
        self.__n_estimators = 0
        self.cum_proba = 0
        self.logits = None
        if save_logits:
            self.logits = []

    def add_estimator(self, logits):
        """
        Add estimator to current ensemble. First call define number of objects (N) and number of classes (K).
        :param logits: ndarray of logits with shape (N, K)
        """
        if self.logits is not None:
            self.logits.append(np.copy(logits))
        l = np.exp(logits - logits.max(1)[:, np.newaxis])
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            self.cum_proba += l / l.sum(1)[:, np.newaxis]
            assert not np.isnan(self.cum_proba).any(), 'NaNs while computing softmax'
        except Exception as e:
            raise e
        self.__n_estimators += 1

    def get_proba(self):
        """
        :return: ndarray with probabilities of shape (N, K)
        """
        return self.cum_proba / self.__n_estimators

    def get_logits(self):
        return np.array(self.logits)


class AccCounter:
    """
    Class for count accuracy during pass through data with mini-batches.
    """
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        """
        Compute and save stats needed for overall accuracy.
        :param outputs: ndarray of predicted values (logits or probabilities)
        :param targets: ndarray of labels with the same length as first dimension of _outputs_
        """
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        """
        Compute current accuracy.
        :return: float accuracy.
        """
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        """
        Flush stats.
        :return:
        """
        self.__n_objects = 0
        self.__sum = 0


def softmax(logits, temp=1.):
    assert not np.isnan(logits).any(), 'NaNs in logits for softmax'
    if len(logits.shape) == 2:
        l = np.exp((logits - logits.max(1)[:, np.newaxis]) / temp)
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            return l / l.sum(1)[:, np.newaxis]
        except Exception as e:
            raise e
    else:
        l = np.exp((logits - logits.max(2)[:, :, np.newaxis]) / temp)
        assert not np.isnan(l).any(), 'NaNs while computing softmax with temp={}'.format(temp)
        l /= l.sum(2)[:, :, np.newaxis]
        return np.mean(l, axis=0)


def entropy_plot_xy(p):
    e = entropy(p)
    n = len(e)
    return sorted(e), np.arange(1, n + 1) / 1. / n


def acc_vs_conf(p, y, t_grid=None, fract=False):
    if t_grid is None:
        t_grid = np.linspace(0, 1, 100)
    c = y == p.argmax(1)
    conf = p.max(axis=1)
    m = conf[:, np.newaxis] > t_grid[np.newaxis]
    acc = np.logical_or(c[:, np.newaxis], ~m)
    if fract:
        return t_grid, acc.mean(0), m.mean(0)
    return t_grid, acc.mean(0)


def adjust_betas(opt, new_betas):
    """
    Update betas for Adam optimizer for all param groups.
    """
    assert isinstance(opt, torch.optim.Adam)
    for pg in opt.param_groups:
        pg['betas'] = new_betas


# class LRPolicy(object):
#     def __init__(self, opt, ):


class BetasPolicy(object):
    """
    Class for handling betas for Adam optimizer. Use if net have SBN.
    """
    def __init__(self, opt, dec=0.1):
        self.opt = opt
        self.n_step = 0
        self.init_betas = []
        for pg in opt.param_groups:
            self.init_betas.append(pg['betas'])

        self.prev_sw = 0.
        self.mult = 1.
        self.dec = dec

        self.max = 0.99999

    def step(self, sample_weight):
        """
        Call it every epoch.
        """
        self.n_step += 1
        for init_b, pg in zip(self.init_betas, self.opt.param_groups):
            b1, b2 = init_b
            if self.prev_sw < 1.:
                pg['betas'] = (b1 + (self.max - b1) * sample_weight, b2 + (self.max - b2) * sample_weight)
            else:
                pg['betas'] = (b1 + (self.max - b1) * max(0., self.mult), b2 + (self.max - b2) * max(0., self.mult))
                self.mult -= self.dec

        self.prev_sw = sample_weight


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


def get_model(model='ResNet18', **kwargs):
    # if 'ResNet' in model:
    #     return class_for_name('models', model)(n_classes=kwargs.get('n_classes', 10))
    if model == 'ResNet18':
        return ResNet18(n_classes=kwargs.get('n_classes', 10))
    elif 'VGG' in model:
        return VGG(vgg_name=model, k=kwargs['k'], dropout=kwargs.get('dropout', None),
                   n_classes=kwargs.get('n_classes', 10), )
    elif 'LeNet' == model:
        return LeNet()
    elif 'FC' == model:
        return FC()
    # elif 'LeNetCifar' == model:
    #     return LeNetCifar(n_classes=kwargs.get('n_classes', None), dropout=kwargs.get('dropout', None))
    else:
        raise NotImplementedError('unknown {} model'.format(model))


def get_dataloader(data='cifar', train_bs=128, test_bs=200, augmentation=True,
                   noiid=False, shuffle=True, data_root='./data',
                   drop_last_train=False, drop_last_test=False):
    transform_train = transforms.Compose([
        MyPad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if data == 'cifar':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    elif data == 'SVHN':
        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True,
                                             transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform_test)
    elif data == 'mnist':
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
    elif data == 'cifar5':
        CIFAR5_CLASSES = [0, 1, 2, 3, 4]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    elif data == 'cifar5-rest':
        CIFAR5_CLASSES = [5, 6, 7, 8, 9]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    else:
        raise NotImplementedError

    if noiid:
        if data != 'cifar':
            raise NotImplementedError
        noiidsampler = CIFARNoIIDSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, sampler=noiidsampler,
                                                  num_workers=2, drop_last=drop_last_train)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle,
                                                  num_workers=2, drop_last=drop_last_train)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False,
                                             num_workers=2, drop_last=drop_last_test)

    return trainloader, testloader


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


# def test_batch_avg(net, data, labels, n_tries=1, seed=42):
#     np.random.seed(42)
#
#     ens = Ensemble()
#     for _ in range(n_infer):
#         logits = np.zeros([acc_data.shape[0], 10])
#         if args.augmentation:
#             acc_data = np.array(list(map(lambda x: transform_train(x).numpy(),
#                                          testset.test_data if args.acc == 'test' else trainset.train_data)))
#         else:
#             acc_data = np.array(list(map(lambda x: transform_test(x).numpy(),
#                                          testset.test_data if args.acc == 'test' else trainset.train_data)))
#
#         if args.permute:
#             perm = np.random.permutation(np.arange(acc_data.shape[0]))
#         else:
#             perm = np.arange(acc_data.shape[0])
#
#         for i in range(0, len(perm), BS):
#             idxs = perm[i: i + BS]
#             inputs = Variable(torch.Tensor(acc_data[idxs]).cuda(async=True))
#             outputs = net(inputs)
#             assert np.allclose(logits[idxs], 0.)
#             logits[idxs] = outputs.cpu().data.numpy()
#
#         ens.add_estimator(logits)
#
#     counter = AccCounter()
#     counter.add(ens.get_proba(), acc_labels)


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
    # net = get_model(chekpoint['name'], **chekpoint.get('model_args', {}))
    net = get_model(**chekpoint.get('script_args', {}))
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


def set_bn_mode(net, mode, update_policy='after', sample_policy='one'):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.set_mode_policy(mode, update_policy=update_policy, sample_policy=sample_policy)


def set_sample_policy(net, sample_policy='one'):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.set_sample_policy(sample_policy=sample_policy)


def set_bn_sample_weight(net, val):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.sample_weight = val


def set_bn_params(net, **kwargs):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            for k, v in kwargs.iteritems():
                setattr(m, k, v)


def to_np(x):
    return x.data.cpu().numpy()


def entropy(p):
    eps = 1e-8
    assert np.all(p >= 0)
    return np.apply_along_axis(lambda x: -np.sum(x[x > eps] * np.log(x[x > eps])), 1, p)


def log_params_info(net, writer, step):
    for name, param in net.named_parameters():
        try:
            writer.add_histogram('{}/grad'.format(name), to_np(param.grad), step, bins='auto')
        except:
            print('---- ', name)


def ensemble(net, data, bs, n_infer=50, return_logits=False):
    """ Ensemble for net training with Vanilla BN """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ens = Ensemble(save_logits=return_logits)
    acc_data = np.array(list(map(lambda x: transform_test(x).numpy(), data)))
    logits = []
    for _ in range(n_infer):
        logits = np.zeros([acc_data.shape[0], 5])
        perm = np.random.permutation(np.arange(acc_data.shape[0]))

        for i in range(0, len(perm), bs):
            idxs = perm[i: i + bs]
            inputs = Variable(torch.Tensor(acc_data[idxs]).cuda(async=True))
            outputs = net(inputs)
            assert np.allclose(logits[idxs], 0.)
            logits[idxs] = outputs.cpu().data.numpy()

        ens.add_estimator(logits)
    return ens.get_proba(), ens.get_logits()


def predict_proba_batch(batch, net, ensembles=1):
    ens = Ensemble()
    img = Variable(batch).cuda()
    for _ in range(ensembles):
        pred = net(img).data.cpu().numpy()
        ens.add_estimator(pred)
    return ens.get_proba()


def bn_ensemble(net, testloader, virtualloader, n_ensembles=50, use_cuda=True, return_logits=False,
                vanilla=False):
    def foo():
        while True:
            for x, _ in virtualloader:
                yield x

    virt_iterator = foo()

    proba = []
    labels = []
    logits = []

    for x, y in testloader:
        ens = Ensemble(save_logits=return_logits)
        set_bn_params(net, nobj=x.shape[0])
        for _ in range(n_ensembles):
            virt_batch = next(virt_iterator)
            batch = Variable(torch.cat((x, virt_batch)))
            if use_cuda:
                batch = batch.cuda()
            pred = net(batch).data.cpu().numpy()
            if vanilla:
                ens.add_estimator(pred[: x.shape[0]])
            else:
                ens.add_estimator(pred[:])
        proba.append(ens.get_proba())
        labels.append(y.tolist())
        if return_logits:
            logits.append(ens.get_logits())

    if return_logits:
        logits = np.stack(logits)
        logits = logits.transpose(0, 2, 1, 3)
        logits = np.concatenate(logits, axis=0)
        logits = logits.transpose(1, 0, 2)

        return np.concatenate(proba), np.concatenate(labels), logits
    return np.concatenate(proba), np.concatenate(labels)


def predict_proba(dataloader, net, ensembles=1, n_classes=10, return_logits=False):
    proba = np.zeros((len(dataloader.dataset), n_classes))
    labels = []
    logits = []
    p = 0
    for img, label in dataloader:
        ens = Ensemble(save_logits=return_logits)
        img = Variable(img).cuda()
        for _ in range(ensembles):
            pred = net(img).data.cpu().numpy()
            ens.add_estimator(pred)
        proba[p: p + pred.shape[0]] = ens.get_proba()
        p += pred.shape[0]
        labels += label.tolist()
        if return_logits:
            logits.append(ens.get_logits())

    if return_logits:
        logits = np.stack(logits)
        logits = logits.transpose(0, 2, 1, 3)
        logits = np.concatenate(logits, axis=0)
        logits = logits.transpose(1, 0, 2)
        return proba, np.array(labels), logits
    return proba, np.array(labels)


def uncertainty_acc(net, known=None, unknown=None, ensembles=50, bn_type='StochBN', n_classes=5,
                    sample_policy='one', bs=None, vanilla_known=None, vanilla_unknown=None, n_sbn=None):
    net.eval()
    set_MyBN_strategy(net, mean_strategy='running', var_strategy='running')
    bns = [m for m in net.modules() if isinstance(m, _MyBatchNorm)]
    kn, unkn = {}, {}
    net.eval()
    p, l = predict_proba(unknown, net, n_classes=n_classes)
    unkn['eval/entropy'] = entropy(p)
    unkn['eval/proba'] = np.copy(p)

    p, l = predict_proba(known, net, n_classes=n_classes)
    kn['eval/entropy'] = entropy(p)
    kn['eval/acc'] = np.mean(p.argmax(1) == l)
    kn['eval/proba'] = np.copy(p)
    kn['eval/labels'] = np.copy(l)

    if bn_type == 'StochBN':
        set_MyBN_strategy(net, mean_strategy='sample', var_strategy='sample')
        if n_sbn:
            for bn in bns[:len(bns) - n_sbn]:
                bn.mean_strategy = 'batch'
                bn.var_strategy = 'batch'

        p, l = predict_proba(unknown, net, ensembles=ensembles, n_classes=n_classes)
        unkn['ensemble/entropy'] = entropy(p)
        unkn['ensemble/proba'] = np.copy(p)

        p, l = predict_proba(known, net, ensembles=ensembles, n_classes=n_classes)
        kn['ensemble/entropy'] = entropy(p)
        kn['ensemble/acc'] = np.mean(p.argmax(1) == l)
        kn['ensemble/proba'] = np.copy(p)
        kn['ensemble/labels'] = np.copy(l)

        set_MyBN_strategy(net, mean_strategy='sample', var_strategy='sample')
        if n_sbn:
            for bn in bns[:len(bns) - n_sbn]:
                bn.mean_strategy = 'batch'
                bn.var_strategy = 'batch'

        p, l = predict_proba(unknown, net, ensembles=1, n_classes=n_classes)
        unkn['one_shot/entropy'] = entropy(p)
        unkn['one_shot/proba'] = np.copy(p)

        p, l = predict_proba(known, net, ensembles=1, n_classes=n_classes)
        kn['one_shot/entropy'] = entropy(p)
        kn['one_shot/acc'] = np.mean(p.argmax(1) == l)
        kn['one_shot/proba'] = np.copy(p)
        kn['one_shot/labels'] = np.copy(l)

    elif bn_type == 'BN':
        data, labels = vanilla_unknown
        set_MyBN_strategy(net, mean_strategy='batch', var_strategy='batch')
        ens_p = ensemble(net, data, bs, n_infer=ensembles)
        unkn['ensemble/entropy'] = entropy(ens_p)
        unkn['ensemble/proba'] = np.copy(ens_p)

        data, labels = vanilla_known
        ens_p = ensemble(net, data, bs, n_infer=ensembles)
        kn['ensemble/entropy'] = entropy(ens_p)
        kn['ensemble/proba'] = np.copy(ens_p)
        kn['ensemble/labels'] = np.copy(labels)
        kn['ensemble/acc'] = np.mean(ens_p.argmax(1) == labels)

        data, labels = vanilla_unknown
        set_MyBN_strategy(net, mean_strategy='batch', var_strategy='batch')
        ens_p = ensemble(net, data, bs, n_infer=1)
        unkn['one_shot/entropy'] = entropy(ens_p)
        unkn['one_shot/proba'] = np.copy(ens_p)

        data, labels = vanilla_known
        ens_p = ensemble(net, data, bs, n_infer=1)
        kn['one_shot/entropy'] = entropy(ens_p)
        kn['one_shot/proba'] = np.copy(ens_p)
        kn['one_shot/labels'] = np.copy(labels)
        kn['one_shot/acc'] = np.mean(ens_p.argmax(1) == labels)
    else:
        raise NotImplementedError

    return unkn, kn


class CIFARNoIIDSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        idxs = np.array(list(range(len(self.data_source))))
        idxs = idxs[np.argsort(self.data_source.train_labels)]
        return iter(idxs)

    def __len__(self):
        return len(self.data_source)


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
                 download=False, classes=None, random_labeling=False):

        if classes is None:
            classes = np.arange(10).tolist()

        self.classes = classes[:]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.random_labeling = random_labeling

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
            if self.random_labeling:
                self.train_labels = np.random.permutation(self.train_labels)

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


def fast_adversarial(x, y, net, loss, eps, is_cuda=True):
    """
    Implement fast gradient sign method for constructing adversarial example

    :param x: (torch.Tensor) Batch of objects
    :param y: target for x
    :param net: (nn.Module) model for which adversarial example constructed
    :param loss: loss function which gradient w.t. net parameters
    :param eps:
    :param is_cuda:
    :return: (torch.Tensor) batch of adversarial examples for each
    """

    if is_cuda:
        inp = Variable(x.cuda(), requires_grad=True)
        target = Variable(y.cuda())
    else:
        inp = Variable(x, requires_grad=True)
        target = Variable(y)

    pred = net(inp)
    _l = loss(pred, target)
    _l.backward()

    return inp.data.cpu() + eps * torch.sign(inp.grad.data.cpu())
