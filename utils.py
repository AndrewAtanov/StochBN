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
    def __init__(self):
        self.__n_estimators = 0
        self.cum_proba = 0

    def add_estimator(self, logits):
        """
        Add estimator to current ensemble. First call define number of objects (N) and number of classes (K).
        :param logits: ndarray of logits with shape (N, K)
        """
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
    if 'ResNet' in model:
        return class_for_name('models', model)()
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


def get_dataloader(data='cifar', bs=128, augmentation=True, noiid=False, shuffle=True, data_root='./data'):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=noiidsampler, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=shuffle, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

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


def ensemble(net, data, bs, n_infer=50):
    """ Ensemble for net training with Vanilla BN """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ens = Ensemble()
    acc_data = np.array(list(map(lambda x: transform_test(x).numpy(), data)))
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
    return ens.get_proba()


def predict_proba_batch(batch, net, ensembles=1):
    ens = Ensemble()
    img = Variable(batch).cuda()
    for _ in range(ensembles):
        pred = net(img).data.cpu().numpy()
        ens.add_estimator(pred)
    return ens.get_proba()


def predict_proba(dataloader, net, ensembles=1, n_classes=10):
    proba = np.zeros((len(dataloader.dataset), n_classes))
    labels = []
    p = 0
    for img, label in dataloader:
        ens = Ensemble()
        img = Variable(img).cuda()
        for _ in range(ensembles):
            pred = net(img).data.cpu().numpy()
            ens.add_estimator(pred)
        proba[p: p + pred.shape[0]] = ens.get_proba()
        p += pred.shape[0]
        labels += label.tolist()
    return proba, np.array(labels)


def uncertainty_acc(net, known, unknown, ensembles=50, bn_type='StochBN', n_classes=5,
                    sample_policy='one', bs=None, vanilla_known=None, vanilla_unknown=None):
    net.eval()
    set_MyBN_strategy(net, mean_strategy='running', var_strategy='running')
    kn, unkn = {}, {}
    net.eval()
    p, l = predict_proba(unknown, net, n_classes=n_classes)
    unkn['eval/entropy'] = entropy(p)

    p, l = predict_proba(known, net, n_classes=n_classes)
    kn['eval/entropy'] = entropy(p)
    kn['eval/acc'] = np.mean(p.argmax(1) == l)

    if bn_type == 'StochBN':
        set_MyBN_strategy(net, mean_strategy='sample', var_strategy='sample')
        p, l = predict_proba(unknown, net, ensembles=ensembles, n_classes=n_classes)
        unkn['ensemble/entropy'] = entropy(p)

        p, l = predict_proba(known, net, ensembles=ensembles, n_classes=n_classes)
        kn['ensemble/entropy'] = entropy(p)
        kn['ensemble/acc'] = np.mean(p.argmax(1) == l)

        set_MyBN_strategy(net, mean_strategy='sample', var_strategy='sample')
        p, l = predict_proba(unknown, net, ensembles=1, n_classes=n_classes)
        unkn['one_shot/entropy'] = entropy(p)

        p, l = predict_proba(known, net, ensembles=1, n_classes=n_classes)
        kn['one_shot/entropy'] = entropy(p)
        kn['one_shot/acc'] = np.mean(p.argmax(1) == l)

    elif bn_type == 'BN':
        data, labels = vanilla_unknown
        set_MyBN_strategy(net, mean_strategy='batch', var_strategy='batch')
        ens_p = ensemble(net, data, bs, n_infer=ensembles)
        unkn['ensemble/entropy'] = entropy(ens_p)

        data, labels = vanilla_known
        ens_p = ensemble(net, data, bs, n_infer=ensembles)
        kn['ensemble/entropy'] = entropy(ens_p)
        kn['ensemble/acc'] = np.mean(ens_p.argmax(1) == labels)

        data, labels = vanilla_unknown
        set_MyBN_strategy(net, mean_strategy='batch', var_strategy='batch')
        ens_p = ensemble(net, data, bs, n_infer=1)
        unkn['one_shot/entropy'] = entropy(ens_p)

        data, labels = vanilla_known
        ens_p = ensemble(net, data, bs, n_infer=1)
        kn['one_shot/entropy'] = entropy(ens_p)
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

    return inp.data + eps * torch.sign(inp.grad.data)
