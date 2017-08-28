import torch
import torch.nn as nn
import torch.nn.functional as F
# from .module import Module
from torch.nn.parameter import Parameter
import numpy as np


class _MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.register_buffer('cur_mean', torch.zeros(num_features))
        self.register_buffer('cur_var', torch.ones(num_features))

        self.collect = False
        self.mode = 'vanilla'
        self.n_samples = 0
        self.mean_strategy = 'vanilla'
        self.var_strategy = 'vanilla'
        self.train_mode = 'vanilla'
        self.test_mode = 'standart'
        self.s = None

        self._sum_m = 0

        self.register_buffer('sum_m', torch.zeros(num_features))
        self.register_buffer('sum_m2', torch.zeros(num_features))
        self.register_buffer('sum_logvar', torch.zeros(num_features))
        self.register_buffer('sum_var', torch.zeros(num_features))

        self.register_buffer('mean_mean', torch.zeros(num_features))
        self.register_buffer('mean_var', torch.ones(num_features))
        self.register_buffer('var_shape', torch.zeros(num_features))
        self.register_buffer('var_scale', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def update_stats(self):
        self._sum_m += self.cur_mean.cpu().numpy()

        self.sum_m += self.cur_mean
        self.sum_m2 += self.cur_mean ** 2

        self.sum_logvar += torch.log(self.cur_var)
        self.sum_var += self.cur_var

        self.n_samples += 1

        if self.n_samples > 1:
            self.mean_mean = self.sum_m / self.n_samples
            self.mean_var.copy_(self.sum_m2 - 2 * self.mean_mean * self.sum_m + self.n_samples * (self.mean_mean ** 2))
            self.mean_var /= (self.n_samples - 1)

            s = torch.log(self.sum_var / self.n_samples) - self.sum_logvar / self.n_samples
            self.s = s.cpu().numpy()
            self.var_shape.copy_((3 - s + torch.sqrt((s - 3) ** 2 + 24 * s)) / 12. / s)
            self.var_scale.copy_(self.sum_var / self.var_shape / self.n_samples)

    def flush_stats(self):
        self.sum_m.zero_()
        self.sum_m2.zero_()

        self.sum_logvar.zero_()
        self.sum_var.zero_()

        self.n_samples = 0

    def forward(self, input):
        self._check_input_dim(input)

        if self.collect:
            res = F.batch_norm(
                input, self.cur_mean, self.cur_var, self.weight, self.bias,
                True, 1., self.eps)
            self.update_stats()

            return res

        if self.training:
            if self.train_mode == 'vanilla':
                return F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight,
                    self.bias, self.training, self.momentum, self.eps)
            elif self.train_mode == 'collected-stats':
                F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight,
                    self.bias, True, self.momentum, self.eps)

                return F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight,
                    self.bias, False, self.momentum, self.eps)

        if 'sample-batch' in self.test_mode:
            is_cuda = self.running_mean.is_cuda
            dims = [-1] + [1] * (len(input.data[0].size()) - 1)
            size = input.data[0].size()
            bs = float(self.test_mode.split('-')[-1])
            k = float(sum(input.data.size()[2:]))

            x_mean = input.data[0].mean(1).mean(1)

            chi2 = torch.randn(int((bs - 1) * k - 1), self.num_features).sum(dim=0).squeeze()
            norm = torch.normal(self.running_mean, torch.sqrt(self.running_var))

            if is_cuda:
                norm = norm.cuda()
                chi2 = chi2.cuda()

            chi2 *= torch.sqrt(self.running_var) / (bs - 1) / k

            self.cur_mean.copy_(norm * (bs - 1) / bs + x_mean / bs)
            tmp = ((input.data[0] - self.cur_mean.view(dims).expand(size)) ** 2).sum(dim=1).sum(dim=1) / (bs * k - 1)
            tmp += ((bs - 1) * k - 1) / (bs * k - 1) / ((bs - 1) * k) * chi2
            tmp += ((x_mean - norm) ** 2) * (bs - 1) * k / bs / (bs * k - 1)
            self.cur_var.copy_(tmp)
            return F.batch_norm(input, self.cur_mean, self.cur_var, self.weight, self.bias,
                                False, self.momentum, self.eps)

        self.cur_mean.copy_(self.running_mean)


        if self.mean_strategy == 'random':
            self.cur_mean.copy_(torch.normal(self.mean_mean, torch.sqrt(self.mean_var)))
        elif self.mean_strategy == 'collected':
            self.cur_mean.copy_(self.mean_mean)

        self.cur_var.copy_(self.running_var)

        if self.var_strategy == 'random':
            cond = np.isnan(self.s) | (self.s < 1e-8)
            shape = self.var_shape.cpu().numpy()
            scale = self.var_scale.cpu().numpy()

            shape[cond] = 1
            scale[cond] = 1

            val = np.random.gamma(shape, scale)

            val[cond] = (self.sum_var / (self.n_samples * 1.)).cpu().numpy()[cond]
            self.cur_var.copy_(torch.Tensor(val))
        elif self.var_strategy == 'collected':
            self.cur_var.copy_(self.sum_var / (self.n_samples * 1.))

        try:
            return F.batch_norm(
                input, self.cur_mean, self.cur_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        except:
            print self, self.cur_mean, self.cur_var
            raise

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class MyBatchNorm1d(_MyBatchNorm):
    def __init__(self, *args, **kwargs):
        raise NotImplemented('')

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)


class MyBatchNorm2d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)


class MyBatchNorm3d(_MyBatchNorm):
    def __init__(self, *args, **kwargs):
        raise NotImplemented('')

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm3d, self)._check_input_dim(input)
