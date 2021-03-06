import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable


def mean_features(x):
    out = x.mean(0)
    while out.dim() > 1:
        out = out.mean(1)
    return out


class _MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, mode=None,
                 learn_stats=False, batch_size=0):
        super(_MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.stats_momentum = momentum
        self.sample_weight = 1.
        self.learn_stats = learn_stats
        self.bs = batch_size
        self.uncorr_type = 'many-batch'
        self.correction = False
        self.nobj = None
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

        self.__global_mode = mode
        self.__update_policy = 'after'
        self.__sample_policy = 'one'
        self.mode = 'vanilla'
        self.n_samples = 0
        self.mean_strategy = 'batch'
        self.var_strategy = 'batch'

        self._sum_m = 0

        self.register_buffer('running_m', torch.zeros(num_features))
        self.register_buffer('running_m2', torch.zeros(num_features))
        self.register_buffer('running_logvar', torch.zeros(num_features))
        self.register_buffer('running_logvar2', torch.zeros(num_features))

        if self.learn_stats:
            self.running_mean_mean = Parameter(torch.Tensor(self.num_features))
            self.running_mean_var = Parameter(torch.Tensor(self.num_features))
            self.running_logvar_mean = Parameter(torch.Tensor(self.num_features))
            self.running_logvar_var = Parameter(torch.Tensor(self.num_features))
        else:
            self.register_buffer('running_mean_mean', torch.zeros(num_features))
            self.register_buffer('running_mean_var', torch.ones(num_features))
            self.register_buffer('running_logvar_mean', torch.zeros(num_features))
            self.register_buffer('running_logvar_var', torch.ones(num_features))

        self.reset_parameters()

    def global_mode(self):
        return self.__global_mode

    def set_mode_policy(self, mode, update_policy='after', sample_policy='one'):
        if self.__global_mode is None:
            self.__global_mode = mode
            self.__update_policy = update_policy
            self.__sample_policy = sample_policy
        else:
            raise AssertionError("Don't change type of BN layer!")

    def set_sample_policy(self, sample_policy='one'):
        self.__sample_policy = sample_policy

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
        self.sum_var2 += self.cur_var ** 2

        self.n_samples += 1

        if self.n_samples > 1:
            self.mean_mean = self.sum_m / self.n_samples
            self.mean_var.copy_(self.sum_m2 - 2 * self.mean_mean * self.sum_m + self.n_samples * (self.mean_mean ** 2))
            self.mean_var /= (self.n_samples - 1)

            s = torch.log(self.sum_var / self.n_samples) - self.sum_logvar / self.n_samples
            self.s = s.cpu().numpy()
            self.var_shape.copy_((3 - s + torch.sqrt((s - 3) ** 2 + 24 * s)) / 12. / s)
            self.var_scale.copy_(self.sum_var / self.var_shape / self.n_samples)

    def update_smoothed_stats(self):
        eps = 1e-6
        self.running_m = (1 - self.stats_momentum) * self.running_m + self.stats_momentum * self.cur_mean
        self.running_m2 = (1 - self.stats_momentum) * self.running_m2 + self.stats_momentum * (self.cur_mean ** 2)

        self.running_logvar = (1 - self.stats_momentum) * self.running_logvar + self.stats_momentum * torch.log(self.cur_var + eps)
        self.running_logvar2 = (1 - self.stats_momentum) * self.running_logvar2 + self.stats_momentum * (torch.log(self.cur_var + eps) ** 2)

        self.running_mean_mean.copy_(self.running_m)
        self.running_mean_var.copy_(self.running_m2 - (self.running_m ** 2))

        # assert not np.any(np.isnan(self.running_logvar.cpu().numpy()))
        # assert not np.any(np.isnan(self.running_logvar2.cpu().numpy()))

        # assert not np.any(np.isinf(self.running_logvar.cpu().numpy()))
        # assert not np.any(np.isinf(self.running_logvar2.cpu().numpy()))

        self.running_logvar_mean.copy_(self.running_logvar)
        self.running_logvar_var.copy_(self.running_logvar2 - (self.running_logvar ** 2))

        self.running_mean.copy_(self.running_m)
        self.running_var.copy_(torch.exp(self.running_logvar))

    def flush_stats(self):
        self.sum_m.zero_()
        self.sum_m2.zero_()

        self.sum_logvar.zero_()
        self.sum_var.zero_()

        self.n_samples = 0

    def forward_id(self, input):
        """
        Implement identity BN layer for easily testing models with no BN layers without changing models implementation.
        :param input: tensor
        :return: input
        """
        return input

    def forward_stochbn(self, input):
        cur_mean = mean_features(input)
        cur_var = F.relu(mean_features(input**2) - cur_mean**2)

        # try:
        #     assert not np.any(np.isnan(input.data.cpu().numpy()))
        #     assert not np.any(np.isnan(cur_mean.data.cpu().numpy()))
        #     assert not np.any(np.isnan(cur_var.data.cpu().numpy()))
        #     assert np.all(cur_var.data.cpu().numpy() >= 0)
        # except:
        #     np.save('cur_var', cur_var.data.cpu().numpy())
        #     # print(self.n_bn_layer)
        #     raise

        self.cur_var.copy_(cur_var.data)
        self.cur_mean.copy_(cur_mean.data)

        running_mean_mean = Variable(self.running_mean_mean, requires_grad=False)
        running_mean_var = Variable(self.running_mean_var, requires_grad=False)

        running_logvar_mean = Variable(self.running_logvar_mean, requires_grad=False)
        running_logvar_var = Variable(self.running_logvar_var, requires_grad=False)

        if self.__update_policy == 'before' and self.training and (not self.learn_stats):
            running_m = Variable((1 - self.stats_momentum) * self.running_m) + self.stats_momentum * cur_mean
            running_m2 = Variable((1 - self.stats_momentum) * self.running_m2) + self.stats_momentum * (cur_mean ** 2)

            running_logvar = Variable((1 - self.stats_momentum) * self.running_logvar) + self.stats_momentum * torch.log(cur_var)
            running_logvar2 = Variable((1 - self.stats_momentum) * self.running_logvar2) + self.stats_momentum * (torch.log(cur_var) ** 2)

            running_mean_mean = running_m
            running_mean_var = running_m2 - (running_m ** 2)

            running_logvar_mean = running_logvar
            running_logvar_var = running_logvar2 - (running_logvar ** 2)

        if self.var_strategy == 'sample':
            if self.__sample_policy == 'one':
                eps = Variable(torch.randn(self.num_features))
                if self.weight.data.is_cuda:
                    eps = eps.cuda()
                sampled_var = torch.exp(eps * torch.sqrt(running_logvar_var) + running_logvar_mean)
                vars = cur_var * (1. - self.sample_weight) + self.sample_weight * sampled_var
            elif self.__sample_policy == 'bs':
                eps = Variable(torch.randn(input.size(0), self.num_features))
                if self.weight.data.is_cuda:
                    eps = eps.cuda()
                logvar = eps * torch.sqrt(running_logvar_var).view(1, -1) + running_logvar_mean.view(1, -1)
                sampled_var = torch.exp(logvar)
                vars = cur_var.view(1, self.num_features) * (1. - self.sample_weight) + self.sample_weight * sampled_var
            else:
                raise NotImplementedError                
        elif self.var_strategy == 'running-mean':
            vars = Variable(torch.exp(self.running_logvar_mean + self.running_logvar_var / 2))
        elif self.var_strategy in ['running', 'running-median']:
            vars = Variable(self.running_var)
        elif self.var_strategy == 'batch':
            vars = cur_var
        else:
            raise NotImplementedError('Unknown var strategy: {}'.format(self.var_strategy))

        if self.mean_strategy == 'sample':
            if self.__sample_policy == 'one':
                eps = Variable(torch.randn(self.num_features))
                if self.weight.data.is_cuda:
                    eps = eps.cuda()

                sampled_mean = eps * torch.sqrt(running_mean_var) + running_mean_mean
                means = cur_mean * (1. - self.sample_weight) + self.sample_weight * sampled_mean
            elif self.__sample_policy == 'bs':
                eps = Variable(torch.randn(input.size(0), self.num_features))
                if self.weight.data.is_cuda:
                    eps = eps.cuda()
                sampled_mean = eps * torch.sqrt(running_mean_var).view(1, -1) + running_mean_mean.view(1, -1)
                means = cur_mean.view(1, -1) * (1. - self.sample_weight) + self.sample_weight * sampled_mean
            else:
                raise NotImplementedError
        elif self.mean_strategy == 'running':
            means = Variable(self.running_mean)
        elif self.mean_strategy == 'batch':
            means = cur_mean
        else:
            raise NotImplementedError('Unknown mean strategy: {}'.format(self.mean_strategy))

        if self.training and (not self.learn_stats):
            self.update_smoothed_stats()
        else:
            self.running_mean.copy_(self.running_mean_mean)
            self.running_var.copy_(torch.exp(self.running_logvar_mean))

        if not self.training and self.correction and self.mean_strategy == 'sample':
            means = means.view(-1, self.num_features, 1, 1)
            vars = vars.view(-1, self.num_features, 1, 1) + means ** 2
            means = (means * float(self.bs) + input.mean(2, keepdim=True).mean(3, keepdim=True)) / (1. + self.bs)
            vars = (vars * self.bs + (input ** 2).mean(2, keepdim=True).mean(3, keepdim=True)) / (self.bs + 1.)
            vars = vars - float(self.bs) / (self.bs + 1.) * (means ** 2)

        return self.batch_norm(input, means, vars + self.eps)

    def forward_vanilla(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

    def forward_smoothed_bn(self, input):
        cur_mean = mean_features(input)
        cur_var = mean_features(input**2) - cur_mean**2

        self.cur_var.copy_(cur_var.data)
        self.cur_mean.copy_(cur_mean.data)

        if np.isclose(self.sample_weight, 1.):
            means = Variable(self.running_mean, requires_grad=False)
            vars = Variable(self.running_var, requires_grad=False)
        elif np.isclose(self.sample_weight, 0.):
            means = cur_mean
            vars = cur_var
        else:
            means = Variable(self.sample_weight * self.running_mean, requires_grad=False) + (1 - self.sample_weight) * cur_mean
            vars = Variable(self.sample_weight * self.running_var, requires_grad=False) + (1 - self.sample_weight) * cur_var

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.cur_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.cur_var

        return self.batch_norm(input, means, vars + self.eps)

    def forward_uncorr(self, input):
        """
        Only for test phase! Not shure in grad flow.
        """
        assert (input.shape[0] - self.nobj) % self.bs == 0
        nvirt = (input.shape[0] - self.nobj) // self.bs

        cur_mean = mean_features(input[-self.bs:])
        cur_var = F.relu(mean_features(input[-self.bs:] ** 2) - cur_mean ** 2)

        if self.uncorr_type == 'one-batch':
            return self.batch_norm(input, cur_mean, cur_var + self.eps)

        virt = []
        for i in range(nvirt - 1):
            virt.append(F.batch_norm(input[self.nobj + self.bs * i: self.nobj + self.bs * (i + 1)],
                                     self.running_mean, self.running_var, self.weight, self.bias,
                                     True, 0., self.eps))

        obj = self.batch_norm(input[:self.nobj], cur_mean, cur_var + self.eps)

        return torch.cat([obj, ] + virt)

    def forward(self, input):
        self._check_input_dim(input)

        if self.__global_mode == 'StochBN':
            return self.forward_stochbn(input)
        elif self.__global_mode == 'vanilla':
            return self.forward_vanilla(input)
        elif self.__global_mode == 'no_bn':
            return self.forward_id(input)
        elif self.__global_mode == 'smoothed_bn':
            return self.forward_smoothed_bn(input)
        elif self.__global_mode == 'uncorr':
            return self.forward_uncorr(input)
        else:
            # warnings.warn('May be problem with this mode, because of refactoring!!!!', DeprecationWarning)
            raise NotImplementedError('No such BN mode {}'.format(self.__global_mode))

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class MyBatchNorm1d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def batch_norm(self, input, means, vars):
        # TODO: implement for any dimensionality
        out = input - means.view(-1, self.num_features)
        out = out / torch.sqrt(vars.view(-1, self.num_features))
        out = out * self.weight.view(1, self.num_features)
        out = out + self.bias.view(1, self.num_features)
        return out


class MyBatchNorm2d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)

    def batch_norm(self, input, means, vars):
        # TODO: implement for any dimensionality
        if means.dim() == 1:
            means = means.view(-1, self.num_features, 1, 1)
            vars = vars.view(-1, self.num_features, 1, 1)
        out = input - means
        out = out / torch.sqrt(vars)
        out = out * self.weight.view(1, self.num_features, 1, 1)
        out = out + self.bias.view(1, self.num_features, 1, 1)
        return out
