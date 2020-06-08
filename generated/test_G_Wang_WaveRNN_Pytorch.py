import sys
_module = sys.modules[__name__]
del sys
audio = _module
dataset = _module
distributions = _module
hparams = _module
loss_function = _module
lrschedule = _module
model = _module
preprocess = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


from torch.distributions import Beta


from torch.distributions import Normal


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch import optim


class ResBlock(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias
            =False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):

    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):

    def __init__(self, feat_dims, upsample_scales, compute_dims, res_blocks,
        res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims,
            res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = 1, scale * 2 + 1
            padding = 0, scale
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding,
                bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


def num_params(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    print('Trainable Parameters: %.3f million' % parameters)


def sample_from_beta_dist(y_hat):
    """
    y_hat (batch_size x seq_len x 2):
    
    """
    loc_y = y_hat.exp()
    alpha = loc_y[:, :, (0)].unsqueeze(-1)
    beta = loc_y[:, :, (1)].unsqueeze(-1)
    dist = Beta(alpha, beta)
    sample = dist.sample()
    sample = 2.0 * sample - 1.0
    return sample


def sample_from_gaussian(y_hat, log_std_min=-7.0, scale_factor=1.0):
    """y_hat (batch_size x seq_len x 2)
        y (batch_size x seq_len x 1)
    """
    assert y_hat.size(2) == 2
    mean = y_hat[:, :, :1]
    log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)
    dist = Normal(mean, torch.exp(log_std))
    sample = dist.sample()
    sample = torch.clamp(torch.clamp(sample, min=-scale_factor), max=
        scale_factor)
    del dist
    return sample


def inv_mulaw_quantize(x_mu, quantization_channels=256, cuda=False):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels. default: 256

    """
    mu = quantization_channels - 1.0
    if isinstance(x_mu, np.ndarray):
        x = x_mu / mu * 2 - 1.0
        x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.0) / mu
    elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
        if isinstance(x_mu, (torch.LongTensor, torch.cuda.LongTensor)):
            x_mu = x_mu.float()
        if cuda:
            mu = torch.FloatTensor([mu]).cuda()
        else:
            mu = torch.FloatTensor([mu])
        x = x_mu / mu * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0
            ) / mu
    return x


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_G_Wang_WaveRNN_Pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(ResBlock(*[], **{'dims': 4}), [torch.rand([4, 4, 64])], {})

    def test_001(self):
        self._check(MelResNet(*[], **{'res_blocks': 1, 'in_dims': 4, 'compute_dims': 4, 'res_out_dims': 4}), [torch.rand([4, 4, 64])], {})
