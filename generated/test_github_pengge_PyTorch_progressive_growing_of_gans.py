import sys
_module = sys.modules[__name__]
del sys
began = _module
debug = _module
h5tool = _module
base_model = _module
model = _module
test = _module
train = _module
train_no_tanh = _module
data = _module
logger = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import numpy as np


import matplotlib as mpl


import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec


import time


from torch.autograd import Variable


import torch


import torch.nn as nn


import torch.optim as optim


from torch.nn.parameter import Parameter


from torch.nn import functional as F


from torch.nn.init import kaiming_normal


from torch.nn.init import calculate_gain


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-08):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % self.eps


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """

    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = torch.mean(self.incoming.weight.data ** 2) ** 0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % self.incoming.__class__.__name__
        return self.__class__.__name__ + param_str


def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, axis=ax, **kwargs)
    return tensor


class MinibatchStatConcatLayer(nn.Module):
    """Minibatch stat concatenation layer.
    - averaging tells how much averaging to use ('all', 'spatial', 'none')
    """

    def __init__(self, averaging='all'):
        super(MinibatchStatConcatLayer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode' % self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-08)

    def forward(self, x):
        shape = list(x.size())
        target_shape = shape.copy()
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2, 3], keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0, 2, 3], keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] / self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % self.averaging


class MinibatchDiscriminationLayer(nn.Module):

    def __init__(self, num_kernels):
        super(MinibatchDiscriminationLayer, self).__init__()
        self.num_kernels = num_kernels

    def forward(self, x):
        pass


class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """

    def __init__(self, mode='mul', strength=0.2, axes=(0, 1), normalize=False):
        super(GDropLayer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x
        rnd_shape = [(s if axis in self.axes else 1) for axis, s in enumerate(x.size())]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1
        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str


class LayerNormLayer(nn.Module):
    """
    Layer normalization. Custom reimplementation based on the paper: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, incoming, eps=0.0001):
        super(LayerNormLayer, self).__init__()
        self.incoming = incoming
        self.eps = eps
        self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = x - mean(x, axis=range(1, len(x.size())))
        x = x * 1.0 / torch.sqrt(mean(x ** 2, axis=range(1, len(x.size())), keepdim=True) + self.eps)
        x = x * self.gain
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        param_str = '(incoming = %s, eps = %s)' % (self.incoming.__class__.__name__, self.eps)
        return self.__class__.__name__ + param_str


DEBUG = False


def resize_activations(v, so):
    """
    Resize activation tensor 'v' of shape 'si' to match shape 'so'.
    :param v:
    :param so:
    :return:
    """
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so) and si[0] == so[0]
    if si[1] > so[1]:
        v = v[:, :so[1]]
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = si[2] // so[2], si[3] // so[3]
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)
    if si[2] < so[2]:
        assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]
        v = F.upsample(v, scale_factor=so[2] // si[2], mode='nearest')
    if si[1] < so[1]:
        z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
        v = torch.cat([v, z], 1)
    return v


class GSelectLayer(nn.Module):

    def __init__(self, pre, chain, post):
        super(GSelectLayer, self).__init__()
        assert len(chain) == len(post)
        self.pre = pre
        self.chain = chain
        self.post = post
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N
        if y is not None:
            assert insert_y_at is not None
        min_level, max_level = int(np.floor(cur_level - 1)), int(np.ceil(cur_level - 1))
        min_level_weight, max_level_weight = int(cur_level + 1) - cur_level, cur_level - int(cur_level)
        _from, _to, _step = 0, max_level + 1, 1
        if self.pre is not None:
            x = self.pre(x)
        out = {}
        if DEBUG:
            None
        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)
            if DEBUG:
                None
            if level == min_level:
                out['min_level'] = self.post[level](x)
            if level == max_level:
                out['max_level'] = self.post[level](x)
                x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + out['max_level'] * max_level_weight
        if DEBUG:
            None
        return x


class DSelectLayer(nn.Module):

    def __init__(self, pre, chain, inputs):
        super(DSelectLayer, self).__init__()
        assert len(chain) == len(inputs)
        self.pre = pre
        self.chain = chain
        self.inputs = inputs
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N
        if y is not None:
            assert insert_y_at is not None
        max_level, min_level = int(np.floor(self.N - cur_level)), int(np.ceil(self.N - cur_level))
        min_level_weight, max_level_weight = int(cur_level + 1) - cur_level, cur_level - int(cur_level)
        _from, _to, _step = min_level + 1, self.N, 1
        if self.pre is not None:
            x = self.pre(x)
        if DEBUG:
            None
        if max_level == min_level:
            x = self.inputs[max_level](x)
            if max_level == insert_y_at:
                x = self.chain[max_level](x, y)
            else:
                x = self.chain[max_level](x)
        else:
            out = {}
            tmp = self.inputs[max_level](x)
            if max_level == insert_y_at:
                tmp = self.chain[max_level](tmp, y)
            else:
                tmp = self.chain[max_level](tmp)
            out['max_level'] = tmp
            out['min_level'] = self.inputs[min_level](x)
            x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + out['max_level'] * max_level_weight
            if min_level == insert_y_at:
                x = self.chain[min_level](x, y)
            else:
                x = self.chain[min_level](x)
        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)
            if DEBUG:
                None
        return x


class AEDSelectLayer(nn.Module):

    def __init__(self, pre, chain, nins):
        super(AEDSelectLayer, self).__init__()
        assert len(chain) == len(nins)
        self.pre = pre
        self.chain = chain
        self.nins = nins
        self.N = len(self.chain) // 2

    def forward(self, x, cur_level=None):
        if cur_level is None:
            cur_level = self.N
        max_level, min_level = int(np.floor(self.N - cur_level)), int(np.ceil(self.N - cur_level))
        min_level_weight, max_level_weight = int(cur_level + 1) - cur_level, cur_level - int(cur_level)
        _from, _to, _step = min_level, self.N, 1
        if self.pre is not None:
            x = self.pre(x)
        if DEBUG:
            None
        if max_level == min_level:
            in_max_level = 0
        else:
            in_max_level = self.chain[max_level](self.nins[max_level](x))
            if DEBUG:
                None
        for level in range(_from, _to, _step):
            if level == min_level:
                in_min_level = self.nins[level](x)
                target_shape = in_max_level.size() if max_level != min_level else in_min_level.size()
                x = min_level_weight * resize_activations(in_min_level, target_shape) + max_level_weight * in_max_level
            x = self.chain[level](x)
            if DEBUG:
                None
        from_, to_, step_ = self.N, 2 * self.N - min_level, 1
        for level in range(from_, to_, step_):
            x = self.chain[level](x)
            if level == 2 * self.N - min_level - 1:
                out_min_level = self.nins[level](x)
            if DEBUG:
                None
        if max_level == min_level:
            out_max_level = 0
        else:
            out_max_level = self.nins[2 * self.N - max_level - 1](self.chain[2 * self.N - max_level - 1](x))
        target_shape = out_max_level.size() if max_level != min_level else out_min_level.size()
        x = min_level_weight * resize_activations(out_min_level, target_shape) + max_level_weight * out_max_level
        if DEBUG:
            None
        return x


class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], 1)


class ReshapeLayer(nn.Module):

    def __init__(self, new_shape):
        super(ReshapeLayer, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        assert reduce(lambda u, v: u * v, self.new_shape) == reduce(lambda u, v: u * v, x.size()[1:])
        return x.view(-1, *self.new_shape)


def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)


def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_batchnorm:
        layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None, to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
    he_init(layers[-1], init, param)
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not nonlinearity == 'linear':
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Generator(nn.Module):

    def __init__(self, num_channels=1, resolution=32, label_size=0, fmap_base=4096, fmap_decay=1.0, fmap_max=256, latent_size=None, normalize_latents=True, use_wscale=True, use_pixelnorm=True, use_leakyrelu=True, use_batchnorm=False, tanh_at_end=None):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_size = latent_size
        self.normalize_latents = normalize_latents
        self.use_wscale = use_wscale
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.use_batchnorm = use_batchnorm
        self.tanh_at_end = tanh_at_end
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        if latent_size is None:
            latent_size = self.get_nf(0)
        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope) if self.use_leakyrelu else nn.ReLU()
        iact = 'leaky_relu' if self.use_leakyrelu else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_iact = 'tanh' if self.tanh_at_end else 'linear'
        pre = None
        lods = nn.ModuleList()
        nins = nn.ModuleList()
        layers = []
        if self.normalize_latents:
            pre = PixelNormLayer()
        if self.label_size:
            layers += [ConcatLayer()]
        layers += [ReshapeLayer([latent_size, 1, 1])]
        layers = G_conv(layers, latent_size, self.get_nf(1), 4, 3, act, iact, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
        net = G_conv(layers, latent_size, self.get_nf(1), 3, 1, act, iact, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
        lods.append(net)
        nins.append(NINLayer([], self.get_nf(1), self.num_channels, output_act, output_iact, None, True, self.use_wscale))
        for I in range(2, R):
            ic, oc = self.get_nf(I - 1), self.get_nf(I)
            layers = [nn.Upsample(scale_factor=2, mode='nearest')]
            layers = G_conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            net = G_conv(layers, oc, oc, 3, 1, act, iact, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            lods.append(net)
            nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True, self.use_wscale))
        self.output_layer = GSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / 2.0 ** (stage * self.fmap_decay)), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)


def D_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_layernorm:
        layers += [LayerNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Discriminator(nn.Module):

    def __init__(self, num_channels=1, resolution=32, label_size=0, fmap_base=4096, fmap_decay=1.0, fmap_max=256, mbstat_avg='all', mbdisc_kernels=None, use_wscale=True, use_gdrop=True, use_layernorm=False, sigmoid_at_end=False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        gdrop_strength = 0.0
        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        iact = 'leaky_relu'
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}
        nins = nn.ModuleList()
        lods = nn.ModuleList()
        pre = None
        nins.append(NINLayer([], self.num_channels, self.get_nf(R - 1), act, iact, negative_slope, True, self.use_wscale))
        for I in range(R - 1, 1, -1):
            ic, oc = self.get_nf(I), self.get_nf(I - 1)
            net = D_conv([], ic, ic, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            lods.append(nn.Sequential(*net))
            nin = []
            nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)
        net = []
        ic = oc = self.get_nf(1)
        if self.mbstat_avg is not None:
            net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            ic += 1
        net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, self.get_nf(0), 4, 0, act, iact, negative_slope, False, self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        if self.mbdisc_kernels:
            net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]
        oc = 1 + self.label_size
        lods.append(NINLayer(net, self.get_nf(0), oc, output_act, output_iact, None, True, self.use_wscale))
        self.output_layer = DSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / 2.0 ** (stage * self.fmap_decay)), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConcatLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (GDropLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinibatchDiscriminationLayer,
     lambda: ([], {'num_kernels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MinibatchStatConcatLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_github_pengge_PyTorch_progressive_growing_of_gans(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

