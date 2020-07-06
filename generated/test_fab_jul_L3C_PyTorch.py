import sys
_module = sys.modules[__name__]
del sys
auto_crop = _module
bitcoding = _module
bitcoding = _module
coders = _module
coders_helpers = _module
part_suffix_helper = _module
blueprints = _module
multiscale_blueprint = _module
criterion = _module
logistic_mixture = _module
dataloaders = _module
images_loader = _module
helpers = _module
aligned_printer = _module
config_checker = _module
global_config = _module
logdir_helpers = _module
pad = _module
paths = _module
rolling_buffer = _module
saver = _module
testset = _module
import_train_images = _module
import_train_images_v1 = _module
l3c = _module
modules = _module
edsr = _module
head = _module
multiscale_network = _module
net = _module
prob_clf = _module
quantizer = _module
pytorch_ext = _module
test = _module
cuda_timer = _module
image_saver = _module
multiscale_tester = _module
setup = _module
torchac = _module
train = _module
lr_schedule = _module
multiscale_trainer = _module
train_restorer = _module
trainer = _module
vis = _module
figure_plotter = _module
grid = _module
histogram_plot = _module
histogram_plotter = _module
image_summaries = _module
safe_summary_writer = _module
summarizable_module = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import itertools


import torch


import functools


import numpy as np


from collections import namedtuple


import torch.nn.functional as F


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.nn import functional as F


import time


import re


from torch.optim import optimizer


import torch.backends.cudnn


import torch.nn as nn


from torch import nn


from torch import nn as nn


from collections import defaultdict


import collections


from torchvision import transforms


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch import optim


from torch.utils.data import DataLoader


from torchvision import transforms as transforms


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), atrous=False):
        super(ResBlock, self).__init__()
        m = []
        _repr = []
        for i in range(2):
            atrous_rate = 1 if not atrous or i == 0 else atrous
            m.append(conv(n_feats, n_feats, kernel_size, rate=atrous_rate, bias=bias))
            _repr.append(f'Conv({n_feats}x{kernel_size}' + (f';A*{atrous_rate})' if atrous_rate != 1 else '') + ')')
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f'BN({n_feats})')
            if i == 0:
                m.append(act)
                _repr.append(f'Act')
        self.body = nn.Sequential(*m)
        self._repr = '/'.join(_repr)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def __repr__(self):
        return f'ResBlock({self._repr})'


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class Head(nn.Module):
    """
    Go from Cin channels to Cf channels.
    For L3C, Cin=Cf, and this is the convolution yielding E^{s+1}_in in Fig. 2.

    """

    def __init__(self, config_ms, Cin):
        super(Head, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        self.head = conv(Cin, config_ms.Cf, config_ms.kernel_size)
        self._repr = f'Conv({config_ms.Cf})'

    def __repr__(self):
        return f'Head({self._repr})'

    def forward(self, x):
        return self.head(x)


class RGBHead(nn.Module):
    """ Go from 3 channels (RGB) to Cf channels, also normalize RGB """

    def __init__(self, config_ms):
        super(RGBHead, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        self.head = nn.Sequential(edsr.MeanShift(0, (0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), Head(config_ms, Cin=3))
        self._repr = 'MeanShift//Head(C=3)'

    def __repr__(self):
        return f'RGBHead({self._repr})'

    def forward(self, x):
        return self.head(x)


EncOut = namedtuple('EncOut', ['bn', 'bn_q', 'S', 'L', 'F'])


def _tensor_to_image(t):
    assert t.shape[0] == 3, t.shape
    return Image.fromarray(t.permute(1, 2, 0).detach().cpu().numpy())


def to_tensor_not_normalized(pic):
    """ copied from PyTorch functional.to_tensor, removed final .float().div(255.) """
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    return img


def resize_bicubic(t, fac):
    img = _tensor_to_image(t)
    h, w = img.size
    img = img.resize((int(h * fac), int(w * fac)), Image.BICUBIC)
    t = to_tensor_not_normalized(img)
    return t


def resize_bicubic_batch(t, fac):
    assert len(t.shape) == 4
    N = t.shape[0]
    return torch.stack([resize_bicubic(t[n, ...], fac) for n in range(N)], dim=0)


DecOut = namedtuple('DecOut', ['F'])


class EDSRDec(nn.Module):

    def __init__(self, config_ms, scale):
        super(EDSRDec, self).__init__()
        self.scale = scale
        n_resblock = config_ms.dec.num_blocks
        Cf = config_ms.Cf
        kernel_size = config_ms.kernel_size
        C = config_ms.q.C
        after_q_kernel = 1
        self.head = conv(C, config_ms.Cf, after_q_kernel)
        m_body = [edsr.ResBlock(conv, Cf, kernel_size, act=nn.ReLU(True)) for _ in range(n_resblock)]
        m_body.append(conv(Cf, Cf, kernel_size))
        self.body = nn.Sequential(*m_body)
        self.tail = edsr.Upsampler(conv, 2, Cf, act=False)

    def forward(self, x, features_to_fuse=None):
        """
        :param x: NCHW
        :return:
        """
        x = self.head(x)
        if features_to_fuse is not None:
            x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return DecOut(x)


class Quantizer(nn.Module):

    def __init__(self, levels, sigma=1.0):
        super(Quantizer, self).__init__()
        assert levels.dim() == 1, 'Expected 1D levels, got {}'.format(levels)
        self.levels = levels
        self.sigma = sigma
        self.L = self.levels.size()[0]

    def __repr__(self):
        return '{}(sigma={})'.format(self._get_name(), self.sigma)

    def forward(self, x):
        """
        :param x: NCHW
        :return:, x_soft, symbols
        """
        assert x.dim() == 4, 'Expected NCHW, got {}'.format(x.size())
        N, C, H, W = x.shape
        x = x.view(N, C, H * W, 1)
        d = torch.pow(x - self.levels, 2)
        phi_soft = F.softmax(-self.sigma * d, dim=-1)
        x_soft = torch.sum(self.levels * phi_soft, dim=-1)
        x_soft = x_soft.view(N, C, H, W)
        _, symbols_hard = torch.min(d.detach(), dim=-1)
        symbols_hard = symbols_hard.view(N, C, H, W)
        x_hard = self.levels[symbols_hard]
        x_soft.data = x_hard
        return x_soft, x_hard, symbols_hard


class Net(nn.Module):

    def __init__(self, config_ms, scale):
        super(Net, self).__init__()
        self.config_ms = config_ms
        self.enc = {'EDSRLikeEnc': EDSRLikeEnc, 'BicubicSubsampling': BicubicDownsamplingEnc}[config_ms.enc.cls](config_ms, scale)
        self.dec = {'EDSRDec': EDSRDec}[config_ms.dec.cls](config_ms, scale)

    def forward(self, x):
        raise NotImplementedError()


class StackedAtrousConvs(nn.Module):

    def __init__(self, atrous_rates_str, Cin, Cout, bias=True, kernel_size=3):
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList([conv(Cin, Cin, kernel_size, rate=rate) for rate in atrous_rates])
        self.lin = conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str):
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self):
        return self._extra_repr

    def forward(self, x):
        x = torch.cat([atrous(x) for atrous in self.atrous], dim=1)
        x = self.lin(x)
        return x


_NUM_PARAMS_OTHER = 3


_NUM_PARAMS_RGB = 4


def non_shared_get_Kp(K, C):
    """ Get Kp=number of channels to predict. See note where we define _NUM_PARAMS_RGB above """
    if C == 3:
        return _NUM_PARAMS_RGB * C * K
    else:
        return _NUM_PARAMS_OTHER * C * K


class AtrousProbabilityClassifier(nn.Module):

    def __init__(self, config_ms, C=3, atrous_rates_str='1,2,4'):
        super(AtrousProbabilityClassifier, self).__init__()
        K = config_ms.prob.K
        Kp = non_shared_get_Kp(K, C)
        self.atrous = StackedAtrousConvs(atrous_rates_str, config_ms.Cf, Kp, kernel_size=config_ms.kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def __repr__(self):
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x):
        """
        :param x: NCfHW
        :return: NKpHW
        """
        return self.atrous(x)


class LambdaModule(nn.Module):

    def __init__(self, forward_lambda, name=''):
        super(LambdaModule, self).__init__()
        self.forward_lambda = forward_lambda
        self.description = 'LambdaModule({})'.format(name)

    def __repr__(self):
        return self.description

    def forward(self, x):
        return self.forward_lambda(x)


class ChannelToLogitsTranspose(nn.Module):

    def __init__(self, Cout, Lout):
        super(ChannelToLogitsTranspose, self).__init__()
        self.Cout = Cout
        self.Lout = Lout

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, self.Lout, self.Cout, H, W)
        return x

    def __repr__(self):
        return 'ChannelToLogitsTranspose(Cout={}, Lout={})'.format(self.Cout, self.Lout)


class LogitsToChannelTranspose(nn.Module):

    def __init__(self):
        super(LogitsToChannelTranspose, self).__init__()

    def forward(self, x):
        N, L, C, H, W = x.shape
        x = x.view(N, C * L, H, W)
        return x

    def __repr__(self):
        return 'LogitsToChannelTranspose()'


def one_hot(x, L, Ldim):
    """ add dim L at Ldim """
    assert Ldim >= 0 or Ldim == -1, f'Only supporting Ldim >= 0 or Ldim == -1: {Ldim}'
    out_shape = list(x.shape)
    if Ldim == -1:
        out_shape.append(L)
    else:
        out_shape.insert(Ldim, L)
    x = x.unsqueeze(Ldim)
    assert x.dim() == len(out_shape), (x.shape, out_shape)
    oh = torch.zeros(*out_shape, dtype=torch.float32, device=x.device)
    oh.scatter_(Ldim, x, 1)
    return oh


class OneHot(nn.Module):
    """
    Take long tensor x of some shape (N,d1,d2,...,dN) containing integers in [0, L),
    produces one hot encoding `out` of out_shape (N, d1, ..., L, ..., dN), where out_shape[Ldim] = L, containing
        out[n, i, ..., l, ..., j] == {1 if x[n, i, ..., j] == l
                                      0 otherwise
    """

    def __init__(self, L, Ldim=1):
        super(OneHot, self).__init__()
        self.L = L
        self.Ldim = Ldim

    def forward(self, x):
        return one_hot(x, self.L, self.Ldim)


def _convert_if_callable(v):
    if hasattr(v, '__call__'):
        return v()
    return v


def normalize_to_0_1(t):
    return t.add(-t.min()).div(t.max() - t.min() + 1e-05)


def iter_modules_of_class(root_module: nn.Module, cls):
    """
    Helpful for extending nn.Module. How to use:
    1. define new nn.Module subclass with some new instance methods, cls
    2. make your root module inherit from cls
    3. make some leaf module inherit from cls
    """
    for m in root_module.modules():
        if isinstance(m, cls):
            yield m


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LambdaModule,
     lambda: ([], {'forward_lambda': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogitsToChannelTranspose,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (OneHot,
     lambda: ([], {'L': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_fab_jul_L3C_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

