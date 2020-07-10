import sys
_module = sys.modules[__name__]
del sys
src = _module
configs = _module
data = _module
decode = _module
encode = _module
eval = _module
l3c = _module
bitcoding = _module
coders = _module
coders_helpers = _module
edsr = _module
logistic_mixture = _module
prob_clf = _module
quantizer = _module
timer = _module
network = _module
setup = _module
torchac = _module
train = _module
util = _module

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
xrange = range
wraps = functools.wraps


from typing import Iterable


from typing import List


from typing import Tuple


import numpy as np


import torch


import torch.utils.data as data


from torch.nn import functional as F


import torchvision.transforms as T


from torch.utils import data


from torch import nn


from typing import Union


import math


import torch.nn as nn


from typing import NamedTuple


from typing import Optional


from collections import defaultdict


from typing import DefaultDict


from typing import Generator


from typing import KeysView


import re


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch import optim


from torch.utils import tensorboard


def get_act(act: str, n_feats: int=0) ->nn.Module:
    """ param act: Name of activation used.
        n_feats: channel size.
        returns the respective activation module, or raise
            NotImplementedError if act is not implememted.
    """
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'prelu':
        return nn.PReLU(n_feats)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'none':
        return nn.Identity()
    raise NotImplementedError(f'{act} is not implemented')


class ResBlock(nn.Module):
    """ Implementation for ResNet block. """

    def __init__(self, n_feats: int, kernel_size: int, act: str='leaky_relu', atrous: int=1, bn: bool=False) ->None:
        """ param n_feats: Channel size.
            param kernel_size: kernel size.
            param act: string of activation to use.
            param atrous: controls amount of dilation to use in final conv.
            param bn: Turns on batch norm. 
        """
        super().__init__()
        m: List[nn.Module] = []
        _repr = []
        for i in range(2):
            atrous_rate = 1 if i == 0 else atrous
            conv_filter = util.conv(n_feats, n_feats, kernel_size, rate=atrous_rate, bias=True)
            m.append(conv_filter)
            _repr.append(f'Conv({n_feats}x{kernel_size}' + (f';A*{atrous_rate})' if atrous_rate != 1 else '') + ')')
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f'BN({n_feats})')
            if i == 0:
                m.append(get_act(act))
                _repr.append('Act')
        self.body = nn.Sequential(*m)
        self._repr = '/'.join(_repr)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        res = self.body(x)
        res += x
        return res

    def __repr__(self) ->str:
        return f'ResBlock({self._repr})'


class Upsampler(nn.Sequential):

    def __init__(self, scale: int, n_feats: int, bn: bool=False, act: str='none', bias: bool=True) ->None:
        m: List[nn.Module] = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(util.conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                m.append(get_act(act))
        elif scale == 3:
            m.append(util.conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            m.append(get_act(act))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class EDSRDec(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, resblocks: int=8, kernel_size: int=3, tail: str='none', channel_attention: bool=False) ->None:
        super().__init__()
        self.head = util.conv(in_ch, out_ch, 1)
        m_body: List[nn.Module] = [ResBlock(out_ch, kernel_size) for _ in range(resblocks)]
        m_body.append(util.conv(out_ch, out_ch, kernel_size))
        self.body = nn.Sequential(*m_body)
        self.tail: nn.Module
        if tail == 'conv':
            self.tail = util.conv(out_ch, out_ch, 1)
        elif tail == 'none':
            self.tail = nn.Identity()
        elif tail == 'upsample':
            self.tail = Upsampler(scale=2, n_feats=out_ch)
        else:
            raise NotImplementedError(f'{tail} is not implemented.')

    def forward(self, x: torch.Tensor, features_to_fuse: torch.Tensor=0.0) ->torch.Tensor:
        """
        :param x: N C H W
        :return: N C" H W
        """
        x = self.head(x)
        x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return x


class CDFOut(NamedTuple):
    logit_probs_c_sm: torch.Tensor
    means_c: torch.Tensor
    log_scales_c: torch.Tensor
    K: int
    targets: torch.Tensor


_LOG_SCALES_MIN = -7.0


_NUM_PARAMS_OTHER = 3


_NUM_PARAMS_RGB = 4


def non_shared_get_K(Kp: int, C: int, num_params: int) ->int:
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    return Kp // (num_params * C)


class DiscretizedMixLogisticLoss(nn.Module):

    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        self.use_coeffs = rgb_scale
        self._num_params = _NUM_PARAMS_RGB if rgb_scale else _NUM_PARAMS_OTHER
        self._nonshared_coeffs_act = torch.sigmoid
        self.bin_width = (x_max - x_min) / (L - 1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001
        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format((self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N * C * H * W)

    def to_sym(self, x):
        return quantizer.to_sym(x, self.x_min, self.x_max, self.L)

    def to_bn(self, S):
        return quantizer.to_bn(S, self.x_min, self.x_max, self.L)

    def cdf_step_non_shared(self, l, targets, c_cur, C, x_c=None) ->CDFOut:
        assert c_cur < C
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(c_cur, C, l, x_c)
        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)
        return CDFOut(logit_probs_c_softmax, means_c, log_scales_c, K, targets)

    def sample(self, l, C):
        return self._non_shared_sample(l, C)

    def log_cdf(self, lo, hi, means, log_scales):
        assert torch.all(lo <= hi), f'{lo[lo > hi]} > {hi[lo > hi]}'
        assert lo.min() >= self.x_min and hi.max() <= self.x_max, '{},{} not in {},{}'.format(lo.min(), hi.max(), self.x_min, self.x_max)
        centered_lo = lo - means
        centered_hi = hi - means
        inv_stdv = torch.exp(-log_scales)
        normalized_lo = inv_stdv * (centered_lo - self.bin_width / 2)
        lo_cond = (lo >= self.x_lower_bound).float()
        cdf_lo = lo_cond * torch.sigmoid(normalized_lo)
        normalized_hi = inv_stdv * (centered_hi + self.bin_width / 2)
        hi_cond = (hi <= self.x_upper_bound).float()
        cdf_hi = hi_cond * torch.sigmoid(normalized_hi) + (1 - hi_cond)
        cdf_delta = cdf_hi - cdf_lo
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))
        assert not torch.any(log_cdf_delta > 1e-06), f'{log_cdf_delta[log_cdf_delta > 1e-06]}'
        return log_cdf_delta

    def forward(self, x: torch.Tensor, l: torch.Tensor) ->torch.Tensor:
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, f'{x.min()},{x.max()} not in {self.x_min},{self.x_max}'
        x, logit_pis, means, log_scales, _ = self._extract_non_shared(x, l)
        log_probs = self.log_cdf(x, x, means, log_scales)
        log_weights = F.log_softmax(logit_pis, dim=2)
        log_probs_weighted = log_weights + log_probs
        nll = -torch.logsumexp(log_probs_weighted, dim=2)
        return nll

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \\pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]
        K = non_shared_get_K(Kp, C, self._num_params)
        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs = l[:, (0), (...)]
        means = l[:, (1), (...)]
        log_scales = torch.clamp(l[:, (2), (...)], min=_LOG_SCALES_MIN)
        x = x.reshape(N, C, 1, H, W)
        if self.use_coeffs:
            assert C == 3, C
            coeffs = self._nonshared_coeffs_act(l[:, (3), (...)])
            coeffs_g_r = coeffs[:, (0), (...)]
            coeffs_b_r = coeffs[:, (1), (...)]
            coeffs_b_g = coeffs[:, (2), (...)]
            means = torch.stack((means[:, (0), (...)], means[:, (1), (...)] + coeffs_g_r * x[:, (0), (...)], means[:, (2), (...)] + coeffs_b_r * x[:, (0), (...)] + coeffs_b_g * x[:, (1), (...)]), dim=1)
        means = torch.clamp(means, min=self.x_min, max=self.x_max)
        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K

    def _extract_non_shared_c(self, c: int, C: int, l: torch.Tensor, x: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        assert c < C, f'{c} >= {C}'
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)
        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs_c = l[:, (0), (c), (...)]
        means_c = l[:, (1), (c), (...)]
        log_scales_c = torch.clamp(l[:, (2), (c), (...)], min=_LOG_SCALES_MIN)
        if self.use_coeffs and c != 0:
            unscaled_coeffs = l[:, (3), (...)]
            if c == 1:
                assert x is not None
                coeffs_g_r = self._nonshared_coeffs_act(unscaled_coeffs[:, (0), (...)])
                means_c += coeffs_g_r * x[:, (0), (...)]
            elif c == 2:
                assert x is not None
                coeffs_b_r = self._nonshared_coeffs_act(unscaled_coeffs[:, (1), (...)])
                coeffs_b_g = self._nonshared_coeffs_act(unscaled_coeffs[:, (2), (...)])
                means_c += coeffs_b_r * x[:, (0), (...)] + coeffs_b_g * x[:, (1), (...)]
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l, C):
        """ sample from model """
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)
        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs = l[:, (0), (...)]
        u = torch.zeros_like(logit_probs).uniform_(1e-05, 1.0 - 1e-05)
        sel = torch.argmax(logit_probs - torch.log(-torch.log(u)), dim=2)
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))
        sel = sel.unsqueeze(2)
        means = torch.gather(l[:, (1), (...)], 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(l[:, (2), (...)], 2, sel).squeeze(2), min=_LOG_SCALES_MIN)
        u = torch.zeros_like(means).uniform_(1e-05, 1.0 - 1e-05)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
        if self.use_coeffs:
            assert C == 3

            def clamp(x_):
                return torch.clamp(x_, 0, 255.0)
            coeffs = torch.sigmoid(l[:, (3), (...)])
            sel_g, sel_b = sel[:, (1), (...)], sel[:, (2), (...)]
            coeffs_g_r = torch.gather(coeffs[:, (0), (...)], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, (1), (...)], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, (2), (...)], 1, sel_b).squeeze(1)
            x0 = clamp(x[:, (0), (...)])
            x1 = clamp(x[:, (1), (...)] + coeffs_g_r * x0)
            x2 = clamp(x[:, (2), (...)] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)
        return x


class StackedAtrousConvs(nn.Module):

    def __init__(self, atrous_rates_str: Union[str, int], Cin: int, Cout: int, bias: bool=True, kernel_size: int=3) ->None:
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList([util.conv(Cin, Cin, kernel_size, rate=rate) for rate in atrous_rates])
        self.lin = util.conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str: Union[str, int]) ->List[int]:
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self) ->str:
        return self._extra_repr

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = torch.cat([atrous(x) for atrous in self.atrous], dim=1)
        x = self.lin(x)
        return x


def non_shared_get_Kp(K, C, num_params):
    """ Get Kp=number of channels to predict. 
        See note where we define _NUM_PARAMS_RGB above """
    return num_params * C * K


class AtrousProbabilityClassifier(nn.Module):

    def __init__(self, in_ch: int, C: int, num_params: int, K: int=10, kernel_size: int=3, atrous_rates_str: str='1,2,4') ->None:
        super(AtrousProbabilityClassifier, self).__init__()
        Kp = non_shared_get_Kp(K, C, num_params)
        self.atrous = StackedAtrousConvs(atrous_rates_str, in_ch, Kp, kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def __repr__(self) ->str:
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        :param x: N C H W
        :return: N Kp H W
        """
        return self.atrous(x)


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


Probs = Tuple[torch.Tensor, Optional[LogisticMixtureProbability], int]


def group_2x2(x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Group 2x2 patches of x on its own channel
        param x: N C H W
        returns: Tuple[N 4 C H/2 W/2]
    """
    _, _, h, w = x.size()
    x_even_height = x[:, :, 0:h:2, :]
    x_odd_height = x[:, :, 1:h:2, :]
    return x_even_height[:, :, :, 0:w:2], x_even_height[:, :, :, 1:w:2], x_odd_height[:, :, :, 0:w:2], x_odd_height[:, :, :, 1:w:2]

