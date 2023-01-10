import sys
_module = sys.modules[__name__]
del sys
config = _module
models = _module
layers_quant = _module
ptq = _module
bit_type = _module
layers = _module
observer = _module
base = _module
build = _module
ema = _module
minmax = _module
omse = _module
percentile = _module
ptf = _module
utils = _module
quantizer = _module
base = _module
log2 = _module
uniform = _module
swin_quant = _module
utils = _module
vit_quant = _module
test_quant = _module

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


import collections.abc


import math


import warnings


from itertools import repeat


import torch


import torch.nn.functional as F


from torch import nn


import numpy as np


import torch.nn as nn


from torch.nn import functional as F


from typing import Optional


import torch.utils.checkpoint as checkpoint


import re


from collections import OrderedDict


from functools import partial


import time


import torchvision.datasets as datasets


import torchvision.transforms as transforms


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class BitType:

    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        if not self.signed:
            return 2 ** self.bits - 1
        return 2 ** (self.bits - 1) - 1

    @property
    def lower_bound(self):
        if not self.signed:
            return 0
        return -2 ** (self.bits - 1)

    @property
    def range(self):
        return 2 ** self.bits

    def update_name(self):
        self.name = ''
        if not self.signed:
            self.name += 'uint'
        else:
            self.name += 'int'
        self.name += '{}'.format(self.bits)


BIT_TYPE_LIST = [BitType(4, False, 'uint4'), BitType(8, True, 'int8'), BitType(8, False, 'uint8')]


BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}


class BaseObserver:

    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = torch.finfo(torch.float32).eps

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.module_type in ['conv_weight', 'linear_weight']:
            v = v.reshape(v.shape[0], -1)
        elif self.module_type == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
            v = v.reshape(-1, v.shape[-1])
            v = v.transpose(0, 1)
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError


class EmaObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode, ema_sigma=0.01):
        super(EmaObserver, self).__init__(module_type, bit_type, calibration_mode)
        self.ema_sigma = ema_sigma
        self.symmetric = self.bit_type.signed

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + self.ema_sigma * (cur_max - self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + self.ema_sigma * (cur_min - self.min_val)
        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type, calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class OmseObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(OmseObserver, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        best_score = 10000000000.0
        for i in range(90):
            new_max = max_val * (1.0 - i * 0.01)
            new_min = min_val * (1.0 - i * 0.01)
            new_scale = (new_max - new_min) / float(qmax - qmin)
            new_scale.clamp_(self.eps)
            new_zero_point = qmin - torch.round(new_min / new_scale)
            new_zero_point.clamp_(qmin, qmax)
            inputs_q = ((inputs / new_scale + new_zero_point).round().clamp(qmin, qmax) - new_zero_point) * new_scale
            score = lp_loss(inputs, inputs_q, p=2.0, reduction='all')
            if score < best_score:
                best_score = score
                self.max_val = new_max
                self.min_val = new_min
                scale = new_scale
                zero_point = new_zero_point
        return scale, zero_point


class PercentileObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode, percentile_sigma=0.01, percentile_alpha=0.99999):
        super(PercentileObserver, self).__init__(module_type, bit_type, calibration_mode)
        self.percentile_sigma = 0.01
        self.percentile_alpha = 0.99999
        self.symmetric = self.bit_type.signed

    def update(self, v):
        assert self.calibration_mode == 'layer_wise'
        v = self.reshape_tensor(v)
        try:
            cur_max = torch.quantile(v.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(v.reshape(-1), 1.0 - self.percentile_alpha)
        except:
            cur_max = torch.tensor(np.percentile(v.reshape(-1).cpu(), self.percentile_alpha * 100), device=v.device, dtype=torch.float32)
            cur_min = torch.tensor(np.percentile(v.reshape(-1).cpu(), (1 - self.percentile_alpha) * 100), device=v.device, dtype=torch.float32)
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + self.percentile_sigma * (cur_max - self.max_val)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point


class PtfObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        best_score = 10000000000.0
        max_val_t = max_val.max()
        min_val_t = min_val.min()
        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8.clamp_(self.eps)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale8)
        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        for j in range(inputs.shape[2]):
            data = inputs[..., j].unsqueeze(-1)
            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) - zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) - zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) - zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) - zero_point) * scale8
            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score = [score1, score2, score4, score8]
            scale_mask[j] *= 2 ** score.index(min(score))
        scale = scale1 * scale_mask
        return scale, zero_point


str2observer = {'minmax': MinmaxObserver, 'ema': EmaObserver, 'omse': OmseObserver, 'percentile': PercentileObserver, 'ptf': PtfObserver}


def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)


class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = -1, 1, 1, 1
        elif self.module_type == 'linear_weight':
            range_shape = -1, 1
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = 1, -1
            elif len(inputs.shape) == 3:
                range_shape = 1, 1, -1
            elif len(inputs.shape) == 4:
                range_shape = 1, -1, 1, 1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)
        return outputs


class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(Log2Quantizer, self).__init__(bit_type, observer, module_type)
        self.softmax_mask = None

    def quant(self, inputs):
        rounds = torch.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2 ** self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2 ** self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs):
        outputs = 2 ** (-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs


class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):
        self.scale, self.zero_point = self.observer.get_quantization_params(*args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(self.bit_type.lower_bound, self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs


str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)


class QAct(nn.Module):

    def __init__(self, quant=False, calibrate=False, last_calibrate=False, bit_type=BIT_TYPE_DICT['int8'], calibration_mode='layer_wise', observer_str='minmax', quantizer_str='uniform'):
        super(QAct, self).__init__()
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type, self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, quant=False, calibrate=False, last_calibrate=False, bit_type=BIT_TYPE_DICT['int8'], calibration_mode='layer_wise', observer_str='minmax', quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type, self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, quant=False, calibrate=False, cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QLinear(in_features, hidden_features, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.act = act_layer()
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.fc2 = QLinear(hidden_features, out_features, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.qact1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.qact2(x)
        x = self.drop(x)
        return x


class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, quant=False, calibrate=False, last_calibrate=False, bit_type=BIT_TYPE_DICT['int8'], calibration_mode='layer_wise', observer_str='minmax', quantizer_str='uniform'):
        super(QConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type, self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, quant=False, calibrate=False, cfg=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = QConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        if norm_layer:
            self.qact_before_norm = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
            self.norm = norm_layer(embed_dim)
            self.qact = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        else:
            self.qact_before_norm = nn.Identity()
            self.norm = nn.Identity()
            self.qact = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.qact_before_norm(x)
        if isinstance(self.norm, nn.Identity):
            x = self.norm(x)
        else:
            x = self.norm(x, self.qact_before_norm.quantizer, self.qact.quantizer)
        x = self.qact(x)
        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2 ** (bit + 1) - 1)
        return M, N

    def forward(self, x, in_quantizer=None, out_quantizer=None, in_scale_expand=1):
        if self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(-1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()
            x_q = x_q * in_scale_mask
            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = in_scale1 / channel_nums * torch.sqrt(channel_nums * (x_q ** 2).sum(dim=-1) - x_q.sum(dim=-1) ** 2)
            A = (in_scale1 / std_x_q).unsqueeze(-1) * self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) - (mean_x_q / std_x_q).unsqueeze(-1) * self.weight.reshape(1, 1, -1)) / out_scale * torch.pow(2, N)).round()
            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):

    def __init__(self, log_i_softmax=False, quant=False, calibrate=False, last_calibrate=False, bit_type=BIT_TYPE_DICT['int8'], calibration_mode='layer_wise', observer_str='minmax', quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()
        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type, self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = x - 2 ** big >= 2 ** (big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.0]
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor ** 2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor ** 2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931
            n = 30
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2 ** (n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2 ** n
            return exp_int, scaling_factor
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2 ** self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2 ** self.bit_type.bits - 1)
            deq_softmax = 2 ** -qlog
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = QLinear(dim, dim * 3, bias=qkv_bias, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact_attn1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact_table = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop)
        self.log_int_softmax = QIntSoftmax(log_i_softmax=cfg.INT_SOFTMAX, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_S, calibration_mode=cfg.CALIBRATION_MODE_S, observer_str=cfg.OBSERVER_S, quantizer_str=cfg.QUANTIZER_S)
        self.qact3 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact4 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.proj = QLinear(dim, dim, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.qact_attn1(attn)
        relative_position_bias_table_q = self.qact_table(self.relative_position_bias_table)
        relative_position_bias = relative_position_bias_table_q[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.qact2(attn)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.log_int_softmax(attn, self.qact2.quantizer.scale)
        else:
            attn = self.log_int_softmax(attn, self.qact2.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.qact3(x)
        x = self.proj(x)
        x = self.qact4(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, quant=quant, calibrate=calibrate, cfg=cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)
        self.norm2 = norm_layer(dim)
        self.qact3 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, quant=quant, calibrate=calibrate, cfg=cfg)
        self.qact4 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x, last_quantizer=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x, last_quantizer, self.qact1.quantizer)
        x = self.qact1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = self.qact2(x)
        x = x + self.drop_path(self.mlp(self.qact3(self.norm2(x, self.qact2.quantizer, self.qact3.quantizer))))
        x = self.qact4(x)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.reduction = QLinear(4 * dim, 2 * dim, bias=False, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)

    def forward(self, x, last_quantizer=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x, last_quantizer, self.qact1.quantizer, 4)
        x = self.qact1(x)
        x = self.reduction(x)
        x = self.qact2(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, quant=quant, calibrate=calibrate, cfg=cfg) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, quant=quant, calibrate=calibrate, cfg=cfg)
        else:
            self.downsample = None

    def forward(self, x, last_quantizer=None):
        for i, blk in enumerate(self.blocks):
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            elif i == 0:
                x = blk(x, last_quantizer)
            else:
                x = blk(x, self.blocks[i - 1].qact4.quantizer)
        if self.downsample is not None:
            x = self.downsample(x, self.blocks[-1].qact4.quantizer)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, quant=False, calibrate=False, input_quant=False, cfg=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.input_quant = input_quant
        self.cfg = cfg
        if input_quant:
            self.qact_input = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None, quant=quant, calibrate=calibrate, cfg=cfg)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
            self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_LN, observer_str=cfg.OBSERVER_LN, quantizer_str=cfg.QUANTIZER_LN)
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i_layer in range(self.num_layers):
            layers += [BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(self.patch_grid[0] // 2 ** i_layer, self.patch_grid[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, quant=quant, calibrate=calibrate, cfg=cfg)]
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.qact3 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.head = QLinear(self.num_features, num_classes, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W) if num_classes > 0 else nn.Identity()
        self.act_out = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = QLinear(self.num_features, num_classes, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W) if num_classes > 0 else nn.Identity()

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x):
        if self.input_quant:
            x = self.qact_input(x)
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
            x = self.qact1(x)
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            last_quantizer = self.patch_embed.qact.quantizer if i == 0 else self.layers[i - 1].downsample.qact2.quantizer
            x = layer(x, last_quantizer)
        x = self.norm(x, self.layers[-1].blocks[-1].qact4.quantizer, self.qact2.quantizer)
        x = self.qact2(x)
        x = self.avgpool(x.transpose(1, 2))
        x = self.qact3(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.act_out(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = QLinear(dim, dim * 3, bias=qkv_bias, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.proj = QLinear(dim, dim, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W)
        self.qact3 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact_attn1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = QIntSoftmax(log_i_softmax=cfg.INT_SOFTMAX, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_S, calibration_mode=cfg.CALIBRATION_MODE_S, observer_str=cfg.OBSERVER_S, quantizer_str=cfg.QUANTIZER_S)

    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.qact_attn1(attn)
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, cfg=cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)
        self.norm2 = norm_layer(dim)
        self.qact3 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, quant=quant, calibrate=calibrate, cfg=cfg)
        self.qact4 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)

    def forward(self, x, last_quantizer=None):
        x = self.qact2(x + self.drop_path(self.attn(self.qact1(self.norm1(x, last_quantizer, self.qact1.quantizer)))))
        x = self.qact4(x + self.drop_path(self.mlp(self.qact3(self.norm2(x, self.qact2.quantizer, self.qact3.quantizer)))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, quant=False, calibrate=False, input_quant=False, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.cfg = cfg
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, quant=quant, calibrate=calibrate, cfg=cfg)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.qact_embed = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact_pos = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        self.qact1 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, quant=quant, calibrate=calibrate, cfg=cfg) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.qact2 = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        self.head = QLinear(self.num_features, num_classes, quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_W, calibration_mode=cfg.CALIBRATION_MODE_W, observer_str=cfg.OBSERVER_W, quantizer_str=cfg.QUANTIZER_W) if num_classes > 0 else nn.Identity()
        self.act_out = QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x):
        B = x.shape[0]
        if self.input_quant:
            x = self.qact_input(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.qact_embed(x)
        x = x + self.qact_pos(self.pos_embed)
        x = self.qact1(x)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[i - 1].qact4.quantizer
            x = blk(x, last_quantizer)
        x = self.norm(x, self.blocks[-1].qact4.quantizer, self.qact2.quantizer)[:, 0]
        x = self.qact2(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.act_out(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QIntLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QIntSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (QLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_megvii_research_FQ_ViT(_paritybench_base):
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

