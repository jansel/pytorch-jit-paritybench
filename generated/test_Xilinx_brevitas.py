import sys
_module = sys.modules[__name__]
del sys
gen_github_actions = _module
gen_vitis_ai_actions = _module
utils = _module
conf = _module
noxfile = _module
setup = _module
brevitas = _module
config = _module
core = _module
bit_width = _module
const = _module
parameter = _module
function_wrapper = _module
clamp = _module
misc = _module
ops_ste = _module
shape = _module
quant = _module
binary = _module
delay = _module
int_base = _module
ternary = _module
restrict_val = _module
scaling = _module
int_scaling = _module
runtime = _module
standalone = _module
stats = _module
stats_op = _module
stats_wrapper = _module
view_wrapper = _module
utils = _module
zero_point = _module
export = _module
handler = _module
manager = _module
onnx = _module
debug = _module
finn = _module
function = _module
acc = _module
act = _module
parameter = _module
acc = _module
act = _module
base = _module
parameter = _module
manager = _module
transform = _module
generic = _module
function = _module
handler = _module
manager = _module
handler = _module
manager = _module
standard = _module
function = _module
qdq = _module
qoperator = _module
function = _module
act = _module
base = _module
parameter = _module
pool = _module
manager = _module
vitis_ai = _module
handler = _module
manager = _module
pyxir = _module
function = _module
handler = _module
manager = _module
xir = _module
function = _module
handler = _module
manager = _module
pytorch = _module
handler = _module
act = _module
base = _module
parameter = _module
manager = _module
autograd_ste_ops = _module
ops = _module
ops_ste = _module
shape = _module
fx = _module
backport = _module
graph = _module
graph_module = _module
immutable_collections = _module
interpreter = _module
node = _module
proxy = _module
subgraph_rewriter = _module
symbolic_trace = _module
torch_function = _module
_overrides = _module
patch = _module
signatures = _module
brevitas_tracer = _module
value_tracer = _module
base = _module
calibrate = _module
equalize = _module
fixed_point = _module
per_input = _module
standardize = _module
target = _module
flexml = _module
utils = _module
inject = _module
defaults = _module
enum = _module
jit = _module
loss = _module
base_loss = _module
weighted_bit_width = _module
nn = _module
hadamard_classifier = _module
mixin = _module
act = _module
base = _module
quant_accumulator = _module
quant_activation = _module
quant_avg_pool = _module
quant_bn = _module
quant_conv = _module
quant_convtranspose = _module
quant_dropout = _module
quant_eltwise = _module
quant_layer = _module
quant_linear = _module
quant_max_pool = _module
quant_scale_bias = _module
quant_upsample = _module
utils = _module
parameter_quant = _module
quant_proxy = _module
runtime_quant = _module
utils = _module
none = _module
scaled_int = _module
shifted_scaled_int = _module
solver = _module
act = _module
bias = _module
common = _module
parameter = _module
trunc = _module
weight = _module
quant_tensor = _module
torch_handler = _module
jit_utils = _module
logging = _module
python_utils = _module
quant_utils = _module
torch_utils = _module
brevitas_examples = _module
bnn_pynq = _module
bnn_pynq_train = _module
cfg = _module
logger = _module
CNV = _module
FC = _module
models = _module
losses = _module
tensor_norm = _module
trainer = _module
imagenet_classification = _module
imagenet_val = _module
models = _module
mobilenetv1 = _module
proxylessnas = _module
vgg = _module
speech_to_text = _module
topology = _module
get_librispeech_data = _module
quartznet = _module
audio_preprocessing = _module
data_layer = _module
greedy_ctc_decoder = _module
helpers = _module
losses = _module
metrics = _module
parts = _module
cleaners = _module
dataset = _module
features = _module
manifest = _module
perturb = _module
quartznet = _module
segment = _module
spectr_augment = _module
quartznet = _module
quartznet_val = _module
text_to_speech = _module
melgan = _module
common = _module
generator_brevitas = _module
res_stack_brevitas = _module
melgan_val = _module
preprocess_dataset = _module
utilities = _module
audio_processing = _module
stft = _module
tests = _module
common = _module
binary_quant_fixture = _module
bit_width_fixture = _module
int_quant_fixture = _module
shared_quant_fixture = _module
ternary_quant_fixture = _module
test_binary_quant = _module
test_bit_width = _module
test_int_quant = _module
test_stats = _module
test_ternary_quant = _module
test_generic_export = _module
test_pytorch_qf_export = _module
hyp_helper = _module
test_autograd_ste_ops = _module
test_ops = _module
test_ops_ste = _module
test_shape = _module
test_tracer = _module
test_transforms = _module
hyp_helper = _module
test_weighted_bit_width = _module
test_act = _module
test_conv2d = _module
test_linear = _module
test_merge_bn = _module
test_wbiol = _module
test_act_scaling = _module
test_weight_scaling = _module
test_brevitas_import = _module
test_python_utils = _module
test_examples_import = _module
test_jit_trace = _module
test_pretrained_accuracy = _module
brevitas_finn = _module
test_brevitas_avg_pool_export = _module
test_debug_export = _module
test_wbiol = _module
test_bnn_pynq_finn_export = _module
test_mobilenet_finn_export = _module
test_quartznet_finn_export = _module
brevitas_ort = _module
test_onnx_standard = _module
brevitas_pyxir = _module
test_dpu_export = _module
test_xir_export = _module
conftest = _module
marker = _module

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


import warnings


from typing import List


from typing import Optional


from torch.utils import cpp_extension


import torch


from torch import Tensor


from torch.nn import Module


from torch.nn import Parameter


from typing import Tuple


from typing import Callable


from typing import Union


import math


from torch import nn


from abc import ABC


from abc import abstractmethod


from functools import partial


from torch.autograd import Function


from typing import TYPE_CHECKING


from torch.nn import Sequential


import copy


from copy import copy


import torch.onnx


from torch.nn.functional import max_pool1d


from torch.nn.functional import max_pool2d


from torch.nn import functional as F


import inspect


import torch.nn.functional as F


import numpy as np


from typing import Any


from typing import Dict


from typing import Set


import types


import re


import torch.nn as nn


from typing import Type


from typing import Iterator


from typing import Iterable


from typing import NamedTuple


import functools


from types import CodeType


from types import FunctionType


from types import ModuleType


from itertools import chain


from inspect import getfullargspec


from inspect import getcallargs


import collections


from random import sample


from functools import reduce


from copy import deepcopy


from inspect import signature


from abc import ABCMeta


from warnings import warn


import torch.jit


from inspect import isclass


from torch.nn import AvgPool2d


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Conv1d


from torch.nn import Conv2d


from torch.nn.functional import conv2d


from torch.nn import ConvTranspose1d


from torch.nn import ConvTranspose2d


from torch.nn.functional import conv_transpose1d


from torch.nn.functional import conv_transpose2d


from torch.nn import Dropout


from torch.nn import Linear


from torch.nn.functional import linear


from torch.nn import MaxPool1d


from torch.nn import MaxPool2d


from torch.nn import Upsample


from torch.nn import UpsamplingBilinear2d


from torch.nn import UpsamplingNearest2d


from torch.nn.functional import interpolate


from torch import tensor


from torch.nn import Identity


from enum import Enum


from torch.nn import ModuleList


from torch.nn import BatchNorm2d


from torch.nn import BatchNorm1d


from torch import hub


import torch.nn.init as init


import random


import time


import torch.optim as optim


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import MNIST


from torchvision.datasets import CIFAR10


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torch.utils.data import Dataset


from scipy.io.wavfile import write


from scipy.signal import get_window


from scipy.io.wavfile import read


from torch.autograd import Variable


from torchvision.models import resnet18


from torchvision.models import mobilenet_v2


from torchvision.models import alexnet


from torchvision.models import squeezenet1_0


from torchvision.models import shufflenet_v2_x0_5


from torchvision.models import mnasnet0_5


from torchvision.models import densenet121


from torchvision import models


class InplaceLogTwo(torch.nn.Module):
    """
    Module wrapper for :func:`~torch.log2_`.

    Examples:
        >>> inplace_log_two = InplaceLogTwo()
        >>> x = torch.tensor(8.0)
        >>> inplace_log_two(x)
        >>> x
        tensor(3.)

    Notes:
        Inplace operations in TorchScript can be problematic, compilation is disabled.
    """

    def __init__(self) ->None:
        super(InplaceLogTwo, self).__init__()

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x.log2_()
        return x


class KLMinimizerThreshold(torch.nn.Module):
    """
    Based on:
    https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """

    def __init__(self, signed, bit_width_impl, num_bins=1000 + 1, smoothing_eps=0.0001):
        super(KLMinimizerThreshold, self).__init__()
        self.num_bins = num_bins
        self.smoothing_eps = smoothing_eps
        self.signed = signed
        self.bit_width_impl = bit_width_impl
        self.absmax_impl = AbsMax()

    def smooth_normalize_distribution(self, p, eps):
        is_zeros = (p == 0).float()
        n_zeros = is_zeros.sum()
        n_nonzeros = torch.numel(p) - n_zeros
        if not n_nonzeros:
            return None
        eps1 = eps * n_zeros / n_nonzeros
        hist = p.float()
        hist += eps * is_zeros + -eps1 * n_nonzeros
        dist = torch.distributions.categorical.Categorical(logits=hist)
        return dist

    def forward(self, x: Tensor):
        absmax = self.absmax_impl(x)
        bit_width = self.bit_width_impl()
        num_quantized_bins = max_int(self.signed, False, bit_width).int()
        thresholds = torch.zeros(self.num_bins // 2 + 1 - num_quantized_bins // 2, device=x.device)
        divergence = torch.zeros_like(thresholds)
        quantized_bins = torch.zeros(num_quantized_bins, device=x.device)
        hist = torch.histc(x, bins=self.num_bins, min=-absmax, max=absmax).int()
        hist_edges = torch.linspace(-absmax, absmax, self.num_bins + 1)
        for i in range(num_quantized_bins // 2, self.num_bins // 2 + 1):
            p_bin_idx_start = self.num_bins // 2 - i
            p_bin_idx_stop = self.num_bins // 2 + i + 1
            thresholds[i - num_quantized_bins // 2] = hist_edges[p_bin_idx_stop]
            sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]
            p = sliced_nd_hist.clone()
            left_outlier_count = torch.sum(hist[0:p_bin_idx_start])
            p[0] += left_outlier_count
            right_outlier_count = torch.sum(hist[p_bin_idx_stop:])
            p[-1] += right_outlier_count
            is_nonzeros = (sliced_nd_hist != 0).float()
            num_merged_bins = torch.numel(p) // num_quantized_bins
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
            q = torch.zeros_like(p, dtype=torch.float32, device=x.device)
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                if j == num_quantized_bins - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = quantized_bins[j] / norm
            q[sliced_nd_hist == 0] = 0.0
            p = self.smooth_normalize_distribution(p, self.smoothing_eps)
            q = self.smooth_normalize_distribution(q, self.smoothing_eps)
            if q is None:
                divergence[i - num_quantized_bins // 2] = float('inf')
            else:
                divergence[i - num_quantized_bins // 2] = torch.distributions.kl.kl_divergence(p, q)
        min_divergence_idx = torch.argmin(divergence)
        opt_threshold = thresholds[min_divergence_idx]
        return opt_threshold


class BaseHandler(Module, ABC):

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        pass

    def reset(self):
        pass


class _JitTraceExportWrapper(nn.Module):

    def __init__(self, model_to_trace):
        super(_JitTraceExportWrapper, self).__init__()
        self.fn_to_trace = lambda *args, **kwargs: model_to_trace(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.fn_to_trace(*args, **kwargs)


class _InputQuantTensorFunction(Function):
    """Account symbolically for scale and zero-point of an input quant tensor"""

    @staticmethod
    def symbolic(g, x, scale, zero_point):
        if zero_point is not None:
            x = g.op('Sub', x, zero_point)
        if scale is not None:
            x = g.op('Mul', x, scale)
        return x

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        return x


class _InputPreprocessingModule(Module):

    def __init__(self, scale, zero_point):
        super(_InputPreprocessingModule, self).__init__()
        if scale is not None:
            self.register_buffer('scale', scale)
        else:
            self.scale = None
        if zero_point is not None:
            self.register_buffer('zero_point', zero_point)
        else:
            self.zero_point = None

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            x = _InputQuantTensorFunction.apply(x, self.scale, self.zero_point)
        return x


class DebugMarkerFunction(Function):

    @staticmethod
    def symbolic(g, input, export_debug_name):
        ret = g.op('brevitas.onnx::DebugMarker', input, export_debug_name_s=export_debug_name)
        return ret

    @staticmethod
    def forward(ctx, input, export_debug_name):
        return input


class ONNXBaseHandler(BaseHandler, ABC):

    def __init__(self):
        super().__init__()
        self.symbolic_kwargs = None
        self.export_debug_name = None
        self.debug_input = False
        self.debug_output = False

    @abstractmethod
    def prepare_for_export(self, module):
        pass

    @abstractmethod
    def symbolic_execution(self, *args, **kwargs):
        pass

    def reset(self):
        self.symbolic_kwargs = None

    def attach_debug_info(self, m):
        self.export_debug_name = m.export_debug_name
        self.debug_input = m.export_input_debug
        self.debug_output = m.export_output_debug

    def forward(self, inp: Tensor, *args, **kwargs):
        debug_fn = lambda x, name: DebugMarkerFunction.apply(x, self.export_debug_name + name)
        if self.export_debug_name is not None and self.debug_input:
            inp = debug_fn(inp, '.input')
        out = self.symbolic_execution(inp, *args, **kwargs)
        if self.export_debug_name is not None and self.debug_output:
            if isinstance(out, Tensor):
                out = debug_fn(out, '.output')
            elif isinstance(out, tuple) and isinstance(out[0], Tensor):
                out = list(out)
                out[0] = debug_fn(out[0], '.output')
                out = tuple(out)
        return out


class BitWidthHandlerMixin(object):

    @classmethod
    def validate_bit_width(cls, bit_width: Tensor, reference: int, le_then=False):
        if bit_width is None:
            raise RuntimeError('Bit width cannot be None')
        bit_width = int(bit_width.item())
        if bit_width > reference:
            raise RuntimeError(f'Bit width {bit_width} is not supported.')
        elif bit_width < reference and not le_then:
            raise RuntimeError(f'Bit width {bit_width} is not supported, should be {reference}b.')
        return bit_width

    @classmethod
    def validate_8b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 8, le_then)

    @classmethod
    def validate_16b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 16, le_then)

    @classmethod
    def validate_32b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 32, le_then)


QUANT_TENSOR_FN_HANDLER = {}


class QuantTensorBase(NamedTuple):
    value: Tensor
    scale: Optional[Tensor]
    zero_point: Optional[Tensor]
    bit_width: Optional[Tensor]
    signed_t: Optional[Tensor]
    training_t: Optional[Tensor]


def _is_all_nested_not_none(input_data):
    if isinstance(input_data, QuantTensor):
        return input_data.is_not_none
    elif isinstance(input_data, (tuple, list)):
        return all([_is_all_nested_not_none(v) for v in input_data])
    elif isinstance(input_data, dict):
        return all([_is_all_nested_not_none(v) for v in input_data.values()])
    else:
        return True


def _unpack_quant_tensor(input_data):
    if isinstance(input_data, QuantTensor):
        return input_data.tensor
    elif isinstance(input_data, tuple):
        return tuple([_unpack_quant_tensor(v) for v in input_data])
    elif isinstance(input_data, list):
        return [_unpack_quant_tensor(v) for v in input_data]
    elif isinstance(input_data, dict):
        return {k: _unpack_quant_tensor(v) for k, v in input_data.items()}
    else:
        return input_data


class QuantTensor(QuantTensorBase):

    def __new__(cls, value, scale=None, zero_point=None, bit_width=None, signed=None, training=None):
        if scale is not None and not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if zero_point is not None and not isinstance(zero_point, torch.Tensor):
            zero_point = torch.tensor(zero_point, dtype=torch.float)
        if bit_width is not None and not isinstance(bit_width, torch.Tensor):
            bit_width = torch.tensor(bit_width, dtype=torch.float)
        if signed is not None:
            signed = torch.tensor(signed, dtype=torch.bool)
        if training is not None:
            training = torch.tensor(training, dtype=torch.bool)
        return super().__new__(cls, value, scale, zero_point, bit_width, signed, training)

    @property
    def signed(self):
        if self.signed_t is not None:
            return self.signed_t.item()
        else:
            return None

    @property
    def training(self):
        if self.training_t is not None:
            return self.training_t.item()
        else:
            return None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in QUANT_TENSOR_FN_HANDLER or not all(issubclass(t, QuantTensor) for t in types) or not (_is_all_nested_not_none(args) and _is_all_nested_not_none(kwargs)):
            args = _unpack_quant_tensor(args)
            kwargs = _unpack_quant_tensor(kwargs)
            return func(*args, **kwargs)
        return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)

    @property
    def tensor(self):
        return self.value

    @property
    def is_not_none(self):
        return self.value is not None and self.scale is not None and self.zero_point is not None and self.bit_width is not None and self.signed is not None

    @property
    def _pre_round_int_value(self):
        int_value = self.value / self.scale
        int_value = int_value + self.zero_point
        return int_value

    @property
    def is_valid(self):
        if self.is_not_none:
            with torch.no_grad():
                pre_round_int_value = self._pre_round_int_value
                rounded_int_value = torch.round(pre_round_int_value)
                is_int = torch.isclose(pre_round_int_value, rounded_int_value).all()
                if self.bit_width >= 2:
                    if self.signed:
                        is_upper_b = (2.0 ** (self.bit_width - 1) - 1 >= rounded_int_value).all()
                        is_lower_b = (-2.0 ** (self.bit_width - 1) <= rounded_int_value).all()
                    else:
                        is_upper_b = (2.0 ** self.bit_width - 1 >= rounded_int_value).all()
                        is_lower_b = (0.0 <= rounded_int_value).all()
                    return (is_int & is_upper_b & is_lower_b).item()
                else:
                    unique_vals = rounded_int_value.unique(sorted=False, return_counts=False, return_inverse=False)
                    is_binary = unique_vals.view(-1).size()[0] == 2
                    is_signed = (unique_vals < 0.0).any().item()
                    sign_match = is_signed == self.signed
                    return is_int.item() and is_binary and sign_match
        else:
            return False

    @property
    def device(self):
        value_device = self.value.device
        is_same_device = True
        for t in [self.scale, self.zero_point, self.bit_width]:
            if t is not None:
                is_same_device &= value_device == t.device
        if not is_same_device:
            raise RuntimeError('Value and metadata are on different devices')
        return value_device

    def set(self, **kwargs):
        return self._replace(**kwargs)

    def detach_(self):
        self.value.detach_()
        self.scale.detach_()
        self.zero_point.detach_()
        self.bit_width.detach_()

    def detach(self):
        return QuantTensor(self.value.detach(), self.scale.detach() if self.scale is not None else None, self.zero_point.detach() if self.zero_point is not None else None, self.bit_width.detach() if self.bit_width is not None else None, self.signed, self.training)

    def contiguous(self):
        return QuantTensor(self.value.contiguous(), self.scale.contiguous() if self.scale is not None else None, self.zero_point.contiguous() if self.zero_point is not None else None, self.bit_width.contiguous() if self.bit_width is not None else None, self.signed, self.training)

    def int(self, float_datatype=False):
        if self.is_valid:
            int_value = round_ste(self._pre_round_int_value)
            if float_datatype:
                return int_value
            else:
                return int_value.int()
        else:
            raise RuntimeError(f'QuantTensor not valid.')

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, QuantTensor):
            raise RuntimeError('Tensor is not a QuantTensor')

    @staticmethod
    def is_zero_zero_point(tensor):
        QuantTensor.check_input_type(tensor)
        if tensor.zero_point is not None:
            return (tensor.zero_point == 0.0).all()
        else:
            return None

    def check_scaling_factors_same(self, other):
        if self.training is not None and self.training:
            return True
        if not torch.allclose(self.scale, other.scale):
            raise RuntimeError('Scaling factors are different')

    def check_zero_points_same(self, other):
        if self.training is not None and self.training:
            return True
        if not torch.allclose(self.zero_point, other.zero_point):
            raise RuntimeError('Zero points are different')

    def check_bit_width_same(self, other):
        if not torch.allclose(self.bit_width, other.bit_width):
            raise RuntimeError('Bit widths are different')

    def check_sign_same(self, other):
        if not self.signed == other.signed:
            raise RuntimeError('Signs are different')

    def view(self, *args, **kwargs):
        return self.set(value=self.value.view(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.set(value=self.value.reshape(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return self.set(value=self.value.flatten(*args, **kwargs))

    def transpose(self, *args, **kwargs):
        value = self.value.transpose(*args, **kwargs)
        tensor_meta = {'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if tm is not None and len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.transpose(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    def permute(self, *args, **kwargs):
        value = self.value.permute(*args, **kwargs)
        tensor_meta = {'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if tm is not None and len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.permute(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)

    @property
    def shape(self):
        return self.value.shape

    def dim(self):
        return self.value.dim()

    def add(self, other):
        return self + other

    @staticmethod
    def cat(tensors, dim, out=None):
        if out is not None:
            raise RuntimeError('Out not supported.')
        if len(tensors) < 2:
            return tensors[0]
        else:
            first_qt = tensors[0]
            if all([(isinstance(qt, QuantTensor) and qt.is_not_none) for qt in tensors]):
                for qt in tensors[1:]:
                    first_qt.check_scaling_factors_same(qt)
                    first_qt.check_zero_points_same(qt)
                    first_qt.check_bit_width_same(qt)
                    first_qt.check_sign_same(qt)
                output_value = torch.cat([qt.value for qt in tensors], dim=dim)
                output_scale = sum([qt.scale for qt in tensors]) / len(tensors)
                output_zero_point = sum([qt.zero_point for qt in tensors]) / len(tensors)
                output_bit_width = sum([qt.bit_width for qt in tensors]) / len(tensors)
                output_signed = first_qt.signed
                output_training = any([qt.training for qt in tensors])
                return QuantTensor(value=output_value, scale=output_scale, zero_point=output_zero_point, bit_width=output_bit_width, signed=output_signed, training=output_training)
            else:
                tensors = [(qt.value if isinstance(qt, QuantTensor) else qt) for qt in tensors]
                output_value = torch.cat(tensors, dim=dim)
                return QuantTensor(output_value)

    def __neg__(self):
        neg_value = (-self.int(float_datatype=True) - self.zero_point) * self.scale
        if self.signed:
            return QuantTensor(value=neg_value, scale=self.scale, zero_point=self.zero_point, bit_width=self.bit_width, signed=self.signed, training=self.training)
        else:
            return QuantTensor(value=neg_value, scale=self.scale, zero_point=self.zero_point, bit_width=self.bit_width + 1, signed=True, training=self.training)

    def __add__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            self.check_scaling_factors_same(other)
            output_value = self.value + other.value
            output_scale = (self.scale + other.scale) / 2
            output_zero_point = self.zero_point + other.zero_point
            max_val = max_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            max_val += max_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            min_val = min_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            min_val += min_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            output_bit_width = ceil_ste(torch.log2(max_val - min_val))
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            output = QuantTensor(value=output_value, scale=output_scale, zero_point=output_zero_point, bit_width=output_bit_width, signed=output_signed, training=output_training)
        elif isinstance(other, QuantTensor):
            output = QuantTensor(self.value + other.value)
        else:
            output = QuantTensor(self.value + other)
        return output

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            output_value = self.value * other.value
            output_scale = self.scale * other.scale
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point
            else:
                raise RuntimeError('Zero-points of mul operands are non-zero, not supported.')
            output = QuantTensor(value=output_value, scale=output_scale, zero_point=output_zero_point, bit_width=output_bit_width, signed=output_signed, training=output_training)
        elif isinstance(other, QuantTensor):
            output = QuantTensor(self.value * other.value)
        else:
            output = QuantTensor(self.value * other)
        return output

    def __sub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            output_tensor = self.value / other.tensor
            output_scale = self.scale / other.scale
            output_bit_width = self.bit_width - other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point / other.zero_point
            else:
                output_zero_point = None
            output = QuantTensor(value=output_tensor, scale=output_scale, zero_point=output_zero_point, bit_width=output_bit_width, signed=output_signed, training=output_training)
        elif isinstance(other, QuantTensor):
            output = QuantTensor(self.value / other.value)
        else:
            output = QuantTensor(self.value / other)
        return output

    def __abs__(self):
        if self.signed:
            abs_value = (torch.abs(self.int(float_datatype=True)) - self.zero_point) * self.scale
            return QuantTensor(value=abs_value, scale=self.scale, zero_point=self.zero_point, bit_width=self.bit_width - 1, signed=False, training=self.training)
        else:
            return self

    def __pos__(self):
        return self


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


class QuantLayerMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor: bool, export_mode: bool=False, export_debug_name: Optional[str]=None, export_handler: Optional=None, cache_inference_quant_inp: bool=False, cache_inference_quant_out: bool=False, cache_quant_io_metadata_only: bool=True):
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor
        self.export_handler = export_handler
        self.cache_inference_quant_inp = cache_inference_quant_inp
        self.cache_inference_quant_out = cache_inference_quant_out
        self.cache_quant_io_metadata_only = cache_quant_io_metadata_only
        self._export_mode = export_mode
        self._export_debug_name = export_debug_name
        self._cached_inp = None
        self._cached_out = None
        self.export_input_debug = False
        self.export_output_debug = False

    @property
    def export_debug_name(self):
        return self._export_debug_name

    @export_debug_name.setter
    def export_debug_name(self, value):
        self._export_debug_name = value

    @property
    @abstractmethod
    def channelwise_separable(self) ->bool:
        pass

    @property
    @abstractmethod
    def requires_export_handler(self):
        pass

    @property
    def export_mode(self):
        if self._export_mode and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return
        elif value and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler.reset()
        self._export_mode = value

    @property
    def is_quant_input_signed(self) ->Optional[bool]:
        if self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    def _set_global_is_quant_layer(self, value):
        config._IS_INSIDE_QUANT_LAYER = value

    def quant_input_scale(self):
        if self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    @property
    def is_quant_output_signed(self) ->Optional[bool]:
        if self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_output_scale(self):
        if self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_zero_point(self):
        if self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_bit_width(self):
        if self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]):
        self._set_global_is_quant_layer(True)
        if torch._C._get_tracing_state() is not None and isinstance(inp, tuple) and len(inp) == len(QuantTensor._fields) and all([isinstance(t, Tensor) for t in inp]):
            inp = QuantTensor(*inp)
        if isinstance(inp, QuantTensor):
            if not self.training and not self._export_mode and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp
        else:
            inp = QuantTensor(inp, training=self.training)
            if not self.training and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_quant_out:
            self._cached_out = _CachedIO(quant_output.detach(), self.cache_quant_io_metadata_only)
        self._set_global_is_quant_layer(False)
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class ScaleHandlerMixin(object):

    @classmethod
    def validate_scalar_scale(cls, scale: Tensor):
        if scale is None:
            raise RuntimeError('Scale cannot be None.')
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError('Only per-tensor scaling is supported.')
        return scale.item()

    @classmethod
    def validate_scalar_int_exponent(cls, scale: Tensor):
        cls.validate_scalar_scale(scale)
        exponent = math.log2(scale)
        if not exponent.is_integer():
            raise RuntimeError('Only power-of-two scale factors are supported.')
        exponent = int(exponent)
        return exponent

    @classmethod
    def validate_neg_scalar_int_exponent(cls, scale: Tensor):
        return -cls.validate_scalar_int_exponent(scale)


class DPUQuantLayerHandler(ONNXBaseHandler, BitWidthHandlerMixin, ScaleHandlerMixin, ABC):

    @classmethod
    def quant_input_scale(cls, module: QuantLayerMixin):
        scale = module.quant_input_scale()
        return cls.validate_neg_scalar_int_exponent(scale)

    @classmethod
    def quant_output_scale(cls, module: QuantLayerMixin):
        scale = module.quant_output_scale()
        return cls.validate_neg_scalar_int_exponent(scale)

    @classmethod
    def quant_input_bit_width(cls, module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return cls.validate_8b_bit_width(bit_width)

    @classmethod
    def quant_output_bit_width(cls, module: QuantLayerMixin):
        bit_width = module.quant_output_bit_width()
        return cls.validate_8b_bit_width(bit_width)

    @classmethod
    def quant_output_shape(cls, module: QuantLayerMixin):
        cached_out = module._cached_out
        if cached_out is None:
            raise RuntimeError('Caching of outputs is required')
        return cached_out.shape

    def prepare_from_cached_io(self, cached_io):
        cached_inp, cached_out = cached_io
        self.symbolic_kwargs = {'output_shape': cached_out.shape, 'input_bit_width': self.validate_8b_bit_width(cached_inp.bit_width), 'input_scale': self.validate_neg_scalar_int_exponent(cached_inp.scale), 'output_bit_width': self.validate_8b_bit_width(cached_out.bit_width), 'output_scale': self.validate_neg_scalar_int_exponent(cached_out.scale)}


DOMAIN_STRING = 'xir.onnx'


class DPUQuantReLUFn(Function):

    @staticmethod
    def symbolic(g, x, output_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        ret = g.op(f'{DOMAIN_STRING}::Relu', x, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(ctx, x, output_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        return x


def _replace_dependency(injector, current_attr, spec):
    replaced_dependency = injector.__dependencies__[current_attr]
    injector.__dependencies__[current_attr] = spec
    _check_loops(injector.__name__, injector.__dependencies__)
    _check_circles(injector.__dependencies__)
    return replaced_dependency


VALUE_ATTR_NAME = 'value'


def _is_narrow_range(quant_injector):
    if 'narrow_range' in quant_injector:
        return quant_injector.narrow_range
    return None


def _is_signed(quant_injector):
    if 'signed' in quant_injector:
        return quant_injector.signed
    return None


class AutoName(str, Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(self).lower() == str(other).lower()


def float_to_int_impl_to_enum(module):
    if isinstance(module, RoundSte):
        return FloatToIntImplType.ROUND
    elif isinstance(module, RoundToZeroSte):
        return FloatToIntImplType.ROUND_TO_ZERO
    elif isinstance(module, FloorSte):
        return FloatToIntImplType.FLOOR
    elif isinstance(module, CeilSte):
        return FloatToIntImplType.CEIL
    elif isinstance(module, DPURoundSte):
        return DPURoundSte
    else:
        return None


def _rounding_mode(quant_injector):
    if 'float_to_int_impl_type' in quant_injector:
        return str(quant_injector.float_to_int_impl_type)
    elif 'float_to_int_impl' in quant_injector:
        try:
            return str(float_to_int_impl_to_enum(quant_injector.float_to_int_impl))
        except:
            return None
    else:
        return None


def _update_state_dict_impl(quant_injector):
    try:
        impl = quant_injector.update_state_dict_impl
    except:
        impl = None
    return impl


def _is_act_enabled(act_impl, tensor_quant):
    if act_impl is None:
        return False
    elif isinstance(act_impl, nn.Hardtanh) and tensor_quant is not None:
        return False
    else:
        return True


def _is_passthrough_act(quant_injector):
    if 'passthrough_act' in quant_injector:
        return quant_injector.passthrough_act
    return False


DEFAULT_MOMENTUM = 0.1


SCALAR_SHAPE = ()


@torch.jit.ignore
def inplace_momentum_update(tensor: torch.Tensor, update: torch.Tensor, momentum: Optional[float], counter: int, new_counter: int) ->torch.Tensor:
    if momentum is None:
        tensor.mul_(counter / new_counter)
        tensor.add_(update / new_counter)
    else:
        tensor.mul_(1 - momentum)
        tensor.add_(momentum * update)
    return tensor


@torch.jit.ignore
def inplace_tensor_mul(tensor: torch.Tensor, value: torch.Tensor) ->torch.Tensor:
    tensor.mul_(value)
    return tensor


class MinMaxScalingInit:

    def __init__(self, min_val: float, max_val: float):
        self.scaling_init = torch.tensor(max(abs(float(min_val)), abs(float(max_val))))

    def __call__(self):
        return self.scaling_init


MIN_INT_BIT_WIDTH = 2


def solve_bit_width_impl_from_enum(impl_type):
    if impl_type == BitWidthImplType.CONST:
        return BitWidthConst
    elif impl_type == BitWidthImplType.PARAMETER:
        return BitWidthParameter
    else:
        raise Exception(f'{impl_type} not recognized.')


def solve_restrict_value_impl_from_enum(impl_type):
    if impl_type == RestrictValueType.FP:
        return FloatRestrictValue
    elif impl_type == RestrictValueType.LOG_FP:
        return LogFloatRestrictValue
    elif impl_type == RestrictValueType.POWER_OF_TWO:
        return PowerOfTwoRestrictValue
    else:
        raise RuntimeError(f'{impl_type} not recognized.')


DEFAULT_STD_DEV_EPSILON = 1e-08


SCALING_STATS_REDUCE_DIM = 1


def solve_float_to_int_impl_from_enum(impl_type):
    if impl_type == FloatToIntImplType.ROUND:
        return RoundSte
    elif impl_type == FloatToIntImplType.FLOOR:
        return FloorSte
    elif impl_type == FloatToIntImplType.CEIL:
        return CeilSte
    elif impl_type == FloatToIntImplType.ROUND_TO_ZERO:
        return RoundToZeroSte
    elif impl_type == FloatToIntImplType.DPU:
        return DPURoundSte
    else:
        raise Exception(f'{impl_type} not recognized.')


class ConvertRuntimeStatsToParameter:

    def __init__(self, restrict_scaling_impl: Module):
        self.restrict_scaling_impl = restrict_scaling_impl
        scaling_impl_postfix = 'fused_activation_quant_proxy.tensor_quant.scaling_impl'
        self.scaling_impl_postfix = scaling_impl_postfix
        self.runtime_stats_postfix = scaling_impl_postfix + '.runtime_stats'
        self.running_stats_postfix = scaling_impl_postfix + '.runtime_stats.running_stats'
        self.scaling_parameter_postfix = scaling_impl_postfix + '.value'

    def __call__(self, prefix, state_dict):
        running_stats_key = prefix + self.running_stats_postfix
        scaling_parameter_key = prefix + self.scaling_parameter_postfix
        if running_stats_key in state_dict and not scaling_parameter_key in state_dict:
            scaling_init = state_dict[running_stats_key]
            scaling_init = scaling_init.abs()
            scaling_init = self.restrict_scaling_impl.restrict_init_tensor(scaling_init)
            state_dict[scaling_parameter_key] = scaling_init
        for k in list(state_dict.keys()):
            if k.startswith(prefix + self.runtime_stats_postfix):
                del state_dict[k]


class DPUQuantEltwiseAddFn(Function):

    @staticmethod
    def symbolic(g, x, y, input_bit_width, input_scale, other_bit_width, other_scale, output_bit_width, output_scale):
        ret = g.op(f'{DOMAIN_STRING}::Add', x, y, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale, other_bit_width, other_scale], vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(ctx, x, y, input_bit_width, input_scale, other_bit_width, other_scale, output_bit_width, output_scale):
        return x


def filter_kwargs(kwargs_prefix, kwargs: dict):
    return {k[len(kwargs_prefix):]: v for k, v in kwargs.items() if k.startswith(kwargs_prefix)}


class QuantProxyMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self, quant, proxy_protocol, none_quant_injector, proxy_prefix: str, kwargs_prefix: str, **kwargs):
        proxy_name = proxy_prefix + 'quant'
        if quant is None:
            quant_injector = none_quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        elif isclass(quant) and issubclass(quant, (Injector, ExtendedInjector)):
            quant_injector = quant
            quant_injector = quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        else:
            if not isinstance(quant, proxy_protocol):
                raise RuntimeError('The quantizer passed does not adhere to the quantization protocol.')
            quant.add_tracked_module(self)
            if filter_kwargs(kwargs_prefix, kwargs):
                warn('Keyword arguments are being passed but they not being used.')
        setattr(self, proxy_name, quant)


class DPUQuantMaxPoolFn(Function):

    @staticmethod
    def symbolic(g, x, kernel_shape, pads, strides, ceil_mode, dilations, out_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        if isinstance(pads, int) and pads != 0 or isinstance(pads, (list, tuple)) and any([(p != 0) for p in pads]):
            x = g.op(f'{DOMAIN_STRING}::Pad', x, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[input_bit_width, input_scale], pads_i=pads)
        ret = g.op(f'{DOMAIN_STRING}::MaxPool', x, kernel_shape_i=kernel_shape, strides_i=strides, auto_pad_s='VALID', dilations_i=dilations, ceil_mode_i=ceil_mode, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(ctx, x, kernel_shape, pads, strides, ceil_mode, dilations, out_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class Kernel2dApplHandlerMixin(ABC):

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 4
        else:
            padding = list(module.padding) + list(module.padding)
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride] * 2
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module):
        if isinstance(module.dilation, int):
            return [module.dilation] * 2
        else:
            return list(module.dilation)

    @staticmethod
    def kernel_shape(module):
        if isinstance(module.kernel_size, int):
            return [module.kernel_size] * 2
        else:
            return list(module.kernel_size)


class QuantMaxPool2d(QuantLayerMixin, MaxPool2d):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, return_quant_tensor: bool=True):
        MaxPool2d.__init__(self, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)


class DPUQuantMaxPool2dHandler(DPUQuantLayerHandler, Kernel2dApplHandlerMixin, ABC):
    handled_layer = QuantMaxPool2d

    @staticmethod
    def _solve_max_pool2d_kwargs(inp, args, kwargs):
        signature = inspect.signature(F._max_pool2d)
        ba = signature.bind(inp, *args, **kwargs)
        ba.apply_defaults()
        if 'return_indices' in ba.arguments:
            assert not ba.arguments['return_indices']
            del ba.arguments['return_indices']
        return ba.arguments

    def prepare_for_export(self, module: QuantMaxPool2d):
        self.symbolic_kwargs = {'kernel_shape': self.kernel_shape(module), 'pads': self.padding(module), 'strides': self.stride(module), 'ceil_mode': module.ceil_mode, 'dilations': self.dilation(module), 'output_shape': self.quant_output_shape(module), 'input_bit_width': self.quant_input_bit_width(module), 'input_scale': self.quant_input_scale(module), 'output_bit_width': self.quant_output_bit_width(module), 'output_scale': self.quant_output_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = DPUQuantMaxPoolFn.apply(inp, *self.symbolic_kwargs.values())
        return ret

    def cached_symbolic_execution(self, inp: Tensor, *args, **kwargs):
        solved_kwargs = self._solve_max_pool2d_kwargs(inp, args, kwargs)
        return DPUQuantMaxPoolFn.apply(*solved_kwargs.values(), *self.symbolic_kwargs.values())


class DPUQuantAvgPoolFn(Function):

    @staticmethod
    def symbolic(g, x, kernel_shape, strides, pads, out_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        if kernel_shape == [1, 1]:
            return x
        if list(out_shape[2:]) == [1, 1]:
            ret = g.op(f'{DOMAIN_STRING}::GlobalAveragePool', x, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale])
        else:
            ret = g.op(f'{DOMAIN_STRING}::AveragePool', x, kernel_shape_i=kernel_shape, strides_i=strides, pads_i=pads, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(ctx, x, kernel_shape, strides, pads, out_shape, input_bit_width, input_scale, output_bit_width, output_scale):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class DPUQuantLinearFn(Function):

    @staticmethod
    def symbolic(g, x, int_weight, int_bias, out_shape, input_bit_width, input_scale, output_bit_width, output_scale, weight_bit_width, weight_scale, bias_bit_width, bias_scale):
        vai_quant_s = ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights']
        if int_bias is not None:
            vai_quant_s += ['vai_quant_biases']
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, int_weight, int_bias, transB_i=1, vai_quant_s=vai_quant_s, vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale], vai_quant_weights_i=[weight_bit_width, weight_scale], vai_quant_biases_i=[bias_bit_width, bias_scale])
        elif int_bias is None and torch_version <= version.parse('1.4.0'):
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, int_weight, torch.tensor(0), transB_i=1, vai_quant_s=vai_quant_s, vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale], vai_quant_weights_i=[weight_bit_width, weight_scale])
        else:
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, int_weight, transB_i=1, vai_quant_s=vai_quant_s, vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale], vai_quant_weights_i=[weight_bit_width, weight_scale])
        return ret

    @staticmethod
    def forward(ctx, x, int_weight, int_bias, out_shape, input_bit_width, input_scale, output_bit_width, output_scale, weight_bit_width, weight_scale, int_bias_bit_width, int_bias_scale):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class HeScalingInit:

    def __init__(self, tracked_parameter_list: List[torch.nn.Parameter]):
        self.tracked_parameter_list = tracked_parameter_list

    def __call__(self):
        scaling_init = 0.0
        for param in self.tracked_parameter_list:
            two_dim_param = param.view(param.shape[0], -1)
            scaling_init += math.sqrt(2.0 / two_dim_param.shape[1])
        scaling_init /= len(self.tracked_parameter_list)
        return torch.tensor(scaling_init)


class ParameterFromStatsScalingInit:

    def __init__(self, parameter_stats_scaling_init_impl):
        self.init_impl = parameter_stats_scaling_init_impl

    def __call__(self):
        return self.init_impl(torch.tensor(0.0))


class ScalingConstInit:

    def __init__(self, scaling_const):
        self.scaling_const = scaling_const

    def __call__(self):
        return self.scaling_const


class DPUQuantConv2dFn(Function):

    @staticmethod
    def symbolic(g, x, int_weight, int_bias, out_shape, input_bit_width, input_scale, output_bit_width, output_scale, weight_bit_width, weight_scale, bias_bit_width, bias_scale, kernel_size, padding, stride, groups, dilation):
        if isinstance(padding, int) and padding != 0 or isinstance(padding, (list, tuple)) and any([(p != 0) for p in padding]):
            x = g.op(f'{DOMAIN_STRING}::Pad', x, vai_quant_s=['vai_quant_in', 'vai_quant_out'], vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[input_bit_width, input_scale], pads_i=padding)
        vai_quant_s = ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights']
        if int_bias is not None:
            vai_quant_s += ['vai_quant_biases']
            ret = g.op(f'{DOMAIN_STRING}::Conv', x, int_weight, int_bias, vai_quant_s=vai_quant_s, vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale], vai_quant_weights_i=[weight_bit_width, weight_scale], vai_quant_biases_i=[bias_bit_width, bias_scale], kernel_shape_i=kernel_size, strides_i=stride, auto_pad_s='VALID', group_i=groups, dilations_i=dilation)
        else:
            ret = g.op(f'{DOMAIN_STRING}::Conv', x, int_weight, vai_quant_s=vai_quant_s, vai_quant_in_i=[input_bit_width, input_scale], vai_quant_out_i=[output_bit_width, output_scale], vai_quant_weights_i=[weight_bit_width, weight_scale], kernel_shape_i=kernel_size, strides_i=stride, auto_pad_s='VALID', group_i=groups, dilations_i=dilation)
        return ret

    @staticmethod
    def forward(ctx, x, int_weight, int_bias, out_shape, input_bit_width, input_scale, output_bit_width, output_scale, weight_bit_width, weight_scale, int_bias_bit_width, int_bias_scale, kernel_size, padding, stride, groups, dilation):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class XIRFixFn(Function):

    @staticmethod
    def symbolic(g, x, bit_width, fix_point, signed):
        ret = g.op(f'{DOMAIN_STRING}::Fix', x, bit_width_i=bit_width, fix_point_i=fix_point, signed_i=int(signed))
        return ret

    @staticmethod
    def forward(ctx, x, bit_width, fix_point, signed):
        return x


class XIRConv2dFn(Function):

    @staticmethod
    def symbolic(g, x, weight, bias, is_depthwise, kernel_size, padding, padding_type, stride, dilation, output_shape):
        if is_depthwise and bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::DepthwiseConv2d', x, weight, bias, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        elif is_depthwise and bias is None:
            ret = g.op(f'{DOMAIN_STRING}::DepthwiseConv2d', x, weight, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        elif not is_depthwise and bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::Conv2d', x, weight, bias, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        else:
            ret = g.op(f'{DOMAIN_STRING}::Conv2d', x, weight, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        return ret

    @staticmethod
    def forward(ctx, x, weight, bias, is_depthwise, kernel_size, padding, padding_type, stride, dilation, output_shape):
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)


class XIRConvTranpose2dFn(Function):

    @staticmethod
    def symbolic(g, x, weight, bias, is_depthwise, kernel_size, padding, padding_type, stride, dilation):
        if is_depthwise and bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::DepthwiseConvTranpose2d', x, weight, bias, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        elif is_depthwise and bias is None:
            ret = g.op(f'{DOMAIN_STRING}::DepthwiseConvTranpose2d', x, weight, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        elif not is_depthwise and bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::ConvTranspose2d', x, weight, bias, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        else:
            ret = g.op(f'{DOMAIN_STRING}::ConvTranspose2d', x, weight, kernel_shape_i=kernel_size, padding_type_s=padding_type, pads_i=padding, strides_i=stride, dilations_i=dilation)
        return ret

    @staticmethod
    def forward(ctx, x, weight, bias, is_depthwise, kernel_size, padding, padding_type, stride, dilation, output_shape):
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)


class XIRGemmFn(Function):

    @staticmethod
    def symbolic(g, x, weight, bias):
        if bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, bias, transA_i=0, transB_i=1)
        elif bias is None and torch_version <= version.parse('1.4.0'):
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, torch.tensor(0), transA_i=0, transB_i=1)
        else:
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, transA_i=0, transB_i=1)
        return ret

    @staticmethod
    def forward(ctx, x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)


class ZeroPointHandlerMixin(object):

    @classmethod
    def zero_point_with_dtype(cls, signed, zero_point):
        if not signed:
            if (zero_point < 0).any():
                raise RuntimeError('Zero points have to be positive under unsigned quantization')
            return zero_point.type(torch.uint8)
        else:
            return zero_point.type(torch.int8)

    @classmethod
    def quant_input_zero_point(cls, module):
        signed = module.is_quant_input_signed
        zero_point = module.quant_input_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)

    @classmethod
    def quant_weight_zero_point(cls, module):
        signed = module.is_quant_weight_signed
        zero_point = module.quant_weight_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)

    @classmethod
    def quant_output_zero_point(cls, module):
        signed = module.is_quant_output_signed
        zero_point = module.quant_output_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)


def _is_scalar(x: Tensor):
    return x.shape == SCALAR_SHAPE


class PytorchQuantLayerHandler(BaseHandler, BitWidthHandlerMixin, ZeroPointHandlerMixin, ABC):

    @classmethod
    @abstractmethod
    def explicit_output_dtype(cls) ->bool:
        pass

    @classmethod
    @abstractmethod
    def prepare_qf(cls, module):
        pass

    @classmethod
    @abstractmethod
    def validate(cls, module):
        pass

    @classmethod
    def gen_quant_impl_kwargs(cls, scale: Tensor, zero_point: Tensor, signed: bool, include_dtype=True):
        if _is_scalar(scale):
            assert _is_scalar(zero_point), 'Scalar zero point required'
            scale, zero_point = scale.item(), zero_point.item()
            quant_impl = torch.quantize_per_tensor
        else:
            if _is_scalar(zero_point):
                zero_point = zero_point.expand_as(scale)
            quant_impl = torch.quantize_per_channel
        quant_kwargs = {'scale': scale, 'zero_point': zero_point}
        if include_dtype and signed:
            quant_kwargs['dtype'] = torch.qint8
        elif include_dtype and not signed:
            quant_kwargs['dtype'] = torch.quint8
        return quant_impl, quant_kwargs

    @classmethod
    def prepare_input_quant(cls, module):
        scale = module.quant_input_scale()
        zero_point = cls.quant_input_zero_point(module)
        signed = module.is_quant_input_signed
        quant_impl, quant_kwargs = cls.gen_quant_impl_kwargs(scale, zero_point, signed)
        return quant_impl, quant_kwargs

    @classmethod
    def prepare_output_quant(cls, module):
        scale = module.quant_output_scale()
        zero_point = cls.quant_output_zero_point(module)
        signed = module.is_quant_output_signed
        incl_dtype = cls.explicit_output_dtype()
        quant_impl, quant_kwargs = cls.gen_quant_impl_kwargs(scale, zero_point, signed, incl_dtype)
        return quant_impl, quant_kwargs


BaseArgumentTypes = Union[str, int, float, bool, torch.dtype, torch.Tensor]


Argument = Optional[Union[Tuple[Any, ...], List[Any], Dict[str, Any], slice, 'Node', BaseArgumentTypes]]


_help_mutation = """If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:
    new_args = ... # copy and mutate args
    node.args = new_args
"""


def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")


def _create_immutable_container(base, mutable_functions):
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container


immutable_dict = _create_immutable_container(dict, ['__delitem__', '__setitem__', 'clear', 'pop', 'popitem', 'update'])


immutable_list = _create_immutable_container(list, ['__delitem__', '__iadd__', '__imul__', '__setitem__', 'append', 'clear', 'extend', 'insert', 'pop', 'remove'])


def map_aggregate(a: Argument, fn: Callable[[Argument], Argument]) ->Argument:
    """ Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    if isinstance(a, tuple):
        return tuple(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, list):
        return immutable_list(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return immutable_dict((k, map_aggregate(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
    else:
        return fn(a)


class Node:
    """
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *including the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """

    def __init__(self, graph: 'Graph', name: str, op: str, target: 'Target', args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument'], type: Optional[Any]=None) ->None:
        self.graph = graph
        self.name = name
        assert op in ['placeholder', 'call_method', 'call_module', 'call_function', 'get_attr', 'output', 'root']
        self.op = op
        if op in ['call_method', 'call_module']:
            assert isinstance(target, str)
        self.target = target
        self._input_nodes: Dict[Node, None] = {}
        self.__update_args_kwargs(map_arg(args, lambda x: x), map_arg(kwargs, lambda x: x))
        self.users: Dict['Node', None] = {}
        self.type: Optional[Any] = type
        self._prev = self
        self._next = self
        self._erased = False

    @property
    def next(self) ->'Node':
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    @property
    def prev(self) ->'Node':
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev

    def prepend(self, x: 'Node') ->None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """
        assert self.graph == x.graph, 'Attempting to move a Node into a different Graph'
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def append(self, x: 'Node') ->None:
        """
        Insert x after this node in the list of nodes in the graph.
        Equvalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """
        self._next.prepend(x)

    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p

    @property
    def args(self) ->Tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args

    @args.setter
    def args(self, a: Tuple[Argument, ...]):
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        self.__update_args_kwargs(map_arg(a, lambda x: x), self._kwargs)

    @property
    def kwargs(self) ->Dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k: Dict[str, Argument]):
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        self.__update_args_kwargs(self._args, map_arg(k, lambda x: x))

    @property
    def all_input_nodes(self) ->List['Node']:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    def __update_args_kwargs(self, new_args: Tuple['Argument', ...], new_kwargs: Dict[str, 'Argument']):
        """
        This API is internal. Do *not* call it directly.
        """
        self._args = new_args
        self._kwargs = new_kwargs
        for old_use in self._input_nodes.keys():
            old_use.users.pop(self)
        self._input_nodes = {}
        map_arg(self._args, lambda n: self._input_nodes.setdefault(n))
        map_arg(self._kwargs, lambda n: self._input_nodes.setdefault(n))
        for new_use in self._input_nodes.keys():
            new_use.users.setdefault(self)

    def __repr__(self) ->str:
        return self.name

    def replace_all_uses_with(self, replace_with: 'Node') ->List['Node']:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.

        Returns:

            The list of Nodes on which this change was made.
        """
        to_process = list(self.users)
        for use_node in to_process:

            def maybe_replace_node(n: Node) ->Node:
                if n == self:
                    return replace_with
                else:
                    return n
            new_args = map_arg(use_node.args, maybe_replace_node)
            new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            use_node.__update_args_kwargs(new_args, new_kwargs)
        assert len(self.users) == 0
        return to_process


Target = Union[Callable[..., Any], str]


class _InsertPoint:

    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert


def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) ->str:
    args_s = ', '.join(repr(a) for a in args)
    kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f'{args_s}, {kwargs_s}'
    return args_s or kwargs_s


def _format_target(base: str, target: str) ->str:
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r


def _is_magic(x: str) ->bool:
    return x.startswith('__') and x.endswith('__')


class _node_list:

    def __init__(self, graph: 'Graph', direction: str='_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            if not cur._erased:
                yield cur
            cur = getattr(cur, direction)

    def __reversed__(self):
        return _node_list(self.graph, '_next' if self.direction == '_prev' else '_prev')


def _shadows_builtin_name(name: str) ->bool:
    return name in builtins.__dict__ or name in keyword.kwlist or name in {'inf', 'nan', 'NoneType'}


def _snake_case(s: str) ->str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)


def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type) and obj.__module__ != 'typing':
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return '...'
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _find_module_of_method(orig_method: Callable[..., Any]) ->str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')


def get_qualified_name(func: Callable[..., Any]) ->str:
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')
    return f'{module}.{name}'


reflectable_magic_methods = {'add': '{} + {}', 'sub': '{} - {}', 'mul': '{} * {}', 'floordiv': '{} // {}', 'truediv': '{} / {}', 'div': '{} / {}', 'mod': '{} % {}', 'pow': '{} ** {}', 'lshift': '{} << {}', 'rshift': '{} >> {}', 'and': '{} & {}', 'or': '{} | {}', 'xor': '{} ^ {}', 'getitem': '{}[{}]'}


magic_methods = dict({'eq': '{} == {}', 'ne': '{} != {}', 'lt': '{} < {}', 'gt': '{} > {}', 'le': '{} <= {}', 'ge': '{} >= {}', 'pos': '+{}', 'neg': '-{}', 'invert': '~{}'}, **reflectable_magic_methods)


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ' + line) for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _assign_attr(from_obj: Any, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        t = getattr(to_module, item, None)
        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t
    setattr(to_module, field, from_obj)


def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            return
        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = f, t
    orig = getattr(from_module, field)
    if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)


def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [(line + '\n') for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)


def _forward_from_src(src: str):
    gbls: Dict[str, Any] = {'inf': math.inf, 'nan': math.nan, 'NoneType': type(None)}
    exec_with_source(src, gbls)
    return gbls['forward']


def deserialize_graphmodule(body: dict) ->torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """


    class CodeOnlyModule(torch.nn.Module):

        def __init__(self, body):
            super().__init__()
            self.__dict__ = body
    try:
        CodeOnlyModule.forward = _forward_from_src(body['_code'])
    except KeyError:
        CodeOnlyModule.forward = _forward_from_src(body['code'])


    class KeepModules(Tracer):

        def is_leaf_module(self, _: torch.nn.Module, __: str) ->bool:
            return True
    com = CodeOnlyModule(body)
    return GraphModule(com, KeepModules().trace(com))


class HadamardClassifier(QuantLayerMixin, nn.Module):

    def __init__(self, in_channels, out_channels, fixed_scale=False, return_quant_tensor: bool=False):
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)
        nn.Module.__init__(self)
        if hadamard is None:
            raise Exception('Hadamard layer requires scipy to be installed.')
        self.out_channels = out_channels
        self.in_channels = in_channels
        sz = 2 ** int(math.ceil(math.log(max(in_channels, out_channels), 2)))
        mat = torch.from_numpy(hadamard(sz)).float()
        self.register_buffer('proj', mat)
        init_scale = 1.0 / math.sqrt(self.out_channels)
        if fixed_scale:
            self.register_buffer('scale', torch.tensor(init_scale))
        else:
            self.scale = nn.Parameter(torch.tensor(init_scale))
        self.eps = 1e-08

    def forward(self, inp):
        output_scale = None
        output_zp = None
        output_bit_width = None
        inp = self.unpack_input(inp)
        norm = inp.value.norm(p='fro', keepdim=True) + self.eps
        out = inp.value / norm
        out = nn.functional.linear(out, self.proj[:self.out_channels, :self.in_channels])
        out = -self.scale * out
        if inp.scale is not None:
            output_scale = inp.scale * self.scale / norm
        if inp.bit_width is not None:
            output_bit_width = self.max_output_bit_width(inp.bit_width)
        if self.return_quant_tensor and inp.zero_point is not None and (inp.zero_point != 0.0).any():
            raise RuntimeError('Computing zero point of output accumulator not supported yet.')
        else:
            output_zp = inp.zero_point
        out = QuantTensor(value=out, scale=output_scale, zero_point=output_zp, bit_width=output_bit_width, signed=True, training=self.training)
        return out

    def max_output_bit_width(self, input_bit_width):
        max_input_val = max_int(bit_width=input_bit_width, narrow_range=False, signed=False)
        max_output_val = max_input_val * self.in_channels
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(HadamardClassifier, self).state_dict(destination, prefix, keep_vars)
        del state_dict[prefix + 'proj']
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(HadamardClassifier, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        proj_key = prefix + 'proj'
        if proj_key in missing_keys:
            missing_keys.remove(proj_key)


NON_ZERO_EPSILON = 1e-06


REMOVE_ZERO_BIT_WIDTH = 0.1


class QuantDropout(QuantLayerMixin, Dropout):

    def __init__(self, p: float=0.5, return_quant_tensor: bool=True):
        Dropout.__init__(self, p=p, inplace=False)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)


def rename_state_dict_by_prefix(old_prefix, new_prefix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)


class QuantMaxPool1d(QuantLayerMixin, MaxPool1d):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, return_quant_tensor: bool=True):
        MaxPool1d.__init__(self, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            return self.export_handler(x.value)
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)


class ScaleBias(Module):

    def __init__(self, num_features: int, bias: bool, runtime_shape=(1, -1, 1, 1)):
        super(ScaleBias, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features)) if bias else None
        self.runtime_shape = runtime_shape

    def forward(self, input):
        return input * self.weight.view(self.runtime_shape) + self.bias.view(self.runtime_shape)


class QuantUpsample(QuantLayerMixin, Upsample):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, return_quant_tensor: bool=True):
        Upsample.__init__(self, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        if self.mode != 'nearest':
            assert x.scale is not None, 'Input scale factor required to interpolate correctly'
            y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingBilinear2d(QuantLayerMixin, UpsamplingBilinear2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool=True):
        UpsamplingBilinear2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        assert x.scale is not None, 'Input scale factor required to interpolate correctly'
        y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool=True):
        UpsamplingNearest2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) ->bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        y = x.set(value=y_value)
        return self.pack_output(y)


class TupleSequential(Sequential):

    def output(self, mod, input):
        if isinstance(input, tuple):
            return mod(*input)
        else:
            return mod(input)

    def forward(self, *input):
        modules = list(self._modules.values())
        out = self.output(modules[0], input)
        for mod in modules[1:]:
            out = self.output(mod, out)
        return out


CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]


INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]


KERNEL_SIZE = 3


LAST_FC_IN_FEATURES = 512


class TensorNorm(nn.Module):

    def __init__(self, eps=0.0001, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return (x - self.running_mean) / (self.running_var + self.eps).pow(0.5) * self.weight + self.bias


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width, min_val=-1.0, max_val=1.0 - 2.0 ** -7, narrow_range=False, restrict_scaling_type=RestrictValueType.POWER_OF_TWO))
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(kernel_size=KERNEL_SIZE, in_channels=in_ch, out_channels=out_ch, bias=False, weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=0.0001))
            self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(in_features=in_features, out_features=out_features, bias=False, weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=0.0001))
            self.linear_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
        self.linear_features.append(QuantLinear(in_features=LAST_FC_IN_FEATURES, out_features=num_classes, bias=False, weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


DROPOUT = 0.2


class FC(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_channels, out_features, in_features=(28, 28)):
        super(FC, self).__init__()
        self.features = ModuleList()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width))
        self.features.append(Dropout(p=DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in out_features:
            self.features.append(QuantLinear(in_features=in_features, out_features=out_features, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            self.features.append(Dropout(p=DROPOUT))
        self.features.append(QuantLinear(in_features=in_features, out_features=num_classes, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant))
        self.features.append(TensorNorm())
        for m in self.modules():
            if isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x


class squared_hinge_loss(Function):

    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.0).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


class SqrHingeLoss(nn.Module):

    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight_bit_width, act_bit_width, act_scaling_per_channel, bias, groups=1, bn_eps=1e-05, shared_act=None, return_quant_tensor=False):
        super(ConvBlock, self).__init__()
        self.conv = QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, weight_bit_width=weight_bit_width, weight_quant=CommonIntWeightPerChannelQuant)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if shared_act is None:
            self.activ = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, scaling_per_channel=act_scaling_per_channel, per_channel_broadcastable_shape=(1, out_channels, 1, 1), return_quant_tensor=return_quant_tensor)
        else:
            self.activ = shared_act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DwsConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, bit_width, pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=3, padding=1, stride=stride, weight_bit_width=bit_width, act_bit_width=bit_width)
        self.pw_conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, weight_bit_width=bit_width, act_bit_width=bit_width, activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


FIRST_LAYER_BIT_WIDTH = 8


class MobileNet(nn.Module):

    def __init__(self, channels, first_stage_stride, bit_width, in_channels=3, num_classes=1000):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]
        self.features = Sequential()
        init_block = ConvBlock(in_channels=in_channels, out_channels=init_block_channels, kernel_size=3, stride=2, weight_bit_width=FIRST_LAYER_BIT_WIDTH, activation_scaling_per_channel=True, act_bit_width=bit_width)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if j == 0 and (i != 0 or first_stage_stride) else 1
                mod = DwsConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, bit_width=bit_width, pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width)
        self.output = QuantLinear(in_channels, num_classes, bias=True, bias_quant=IntBias, weight_quant=CommonIntWeightPerTensorQuant, weight_bit_width=bit_width)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


class ProxylessBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bn_eps, expansion, bit_width, depthwise_bit_width, shared_act):
        super(ProxylessBlock, self).__init__()
        self.use_bc = expansion > 1
        mid_channels = in_channels * expansion
        if self.use_bc:
            self.bc_conv = ConvBlock(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, groups=1, bn_eps=bn_eps, act_scaling_per_channel=True, weight_bit_width=bit_width, bias=False, act_bit_width=depthwise_bit_width)
        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_channels, bn_eps=bn_eps, act_scaling_per_channel=False, weight_bit_width=depthwise_bit_width, act_bit_width=bit_width, bias=False)
        self.pw_conv = ConvBlock(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, bn_eps=bn_eps, weight_bit_width=bit_width, shared_act=shared_act, bias=False, act_bit_width=None, act_scaling_per_channel=None)

    def forward(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bn_eps, expansion, residual, shortcut, bit_width, depthwise_bit_width, shared_act):
        super(ProxylessUnit, self).__init__()
        assert residual or shortcut
        assert shared_act is not None
        self.residual = residual
        self.shortcut = shortcut
        if self.residual:
            self.body = ProxylessBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bn_eps=bn_eps, expansion=expansion, bit_width=bit_width, depthwise_bit_width=depthwise_bit_width, shared_act=shared_act)
            self.shared_act = shared_act

    def forward(self, x):
        if not self.residual:
            return x
        if not self.shortcut:
            x = self.body(x)
            return x
        identity = x
        x = self.body(x)
        x = identity + x
        x = self.shared_act(x)
        return x


class ProxylessNAS(nn.Module):

    def __init__(self, channels, init_block_channels, final_block_channels, residuals, shortcuts, kernel_sizes, expansions, bit_width, depthwise_bit_width, first_layer_weight_bit_width, hadamard_classifier, bn_eps=0.001, in_channels=3, num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.features = nn.Sequential()
        init_block = ConvBlock(in_channels=in_channels, out_channels=init_block_channels, kernel_size=3, stride=2, padding=1, groups=1, bn_eps=bn_eps, act_scaling_per_channel=False, bias=False, act_bit_width=bit_width, weight_bit_width=first_layer_weight_bit_width)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        shared_act = None
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            residuals_per_stage = residuals[i]
            shortcuts_per_stage = shortcuts[i]
            kernel_sizes_per_stage = kernel_sizes[i]
            expansions_per_stage = expansions[i]
            for j, out_channels in enumerate(channels_per_stage):
                residual = residuals_per_stage[j] == 1
                shortcut = shortcuts_per_stage[j] == 1
                kernel_size = kernel_sizes_per_stage[j]
                expansion = expansions_per_stage[j]
                stride = 2 if j == 0 and i != 0 else 1
                if not shortcut:
                    shared_act = QuantIdentity(bit_width=bit_width, act_quant=CommonIntActQuant, return_quant_tensor=True)
                unit = ProxylessUnit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bn_eps=bn_eps, expansion=expansion, residual=residual, shortcut=shortcut, bit_width=bit_width, depthwise_bit_width=depthwise_bit_width, shared_act=shared_act)
                stage.add_module('unit{}'.format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        final_block = ConvBlock(in_channels=in_channels, out_channels=final_block_channels, kernel_size=1, stride=1, padding=0, groups=1, bn_eps=bn_eps, act_scaling_per_channel=False, act_bit_width=bit_width, weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.features.add_module('final_block', final_block)
        in_channels = final_block_channels
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width)
        if hadamard_classifier:
            self.output = HadamardClassifier(in_channels=in_channels, out_channels=num_classes, fixed_scale=False)
        else:
            self.output = QuantLinear(in_features=in_channels, out_features=num_classes, bias=True, bias_quant=IntBias, weight_bit_width=bit_width, weight_quant=CommonIntWeightPerTensorQuant)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QuantConv2d(in_channels, v, kernel_size=3, stride=1, padding=1, groups=1, bias=not batch_norm, weight_bit_width=bit_width, weight_quant=CommonIntWeightPerChannelQuant)
            act = QuantReLU(act_quant=CommonUintActQuant, bit_width=bit_width, return_quant_tensor=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)


class QuantVGG(nn.Module):

    def __init__(self, cfg, batch_norm, bit_width=8, num_classes=1000):
        super(QuantVGG, self).__init__()
        self.features = make_layers(cfg, batch_norm, bit_width)
        self.avgpool = QuantAvgPool2d(kernel_size=(7, 7), stride=1, bit_width=bit_width)
        self.classifier = nn.Sequential(QuantLinear(512 * 7 * 7, 4096, bias=True, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=bit_width), QuantReLU(act_quant=CommonUintActQuant, bit_width=bit_width), nn.Dropout(), QuantLinear(4096, 4096, bias=True, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=bit_width), QuantReLU(act_quant=CommonUintActQuant, bit_width=bit_width), nn.Dropout(), QuantLinear(4096, num_classes, bias=False, weight_quant=CommonIntWeightPerTensorQuant, weight_bit_width=bit_width))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AudioPreprocessor(nn.Module):
    """
    A base class for Neural Modules that performs audio preprocessing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length, **kwargs):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.torch_windows = {'hann': torch.hann_window, 'hamming': torch.hamming_window, 'blackman': torch.blackman_window, 'bartlett': torch.bartlett_window, 'ones': torch.ones, None: torch.ones}

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.get_features(input_signal, length)
        processed_length = self.get_seq_len(length.float())
        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        pass

    def get_seq_len(self, length):
        return torch.ceil(length / self.hop_length)


class AudioToSpectrogramPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to spectrograms.
    Uses torchaudio's Spectrogram class as a featurizer.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null']
            Defaults to "hann"
        normalized (bool): Whether to normalize by magnitude after stft
    """

    def __init__(self, *, sample_rate=16000, window_size=0.02, window_stride=0.01, n_window_size=None, n_window_stride=None, n_fft=None, window='hann', normalized=True, **kwargs):
        if not have_torchaudio:
            raise ModuleNotFoundError('torchaudio is not installed but is necessary for AudioToSpectrogramPreprocessor. We recommend you try building it from source for the PyTorch version you have.')
        if window_size and n_window_size:
            raise ValueError(f'{self} received both window_size and n_window_size. Only one should be specified.')
        if window_stride and n_window_stride:
            raise ValueError(f'{self} received both window_stride and n_window_stride. Only one should be specified.')
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)
        super().__init__(n_window_size, n_window_stride, **kwargs)
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(f"Window argument for AudioProcessor is invalid: {window}.For no window function, use 'ones' or None.")
        self.featurizer = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, window_fn=window_fn, normalized=normalized)
        self.featurizer

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)


CONSTANT = 1e-05


def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        if torch.cuda.is_available():
            forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0).cpu()
        else:
            forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0).cpu()
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == 'per_feature':
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == 'all_features':
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :seq_len[i].item()].mean()
            x_std[i] = x[i, :, :seq_len[i].item()].std()
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(self, *, sample_rate=16000, n_window_size=320, n_window_stride=160, window='hann', normalize='per_feature', n_fft=None, preemph=0.97, nfilt=64, lowfreq=0, highfreq=None, log=True, log_zero_guard_type='add', log_zero_guard_value=2 ** -24, dither=CONSTANT, pad_to=16, max_duration=16.7, frame_splicing=1, stft_conv=False, pad_value=0, mag_power=2.0, logger=None):
        super(FilterbankFeatures, self).__init__()
        if n_window_size is None or n_window_stride is None or not isinstance(n_window_size, int) or not isinstance(n_window_stride, int) or n_window_size <= 0 or n_window_stride <= 0:
            raise ValueError(f'{self} got an invalid value for either n_window_size or n_window_stride. Both must be positive ints.')
        if logger:
            logger.info(f'PADDING: {pad_to}')
        else:
            None
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_conv = stft_conv
        if stft_conv:
            if logger:
                logger.info('STFT using conv')
            else:
                None


            class STFTPatch(STFT):

                def __init__(self, *params, **kw_params):
                    super(STFTPatch, self).__init__(*params, **kw_params)

                def forward(self, input_data):
                    return super(STFTPatch, self).transform(input_data)[0]
            self.stft = STFTPatch(self.n_fft, self.hop_length, self.win_length, window)
        else:
            None
            torch_windows = {'hann': torch.hann_window, 'hamming': torch.hamming_window, 'blackman': torch.blackman_window, 'bartlett': torch.bartlett_window, 'none': None}
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
            self.register_buffer('window', window_tensor)
            self.stft = lambda x: torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True, window=self.window)
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        filterbanks = torch.tensor(librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float).unsqueeze(0)
        self.register_buffer('fb', filterbanks)
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - max_length % pad_to
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power
        if log_zero_guard_type not in ['add', 'clamp']:
            raise ValueError(f"{self} received {log_zero_guard_type} for the log_zero_guard_type parameter. It must be either 'add' or 'clamp'.")
        self.log_zero_guard_value = lambda _: log_zero_guard_value
        if isinstance(log_zero_guard_value, str):
            if log_zero_guard_value == 'tiny':
                self.log_zero_guard_value = lambda x: torch.finfo(x.dtype).tiny
            elif log_zero_guard_value == 'eps':
                self.log_zero_guard_value = lambda x: torch.finfo(x.dtype).eps
            else:
                raise ValueError(f"{self} received {log_zero_guard_value} for the log_zero_guard_type parameter. It must be either a number, 'tiny', or 'eps'")
        self.log_zero_guard_type = log_zero_guard_type

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len / self.hop_length)

    @property
    def filter_banks(self):
        return self.fb

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = self.stft(x)
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
        if not self.stft_conv:
            x = x.sum(-1)
        x = torch.matmul(self.fb, x)
        if self.log:
            if self.log_zero_guard_type == 'add':
                x = torch.log(x + self.log_zero_guard_value(x))
            elif self.log_zero_guard_type == 'clamp':
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value(x)))
            else:
                raise ValueError('log_zero_guard_type was not understood')
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)
        max_len = x.size(-1)
        mask = torch.arange(max_len)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool), self.pad_value)
        del mask
        pad_to = self.pad_to
        if not self.training:
            pad_to = 16
        if pad_to == 'max':
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):
    """Featurizer that converts wavs to mel spectrograms.
    We don't use torchaudio's implementation here because the original
    implementation is not the same, so for the sake of backwards-compatibility
    this will use the old FilterbankFeatures for now.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        stft_conv (bool): If True, uses pytorch_stft and convolutions. If
            False, uses torch.stft.
            Defaults to False
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
    """

    def __init__(self, *, sample_rate=16000, window_size=0.02, window_stride=0.01, n_window_size=None, n_window_stride=None, window='hann', normalize='per_feature', n_fft=None, preemph=0.97, features=64, lowfreq=0, highfreq=None, log=True, log_zero_guard_type='add', log_zero_guard_value=2 ** -24, dither=1e-05, pad_to=16, frame_splicing=1, stft_conv=False, pad_value=0, mag_power=2.0, **kwargs):
        if window_size and n_window_size:
            raise ValueError(f'{self} received both window_size and n_window_size. Only one should be specified.')
        if window_stride and n_window_stride:
            raise ValueError(f'{self} received both window_stride and n_window_stride. Only one should be specified.')
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)
        super().__init__(n_window_size, n_window_stride, **kwargs)
        self.featurizer = FilterbankFeatures(sample_rate=sample_rate, n_window_size=n_window_size, n_window_stride=n_window_stride, window=window, normalize=normalize, n_fft=n_fft, preemph=preemph, nfilt=features, lowfreq=lowfreq, highfreq=highfreq, log=log, log_zero_guard_type=log_zero_guard_type, log_zero_guard_value=log_zero_guard_value, dither=dither, pad_to=pad_to, frame_splicing=frame_splicing, stft_conv=stft_conv, pad_value=pad_value, mag_power=mag_power, logger=None)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    def get_seq_len(self, seq_len):
        return self.featurizer.get_seq_len(seq_len)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    def __init__(self, *, sample_rate=16000, window_size=0.02, window_stride=0.01, n_window_size=None, n_window_stride=None, window='hann', n_fft=None, lowfreq=0.0, highfreq=None, n_mels=64, n_mfcc=64, dct_type=2, norm='ortho', log=True, **kwargs):
        if not have_torchaudio:
            raise ModuleNotFoundError('torchaudio is not installed but is necessary for AudioToMFCCPreprocessor. We recommend you try building it from source for the PyTorch version you have.')
        if window_size and n_window_size:
            raise ValueError(f'{self} received both window_size and n_window_size. Only one should be specified.')
        if window_stride and n_window_stride:
            raise ValueError(f'{self} received both window_stride and n_window_stride. Only one should be specified.')
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)
        super().__init__(n_window_size, n_window_stride, **kwargs)
        mel_kwargs = {}
        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels
        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))
        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(f"Window argument for AudioProcessor is invalid: {window}.For no window function, use 'ones' or None.")
        mel_kwargs['window_fn'] = window_fn
        self.featurizer = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, log_mels=log, melkwargs=mel_kwargs)
        self.featurizer

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment
    """

    def __init__(self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, rng=None):
        super(SpecAugment, self).__init__()
        self._rng = random.Random() if rng is None else rng
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = int(self._rng.uniform(0, sh[1] - self.freq_width))
                w = int(self._rng.uniform(0, self.freq_width))
                mask[idx, x_left:x_left + w, :] = 1
            for i in range(self.time_masks):
                y_left = int(self._rng.uniform(0, sh[2] - self.time_width))
                w = int(self._rng.uniform(0, self.time_width))
                mask[idx, :, y_left:y_left + w] = 1
        x = x.masked_fill(mask.type(torch.bool), 0)
        return x


class SpecCutout(nn.Module):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()
        self._rng = random.Random() if rng is None else rng
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = int(self._rng.uniform(0, sh[1] - self.rect_freq))
                rect_y = int(self._rng.uniform(0, sh[2] - self.rect_time))
                w_x = int(self._rng.uniform(0, self.rect_time))
                w_y = int(self._rng.uniform(0, self.rect_freq))
                mask[idx, rect_x:rect_x + w_x, rect_y:rect_y + w_y] = 1
        x = x.masked_fill(mask.type(torch.bool), 0)
        return x


class SpectrogramAugmentation(nn.Module):
    """
    Performs time and freq cuts in one of two ways.

    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.

    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """

    def __init__(self, *, freq_masks=0, time_masks=0, freq_width=10, time_width=10, rect_masks=0, rect_time=5, rect_freq=20, rng=None, **kwargs):
        nn.Module.__init__(self)
        if rect_masks > 0:
            self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq, rng=rng)
        else:
            self.spec_cutout = lambda x: x
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(freq_masks=freq_masks, time_masks=time_masks, freq_width=freq_width, time_width=time_width, rng=rng)
        else:
            self.spec_augment = lambda x: x

    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class MultiplyBatch(nn.Module):
    """
    Augmentation that repeats each element in a batch.
    Other augmentations can be applied afterwards.

    Args:
        mult_batch (int): number of repeats
    """

    def __init__(self, *, mult_batch=1):
        nn.Module.__init__(self)
        self.mult = mult_batch

    @torch.no_grad()
    def forward(self, in_x, in_x_len, in_y, in_y_len):
        out_x = in_x.repeat(self.mult, 1, 1)
        out_y = in_y.repeat(self.mult, 1)
        out_x_len = in_x_len.repeat(self.mult)
        out_y_len = in_y_len.repeat(self.mult)
        return out_x, out_x_len, out_y, out_y_len


class ManifestBase:

    def __init__(self, manifest_paths, labels, max_duration=None, min_duration=None, sort_by_duration=False, max_utts=0, blank_index=-1, unk_index=-1, normalize=True, logger=None):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.blank_index = blank_index
        self.unk_index = unk_index
        self.normalize = normalize
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.logger = None
        data = []
        duration = 0.0
        filtered_duration = 0.0
        for item in self.json_item_gen(manifest_paths):
            if min_duration and item['duration'] < min_duration:
                filtered_duration += item['duration']
                continue
            if max_duration and item['duration'] > max_duration:
                filtered_duration += item['duration']
                continue
            text = ''
            if 'text' in item:
                text = item['text']
            elif 'text_filepath' in item:
                text = self.load_transcript(item['text_filepath'])
            else:
                filtered_duration += item['duration']
                continue
            if normalize:
                text = self.normalize_text(text, labels, logger=self.logger)
            if not isinstance(text, str):
                self.logger.warning('WARNING: Got transcript: {}. It is not a string. Dropping data point'.format(text))
                filtered_duration += item['duration']
                continue
            item['tokens'] = self.tokenize_transcript(text, self.labels_map, self.unk_index, self.blank_index)
            if 'audio_filename' in item and 'audio_filepath' not in item:
                self.logger.warning('Malformed manifest: The key audio_filepath was not found in the manifest. Using audio_filename instead.')
                item['audio_filepath'] = item['audio_filename']
            data.append(item)
            duration += item['duration']
            if max_utts > 0 and len(data) >= max_utts:
                self.logger.info('Stop parsing due to max_utts ({})'.format(max_utts))
                break
        if sort_by_duration:
            data = sorted(data, key=lambda x: x['duration'])
        self._data = data
        self._size = len(data)
        self._duration = duration
        self._filtered_duration = filtered_duration

    @staticmethod
    def normalize_text(text, labels):
        """for the base class remove surrounding whitespace only"""
        return text.strip()

    @staticmethod
    def tokenize_transcript(transcript, labels_map, unk_index, blank_index):
        """tokenize transcript to convert words/characters to indices"""
        special_labels = set([l for l in labels_map.keys() if len(l) > 1])
        tokens = []
        for i, word in enumerate(transcript.split(' ')):
            if i > 0:
                tokens.append(labels_map.get(' ', unk_index))
            if word in special_labels:
                tokens.append(labels_map.get(word))
                continue
            for char in word:
                tokens.append(labels_map.get(char, unk_index))
        tokens = [x for x in tokens if x != blank_index]
        return tokens

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def json_item_gen(manifest_paths):
        for manifest_path in manifest_paths:
            with open(manifest_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    yield json.loads(line)

    @staticmethod
    def load_transcript(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)


ABBREVIATIONS_COMMON = [(re.compile('\\b%s\\.' % x[0]), x[1]) for x in [('ms', 'miss'), ('mrs', 'misess'), ('mr', 'mister'), ('messrs', 'messeurs'), ('dr', 'doctor'), ('drs', 'doctors'), ('st', 'saint'), ('co', 'company'), ('jr', 'junior'), ('sr', 'senior'), ('rev', 'reverend'), ('hon', 'honorable'), ('sgt', 'sergeant'), ('capt', 'captain'), ('maj', 'major'), ('col', 'colonel'), ('lt', 'lieutenant'), ('gen', 'general'), ('prof', 'professor'), ('lb', 'pounds'), ('rep', 'representative'), ('st', 'street'), ('ave', 'avenue'), ('etc', 'et cetera'), ('jan', 'january'), ('feb', 'february'), ('mar', 'march'), ('apr', 'april'), ('jun', 'june'), ('jul', 'july'), ('aug', 'august'), ('sep', 'september'), ('oct', 'october'), ('nov', 'november'), ('dec', 'december')]]


ABBREVIATIONS_EXPANDED = [(re.compile('\\b%s\\.' % x[0]), x[1]) for x in [('ltd', 'limited'), ('fig', 'figure'), ('figs', 'figures'), ('gent', 'gentlemen'), ('ft', 'fort'), ('esq', 'esquire'), ('prep', 'preperation'), ('bros', 'brothers'), ('ind', 'independent'), ('mme', 'madame'), ('pro', 'professional'), ('vs', 'versus'), ('inc', 'include')]]


def clean_abbreviations(string, expanded=False):
    for regex, replacement in ABBREVIATIONS_COMMON:
        string = re.sub(regex, replacement, string)
    if expanded:
        for regex, replacement in ABBREVIATIONS_EXPANDED:
            string = re.sub(regex, replacement, string)
    return string


NUM_CHECK = re.compile('([$]?)(^|\\s)(\\S*[0-9]\\S*)(?=(\\s|$)((\\S*)(\\s|$))?)')


CURRENCY_CHECK = re.compile('\\$')


DECIMAL_CHECK = re.compile('([.,][0-9]{1,2})$')


ORD_CHECK = re.compile('([0-9]+)(st|nd|rd|th)')


THREE_CHECK = re.compile('([0-9]{3})([.,][0-9]{1,2})?([!.?])?$')


TIME_CHECK = re.compile('([0-9]{1,2}):([0-9]{2})(am|pm)?')


class NumberCleaner:

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.curr_num = []
        self.currency = None

    def format_final_number(self, whole_num, decimal):
        if self.currency:
            return_string = inflect.number_to_words(whole_num)
            return_string += ' dollar' if whole_num == 1 else ' dollars'
            if decimal:
                return_string += ' and ' + inflect.number_to_words(decimal)
                return_string += ' cent' if whole_num == decimal else ' cents'
            self.reset()
            return return_string
        self.reset()
        if decimal:
            whole_num += '.' + decimal
            return inflect.number_to_words(whole_num)
        else:

            def convert_to_word(match):
                return ' ' + inflect.number_to_words(match.group(0)) + ' '
            return re.sub('[0-9,]+', convert_to_word, whole_num)

    def clean(self, match):
        ws = match.group(2)
        number = match.group(3)
        _proceeding_symbol = match.group(7)
        time_match = TIME_CHECK.match(number)
        if time_match:
            string = ws + inflect.number_to_words(time_match.group(1)) + '{}{}'
            mins = int(time_match.group(2))
            min_string = ''
            if mins != 0:
                min_string = ' ' + inflect.number_to_words(time_match.group(2))
            ampm_string = ''
            if time_match.group(3):
                ampm_string = ' ' + time_match.group(3)
            return string.format(min_string, ampm_string)
        ord_match = ORD_CHECK.match(number)
        if ORD_CHECK.match(number):
            return ws + inflect.number_to_words(ord_match.group(0))
        if self.currency is None:
            self.currency = match.group(1) or CURRENCY_CHECK.match(number)
        three_match = THREE_CHECK.match(match.group(6))
        if three_match:
            self.curr_num.append(number)
            return ' '
        else:
            whole_num = ''.join(self.curr_num) + number
            decimal = None
            decimal_match = DECIMAL_CHECK.search(whole_num)
            if decimal_match:
                decimal = decimal_match.group(1)[1:]
                whole_num = whole_num[:-len(decimal) - 1]
            whole_num = re.sub('\\.', '', whole_num)
            return ws + self.format_final_number(whole_num, decimal)


def clean_numbers(string):
    cleaner = NumberCleaner()
    string = NUM_CHECK.sub(cleaner.clean, string)
    return string


def clean_punctuations(string, table, punctuation_to_replace):
    for punc, replacement in punctuation_to_replace.items():
        string = re.sub('\\{}'.format(punc), ' {} '.format(replacement), string)
    string = string.translate(table)
    return string


def warn_common_chars(string):
    if re.search('[£€]', string):
        None


def clean_text(string, table, punctuation_to_replace):
    warn_common_chars(string)
    string = unidecode(string)
    string = string.lower()
    string = re.sub('\\s+', ' ', string)
    string = clean_numbers(string)
    string = clean_abbreviations(string)
    string = clean_punctuations(string, table, punctuation_to_replace)
    string = re.sub('\\s+', ' ', string).strip()
    return string


class ManifestEN(ManifestBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalize_text(text, labels, logger=None):
        punctuation = string.punctuation
        punctuation_to_replace = {'+': 'plus', '&': 'and', '%': 'percent'}
        for char in punctuation_to_replace:
            punctuation = punctuation.replace(char, '')
        for l in labels:
            punctuation = punctuation.replace(l, '')
        table = str.maketrans(punctuation, ' ' * len(punctuation))
        try:
            text = clean_text(text, table, punctuation_to_replace)
        except BaseException:
            if logger:
                logger.warning('WARNING: Normalizing {} failed'.format(text))
            else:
                None
            return None
        return text


class AudioDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(self, manifest_filepath, labels, featurizer, max_duration=None, min_duration=None, max_utts=0, blank_index=-1, unk_index=-1, normalize=True, trim=False, bos_id=None, eos_id=None, logger=False, load_audio=True, manifest_class=ManifestEN):
        m_paths = manifest_filepath.split(',')
        self.manifest = manifest_class(m_paths, labels, max_duration=max_duration, min_duration=min_duration, max_utts=max_utts, blank_index=blank_index, unk_index=unk_index, normalize=normalize, logger=logger)
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        if logger:
            logger.info('Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.'.format(self.manifest.duration / 3600, self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(sample['audio_filepath'], offset=offset, duration=duration, trim=self.trim)
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None
        t, tl = sample['tokens'], len(sample['tokens'])
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1
        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.manifest)


class Perturbation(object):

    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class GainPerturbation(Perturbation):

    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        data._samples = data._samples * 10.0 ** (gain / 20.0)


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=None, trim=False, trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return '%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB' % (type(self), self.num_samples, self.sample_rate, self.duration, self.rms_db)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError('Unsupported sample type: %s.' % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=None, int_values=False, offset=0, duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)
        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @classmethod
    def segment_from_file(cls, filename, target_sr=None, n_segments=0, trim=False):
        """Grabs n_segments number of samples from filename randomly from the
        file as opposed to at a specified offset.
        """
        with sf.SoundFile(filename, 'r') as f:
            sample_rate = f.samplerate
            if n_segments > 0 and len(f) > n_segments:
                max_audio_start = len(f) - n_segments
                audio_start = random.randint(0, max_audio_start)
                f.seek(audio_start)
                samples = f.read(n_segments, dtype='float32')
            else:
                samples = f.read(dtype='float32')
        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        self._samples = np.pad(self._samples, (pad_size if symmetric else 0, pad_size), mode='constant')

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set,
        e.g. out
                           of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError('The slice start position (%f s) is out of bounds.' % start_time)
        if end_time < 0.0:
            raise ValueError('The slice end position (%f s) is out of bounds.' % end_time)
        if start_time > end_time:
            raise ValueError('The slice start position (%f s) is later than the end position (%f s).' % (start_time, end_time))
        if end_time > self.duration:
            raise ValueError('The slice end position (%f s) is out of bounds (> %f s)' % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]


class ImpulsePerturbation(Perturbation):

    def __init__(self, manifest_path=None, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse_record = self._rng.sample(self._manifest.data, 1)[0]
        impulse = AudioSegment.from_file(impulse_record['audio_filepath'], target_sr=data.sample_rate)
        data._samples = signal.fftconvolve(data.samples, impulse.samples, 'full')


class NoisePerturbation(Perturbation):

    def __init__(self, manifest_path=None, min_snr_db=40, max_snr_db=50, max_gain_db=300.0, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    def perturb(self, data):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        noise_record = self._rng.sample(self._manifest.data, 1)[0]
        noise = AudioSegment.from_file(noise_record['audio_filepath'], target_sr=data.sample_rate)
        noise_gain_db = min(data.rms_db - noise.rms_db - snr_db, self._max_gain_db)
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        noise.subsegment(start_time=start_time, end_time=start_time + data.duration)
        noise.gain_db(noise_gain_db)
        data._samples = data._samples + noise.samples


class ShiftPerturbation(Perturbation):

    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class SpeedPerturbation(Perturbation):

    def __init__(self, min_speed_rate=0.85, max_speed_rate=1.15, rng=None):
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        if speed_rate <= 0:
            raise ValueError('speed_rate should be greater than zero.')
        data._samples = librosa.effects.time_stretch(data._samples, speed_rate)


perturbation_types = {'speed': SpeedPerturbation, 'gain': GainPerturbation, 'impulse': ImpulsePerturbation, 'shift': ShiftPerturbation, 'noise': NoisePerturbation}


class AudioAugmentor(object):

    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for prob, p in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for prob, p in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                None
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


class WaveformFeaturizer(object):

    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(file_path, target_sr=self.sample_rate, int_values=self.int_values, offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None
        sample_rate = input_config.get('sample_rate', 16000)
        int_values = input_config.get('int_values', False)
        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)


def seq_collate_fn(batch, token_pad_value=0):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()
    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = 0, max_audio_len - sig_len
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = 0, max_tokens_len - tokens_i_len
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=token_pad_value)
        tokens.append(tokens_i)
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    return audio_signal, audio_lengths, tokens, tokens_lengths


class AudioToTextDataLayer(nn.Module):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (str): Dataset parameter.
            End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    def __init__(self, *, manifest_filepath, labels, batch_size, sample_rate=16000, int_values=False, bos_id=None, eos_id=None, pad_id=None, min_duration=0.1, max_duration=None, normalize_transcripts=True, trim_silence=False, load_audio=True, drop_last=False, shuffle=True, num_workers=4, placement='cpu', **kwargs):
        super().__init__()
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=None)
        dataset_params = {'manifest_filepath': manifest_filepath, 'labels': labels, 'featurizer': self._featurizer, 'max_duration': max_duration, 'min_duration': min_duration, 'normalize': normalize_transcripts, 'trim': trim_silence, 'bos_id': bos_id, 'eos_id': eos_id, 'logger': None, 'load_audio': load_audio}
        self._dataset = AudioDataset(**dataset_params)
        if placement == 'cuda':
            None
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None
        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(dataset=self._dataset, batch_size=batch_size, collate_fn=partial(seq_collate_fn, token_pad_value=pad_id), drop_last=drop_last, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=num_workers)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


class GreedyCTCDecoder(nn.Module):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False)
            return argmx


class CTCLossNM(nn.Module):
    """
    Neural Module wrapper for pytorch's ctcloss

    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
    """

    def __init__(self, *, num_classes, **kwargs):
        nn.Module.__init__(self)
        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none')

    def _loss(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length, target_length)
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*kwargs.values())


BIAS_CONFIGS = False


SCALING_MIN_VAL = 2e-09


WEIGHT_NARROW_RANGE = True


def make_quantconv1d(feat_in, feat_out, kernel_size, stride, padding, bit_width, dilation=1, group=1):
    return quant_nn.QuantConv1d(in_channels=feat_in, out_channels=feat_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=group, weight_bit_width=bit_width, weight_quant_type=QUANT_TYPE, weight_narrow_range=WEIGHT_NARROW_RANGE, weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE, weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP, weight_scaling_min_val=SCALING_MIN_VAL, bias_bit_width=bit_width, bias_quant_type=QUANT_TYPE_BIAS, bias_narrow_range=BIAS_CONFIGS, compute_output_scale=BIAS_CONFIGS, compute_output_bit_width=BIAS_CONFIGS, return_quant_tensor=False)


class MaskedConv1d(nn.Module):
    __constants__ = ['use_conv_mask', 'real_out_channels', 'heads']

    def __init__(self, in_channels, out_channels, kernel_size, scaling_per_channel, bit_width, stride=1, padding=0, dilation=1, groups=1, heads=-1, bias=False, use_mask=True):
        super(MaskedConv1d, self).__init__()
        if not (heads == -1 or groups == in_channels):
            raise ValueError('Only use heads for depthwise convolutions')
        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads
        self.conv = make_quantconv1d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, scaling_per_channel=scaling_per_channel, bit_width=bit_width)
        self.channelwise_separable = in_channels == out_channels and in_channels == groups
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) / self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens
            max_len = x.size(2)
            mask = torch.arange(max_len).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1), 0)
            lens = self.get_seq_len(lens)
        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])
        out = self.conv(x)
        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)
        return out, lens


class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        return x


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError('Only stride OR dilation may be greater than 1')
    if dilation > 1:
        return dilation * kernel_size // 2 - 1
    return kernel_size // 2


def make_jasper_activation(activation, channels, bit_width, absolute_act_val, scaling_per_channel):
    brevitas_activation = brevitas_activations[activation]
    return brevitas_activation(bit_width=bit_width, scaling_per_channel=scaling_per_channel, quant_type=QUANT_TYPE, scaling_impl_type=ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, restrict_scaling_type=ACT_RESTRICT_SCALING_TYPE, max_val=absolute_act_val, per_channel_broadcastable_shape=(1, channels, 1), scaling_stats_permute_dims=(1, 0, 2), return_quant_tensor=False)


def make_norm_scale(bit_width, absolute_act_val, scaling_per_channel):
    return quant_nn.QuantHardTanh(bit_width=bit_width, scaling_per_channel=scaling_per_channel, quant_type=QUANT_TYPE, scaling_impl_type=ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, restrict_scaling_type=ACT_RESTRICT_SCALING_TYPE, max_val=absolute_act_val, min_val=-absolute_act_val, scaling_stats_permute_dims=(1, 0, 2), return_quant_tensor=True)


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias):
    denom = torch.sqrt(bn_var + bn_eps)
    mul_factor = bn_weight / denom
    add_factor = -bn_mean * mul_factor + bn_bias
    return mul_factor, add_factor


def rename_state_dict_by_postfix(old_postfix, new_postfix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.endswith(old_postfix):
            new_key = k[:len(k) - len(old_postfix)] + new_postfix
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)


class JasperBlock(nn.Module):
    __constants__ = ['conv_mask', 'separable', 'residual_mode', 'res', 'mconv']

    def __init__(self, inplanes, planes, bit_width, absolute_act_val, activation_inner_scaling_per_output_channel, activation_other_scaling_per_output_channel, weight_scaling_per_output_channel, repeat=3, kernel_size=11, stride=1, dilation=1, padding='same', dropout=0.2, activation=None, residual=True, groups=1, separable=False, heads=-1, normalization='batch', norm_groups=1, residual_mode='add', residual_panes=[], conv_mask=False, fused_bn=False):
        super(JasperBlock, self).__init__()
        if padding != 'same':
            raise ValueError("currently only 'same' padding is supported")
        self.fused_bn = fused_bn
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.conv_module_to_merge = []
        inplanes_loop = inplanes
        conv = nn.ModuleList()
        self.norm_depthwise = nn.ModuleList()
        for _ in range(repeat - 1):
            if separable:
                self.norm_depthwise.extend([make_norm_scale(bit_width=bit_width, absolute_act_val=absolute_act_val, scaling_per_channel=activation_other_scaling_per_output_channel)])
            conv.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val, groups=groups, heads=heads, separable=separable, normalization=normalization, norm_groups=norm_groups, bit_width=bit_width, scaling_per_channel=weight_scaling_per_output_channel))
            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation, channels=planes, bit_width=bit_width, absolute_act_val=absolute_act_val, scaling_per_channel=activation_inner_scaling_per_output_channel))
            inplanes_loop = planes
        if separable:
            self.norm_depthwise.extend([make_norm_scale(bit_width=bit_width, absolute_act_val=absolute_act_val, scaling_per_channel=activation_other_scaling_per_output_channel)])
        conv.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val, groups=groups, heads=heads, separable=separable, normalization=normalization, norm_groups=norm_groups, bit_width=bit_width, scaling_per_channel=weight_scaling_per_output_channel))
        self.mconv = conv
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            res_list = nn.ModuleList()
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res_list.append(nn.ModuleList(self._get_conv_bn_layer(ip, planes, kernel_size=1, normalization=normalization, norm_groups=norm_groups, bit_width=bit_width, scaling_per_channel=weight_scaling_per_output_channel)))
            self.res = res_list
            self.quant_normalization = make_norm_scale(bit_width=bit_width, absolute_act_val=absolute_act_val, scaling_per_channel=activation_other_scaling_per_output_channel)
        else:
            self.res = None
            self.quant_normalization = None
        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation, channels=inplanes_loop, absolute_act_val=absolute_act_val, scaling_per_channel=activation_other_scaling_per_output_channel, bit_width=bit_width))

    def _get_conv(self, in_channels, out_channels, bit_width, scaling_per_channel, kernel_size=11, stride=1, dilation=1, padding=0, bias=False, groups=1, heads=-1, separable=False):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, groups=groups, heads=heads, use_mask=use_mask, scaling_per_channel=scaling_per_channel, bit_width=bit_width)
        else:
            return make_quantconv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, groups=groups, bias=bias, scaling_per_channel=scaling_per_channel, bit_width=bit_width)

    def _get_conv_bn_layer(self, in_channels, out_channels, bit_width, scaling_per_channel, kernel_size=11, stride=1, dilation=1, padding=0, bias=False, groups=1, heads=-1, separable=False, normalization='batch', norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels
        if separable:
            layers = [self._get_conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=in_channels, heads=heads, bias=bias, scaling_per_channel=scaling_per_channel, bit_width=bit_width), self._get_conv(in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, groups=groups, bias=bias, scaling_per_channel=scaling_per_channel, bit_width=bit_width)]
        else:
            layers = [self._get_conv(in_channels, out_channels, kernel_size=kernel_size, scaling_per_channel=scaling_per_channel, bit_width=bit_width, stride=stride, bias=bias, dilation=dilation, padding=padding, groups=groups)]
        if normalization == 'group':
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == 'instance':
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == 'layer':
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == 'batch':
            if self.fused_bn:
                self.conv_module_to_merge.append(layers[-1])
                layers.append(nn.Identity())
            else:
                layers.append(nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1))
        else:
            raise ValueError(f'Normalization method ({normalization}) does not match one of [batch, layer, group, instance].')
        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, channels, bit_width, absolute_act_val, scaling_per_channel, drop_prob=0.2, activation=None):
        if activation is None:
            raise Exception('Activation required')
        layers = [make_jasper_activation(activation, channels, bit_width=bit_width, absolute_act_val=absolute_act_val, scaling_per_channel=scaling_per_channel), nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_: Tuple[List[Tensor], Optional[Tensor]]):
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_
        out = xs[-1]
        count_norm = 0
        lens = lens_orig
        check_flag = False
        for i, l in enumerate(self.mconv):
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)
            if isinstance(l, (MaskedConv1d, QuantConv1d)):
                check_flag = check_flag or l.channelwise_separable
                if l.channelwise_separable:
                    out = self.norm_depthwise[count_norm](out)
                    if isinstance(out, QuantTensor):
                        out = out.value
                    count_norm += 1
        if check_flag:
            assert len(self.norm_depthwise) == count_norm
        if self.res is not None:
            out = self.quant_normalization(out)
            if self.training:
                out = out.value
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)
                res_out = self.quant_normalization(res_out)
                if self.training:
                    res_out = res_out.value
                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)
        if isinstance(out, QuantTensor):
            out = out.value
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens
        return [out], lens

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if not self.conv_mask:
            rename_state_dict_by_postfix('conv.weight', 'weight', state_dict)
        if self.fused_bn:
            self.fuse_bn(state_dict, prefix)
        super(JasperBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        extra_k = 'quant_normalization'
        is_prefix_to_fix = any([(prefix == 'encoder.' + p) for p in ['0.', '16.', '17.']])
        if is_prefix_to_fix:
            for i, k in enumerate(unexpected_keys):
                if extra_k in k:
                    del unexpected_keys[i]

    def fuse_bn(self, state_dict, prefix):
        index = 0
        flag = False
        keys_to_check = []
        keys_to_delete = []
        for k in state_dict.keys():
            if k.startswith(prefix):
                keys_to_check.append(k)
                if k.split('.')[-1] == 'running_mean':
                    flag = True
        if flag:
            for name in keys_to_check:
                prefix_long = name.split('.')[:-1]
                if name.split('.')[-1] == 'running_mean':
                    bn_prefix = '.'.join(prefix_long)
                    module_number = int(prefix_long[-1])
                    conv_name = prefix_long[:-1] + [str(module_number - 1)]
                    if self.conv_mask:
                        conv_name = conv_name + ['conv']
                    conv_name = '.'.join(conv_name)
                    conv_mod = self.conv_module_to_merge[index]
                    index = index + 1
                    bn_weight_key = '.'.join([bn_prefix, 'weight'])
                    bn_bias_key = '.'.join([bn_prefix, 'bias'])
                    bn_running_mean_key = '.'.join([bn_prefix, 'running_mean'])
                    bn_running_var_key = '.'.join([bn_prefix, 'running_var'])
                    bn_num_batches_traked_key = '.'.join([bn_prefix, 'num_batches_tracked'])
                    keys_to_delete = keys_to_delete + [bn_bias_key]
                    keys_to_delete = keys_to_delete + [bn_weight_key]
                    keys_to_delete = keys_to_delete + [bn_running_mean_key]
                    keys_to_delete = keys_to_delete + [bn_running_var_key]
                    keys_to_delete = keys_to_delete + [bn_num_batches_traked_key]
                    mul_factor, add_factor = mul_add_from_bn(bn_mean=state_dict[bn_running_mean_key], bn_var=state_dict[bn_running_var_key], bn_eps=0.001, bn_weight=state_dict[bn_weight_key], bn_bias=state_dict[bn_bias_key])
                    if isinstance(conv_mod, MaskedConv1d):
                        conv_mod = conv_mod.conv
                    conv_weight_key = conv_name + '.weight'
                    conv_bias_key = conv_name + '.bias'
                    result = state_dict[conv_weight_key] * mul_factor.view(-1, 1, 1)
                    state_dict[conv_weight_key] = result
                    if conv_mod.bias is not None and conv_bias_key in state_dict:
                        state_dict[conv_bias_key] += add_factor
                    elif conv_mod.bias is not None and not conv_bias_key in state_dict:
                        state_dict[conv_bias_key] = add_factor
                    else:
                        if torch.cuda.is_available():
                            add_factor = add_factor
                        conv_mod.bias = nn.Parameter(add_factor)
                        state_dict[conv_bias_key] = add_factor
                else:
                    state_dict[name] = state_dict[name]
        for k in list(state_dict.keys()):
            if k in keys_to_delete:
                del state_dict[k]
        assert len(self.conv_module_to_merge) == index


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, nn.Conv1d):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            raise ValueError('Unknown Initialization mode: {0}'.format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class JasperEncoder(nn.Module):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu"].
        feat_in (int): Number of channels being input to this module
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add" or "max".
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(self, *, jasper, outer_bit_width, inner_bit_width, weight_scaling_per_output_channel, absolute_act_val, activation_inner_scaling_per_output_channel, activation_other_scaling_per_output_channel, activation, feat_in, fused_bn=False, normalization_mode='batch', residual_mode='add', norm_groups=-1, conv_mask=True, frame_splicing=1, init_mode='xavier_uniform', **kwargs):
        nn.Module.__init__(self)
        feat_in = feat_in * frame_splicing
        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for it, lcfg in enumerate(jasper):
            if it == 0:
                bit_width = outer_bit_width
            else:
                bit_width = inner_bit_width
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            encoder_layers.append(JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'], kernel_size=lcfg['kernel'], stride=lcfg['stride'], dilation=lcfg['dilation'], dropout=lcfg['dropout'], residual=lcfg['residual'], groups=groups, fused_bn=fused_bn, separable=separable, heads=heads, residual_mode=residual_mode, normalization=normalization_mode, norm_groups=norm_groups, activation=activation, residual_panes=dense_res, conv_mask=conv_mask, bit_width=bit_width, absolute_act_val=absolute_act_val, activation_inner_scaling_per_output_channel=activation_inner_scaling_per_output_channel, activation_other_scaling_per_output_channel=activation_other_scaling_per_output_channel, weight_scaling_per_output_channel=weight_scaling_per_output_channel))
            feat_in = lcfg['filters']
        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]
        return s_input[-1], length


class JasperDecoderForCTC(nn.Module):
    """
    Jasper Decoder creates the final layer in Jasper that maps from the outputs
    of Jasper Encoder to the vocabulary of interest.

    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(self, *, feat_in, num_classes, bit_width, weight_scaling_per_channel, init_mode='xavier_uniform', **kwargs):
        nn.Module.__init__(self)
        self._feat_in = feat_in
        self._num_classes = num_classes + 1
        self.decoder_layers = nn.Sequential(make_quantconv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True, bit_width=bit_width, scaling_per_channel=weight_scaling_per_channel))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        return F.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)


class Quartznet(nn.Module):

    def __init__(self, preprocessing, encoder, decoder, greedyctcdecoder):
        super(Quartznet, self).__init__()
        self.preprocessing = preprocessing
        self.encoder = encoder
        self.decoder = decoder
        self.greedy_ctc_decoder = greedyctcdecoder

    def forward(self, input_tensors):
        if self.preprocessing is not None:
            audio_signal_e1, a_sig_length_e1, _, _ = input_tensors
            processed_signal_e1, p_length_e1 = self.preprocessing(input_signal=audio_signal_e1, length=a_sig_length_e1)
            encoded_e1, encoded_len_e1 = self.encoder(audio_signal=processed_signal_e1, length=p_length_e1)
        else:
            encoded_e1 = self.encoder(input_tensors)
        log_probs_e1 = self.decoder(encoder_output=encoded_e1)
        predictions_e1 = self.greedy_ctc_decoder(log_probs=log_probs_e1)
        return predictions_e1

    def restore_checkpoints(self, encoder_state_dict, decoder_state_dict):
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        None


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


MAX_WAV_VALUE = 32768.0


ACT_MAX_VAL = 1


ACT_MIN_VAL = -1


def make_hardtanh_activation(bit_width, return_quant_tensor=False):
    return quant_nn.QuantHardTanh(bit_width=bit_width, max_val=ACT_MAX_VAL, min_val=ACT_MIN_VAL, quant_type=QUANT_TYPE, scaling_impl_type=ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant_tensor)


def make_leakyRelu_activation(bit_width):
    el1 = nn.LeakyReLU()
    el2 = make_hardtanh_activation(bit_width=bit_width)
    layer = nn.Sequential(el1, el2)
    return layer


class ResStack(nn.Module):

    def __init__(self, channel, bit_width):
        super(ResStack, self).__init__()
        self.scale_norm = make_hardtanh_activation(bit_width=bit_width, return_quant_tensor=True)
        self.layers = nn.ModuleList([nn.Sequential(make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_quantconv1d(channel, channel, kernel_size=3, stride=1, padding=3 ** i, dilation=3 ** i, bit_width=bit_width)), make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_quantconv1d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, bit_width=bit_width))) for i in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = self.scale_norm(x)
            if isinstance(x, QuantTensor):
                x_unp, _, _ = x
            else:
                x_unp = x
            x_layer = self.scale_norm(layer(x_unp))
            if isinstance(x_layer, QuantTensor):
                x_layer_unp, _, _ = x_layer
            else:
                x_layer_unp = x_layer
            if self.training:
                x = x_unp + x_layer_unp
            else:
                x = x + x_layer
        if isinstance(x, QuantTensor):
            x, _, _ = x
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            nn.utils.remove_weight_norm(layer[1])
            nn.utils.remove_weight_norm(layer[3])


def make_tanh_activation(bit_width):
    return quant_nn.QuantTanh(bit_width=bit_width, quant_type=QUANT_TYPE, scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=False)


def make_transpconv1d(feat_in, feat_out, kernel_size, stride, padding, bit_width, dilation=1):
    return quant_nn.QuantConvTranspose1d(in_channels=feat_in, out_channels=feat_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, weight_bit_width=bit_width, weight_quant_type=QUANT_TYPE, weight_narrow_range=WEIGHT_NARROW_RANGE, weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE, weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP, weight_scaling_min_val=SCALING_MIN_VAL, bias_bit_width=bit_width, bias_quant_type=QUANT_TYPE_BIAS, bias_narrow_range=BIAS_CONFIGS, compute_output_scale=BIAS_CONFIGS, compute_output_bit_width=BIAS_CONFIGS, return_quant_tensor=False)


class Generator(nn.Module):

    def __init__(self, mel_channel, bit_width, last_layer_bit_width):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.generator = nn.Sequential(nn.utils.weight_norm(make_quantconv1d(mel_channel, 512, kernel_size=7, stride=1, padding=3, bit_width=bit_width)), make_leakyRelu_activation(bit_width=bit_width), nn.utils.weight_norm(make_transpconv1d(512, 256, kernel_size=16, stride=8, padding=4, bit_width=bit_width)), ResStack(256, bit_width=bit_width), make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_transpconv1d(256, 128, kernel_size=16, stride=8, padding=4, bit_width=bit_width)), ResStack(128, bit_width=bit_width), make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_transpconv1d(128, 64, kernel_size=4, stride=2, padding=1, bit_width=bit_width)), ResStack(64, bit_width=bit_width), make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_transpconv1d(64, 32, kernel_size=4, stride=2, padding=1, bit_width=bit_width)), ResStack(32, bit_width=bit_width), make_leakyRelu_activation(bit_width), nn.utils.weight_norm(make_quantconv1d(32, 1, kernel_size=7, stride=1, padding=3, bit_width=bit_width)), make_tanh_activation(bit_width=last_layer_bit_width))

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        zero = torch.full((1, self.mel_channel, 10), -11.5129)
        mel = torch.cat((mel, zero), dim=2)
        audio = self.forward(mel)
        audio = audio.squeeze()
        audio = audio[:-(hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()
        return audio


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class UnpackShape(Module):

    def forward(self, x):
        size = x.size()
        batchsize, num_channels, height, width = size
        return x


class ReshapeModule(Module):

    def forward(self, x):
        groups = 1
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        return x


class CatChunkUnrolledModule(Module):

    def forward(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        return x


class CatChunkRolledModule(Module):

    def forward(self, x: Tensor):
        x = x.chunk(2, dim=1)
        x = torch.cat(x, dim=1)
        return x


class InPlaceTorchAddModule(Module):

    def forward(self, x: Tensor):
        x.add_(x)
        return x


class InPlacePythonAddModule(Module):

    def forward(self, x: Tensor):
        x += x
        return x


class PythonAddModule(Module):

    def forward(self, x: Tensor):
        x = x + x
        return x


class TorchAddModule(Module):

    def forward(self, x: Tensor):
        x = torch.add(x, x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AudioPreprocessor,
     lambda: ([], {'win_length': 4, 'hop_length': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CatChunkRolledModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CatChunkUnrolledModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GreedyCTCDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupShuffle,
     lambda: ([], {'groups': 1, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InPlacePythonAddModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InPlaceTorchAddModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InplaceLogTwo,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiplyBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4]), torch.rand([4, 4]), torch.rand([4])], {}),
     True),
    (PythonAddModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QuantDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantMaxPool1d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (QuantMaxPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReshapeModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleBias,
     lambda: ([], {'num_features': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpecAugment,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpecCutout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpectrogramAugmentation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SqrHingeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TensorNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchAddModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnpackShape,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Xilinx_brevitas(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

