import sys
_module = sys.modules[__name__]
del sys
gen_github_actions = _module
brevitas = _module
config = _module
core = _module
bit_width = _module
function_wrapper = _module
quant = _module
restrict_val = _module
scaling = _module
stats = _module
function = _module
autograd_ops = _module
ops = _module
shape = _module
loss = _module
base_loss = _module
weighted_bit_width = _module
nn = _module
hadamard_classifier = _module
quant_accumulator = _module
quant_activation = _module
quant_avg_pool = _module
quant_bn = _module
quant_conv = _module
quant_conv1d = _module
quant_convtranspose1d = _module
quant_layer = _module
quant_linear = _module
quant_scale_bias = _module
proxy = _module
parameter_quant = _module
quant_proxy = _module
runtime_quant = _module
quant_tensor = _module
utils = _module
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
LFC = _module
SFC = _module
TFC = _module
models = _module
common = _module
losses = _module
tensor_norm = _module
trainer = _module
imagenet_classification = _module
imagenet_val = _module
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
conf = _module
noxfile = _module
setup = _module
generate_quant_input = _module
test_act_scaling = _module
test_conv1d = _module
test_import = _module
test_ops = _module
test_quant = _module
test_transposed_conv1d = _module
test_pretrained_accuracy = _module
conftest = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from typing import Optional


from enum import auto


import torch


from torch import Tensor


from torch.nn import Parameter


from typing import Tuple


from typing import Union


from torch.nn import Module


from typing import Callable


import math


from torch.nn import Sequential


from typing import List


from torch import nn


from abc import ABCMeta


from abc import abstractmethod


from functools import reduce


import torch.nn as nn


from torch.nn import AvgPool2d


import re


from torch.nn import Conv2d


from torch.nn import functional as F


from torch.nn.functional import conv2d


from torch.nn.parameter import Parameter


from torch.nn import Conv1d


from torch.nn.functional import conv1d


from torch.nn import ConvTranspose1d


from torch.nn.functional import conv_transpose1d


from torch.nn import Linear


from torch.nn.functional import linear


from functools import partial


from typing import Dict


from torch.nn import ModuleList


from torch.nn import BatchNorm2d


from torch.nn import MaxPool2d


from torch.nn import BatchNorm1d


from torch.nn import Dropout


from torch.autograd import Function


import torch.nn.init as init


import random


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torch.utils.data import Dataset


import torch.nn.functional as F


import numpy as np


from torch.autograd import Variable


from scipy.signal import get_window


class IdentityBitWidth(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tensor:
        return x


class ZeroLsbTruncBitWidth(torch.jit.ScriptModule):

    def forward(self, input_bit_width: Tensor, zero_hw_sentinel: Tensor):
        return zero_hw_sentinel


MIN_INT_BIT_WIDTH = 2


_global_config['IGNORE_MISSING_KEYS'] = 4


NON_ZERO_EPSILON = 1e-06


REMOVE_ZERO_BIT_WIDTH = 0.1


class RemoveBitwidthParameter(torch.jit.ScriptModule):
    __constants__ = ['min_overall_bit_width', 'non_zero_epsilon',
        'override_pretrained', 'remove_at_least_init_val']

    def __init__(self, bit_width_to_remove, remove_at_least_init_val,
        restrict_bit_width_impl, override_pretrained):
        super(RemoveBitwidthParameter, self).__init__()
        if bit_width_to_remove < 0:
            raise Exception(
                'Bit width to clamp has to be at least 0, instead is {}.'.
                format(bit_width_to_remove))
        elif bit_width_to_remove == 0:
            bit_width_coeff_init = 1 / REMOVE_ZERO_BIT_WIDTH
        else:
            bit_width_coeff_init = 1 / bit_width_to_remove
        self.bit_width_coeff = Parameter(torch.tensor(bit_width_coeff_init))
        self.restrict_bit_width_impl = restrict_bit_width_impl
        self.non_zero_epsilon = NON_ZERO_EPSILON
        self.override_pretrained = override_pretrained
        self.remove_at_least_init_val = remove_at_least_init_val

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel) ->Tensor:
        bit_width_to_remove = 1.0 / (self.non_zero_epsilon + torch.abs(self
            .bit_width_coeff))
        bit_width_to_remove = self.restrict_bit_width_impl(bit_width_to_remove)
        return bit_width_to_remove

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        bit_width_coeff_key = prefix + 'bit_width_coeff'
        if self.override_pretrained and bit_width_coeff_key in state_dict:
            del state_dict[bit_width_coeff_key]
        super(RemoveBitwidthParameter, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)
        if config.IGNORE_MISSING_KEYS and bit_width_coeff_key in missing_keys:
            missing_keys.remove(bit_width_coeff_key)


@torch.jit.script
def tensor_clamp(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) ->torch.Tensor:
    """

    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the clamp operation
    min_val : Tensor
        Tensor containing the minimum values for the clamp operation. Must have the same shape of `x`
    max_val : Tensor
        Tensor containing the maximum values for the clamp operation. Must have the same shape of `x`

    Returns
    -------
    Tensor
        Tensor for which every element of `x` is clamped between the corresponding minimum and maximum values.
    """
    out = torch.where(x > max_val, max_val, x)
    out = torch.where(out < min_val, min_val, out)
    return out


class IdentityQuant(torch.jit.ScriptModule):
    """ Placeholder Class that returns the input without performing any operation. The scale and bit_width output
    arguments are set to zero_hw_sentinel (0).
    """

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tuple[Tensor,
        Tensor, Tensor]:
        return x, zero_hw_sentinel, zero_hw_sentinel


class BinaryQuant(torch.jit.ScriptModule):
    """ Class that implement the binary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    The scale factor is determined internally through the scaling_impl module.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor

    Attributes
    ----------
    scaling_impl: Module
       Module that determines the value of the scale factor
    bit_width: Int
        For binary quantization, the bit_width is constant and fixed to 1

    Methods
    -------
    forward(x, zero_hw_sentinel)
        Perform the binary quantization using :func:`~brevitas.function.ops_ste.binary_sign_ste`. After that, the
        result is converted to floating point through the scale factor.
        The scale factor is determined by the attribute `scaling_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.

    """
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(BinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = 1

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tuple[Tensor,
        Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        y = binary_sign_ste(x) * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class ClampedBinaryQuant(torch.jit.ScriptModule):
    """ Class that implement the binary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    Before performing the binarization, the input tensor is clamped in the range of admissible values, determined by the
    scale factor.
    The scale factor is determined internally through the scaling_impl module.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor

    Attributes
    ----------
    scaling_impl : Module
       Module that determines the value of the scale factor
    bit_width : Int
        For binary quantization, the bit_width is constant and fixed to 1

    Methods
    -------
    forward(x, zero_hw_sentinel)
        Perform the binary quantization using :func:`~brevitas.function.ops_ste.binary_sign_ste`. After that, the
        result is converted to floating point through the scale factor.
        The scale factor is determined by the attribute `scaling_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.

    """
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(ClampedBinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = 1

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tuple[Tensor,
        Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        y = tensor_clamp(x, -scale, scale)
        y = binary_sign_ste(y) * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class TernaryQuant(torch.jit.ScriptModule):
    """ Class that implement the ternary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    The scale factor is determined internally through the scaling_impl module. The threshold is a user-defined value in
    the range (0,1).

    The quantization is performed in such a way that all input values in the range
    (-scale*threshold, scale*threshold) are quantized to 0. Values greater than the upper bound are quantized to 'scale'
    . Values lower than the lower bound are quantized to '-scale'.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor
    threshold: Float
        User-defined value that determines, together with the scale factor, the range of values that are quantized to 0.

    Attributes
    ----------
    scaling_impl : Module
       Module that determines the value of the scale factor
    bit_width : Int
        For binary quantization, the bit_width is constant and fixed to 2
    threshold: Float
        User-defined value that determines, together with the scale factor, the range of values that are quantized to 0.

    Methods
    -------
    forward(x, zero_hw_sentinel)
        Perform the ternary quantization using :func:`~brevitas.function.ops_ste.ternary_sign_ste`. After that, the
        result is converted to floating point through the scale factor.
        The scale factor is determined by the attribute `scaling_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """
    __constants__ = ['threshold', 'bit_width']

    def __init__(self, scaling_impl: Module, threshold: float):
        super(TernaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.threshold = threshold
        self.bit_width = 2

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tuple[Tensor,
        Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        mask = x.abs().ge(self.threshold * scale)
        y = mask.float() * ternary_sign_ste(x)
        y = y * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class PrescaledRestrictIntQuantWithInputBitWidth(torch.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width.
    Scale is determined externally, int_scale is set to 1, while bit_width is determined internally through
    msb_clamp_bit_width_impl.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant : Module
       Module that performs the actual quantization
    msb_clamp_bit_width_impl : Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x, scale, input_bit_width, zero_hw_sentinel)
        After determining internally the bit_width value, it calls IntQuant to perform the quantization of the input

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Scale factor that regulates the conversion between integer and floating point version of the input tensor
        input_bit_width
            Bit_width that, going in `msb_clamp_bit_with`, is used to determine the bit_width for the quantization
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """

    def __init__(self, narrow_range: bool, signed: bool, tensor_clamp_impl:
        Module, msb_clamp_bit_width_impl: Module, float_to_int_impl: Module):
        super(PrescaledRestrictIntQuantWithInputBitWidth, self).__init__()
        self.int_quant = IntQuant(signed=signed, narrow_range=narrow_range,
            tensor_clamp_impl=tensor_clamp_impl, float_to_int_impl=
            float_to_int_impl)
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl

    @torch.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, input_bit_width: Tensor,
        zero_hw_sentinel: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(input_bit_width,
            zero_hw_sentinel)
        y = self.int_quant(scale, zero_hw_sentinel + 1, msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class PrescaledRestrictIntQuant(torch.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width.
    Scale is determined externally, int_scale is set to 1, while bit_width is determined internally through
    msb_clamp_bit_width_impl.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant: Module
       Module that performs the actual quantization
    msb_clamp_bit_width_impl: Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x, scale, zero_hw_sentinel)
        After determining internally the bit_width value, it calls IntQuant to perform the quantization of the input

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Scale factor that regulates the conversion between integer and floating point version of the input tensor
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """

    def __init__(self, narrow_range: bool, signed: bool, tensor_clamp_impl:
        Module, msb_clamp_bit_width_impl: Module, float_to_int_impl: Module):
        super(PrescaledRestrictIntQuant, self).__init__()
        self.int_quant = IntQuant(signed=signed, narrow_range=narrow_range,
            tensor_clamp_impl=tensor_clamp_impl, float_to_int_impl=
            float_to_int_impl)
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl

    @torch.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, zero_hw_sentinel: Tensor
        ) ->Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(zero_hw_sentinel)
        y = self.int_quant(scale, zero_hw_sentinel + 1, msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class IdentityPrescaledIntQuant(torch.jit.ScriptModule):
    """ Placeholder Class that returns the input without performing any operation.
    """

    @torch.jit.script_method
    def forward(self, x, input_scale, input_bit_width, zero_hw_sentinel
        ) ->Tuple[Tensor, Tensor, Tensor]:
        return x, input_scale, input_bit_width


class RescalingIntQuant(torch.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    The `runtime` boolean is required to determine how to compute the scale factor.
    The `int_scaling_impl` module is required to  determine int_scale.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width. All values are determined internally.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant: Module
       Module that performs the actual quantization
    runtime: Bool
        Value that determines how the scaling factor is computed in `scaling_impl`
    scaling_impl: Module
        Module that is responsible for the computation of the scale factor
    int_scaling_impl: Module
        Module that is responsible for the computation of the int_scale factor
    msb_clamp_bit_width_impl: Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x, zero_hw_sentinel)
        After determining internally the bit_width value, the scale factor, and the int_scale factor
        the method calls IntQuant to perform the quantization of the input.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        zero_hw_sentinel: Tensor
            Constant buffer required to move stateless (as in, not part of the model's state_dict) constant values
            to the appropriate device and converting them to Tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """
    __constants__ = ['runtime']

    def __init__(self, narrow_range: bool, runtime: bool, signed: bool,
        scaling_impl: Module, int_scaling_impl: Module, tensor_clamp_impl:
        Module, msb_clamp_bit_width_impl: Module, float_to_int_impl: Module):
        super(RescalingIntQuant, self).__init__()
        self.int_quant = IntQuant(signed=signed, narrow_range=narrow_range,
            tensor_clamp_impl=tensor_clamp_impl, float_to_int_impl=
            float_to_int_impl)
        self.runtime = runtime
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl

    @staticmethod
    def scaling_init_from_min_max(min_val_init: Union[int, float],
        max_val_init: Union[int, float]) ->torch.Tensor:
        """ Static Method that is used in the step of initializing the scale factor

        Parameters
        ----------
        min_val_init: Tensor
            Minimum value used for initialization
        max_val_init: Tensor
            Maximum value used for initialization

        Returns
        -------
        Tensor
            The largest number, in absolute value, between `max_val_init` and `min_val_init`
        """
        scaling_init = max(abs(float(min_val_init)), abs(float(max_val_init)))
        return torch.tensor(scaling_init)

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) ->Tuple[Tensor,
        Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(zero_hw_sentinel)
        if self.runtime:
            scale = self.scaling_impl(x)
        else:
            scale = self.scaling_impl(zero_hw_sentinel)
        int_scale = self.int_scaling_impl(msb_clamp_bit_width)
        y = self.int_quant(scale, int_scale, msb_clamp_bit_width, x)
        output_bit_width = msb_clamp_bit_width
        output_scale = scale / int_scale
        return y, output_scale, output_bit_width


class IntQuant(torch.jit.ScriptModule):
    """ Class that implement the quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor (i.e. scale/int_scale).

    All values required for the quantization are determined externally.


    Parameters
    ----------
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.

    Methods
    -------
    to_int(scale, int_scale_msb_clamp_bit_width, x)
        Perform the conversion to integer of the input tensor.
        After diving by the scale factor (i.e. scale/int_scale), the input tensor is clamped in the range of admissible
        integer values, and then converted to integer according to the strategy defined by `float_to_int_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Floating point component of the scale factor
        int_scale: Tensor
            Integer component of the scale factor
        msb_clamp_bit_width: Tensor
            Bit_width to be used for the conversion to integer

    forward(scale, int_scale, msb_clamp_bit_width, x)
        Perform the quantization of the input tensor. The value is first converted to its integer representation and
        quantized, then converted to its floating representation multiplying it by the scale factor
        (i.e. scale/scale_int)

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Floating point component of the scale factor
        int_scale: Tensor
            Integer component of the scale factor
        msb_clamp_bit_width: Tensor
            Bit_width to be used for the conversion to integer

        Returns
        -------
        Tensor
            The quantized tensor after its conversion to floating point

    min_int(bit_width)
        Determines the minimum integer representable according to the values of `signed`, `narrow_range`, and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the minimum integer representable

        Returns
        -------
        Tensor
            The minimum integer representable

    max_int(bit_width)
        Determines the maximum signed integer representable according to the values of `signed`, `narrow_range`, and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the maximum integer representable

        Returns
        -------
        Tensor
            The maximum integer representable

    max_uint(bit_width)
        Determines the maximum unsigned integer representable according to the values of `narrow_range` and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the maximum integer representable

        Returns
        -------
        Tensor
            The maximum integer representable
    """
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, narrow_range: bool, signed: bool, float_to_int_impl:
        Module, tensor_clamp_impl: Module):
        super(IntQuant, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.signed = signed
        self.narrow_range = narrow_range

    def to_int(self, scale: Tensor, int_scale: Tensor, msb_clamp_bit_width:
        Tensor, x: Tensor) ->Tensor:
        y = x / scale
        y = y * int_scale
        min_int_val = self.min_int(msb_clamp_bit_width)
        max_int_val = self.max_int(msb_clamp_bit_width)
        y = self.tensor_clamp_impl(y, min_val=min_int_val, max_val=max_int_val)
        y = self.float_to_int_impl(y)
        return y

    @torch.jit.script_method
    def min_int(self, bit_width):
        return min_int(self.signed, self.narrow_range, bit_width)

    @torch.jit.script_method
    def max_int(self, bit_width):
        return max_int(self.signed, bit_width)

    @torch.jit.script_method
    def max_uint(self, bit_width):
        return max_uint(self.narrow_range, bit_width)

    @torch.jit.script_method
    def forward(self, scale: Tensor, int_scale: Tensor, msb_clamp_bit_width:
        Tensor, x: Tensor) ->Tensor:
        y_int = self.to_int(scale, int_scale, msb_clamp_bit_width, x)
        y = y_int / int_scale
        y = y * scale
        return y


class CeilSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(CeilSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return ceil_ste(x)


class ClampMin(torch.jit.ScriptModule):
    __constants__ = ['min_val']

    def __init__(self, min_val: float) ->None:
        super(ClampMin, self).__init__()
        self.min_val = min_val

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x.clamp_min(self.min_val)


class FloorSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(FloorSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return floor_ste(x)


@torch.jit.script
def identity(x: torch.Tensor) ->torch.Tensor:
    """ Identity function

    Parameters
    ----------
    x : Tensor
        Input Tensor

    Returns
    -------
    Tensor
        Unaltered input tensor

    """
    return x


class Identity(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(Identity, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return identity(x)


class LogTwo(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(LogTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.log2(x)


class PowerOfTwo(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(PowerOfTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return 2.0 ** x


class RoundSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(RoundSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return round_ste(x)


class AffineRescaling(torch.jit.ScriptModule):

    def __init__(self, affine_shape):
        super(AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(affine_shape))
        self.affine_bias = Parameter(torch.zeros(affine_shape))

    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias
        out = torch.abs(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(AffineRescaling, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)
        affine_weight_key = prefix + 'affine_weight'
        affine_bias_key = prefix + 'affine_bias'
        if config.IGNORE_MISSING_KEYS and affine_weight_key in missing_keys:
            missing_keys.remove(affine_weight_key)
        if config.IGNORE_MISSING_KEYS and affine_bias_key in missing_keys:
            missing_keys.remove(affine_bias_key)


SCALING_SCALAR_SHAPE = ()


@torch.jit.script
def over_batch_over_output_channels(x):
    return x.shape[0], x.shape[1], -1


class OverBatchOverOutputChannelView(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(OverBatchOverOutputChannelView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_batch_over_output_channels(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


@torch.jit.script
def over_batch_over_tensor(x):
    return x.shape[0], -1


class OverBatchOverTensorView(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(OverBatchOverTensorView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_batch_over_tensor(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


@torch.jit.script
def over_output_channels(x):
    return x.shape[0], -1


class OverOutputChannelView(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(OverOutputChannelView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_output_channels(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


@torch.jit.script
def over_tensor(x):
    return -1


class OverTensorView(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(OverTensorView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_tensor(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


class StatsInputViewShapeImpl(object):
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView


@torch.jit.script
def min_int(signed: bool, narrow_range: bool, bit_width: torch.Tensor
    ) ->torch.Tensor:
    """ Compute the minimum integer representable

    The minimum integer representable depends on the number of bits, whether the negative numbers are included
    in the representation, and whether the narrow range setting is used.
    For positive-only number, the minimum value will always be zero.
    If the sign and narrow range flags are both set, then the representation will be such that there is symmetry
    between positive and negative values.
    For example, for 3 bit representation, with sign and narrow range, the
    values representable are in the range [-3, 3].
    If the narrow range is not enabled, then the possible values will be in the range [-4, 3].

    Parameters
    ----------
    signed : Bool
        Flag that indicates whether negative numbers must be included or not
    narrow_range : Bool
        Flag that indicates whether the narrow range setting is enabled or not
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Minimum integer that can be represented according to the input parameters

    """
    if signed and narrow_range:
        value = -2 ** (bit_width - 1) + 1
    elif signed and not narrow_range:
        value = -2 ** (bit_width - 1)
    else:
        value = 0 * bit_width
    return value


class SignedFpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, narrow_range):
        super(SignedFpIntScale, self).__init__()
        self.signed = True
        self.narrow_range = narrow_range

    @torch.jit.script_method
    def forward(self, bit_width):
        return -min_int(self.signed, self.narrow_range, bit_width)


@torch.jit.script
def max_int(signed: bool, bit_width: torch.Tensor) ->torch.Tensor:
    """ Compute the maximum integer representable

    The maximum integer representable depends on the number of bits, and whether the negative numbers are included
    in the representation. If so, one bit is lost in the computation of the maximum value.

    Parameters
    ----------
    signed : Bool
        Flag that indicates whether negative numbers must be included or not
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Maximum integer that can be represented according to the input parameters

    """
    if signed:
        value = 2 ** (bit_width - 1) - 1
    else:
        value = 2 ** bit_width - 1
    return value


class UnsignedFpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self):
        super(UnsignedFpIntScale, self).__init__()
        self.signed = False

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width)


class PowerOfTwoIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed):
        super(PowerOfTwoIntScale, self).__init__()
        self.signed = signed

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width) + 1


class _ViewParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape']

    def __init__(self, parameter, view_shape_impl):
        super(_ViewParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl().shape(parameter)

    @torch.jit.script_method
    def forward(self):
        return self.parameter.view(self.shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(_ViewParameterWrapper, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewParameterWrapper, self).state_dict(destination
            , prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class _ViewCatParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape', 'cat_dim']

    def __init__(self, parameter, view_shape_impl, cat_dim):
        super(_ViewCatParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl().shape(parameter)
        self.cat_dim = cat_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.cat([self.parameter.view(self.shape), x], dim=self.cat_dim
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(_ViewCatParameterWrapper, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewCatParameterWrapper, self).state_dict(
            destination, prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class AbsMax(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) ->None:
        super(AbsMax, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.reduce_dim is None:
            return torch.max(torch.abs(x))
        else:
            return torch.max(torch.abs(x), dim=self.reduce_dim)[0]


class AbsMaxAve(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) ->None:
        super(AbsMaxAve, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.mean(torch.max(torch.abs(x), dim=self.reduce_dim)[0])


class AbsAve(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) ->None:
        super(AbsAve, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.reduce_dim is None:
            return torch.mean(torch.abs(x))
        else:
            return torch.mean(torch.abs(x), dim=self.reduce_dim)


STD_DEV_EPSILON = 1e-08


class MeanSigmaStd(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim', 'output_shape', 'std_dev_epsilon',
        'const_sigma']

    def __init__(self, reduce_dim, const_sigma, learned_sigma, output_shape
        ) ->None:
        super(MeanSigmaStd, self).__init__()
        self.reduce_dim = reduce_dim
        self.const_sigma = const_sigma
        self.learned_sigma = learned_sigma
        self.output_shape = output_shape
        self.std_dev_epsilon = STD_DEV_EPSILON

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        abs_val = torch.abs(x)
        if self.reduce_dim is None:
            mean_val = torch.mean(abs_val)
            std_val = torch.sqrt(torch.var(abs_val) + self.std_dev_epsilon)
        else:
            mean_val = torch.mean(torch.abs(x), dim=self.reduce_dim)
            mean_val = mean_val.view(self.output_shape)
            std_val = torch.sqrt(torch.var(abs_val, dim=self.reduce_dim) +
                self.std_dev_epsilon)
            std_val = std_val.view(self.output_shape)
        if self.const_sigma is not None:
            return mean_val + self.const_sigma * std_val
        else:
            return mean_val + self.learned_sigma * std_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(MeanSigmaStd, self)._load_from_state_dict(state_dict, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        sigma_key = prefix + 'learned_sigma'
        if config.IGNORE_MISSING_KEYS and sigma_key in missing_keys:
            missing_keys.remove(sigma_key)


@torch.jit.script
def max_uint(narrow_range: bool, bit_width: torch.Tensor) ->torch.Tensor:
    """ Compute the maximum unsigned integer representable

    The maximum unsigned integer representable depends on the number of bits, and whether the narrow range setting
    is used. If so, the maximum value represented is decreased by one unit.

    Parameters
    ----------
    narrow_range : Bool
        Flag that indicates whether to decrease the possible maximum value represented
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Maximum unsigned integer that can be represented according to the input parameters

    """
    if narrow_range:
        value = 2 ** bit_width - 2
    else:
        value = 2 ** bit_width - 1
    return value


def pack_quant_tensor(tensor, scale, bit_width):
    return QuantTensor._make([tensor, scale, bit_width])


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, compute_output_scale, compute_output_bit_width,
        return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self, output, output_scale, output_bit_width):
        if self.return_quant_tensor:
            return QuantTensor(tensor=output, scale=output_scale, bit_width
                =output_bit_width)
        else:
            return output


class HadamardClassifier(QuantLayer, nn.Module):

    def __init__(self, in_channels, out_channels, fixed_scale=False,
        compute_output_scale: bool=False, compute_output_bit_width: bool=
        False, return_quant_tensor: bool=False):
        QuantLayer.__init__(self, compute_output_scale=compute_output_scale,
            compute_output_bit_width=compute_output_bit_width,
            return_quant_tensor=return_quant_tensor)
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

    def forward(self, x):
        output_scale = None
        output_bit_width = None
        x, input_scale, input_bit_width = self.unpack_input(x)
        norm = x.norm(p='fro', keepdim=True) + self.eps
        x = x / norm
        out = -self.scale * nn.functional.linear(x, self.proj[:self.
            out_channels, :self.in_channels])
        if self.compute_output_scale:
            output_scale = input_scale * self.scale / norm
        if self.compute_output_bit_width:
            output_bit_width = self.max_output_bit_width(input_bit_width)
        return self.pack_output(out, output_scale, output_bit_width)

    def max_output_bit_width(self, input_bit_width):
        max_input_val = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_output_val = max_input_val * self.in_channels
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(HadamardClassifier, self).state_dict(destination,
            prefix, keep_vars)
        del state_dict[prefix + 'proj']
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(HadamardClassifier, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)
        proj_key = prefix + 'proj'
        if proj_key in missing_keys:
            missing_keys.remove(proj_key)


class QuantAccumulator(QuantLayer, Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        QuantLayer.__init__(self, compute_output_scale=True,
            compute_output_bit_width=True, return_quant_tensor=True)
        Module.__init__(self)

    @property
    def acc_quant_proxy(self):
        return self._act_quant_proxy

    @acc_quant_proxy.setter
    def acc_quant_proxy(self, act_quant_proxy):
        self._acc_quant_proxy = act_quant_proxy

    def forward(self, input):
        tensor, input_scale, input_bit_width = self.unpack_input(input)
        output, output_scale, output_bit_width = self.acc_quant_proxy(tensor,
            input_scale, input_bit_width)
        return self.pack_output(output, output_scale, output_bit_width)


class QuantActivation(QuantLayer, Module):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        QuantLayer.__init__(self, compute_output_scale=True,
            compute_output_bit_width=True, return_quant_tensor=
            return_quant_tensor)
        Module.__init__(self)

    @property
    def act_quant_proxy(self):
        return self._act_quant_proxy

    @act_quant_proxy.setter
    def act_quant_proxy(self, act_quant_proxy):
        self._act_quant_proxy = act_quant_proxy

    def quant_act_scale(self):
        if isinstance(self.act_quant_proxy.fused_activation_quant_proxy.
            tensor_quant, IdentityQuant):
            raise Exception(
                "Can't generate scaling factor without quantization enabled")
        zero_hw_sentinel = self.act_quant_proxy.zero_hw_sentinel
        scaling_impl = (self.act_quant_proxy.fused_activation_quant_proxy.
            tensor_quant.scaling_impl)
        current_status = scaling_impl.training
        scaling_impl.eval()
        _, out, _ = self.act_quant_proxy(zero_hw_sentinel)
        scaling_impl.train(current_status)
        return out

    def forward(self, input):
        tensor, _, _ = self.unpack_input(input)
        output, output_scale, output_bit_width = self.act_quant_proxy(tensor)
        return self.pack_output(output, output_scale, output_bit_width)


class TensorClamp(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(TensorClamp, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val:
        torch.Tensor):
        return tensor_clamp(x, min_val=min_val, max_val=max_val)


OVER_BATCH_OVER_CHANNELS_4D_SHAPE = 1, -1, 1, 1


ZERO_HW_SENTINEL_NAME = 'zero_hw_sentinel'


SCALING_MIN_VAL = 2e-09


class TensorClampSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(TensorClampSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val:
        torch.Tensor):
        return tensor_clamp_ste(x, min_val, max_val)


_global_config['REINIT_WEIGHT_QUANT_ON_LOAD'] = 4


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias, affine_only):
    mul_factor = bn_weight
    add_factor = bn_bias * torch.sqrt(bn_var + bn_eps)
    add_factor = add_factor - bn_mean * (bn_weight - 1.0)
    if not affine_only:
        mul_factor = mul_factor / torch.sqrt(bn_var + bn_eps)
        add_factor = add_factor - bn_mean
        add_factor = add_factor / torch.sqrt(bn_var + bn_eps)
    return mul_factor, add_factor


class ScaleBias(nn.Module):

    def __init__(self, num_features):
        super(ScaleBias, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.weight + self.bias


class WeightReg(nn.Module):

    def __init__(self):
        super(WeightReg, self).__init__()
        pass

    def forward(self, weight):
        return weight + 0


ZERO_HW_SENTINEL_VALUE = 0.0


class QuantProxy(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(QuantProxy, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(
            ZERO_HW_SENTINEL_VALUE))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        super(QuantProxy, self)._load_from_state_dict(state_dict, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
        if zero_hw_sentinel_key in unexpected_keys:
            unexpected_keys.remove(zero_hw_sentinel_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(QuantProxy, self).state_dict(destination,
            prefix, keep_vars)
        del output_dict[prefix + ZERO_HW_SENTINEL_NAME]
        return output_dict


class FusedActivationQuantProxy(torch.jit.ScriptModule):

    def __init__(self, activation_impl, tensor_quant):
        super(FusedActivationQuantProxy, self).__init__()
        self.activation_impl = activation_impl
        self.tensor_quant = tensor_quant

    @torch.jit.script_method
    def forward(self, x, zero_hw_sentinel):
        x = self.activation_impl(x)
        x, output_scale, output_bit_width = self.tensor_quant(x,
            zero_hw_sentinel)
        return x, output_scale, output_bit_width


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


CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256,
    False), (256, False)]


INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]


INTERMEDIATE_FC_PER_OUT_CH_SCALING = True


LAST_FC_IN_FEATURES = 512


LAST_FC_PER_OUT_CH_SCALING = False


class ConstScalarClamp(torch.jit.ScriptModule):
    __constants__ = ['min_val', 'max_val']

    def __init__(self, min_val, max_val) ->None:
        super(ConstScalarClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


ACT_PER_OUT_CH_SCALING = False


HARD_TANH_MAX = 1.0


HARD_TANH_MIN = -1.0


NARROW_RANGE_ENABLED = True


def get_act_quant(act_bit_width, act_quant_type):
    return QuantHardTanh(quant_type=act_quant_type, bit_width=act_bit_width,
        bit_width_impl_type=BIT_WIDTH_IMPL_TYPE, min_val=HARD_TANH_MIN,
        max_val=HARD_TANH_MAX, scaling_impl_type=ACT_SCALING_IMPL_TYPE,
        restrict_scaling_type=SCALING_VALUE_TYPE, scaling_per_channel=
        ACT_PER_OUT_CH_SCALING, narrow_range=NARROW_RANGE_ENABLED)


BIAS_ENABLED = False


CONV_PER_OUT_CH_SCALING = False


KERNEL_SIZE = 3


WEIGHT_SCALING_CONST = 1.0


def get_quant_conv2d(in_ch, out_ch, bit_width, quant_type):
    return QuantConv2d(in_channels=in_ch, kernel_size=KERNEL_SIZE,
        out_channels=out_ch, weight_quant_type=quant_type, weight_bit_width
        =bit_width, weight_narrow_range=NARROW_RANGE_ENABLED,
        weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
        weight_scaling_const=WEIGHT_SCALING_CONST,
        weight_scaling_per_output_channel=CONV_PER_OUT_CH_SCALING,
        weight_restrict_scaling_type=SCALING_VALUE_TYPE,
        weight_bit_width_impl_type=BIT_WIDTH_IMPL_TYPE, bias=BIAS_ENABLED)


def get_quant_linear(in_features, out_features, per_out_ch_scaling,
    bit_width, quant_type):
    return QuantLinear(bias=BIAS_ENABLED, in_features=in_features,
        out_features=out_features, weight_quant_type=quant_type,
        weight_bit_width=bit_width, weight_scaling_const=
        WEIGHT_SCALING_CONST, weight_bit_width_impl_type=
        BIT_WIDTH_IMPL_TYPE, weight_scaling_per_output_channel=
        per_out_ch_scaling, weight_scaling_impl_type=
        WEIGHT_SCALING_IMPL_TYPE, weight_narrow_range=NARROW_RANGE_ENABLED)


def get_quant_type(bit_width):
    if bit_width is None:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    else:
        return QuantType.INT


class CNV(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width
        =None, in_bit_width=None, in_ch=3):
        super(CNV, self).__init__()
        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        max_in_val = 1 - 2 ** -7
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(QuantHardTanh(bit_width=in_bit_width,
            quant_type=in_quant_type, max_val=max_in_val,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_impl_type=ScalingImplType.CONST))
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(get_quant_conv2d(in_ch=in_ch, out_ch=
                out_ch, bit_width=weight_bit_width, quant_type=
                weight_quant_type))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=0.0001))
            self.conv_features.append(get_act_quant(act_bit_width,
                act_quant_type))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(get_quant_linear(in_features=
                in_features, out_features=out_features, per_out_ch_scaling=
                INTERMEDIATE_FC_PER_OUT_CH_SCALING, bit_width=
                weight_bit_width, quant_type=weight_quant_type))
            self.linear_features.append(BatchNorm1d(out_features, eps=0.0001))
            self.linear_features.append(get_act_quant(act_bit_width,
                act_quant_type))
        self.linear_features.append(get_quant_linear(in_features=
            LAST_FC_IN_FEATURES, out_features=num_classes,
            per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING, bit_width=
            weight_bit_width, quant_type=weight_quant_type))
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


FC_OUT_FEATURES = [64, 64, 64]


HIDDEN_DROPOUT = 0.2


IN_DROPOUT = 0.2


class LFC(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width
        =None, in_bit_width=None, in_ch=1, in_features=(28, 28)):
        super(LFC, self).__init__()
        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        self.features = ModuleList()
        self.features.append(get_act_quant(in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(get_quant_linear(in_features=in_features,
                out_features=out_features, per_out_ch_scaling=
                INTERMEDIATE_FC_PER_OUT_CH_SCALING, bit_width=
                weight_bit_width, quant_type=weight_quant_type))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(get_act_quant(act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.features.append(get_quant_linear(in_features=in_features,
            out_features=num_classes, per_out_ch_scaling=
            LAST_FC_PER_OUT_CH_SCALING, bit_width=weight_bit_width,
            quant_type=weight_quant_type))
        self.features.append(BatchNorm1d(num_features=num_classes))
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


class SFC(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width
        =None, in_bit_width=None, in_ch=1, in_features=(28, 28)):
        super(SFC, self).__init__()
        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        self.features = ModuleList()
        self.features.append(get_act_quant(in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(get_quant_linear(in_features=in_features,
                out_features=out_features, per_out_ch_scaling=
                INTERMEDIATE_FC_PER_OUT_CH_SCALING, bit_width=
                weight_bit_width, quant_type=weight_quant_type))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(get_act_quant(act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.features.append(get_quant_linear(in_features=in_features,
            out_features=num_classes, per_out_ch_scaling=
            LAST_FC_PER_OUT_CH_SCALING, bit_width=weight_bit_width,
            quant_type=weight_quant_type))
        self.features.append(BatchNorm1d(num_features=num_classes))
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


class TFC(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width
        =None, in_bit_width=None, in_ch=1, in_features=(28, 28)):
        super(TFC, self).__init__()
        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        self.features = ModuleList()
        self.features.append(get_act_quant(in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(get_quant_linear(in_features=in_features,
                out_features=out_features, per_out_ch_scaling=
                INTERMEDIATE_FC_PER_OUT_CH_SCALING, bit_width=
                weight_bit_width, quant_type=weight_quant_type))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(get_act_quant(act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.features.append(get_quant_linear(in_features=in_features,
            out_features=num_classes, per_out_ch_scaling=
            LAST_FC_PER_OUT_CH_SCALING, bit_width=weight_bit_width,
            quant_type=weight_quant_type))
        self.features.append(BatchNorm1d(num_features=num_classes))
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
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.0).mul_(
            output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


class SqrHingeLoss(nn.Module):

    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)


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
            self.running_mean = (1 - self.momentum
                ) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return (x - self.running_mean) / (self.running_var + self.eps).pow(
                0.5) * self.weight + self.bias


class DwsConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, bit_width,
        pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(in_channels=in_channels, out_channels=
            in_channels, groups=in_channels, kernel_size=3, padding=1,
            stride=stride, weight_bit_width=bit_width, act_bit_width=bit_width)
        self.pw_conv = ConvBlock(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, padding=0, weight_bit_width=
            bit_width, act_bit_width=bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


ENABLE_BIAS_QUANT = False


WEIGHT_NARROW_RANGE = True


WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True


ACT_MAX_VAL = 1


ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None


ACT_RETURN_QUANT_TENSOR = False


ACT_SCALING_PER_CHANNEL = False


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        weight_bit_width, act_bit_width, stride=1, padding=0, groups=1,
        bn_eps=1e-05, activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = make_quant_conv2d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, groups=groups, bias=False, bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = make_quant_relu(bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


FIRST_LAYER_BIT_WIDTH = 8


class MobileNet(nn.Module):

    def __init__(self, channels, first_stage_stride, bit_width, in_channels
        =3, num_classes=1000):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]
        self.features = Sequential()
        init_block = ConvBlock(in_channels=in_channels, out_channels=
            init_block_channels, kernel_size=3, stride=2, weight_bit_width=
            FIRST_LAYER_BIT_WIDTH, activation_scaling_per_channel=True,
            act_bit_width=bit_width)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if j == 0 and (i != 0 or first_stage_stride) else 1
                mod = DwsConvBlock(in_channels=in_channels, out_channels=
                    out_channels, stride=stride, bit_width=bit_width,
                    pw_activation_scaling_per_channel=
                    pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = make_quant_avg_pool(kernel_size=7, stride=1,
            signed=False, bit_width=bit_width)
        self.output = make_quant_linear(in_channels, num_classes, bias=True,
            enable_bias_quant=True, bit_width=bit_width,
            weight_scaling_per_output_channel=False)

    def forward(self, x):
        quant_tensor = self.features(x)
        x, scale, bit_width = self.final_pool(quant_tensor)
        x = x.view(x.size(0), -1)
        out = self.output(pack_quant_tensor(x, scale, bit_width))
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, weight_bit_width, act_bit_width, act_scaling_per_channel,
        bias, groups=1, bn_eps=1e-05, shared_act=None, return_quant_tensor=
        False):
        super(ConvBlock, self).__init__()
        self.conv = make_quant_conv2d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, groups=groups, bias=bias, bit_width=weight_bit_width,
            weight_scaling_per_output_channel=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if shared_act is None:
            self.activ = make_quant_relu(bit_width=act_bit_width,
                scaling_per_channel=act_scaling_per_channel,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                return_quant_tensor=return_quant_tensor)
        else:
            self.activ = shared_act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ProxylessBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        bn_eps, expansion, bit_width, depthwise_bit_width, shared_act):
        super(ProxylessBlock, self).__init__()
        self.use_bc = expansion > 1
        mid_channels = in_channels * expansion
        if self.use_bc:
            self.bc_conv = ConvBlock(in_channels=in_channels, out_channels=
                mid_channels, kernel_size=1, stride=1, padding=0, groups=1,
                bn_eps=bn_eps, act_scaling_per_channel=True,
                weight_bit_width=bit_width, bias=False, act_bit_width=
                depthwise_bit_width)
        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(in_channels=mid_channels, out_channels=
            mid_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, groups=mid_channels, bn_eps=bn_eps,
            act_scaling_per_channel=False, weight_bit_width=
            depthwise_bit_width, act_bit_width=bit_width, bias=False)
        self.pw_conv = ConvBlock(in_channels=mid_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, groups=1,
            bn_eps=bn_eps, weight_bit_width=bit_width, shared_act=
            shared_act, bias=False, act_bit_width=None,
            act_scaling_per_channel=None)

    def forward(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        bn_eps, expansion, residual, shortcut, bit_width,
        depthwise_bit_width, shared_act):
        super(ProxylessUnit, self).__init__()
        assert residual or shortcut
        assert shared_act is not None
        self.residual = residual
        self.shortcut = shortcut
        if self.residual:
            self.body = ProxylessBlock(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, stride=
                stride, bn_eps=bn_eps, expansion=expansion, bit_width=
                bit_width, depthwise_bit_width=depthwise_bit_width,
                shared_act=shared_act)
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


HADAMARD_FIXED_SCALE = False


def make_hadamard_classifier(in_channels, out_channels, fixed_scale=
    HADAMARD_FIXED_SCALE):
    return qnn.HadamardClassifier(in_channels=in_channels, out_channels=
        out_channels, fixed_scale=fixed_scale)


HARD_TANH_THRESHOLD = 10.0


class ProxylessNAS(nn.Module):

    def __init__(self, channels, init_block_channels, final_block_channels,
        residuals, shortcuts, kernel_sizes, expansions, bit_width,
        depthwise_bit_width, first_layer_weight_bit_width,
        hadamard_classifier, bn_eps=0.001, in_channels=3, num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.features = nn.Sequential()
        init_block = ConvBlock(in_channels=in_channels, out_channels=
            init_block_channels, kernel_size=3, stride=2, padding=1, groups
            =1, bn_eps=bn_eps, act_scaling_per_channel=False, bias=False,
            act_bit_width=bit_width, weight_bit_width=
            first_layer_weight_bit_width)
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
                    shared_act = make_quant_hard_tanh(bit_width=bit_width,
                        return_quant_tensor=True)
                unit = ProxylessUnit(in_channels=in_channels, out_channels=
                    out_channels, kernel_size=kernel_size, stride=stride,
                    bn_eps=bn_eps, expansion=expansion, residual=residual,
                    shortcut=shortcut, bit_width=bit_width,
                    depthwise_bit_width=depthwise_bit_width, shared_act=
                    shared_act)
                stage.add_module('unit{}'.format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        final_block = ConvBlock(in_channels=in_channels, out_channels=
            final_block_channels, kernel_size=1, stride=1, padding=0,
            groups=1, bn_eps=bn_eps, act_scaling_per_channel=False,
            act_bit_width=bit_width, weight_bit_width=bit_width, bias=False,
            return_quant_tensor=True)
        self.features.add_module('final_block', final_block)
        in_channels = final_block_channels
        self.final_pool = make_quant_avg_pool(kernel_size=7, stride=1,
            signed=False, bit_width=bit_width)
        if hadamard_classifier:
            self.output = make_hadamard_classifier(in_channels=in_channels,
                out_channels=num_classes)
        else:
            self.output = make_quant_linear(in_channels=in_channels,
                out_channels=num_classes, bias=True, enable_bias_quant=True,
                bit_width=bit_width, weight_scaling_per_output_channel=False)

    def forward(self, x):
        x = self.features(x)
        x, scale, bit_width = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(pack_quant_tensor(x, scale, bit_width))
        return x


def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = make_quant_conv2d(in_channels, v, kernel_size=3,
                stride=1, padding=1, groups=1, bias=not batch_norm,
                bit_width=bit_width)
            act = make_quant_relu(bit_width)
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
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(make_quant_linear(512 * 7 * 7, 4096,
            bias=True, bit_width=bit_width), make_quant_relu(bit_width), nn
            .Dropout(), make_quant_linear(4096, 4096, bias=True, bit_width=
            bit_width), make_quant_relu(bit_width), nn.Dropout(),
            make_quant_linear(4096, num_classes, bias=False, bit_width=
            bit_width, weight_scaling_per_output_channel=False))
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        self.torch_windows = {'hann': torch.hann_window, 'hamming': torch.
            hamming_window, 'blackman': torch.blackman_window, 'bartlett':
            torch.bartlett_window, 'ones': torch.ones, None: torch.ones}

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

    def __init__(self, *, freq_masks=0, time_masks=0, freq_width=10,
        time_width=10, rect_masks=0, rect_time=5, rect_freq=20, rng=None,
        **kwargs):
        nn.Module.__init__(self)
        if rect_masks > 0:
            self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=
                rect_time, rect_freq=rect_freq, rng=rng)
        else:
            self.spec_cutout = lambda x: x
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(freq_masks=freq_masks,
                time_masks=time_masks, freq_width=freq_width, time_width=
                time_width, rng=rng)
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

    def __init__(self, manifest_paths, labels, max_duration=None,
        min_duration=None, sort_by_duration=False, max_utts=0, blank_index=
        -1, unk_index=-1, normalize=True, logger=None):
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
                self.logger.warning(
                    'WARNING: Got transcript: {}. It is not a string. Dropping data point'
                    .format(text))
                filtered_duration += item['duration']
                continue
            item['tokens'] = self.tokenize_transcript(text, self.labels_map,
                self.unk_index, self.blank_index)
            if 'audio_filename' in item and 'audio_filepath' not in item:
                self.logger.warning(
                    'Malformed manifest: The key audio_filepath was not found in the manifest. Using audio_filename instead.'
                    )
                item['audio_filepath'] = item['audio_filename']
            data.append(item)
            duration += item['duration']
            if max_utts > 0 and len(data) >= max_utts:
                self.logger.info('Stop parsing due to max_utts ({})'.format
                    (max_utts))
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


ABBREVIATIONS_COMMON = [(re.compile('\\b%s\\.' % x[0]), x[1]) for x in [(
    'ms', 'miss'), ('mrs', 'misess'), ('mr', 'mister'), ('messrs',
    'messeurs'), ('dr', 'doctor'), ('drs', 'doctors'), ('st', 'saint'), (
    'co', 'company'), ('jr', 'junior'), ('sr', 'senior'), ('rev',
    'reverend'), ('hon', 'honorable'), ('sgt', 'sergeant'), ('capt',
    'captain'), ('maj', 'major'), ('col', 'colonel'), ('lt', 'lieutenant'),
    ('gen', 'general'), ('prof', 'professor'), ('lb', 'pounds'), ('rep',
    'representative'), ('st', 'street'), ('ave', 'avenue'), ('etc',
    'et cetera'), ('jan', 'january'), ('feb', 'february'), ('mar', 'march'),
    ('apr', 'april'), ('jun', 'june'), ('jul', 'july'), ('aug', 'august'),
    ('sep', 'september'), ('oct', 'october'), ('nov', 'november'), ('dec',
    'december')]]


ABBREVIATIONS_EXPANDED = [(re.compile('\\b%s\\.' % x[0]), x[1]) for x in [(
    'ltd', 'limited'), ('fig', 'figure'), ('figs', 'figures'), ('gent',
    'gentlemen'), ('ft', 'fort'), ('esq', 'esquire'), ('prep',
    'preperation'), ('bros', 'brothers'), ('ind', 'independent'), ('mme',
    'madame'), ('pro', 'professional'), ('vs', 'versus'), ('inc', 'include')]]


def clean_abbreviations(string, expanded=False):
    for regex, replacement in ABBREVIATIONS_COMMON:
        string = re.sub(regex, replacement, string)
    if expanded:
        for regex, replacement in ABBREVIATIONS_EXPANDED:
            string = re.sub(regex, replacement, string)
    return string


NUM_CHECK = re.compile(
    '([$]?)(^|\\s)(\\S*[0-9]\\S*)(?=(\\s|$)((\\S*)(\\s|$))?)')


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
        string = re.sub('\\{}'.format(punc), ' {} '.format(replacement), string
            )
    string = string.translate(table)
    return string


def warn_common_chars(string):
    if re.search('[£€]', string):
        print(
            "WARNING: Your transcript contains one of '£' or '€' which we donot currently handle"
            )


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
                print('WARNING: Normalizing {} failed'.format(text))
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

    def __init__(self, manifest_filepath, labels, featurizer, max_duration=
        None, min_duration=None, max_utts=0, blank_index=-1, unk_index=-1,
        normalize=True, trim=False, bos_id=None, eos_id=None, logger=False,
        load_audio=True, manifest_class=ManifestEN):
        m_paths = manifest_filepath.split(',')
        self.manifest = manifest_class(m_paths, labels, max_duration=
            max_duration, min_duration=min_duration, max_utts=max_utts,
            blank_index=blank_index, unk_index=unk_index, normalize=
            normalize, logger=logger)
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        if logger:
            logger.info(
                'Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.'
                .format(self.manifest.duration / 3600, self.manifest.
                filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(sample['audio_filepath'],
                offset=offset, duration=duration, trim=self.trim)
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

    def __init__(self, samples, sample_rate, target_sr=None, trim=False,
        trim_db=60):
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
        return (
            '%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB'
             % (type(self), self.num_samples, self.sample_rate, self.
            duration, self.rms_db))

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
    def from_file(cls, filename, target_sr=None, int_values=False, offset=0,
        duration=0, trim=False):
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
    def segment_from_file(cls, filename, target_sr=None, n_segments=0, trim
        =False):
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
        self._samples = np.pad(self._samples, (pad_size if symmetric else 0,
            pad_size), mode='constant')

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
            raise ValueError(
                'The slice start position (%f s) is out of bounds.' %
                start_time)
        if end_time < 0.0:
            raise ValueError(
                'The slice end position (%f s) is out of bounds.' % end_time)
        if start_time > end_time:
            raise ValueError(
                'The slice start position (%f s) is later than the end position (%f s).'
                 % (start_time, end_time))
        if end_time > self.duration:
            raise ValueError(
                'The slice end position (%f s) is out of bounds (> %f s)' %
                (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]


class ImpulsePerturbation(Perturbation):

    def __init__(self, manifest_path=None, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse_record = self._rng.sample(self._manifest.data, 1)[0]
        impulse = AudioSegment.from_file(impulse_record['audio_filepath'],
            target_sr=data.sample_rate)
        data._samples = signal.fftconvolve(data.samples, impulse.samples,
            'full')


class NoisePerturbation(Perturbation):

    def __init__(self, manifest_path=None, min_snr_db=40, max_snr_db=50,
        max_gain_db=300.0, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    def perturb(self, data):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        noise_record = self._rng.sample(self._manifest.data, 1)[0]
        noise = AudioSegment.from_file(noise_record['audio_filepath'],
            target_sr=data.sample_rate)
        noise_gain_db = min(data.rms_db - noise.rms_db - snr_db, self.
            _max_gain_db)
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        noise.subsegment(start_time=start_time, end_time=start_time + data.
            duration)
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


perturbation_types = {'speed': SpeedPerturbation, 'gain': GainPerturbation,
    'impulse': ImpulsePerturbation, 'shift': ShiftPerturbation, 'noise':
    NoisePerturbation}


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
                print(p['aug_type'], 'perturbation not known. Skipping.')
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


class WaveformFeaturizer(object):

    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = (augmentor if augmentor is not None else
            AudioAugmentor())
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(file_path, target_sr=self.
            sample_rate, int_values=self.int_values, offset=offset,
            duration=duration, trim=trim)
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
        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa
            )


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
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=
                token_pad_value)
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

    def __init__(self, *, manifest_filepath, labels, batch_size,
        sample_rate=16000, int_values=False, bos_id=None, eos_id=None,
        pad_id=None, min_duration=0.1, max_duration=None,
        normalize_transcripts=True, trim_silence=False, load_audio=True,
        drop_last=False, shuffle=True, num_workers=4, placement='cpu', **kwargs
        ):
        super().__init__()
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate,
            int_values=int_values, augmentor=None)
        dataset_params = {'manifest_filepath': manifest_filepath, 'labels':
            labels, 'featurizer': self._featurizer, 'max_duration':
            max_duration, 'min_duration': min_duration, 'normalize':
            normalize_transcripts, 'trim': trim_silence, 'bos_id': bos_id,
            'eos_id': eos_id, 'logger': None, 'load_audio': load_audio}
        self._dataset = AudioDataset(**dataset_params)
        if placement == 'cuda':
            None
            sampler = torch.utils.data.distributed.DistributedSampler(self.
                _dataset)
        else:
            sampler = None
        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(dataset=self.
            _dataset, batch_size=batch_size, collate_fn=partial(
            seq_collate_fn, token_pad_value=pad_id), drop_last=drop_last,
            shuffle=shuffle if sampler is None else False, sampler=sampler,
            num_workers=num_workers)

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
        loss = self._criterion(log_probs.transpose(1, 0), targets,
            input_length, target_length)
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*kwargs.values())


CONSTANT = 1e-05


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == 'per_feature':
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
            device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
            device=x.device)
        for i in range(x.shape[0]):
            x_mean[(i), :] = x[(i), :, :seq_len[i]].mean(dim=1)
            x_std[(i), :] = x[(i), :, :seq_len[i]].std(dim=1)
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == 'all_features':
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[(i), :, :seq_len[i].item()].mean()
            x_std[i] = x[(i), :, :seq_len[i].item()].std()
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

    def __init__(self, *, sample_rate=16000, n_window_size=320,
        n_window_stride=160, window='hann', normalize='per_feature', n_fft=
        None, preemph=0.97, nfilt=64, lowfreq=0, highfreq=None, log=True,
        log_zero_guard_type='add', log_zero_guard_value=2 ** -24, dither=
        CONSTANT, pad_to=16, max_duration=16.7, frame_splicing=1, stft_conv
        =False, pad_value=0, mag_power=2.0, logger=None):
        super(FilterbankFeatures, self).__init__()
        if n_window_size is None or n_window_stride is None or not isinstance(
            n_window_size, int) or not isinstance(n_window_stride, int
            ) or n_window_size <= 0 or n_window_stride <= 0:
            raise ValueError(
                f'{self} got an invalid value for either n_window_size or n_window_stride. Both must be positive ints.'
                )
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
            self.stft = STFTPatch(self.n_fft, self.hop_length, self.
                win_length, window)
        else:
            None
            torch_windows = {'hann': torch.hann_window, 'hamming': torch.
                hamming_window, 'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window, 'none': None}
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length, periodic=False
                ) if window_fn else None
            self.register_buffer('window', window_tensor)
            self.stft = lambda x: torch.stft(x, n_fft=self.n_fft,
                hop_length=self.hop_length, win_length=self.win_length,
                center=True, window=self.window)
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        filterbanks = torch.tensor(librosa.filters.mel(sample_rate, self.
            n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.
            float).unsqueeze(0)
        self.register_buffer('fb', filterbanks)
        max_length = self.get_seq_len(torch.tensor(max_duration *
            sample_rate, dtype=torch.float))
        max_pad = pad_to - max_length % pad_to
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power
        if log_zero_guard_type not in ['add', 'clamp']:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the log_zero_guard_type parameter. It must be either 'add' or 'clamp'."
                )
        self.log_zero_guard_value = lambda _: log_zero_guard_value
        if isinstance(log_zero_guard_value, str):
            if log_zero_guard_value == 'tiny':
                self.log_zero_guard_value = lambda x: torch.finfo(x.dtype).tiny
            elif log_zero_guard_value == 'eps':
                self.log_zero_guard_value = lambda x: torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {log_zero_guard_value} for the log_zero_guard_type parameter. It must be either a number, 'tiny', or 'eps'"
                    )
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
            x = torch.cat((x[:, (0)].unsqueeze(1), x[:, 1:] - self.preemph *
                x[:, :-1]), dim=1)
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
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)),
                value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.
                    pad_value)
        return x


BIAS_CONFIGS = False


def make_quantconv1d(feat_in, feat_out, kernel_size, stride, padding,
    bit_width, dilation=1, group=1):
    return quant_nn.QuantConv1d(in_channels=feat_in, out_channels=feat_out,
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=
        dilation, groups=group, weight_bit_width=bit_width,
        weight_quant_type=QUANT_TYPE, weight_narrow_range=
        WEIGHT_NARROW_RANGE, weight_scaling_impl_type=
        WEIGHT_SCALING_IMPL_TYPE, weight_scaling_stats_op=
        WEIGHT_SCALING_STATS_OP, weight_scaling_min_val=SCALING_MIN_VAL,
        bias_bit_width=bit_width, bias_quant_type=QUANT_TYPE_BIAS,
        bias_narrow_range=BIAS_CONFIGS, compute_output_scale=BIAS_CONFIGS,
        compute_output_bit_width=BIAS_CONFIGS, return_quant_tensor=False)


class MaskedConv1d(nn.Module):
    __constants__ = ['use_conv_mask', 'real_out_channels', 'heads']

    def __init__(self, in_channels, out_channels, kernel_size,
        scaling_per_channel, bit_width, stride=1, padding=0, dilation=1,
        groups=1, heads=-1, bias=False, use_mask=True):
        super(MaskedConv1d, self).__init__()
        if not (heads == -1 or groups == in_channels):
            raise ValueError('Only use heads for depthwise convolutions')
        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads
        self.conv = make_quantconv1d(in_channels, out_channels, kernel_size,
            bias=bias, stride=stride, padding=padding, dilation=dilation,
            groups=groups, scaling_per_channel=scaling_per_channel,
            bit_width=bit_width)
        self.is_depthwise = (in_channels == out_channels and in_channels ==
            groups)
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (
            self.conv.kernel_size[0] - 1) - 1) / self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens
            max_len = x.size(2)
            mask = torch.arange(max_len).expand(len(lens), max_len
                ) >= lens.unsqueeze(1)
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


def make_jasper_activation(activation, channels, bit_width,
    absolute_act_val, scaling_per_channel):
    brevitas_activation = brevitas_activations[activation]
    return brevitas_activation(bit_width=bit_width, scaling_per_channel=
        scaling_per_channel, quant_type=QUANT_TYPE, scaling_impl_type=
        ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, max_val=
        absolute_act_val, per_channel_broadcastable_shape=(1, channels, 1),
        scaling_stats_permute_dims=(1, 0, 2), return_quant_tensor=False)


def make_norm_scale(bit_width, absolute_act_val, scaling_per_channel):
    return quant_nn.QuantHardTanh(bit_width=bit_width, scaling_per_channel=
        scaling_per_channel, quant_type=QUANT_TYPE, scaling_impl_type=
        ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, max_val=
        absolute_act_val, min_val=-absolute_act_val,
        scaling_stats_permute_dims=(1, 0, 2), return_quant_tensor=True)


class JasperBlock(nn.Module):
    __constants__ = ['conv_mask', 'separable', 'residual_mode', 'res', 'mconv']

    def __init__(self, inplanes, planes, bit_width, absolute_act_val,
        activation_inner_scaling_per_output_channel,
        activation_other_scaling_per_output_channel,
        weight_scaling_per_output_channel, repeat=3, kernel_size=11, stride
        =1, dilation=1, padding='same', dropout=0.2, activation=None,
        residual=True, groups=1, separable=False, heads=-1, normalization=
        'batch', norm_groups=1, residual_mode='add', residual_panes=[],
        conv_mask=False, fused_bn=False):
        super(JasperBlock, self).__init__()
        if padding != 'same':
            raise ValueError("currently only 'same' padding is supported")
        self.fused_bn = fused_bn
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.quant_normalization = make_norm_scale(bit_width=bit_width,
            absolute_act_val=absolute_act_val, scaling_per_channel=
            activation_other_scaling_per_output_channel)
        self.conv_module_to_merge = []
        inplanes_loop = inplanes
        conv = nn.ModuleList()
        self.norm_depthwise = nn.ModuleList()
        for _ in range(repeat - 1):
            if separable:
                self.norm_depthwise.extend([make_norm_scale(bit_width=
                    bit_width, absolute_act_val=absolute_act_val,
                    scaling_per_channel=
                    activation_other_scaling_per_output_channel)])
            conv.extend(self._get_conv_bn_layer(inplanes_loop, planes,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                padding=padding_val, groups=groups, heads=heads, separable=
                separable, normalization=normalization, norm_groups=
                norm_groups, bit_width=bit_width, scaling_per_channel=
                weight_scaling_per_output_channel))
            conv.extend(self._get_act_dropout_layer(drop_prob=dropout,
                activation=activation, channels=planes, bit_width=bit_width,
                absolute_act_val=absolute_act_val, scaling_per_channel=
                activation_inner_scaling_per_output_channel))
            inplanes_loop = planes
        if separable:
            self.norm_depthwise.extend([make_norm_scale(bit_width=bit_width,
                absolute_act_val=absolute_act_val, scaling_per_channel=
                activation_other_scaling_per_output_channel)])
        conv.extend(self._get_conv_bn_layer(inplanes_loop, planes,
            kernel_size=kernel_size, stride=stride, dilation=dilation,
            padding=padding_val, groups=groups, heads=heads, separable=
            separable, normalization=normalization, norm_groups=norm_groups,
            bit_width=bit_width, scaling_per_channel=
            weight_scaling_per_output_channel))
        self.mconv = conv
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            res_list = nn.ModuleList()
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res_list.append(nn.ModuleList(self._get_conv_bn_layer(ip,
                    planes, kernel_size=1, normalization=normalization,
                    norm_groups=norm_groups, bit_width=bit_width,
                    scaling_per_channel=weight_scaling_per_output_channel)))
            self.res = res_list
        else:
            self.res = None
        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=
            dropout, activation=activation, channels=inplanes_loop,
            absolute_act_val=absolute_act_val, scaling_per_channel=
            activation_other_scaling_per_output_channel, bit_width=bit_width))

    def _get_conv(self, in_channels, out_channels, bit_width,
        scaling_per_channel, kernel_size=11, stride=1, dilation=1, padding=
        0, bias=False, groups=1, heads=-1, separable=False):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding, bias=
                bias, groups=groups, heads=heads, use_mask=use_mask,
                scaling_per_channel=scaling_per_channel, bit_width=bit_width)
        else:
            return make_quantconv1d(in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding, groups=
                groups, bias=bias, scaling_per_channel=scaling_per_channel,
                bit_width=bit_width)

    def _get_conv_bn_layer(self, in_channels, out_channels, bit_width,
        scaling_per_channel, kernel_size=11, stride=1, dilation=1, padding=
        0, bias=False, groups=1, heads=-1, separable=False, normalization=
        'batch', norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels
        if separable:
            layers = [self._get_conv(in_channels, in_channels, kernel_size=
                kernel_size, stride=stride, dilation=dilation, padding=
                padding, groups=in_channels, heads=heads, bias=bias,
                scaling_per_channel=scaling_per_channel, bit_width=
                bit_width), self._get_conv(in_channels, out_channels,
                kernel_size=1, stride=1, dilation=1, padding=0, groups=
                groups, bias=bias, scaling_per_channel=scaling_per_channel,
                bit_width=bit_width)]
        else:
            layers = [self._get_conv(in_channels, out_channels, kernel_size
                =kernel_size, scaling_per_channel=scaling_per_channel,
                bit_width=bit_width, stride=stride, bias=bias, dilation=
                dilation, padding=padding, groups=groups)]
        if normalization == 'group':
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels
                =out_channels))
        elif normalization == 'instance':
            layers.append(nn.GroupNorm(num_groups=out_channels,
                num_channels=out_channels))
        elif normalization == 'layer':
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels)
                )
        elif normalization == 'batch':
            if self.fused_bn:
                self.conv_module_to_merge.append(layers[-1])
                layers.append(nn.Identity())
            else:
                layers.append(nn.BatchNorm1d(out_channels, eps=0.001,
                    momentum=0.1))
        else:
            raise ValueError(
                f'Normalization method ({normalization}) does not match one of [batch, layer, group, instance].'
                )
        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, channels, bit_width, absolute_act_val,
        scaling_per_channel, drop_prob=0.2, activation=None):
        if activation is None:
            raise Exception('Activation required')
        layers = [make_jasper_activation(activation, channels, bit_width=
            bit_width, absolute_act_val=absolute_act_val,
            scaling_per_channel=scaling_per_channel), nn.Dropout(p=drop_prob)]
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
                check_flag = check_flag or l.is_depthwise
                if l.is_depthwise:
                    out, scale, bit = self.norm_depthwise[count_norm](out)
                    count_norm += 1
            else:
                out = l(out)
        if check_flag:
            assert len(self.norm_depthwise) == count_norm
        if self.res is not None:
            out = self.quant_normalization(out)
            if self.training:
                out, scale, bit = out
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)
                res_out = self.quant_normalization(res_out)
                if self.training:
                    res_out, scale, bit = res_out
                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)
        if isinstance(out, QuantTensor):
            out, scale, bit = out
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens
        return [out], lens

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        if self.fused_bn:
            self.fuse_bn(state_dict, prefix)
        super(JasperBlock, self)._load_from_state_dict(state_dict, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

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
                    conv_name = prefix_long[:-1] + [str(module_number - 1)] + [
                        'conv']
                    conv_name = '.'.join(conv_name)
                    conv_mod = self.conv_module_to_merge[index]
                    index = index + 1
                    bn_weight_key = '.'.join([bn_prefix, 'weight'])
                    bn_bias_key = '.'.join([bn_prefix, 'bias'])
                    bn_running_mean_key = '.'.join([bn_prefix, 'running_mean'])
                    bn_running_var_key = '.'.join([bn_prefix, 'running_var'])
                    bn_num_batches_traked_key = '.'.join([bn_prefix,
                        'num_batches_tracked'])
                    keys_to_delete = keys_to_delete + [bn_bias_key]
                    keys_to_delete = keys_to_delete + [bn_weight_key]
                    keys_to_delete = keys_to_delete + [bn_running_mean_key]
                    keys_to_delete = keys_to_delete + [bn_running_var_key]
                    keys_to_delete = keys_to_delete + [
                        bn_num_batches_traked_key]
                    mul_factor, add_factor = mul_add_from_bn(bn_mean=
                        state_dict[bn_running_mean_key], bn_var=state_dict[
                        bn_running_var_key], bn_eps=0.001, bn_weight=
                        state_dict[bn_weight_key], bn_bias=state_dict[
                        bn_bias_key], affine_only=False)
                    if isinstance(conv_mod, MaskedConv1d):
                        conv_mod = conv_mod.conv
                    mul_shape = conv_mod.per_output_channel_broadcastable_shape
                    conv_weight_key = conv_name + '.weight'
                    conv_bias_key = conv_name + '.bias'
                    result = state_dict[conv_weight_key] * mul_factor.view(
                        mul_shape)
                    state_dict[conv_weight_key] = result
                    if (conv_mod.bias is not None and conv_bias_key in
                        state_dict):
                        state_dict[conv_bias_key] += add_factor
                    elif conv_mod.bias is not None and not conv_bias_key in state_dict:
                        state_dict[conv_bias_key] = add_factor
                    else:
                        if torch.is_available():
                            add_factor = add_factor
                        conv_mod.bias = nn.Parameter(add_factor)
                        state_dict[conv_bias_key] = add_factor
                else:
                    state_dict[name] = state_dict[name]
        for k in list(state_dict.keys()):
            if k in keys_to_delete:
                del state_dict[k]
        assert len(self.conv_module_to_merge) == index


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

    def __init__(self, freq_masks=0, time_masks=0, freq_width=10,
        time_width=10, rng=None):
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
                mask[(idx), x_left:x_left + w, :] = 1
            for i in range(self.time_masks):
                y_left = int(self._rng.uniform(0, sh[2] - self.time_width))
                w = int(self._rng.uniform(0, self.time_width))
                mask[(idx), :, y_left:y_left + w] = 1
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
                mask[(idx), rect_x:rect_x + w_x, rect_y:rect_y + w_y] = 1
        x = x.masked_fill(mask.type(torch.bool), 0)
        return x


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

    def __init__(self, *, jasper, outer_bit_width, inner_bit_width,
        weight_scaling_per_output_channel, absolute_act_val,
        activation_inner_scaling_per_output_channel,
        activation_other_scaling_per_output_channel, activation, feat_in,
        fused_bn=False, normalization_mode='batch', residual_mode='add',
        norm_groups=-1, conv_mask=True, frame_splicing=1, init_mode=
        'xavier_uniform', **kwargs):
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
            encoder_layers.append(JasperBlock(feat_in, lcfg['filters'],
                repeat=lcfg['repeat'], kernel_size=lcfg['kernel'], stride=
                lcfg['stride'], dilation=lcfg['dilation'], dropout=lcfg[
                'dropout'], residual=lcfg['residual'], groups=groups,
                fused_bn=fused_bn, separable=separable, heads=heads,
                residual_mode=residual_mode, normalization=
                normalization_mode, norm_groups=norm_groups, activation=
                activation, residual_panes=dense_res, conv_mask=conv_mask,
                bit_width=bit_width, absolute_act_val=absolute_act_val,
                activation_inner_scaling_per_output_channel=
                activation_inner_scaling_per_output_channel,
                activation_other_scaling_per_output_channel=
                activation_other_scaling_per_output_channel,
                weight_scaling_per_output_channel=
                weight_scaling_per_output_channel))
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

    def __init__(self, *, feat_in, num_classes, bit_width,
        weight_scaling_per_channel, init_mode='xavier_uniform', **kwargs):
        nn.Module.__init__(self)
        self._feat_in = feat_in
        self._num_classes = num_classes + 1
        self.decoder_layers = nn.Sequential(make_quantconv1d(self._feat_in,
            self._num_classes, kernel_size=1, bias=True, bit_width=
            bit_width, scaling_per_channel=weight_scaling_per_channel))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        return F.log_softmax(self.decoder_layers(encoder_output).transpose(
            1, 2), dim=-1)


class Quartznet(nn.Module):

    def __init__(self, preprocessing, encoder, decoder, greedyctcdecoder):
        super(Quartznet, self).__init__()
        self.preprocessing = preprocessing
        self.encoder = encoder
        self.decoder = decoder
        self.greedy_ctc_decoder = greedyctcdecoder

    def forward(self, input_tensors):
        audio_signal_e1, a_sig_length_e1, _, _ = input_tensors
        processed_signal_e1, p_length_e1 = self.preprocessing(input_signal=
            audio_signal_e1, length=a_sig_length_e1)
        encoded_e1, encoded_len_e1 = self.encoder(audio_signal=
            processed_signal_e1, length=p_length_e1)
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


ACT_MIN_VAL = -1


def make_hardtanh_activation(bit_width, return_quant_tensor=False):
    return quant_nn.QuantHardTanh(bit_width=bit_width, max_val=ACT_MAX_VAL,
        min_val=ACT_MIN_VAL, quant_type=QUANT_TYPE, scaling_impl_type=
        ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL,
        return_quant_tensor=return_quant_tensor)


def make_leakyRelu_activation(bit_width):
    el1 = nn.LeakyReLU()
    el2 = make_hardtanh_activation(bit_width=bit_width)
    layer = nn.Sequential(el1, el2)
    return layer


def make_tanh_activation(bit_width):
    return quant_nn.QuantTanh(bit_width=bit_width, quant_type=QUANT_TYPE,
        scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=False)


def make_transpconv1d(feat_in, feat_out, kernel_size, stride, padding,
    bit_width, dilation=1):
    return quant_nn.QuantConvTranspose1d(in_channels=feat_in, out_channels=
        feat_out, kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, weight_bit_width=bit_width, weight_quant_type=
        QUANT_TYPE, weight_narrow_range=WEIGHT_NARROW_RANGE,
        weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
        weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
        weight_scaling_min_val=SCALING_MIN_VAL, bias_bit_width=bit_width,
        bias_quant_type=QUANT_TYPE_BIAS, bias_narrow_range=BIAS_CONFIGS,
        compute_output_scale=BIAS_CONFIGS, compute_output_bit_width=
        BIAS_CONFIGS, return_quant_tensor=False)


class Generator(nn.Module):

    def __init__(self, mel_channel, bit_width, last_layer_bit_width):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.generator = nn.Sequential(nn.utils.weight_norm(
            make_quantconv1d(mel_channel, 512, kernel_size=7, stride=1,
            padding=3, bit_width=bit_width)), make_leakyRelu_activation(
            bit_width=bit_width), nn.utils.weight_norm(make_transpconv1d(
            512, 256, kernel_size=16, stride=8, padding=4, bit_width=
            bit_width)), ResStack(256, bit_width=bit_width),
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_transpconv1d(256, 128, kernel_size=16, stride=8, padding=4,
            bit_width=bit_width)), ResStack(128, bit_width=bit_width),
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_transpconv1d(128, 64, kernel_size=4, stride=2, padding=1,
            bit_width=bit_width)), ResStack(64, bit_width=bit_width),
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_transpconv1d(64, 32, kernel_size=4, stride=2, padding=1,
            bit_width=bit_width)), ResStack(32, bit_width=bit_width),
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_quantconv1d(32, 1, kernel_size=7, stride=1, padding=3,
            bit_width=bit_width)), make_tanh_activation(bit_width=
            last_layer_bit_width))

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


class ResStack(nn.Module):

    def __init__(self, channel, bit_width):
        super(ResStack, self).__init__()
        self.scale_norm = make_hardtanh_activation(bit_width=bit_width,
            return_quant_tensor=True)
        self.layers = nn.ModuleList([nn.Sequential(
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_quantconv1d(channel, channel, kernel_size=3, stride=1,
            padding=3 ** i, dilation=3 ** i, bit_width=bit_width)),
            make_leakyRelu_activation(bit_width), nn.utils.weight_norm(
            make_quantconv1d(channel, channel, kernel_size=3, stride=1,
            padding=1, dilation=1, bit_width=bit_width))) for i in range(3)])

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


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
    n_fft=800, dtype=np.float32, norm=None):
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
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n -
            sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800,
        window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.
            imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, (None), :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale *
            fourier_basis).T[:, (None), :])
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
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length /
            2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        if torch.is_available():
            forward_transform = F.conv1d(input_data, Variable(self.
                forward_basis, requires_grad=False), stride=self.hop_length,
                padding=0).cpu()
        else:
            forward_transform = F.conv1d(input_data, Variable(self.
                forward_basis, requires_grad=False), stride=self.hop_length,
                padding=0).cpu()
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data,
            real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase),
            magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False), stride=self.
            hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1),
                hop_length=self.hop_length, win_length=self.win_length,
                n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum >
                tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(
                window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, (approx_nonzero_indices)] /= window_sum[
                approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length /
            2):]
        inverse_transform = inverse_transform[:, :, :-int(self.
            filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


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

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
        n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length,
            n_mel_channels, mel_fmin, mel_fmax)
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Xilinx_brevitas(_paritybench_base):
    pass
    def test_000(self):
        self._check(AffineRescaling(*[], **{'affine_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(AudioPreprocessor(*[], **{'win_length': 4, 'hop_length': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GreedyCTCDecoder(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GroupShuffle(*[], **{'groups': 1, 'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(MultiplyBatch(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4]), torch.rand([4, 4]), torch.rand([4])], {})

    def test_006(self):
        self._check(ScaleBias(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(SpecAugment(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SpecCutout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(SpectrogramAugmentation(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(SqrHingeLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(TensorNorm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(WeightReg(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(ZeroLsbTruncBitWidth(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

