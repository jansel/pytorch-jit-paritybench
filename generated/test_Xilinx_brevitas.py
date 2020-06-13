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


REMOVE_ZERO_BIT_WIDTH = 0.1


NON_ZERO_EPSILON = 1e-06


_global_config['IGNORE_MISSING_KEYS'] = 4


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


class LogTwo(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(LogTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.log2(x)


class RoundSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(RoundSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return round_ste(x)


class CeilSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(CeilSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return ceil_ste(x)


class PowerOfTwo(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(PowerOfTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return 2.0 ** x


class ClampMin(torch.jit.ScriptModule):
    __constants__ = ['min_val']

    def __init__(self, min_val: float) ->None:
        super(ClampMin, self).__init__()
        self.min_val = min_val

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x.clamp_min(self.min_val)


class Identity(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(Identity, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return identity(x)


class FloorSte(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(FloorSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return floor_ste(x)


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


class TensorClamp(torch.jit.ScriptModule):

    def __init__(self) ->None:
        super(TensorClamp, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val:
        torch.Tensor):
        return tensor_clamp(x, min_val=min_val, max_val=max_val)


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


class StatsInputViewShapeImpl(object):
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView


SCALING_SCALAR_SHAPE = ()


OVER_BATCH_OVER_CHANNELS_4D_SHAPE = 1, -1, 1, 1


ZERO_HW_SENTINEL_NAME = 'zero_hw_sentinel'


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


BIAS_ENABLED = False


WEIGHT_SCALING_CONST = 1.0


NARROW_RANGE_ENABLED = True


def get_quant_type(bit_width):
    if bit_width is None:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    else:
        return QuantType.INT


IN_DROPOUT = 0.2


FC_OUT_FEATURES = [64, 64, 64]


INTERMEDIATE_FC_PER_OUT_CH_SCALING = True


HIDDEN_DROPOUT = 0.2


LAST_FC_PER_OUT_CH_SCALING = False


HARD_TANH_MIN = -1.0


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


ACT_RETURN_QUANT_TENSOR = False


ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None


ENABLE_BIAS_QUANT = False


WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True


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


HARD_TANH_THRESHOLD = 10.0


ACT_SCALING_PER_CHANNEL = False


SCALING_MIN_VAL = 2e-09


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
        return torch.ceil(length / self.hop_length).to(dtype=torch.long)


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


def warn_common_chars(string):
    if re.search('[£€]', string):
        print(
            "WARNING: Your transcript contains one of '£' or '€' which we donot currently handle"
            )


NUM_CHECK = re.compile(
    '([$]?)(^|\\s)(\\S*[0-9]\\S*)(?=(\\s|$)((\\S*)(\\s|$))?)')


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
                center=True, window=self.window.to(dtype=torch.float))
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
        return torch.ceil(seq_len / self.hop_length).to(dtype=torch.long)

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
        x = torch.matmul(self.fb.to(x.dtype), x)
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
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.
            device), self.pad_value)
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


def make_norm_scale(bit_width, absolute_act_val, scaling_per_channel):
    return quant_nn.QuantHardTanh(bit_width=bit_width, scaling_per_channel=
        scaling_per_channel, quant_type=QUANT_TYPE, scaling_impl_type=
        ACT_SCALING_IMPL_TYPE, scaling_min_val=SCALING_MIN_VAL, max_val=
        absolute_act_val, min_val=-absolute_act_val,
        scaling_stats_permute_dims=(1, 0, 2), return_quant_tensor=True)


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
        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)
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
        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)
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


def make_tanh_activation(bit_width):
    return quant_nn.QuantTanh(bit_width=bit_width, quant_type=QUANT_TYPE,
        scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=False)


ACT_MAX_VAL = 1


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
        if torch.cuda.is_available():
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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Xilinx_brevitas(_paritybench_base):
    pass
    def test_000(self):
        self._check(ZeroLsbTruncBitWidth(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(AffineRescaling(*[], **{'affine_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ScaleBias(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(WeightReg(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(SqrHingeLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(TensorNorm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(AudioPreprocessor(*[], **{'win_length': 4, 'hop_length': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SpectrogramAugmentation(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(MultiplyBatch(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4]), torch.rand([4, 4]), torch.rand([4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(GreedyCTCDecoder(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(GroupShuffle(*[], **{'groups': 1, 'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(SpecAugment(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(SpecCutout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

