import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
activations_test = _module
attributes_test = _module
convolution_test = _module
general_test = _module
jit_test = _module
linear_test = _module
normalization_test = _module
padding_test = _module
pickle_test = _module
recurrent_test = _module
regularization_test = _module
torchlayers_test = _module
torchlayers = _module
_dev_utils = _module
helpers = _module
infer = _module
_name = _module
_version = _module
activations = _module
convolution = _module
module = _module
normalization = _module
pooling = _module
regularization = _module
upsample = _module

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


import itertools


import torch


import inspect


import types


import typing


import warnings


import collections


import math


import abc


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torchlayers.Sequential(torchlayers.Conv(64,
            kernel_size=7), torchlayers.activations.Swish(), torchlayers.
            InvertedResidualBottleneck(squeeze_excitation=False),
            torchlayers.AvgPool(), torchlayers.HardSwish(), torchlayers.
            SeparableConv(128), torchlayers.InvertedResidualBottleneck(),
            torch.nn.ReLU(), torchlayers.AvgPool(), torchlayers.
            DepthwiseConv(256), torchlayers.Poly(torchlayers.
            InvertedResidualBottleneck(), order=3), torchlayers.ReLU(),
            torchlayers.MaxPool(), torchlayers.Fire(out_channels=512),
            torchlayers.SqueezeExcitation(hidden=64), torchlayers.
            InvertedResidualBottleneck(), torchlayers.MaxPool(),
            torchlayers.InvertedResidualBottleneck(squeeze_excitation=False
            ), torchlayers.Dropout(), torchlayers.StochasticDepth(torch.nn.
            Sequential(torchlayers.InvertedResidualBottleneck(
            squeeze_excitation=False), torchlayers.
            InvertedResidualBottleneck(squeeze_excitation=False)), p=0.5),
            torchlayers.AvgPool())
        self.decoder = torchlayers.Sequential(torchlayers.Poly(torchlayers.
            InvertedResidualBottleneck(), order=2), torchlayers.
            ConvPixelShuffle(out_channels=512, upscale_factor=2),
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(),
            order=3), torchlayers.ConvPixelShuffle(out_channels=256,
            upscale_factor=2), torchlayers.StandardNormalNoise(),
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(),
            order=3), torchlayers.ConvPixelShuffle(out_channels=128,
            upscale_factor=2), torchlayers.Poly(torchlayers.
            InvertedResidualBottleneck(), order=4), torchlayers.
            ConvPixelShuffle(out_channels=64, upscale_factor=2),
            torchlayers.InvertedResidualBottleneck(), torchlayers.Conv(256),
            torchlayers.Swish(), torchlayers.BatchNorm(), torchlayers.
            ConvPixelShuffle(out_channels=32, upscale_factor=2),
            torchlayers.Conv(16), torchlayers.Swish(), torchlayers.Conv(3))

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class ConcatenateProxy(torch.nn.Module):

    def forward(self, tensor):
        return tensor, tensor, tensor


class _CustomLinearImpl(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias: bool=True):
        super().__init__(in_features, out_features, bias)
        self.some_params = torch.nn.Parameter(torch.randn(2, out_features))


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom in
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input::

        import torch
        import torchlayers as tl

        model = torch.nn.Sequential(tl.Lambda(lambda tensor: tensor ** 2))
        model(torch.randn(64 , 20))

    Parameters
    ----------
    function : Callable
        Any user specified function

    Returns
    -------
    Any
        Anything `function` returns

    """

    def __init__(self, function: typing.Callable):
        super().__init__()
        self.function: typing.Callable = function

    def forward(self, *args, **kwargs) ->typing.Any:
        return self.function(*args, **kwargs)


class Concatenate(torch.nn.Module):
    """Concatenate list of tensors.

    Mainly useful in `torch.nn.Sequential` when previous layer returns multiple
    tensors, e.g.::

        import torch
        import torchlayers as tl

        class Foo(torch.nn.Module):
            # Return same tensor three times
            # You could explicitly return a list or tuple as well
            def forward(tensor):
                return tensor, tensor, tensor


        model = torch.nn.Sequential(Foo(), tl.Concatenate())
        model(torch.randn(64 , 20))

    All tensors must have the same shape (except in the concatenating dimension).

    Parameters
    ----------
    dim : int
        Dimension along which tensors will be concatenated

    Returns
    -------
    torch.Tensor
        Concatenated tensor along specified `dim`.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Reshape(torch.nn.Module):
    """Reshape tensor excluding `batch` dimension

    Reshapes input `torch.Tensor` features while preserving batch dimension.
    Standard `torch.reshape` values (e.g. `-1`) are supported, e.g.::

        import torch
        import torchlayers as tl

        layer = tl.Reshape(20, -1)
        layer(torch.randn(64, 80)) # shape (64, 20, 4)

    All tensors must have the same shape (except in the concatenating dimension).
    If possible, no copy of `tensor` will be performed.

    Parameters
    ----------
    shapes: *int
        Variable length list of shapes used in view function

    Returns
    -------
    torch.Tensor
        Concatenated tensor

    """

    def __init__(self, *shapes: int):
        super().__init__()
        self.shapes: typing.Tuple[int] = shapes

    def forward(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0], *self.shapes))


def hard_sigmoid(tensor: torch.Tensor, inplace: bool=False) ->torch.Tensor:
    """
    Applies HardSigmoid function element-wise.

    See :class:`torchlayers.activations.HardSigmoid` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise
    inplace : bool, optional
        Whether operation should be performed `in-place`. Default: `False`

    Returns
    -------
    torch.Tensor
    """
    return torch.nn.functional.hardtanh(tensor, min_val=0, inplace=inplace)


class HardSigmoid(torch.nn.Module):
    """
    Applies HardSigmoid function element-wise.

    Uses `torch.nn.functional.hardtanh` internally with `0` and `1` ranges.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    """

    def forward(self, tensor: torch.Tensor):
        return hard_sigmoid(tensor)


def swish(tensor: torch.Tensor, beta: float=1.0) ->torch.Tensor:
    """
    Applies Swish function element-wise.

    See :class:`torchlayers.activations.Swish` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise
    beta : float, optional
        Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    Returns
    -------
    torch.Tensor
    """
    return torch.sigmoid(beta * tensor) * tensor


class Swish(torch.nn.Module):
    """
    Applies Swish function element-wise.

    .. math::

        Swish(x) = x / (1 + \\exp(-beta * x))

    This form was originally proposed by Prajit Ramachandran et al. in
    `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941.pdf>`__

    Parameters
    ----------
    beta : float, optional
        Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    """

    def __init__(self, beta: float=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, tensor: torch.Tensor):
        return swish(tensor, self.beta)


def hard_swish(tensor: torch.Tensor) ->torch.Tensor:
    """
    Applies HardSwish function element-wise.

    See :class:`torchlayers.activations.HardSwish` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    Returns
    -------
    torch.Tensor
    """
    return tensor * torch.nn.functional.relu6(tensor + 3) / 6


class HardSwish(torch.nn.Module):
    """
    Applies HardSwish function element-wise.

    .. math::

        HardSwish(x) = x * \\min(\\max(0,x + 3), 6) / 6


    While similar in effect to `Swish` should be more CPU-efficient.
    Above formula proposed by in Andrew Howard et al. in `Searching for MobileNetV3 <https://arxiv.org/pdf/1905.02244.pdf>`__.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    """

    def forward(self, tensor: torch.Tensor):
        return hard_swish(tensor)


class SeparableConv(torch.nn.Module):
    """Separable convolution layer (a.k.a. depthwise separable convolution).

    Based on input shape it either creates 1D, 2D or 3D separable convolution
    for inputs of shape 3D, 4D, 5D respectively (including batch as first dimension).

    Additional `same` `padding` mode was added and set as default.
    This mode preserves all dimensions excepts channels.

    `kernel_size` got a default value of `3`.

    .. note::
                **IMPORTANT**: `same` currently works only for odd values of `kernel_size`,
                `dilation` and `stride`. If any of those is even you should explicitly pad
                your input asymmetrically with `torch.functional.pad` or a-like.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : Union[int, Tuple[int, int], Tuple[int, int, int]], optional
        Size of the convolving kernel. User can specify `int` or 2-tuple (for `Conv2d`)
        or 3-tuple (for `Conv3d`). Default: `3`
    stride : Union[int, Tuple[int, int], Tuple[int, int, int]], optional
        Stride of the convolution. User can specify `int` or 2-tuple (for `Conv2d`)
        or 3-tuple (for `Conv3d`). Default: `3`
    padding : Union[str, int, Tuple[int, int], Tuple[int, int, int]], optional
        Padding added to both sides of the input. String "same" can be used with odd
        `kernel_size`, `stride` and `dilation`
        User can specify `int` or 2-tuple (for `Conv2d`)
        or 3-tuple (for `Conv3d`). Default: `same`
    dilation : Union[int, Tuple[int, int], Tuple[int, int, int]], optional
        Spacing between kernel elements. String "same" can be used with odd
        `kernel_size`, `stride` and `dilation`
        User can specify `int` or 2-tuple (for `Conv2d`)
        or 3-tuple (for `Conv3d`). Default: `1`
    bias : bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
    padding_mode : string, optional
        Accepted values `zeros` and `circular` Default: `zeros`

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3,
        stride=1, padding='same', dilation=1, bias: bool=True, padding_mode:
        str='zeros'):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: typing.Union[int, typing.Tuple[int, int], typing.
            Tuple[int, int, int]] = kernel_size
        self.stride: typing.Union[int, typing.Tuple[int, int], typing.Tuple
            [int, int, int]] = stride
        self.padding: typing.Union[str, int, typing.Tuple[int, int], typing
            .Tuple[int, int, int]] = padding
        self.dilation: typing.Union[int, typing.Tuple[int, int], typing.
            Tuple[int, int, int]] = dilation
        self.bias: bool = bias
        self.padding_mode: str = padding_mode
        self.depthwise = Conv(in_channels=in_channels, out_channels=
            in_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=in_channels, bias=bias,
            padding_mode=padding_mode)
        self.pointwise = Conv(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False, padding_mode=padding_mode)

    def forward(self, inputs):
        return self.pointwise(self.depthwise(inputs))


class ChannelShuffle(torch.nn.Module):
    """Shuffle output channels.

    When using group convolution knowledge transfer between next layers is reduced
    (as the same input channels are convolved with the same output channels).

    This layer reshuffles output channels via simple `reshape` in order to mix the representation
    from separate groups and improve knowledge transfer.

    Originally proposed by Xiangyu Zhang et al. in:
    `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`__

    Example::


        import torchlayers as tl

        model = tl.Sequential(
            tl.Conv(64),
            tl.Swish(),
            tl.Conv(128, groups=16),
            tl.ChannelShuffle(groups=16),
            tl.Conv(256),
            tl.GlobalMaxPool(),
            tl.Linear(10),
        )

    Parameters
    ----------
    groups : int
        Number of groups used in the previous convolutional layer.

    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups: int = groups

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], self.groups, -1, *inputs.
            shape[2:]).transpose(1, 2).reshape(*inputs.shape)


class ChannelSplit(torch.nn.Module):
    """Convenience layer splitting tensor using `p`.

    Returns two outputs, splitted accordingly to parameters.

    Example::

        import torchlayers as tl


        class Net(tl.Module):
            def __init__(self):
                super().__init__()
                self.layer = tl.Conv(256, groups=16)
                self.splitter = tl.ChannelSplit(0.5)

            def forward(x):
                outputs = self.layer(x)
                half, rest = self.splitter(outputs)
                return half # for some reason


    Parameters
    ----------
    p : float
        Percentage of channels to go into first group
    dim : int, optional
        Dimension along which input will be splitted. Default: `1` (channel dimension)

    """

    def __init__(self, p: float, dim: int=1):
        super().__init__()
        if not 0.0 < p < 1.0:
            raise ValueError(
                'Ratio of small expand fire module has to be between 0 and 1.')
        self.p: float = p
        self.dim: int = dim

    def forward(self, inputs):
        return torch.split(inputs, int(inputs.shape[1] * self.p), dim=self.dim)


class Residual(torch.nn.Module):
    """Residual connection adding input to output of provided module.

    Originally proposed by He et al. in `ResNet <www.arxiv.org/abs/1512.03385>`__

    For correct usage it is advised to keep input line (skip connection) without
    any layer or activation and implement transformations only in module arguments
    (as per `Identity Mappings in Deep Residual Networks <https://arxiv.org/pdf/1603.05027.pdf>`__).

    Example::


        import torch
        import torchlayers as tl

        # ResNet-like block
        class _BlockImpl(tl.Module):
            def __init__(self, in_channels: int):
                self.block = tl.Residual(
                    tl.Sequential(
                        tl.Conv(in_channels),
                        tl.ReLU(),
                        tl.Conv(4 * in_channels),
                        tl.ReLU(),
                        tl.Conv(in_channels),
                    )
                )

            def forward(self, x):
                return self.block(x)


        Block = tl.infer(_BlockImpl)



    Parameters
    ----------
    module : torch.nn.Module
        Convolutional PyTorch module (or other compatible module).
        Shape of module's `inputs` has to be equal to it's `outputs`, both
        should be addable `torch.Tensor` instances.
    projection : torch.nn.Module, optional
        If shapes of `inputs` and `module` results are different, it's user
        responsibility to add custom `projection` module (usually `1x1` convolution).
        Default: `None`

    """

    def __init__(self, module: torch.nn.Module, projection: torch.nn.Module
        =None):
        super().__init__()
        self.module: torch.nn.Module = module
        self.projection: torch.nn.Module = projection

    def forward(self, inputs):
        output = self.module(inputs)
        if self.projection is not None:
            inputs = self.projections(inputs)
        return output + inputs


class Dense(torch.nn.Module):
    """Dense residual connection concatenating input channels and output channels of provided module.

    Originally proposed by Gao Huang et al. in `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__

    Can be used just like `torchlayers.convolution.Residual` but concatenates
    channels (dimension can be specified) instead of adding.

    Parameters
    ----------
    module : torch.nn.Module
        Convolutional PyTorch module (or other compatible module).
        Shape of module's `inputs` has to be equal to it's `outputs`, both
        should be addable `torch.Tensor` instances.
    dim : int, optional
        Dimension along which `input` and module's `output` will be concatenated.
        Default: `1` (channel-wise)

    """

    def __init__(self, module: torch.nn.Module, dim: int=1):
        super().__init__()
        self.module: torch.nn.Module = module
        self.dim: int = dim

    def forward(self, inputs):
        return torch.cat(self.module(inputs), inputs, dim=self.dim)


class Poly(torch.nn.Module):
    """Apply one module to input multiple times and sum.

    It's equation for `order` equal to :math:`N` could be expressed as

    .. math::

        I + F + F^2 + ... + F^N

    where :math:`I` is identity mapping and :math:`F` is output of `module` applied :math:`^N` times.

    Originally proposed by Xingcheng Zhang et al. in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Example::

        import torchlayers as tl

        # Any input will be passed 3 times
        # Through the same convolutional layer (weights and biases shared)
        layer = tl.Sequential(tl.Conv(64), tl.Poly(tl.Conv(64), order=3))
        layer(torch.randn(1, 3, 32, 32))

    Above can be rewritten by the following::

        x = torch.randn(1, 3, 32, 32)

        first_convolution = tl.Conv(64)
        output = first_convolution(x)

        shared_convolution = tl.Conv(64)
        first_level = shared_convolution(output)
        second_level = shared_convolution(first_level)
        third_level = shared_convolution(second_level)

        # That's what tl.Poly would return
        final = output + first_level + second_level + third_level


    Parameters
    ----------
    module : torch.nn.Module
        Convolutional PyTorch module (or other compatible module).
        `inputs` shape has to be equal to it's `output` shape
        (for 2D convolution it would be :math:`(C, H, W)`)
    order : int, optional
        Order of PolyInception module. For order equal to `1` acts just like
        ResNet, order of `2` was used in original paper. Default: `2`
    """

    def __init__(self, module: torch.nn.Module, order: int=2):
        super().__init__()
        if order < 1:
            raise ValueError('Order of Poly cannot be less than 1.')
        self.module: torch.nn.Module = module
        self.order: int = order

    def extra_repr(self):
        return f'order={self.order},'

    def forward(self, inputs):
        outputs = [self.module(inputs)]
        for _ in range(1, self.order):
            outputs.append(self.module(outputs[-1]))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class MPoly(torch.nn.Module):
    """Apply multiple modules to input multiple times and sum.

    It's equation for `poly_modules` length equal to :math:`N` could be expressed by

    .. math::

        I + F_1 + F_1(F_0) + ... + F_N(F_{N-1}...F_0)

    where :math:`I` is identity and consecutive :math:`F_N` are consecutive modules
    applied to output of previous ones.

    Originally proposed by Xingcheng Zhang et al. in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    *poly_modules : torch.nn.Module
        Variable arg of modules to use. If empty, acts as an identity.
        For single module acts like `ResNet`. `2` was used in original paper.
        All modules need `inputs` and `outputs` of equal `shape`.

    """

    def __init__(self, *poly_modules: torch.nn.Module):
        super().__init__()
        self.poly_modules: torch.nn.Module = torch.nn.ModuleList(poly_modules)

    def forward(self, inputs):
        outputs = [self.poly_modules[0](inputs)]
        for module in self.poly_modules[1:]:
            outputs.append(module(outputs[-1]))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class WayPoly(torch.nn.Module):
    """Apply multiple modules to input and sum.

    It's equation for `poly_modules` length equal to :math:`N` could be expressed by

    .. math::

        I + F_1(I) + F_2(I) + ... + F_N

    where :math:`I` is identity and consecutive :math:`F_N` are consecutive `poly_modules`
    applied to input.

    Could be considered as an extension of standard `ResNet` to many parallel modules.

    Originally proposed by Xingcheng Zhang et al. in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    *poly_modules : torch.nn.Module
        Variable arg of modules to use. If empty, acts as an identity.
        For single module acts like `ResNet`. `2` was used in original paper.
        All modules need `inputs` and `outputs` of equal `shape`.
    """

    def __init__(self, *poly_modules: torch.nn.Module):
        super().__init__()
        self.poly_modules: torch.nn.Module = torch.nn.ModuleList(poly_modules)

    def forward(self, inputs):
        outputs = []
        for module in self.poly_modules:
            outputs.append(module(inputs))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class SqueezeExcitation(torch.nn.Module):
    """Learn channel-wise excitation maps for `inputs`.

    Provided `inputs` will be squeezed into `in_channels` via average pooling,
    passed through two non-linear layers, rescaled to :math:`[0, 1]` via `sigmoid`-like function
    and multiplied with original input channel-wise.

    Originally proposed by Xingcheng Zhang et al. in
    `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`__

    Example::


        import torchlayers as tl

        # Assume only 128 channels can be an input in this case
        block = tl.Residual(tl.Conv(128), tl.SqueezeExcitation(), tl.Conv(128))


    Parameters
    ----------
    in_channels : int
        Number of channels in the input
    hidden : int, optional
        Size of the hidden `torch.nn.Linear` layer. Usually smaller than `in_channels`
        (at least in original research paper). Default: `1/16` of `in_channels` as
        suggested by original paper.
    activation : Callable[[Tensor], Tensor], optional
        One argument callable performing activation after hidden layer.
        Default: `torch.nn.ReLU()`
    sigmoid : Callable[[Tensor], Tensor], optional
        One argument callable squashing values after excitation.
        Default: `torch.nn.Sigmoid`

    """

    def __init__(self, in_channels: int, hidden: int=None, activation=None,
        sigmoid=None):
        super().__init__()
        self.in_channels: int = in_channels
        self.hidden: int = hidden if hidden is not None else in_channels // 16
        self.activation: typing.Callable[[torch.Tensor], torch.Tensor
            ] = activation if activation is not None else torch.nn.ReLU()
        self.sigmoid: typing.Callable[[torch.Tensor], torch.Tensor
            ] = sigmoid if sigmoid is not None else torch.nn.Sigmoid()
        self._pooling = pooling.GlobalAvgPool()
        self._first = torch.nn.Linear(in_channels, self.hidden)
        self._second = torch.nn.Linear(self.hidden, in_channels)

    def forward(self, inputs):
        excitation = self.sigmoid(self._second(self.activation(self._first(
            self._pooling(inputs)))))
        return inputs * excitation.view(*excitation.shape, *([1] * (len(
            inputs.shape) - 2)))


class Fire(torch.nn.Module):
    """Squeeze and Expand number of channels efficiently operation-wise.

    First input channels will be squeezed to `hidden` channels and :math:`1 x 1` convolution.
    After that those will be expanded to `out_channels` partially done by :math:`3 x 3` convolution
    and partially by :math:`1 x 1` convolution (as specified by `p` parameter).

    Originally proposed by Forrest N. Iandola et al. in
    `SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size <https://arxiv.org/abs/1602.07360>`__

    Parameters
    ----------
    in_channels : int
        Number of channels in the input
    out_channels : int
        Number of channels produced by Fire module
    hidden_channels : int, optional
        Number of hidden channels (squeeze convolution layer).
        Default: `None` (half of `in_channels`)
    p : float, optional
        Ratio of :math:`1 x 1` convolution taken from total `out_channels`.
        The more, the more :math:`1 x 1` convolution will be used during expanding.
        Default: `0.5` (half of `out_channels`)

    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels
        =None, p: float=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            if in_channels >= 16:
                self.hidden_channels = in_channels // 2
            else:
                self.hidden_channels = 8
        else:
            self.hidden_channels = hidden_channels
        if not 0.0 < p < 1.0:
            raise ValueError("Fire's p has to be between 0 and 1, got {}".
                format(p))
        self.p: float = p
        self.squeeze = Conv(in_channels, self.hidden_channels, kernel_size=1)
        small_out_channels = int(out_channels * self.p)
        self.expand_small = Conv(self.hidden_channels, small_out_channels,
            kernel_size=1)
        self.expand_large = Conv(self.hidden_channels, out_channels -
            small_out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        squeeze = self.squeeze(inputs)
        return torch.cat((self.expand_small(squeeze), self.expand_large(
            squeeze)), dim=1)


class InvertedResidualBottleneck(torch.nn.Module):
    """Inverted residual block used in MobileNetV2, MNasNet, Efficient Net and other architectures.

    Originally proposed by Mark Sandler et al. in
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <0.5MB model size <https://arxiv.org/abs/1801.04381>`__

    Expanded with `SqueezeExcitation` after depthwise convolution by Mingxing Tan et al. in
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile <https://arxiv.org/abs/1807.11626>`__

    Due to it's customizable nature blocks from other research papers could be easily produced, e.g.
    `Searching for MobileNetV3 <https://arxiv.org/pdf/1905.02244.pdf>`__ by providing
    `torchlayers.HardSwish()` as `activation`, `torchlayers.HardSigmoid()` as `squeeze_excitation_activation`
    and `squeeze_excitation_hidden` equal to `hidden_channels // 4`.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input
    hidden_channels: int, optional
        Number of hidden channels (expanded). Should be greater than `in_channels`, usually
        by factor of `4`. Default: `in_channels * 4`
    activation: typing.Callable, optional
        One argument callable performing activation after hidden layer.
        Default: `torch.nn.ReLU6()`
    batchnorm: bool, optional
        Whether to apply Batch Normalization layer after initial convolution,
        depthwise expanding part (before squeeze excitation) and final squeeze.
        Default: `True`
    squeeze_excitation: bool, optional
        Whether to use standard `SqueezeExcitation` (see `SqueezeExcitation` module)
        after depthwise convolution.
        Default: `True`
    squeeze_excitation_hidden: int, optional
        Size of the hidden `torch.nn.Linear` layer. Usually smaller than `in_channels`
        (at least in original research paper). Default: `1/16` of `in_channels` as
        suggested by original paper.
    squeeze_excitation_activation: typing.Callable, optional
        One argument callable performing activation after hidden layer.
        Default: `torch.nn.ReLU()`
    squeeze_excitation_sigmoid: typing.Callable, optional
        One argument callable squashing values after excitation.
        Default: `torch.nn.Sigmoid`

    """

    def __init__(self, in_channels: int, hidden_channels: int=None,
        activation=None, batchnorm: bool=True, squeeze_excitation: bool=
        True, squeeze_excitation_hidden: int=None,
        squeeze_excitation_activation=None, squeeze_excitation_sigmoid=None):

        def _add_batchnorm(block, channels):
            if batchnorm:
                block.append(normalization.BatchNorm(channels))
            return block
        super().__init__()
        self.in_channels: int = in_channels
        self.hidden_channels: int = (hidden_channels if hidden_channels is not
            None else in_channels * 4)
        self.activation: typing.Callable[[torch.Tensor], torch.Tensor
            ] = torch.nn.ReLU6() if activation is None else activation
        self.batchnorm: bool = batchnorm
        self.squeeze_excitation: bool = squeeze_excitation
        self.squeeze_excitation_hidden: int = squeeze_excitation_hidden
        self.squeeze_excitation_activation: typing.Callable[[torch.Tensor],
            torch.Tensor] = squeeze_excitation_activation
        self.squeeze_excitation_sigmoid: typing.Callable[[torch.Tensor],
            torch.Tensor] = squeeze_excitation_sigmoid
        initial = torch.nn.Sequential(*_add_batchnorm([Conv(self.
            in_channels, self.hidden_channels, kernel_size=1), self.
            activation], self.hidden_channels))
        depthwise_modules = _add_batchnorm([Conv(self.hidden_channels, self
            .hidden_channels, kernel_size=3, groups=self.hidden_channels),
            self.activation], self.hidden_channels)
        if squeeze_excitation:
            depthwise_modules.append(SqueezeExcitation(self.hidden_channels,
                squeeze_excitation_hidden, squeeze_excitation_activation,
                squeeze_excitation_sigmoid))
        depthwise = torch.nn.Sequential(*depthwise_modules)
        squeeze = torch.nn.Sequential(*_add_batchnorm([Conv(self.
            hidden_channels, self.in_channels, kernel_size=1)], self.
            in_channels))
        self.block = Residual(torch.nn.Sequential(initial, depthwise, squeeze))

    def forward(self, inputs):
        return self.block(inputs)


class InferDimension(torch.nn.Module):
    """Infer dimensionality of module from input using dispatcher.

    Users can pass provide their own modules to infer dimensionality
    from input tensor by inheriting from this module and providing
    `super().__init__()` with `dispatcher` method::


        import torchlayers as tl

        class BatchNorm(tl.InferDimension):
            def __init__(
                self,
                num_features: int,
                eps: float = 1e-05,
                momentum: float = 0.1,
                affine: bool = True,
                track_running_stats: bool = True,
            ):
                super().__init__(
                    dispatcher={
                        # 5 dimensional tensor -> create torch.nn.BatchNorm3d
                        5: torch.nn.BatchNorm3d,
                        # 4 dimensional tensor -> create torch.nn.BatchNorm2d
                        4: torch.nn.BatchNorm2d,
                        3: torch.nn.BatchNorm1d,
                        2: torch.nn.BatchNorm1d,
                    },
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )

    All dimension-agnostic modules in `torchlayers` are created this way.
    This class can also be mixed with `torchlayers.infer` for dimensionality
    and shape inference in one.

    This class works correctly with `torchlayers.build` and other provided
    functionalities.

    Parameters
    ----------
    dispatcher: Dict[int, torch.nn.Module]
        Key should be length of input's tensor shape. Value should be a `torch.nn.Module`
        to be used for the dimensionality.
    initializer: Callable[[torch.nn.Module, torch.Tensor, **kwargs], torch.nn.Module], optional
        How to initialize dispatched module. Can be used to modify it's creation.
        First argument - dispatched module class, second - input tensor, **kwargs are
        arguments to use for module initialization. Should return module's instance.
        By default dispatched module initialized with **kwargs is returned.
    **kwargs:
        Arguments used to initialize dispatched module

    """

    def __init__(self, dispatcher: typing.Dict[int, torch.nn.Module],
        initializer: typing.Callable=None, **kwargs):
        super().__init__()
        self._dispatcher = dispatcher
        self._inner_module_name = '_inner_module'
        if initializer is None:
            self._initializer = (lambda dispatched_class, _, **kwargs:
                dispatched_class(**kwargs))
        else:
            self._initializer = initializer
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._noninferable_attributes = [key for key in kwargs]
        self._repr = _dev_utils.infer.create_repr(self._inner_module_name,
            **kwargs)
        self._reduce = _dev_utils.infer.create_reduce(self.
            _inner_module_name, self._noninferable_attributes)

    def __repr__(self):
        return self._repr(self)

    def __reduce__(self):
        return self._reduce(self)

    def forward(self, inputs):
        module = getattr(self, self._inner_module_name, None)
        if module is None:
            dimensionality = len(inputs.shape)
            dispatched_class = self._dispatcher.get(dimensionality)
            if dispatched_class is None:
                dispatched_class = self._dispatcher.get('*')
                if dispatched_class is None:
                    raise ValueError(
                        '{} could not be inferred from shape. Got tensor of dimensionality: {} but only {} are allowed'
                        .format(self._module_name, dimensionality, list(
                        self._dispatcher.keys())))
            self.add_module(self._inner_module_name, self._initializer(
                dispatched_class, inputs, **{key: getattr(self, key) for
                key in self._noninferable_attributes}))
        return getattr(self, self._inner_module_name)(inputs)


class GroupNorm(torch.nn.GroupNorm):
    """Apply Group Normalization over a mini-batch of inputs.

    Works exactly like PyTorch's counterpart, but `num_channels` is used as first argument
    so it can be inferred during first forward pass.

    Parameters
    ----------
    num_channels : int
        Number of channels expected in input
    num_groups : int
        Number of groups to separate the channels into
    eps : float, optional
        Value added to the denominator for numerical stability.
        Default: `1e-5`
    affine : bool, optional
        If ``True``, this module has learnable affine parameters.
        Default: ``True``

    """

    def __init__(self, num_channels: int, num_groups: int, eps: float=1e-05,
        affine: bool=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels,
            eps=eps, affine=affine)


class _GlobalPool(torch.nn.Module):

    def forward(self, inputs):
        return self._pooling(inputs).reshape(inputs.shape[0], -1)


class StochasticDepth(torch.nn.Module):
    """Randomly skip module during training with specified `p`, leaving inference untouched.

    Originally proposed by Gao Huang et al. in
    `Deep Networks with Stochastic Depth <www.arxiv.org/abs/1512.03385>`__.

    Originally devised as regularization, though `other research <https://web.stanford.edu/class/cs331b/2016/projects/kaplan_smith_jiang.pdf>`__  suggests:

    - "[...] StochasticDepth Nets are less tuned for low-level feature extraction but more tuned for higher level feature differentiation."
    - "[...] Stochasticity does not help with the ”dead neurons” problem; in fact the problem is actually more pronounced in the early layers. Nonetheless, the Stochastic Depth Network has relatively fewer dead neurons in later layers."

    It might be useful to employ this technique to layers closer to the bottleneck.

    Example::


        import torchlayers as tl

        # Assume only 128 channels can be an input in this case
        block = tl.StochasticDepth(tl.Conv(128), p=0.3)
        # May skip tl.Conv with 0.3 probability
        block(torch.randn(1, 3, 32, 32))

    Parameters
    ----------
    module : torch.nn.Module
        Any module whose output might be skipped
        (output shape of it has to be equal to the shape of inputs).
    p : float, optional
        Probability of survival (e.g. the layer will be kept). Default: ``0.5``

    """

    def __init__(self, module: torch.nn.Module, p: float=0.5):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                'Stochastic Depth p has to be between 0 and 1 but got {}'.
                format(p))
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_():
            return inputs
        return self.p * self.module(inputs)


class StandardNormalNoise(torch.nn.Module):
    """Add noise from standard normal distribution during forward pass.

    .. note::
            Changes input only if `module` is in `train` mode.

    Example::


        import torchlayers as tl

        model = tl.Sequential(
            tl.StandardNormalNoise(), tl.Linear(10), tl.ReLU(), tl.tl.Linear(1)
        )
        tl.build(model, torch.randn(3, 30))

        # Noise from Standard Normal distribution will be added
        model(torch.randn(3, 30))

        model.eval()
        # Eval mode, no noise added
        model(torch.randn(3, 30))

    """

    def forward(self, inputs):
        if self.training:
            return inputs + torch.randn_like(inputs)
        return inputs


class UniformNoise(torch.nn.Module):
    """Add noise from uniform `[0, 1)` distribution during forward pass.

    .. note::
            Changes input only if `module` is in `train` mode.

    Example::


        import torchlayers as tl

        noisy_linear_regression = tl.Sequential(
            tl.UniformNoise(), tl.Linear(1)
        )
        tl.build(model, torch.randn(1, 10))

        # Noise from Uniform distribution will be added
        model(torch.randn(64, 10))

        model.eval()
        # Eval mode, no noise added
        model(torch.randn(64, 10))

    """

    def forward(self, inputs):
        if self.training:
            return inputs + torch.rand_like(inputs)
        return inputs


class WeightDecay(torch.nn.Module):

    def __init__(self, module, weight_decay, name: str=None):
        if weight_decay <= 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}"
                .format(weight_decay))
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name
        self.hook = self.module.register_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (param.grad is None or torch.all(
                    param.grad == 0.0)):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) ->str:
        representation = 'weight_decay={}'.format(self.weight_decay)
        if self.name is not None:
            representation += ', name={}'.format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class ConvPixelShuffle(torch.nn.Module):
    """Two dimensional convolution with ICNR initialization followed by PixelShuffle.

    Increases `height` and `width` of `input` tensor by scale, acts like
    learnable upsampling. Due to `ICNR weight initialization <https://arxiv.org/abs/1707.02937>`__
    of `convolution` it has similar starting point to nearest neighbour upsampling.

    `kernel_size` got a default value of `3`, `upscale_factor` got a default
    value of `2`.

    Example::

        import torchlayers as tl


        class MiniAutoEncoder(tl.Module):
            def __init__(self, out_channels):
                super().__init__()
                self.conv1 = tl.Conv(64)
                # Twice smaller image by default
                self.pooling = tl.MaxPool()
                # Twice larger (upscale_factor=2) by default
                self.upsample = tl.ConvPixelShuffle(out_channels)

            def forward(self, x):
                x = self.conv1(x)
                pooled = self.pooling(x)
                return self.upsample(pooled)


        out_channels = 3
        network = MiniAutoEncoder(out_channels)
        tl.build(network, torch.randn(1, out_channels, 64, 64))
        assert network(torch.randn(5, out_channels, 64, 64)).shape == [5, out_channels, 64, 64]

    .. note::

        Currently only `4D` input is allowed (`[batch, channels, height, width]`),
        due to `torch.nn.PixelShuffle` not supporting `3D` or `5D` inputs.
        See [this PyTorch PR](https://github.com/pytorch/pytorch/pull/6340/files)
        for example of dimension-agnostic implementation.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced after PixelShuffle
    upscale_factor : int, optional
        Factor to increase spatial resolution by. Default: `2`
    kernel_size : int or tuple, optional
        Size of the convolving kernel. Default: `3`
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
    initializer: typing.Callable[[torch.Tensor,], torch.Tensor], optional
        Initializer for ICNR initialization, can be a function from `torch.nn.init`.
        Receive tensor as argument and returns tensor after initialization.
        Default: `torch.nn.init.kaiming_normal_`

    """

    def __init__(self, in_channels, out_channels, upscale_factor: int=2,
        kernel_size: int=3, stride: int=1, padding='same', dilation: int=1,
        groups: int=1, bias: bool=True, padding_mode: str='zeros',
        initializer=None):
        super().__init__()
        self.convolution = convolution.Conv(in_channels, out_channels *
            upscale_factor * upscale_factor, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        self.upsample = torch.nn.PixelShuffle(upscale_factor)
        if initializer is None:
            self.initializer = torch.nn.init.kaiming_normal_
        else:
            self.initializer = initializer

    def post_build(self):
        """Initialize weights after layer was built."""
        self.icnr_initialization(self.convolution.weight.data)

    def icnr_initialization(self, tensor):
        """ICNR initializer for checkerboard artifact free sub pixel convolution.

        Originally presented in
        `Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize <https://arxiv.org/abs/1707.02937>`__
        Initializes convolutional layer prior to `torch.nn.PixelShuffle`.
        Weights are initialized according to `initializer` passed to to `__init__`.

        Parameters
        ----------
        tensor: torch.Tensor
                Tensor to be initialized using ICNR init.

        Returns
        -------
        torch.Tensor
                Tensor initialized using ICNR.

        """
        if self.upsample.upscale_factor == 1:
            return self.initializer(tensor)
        new_shape = [int(tensor.shape[0] / self.upsample.upscale_factor ** 2)
            ] + list(tensor.shape[1:])
        subkernel = self.initializer(torch.zeros(new_shape)).transpose(0, 1)
        kernel = subkernel.reshape(subkernel.shape[0], subkernel.shape[1], -1
            ).repeat(1, 1, self.upsample.upscale_factor ** 2)
        return kernel.reshape([-1, tensor.shape[0]] + list(tensor.shape[2:])
            ).transpose(0, 1)

    def forward(self, inputs):
        return self.upsample(self.convolution(inputs))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_szymonmaszke_torchlayers(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConcatenateProxy(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(GroupNorm(*[], **{'num_channels': 4, 'num_groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(HardSigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(HardSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Lambda(*[], **{'function': _mock_layer()}), [], {'input': torch.rand([4, 4])})

    def test_006(self):
        self._check(Poly(*[], **{'module': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Reshape(*[], **{}), [torch.rand([4])], {})

    def test_008(self):
        self._check(Residual(*[], **{'module': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(StandardNormalNoise(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(StochasticDepth(*[], **{'module': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(UniformNoise(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(WayPoly(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_014(self):
        self._check(WeightDecay(*[], **{'module': _mock_layer(), 'weight_decay': 4}), [], {'input': torch.rand([4, 4])})

    def test_015(self):
        self._check(_CustomLinearImpl(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

