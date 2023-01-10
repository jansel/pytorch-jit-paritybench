import sys
_module = sys.modules[__name__]
del sys
demo_losses = _module
pytorch_toolbelt = _module
datasets = _module
classification = _module
common = _module
mean_std = _module
providers = _module
inria_aerial = _module
segmentation = _module
wrappers = _module
inference = _module
ensembling = _module
functional = _module
tiles = _module
tiles_3d = _module
tta = _module
losses = _module
balanced_bce = _module
bitempered_loss = _module
dice = _module
focal = _module
focal_cosine = _module
functional = _module
jaccard = _module
joint_loss = _module
lovasz = _module
soft_bce = _module
soft_ce = _module
soft_f1 = _module
wing_loss = _module
modules = _module
activations = _module
backbone = _module
inceptionv4 = _module
mobilenet = _module
senet = _module
wider_resnet = _module
coord_conv = _module
decoders = _module
bifpn = _module
can = _module
common = _module
deeplab = _module
fpn_cat = _module
fpn_sum = _module
hrnet = _module
pyramid_pooling = _module
unet = _module
unet_v2 = _module
upernet = _module
dropblock = _module
dsconv = _module
encoders = _module
common = _module
densenet = _module
hourglass = _module
hrnet = _module
inception = _module
resnet = _module
seresnet = _module
squeezenet = _module
swin = _module
timm = _module
common = _module
dpn = _module
efficient_net = _module
efficient_net_v2 = _module
nf_regnet = _module
nfnet = _module
res2net = _module
resnet = _module
unet = _module
wide_resnet = _module
xresnet = _module
fpn = _module
hypercolumn = _module
identity = _module
normalize = _module
ocnet = _module
pooling = _module
scse = _module
simple = _module
srm = _module
unet = _module
upsample = _module
optimization = _module
functional = _module
lr_schedules = _module
utils = _module
bboxes_utils = _module
distributed = _module
fs = _module
namesgenerator = _module
python_utils = _module
random = _module
rle = _module
support = _module
torch_utils = _module
visualization = _module
zoo = _module
segmentation = _module
setup = _module
test_activations = _module
test_decoders = _module
test_encoders = _module
test_filesystem_utils = _module
test_losses = _module
test_model_export = _module
test_models_zoo = _module
test_modules = _module
test_tiles = _module
test_tta = _module
test_utils_functional = _module
test_visualization = _module

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


from torch.nn import BCEWithLogitsLoss


import numpy as np


import torch


import matplotlib.pyplot as plt


from typing import Optional


from typing import List


from typing import Union


from typing import Callable


from torch.utils.data import Dataset


import warnings


from typing import Tuple


import pandas as pd


from sklearn.model_selection import GroupKFold


from functools import partial


import random


from typing import Any


from torch.utils.data.dataloader import default_collate


from torch import nn


from torch import Tensor


from typing import Iterable


from typing import Dict


from collections.abc import Sized


from collections.abc import Iterable


import math


import typing


from collections import defaultdict


from typing import Mapping


import torch.nn.functional as F


from torch.nn.modules.loss import _Loss


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import functional as F


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.utils import model_zoo


from typing import Type


import inspect


from torchvision.models import densenet121


from torchvision.models import densenet161


from torchvision.models import densenet169


from torchvision.models import densenet201


from torchvision.models import DenseNet


from torchvision.models import resnet50


from torchvision.models import resnet34


from torchvision.models import resnet18


from torchvision.models import resnet101


from torchvision.models import resnet152


from torchvision.models import squeezenet1_1


import torch.utils.checkpoint as checkpoint


from torch.nn.modules.module import _IncompatibleKeys


import torch.nn.init


from math import hypot


from typing import Iterator


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from collections import namedtuple


from torchvision.ops import box_iou


import torch.distributed as dist


import collections


from typing import Sequence


from torch.utils.data import ConcatDataset


import re


from torch.utils.data import DataLoader


class ApplySoftmaxTo(nn.Module):
    output_keys: Tuple
    temperature: float
    dim: int

    def __init__(self, model: nn.Module, output_key: Union[str, int, Iterable[str]]='logits', dim: int=1, temperature: float=1):
        """
        Apply softmax activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string, index or list of strings, indicating to what outputs softmax activation should be applied.
        :param dim: Tensor dimension for softmax activation
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.dim = dim
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).softmax(dim=self.dim)
        return output


class ApplySigmoidTo(nn.Module):
    output_keys: Tuple
    temperature: float

    def __init__(self, model: nn.Module, output_key: Union[str, int, Iterable[str]]='logits', temperature=1):
        """
        Apply sigmoid activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string index, or list of strings, indicating to what outputs sigmoid activation should be applied.
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).sigmoid_()
        return output


MaybeStrOrCallable = Optional[Union[str, Callable]]


def _deaugment_averaging(x: Tensor, reduction: MaybeStrOrCallable) ->Tensor:
    """
    Average predictions of TTA-ed model.
    This function assumes TTA dimension is 0, e.g [T, B, C, Ci, Cj, ..]

    Args:
        x: Input tensor of shape [T, B, ... ]
        reduction: Reduction mode ("sum", "mean", "gmean", "hmean", function, None)

    Returns:
        Tensor of shape [B, C, Ci, Cj, ..]
    """
    if reduction == 'mean':
        x = x.mean(dim=0)
    elif reduction == 'sum':
        x = x.sum(dim=0)
    elif reduction in {'gmean', 'geometric_mean'}:
        x = F.geometric_mean(x, dim=0)
    elif reduction in {'hmean', 'harmonic_mean'}:
        x = F.harmonic_mean(x, dim=0)
    elif reduction == 'logodd':
        x = F.logodd_mean(x, dim=0)
    elif callable(reduction):
        x = reduction(x, dim=0)
    elif reduction in {None, 'None', 'none'}:
        pass
    else:
        raise KeyError(f'Unsupported reduction mode {reduction}')
    return x


class Ensembler(nn.Module):
    __slots__ = ['outputs', 'reduction', 'return_some_outputs']
    """
    Compute sum (or average) of outputs of several models.
    """

    def __init__(self, models: List[nn.Module], reduction: str='mean', outputs: Optional[Iterable[str]]=None):
        """

        :param models:
        :param reduction: Reduction key ('mean', 'sum', 'gmean', 'hmean' or None)
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.return_some_outputs = outputs is not None
        self.outputs = tuple(outputs) if outputs else tuple()
        self.models = nn.ModuleList(models)
        self.reduction = reduction

    def forward(self, *input, **kwargs):
        outputs = [model(*input, **kwargs) for model in self.models]
        if self.return_some_outputs:
            keys = self.outputs
        elif isinstance(outputs[0], dict):
            keys = outputs[0].keys()
        elif torch.is_tensor(outputs[0]):
            keys = None
        else:
            raise RuntimeError()
        if keys is None:
            predictions = torch.stack(outputs)
            predictions = _deaugment_averaging(predictions, self.reduction)
            averaged_output = predictions
        else:
            averaged_output = {}
            for key in keys:
                predictions = [output[key] for output in outputs]
                predictions = torch.stack(predictions)
                predictions = _deaugment_averaging(predictions, self.reduction)
                averaged_output[key] = predictions
        return averaged_output


class PickModelOutput(nn.Module):
    """
    Wraps a model that returns dict or list and returns only a specific element.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(PickModelOutput(model, "OUTPUT_MASK")), nn.Sigmoid())
    """
    __slots__ = ['target_key']

    def __init__(self, model: nn.Module, key: Union[str, int]):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, *input, **kwargs) ->Tensor:
        output = self.model(*input, **kwargs)
        return output[self.target_key]


class SelectByIndex(nn.Module):
    """
    Select a single Tensor from the dict or list of output tensors.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(model, SelectByIndex("OUTPUT_MASK"), nn.Sigmoid())
    """
    __slots__ = ['target_key']

    def __init__(self, key: Union[str, int]):
        super().__init__()
        self.target_key = key

    def forward(self, outputs: Dict[str, Tensor]) ->Tensor:
        return outputs[self.target_key]


string_types = type(b''), type('')


def pytorch_toolbelt_deprecated(reason):
    """
    Mark function or class as deprecated.
    It will result in a warning being emitted when the function is used.
    """
    if isinstance(reason, string_types):

        def decorator(func1):
            if inspect.isclass(func1):
                fmt1 = 'Call to deprecated class {name} ({reason}).'
            else:
                fmt1 = 'Call to deprecated function {name} ({reason}).'

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(fmt1.format(name=func1.__name__, reason=reason), category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason
        if inspect.isclass(func2):
            fmt2 = 'Call to deprecated class {name}.'
        else:
            fmt2 = 'Call to deprecated function {name}.'

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2
    else:
        raise TypeError(repr(type(reason)))


@pytorch_toolbelt_deprecated('This class is deprecated. Please use GeneralizedTTA instead')
class TTAWrapper(nn.Module):

    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


class GeneralizedTTA(nn.Module):
    __slots__ = ['augment_fn', 'deaugment_fn']
    """
    Example:
        tta_model = GeneralizedTTA(model,
            augment_fn=tta.d2_image_augment,
            deaugment_fn={
                OUTPUT_MASK_KEY: tta.d2_image_deaugment,
                OUTPUT_EDGE_KEY: tta.d2_image_deaugment,
            },


    Notes:
        Input tensors must be square for D2/D4 or similar types of augmentation
    """

    def __init__(self, model: Union[nn.Module, nn.DataParallel], augment_fn: Union[Callable, Dict[str, Callable], List[Callable]], deaugment_fn: Union[Callable, Dict[str, Callable], List[Callable]]):
        super().__init__()
        self.model = model
        self.augment_fn = augment_fn
        self.deaugment_fn = deaugment_fn

    def forward(self, *input, **kwargs):
        if isinstance(self.augment_fn, dict):
            if len(input) != 0:
                raise ValueError('Input for GeneralizedTTA must not have positional arguments when augment_fn is dictionary')
            augmented_inputs = dict((key, augment(kwargs[key])) for key, augment in self.augment_fn.items())
            outputs = self.model(**augmented_inputs)
        elif isinstance(self.augment_fn, (list, tuple)):
            if len(kwargs) != 0:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            augmented_inputs = [augment(x) for x, augment in zip(input, self.augment_fn)]
            outputs = self.model(*augmented_inputs)
        else:
            if len(input) != 1:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            if len(kwargs) != 0:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            augmented_input = self.augment_fn(input[0])
            outputs = self.model(augmented_input)
        if isinstance(self.deaugment_fn, dict):
            if not isinstance(outputs, dict):
                raise ValueError('Output of the model must be a dict')
            deaugmented_output = dict((key, self.deaugment_fn[key](outputs[key])) for key in self.deaugment_fn.keys())
        elif isinstance(self.deaugment_fn, (list, tuple)):
            if not isinstance(outputs, (dict, tuple)):
                raise ValueError('Output of the model must be a dict')
            deaugmented_output = [deaugment(value) for value, deaugment in zip(outputs, self.deaugment_fn)]
        else:
            deaugmented_output = self.deaugment_fn(outputs)
        return deaugmented_output


def ms_image_augment(image: Tensor, size_offsets: List[Union[int, Tuple[int, int]]], mode='bilinear', align_corners=False) ->List[Tensor]:
    """
    Multi-scale image augmentation. This function create list of resized tensors from the input one.
    """
    batch_size, channels, rows, cols = image.size()
    augmented_inputs = []
    for offset in size_offsets:
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset
        if rows_offset == 0 and cols_offset == 0:
            augmented_inputs.append(image)
        else:
            scale_size = rows + rows_offset, cols + cols_offset
            scaled_input = torch.nn.functional.interpolate(image, size=scale_size, mode=mode, align_corners=align_corners)
            augmented_inputs.append(scaled_input)
    return augmented_inputs


def ms_image_deaugment(images: List[Tensor], size_offsets: List[Union[int, Tuple[int, int]]], reduction: MaybeStrOrCallable='mean', mode: str='bilinear', align_corners: bool=True, stride: int=1) ->Tensor:
    """
    Perform multi-scale deaugmentation of predicted feature maps.

    Args:
        images: List of tensors of shape [B, C, Hi, Wi], [B, C, Hj, Wj], [B, C, Hk, Wk]
        size_offsets:
        reduction:
        mode:
        align_corners:
        stride: Stride of the output feature map w.r.t to model input size.
        Used to correctly scale size_offsets to match with size of output feature maps

    Returns:
        Averaged feature-map of the original size
    """
    if len(images) != len(size_offsets):
        raise ValueError('Number of images must be equal to number of size offsets')
    deaugmented_outputs = []
    for feature_map, offset in zip(images, size_offsets):
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset
        if rows_offset == 0 and cols_offset == 0:
            deaugmented_outputs.append(feature_map)
        else:
            batch_size, channels, rows, cols = feature_map.size()
            original_size = rows - rows_offset // stride, cols - cols_offset // stride
            scaled_image = torch.nn.functional.interpolate(feature_map, size=original_size, mode=mode, align_corners=align_corners)
            deaugmented_outputs.append(scaled_image)
    deaugmented_outputs = torch.stack(deaugmented_outputs)
    return _deaugment_averaging(deaugmented_outputs, reduction=reduction)


class MultiscaleTTA(nn.Module):

    def __init__(self, model: nn.Module, size_offsets: List[int], deaugment_fn: Union[Callable, Dict[str, Callable]]=ms_image_deaugment):
        if isinstance(deaugment_fn, Mapping):
            self.keys = set(deaugment_fn.keys())
        else:
            self.keys = None
        super().__init__()
        self.model = model
        self.size_offsets = size_offsets
        self.deaugment_fn = deaugment_fn

    def forward(self, x):
        ms_inputs = ms_image_augment(x, size_offsets=self.size_offsets)
        ms_outputs = [self.model(x) for x in ms_inputs]
        outputs = {}
        if self.keys is None:
            outputs = self.deaugment_fn(ms_outputs, self.size_offsets)
        else:
            keys = self.keys
            for key in keys:
                deaugment_fn: Callable = self.deaugment_fn[key]
                values = [x[key] for x in ms_outputs]
                outputs[key] = deaugment_fn(values, self.size_offsets)
        return outputs


def balanced_binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor, gamma: float=1.0, ignore_index: Optional[int]=None, reduction: str='mean') ->Tensor:
    """
    Balanced binary cross entropy loss.

    Args:
        logits:
        targets: This loss function expects target values to be hard targets 0/1.
        gamma: Power factor for balancing weights
        ignore_index:
        reduction:

    Returns:
        Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
        shape of `logits` tensor.
    """
    pos_targets: Tensor = targets.eq(1).sum()
    neg_targets: Tensor = targets.eq(0).sum()
    num_targets = pos_targets + neg_targets
    pos_weight = torch.pow(neg_targets / (num_targets + 1e-07), gamma)
    neg_weight = 1.0 - pos_weight
    pos_term = pos_weight.pow(gamma) * targets * torch.nn.functional.logsigmoid(logits)
    neg_term = neg_weight.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-logits)
    loss = -(pos_term + neg_term)
    if ignore_index is not None:
        loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    return loss


class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced binary cross-entropy loss.

    https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
    """
    __constants__ = ['gamma', 'reduction', 'ignore_index']

    def __init__(self, gamma: float=1.0, reduction='mean', ignore_index: Optional[int]=None):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output: Tensor, target: Tensor) ->Tensor:
        return balanced_binary_cross_entropy_with_logits(output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction)


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_binary_search(activations: Tensor, t: float, num_iters: int) ->Tensor:
    """Compute normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    effective_dim = torch.sum((normalized_activations > -1.0 / (1.0 - t)).to(torch.int32), dim=-1, keepdim=True)
    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)
    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True)
        update = sum_probs < 1.0
        lower = torch.reshape(lower * update + (1.0 - update) * logt_partition, shape_partition)
        upper = torch.reshape(upper * (1.0 - update) + update * logt_partition, shape_partition)
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


def compute_normalization_fixed_point(activations: Tensor, t: float, num_iters: int) ->Tensor:
    """Return the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - t)
    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu
    return normalization_constants


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Compute normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction='mean'):
    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels
    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (1 - label_smoothing * num_classes / (num_classes - 1)) * labels_onehot + label_smoothing / (num_classes - 1)
    probabilities = tempered_softmax(activations, t2, num_iters)
    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) - labels_onehot * log_t(probabilities, t1) - labels_onehot.pow(2.0 - t1) / (2.0 - t1) + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim=-1)
    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


class BiTemperedLogisticLoss(nn.Module):
    """

    https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
    https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: float, t2: float, smoothing=0.0, ignore_index=None, reduction: str='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: Tensor, targets: Tensor) ->Tensor:
        loss = bi_tempered_logistic_loss(predictions, targets, t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none')
        if self.ignore_index is not None:
            mask = ~targets.eq(self.ignore_index)
            loss *= mask
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryBiTemperedLogisticLoss(nn.Module):
    """
    Modification of BiTemperedLogisticLoss for binary classification case.
    It's signature matches nn.BCEWithLogitsLoss: Predictions and target tensors must have shape [B,1,...]

    References:
        https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
        https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: float, t2: float, smoothing: float=0.0, ignore_index: Optional[int]=None, reduction: str='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: Tensor, targets: Tensor) ->Tensor:
        """
        Forward method of the loss function

        Args:
            predictions: [B,1,...]
            targets: [B,1,...]

        Returns:
            Zero-sized tensor with reduced loss if self.reduction is `sum` or `mean`; Otherwise returns loss of the
            shape of `predictions` tensor.
        """
        if predictions.size(1) != 1 or targets.size(1) != 1:
            raise ValueError('Channel dimension for predictions and targets must be equal to 1')
        loss = bi_tempered_logistic_loss(torch.cat([-predictions, predictions], dim=1).moveaxis(1, -1), torch.cat([1 - targets, targets], dim=1).moveaxis(1, -1), t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none').unsqueeze(dim=1)
        if self.ignore_index is not None:
            mask = targets.eq(self.ignore_index)
            loss = torch.masked_fill(loss, mask, 0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


BINARY_MODE = 'binary'


MULTICLASS_MODE = 'multiclass'


MULTILABEL_MODE = 'multilabel'


def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float=0.0, eps: float=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) ->torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {'O', 'M', 'U', 'S'}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    raise ValueError('Unsupported input type' + str(type(x)))


class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(self, mode: str, classes: List[int]=None, log_loss=False, from_logits=True, smooth: float=0.0, ignore_index=None, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot(y_true * mask, num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


def focal_loss_with_logits(output: torch.Tensor, target: torch.Tensor, gamma: float=2.0, alpha: Optional[float]=0.25, reduction: str='mean', normalized: bool=False, reduced_threshold: Optional[float]=None, eps: float=1e-06, ignore_index=None, activation: str='sigmoid', softmax_dim: Optional[int]=None) ->torch.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
        activation: Either sigmoid or softmax. If `softmax` is used, `softmax_dim` must be also specified.

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type_as(output)
    if activation == 'sigmoid':
        p = torch.sigmoid(output)
    else:
        p = torch.softmax(output, dim=softmax_dim)
    ce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term = torch.masked_fill(focal_term, pt < reduced_threshold, 1)
    loss = focal_term * ce_loss
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if ignore_index is not None:
        ignore_mask = target.eq(ignore_index)
        loss = torch.masked_fill(loss, ignore_mask, 0)
        if normalized:
            focal_term = torch.masked_fill(focal_term, ignore_mask, 0)
    if normalized:
        norm_factor = focal_term.sum(dtype=torch.float32).clamp_min(eps)
        loss /= norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum(dtype=torch.float32)
    if reduction == 'batchwise_mean':
        loss = loss.sum(dim=0, dtype=torch.float32)
    return loss


class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha: Optional[float]=None, gamma: float=2.0, ignore_index: Optional[int]=None, reduction: str='mean', normalized: bool=False, reduced_threshold: Optional[float]=None, activation: str='sigmoid', softmax_dim: Optional[int]=None):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strength).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param activation: Either `sigmoid` or `softmax`. If `softmax` is used, `softmax_dim` must be also specified.

        """
        super().__init__()
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized, ignore_index=ignore_index, activation=activation, softmax_dim=softmax_dim)

    def forward(self, inputs: Tensor, targets: Tensor) ->Tensor:
        """Compute focal loss for binary classification problem."""
        loss = self.focal_loss_fn(inputs, targets)
        return loss


def softmax_focal_loss_with_logits(output: torch.Tensor, target: torch.Tensor, gamma: float=2.0, reduction: str='mean', normalized: bool=False, reduced_threshold: Optional[float]=None, eps: float=1e-06, ignore_index: int=-100) ->torch.Tensor:
    """
    Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        gamma: Focal loss power factor
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    log_softmax = F.log_softmax(output, dim=1)
    loss = F.nll_loss(log_softmax, target, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-loss)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * loss
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


class CrossEntropyFocalLoss(nn.Module):
    """
    Focal loss for multi-class problem. It uses softmax to compute focal term instead of sigmoid as in
    original paper. This loss expects target labes to have one dimension less (like in nn.CrossEntropyLoss).

    """

    def __init__(self, gamma: float=2.0, reduction: str='mean', normalized: bool=False, reduced_threshold: Optional[float]=None, ignore_index: int=-100):
        """

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.reduced_threshold = reduced_threshold
        self.normalized = normalized
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) ->Tensor:
        return softmax_focal_loss_with_logits(inputs, targets, gamma=self.gamma, reduction=self.reduction, normalized=self.normalized, reduced_threshold=self.reduced_threshold, ignore_index=self.ignore_index)


class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).

    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: float=1, gamma: float=2, xent: float=0.1, reduction='mean'):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        cosine_loss = F.cosine_embedding_loss(input, torch.nn.functional.one_hot(target, num_classes=input.size(-1)), torch.tensor([1], device=target.device), reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction='none')
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss


def soft_jaccard_score(output: torch.Tensor, target: torch.Tensor, smooth: float=0.0, eps: float=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :param dims:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


class JaccardLoss(_Loss):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multi-class and multi-label cases.
    """

    def __init__(self, mode: str, classes: List[int]=None, log_loss=False, from_logits=True, smooth=0, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true = F.one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask.float()
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


def _flatten_binary_scores(scores, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return scores, labels
    valid = labels != ignore_index
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore_index)) for log, lab in zip(logits, labels))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore_index))
    return loss


class BinaryLovaszLoss(_Loss):

    def __init__(self, per_image: bool=False, ignore_index: Optional[Union[int, float]]=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)
    probas = probas.contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _lovasz_softmax(probas, labels, classes='present', per_image=False, ignore_index=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore_index: void class labels
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore_index), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore_index), classes=classes)
    return loss


class LovaszLoss(_Loss):

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """
    __constants__ = ['weight', 'pos_weight', 'reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, weight=None, ignore_index: Optional[int]=-100, reduction='mean', smooth_factor=None, pos_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)
        loss = F.binary_cross_entropy_with_logits(input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask: Tensor = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def label_smoothed_nll_loss(lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction='mean', dim=-1) ->torch.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)
    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """
    __constants__ = ['reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, reduction: str='mean', smooth_factor: float=0.0, ignore_index: Optional[int]=-100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(log_prob, target, epsilon=self.smooth_factor, ignore_index=self.ignore_index, reduction=self.reduction, dim=self.dim)


def soft_micro_f1(preds: Tensor, targets: Tensor, eps=1e-06) ->Tensor:
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        targets (Tensor): targets array of shape (Num Samples, Num Classes)
        preds (Tensor): probability matrix of shape (Num Samples, Num Classes)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    References:
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """
    tp = torch.sum(preds * targets, dim=0)
    fp = torch.sum(preds * (1 - targets), dim=0)
    fn = torch.sum((1 - preds) * targets, dim=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - soft_f1
    return loss.mean()


class BinarySoftF1Loss(nn.Module):

    def __init__(self, ignore_index: Optional[int]=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor) ->Tensor:
        targets = targets.view(-1)
        preds = preds.view(-1)
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        preds = preds.sigmoid().clamp(self.eps, 1 - self.eps)
        return soft_micro_f1(preds.view(-1, 1), targets.view(-1, 1))


class SoftF1Loss(nn.Module):

    def __init__(self, ignore_index: Optional[int]=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor) ->Tensor:
        preds = preds.softmax(dim=1).clamp(self.eps, 1 - self.eps)
        targets = torch.nn.functional.one_hot(targets, preds.size(1))
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        return soft_micro_f1(preds, targets)


class WingLoss(_Loss):

    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(prediction, target, self.width, self.curvature, self.reduction)


def mish_naive(input):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return input * torch.tanh(F.softplus(input))


class MishNaive(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return mish_naive(x)


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


class MishFunction(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish(x):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return MishFunction.apply(x)


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Credit: https://github.com/digantamisra98/Mish
    """

    def __init__(self, inplace=False):
        """
        Init method.
        :param inplace: Not used, exists only for compatibility
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


class HardSigmoid(nn.Module):

    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


def swish_naive(x):
    return x * x.sigmoid()


class SwishNaive(nn.Module):

    def forward(self, input_tensor):
        return swish_naive(input_tensor)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.

    Credit:
        https://blog.ceshine.net/post/pytorch-memory-swish/
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations_jit.py

    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x):
    return SwishFunction.apply(x)


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()

    def forward(self, input_tensor):
        return swish(input_tensor)


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, activation):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, activation):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), activation())


def conv_bn(inp, oup, stride, activation):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), activation())


ACT_CELU = 'celu'


ACT_ELU = 'elu'


ACT_GELU = 'gelu'


ACT_GLU = 'glu'


ACT_HARD_SIGMOID = 'hard_sigmoid'


ACT_HARD_SWISH = 'hard_swish'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_MISH = 'mish'


ACT_MISH_NAIVE = 'mish_naive'


ACT_NONE = 'none'


ACT_PRELU = 'prelu'


ACT_RELU = 'relu'


ACT_RELU6 = 'relu6'


ACT_SELU = 'selu'


ACT_SIGMOID = 'sigmoid'


ACT_SILU = 'silu'


ACT_SOFTPLUS = 'softplus'


ACT_SWISH = 'swish'


ACT_SWISH_NAIVE = 'swish_naive'


class Identity(nn.Module):
    """The most useful module. A pass-through module which does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_activation_block(activation_name: str):
    ACTIVATIONS = {ACT_CELU: nn.CELU, ACT_ELU: nn.ELU, ACT_GELU: nn.GELU, ACT_GLU: nn.GLU, ACT_HARD_SIGMOID: HardSigmoid, ACT_HARD_SWISH: HardSwish, ACT_LEAKY_RELU: nn.LeakyReLU, ACT_MISH: Mish, ACT_MISH_NAIVE: MishNaive, ACT_NONE: Identity, ACT_PRELU: nn.PReLU, ACT_RELU6: nn.ReLU6, ACT_RELU: nn.ReLU, ACT_SELU: nn.SELU, ACT_SILU: nn.SiLU, ACT_SOFTPLUS: nn.Softplus, ACT_SWISH: Swish, ACT_SWISH_NAIVE: SwishNaive, ACT_SIGMOID: nn.Sigmoid}
    return ACTIVATIONS[activation_name.lower()]


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0, activation='relu6'):
        super(MobileNetV2, self).__init__()
        activation_block = get_activation_block(activation)
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.layer0 = conv_bn(3, input_channel, 2, activation_block)
        for layer_index, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)
            blocks = []
            for i in range(n):
                if i == 0:
                    blocks.append(block(input_channel, output_channel, s, expand_ratio=t, activation=activation_block))
                else:
                    blocks.append(block(input_channel, output_channel, 1, expand_ratio=t, activation=activation_block))
                input_channel = output_channel
            self.add_module(f'layer{layer_index + 1}', nn.Sequential(*blocks))
        self.final_layer = conv_1x1_bn(input_channel, self.last_channel, activation=activation_block)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.final_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def instantiate_activation_block(activation_name: str, **kwargs) ->nn.Module:
    block = get_activation_block(activation_name)
    act_params = {}
    if 'inplace' in kwargs and activation_name in {ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_SELU, ACT_SILU, ACT_CELU, ACT_ELU}:
        act_params['inplace'] = kwargs['inplace']
    if 'slope' in kwargs and activation_name in {ACT_LEAKY_RELU}:
        act_params['negative_slope'] = kwargs['slope']
    if activation_name == ACT_PRELU:
        act_params['num_parameters'] = kwargs['num_parameters']
    return block(**act_params)


def ABN(num_features: int, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, activation=ACT_RELU, slope=0.01, inplace=True):
    bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    act = instantiate_activation_block(activation, inplace=inplace, slope=slope)
    return nn.Sequential(OrderedDict([('bn', bn), (activation, act)]))


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, dropout=None):
        """Identity-mapping residual block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class GlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class WiderResNet(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act)))
                in_channels = channels[mod_id][-1]
            if mod_id <= 4:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            out = self.classifier(out)
        return out


class WiderResNetA2(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0, dilation=False):
        """Wider ResNet with pre-activation (identity mapping) blocks.
        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            return self.classifier(out)
        else:
            return out


def append_coords(input_tensor, with_r=False):
    batch_size, _, x_dim, y_dim = input_tensor.size()
    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
    if with_r:
        rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)
    return ret


class AddCoords(nn.Module):
    """
    An alternative implementation for PyTorch with auto-infering the x-y dimensions.
    https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        return append_coords(x, self.with_r)


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class BiFPNDepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, act=nn.ReLU):
        super(BiFPNDepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-05)
        self.act = act(inplace=True)

    def forward(self, x: Tensor) ->Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=nn.ReLU, dilation=1):
        super(BiFPNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-05)
        self.act = act(inplace=True)

    def forward(self, x: Tensor) ->Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size: int=64, epsilon: float=0.0001, act=nn.ReLU):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        self.p3_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p4_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p4_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p7_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.w1 = nn.Parameter(torch.Tensor(2, 4), requires_grad=True)
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 4), requires_grad=True)
        self.w2_relu = nn.ReLU()
        torch.nn.init.constant_(self.w1, 1)
        torch.nn.init.constant_(self.w2, 1)

    def forward(self, inputs: List[Tensor]) ->List[Tensor]:
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, size=p6_x.size()[2:]))
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, size=p5_x.size()[2:]))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, size=p4_x.size()[2:]))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, size=p3_x.size()[2:]))
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * F.interpolate(p3_out, size=p4_x.size()[2:]))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * F.interpolate(p4_out, size=p5_x.size()[2:]))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * F.interpolate(p5_out, size=p6_x.size()[2:]))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * F.interpolate(p6_out, size=p7_x.size()[2:]))
        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class RCM(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        return self.block(x) + x


class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def ds_cfm_branch(in_channels: int, out_channels: int, kernel_size: int):
    return nn.Sequential(DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(out_channels))


class CFM(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=[3, 5, 7, 11]):
        super().__init__()
        self.gp_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.conv_branches = nn.ModuleList(ds_cfm_branch(in_channels, out_channels, ks) for ks in kernel_sizes)

    def forward(self, x):
        gp = self.gp_branch(x)
        gp = gp.expand_as(x)
        conv_branches = [conv(x) for conv in self.conv_branches]
        return torch.cat(conv_branches + [gp], dim=1)


class AMM(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(DepthwiseSeparableConv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, encoder, decoder):
        decoder = F.interpolate(decoder, size=encoder.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([encoder, decoder], dim=1)
        x = self.conv_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, 1) * x
        return encoder + x


class CANDecoder(nn.Module):
    """
    Context Aggregation Network
    """

    def __init__(self, features: List[int], out_channels=256):
        super().__init__()
        self.encoder_rcm = nn.ModuleList(RCM(in_channels, out_channels) for in_channels in features)
        self.cfm = nn.Sequential(CFM(out_channels, out_channels), RCM(out_channels * 5, out_channels))
        self.amm_blocks = nn.ModuleList(AMM(out_channels, out_channels) for in_channels in features[:-1])
        self.rcm_blocks = nn.ModuleList(RCM(out_channels, out_channels) for in_channels in features[:-1])
        self.output_filters = [out_channels] * len(features)

    def forward(self, features):
        features = [rcm(x) for x, rcm in zip(features, self.encoder_rcm)]
        x = self.cfm(features[-1])
        outputs = [x]
        num_blocks = len(self.amm_blocks)
        for index in range(num_blocks):
            block_index = num_blocks - index - 1
            encoder_input = features[block_index]
            x = self.amm_blocks[block_index](encoder_input, x)
            x = self.rcm_blocks[block_index](x)
            outputs.append(x)
        return outputs[::-1]


class DecoderModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)


class SegmentationDecoderModule(DecoderModule):
    """
    A placeholder for future. Indicates sub-class decoders are suitable for segmentation tasks
    """
    pass


class ASPPModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int, norm_layer=nn.BatchNorm2d, activation=ACT_RELU):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class SeparableASPPModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int, norm_layer=nn.BatchNorm2d, activation=ACT_RELU):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, norm_layer=nn.BatchNorm2d, activation: str=ACT_RELU):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x: Tensor) ->Tensor:
        size = x.shape[-2:]
        x = self.pooling(x)
        x = self.conv(x)
        x = self.abn(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, aspp_module: Union[Type[ASPPModule], Type[SeparableASPPModule]], atrous_rates=(12, 24, 36), dropout: float=0.5, activation: str=ACT_RELU):
        super(ASPP, self).__init__()
        aspp_modules = [aspp_module(in_channels, out_channels, 3, padding=1, dilation=1, activation=activation), ASPPPooling(in_channels, out_channels)] + [aspp_module(in_channels, out_channels, 3, padding=ar, dilation=ar) for ar in atrous_rates]
        self.aspp = nn.ModuleList(aspp_modules)
        self.project = nn.Sequential(nn.Conv2d(len(self.aspp) * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), instantiate_activation_block(activation, inplace=False), nn.Dropout(dropout, inplace=True))

    def forward(self, x: Tensor) ->Tensor:
        res = []
        for aspp in self.aspp:
            res.append(aspp(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeeplabV3Decoder(DecoderModule):
    """
    Implements DeepLabV3 model from `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Partially copy-pasted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    """

    def __init__(self, feature_maps: List[int], aspp_channels: int, channels: int, atrous_rates=(12, 24, 36), dropout: float=0.5, activation=ACT_RELU):
        """

        Args:
            feature_maps: List of input channels
            aspp_channels:
            channels: Output channels
            atrous_rates:
            dropout:
            activation:
        """
        super().__init__()
        self.aspp = ASPP(in_channels=feature_maps[-1], out_channels=aspp_channels, aspp_module=ASPPModule, atrous_rates=atrous_rates, dropout=dropout, activation=activation)
        self.final = nn.Sequential(nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False), nn.BatchNorm2d(aspp_channels), instantiate_activation_block(activation, inplace=True), nn.Conv2d(aspp_channels, channels, kernel_size=1))
        self._channels = [channels]
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        high_level_features = feature_maps[-1]
        high_level_features = self.aspp(high_level_features)
        return self.final(high_level_features)

    @property
    def channels(self) ->Tuple[int]:
        return self._channels


class DeeplabV3PlusDecoder(DecoderModule):
    """
    Implements DeepLabV3 model from `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Partially copy-pasted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    """

    def __init__(self, feature_maps: List[int], channels: int, aspp_channels: int, low_level_channels: int=48, atrous_rates=(12, 24, 36), dropout: float=0.5, activation: str=ACT_RELU):
        """

        Args:
            feature_maps: Input feature maps
            aspp_channels:
            channels: Number of output channels
            atrous_rates:
            dropout:
            activation:
            low_level_channels:
        """
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(feature_maps[0], low_level_channels, 1, bias=False), nn.BatchNorm2d(low_level_channels), instantiate_activation_block(activation, inplace=True))
        self.aspp = ASPP(in_channels=feature_maps[-1], out_channels=aspp_channels, atrous_rates=atrous_rates, dropout=dropout, activation=activation, aspp_module=SeparableASPPModule)
        self.final = nn.Sequential(nn.Conv2d(aspp_channels + low_level_channels, channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(channels), instantiate_activation_block(activation, inplace=True))
        self._channels = [channels]
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        low_level_features = self.project(feature_maps[0])
        output_feature = self.aspp(feature_maps[-1])
        high_level_features = F.interpolate(output_feature, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        combined_features = torch.cat([low_level_features, high_level_features], dim=1)
        return [self.final(combined_features)]

    @property
    def channels(self) ->Tuple[int]:
        return self._channels


class FPNCatDecoderBlock(nn.Module):
    """
    Simple prediction block composed of (Conv + BN + Activation) repeated twice
    """

    def __init__(self, input_features: int, output_features: int, abn_block=ABN, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(output_features)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(output_features)
        self.drop2 = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.drop2(x)
        return x


class FPNBottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN, dropout=0.0):
        """

        Args:
            encoder_features:
            decoder_features:
            output_features:
            supervision_channels:
            abn_block:
            dropout:
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.drop1 = nn.Dropout2d(dropout, inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNContextBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN, dropout=0.0):
        """
        Center FPN block that aggregates multi-scale context using strided average poolings

        :param in_channels: Number of input features
        :param out_channels: Number of output features
        :param abn_block: Block for Activation + BatchNorm2d
        :param dropout: Dropout rate after context fusion
        """
        super().__init__()
        self.bottleneck = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.proj4 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.proj8 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.proj_global = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.blend = nn.Conv2d(4 * in_channels // 8, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: Tensor) ->Tensor:
        x = self.bottleneck(x)
        p2 = self.proj2(self.pool2(x))
        p4 = self.proj4(self.pool4(x))
        p8 = self.proj8(self.pool8(x))
        pg = self.proj_global(self.pool_global(x))
        out_size = p2.size()[2:]
        x = torch.cat([p2, F.interpolate(p4, size=out_size, mode='nearest'), F.interpolate(p8, size=out_size, mode='nearest'), F.interpolate(pg, size=out_size, mode='nearest')], dim=1)
        x = self.blend(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


def conv1x1(in_channels: int, out_channels: int, groups=1, bias=True) ->nn.Conv2d:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
    if bias:
        torch.nn.init.zeros_(conv.bias)
    return conv


class FPNCatDecoder(SegmentationDecoderModule):
    """
    Feature pyramid network decoder with concatenation between intermediate layers:

        Input
        fm[0] -> predict(concat(bottleneck[0](fm[0]), upsample(fpn[1]))) -> fpn[0] -> output[0](fpn[0])
        fm[1] -> predict(concat(bottleneck[1](fm[1]), upsample(fpn[2]))) -> fpn[1] -> output[1](fpn[1])
        fm[2] -> predict(concat(bottleneck[2](fm[2]), upsample(fpn[3]))) -> fpn[2] -> output[2](fpn[2])
        ...
        fm[n] -> predict(concat(bottleneck[n](fm[n]), upsample(context)) -> fpn[n] -> output[n](fpn[n])
        fm[n] -> context_block(feature_map[n]) -> context
    """

    def __init__(self, feature_maps: List[int], channels: int, context_block=FPNContextBlock, bottleneck_block=FPNBottleneckBlock, predict_block: Union[nn.Identity, conv1x1, nn.Module]=conv1x1, output_block: Union[nn.Identity, conv1x1, nn.Module]=nn.Identity, prediction_channels: int=None, upsample_block=nn.Upsample):
        """
        Create a new instance of FPN decoder with concatenation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param channels: Output FPN channels
        :param context_block:
        :param bottleneck_block:
        :param predict_block:
        :param output_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()
        self.context = context_block(feature_maps[-1], channels)
        self.bottlenecks = nn.ModuleList([bottleneck_block(in_channels, channels) for in_channels in reversed(feature_maps)])
        self.predicts = nn.ModuleList([predict_block(channels + channels, channels) for _ in reversed(feature_maps)])
        if issubclass(output_block, nn.Identity):
            self.channels = [channels] * len(feature_maps)
            self.outputs = nn.ModuleList([output_block() for _ in reversed(feature_maps)])
        else:
            self.channels = [prediction_channels] * len(feature_maps)
            self.outputs = nn.ModuleList([output_block(channels, prediction_channels) for _ in reversed(feature_maps)])
        if issubclass(upsample_block, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_block(scale_factor=2) for _ in reversed(feature_maps)])
        else:
            self.upsamples = nn.ModuleList([upsample_block(channels, channels) for in_channels in reversed(feature_maps)])

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        last_feature_map = feature_maps[-1]
        feature_maps = reversed(feature_maps)
        outputs = []
        fpn = self.context(last_feature_map)
        for feature_map, bottleneck, upsample, predict_block, output_block in zip(feature_maps, self.bottlenecks, self.upsamples, self.predicts, self.outputs):
            fpn = torch.cat([bottleneck(feature_map), upsample(fpn)], dim=1)
            fpn = predict_block(fpn)
            outputs.append(output_block(fpn))
        return outputs[::-1]


class FPNSumDecoder(SegmentationDecoderModule):
    """
    Feature pyramid network decoder with summation between intermediate layers:

        Input
        feature_map[0] -> bottleneck[0](feature_map[0]) + upsample(fpn[1]) -> fpn[0]
        feature_map[1] -> bottleneck[1](feature_map[1]) + upsample(fpn[2]) -> fpn[1]
        feature_map[2] -> bottleneck[2](feature_map[2]) + upsample(fpn[3]) -> fpn[2]
        ...
        feature_map[n] -> bottleneck[n](feature_map[n]) + upsample(context) -> fpn[n]
        feature_map[n] -> context_block(feature_map[n]) -> context
    """

    def __init__(self, feature_maps: List[int], channels: int, context_block=FPNContextBlock, bottleneck_block=FPNBottleneckBlock, prediction_block: Union[nn.Identity, conv1x1, nn.Module]=nn.Identity, prediction_channels: int=None, upsample_block=nn.Upsample):
        """
        Create a new instance of FPN decoder with summation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param channels: FPN channels
        :param context_block:
        :param bottleneck_block:
        :param prediction_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()
        self.context = context_block(feature_maps[-1], channels)
        self.bottlenecks = nn.ModuleList([bottleneck_block(in_channels, channels) for in_channels in reversed(feature_maps)])
        if inspect.isclass(prediction_block) and issubclass(prediction_block, nn.Identity):
            self.outputs = nn.ModuleList([prediction_block() for _ in reversed(feature_maps)])
            self.channels = [channels] * len(feature_maps)
        else:
            self.outputs = nn.ModuleList([prediction_block(channels, prediction_channels) for _ in reversed(feature_maps)])
            self.channels = [prediction_channels] * len(feature_maps)
        if issubclass(upsample_block, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_block(scale_factor=2) for _ in reversed(feature_maps)])
        else:
            self.upsamples = nn.ModuleList([upsample_block(channels, channels) for in_channels in reversed(feature_maps)])

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        last_feature_map = feature_maps[-1]
        feature_maps = reversed(feature_maps)
        outputs = []
        fpn = self.context(last_feature_map)
        for feature_map, bottleneck, upsample, output_block in zip(feature_maps, self.bottlenecks, self.upsamples, self.outputs):
            fpn = bottleneck(feature_map) + upsample(fpn)
            outputs.append(output_block(fpn))
        return outputs[::-1]


class HRNetSegmentationDecoder(SegmentationDecoderModule):
    channels: int

    def __init__(self, feature_maps: List[int], channels: int, dropout=0.0, interpolation_mode='nearest', align_corners=None):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.channels = channels
        features = sum(feature_maps)
        self.embedding = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(features)), ('relu', nn.ReLU(inplace=True))]))
        self.logits = nn.Sequential(OrderedDict([('drop', nn.Dropout2d(dropout)), ('final', nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1))]))

    def forward(self, feature_maps: List[Tensor]):
        x_size = feature_maps[0].size()[2:]
        resized_feature_maps = [feature_maps[0]]
        for feature_map in feature_maps[1:]:
            feature_map = F.interpolate(feature_map, size=x_size, mode=self.interpolation_mode, align_corners=self.align_corners)
            resized_feature_maps.append(feature_map)
        feature_map = torch.cat(resized_feature_maps, dim=1)
        embedding = self.embedding(feature_map)
        return self.logits(embedding)


class PPMDecoder(DecoderModule):
    """
    Pyramid pooling decoder module

    https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/42b7567a43b1dab568e2bbfcbc8872778fbda92a/models/models.py
    """

    def __init__(self, feature_maps: List[int], num_classes=150, channels=512, pool_scales=(1, 2, 3, 6)):
        super(PPMDecoder, self).__init__()
        fc_dim = feature_maps[-1]
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * channels, channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(channels, num_classes, kernel_size=1))

    def forward(self, feature_maps: List[torch.Tensor]):
        last_fm = feature_maps[-1]
        input_size = last_fm.size()
        ppm_out = [last_fm]
        for pool_scale in self.ppm:
            input_pooled = pool_scale(last_fm)
            input_pooled = F.interpolate(input_pooled, size=input_size[2:], mode='bilinear', align_corners=False)
            ppm_out.append(input_pooled)
        ppm_out = torch.cat(ppm_out, dim=1)
        x = self.conv_last(ppm_out)
        return x


class DeconvolutionUpsample2d(nn.Module):

    def __init__(self, in_channels: int, n=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=2)

    def forward(self, x: Tensor, output_size: Optional[List[int]]=None) ->Tensor:
        return self.conv(x, output_size=output_size)


class UnetBlock(nn.Module):
    """
    Vanilla U-Net block containing of two convolutions interleaved with batch-norm and RELU
    """

    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class UNetDecoder(DecoderModule):

    def __init__(self, feature_maps: List[int], decoder_features: Union[int, List[int]]=None, unet_block=UnetBlock, upsample_block: Union[nn.Upsample, nn.ConvTranspose2d, Type[nn.PixelShuffle]]=None):
        super().__init__()
        if upsample_block is None:
            upsample_block = nn.ConvTranspose2d
        blocks = []
        upsamples = []
        num_blocks = len(feature_maps) - 1
        if decoder_features is None:
            decoder_features = [None] * num_blocks
        elif len(decoder_features) != num_blocks:
            raise ValueError(f'decoder_features must have length of {num_blocks}')
        in_channels_for_upsample_block = feature_maps[-1]
        for block_index in reversed(range(num_blocks)):
            features_from_encoder = feature_maps[block_index]
            if isinstance(upsample_block, nn.Upsample):
                upsamples.append(upsample_block)
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.Upsample):
                upsamples.append(upsample_block(scale_factor=2))
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.PixelShuffle):
                upsamples.append(upsample_block(upscale_factor=2))
                out_channels_from_upsample_block = in_channels_for_upsample_block // 4
            elif issubclass(upsample_block, nn.ConvTranspose2d):
                up = upsample_block(in_channels_for_upsample_block, in_channels_for_upsample_block // 2, kernel_size=3, stride=2, padding=1)
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels
            else:
                up = upsample_block(in_channels_for_upsample_block)
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels
            in_channels = features_from_encoder + out_channels_from_upsample_block
            out_channels = decoder_features[block_index] or in_channels // 2
            blocks.append(unet_block(in_channels, out_channels))
            in_channels_for_upsample_block = out_channels
            decoder_features[block_index] = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.output_filters = decoder_features

    @property
    @torch.jit.unused
    def channels(self) ->List[int]:
        return self.output_filters

    def forward(self, feature_maps: List[torch.Tensor]) ->List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]
            if isinstance(upsample_block, (nn.ConvTranspose2d, DeconvolutionUpsample2d)):
                x = upsample_block(x, output_size=encoder_input.size())
            else:
                x = upsample_block(x)
            x = torch.cat([x, encoder_input], dim=1)
            x = decoder_block(x)
            outputs.append(x)
        return outputs[::-1]


class UnetCentralBlockV2(nn.Module):

    def __init__(self, in_dec_filters, out_filters, mask_channels, abn_block=ABN):
        super().__init__()
        self.bottleneck = nn.Conv2d(in_dec_filters, out_filters, kernel_size=1)
        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)
        self.dsv = nn.Conv2d(out_filters, mask_channels, kernel_size=1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        dsv = self.dsv(x)
        return x, dsv


class UnetDecoderBlockV2(nn.Module):
    """"""

    def __init__(self, in_dec_filters: int, in_enc_filters: int, out_filters: int, mask_channels: int, abn_block=ABN, pre_dropout_rate=0.0, post_dropout_rate=0.0, scale_factor=None, scale_mode='nearest', align_corners=None):
        super(UnetDecoderBlockV2, self).__init__()
        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.align_corners = align_corners
        self.bottleneck = nn.Conv2d(in_dec_filters + in_enc_filters, out_filters, kernel_size=1)
        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)
        self.pre_drop = nn.Dropout2d(pre_dropout_rate, inplace=True)
        self.post_drop = nn.Dropout2d(post_dropout_rate)
        self.dsv = nn.Conv2d(out_filters, mask_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, enc: torch.Tensor) ->Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=self.align_corners)
        else:
            lat_size = enc.size()[2:]
            x = F.interpolate(x, size=lat_size, mode=self.scale_mode, align_corners=self.align_corners)
        x = torch.cat([x, enc], 1)
        x = self.bottleneck(x)
        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.post_drop(x)
        dsv = self.dsv(x)
        return x, dsv


class UNetDecoderV2(DecoderModule):

    def __init__(self, feature_maps: List[int], decoder_features: int, mask_channels: int, dropout=0.0, abn_block=ABN):
        super().__init__()
        if not isinstance(decoder_features, list):
            decoder_features = [(decoder_features * 2 ** i) for i in range(len(feature_maps))]
        blocks = []
        for block_index, in_enc_features in enumerate(feature_maps[:-1]):
            blocks.append(UnetDecoderBlockV2(decoder_features[block_index + 1], in_enc_features, decoder_features[block_index], mask_channels, abn_block=abn_block, post_dropout_rate=dropout))
        self.center = UnetCentralBlockV2(feature_maps[-1], decoder_features[-1], mask_channels, abn_block=abn_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features
        self.final = nn.Conv2d(decoder_features[0], mask_channels, kernel_size=1)

    def forward(self, feature_maps):
        output, dsv = self.center(feature_maps[-1])
        dsv_list = [dsv]
        for decoder_block, encoder_output in zip(reversed(self.blocks), reversed(feature_maps[:-1])):
            output, dsv = decoder_block(output, encoder_output)
            dsv_list.append(dsv)
        dsv_list = list(reversed(dsv_list))
        output = self.final(output)
        return output, dsv_list


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """
    3x3 convolution + BN + relu
    """
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class UPerNet(nn.Module):

    def __init__(self, output_filters: List[int], num_classes=150, pool_scales=(1, 2, 3, 6), fpn_dim=256):
        super(UPerNet, self).__init__()
        last_fm_dim = output_filters[-1]
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(last_fm_dim, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(last_fm_dim + len(pool_scales) * 512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in output_filters[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(output_filters) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(output_filters) * fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, num_classes, kernel_size=1))

    def forward(self, feature_maps):
        last_fm = feature_maps[-1]
        input_size = last_fm.size()
        ppm_out = [last_fm]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(pool_scale(last_fm), (input_size[2], input_size[3]), mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(feature_maps) - 1)):
            conv_x = feature_maps[i]
            conv_x = self.fpn_in[i](conv_x)
            f = F.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        return x


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, 'Expected input with 4 dimensions (bsize, channels, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask, keeped = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * (block_mask.numel() / keeped)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :], kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        keeped = block_mask.numel() - block_mask.sum()
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask, keeped

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 2


class DropBlock3D(DropBlock2D):
    """Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        assert x.dim() == 5, 'Expected input with 5 dimensions (bsize, channels, depth, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :], kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 3


class DropBlockScheduled(nn.Module):

    def __init__(self, dropblock, start_value, stop_value, nr_steps, start_step=0):
        super(DropBlockScheduled, self).__init__()
        self.dropblock = dropblock
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.start_step = start_step
        self.nr_steps = nr_steps
        self.step_size = (stop_value - start_value) / nr_steps

    def forward(self, x):
        if self.training:
            self.step()
        return self.dropblock(x)

    def step(self):
        idx = self.i.item()
        if self.start_step < idx < self.start_step + self.nr_steps:
            self.dropblock.drop_prob += self.step_size
        self.i += 1


def _take_ints(elements: List[int], indexes: List[int]) ->List[int]:
    selected: List[int] = []
    for i in indexes:
        selected.append(elements[i])
    return selected


def _take_tensors(elements: List[Tensor], indexes: List[int]) ->List[Tensor]:
    selected: List[Tensor] = []
    for i in indexes:
        selected.append(elements[i])
    return selected


class EncoderModule(nn.Module):
    __constants__ = ['_layers', '_output_strides', '_output_filters']

    def __init__(self, channels: List[int], strides: List[int], layers: List[int]):
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError('Number of channels must be equal to number of strides')
        self._layers = list(layers)
        self._output_strides = _take_ints(strides, self._layers)
        self._output_filters = _take_ints(channels, self._layers)

    def forward(self, x: Tensor) ->List[Tensor]:
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            x = output
        return _take_tensors(output_features, self._layers)

    @property
    @torch.jit.unused
    def channels(self) ->Tuple[int, ...]:
        return tuple(self._output_filters)

    @property
    @torch.jit.unused
    def strides(self) ->Tuple[int, ...]:
        return tuple(self._output_strides)

    @torch.jit.unused
    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        """
        Change number of channels expected in the input tensor. By default,
        all encoders assume 3-channel image in BCHW notation with C=3.
        This method changes first convolution to have user-defined number of
        channels as input.
        """
        raise NotImplementedError


def _take(elements: List[Any], indexes: List[int]) ->List[Any]:
    selected = []
    for i in indexes:
        selected.append(elements[i])
    return selected


def make_n_channel_input_conv(conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], in_channels: int, mode='auto', **kwargs) ->Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]:
    """
    Create convolution block with same parameters and desired number of channels.

    Args:
        conv: Input nn.Conv2D object to copy settings/weights from
        in_channels: Desired number of input channels
        mode:
        **kwargs: Optional overrides for Conv2D parameters
    """
    conv_cls = conv.__class__
    if conv.in_channels == in_channels:
        warnings.warn('make_n_channel_input call is spurious')
        return conv
    new_conv = conv_cls(in_channels, out_channels=conv.out_channels, kernel_size=kwargs.get('kernel_size', conv.kernel_size), stride=kwargs.get('stride', conv.stride), padding=kwargs.get('padding', conv.padding), dilation=kwargs.get('dilation', conv.dilation), groups=kwargs.get('groups', conv.groups), bias=kwargs.get('bias', conv.bias is not None), padding_mode=kwargs.get('padding_mode', conv.padding_mode))
    w = conv.weight
    if in_channels > conv.in_channels:
        n = math.ceil(in_channels / float(conv.in_channels))
        w = torch.cat([w] * n, dim=1)
        w = w[:, :in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    else:
        w = w[:, 0:in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    return new_conv


def make_n_channel_input(conv: nn.Module, in_channels: int, mode='auto', **kwargs) ->nn.Module:
    """
    Create convolution block with same parameters and desired number of channels.

    Args:
        conv: Input nn.Conv2D object to copy settings/weights from
        in_channels: Desired number of input channels
        mode:
        **kwargs: Optional overrides for Conv2D parameters
    """
    if isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return make_n_channel_input_conv(conv, in_channels=in_channels, mode=mode, **kwargs)
    raise ValueError(f'Unsupported class {conv.__class__.__name__}')


class DenseNetEncoder(EncoderModule):

    def __init__(self, densenet: DenseNet, strides: List[int], channels: List[int], layers: List[int], first_avg_pool=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__(channels, strides, layers)

        def except_pool(block: nn.Module):
            del block.pool
            return block
        self.layer0 = nn.Sequential(OrderedDict([('conv0', densenet.features.conv0), ('bn0', densenet.features.norm0), ('act0', densenet.features.relu0)]))
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool0 = self.avg_pool if first_avg_pool else densenet.features.pool0
        self.layer1 = nn.Sequential(densenet.features.denseblock1, except_pool(densenet.features.transition1))
        self.layer2 = nn.Sequential(densenet.features.denseblock2, except_pool(densenet.features.transition2))
        self.layer3 = nn.Sequential(densenet.features.denseblock3, except_pool(densenet.features.transition3))
        self.layer4 = nn.Sequential(densenet.features.denseblock4)
        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    @torch.jit.unused
    def strides(self):
        return self._output_strides

    @property
    @torch.jit.unused
    def channels(self):
        return self._output_filters

    def forward(self, x):
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            if layer == self.layer0:
                output = self.pool0(output)
            else:
                output = self.avg_pool(output)
            x = output
        return _take(output_features, self._layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv0 = make_n_channel_input(self.layer0.conv0, input_channels, mode=mode, **kwargs)
        return self


class DenseNet121Encoder(DenseNetEncoder):

    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet121(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 512, 1024]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet161Encoder(DenseNetEncoder):

    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet161(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [96, 192, 384, 1056, 2208]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet169Encoder(DenseNetEncoder):

    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet169(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 640, 1664]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet201Encoder(DenseNetEncoder):

    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet201(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 896, 1920]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class HGResidualBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, reduction=2, activation: Callable=nn.ReLU):
        super(HGResidualBlock, self).__init__()
        mid_channels = input_channels // reduction
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.act1 = activation(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = activation(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.act3 = activation(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=True)
        if input_channels == output_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)
            torch.nn.init.zeros_(self.skip_layer.bias)
        torch.nn.init.zeros_(self.conv3.bias)

    def forward(self, x: Tensor) ->Tensor:
        residual = self.skip_layer(x)
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)
        out += residual
        return out


class HGStemBlock(nn.Module):

    def __init__(self, input_channels, output_channels, activation: Callable=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = activation(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = activation(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = activation(inplace=True)
        self.residual1 = HGResidualBlock(64, 128)
        self.residual2 = HGResidualBlock(128, output_channels)

    def forward(self, x: Tensor) ->Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class HGBlock(nn.Module):
    """
    A single Hourglass model block.
    """

    def __init__(self, depth: int, input_features: int, features, increase=0, activation=nn.ReLU, repeats=1, pooling_block=nn.MaxPool2d):
        super(HGBlock, self).__init__()
        nf = features + increase
        if inspect.isclass(pooling_block) and issubclass(pooling_block, (nn.MaxPool2d, nn.AvgPool2d)):
            self.down = pooling_block(kernel_size=2, padding=0, stride=2)
        else:
            self.down = pooling_block(input_features)
        if repeats == 1:
            self.up1 = HGResidualBlock(input_features, features, activation=activation)
            self.low1 = HGResidualBlock(input_features, nf, activation=activation)
        else:
            up_blocks = []
            up_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(up_input_features, features))
                up_input_features = features
            self.up1 = nn.Sequential(*up_blocks)
            down_blocks = []
            down_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(down_input_features, nf))
                down_input_features = nf
            self.low1 = nn.Sequential(*down_blocks)
        self.depth = depth
        if self.depth > 1:
            self.low2 = HGBlock(depth - 1, nf, nf, increase=increase, pooling_block=pooling_block, activation=activation, repeats=repeats)
        else:
            self.low2 = HGResidualBlock(nf, nf, activation=activation)
        self.low3 = HGResidualBlock(nf, features, activation=activation)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: Tensor) ->Tensor:
        up1 = self.up1(x)
        pool1 = self.down(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        hg = up1 + up2
        return hg


def conv1x1_bn_act(in_channels, out_channels, activation=nn.ReLU):
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1)), ('bn', nn.BatchNorm2d(out_channels)), ('act', activation(inplace=True))]))


class HGFeaturesBlock(nn.Module):

    def __init__(self, features: int, activation: Callable, blocks=1):
        super().__init__()
        residual_blocks = [HGResidualBlock(features, features, activation=activation) for _ in range(blocks)]
        self.residuals = nn.Sequential(*residual_blocks)
        self.linear = conv1x1_bn_act(features, features, activation=activation)

    def forward(self, x: Tensor) ->Tensor:
        x = self.residuals(x)
        x = self.linear(x)
        return x


class HGSupervisionBlock(nn.Module):

    def __init__(self, features, supervision_channels: int):
        super().__init__()
        self.squeeze = nn.Conv2d(features, supervision_channels, kernel_size=1)
        self.expand = nn.Conv2d(supervision_channels, features, kernel_size=1)

    def forward(self, x: Tensor) ->Tuple[Tensor, Tensor]:
        sup_mask = self.squeeze(x)
        sup_features = self.expand(sup_mask)
        return sup_mask, sup_features


class StackedHGEncoder(EncoderModule):
    """
    Original implementation: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    """

    def __init__(self, input_channels: int=3, stack_level: int=8, depth: int=4, features: int=256, activation=ACT_RELU, repeats=1, pooling_block=nn.MaxPool2d):
        super().__init__(channels=[features] + [features] * stack_level, strides=[4] + [4] * stack_level, layers=list(range(0, stack_level + 1)))
        self.stack_level = stack_level
        self.depth_level = depth
        self.num_features = features
        act = get_activation_block(activation)
        self.stem = HGStemBlock(input_channels, features, activation=act)
        input_features = features
        modules = []
        for _ in range(stack_level):
            modules.append(HGBlock(depth, input_features, features, increase=0, activation=act, repeats=repeats, pooling_block=pooling_block))
            input_features = features
        self.num_blocks = len(modules)
        self.blocks = nn.ModuleList(modules)
        self.features = nn.ModuleList([HGFeaturesBlock(features, blocks=4, activation=act) for _ in range(stack_level)])
        self.merge_features = nn.ModuleList([nn.Conv2d(features, features, kernel_size=1) for _ in range(stack_level - 1)])

    def __str__(self):
        return f'hg_s{self.stack_level}_d{self.depth_level}_f{self.num_features}'

    def forward(self, x: Tensor) ->List[Tensor]:
        x = self.stem(x)
        outputs = [x]
        for i, hourglass in enumerate(self.blocks):
            features = self.features[i](hourglass(x))
            outputs.append(features)
            if i < self.num_blocks - 1:
                x = x + self.merge_features[i](features)
        return outputs

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode)
        return self

    @property
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem] + list(self.blocks)


class StackedSupervisedHGEncoder(StackedHGEncoder):

    def __init__(self, supervision_channels: int, input_channels: int=3, stack_level: int=8, depth: int=4, features: int=256, activation=ACT_RELU, repeats=1, pooling_block=nn.MaxPool2d, supervision_block=HGSupervisionBlock):
        super().__init__(input_channels=input_channels, stack_level=stack_level, depth=depth, features=features, activation=activation, repeats=repeats, pooling_block=pooling_block)
        self.supervision_blocks = nn.ModuleList([supervision_block(features, supervision_channels) for _ in range(stack_level - 1)])

    def forward(self, x: Tensor) ->Tuple[List[Tensor], List[Tensor]]:
        x = self.stem(x)
        outputs = [x]
        supervision = []
        for i, hourglass in enumerate(self.blocks):
            features = self.features[i](hourglass(x))
            outputs.append(features)
            if i < self.num_blocks - 1:
                sup_mask, sup_features = self.supervision_blocks[i](features)
                supervision.append(sup_mask)
                x = x + self.merge_features[i](features) + sup_features
        return outputs, supervision


HRNETV2_BN_MOMENTUM = 0.1


def hrnet_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class HRNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HRNetBasicBlock, self).__init__()
        self.conv1 = hrnet_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = hrnet_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HRNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HRNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=HRNETV2_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=HRNETV2_BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i], momentum=HRNETV2_BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=HRNETV2_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=HRNETV2_BN_MOMENTUM), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=(height_output, width_output), mode='bilinear', align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNetEncoderBase(EncoderModule):

    def __init__(self, input_channels=3, width=48, layers: List[int]=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        channels = [64, width, width * 2, width * 4, width * 8]
        strides = [4, 4, 8, 16, 32]
        super().__init__(channels=channels, strides=strides, layers=layers)
        blocks_dict = {'BASIC': HRNetBasicBlock, 'BOTTLENECK': HRNetBottleneck}
        extra = {'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4), 'NUM_CHANNELS': (width, width * 2), 'FUSE_METHOD': 'SUM'}, 'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4), 'NUM_CHANNELS': (width, width * 2, width * 4), 'FUSE_METHOD': 'SUM'}, 'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4, 4), 'NUM_CHANNELS': (width, width * 2, width * 4, width * 8), 'FUSE_METHOD': 'SUM'}, 'FINAL_CONV_KERNEL': 1}
        self.layer0 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64, momentum=HRNETV2_BN_MOMENTUM)), ('relu', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64, momentum=HRNETV2_BN_MOMENTUM)), ('relu2', nn.ReLU(inplace=True))]))
        self.layer1 = self._make_layer(HRNetBottleneck, 64, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i], momentum=HRNETV2_BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels, momentum=HRNETV2_BN_MOMENTUM), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=HRNETV2_BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        blocks_dict = {'BASIC': HRNetBasicBlock, 'BOTTLENECK': HRNetBottleneck}
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        layer0 = self.layer0(x)
        x = self.layer1(layer0)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        outputs = _take([layer0] + y_list, self._layers)
        return outputs

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


class HRNetV2Encoder18(HRNetEncoderBase):

    def __init__(self, pretrained=None, layers=None):
        super().__init__(width=18, layers=layers)


class HRNetV2Encoder34(HRNetEncoderBase):

    def __init__(self, pretrained=None, layers=None):
        super().__init__(width=34, layers=layers)


class HRNetV2Encoder48(HRNetEncoderBase):

    def __init__(self, pretrained=None, layers=None):
        super().__init__(width=48, layers=layers)


class ResnetEncoder(EncoderModule):

    def __init__(self, resnet, filters, strides, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__(filters, strides, layers)
        self.layer0 = nn.Sequential(OrderedDict([('conv0', resnet.conv1), ('bn0', resnet.bn1), ('act0', resnet.relu)]))
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            if layer == self.layer0:
                output = self.maxpool(output)
            x = output
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv0 = make_n_channel_input(self.layer0.conv0, input_channels, mode=mode, **kwargs)
        return self


class Resnet18Encoder(ResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet18(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet34Encoder(ResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet34(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet50Encoder(ResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet50(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet101Encoder(ResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet101(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet152Encoder(ResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet152(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnetEncoder(EncoderModule):
    """
    The only difference from vanilla ResNet is that it has 'layer0' module
    """

    def __init__(self, seresnet: SENet, channels, strides, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__(channels, strides, layers)
        self.maxpool = seresnet.layer0.pool
        del seresnet.layer0.pool
        self.layer0 = seresnet.layer0
        self.layer1 = seresnet.layer1
        self.layer2 = seresnet.layer2
        self.layer3 = seresnet.layer3
        self.layer4 = seresnet.layer4
        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    @property
    @torch.jit.unused
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    @torch.jit.unused
    def strides(self):
        return self._output_strides

    @property
    @torch.jit.unused
    def channels(self):
        return self._output_filters

    def forward(self, x: Tensor) ->List[Tensor]:
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            if layer == self.layer0:
                output = self.maxpool(output)
            x = output
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode, **kwargs)
        return self


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


pretrained_settings = {'senet154': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet50': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet101': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet152': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnext50_32x4d': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnext101_32x4d': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}}


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SEResnet50Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet50(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SEResnet101Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet101(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SEResnet152Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet152(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SENet154Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = senet154(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SEResNeXt50Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SEResNeXt101Encoder(SEResnetEncoder):

    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnext101_32x4d(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SqueezenetEncoder(EncoderModule):

    def __init__(self, pretrained=True, layers=[1, 2, 3]):
        super().__init__([64, 128, 256, 512], [4, 8, 16, 16], layers)
        squeezenet = squeezenet1_1(pretrained=pretrained)
        self.layer0 = nn.Sequential(OrderedDict([('conv1', squeezenet.features[0]), ('relu1', nn.ReLU(inplace=True)), ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        self.layer1 = nn.Sequential(squeezenet.features[3], squeezenet.features[4], nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential(squeezenet.features[6], squeezenet.features[7], nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer3 = nn.Sequential(squeezenet.features[9], squeezenet.features[10], squeezenet.features[11], squeezenet.features[12])

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3]

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
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


def window_reverse(windows, window_size, H, W):
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
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer=act_layer, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
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
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(EncoderModule):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        layers (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, activation=ACT_GELU, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, layers=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False, pretrained=None):
        super().__init__(layers=layers, channels=[embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8], strides=[4, 8, 16, 32])
        act_layer = get_activation_block(activation)
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = 0, 1, 2, 3
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], act_layer=act_layer, norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        if pretrained:
            self.init_weights(pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            state = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')
            model_state_dict = state['model']
            self.load_state_dict(model_state_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return _take(outs, self._layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.patch_embed.proj = make_n_channel_input(self.patch_embed.proj, input_channels)
        return self


class SwinT(SwinTransformer):

    def __init__(self, ape=False, attn_drop_rate=0.0, depths=(2, 2, 6, 2), drop_path_rate=0.5, drop_rate=0.0, embed_dim=96, mlp_ratio=4.0, num_heads=(3, 6, 12, 24), layers=(0, 1, 2, 3), patch_norm=True, qk_scale=None, qkv_bias=True, use_checkpoint=False, window_size=7, activation=ACT_GELU, pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'):
        super(SwinT, self).__init__(ape=ape, attn_drop_rate=attn_drop_rate, depths=depths, drop_path_rate=drop_path_rate, drop_rate=drop_rate, embed_dim=embed_dim, mlp_ratio=mlp_ratio, num_heads=num_heads, layers=layers, patch_norm=patch_norm, qk_scale=qk_scale, qkv_bias=qkv_bias, use_checkpoint=use_checkpoint, window_size=window_size, activation=activation, pretrained=pretrained)


class SwinS(SwinTransformer):

    def __init__(self, ape=False, attn_drop_rate=0.0, depths=(2, 2, 18, 2), drop_path_rate=0.3, drop_rate=0.0, embed_dim=96, mlp_ratio=4.0, num_heads=(3, 6, 12, 24), layers=(0, 1, 2, 3), patch_norm=True, qk_scale=None, qkv_bias=True, use_checkpoint=False, window_size=7, activation=ACT_GELU, pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'):
        super().__init__(ape=ape, attn_drop_rate=attn_drop_rate, depths=depths, drop_path_rate=drop_path_rate, drop_rate=drop_rate, embed_dim=embed_dim, mlp_ratio=mlp_ratio, num_heads=num_heads, layers=layers, patch_norm=patch_norm, qk_scale=qk_scale, qkv_bias=qkv_bias, use_checkpoint=use_checkpoint, window_size=window_size, activation=activation, pretrained=pretrained)


class SwinB(SwinTransformer):

    def __init__(self, ape=False, attn_drop_rate=0.0, depths=(2, 2, 18, 2), drop_path_rate=0.5, drop_rate=0.0, embed_dim=128, mlp_ratio=4.0, num_heads=(4, 8, 16, 32), layers=(0, 1, 2, 3), patch_norm=True, qk_scale=None, qkv_bias=True, use_checkpoint=False, window_size=7, activation=ACT_GELU, pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'):
        super().__init__(ape=ape, attn_drop_rate=attn_drop_rate, depths=depths, drop_path_rate=drop_path_rate, drop_rate=drop_rate, embed_dim=embed_dim, mlp_ratio=mlp_ratio, num_heads=num_heads, layers=layers, patch_norm=patch_norm, qk_scale=qk_scale, qkv_bias=qkv_bias, use_checkpoint=use_checkpoint, window_size=window_size, activation=activation, pretrained=pretrained)


class SwinL(SwinTransformer):

    def __init__(self, ape=False, attn_drop_rate=0.0, depths=(2, 2, 18, 2), drop_path_rate=0.3, drop_rate=0.0, embed_dim=192, mlp_ratio=4.0, num_heads=(6, 12, 24, 48), layers=(0, 1, 2, 3), patch_norm=True, qk_scale=None, qkv_bias=True, use_checkpoint=False, window_size=7, activation=ACT_GELU, pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'):
        super().__init__(ape=ape, attn_drop_rate=attn_drop_rate, depths=depths, drop_path_rate=drop_path_rate, drop_rate=drop_rate, embed_dim=embed_dim, mlp_ratio=mlp_ratio, num_heads=num_heads, layers=layers, patch_norm=patch_norm, qk_scale=qk_scale, qkv_bias=qkv_bias, use_checkpoint=use_checkpoint, window_size=window_size, activation=activation, pretrained=pretrained)


class GenericTimmEncoder(EncoderModule):

    def __init__(self, timm_encoder: Union[nn.Module, str], layers: List[int]=None, pretrained=True):
        strides = []
        channels = []
        default_layers = []
        if isinstance(timm_encoder, str):
            timm_encoder = timm.models.factory.create_model(timm_encoder, features_only=True, pretrained=pretrained)
        for i, fi in enumerate(timm_encoder.feature_info):
            strides.append(fi['reduction'])
            channels.append(fi['num_chs'])
            default_layers.append(i)
        if layers is None:
            layers = default_layers
        super().__init__(channels, strides, layers)
        self.encoder = timm_encoder

    def forward(self, x: Tensor) ->List[Tensor]:
        all_feature_maps = self.encoder(x)
        return _take_tensors(all_feature_maps, self._layers)


class DPN68Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn68(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs)
        return self


class DPN68BEncoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn68b(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs)
        return self


class DPN92Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn92(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([336, 704, 1552, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs)
        return self


class DPN107Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn107(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([376, 1152, 2432, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs)
        return self


class DPN131Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn131(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([352, 832, 1984, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs)
        return self


def make_n_channel_input_conv2d_same(conv: nn.Conv2d, in_channels: int, mode='auto', **kwargs):
    assert isinstance(conv, nn.Conv2d)
    if conv.in_channels == in_channels:
        warnings.warn('make_n_channel_input call is spurious')
        return conv
    new_conv = Conv2dSame(in_channels, out_channels=conv.out_channels, kernel_size=kwargs.get('kernel_size', conv.kernel_size), stride=kwargs.get('stride', conv.stride), padding=kwargs.get('padding', conv.padding), dilation=kwargs.get('dilation', conv.dilation), groups=kwargs.get('groups', conv.groups), bias=kwargs.get('bias', conv.bias is not None))
    w = conv.weight
    if in_channels > conv.in_channels:
        n = math.ceil(in_channels / float(conv.in_channels))
        w = torch.cat([w] * n, dim=1)
        w = w[:, :in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    else:
        w = w[:, 0:in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    return new_conv


class TimmBaseEfficientNetEncoder(EncoderModule):

    def __init__(self, encoder, features, layers=[1, 2, 3, 4], first_conv_stride_one: bool=False):
        strides = [2, 4, 8, 16, 32]
        if first_conv_stride_one:
            strides = [1, 2, 4, 8, 16]
            encoder.conv_stem.stride = 1, 1
        super().__init__(features, strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode, **kwargs)
        return self


class TimmB0Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, use_tf=True):
        model_cls = tf_efficientnet_b0_ns if use_tf else efficientnet_b0
        act_layer = get_activation_block(activation)
        encoder = model_cls(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05)
        super().__init__(encoder, features=[16, 24, 40, 112, 320], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB1Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b1_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05)
        super().__init__(encoder, [16, 24, 40, 112, 320], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB2Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate: float=0.1):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b2_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [16, 24, 48, 120, 352], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB3Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.1):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b3_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [24, 32, 48, 136, 384], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB4Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.2):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b4_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [24, 32, 56, 160, 448], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB5Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.2):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b5_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [24, 40, 64, 176, 512], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB6Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.2):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b6_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [32, 40, 72, 200, 576], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB7Encoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.2):
        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b7_ns(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [32, 48, 80, 224, 640], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmMixNetXLEncoder(TimmBaseEfficientNetEncoder):

    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str=ACT_SILU, first_conv_stride_one: bool=False, drop_path_rate=0.2):
        act_layer = get_activation_block(activation)
        encoder = mixnet_xl(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate)
        super().__init__(encoder, [40, 48, 64, 192, 320], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmEfficientNetV2(GenericTimmEncoder):

    def __init__(self, model_name: str='efficientnetv2_rw_s', pretrained=True, layers=None, activation: str=ACT_SILU, drop_rate=0.0, drop_path_rate=0.0):
        act_layer = get_activation_block(activation)
        encoder = create_model(model_name=model_name, pretrained=pretrained, features_only=True, act_layer=act_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        super().__init__(encoder, layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        if isinstance(self.encoder.conv_stem, Conv2dSame):
            self.encoder.conv_stem = make_n_channel_input_conv2d_same(self.encoder.conv_stem, input_channels, mode, **kwargs)
        else:
            self.encoder.conv_stem = make_n_channel_input(self.encoder.conv_stem, input_channels, mode, **kwargs)
        return self


class TimmRes2Net101Encoder(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        act_layer = get_activation_block(activation)
        encoder = res2net101_26w_4s(pretrained=pretrained, act_layer=act_layer, features_only=True)
        super().__init__(encoder, layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class TimmRes2Next50Encoder(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        act_layer = get_activation_block(activation)
        encoder = res2next50(pretrained=pretrained, act_layer=act_layer, features_only=True)
        super().__init__(encoder, layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class TResNetMEncoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        act_layer = get_activation_block(activation)
        encoder = tresnet_m(pretrained=pretrained, act_layer=act_layer)
        super().__init__([64, 64, 128, 1024, 2048], [4, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.body.SpaceToDepth, encoder.body.conv1)
        self.layer1 = encoder.body.layer1
        self.layer2 = encoder.body.layer2
        self.layer3 = encoder.body.layer3
        self.layer4 = encoder.body.layer4

    @property
    @torch.jit.unused
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class SKResNet18Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None, no_first_max_pool=False, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        act_layer = get_activation_block(activation)
        encoder = skresnet18(pretrained=pretrained, features_only=True, act_layer=act_layer)
        super().__init__([64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(OrderedDict([('conv1', encoder.conv1), ('bn1', encoder.bn1), ('act1', encoder.act1)]))
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2) if no_first_max_pool else encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SKResNeXt50Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        act_layer = get_activation_block(activation)
        encoder = skresnext50_32x4d(pretrained=pretrained, act_layer=act_layer)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(OrderedDict([('conv1', encoder.conv1), ('bn1', encoder.bn1), ('act1', encoder.act1)]))
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SWSLResNeXt101Encoder(EncoderModule):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        act_layer = get_activation_block(activation)
        encoder = swsl_resnext101_32x8d(pretrained=pretrained, act_layer=act_layer)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(OrderedDict([('conv1', encoder.conv1), ('bn1', encoder.bn1), ('act1', encoder.act1)]))
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class TimmResnet152D(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        act_layer = get_activation_block(activation)
        encoder = resnet152d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmSEResnet152D(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        act_layer = get_activation_block(activation)
        encoder = seresnet152d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet50D(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        act_layer = get_activation_block(activation)
        encoder = resnet50d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet101D(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        act_layer = get_activation_block(activation)
        encoder = resnet101d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet200D(GenericTimmEncoder):

    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        act_layer = get_activation_block(activation)
        encoder = resnet200d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class UnetEncoder(EncoderModule):
    """
    Vanilla U-Net encoder
    """

    def __init__(self, in_channels=3, out_channels=32, num_layers=4, growth_factor=2, pool_block: Union[nn.MaxPool2d, nn.AvgPool2d]=None, unet_block: Union[nn.Module, UnetBlock]=UnetBlock):
        if pool_block is None:
            pool_block = partial(nn.MaxPool2d, kernel_size=2, stride=2)
        feature_maps = [(out_channels * growth_factor ** i) for i in range(num_layers)]
        strides = [(2 ** i) for i in range(num_layers)]
        super().__init__(feature_maps, strides, layers=list(range(num_layers)))
        input_filters = in_channels
        self.num_layers = num_layers
        for layer in range(num_layers):
            block = unet_block(input_filters, feature_maps[layer])
            if layer > 0:
                pool = pool_block()
                block = nn.Sequential(OrderedDict([('pool', pool), ('conv', block)]))
            input_filters = feature_maps[layer]
            self.add_module(f'layer{layer}', block)

    @property
    def encoder_layers(self):
        return [self.__getattr__(f'layer{layer}') for layer in range(self.num_layers)]

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


def make_conv_bn_act(in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, zero_batch_norm: bool=False, use_activation: bool=True, activation: str=ACT_RELU) ->torch.nn.Sequential:
    """
    Create a nn.Conv2d block followed by nn.BatchNorm2d and (optional) activation block.
    """
    batch_norm = nn.BatchNorm2d(out_channels)
    nn.init.constant_(batch_norm.weight, 0.0 if zero_batch_norm else 1.0)
    layers = [('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)), ('bn', batch_norm)]
    if use_activation:
        activation_block = instantiate_activation_block(activation, inplace=True)
        layers.append((activation, activation_block))
    return nn.Sequential(OrderedDict(layers))


class StemBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, activation: str=ACT_RELU):
        super().__init__()
        self.conv_bn_relu_1 = make_conv_bn_act(input_channels, 8, stride=2, activation=activation)
        self.conv_bn_relu_2 = make_conv_bn_act(8, 64, activation=activation)
        self.conv_bn_relu_3 = make_conv_bn_act(64, output_channels, activation=activation)

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.conv_bn_relu_3(x)
        return x


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""

    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int=1, activation: str=ACT_RELU):
        super().__init__()
        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion
        if expansion == 1:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation), make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]
        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module

        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super().__init__()
        assert reduction or squeeze_channels, "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"
        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x: Tensor):
        module_input = x
        x = self.avg_pool(x)
        x = self.squeeze(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2d(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels, reduction=reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class SEXResNetBlock(nn.Module):
    """Creates the Squeeze&Excitation + XResNet block."""

    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int=1, activation: str=ACT_RELU):
        super().__init__()
        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion
        if expansion == 1:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation), make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]
        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)
        self.se = ChannelSpatialGate2d(n_filters, reduction=4)
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class XResNet(EncoderModule):

    def __init__(self, expansion: int, blocks: List[int], input_channels: int=3, activation=ACT_RELU, layers=None, first_pool: Union[nn.MaxPool2d, nn.AvgPool2d]=nn.MaxPool2d, pretrained=None, block: Union[XResNetBlock, SEXResNetBlock]=XResNetBlock):
        assert len(blocks) == 4
        if layers is None:
            layers = [1, 2, 3, 4]
        n_filters = [64 // expansion, 64, 128, 256, 512]
        channels = [64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
        super().__init__(channels, [2, 4, 8, 16, 32], layers)
        res_layers = [self._make_layer(block, expansion, n_filters[i], n_filters[i + 1], n_blocks=l, stride=1 if i == 0 else 2, activation=activation) for i, l in enumerate(blocks)]
        self.stem = StemBlock(input_channels, 64, activation=activation)
        self.layer1 = nn.Sequential(OrderedDict([('pool', first_pool(kernel_size=3, stride=2, padding=1)), ('block', res_layers[0])]))
        self.layer2 = res_layers[1]
        self.layer3 = res_layers[2]
        self.layer4 = res_layers[3]

    @property
    def encoder_layers(self) ->List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    @staticmethod
    def _make_layer(block, expansion, n_inputs: int, n_filters: int, n_blocks: int, stride: int, activation: str):
        return nn.Sequential(*[block(expansion, n_inputs if i == 0 else n_filters, n_filters, stride if i == 0 else 1, activation=activation) for i in range(n_blocks)])

    def change_input_channels(self, input_channels: int, mode='auto', **kwargs):
        self.stem.conv_bn_relu_1.conv = make_n_channel_input(self.stem.conv_bn_relu_1.conv, input_channels, mode)


class FPNFuse(nn.Module):

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[Tensor]):
        layers = []
        dst_size = features[0].size()[2:]
        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))
        return torch.cat(layers, dim=1)


class FPNFuseSum(nn.Module):
    """Compute a sum of individual FPN layers"""

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[Tensor]) ->Tensor:
        output = features[0]
        dst_size = features[0].size()[2:]
        for f in features[1:]:
            output = output + F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners)
        return output


class HFF(nn.Module):
    """
    Hierarchical feature fusion module.
    https://arxiv.org/pdf/1811.11431.pdf
    https://arxiv.org/pdf/1803.06815.pdf

    What it does is easily explained in code:
    feature_map_0 - feature_map of the highest resolution
    feature_map_N - feature_map of the smallest resolution

    >>> feature_map = feature_map_0 + up(feature_map[1] + up(feature_map[2] + up(feature_map[3] + ...))))
    """

    def __init__(self, sizes=None, upsample_scale=2, mode='nearest', align_corners=None):
        super().__init__()
        self.sizes = sizes
        self.interpolation_mode = mode
        self.align_corners = align_corners
        self.upsample_scale = upsample_scale

    def forward(self, features: List[Tensor]) ->Tensor:
        num_feature_maps = len(features)
        current_map = features[-1]
        for feature_map_index in reversed(range(num_feature_maps - 1)):
            if self.sizes is not None:
                prev_upsampled = self._upsample(current_map, self.sizes[feature_map_index])
            else:
                prev_upsampled = self._upsample(current_map)
            current_map = features[feature_map_index] + prev_upsampled
        return current_map

    def _upsample(self, x, output_size=None):
        if output_size is not None:
            x = F.interpolate(x, size=(output_size[0], output_size[1]), mode=self.interpolation_mode, align_corners=self.align_corners)
        else:
            x = F.interpolate(x, scale_factor=self.upsample_scale, mode=self.interpolation_mode, align_corners=self.align_corners)
        return x


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).float().reshape(1, len(mean), 1, 1).contiguous())
        self.register_buffer('std', torch.tensor(std).float().reshape(1, len(std), 1, 1).reciprocal().contiguous())

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return (input.type_as(self.mean) - self.mean) * self.std


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0), abn_block(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=False)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels, key_channels, value_channels, out_channels, scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=[1], abn_block=ABN):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0), abn_block(out_channels), nn.Dropout2d(dropout))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ObjectContextBlock(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=[1], abn_block=ABN):
        super(ObjectContextBlock, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), abn_block(out_channels))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class ASPObjectContextBlock(nn.Module):

    def __init__(self, features, out_features=256, dilations=(12, 24, 36), abn_block=ABN, dropout=0.1):
        super(ASPObjectContextBlock, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=False), abn_block(out_features), ObjectContextBlock(in_channels=out_features, out_channels=out_features, key_channels=out_features // 2, value_channels=out_features, dropout=dropout, sizes=[2]))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False), abn_block(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), abn_block(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), abn_block(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), abn_block(out_features))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False), abn_block(out_features * 2), nn.Dropout2d(dropout))

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), dim=1))
        return z

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, torch.Tensor):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


class _PyramidSelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), abn_block(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, _, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == self.scale - 1:
                    end_x = h
                if j == self.scale - 1:
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)
            sim_map = torch.matmul(query_local, key_local)
            sim_map = self.key_channels ** -0.5 * sim_map
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local)
            local_list.append(context_local)
        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(in_channels, key_channels, value_channels, out_channels, scale)


class PyramidObjectContextBlock(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout=0.05, sizes=[1, 2, 3, 6], abn_block=ABN):
        super(PyramidObjectContextBlock, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, in_channels // 2, in_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False), abn_block(out_channels), nn.Dropout2d(dropout))
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels * self.group, kernel_size=1, padding=0, bias=False), abn_block(in_channels * self.group))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return PyramidSelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output


class GlobalMaxPool2d(nn.Module):

    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalKMaxPool2d(nn.Module):
    """
    K-max global pooling block

    https://arxiv.org/abs/1911.07344
    """

    def __init__(self, k=4, trainable=True, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.k = k
        self.flatten = flatten
        self.trainable = trainable
        weights = torch.ones((1, 1, k))
        if trainable:
            self.register_parameter('weights', torch.nn.Parameter(weights))
        else:
            self.register_buffer('weights', weights)

    def forward(self, x: Tensor):
        input = x.view(x.size(0), x.size(1), -1)
        kmax = input.topk(k=self.k, dim=2)[0]
        kmax = (kmax * self.weights).mean(dim=2)
        if not self.flatten:
            kmax = kmax.view(kmax.size(0), kmax.size(1), 1, 1)
        return kmax

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool=True):
        if not self.trainable:
            return _IncompatibleKeys([], [])
        super().load_state_dict(state_dict, strict)


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class RMSPool(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        x_mean = x.mean(dim=[2, 3])
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class MILCustomPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weight_generator = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1), nn.ReLU(True), nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        weight = self.weight_generator(x)
        loss = self.classifier(x)
        logits = torch.sum(weight * loss, dim=[2, 3]) / (torch.sum(weight, dim=[2, 3]) + 1e-06)
        return logits


class GlobalRankPooling(nn.Module):
    """
    https://arxiv.org/abs/1704.02112
    """

    def __init__(self, num_features, spatial_size, flatten=False):
        super().__init__()
        self.conv = nn.Conv1d(num_features, num_features, spatial_size, groups=num_features)
        self.flatten = flatten

    def forward(self, x: torch.Tensor):
        spatial_size = x.size(2) * x.size(3)
        assert spatial_size == self.conv.kernel_size[0], f'Expected spatial size {self.conv.kernel_size[0]}, got {x.size(2)}x{x.size(3)}'
        x = x.view(x.size(0), x.size(1), -1)
        x_sorted, index = x.topk(spatial_size, dim=2)
        x = self.conv(x_sorted)
        if self.flatten:
            x = x.squeeze(2)
        return x


class GeneralizedMeanPooling2d(nn.Module):
    """

    https://arxiv.org/pdf/1902.05509v2.pdf
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, p: float=3, eps=1e-06, flatten=False):
        super(GeneralizedMeanPooling2d, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x: Tensor) ->Tensor:
        x = F.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(self.p), output_size=1).pow(1.0 / self.p)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.item()) + ', ' + 'eps=' + str(self.eps) + ')'


class SpatialGate2dV2(nn.Module):
    """
    Spatial Squeeze and Channel Excitation module
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeeze_channels = max(1, channels // reduction)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=7, dilation=3, padding=3 * 3)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.squeeze(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2dV2(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2dV2(channels, reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class SRMLayer(nn.Module):
    """An implementation of SRM block, proposed in
    "SRM : A Style-based Recalibration Module for Convolutional Neural Networks".

    """

    def __init__(self, channels: int):
        super(SRMLayer, self).__init__()
        self.cfc = nn.Conv1d(channels, channels, kernel_size=2, bias=False, groups=channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)
        z = self.cfc(u)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)


class UnetCentralBlock(nn.Module):

    def __init__(self, in_dec_filters: int, out_filters: int, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class UnetDecoderBlock(nn.Module):
    """"""

    def __init__(self, in_dec_filters: int, in_enc_filters: int, out_filters: int, abn_block=ABN, dropout_rate=0.0, scale_factor=None, scale_mode='nearest', align_corners=None):
        super(UnetDecoderBlock, self).__init__()
        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(in_dec_filters + in_enc_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_filters)
        self.drop = nn.Dropout2d(dropout_rate, inplace=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)

    def forward(self, x: torch.Tensor, enc: Optional[torch.Tensor]=None) ->torch.Tensor:
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=self.align_corners)
        else:
            lat_size = enc.size()[2:]
            x = F.interpolate(x, size=lat_size, mode=self.scale_mode, align_corners=self.align_corners)
        if enc is not None:
            x = torch.cat([x, enc], dim=1)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class DepthToSpaceUpsample2d(nn.Module):
    """
    NOTE: This block is not fully ready yet. Need to figure out how to correctly initialize
    default weights to have bilinear upsample identical to OpenCV results

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int=2):
        super().__init__()
        n = 2 ** scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * n, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x: Tensor) ->Tensor:
        x = self.shuffle(self.conv(x))
        return x


class BilinearAdditiveUpsample2d(nn.Module):
    """
    https://arxiv.org/abs/1707.05847
    """

    def __init__(self, in_channels: int, scale_factor: int=2, n: int=4):
        super().__init__()
        if in_channels % n != 0:
            raise ValueError(f'Number of input channels ({in_channels})must be divisable by n ({n})')
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.n = n

    def forward(self, x: Tensor) ->Tensor:
        x = self.upsample(x)
        n, c, h, w = x.size()
        x = x.reshape(n, c // self.n, self.n, h, w).mean(2)
        return x


class ResidualDeconvolutionUpsample2d(nn.Module):

    def __init__(self, in_channels, scale_factor=2, n=4):
        if scale_factor != 2:
            raise NotImplementedError('Scale factor other than 2 is not implemented')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=scale_factor, output_padding=1)
        self.residual = BilinearAdditiveUpsample2d(in_channels, scale_factor=scale_factor, n=n)
        self.init_weights()

    def forward(self, x: Tensor) ->Tensor:
        residual_up = self.residual(x)
        return self.conv(x, output_size=residual_up.size()) + residual_up

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        torch.nn.init.zeros_(self.conv.bias)


class UnetSegmentationModel(nn.Module):

    def __init__(self, encoder: EncoderModule, unet_channels: Union[int, List[int]], num_classes: int=1, dropout=0.25, full_size_mask=True, activation=ACT_RELU, upsample_block=nn.UpsamplingNearest2d, last_upsample_block=None):
        super().__init__()
        self.encoder = encoder
        abn_block = partial(ABN, activation=activation)
        self.decoder = UNetDecoder(feature_maps=encoder.channels, decoder_features=unet_channels, unet_block=partial(UnetBlock, abn_block=abn_block), upsample_block=upsample_block)
        if last_upsample_block is not None:
            self.mask = nn.Sequential(OrderedDict([('drop', nn.Dropout2d(dropout)), ('conv', last_upsample_block(unet_channels[0], num_classes))]))
        else:
            self.last_upsample_block = None
            self.mask = nn.Sequential(OrderedDict([('drop', nn.Dropout2d(dropout)), ('conv', conv1x1(unet_channels[0], num_classes))]))
        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) ->Dict[str, Tensor]:
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)
        mask = self.mask(x[0])
        if self.full_size_mask and mask.size()[2:] != x_size:
            mask = F.interpolate(mask, size=x_size[2:], mode='bilinear', align_corners=False)
        return mask


class NoOp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class SumAll(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum(dim=[1, 2, 3])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AMM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPObjectContextBlock,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPPModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BalancedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaseOC_Module,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiFPNConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiFPNDepthwiseConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiTemperedLogisticLoss,
     lambda: ([], {'t1': 4, 't2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BilinearAdditiveUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinaryFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryLovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinarySoftF1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CFM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelGate2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelSpatialGate2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelSpatialGate2dV2,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeconvolutionUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseNet121Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseNet161Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseNet169Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseNet201Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DepthToSpaceUpsample2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseSeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropBlock2D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropBlock3D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DropBlockScheduled,
     lambda: ([], {'dropblock': _mock_layer(), 'start_value': 4, 'stop_value': 4, 'nr_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Ensembler,
     lambda: ([], {'models': [_mock_layer()]}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FPNBottleneckBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FPNCatDecoderBlock,
     lambda: ([], {'input_features': 4, 'output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedMeanPooling2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalKMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalWeightedAvgPool2d,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGBlock,
     lambda: ([], {'depth': 1, 'input_features': 4, 'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGFeaturesBlock,
     lambda: ([], {'features': 4, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGResidualBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGStemBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGSupervisionBlock,
     lambda: ([], {'features': 4, 'supervision_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HRNetBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HRNetEncoderBase,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (HRNetV2Encoder18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (HRNetV2Encoder34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (HRNetV2Encoder48,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityResidualBlock,
     lambda: ([], {'in_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MILCustomPoolingModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MishNaive,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {}),
     True),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NoOp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {'mean': [4, 4], 'std': [4, 4]}),
     lambda: ([torch.rand([4, 2, 4, 4])], {}),
     True),
    (ObjectContextBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PickModelOutput,
     lambda: ([], {'model': _mock_layer(), 'key': 4}),
     lambda: ([], {'input': torch.rand([5, 4])}),
     False),
    (PyramidObjectContextBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PyramidSelfAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RCM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMSPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (ResidualDeconvolutionUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet101Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Resnet152Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Resnet18Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Resnet34Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Resnet50Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEXResNetBlock,
     lambda: ([], {'expansion': 4, 'n_inputs': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 16, 64, 64])], {}),
     True),
    (SRMLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelectByIndex,
     lambda: ([], {'key': 4}),
     lambda: ([torch.rand([5, 4, 4, 4])], {}),
     False),
    (SelfAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SeparableASPPModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
    (SpatialGate2dV2,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezenetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (StackedHGEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (StackedSupervisedHGEncoder,
     lambda: ([], {'supervision_channels': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (StemBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SumAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwishNaive,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetCentralBlock,
     lambda: ([], {'in_dec_filters': 4, 'out_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetCentralBlockV2,
     lambda: ([], {'in_dec_filters': 4, 'out_filters': 4, 'mask_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetDecoderBlockV2,
     lambda: ([], {'in_dec_filters': 4, 'in_enc_filters': 4, 'out_filters': 4, 'mask_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UnetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (XResNet,
     lambda: ([], {'expansion': 4, 'blocks': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (XResNetBlock,
     lambda: ([], {'expansion': 4, 'n_inputs': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 16, 64, 64])], {}),
     True),
    (_PyramidSelfAttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_SelfAttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_BloodAxe_pytorch_toolbelt(_paritybench_base):
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

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

    def test_077(self):
        self._check(*TESTCASES[77])

    def test_078(self):
        self._check(*TESTCASES[78])

    def test_079(self):
        self._check(*TESTCASES[79])

    def test_080(self):
        self._check(*TESTCASES[80])

    def test_081(self):
        self._check(*TESTCASES[81])

    def test_082(self):
        self._check(*TESTCASES[82])

    def test_083(self):
        self._check(*TESTCASES[83])

    def test_084(self):
        self._check(*TESTCASES[84])

    def test_085(self):
        self._check(*TESTCASES[85])

    def test_086(self):
        self._check(*TESTCASES[86])

    def test_087(self):
        self._check(*TESTCASES[87])

    def test_088(self):
        self._check(*TESTCASES[88])

    def test_089(self):
        self._check(*TESTCASES[89])

    def test_090(self):
        self._check(*TESTCASES[90])

    def test_091(self):
        self._check(*TESTCASES[91])

    def test_092(self):
        self._check(*TESTCASES[92])

    def test_093(self):
        self._check(*TESTCASES[93])

    def test_094(self):
        self._check(*TESTCASES[94])

    def test_095(self):
        self._check(*TESTCASES[95])

    def test_096(self):
        self._check(*TESTCASES[96])

    def test_097(self):
        self._check(*TESTCASES[97])

    def test_098(self):
        self._check(*TESTCASES[98])

    def test_099(self):
        self._check(*TESTCASES[99])

    def test_100(self):
        self._check(*TESTCASES[100])

    def test_101(self):
        self._check(*TESTCASES[101])

    def test_102(self):
        self._check(*TESTCASES[102])

    def test_103(self):
        self._check(*TESTCASES[103])

    def test_104(self):
        self._check(*TESTCASES[104])

    def test_105(self):
        self._check(*TESTCASES[105])

    def test_106(self):
        self._check(*TESTCASES[106])

