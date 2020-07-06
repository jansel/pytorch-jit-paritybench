import sys
_module = sys.modules[__name__]
del sys
demo_losses = _module
pytorch_toolbelt = _module
inference = _module
ensembling = _module
functional = _module
tiles = _module
tta = _module
losses = _module
dice = _module
focal = _module
functional = _module
jaccard = _module
joint_loss = _module
lovasz = _module
soft_bce = _module
soft_ce = _module
wing_loss = _module
modules = _module
activations = _module
backbone = _module
inceptionv4 = _module
mobilenet = _module
mobilenetv3 = _module
senet = _module
wider_resnet = _module
coord_conv = _module
decoders = _module
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
efficientnet = _module
hourglass = _module
hrnet = _module
inception = _module
resnet = _module
seresnet = _module
squeezenet = _module
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
catalyst = _module
criterions = _module
loss_adapter = _module
metrics = _module
opl = _module
utils = _module
visualization = _module
dataset_utils = _module
fs = _module
namesgenerator = _module
random = _module
rle = _module
support = _module
torch_utils = _module
zoo = _module
classification = _module
segmentation = _module
setup = _module
test_activations = _module
test_decoders = _module
test_encoders = _module
test_losses = _module
test_modules = _module
test_tiles = _module
test_tta = _module
test_utils_functional = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.nn import BCEWithLogitsLoss


import numpy as np


import torch


from torch import nn


from torch import Tensor


from typing import List


from typing import Union


from collections import Sized


from collections import Iterable


from typing import Tuple


import math


from functools import partial


from torch.nn.functional import interpolate


import torch.nn.functional as F


from torch.nn.modules.loss import _Loss


from typing import Optional


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import functional as F


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.utils import model_zoo


import torch.functional as F


import warnings


from torchvision.models import densenet121


from torchvision.models import densenet161


from torchvision.models import densenet169


from torchvision.models import densenet201


from torchvision.models import DenseNet


from copy import deepcopy


from typing import Callable


from torchvision.models import resnet50


from torchvision.models import resnet34


from torchvision.models import resnet18


from torchvision.models import resnet101


from torchvision.models import resnet152


from torchvision.models import squeezenet1_1


from math import hypot


from typing import Iterator


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.optimizer import Optimizer


from typing import Dict


from sklearn.metrics import f1_score


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


import random


import collections


from typing import Sequence


import re


from torch.utils.data import DataLoader


class ApplySoftmaxTo(nn.Module):

    def __init__(self, model: nn.Module, output_key: Union[str, List[str]]='logits', dim=1, temperature=1):
        """
        Apply softmax activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string or list of strings, indicating to what outputs softmax activation should be applied.
        :param dim: Tensor dimension for softmax activation
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = output_key if isinstance(output_key, (list, tuple)) else [output_key]
        self.output_keys = set(output_key)
        self.model = model
        self.dim = dim
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).softmax(dim=1)
        return output


class ApplySigmoidTo(nn.Module):

    def __init__(self, model: nn.Module, output_key: Union[str, List[str]]='logits', temperature=1):
        """
        Apply sigmoid activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string or list of strings, indicating to what outputs sigmoid activation should be applied.
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = output_key if isinstance(output_key, (list, tuple)) else [output_key]
        self.output_keys = set(output_key)
        self.model = model
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).sigmoid()
        return output


class Ensembler(nn.Module):
    """
    Compute sum (or average) of outputs of several models.
    """

    def __init__(self, models: List[nn.Module], average=True, outputs=None):
        """

        :param models:
        :param average:
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.outputs = outputs
        self.models = nn.ModuleList(models)
        self.average = average

    def forward(self, *input, **kwargs):
        output_0 = self.models[0](*input, **kwargs)
        num_models = len(self.models)
        if self.outputs:
            keys = self.outputs
        else:
            keys = output_0.keys()
        for index in range(1, num_models):
            output_i = self.models[index](*input, **kwargs)
            for key in keys:
                output_0[key] += output_i[key]
        if self.average:
            for key in keys:
                output_0[key] /= num_models
        return output_0


class PickModelOutput(nn.Module):
    """
    Assuming you have a model that outputs a dictionary, this module returns only a given element by it's key
    """

    def __init__(self, model: nn.Module, key: str):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, *input, **kwargs) ->Tensor:
        output = self.model(*input, **kwargs)
        return output[self.target_key]


class TTAWrapper(nn.Module):

    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


class MultiscaleTTAWrapper(nn.Module):
    """
    Multiscale TTA wrapper module
    """

    def __init__(self, model: nn.Module, scale_levels: List[float]=None, size_offsets: List[int]=None):
        """
        Initialize multi-scale TTA wrapper

        :param model: Base model for inference
        :param scale_levels: List of additional scale levels,
            e.g: [0.5, 0.75, 1.25]
        """
        super().__init__()
        assert scale_levels or size_offsets, 'Either scale_levels or size_offsets must be set'
        assert not (scale_levels and size_offsets), 'Either scale_levels or size_offsets must be set'
        self.model = model
        self.scale_levels = scale_levels
        self.size_offsets = size_offsets

    def forward(self, input: Tensor) ->Tensor:
        h = input.size(2)
        w = input.size(3)
        out_size = h, w
        output = self.model(input)
        if self.scale_levels:
            for scale in self.scale_levels:
                dst_size = int(h * scale), int(w * scale)
                input_scaled = interpolate(input, dst_size, mode='bilinear', align_corners=False)
                output_scaled = self.model(input_scaled)
                output_scaled = interpolate(output_scaled, out_size, mode='bilinear', align_corners=False)
                output += output_scaled
            output /= 1.0 + len(self.scale_levels)
        elif self.size_offsets:
            for offset in self.size_offsets:
                dst_size = int(h + offset), int(w + offset)
                input_scaled = interpolate(input, dst_size, mode='bilinear', align_corners=False)
                output_scaled = self.model(input_scaled)
                output_scaled = interpolate(output_scaled, out_size, mode='bilinear', align_corners=False)
                output += output_scaled
            output /= 1.0 + len(self.size_offsets)
        return output


BINARY_MODE = 'binary'


MULTICLASS_MODE = 'multiclass'


MULTILABEL_MODE = 'multilabel'


def soft_dice_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0, eps=1e-07, dims=None) ->torch.Tensor:
    """

    :param y_pred:
    :param y_true:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert y_pred.size() == y_true.size()
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + smooth) / (cardinality.clamp_min(eps) + smooth)
    return dice_score


def to_tensor(x, dtype=None) ->torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
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
        super(DiceLoss, self).__init__()
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
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), self.smooth, self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


def focal_loss_with_logits(input: torch.Tensor, target: torch.Tensor, gamma=2.0, alpha: Optional[float]=0.25, reduction='mean', normalized=False, reduced_threshold: Optional[float]=None) ->torch.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())
    logpt = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(-logpt)
    if reduced_threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * logpt
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if normalized:
        norm_factor = focal_term.sum() + 1e-05
        loss /= norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


class BinaryFocalLoss(_Loss):

    def __init__(self, alpha=None, gamma=2, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strenght).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]
        loss = self.focal_loss_fn(label_input, label_target)
        return loss


class FocalLoss(_Loss):

    def __init__(self, alpha=None, gamma=2, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized)

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index
        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, (cls), (...)]
            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]
            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss


def soft_jaccard_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0.0, eps=1e-07, dims=None) ->torch.Tensor:
    """

    :param y_pred:
    :param y_true:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert y_pred.size() == y_true.size()
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union.clamp_min(eps) + smooth)
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
        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), self.smooth, self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1 - scores
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


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
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
    """Nanmean compatible with generators.
    """
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


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for log, lab in zip(logits, labels))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


class BinaryLovaszLoss(_Loss):

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore=self.ignore)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
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
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, (0)]
        else:
            class_pred = probas[:, (c)]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(_lovasz_grad(fg_sorted))))
    return mean(losses)


def _lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


class LovaszLoss(_Loss):

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore=self.ignore)


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
            soft_targets = (1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)
        else:
            soft_targets = target
        loss = F.binary_cross_entropy_with_logits(input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask = target != self.ignore_index
            size = not_ignored_mask.sum()
            if size == 0:
                return 0
            loss *= not_ignored_mask
        else:
            size = loss.numel()
        if self.reduction == 'mean':
            loss = loss.sum() / size
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', smooth_factor=None, ignore_index: Optional[int]=None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        num_classes = input.size(1)
        log_prb = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(log_prb, target.long(), ignore_index=self.ignore_index, reduction=self.reduction)
        if self.smooth_factor is None:
            return ce_loss
        if self.ignore_index is not None:
            not_ignored_mask = target != self.ignore_index
            log_prb *= not_ignored_mask.unsqueeze(dim=1)
        if self.reduction == 'sum':
            smooth_loss = -log_prb.sum()
        else:
            smooth_loss = -log_prb.sum(dim=1)
            if self.reduction == 'mean':
                smooth_loss = smooth_loss.mean()
        return self.smooth_factor * smooth_loss / num_classes + (1 - self.smooth_factor) * ce_loss


class WingLoss(_Loss):

    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(prediction, target, self.width, self.curvature, self.reduction)


def mish(input):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return input * torch.tanh(F.softplus(input))


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


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.

    Credit: https://blog.ceshine.net/post/pytorch-memory-swish/
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def swish(x):
    return SwishFunction.apply(x)


class Swish(nn.Module):

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


ACT_GLU = 'glu'


ACT_HARD_SIGMOID = 'hard_sigmoid'


ACT_HARD_SWISH = 'hard_swish'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_MISH = 'mish'


ACT_NONE = 'none'


ACT_PRELU = 'prelu'


ACT_RELU = 'relu'


ACT_RELU6 = 'relu6'


ACT_SELU = 'selu'


ACT_SWISH = 'swish'


ACT_SWISH_NAIVE = 'swish_naive'


class Identity(nn.Module):
    """The most useful module. A pass-through module which does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_activation_block(activation_name: str):
    ACTIVATIONS = {ACT_CELU: nn.CELU, ACT_GLU: nn.GLU, ACT_PRELU: nn.PReLU, ACT_ELU: nn.ELU, ACT_HARD_SIGMOID: HardSigmoid, ACT_HARD_SWISH: HardSwish, ACT_LEAKY_RELU: nn.LeakyReLU, ACT_MISH: Mish, ACT_NONE: Identity, ACT_RELU6: nn.ReLU6, ACT_RELU: nn.ReLU, ACT_SELU: nn.SELU, ACT_SWISH: Swish, ACT_SWISH_NAIVE: SwishNaive}
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


class SqEx(nn.Module):
    """Squeeze-Excitation block. Implemented in ONNX & CoreML friendly way.
    Original implementation: https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
    """

    def __init__(self, n_features, reduction=4):
        super(SqEx, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')
        self.linear1 = nn.Conv2d(n_features, n_features // reduction, kernel_size=1, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv2d(n_features // reduction, n_features, kernel_size=1, bias=True)
        self.nonlin2 = HardSigmoid(inplace=True)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, output_size=1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = x * y
        return y


class LinearBottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, expplanes, k=3, stride=1, drop_prob=0, num_steps=300000.0, start_step=0, activation=nn.ReLU, act_params={'inplace': True}, SE=False):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes)
        self.db1 = nn.Dropout2d(drop_prob)
        self.act1 = activation(**act_params)
        self.conv2 = nn.Conv2d(expplanes, expplanes, kernel_size=k, stride=stride, padding=k // 2, bias=False, groups=expplanes)
        self.bn2 = nn.BatchNorm2d(expplanes)
        self.db2 = nn.Dropout2d(drop_prob)
        self.act2 = activation(**act_params)
        self.se = SqEx(expplanes) if SE else Identity()
        self.conv3 = nn.Conv2d(expplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.db3 = nn.Dropout2d(drop_prob)
        self.stride = stride
        self.expplanes = expplanes
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.db2(out)
        out = self.act2(out)
        out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.db3(out)
        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        return out


class LastBlockLarge(nn.Module):

    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockLarge, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1)
        self.act2 = HardSwish(inplace=True)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)
        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))
        return out


class LastBlockSmall(nn.Module):

    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)
        self.se = SqEx(expplanes1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1, bias=False)
        self.act2 = HardSwish(inplace=True)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)
        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.se(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    Ensure that all layers have a channel number that is divisible by 8

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Module):
    """MobileNetV3 implementation.
    """

    def __init__(self, num_classes=1000, scale=1.0, in_channels=3, drop_prob=0.0, num_steps=300000.0, start_step=0, small=False):
        super(MobileNetV3, self).__init__()
        self.num_steps = num_steps
        self.start_step = start_step
        self.scale = scale
        self.num_classes = num_classes
        self.small = small
        self.bottlenecks_setting_large = [[16, 16, 16, 1, 3, 0, False, nn.ReLU], [16, 64, 24, 2, 3, 0, False, nn.ReLU], [24, 72, 24, 1, 3, 0, False, nn.ReLU], [24, 72, 40, 2, 5, 0, True, nn.ReLU], [40, 120, 40, 1, 5, 0, True, nn.ReLU], [40, 120, 40, 1, 5, 0, True, nn.ReLU], [40, 240, 80, 2, 3, drop_prob, False, HardSwish], [80, 200, 80, 1, 3, drop_prob, False, HardSwish], [80, 184, 80, 1, 3, drop_prob, False, HardSwish], [80, 184, 80, 1, 3, drop_prob, False, HardSwish], [80, 480, 112, 1, 3, drop_prob, True, HardSwish], [112, 672, 112, 1, 3, drop_prob, True, HardSwish], [112, 672, 160, 2, 5, drop_prob, True, HardSwish], [160, 672, 160, 1, 5, drop_prob, True, HardSwish], [160, 960, 160, 1, 5, drop_prob, True, HardSwish]]
        self.bottlenecks_setting_small = [[16, 64, 16, 2, 3, 0, True, nn.ReLU], [16, 72, 24, 2, 3, 0, False, nn.ReLU], [24, 88, 24, 1, 3, 0, False, nn.ReLU], [24, 96, 40, 2, 5, 0, True, HardSwish], [40, 240, 40, 1, 5, drop_prob, True, HardSwish], [40, 240, 40, 1, 5, drop_prob, True, HardSwish], [40, 120, 48, 1, 5, drop_prob, True, HardSwish], [48, 144, 96, 1, 5, drop_prob, True, HardSwish], [96, 288, 96, 2, 5, drop_prob, True, HardSwish], [96, 576, 96, 1, 5, drop_prob, True, HardSwish], [96, 576, 96, 1, 5, drop_prob, True, HardSwish]]
        self.bottlenecks_setting = self.bottlenecks_setting_small if small else self.bottlenecks_setting_large
        for l in self.bottlenecks_setting:
            l[0] = _make_divisible(l[0] * self.scale, 8)
            l[1] = _make_divisible(l[1] * self.scale, 8)
            l[2] = _make_divisible(l[2] * self.scale, 8)
        self.conv1 = nn.Conv2d(in_channels, self.bottlenecks_setting[0][0], kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.bottlenecks_setting[0][0])
        self.act1 = HardSwish(inplace=True)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = self._make_bottlenecks()
        self.last_exp2 = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        if small:
            self.last_exp1 = _make_divisible(576 * self.scale, 8)
            self.last_block = LastBlockSmall(self.bottlenecks_setting[-1][2], num_classes, self.last_exp1, self.last_exp2)
        else:
            self.last_exp1 = _make_divisible(960 * self.scale, 8)
            self.last_block = LastBlockLarge(self.bottlenecks_setting[-1][2], num_classes, self.last_exp1, self.last_exp2)

    def _make_bottlenecks(self):
        layers = []
        modules = OrderedDict()
        stage_name = 'Bottleneck'
        for i, setup in enumerate(self.bottlenecks_setting):
            name = stage_name + '_{}'.format(i)
            module = LinearBottleneck(setup[0], setup[2], setup[1], k=setup[4], stride=setup[3], drop_prob=setup[5], num_steps=self.num_steps, start_step=self.start_step, activation=setup[7], act_params={'inplace': True}, SE=setup[6])
            modules[name] = module
            if setup[3] == 2:
                layer = nn.Sequential(modules)
                layers.append(layer)
                modules = OrderedDict()
        if len(modules):
            layer = nn.Sequential(modules)
            layers.append(layer)
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last_block(x)
        return x


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
    if 'inplace' in kwargs and activation_name in {ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_SELU, ACT_CELU, ACT_ELU}:
        act_params['inplace'] = kwargs['inplace']
    if 'slope' in kwargs and activation_name in {ACT_LEAKY_RELU}:
        act_params['slope'] = kwargs['slope']
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
        super(GlobalAvgPool2d, self).__init__()
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

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, abn_block=ABN):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.abn = abn_block(planes)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.abn(x)
        return x


class ASPP(nn.Module):

    def __init__(self, inplanes: int, output_stride: int, output_features: int, dropout=0.5, abn_block=ABN):
        super(ASPP, self).__init__()
        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPPModule(inplanes, output_features, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, output_features, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, output_features, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, output_features, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(inplanes, output_features, 1, stride=1, bias=False), abn_block(output_features))
        self.conv1 = nn.Conv2d(output_features * 5, output_features, 1, bias=False)
        self.abn1 = abn_block(output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.abn1(x)
        return self.dropout(x)


class DeeplabV3Decoder(DecoderModule):

    def __init__(self, feature_maps: List[int], num_classes: int, output_stride=32, high_level_bottleneck=256, low_level_bottleneck=32, dropout=0.5, abn_block=ABN):
        super(DeeplabV3Decoder, self).__init__()
        self.aspp = ASPP(feature_maps[-1], output_stride, high_level_bottleneck, dropout=dropout, abn_block=abn_block)
        self.conv1 = nn.Conv2d(feature_maps[0], low_level_bottleneck, 1, bias=False)
        self.abn1 = abn_block(low_level_bottleneck)
        self.last_conv = nn.Sequential(nn.Conv2d(high_level_bottleneck + low_level_bottleneck, high_level_bottleneck, kernel_size=3, padding=1, bias=False), abn_block(high_level_bottleneck), nn.Dropout(dropout), nn.Conv2d(high_level_bottleneck, high_level_bottleneck, kernel_size=3, padding=1, bias=False), abn_block(high_level_bottleneck), nn.Dropout(dropout * 0.2), nn.Conv2d(high_level_bottleneck, num_classes, kernel_size=1))
        self.dsv = nn.Conv2d(high_level_bottleneck, num_classes, kernel_size=1)

    def forward(self, feature_maps: List[Tensor]) ->List[Tensor]:
        low_level_feat = feature_maps[0]
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.abn1(low_level_feat)
        high_level_features = feature_maps[-1]
        high_level_features = self.aspp(high_level_features)
        mask_dsv = self.dsv(high_level_features)
        high_level_features = F.interpolate(high_level_features, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        high_level_features = torch.cat([high_level_features, low_level_feat], dim=1)
        mask = self.last_conv(high_level_features)
        return [mask, mask_dsv]


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
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)


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

    def __init__(self, feature_maps: List[int], fpn_channels: int, context_block=FPNContextBlock, bottleneck_block=FPNBottleneckBlock, predict_block: Union[nn.Identity, conv1x1, nn.Module]=conv1x1, output_block: Union[nn.Identity, conv1x1, nn.Module]=nn.Identity, prediction_channels: int=None, upsample_block=nn.Upsample):
        """
        Create a new instance of FPN decoder with concatenation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param fpn_channels: FPN channels
        :param context_block:
        :param bottleneck_block:
        :param predict_block:
        :param output_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()
        self.context = context_block(feature_maps[-1], fpn_channels)
        self.bottlenecks = nn.ModuleList([bottleneck_block(in_channels, fpn_channels) for in_channels in reversed(feature_maps)])
        self.predicts = nn.ModuleList([predict_block(fpn_channels + fpn_channels, fpn_channels) for _ in reversed(feature_maps)])
        if issubclass(output_block, nn.Identity):
            self.channels = [fpn_channels] * len(feature_maps)
            self.outputs = nn.ModuleList([output_block() for _ in reversed(feature_maps)])
        else:
            self.channels = [prediction_channels] * len(feature_maps)
            self.outputs = nn.ModuleList([output_block(fpn_channels, prediction_channels) for _ in reversed(feature_maps)])
        if issubclass(upsample_block, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_block(scale_factor=2) for _ in reversed(feature_maps)])
        else:
            self.upsamples = nn.ModuleList([upsample_block(fpn_channels, fpn_channels) for in_channels in reversed(feature_maps)])

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

    def __init__(self, feature_maps: List[int], fpn_channels: int, context_block=FPNContextBlock, bottleneck_block=FPNBottleneckBlock, prediction_block: Union[nn.Identity, conv1x1, nn.Module]=nn.Identity, prediction_channels: int=None, upsample_block=nn.Upsample):
        """
        Create a new instance of FPN decoder with summation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param fpn_channels: FPN channels
        :param context_block:
        :param bottleneck_block:
        :param prediction_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()
        self.context = context_block(feature_maps[-1], fpn_channels)
        self.bottlenecks = nn.ModuleList([bottleneck_block(in_channels, fpn_channels) for in_channels in reversed(feature_maps)])
        if issubclass(prediction_block, nn.Identity):
            self.outputs = nn.ModuleList([prediction_block() for _ in reversed(feature_maps)])
            self.channels = [fpn_channels] * len(feature_maps)
        else:
            self.outputs = nn.ModuleList([prediction_block(fpn_channels, prediction_channels) for _ in reversed(feature_maps)])
            self.channels = [prediction_channels] * len(feature_maps)
        if issubclass(upsample_block, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_block(scale_factor=2) for _ in reversed(feature_maps)])
        else:
            self.upsamples = nn.ModuleList([upsample_block(fpn_channels, fpn_channels) for in_channels in reversed(feature_maps)])

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

    def __init__(self, feature_maps: List[int], output_channels: int, dropout=0.0, interpolation_mode='nearest', align_corners=None):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        features = sum(feature_maps)
        self.embedding = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(features)), ('relu', nn.ReLU(inplace=True))]))
        self.logits = nn.Sequential(OrderedDict([('drop', nn.Dropout2d(dropout)), ('final', nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=1))]))

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

    def __init__(self, feature_maps: List[int], decoder_features: Union[int, List[int]]=None, unet_block=UnetBlock, upsample_block: Union[nn.Upsample, nn.ConvTranspose2d]=None):
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
    def channels(self) ->List[int]:
        return self.output_filters

    def forward(self, feature_maps: List[torch.Tensor]) ->List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]
            if isinstance(upsample_block, nn.ConvTranspose2d):
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
    """
    """

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
            out = x * block_mask[:, (None), :, :]
            out = out * (block_mask.numel() / keeped)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, (None), :, :], kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)
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
            out = x * block_mask[:, (None), :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, (None), :, :, :], kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
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


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


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


class EncoderModule(nn.Module):

    def __init__(self, channels: List[int], strides: List[int], layers: List[int]):
        super().__init__()
        assert len(channels) == len(strides)
        self._layers = layers
        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    def forward(self, x: Tensor) ->List[Tensor]:
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            x = output
        return _take(output_features, self._layers)

    @property
    def channels(self) ->List[int]:
        return self._output_filters

    @property
    def strides(self) ->List[int]:
        return self._output_strides

    @property
    @pytorch_toolbelt_deprecated('This property is deprecated, please use .strides instead.')
    def output_strides(self) ->List[int]:
        return self.strides

    @property
    @pytorch_toolbelt_deprecated('This property is deprecated, please use .channels instead.')
    def output_filters(self) ->List[int]:
        return self.channels

    @property
    @pytorch_toolbelt_deprecated("This property is deprecated, please don't use it")
    def encoder_layers(self) ->List[nn.Module]:
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)

    def change_input_channels(self, input_channels: int, mode='auto'):
        """
        Change number of channels expected in the input tensor. By default,
        all encoders assume 3-channel image in BCHW notation with C=3.
        This method changes first convolution to have user-defined number of
        channels as input.
        """
        raise NotImplementedError


def make_n_channel_input(conv: nn.Conv2d, in_channels: int, mode='auto'):
    assert isinstance(conv, nn.Conv2d)
    if conv.in_channels == in_channels:
        warnings.warn('make_n_channel_input call is spurious')
        return conv
    new_conv = nn.Conv2d(in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None, padding_mode=conv.padding_mode)
    w = conv.weight
    if in_channels > conv.in_channels:
        n = math.ceil(in_channels / float(conv.in_channels))
        w = torch.cat([w] * n, dim=1)
        w = w[:, :in_channels, (...)]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    else:
        w = w[:, 0:in_channels, (...)]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    return new_conv


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
    def output_strides(self):
        return self._output_strides

    @property
    def output_filters(self):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.layer0.conv0 = make_n_channel_input(self.layer0.conv0, input_channels, mode)
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


def round_filters(filters: int, width_coefficient, depth_divisor, min_depth) ->int:
    """
    Calculate and round number of filters based on depth multiplier.
    """
    filters *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_multiplier):
    """
    Round number of filters based on depth multiplier.
    """
    if not depth_multiplier:
        return repeats
    return int(math.ceil(depth_multiplier * repeats))


class EfficientNetBlockArgs:

    def __init__(self, input_filters, output_filters, expand_ratio, repeats=1, kernel_size=3, stride=1, se_reduction=4, dropout=0.0, id_skip=True):
        self.in_channels = input_filters
        self.out_channels = output_filters
        self.expand_ratio = expand_ratio
        self.num_repeat = repeats
        self.se_reduction = se_reduction
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.width_coefficient = 1.0
        self.depth_coefficient = 1.0
        self.depth_divisor = 8
        self.min_filters = None
        self.id_skip = id_skip

    def __repr__(self):
        """Encode a block args class to a string representation."""
        args = ['r%d' % self.num_repeat, 'k%d' % self.kernel_size, 's%d' % self.stride, 'e%s' % self.expand_ratio, 'i%d' % self.in_channels, 'o%d' % self.out_channels]
        if self.se_reduction > 0:
            args.append('se%s' % self.se_reduction)
        return '_'.join(args)

    def copy(self):
        return deepcopy(self)

    def scale(self, width_coefficient: float, depth_coefficient: float, depth_divisor: float=8.0, min_filters: int=None):
        copy = self.copy()
        copy.in_channels = round_filters(self.in_channels, width_coefficient, depth_divisor, min_filters)
        copy.out_channels = round_filters(self.out_channels, width_coefficient, depth_divisor, min_filters)
        copy.num_repeat = round_repeats(self.num_repeat, depth_coefficient)
        copy.width_coefficient = width_coefficient
        copy.depth_coefficient = depth_coefficient
        copy.depth_divisor = depth_divisor
        copy.min_filters = min_filters
        return copy

    @staticmethod
    def B0():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.0) for p in params]
        return params

    @staticmethod
    def B1():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.1) for p in params]
        return params

    @staticmethod
    def B2():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.1, depth_coefficient=1.2) for p in params]
        return params

    @staticmethod
    def B3():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.2, depth_coefficient=1.4) for p in params]
        return params

    @staticmethod
    def B4():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.4, depth_coefficient=1.8) for p in params]
        return params

    @staticmethod
    def B5():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.6, depth_coefficient=2.2) for p in params]
        return params

    @staticmethod
    def B6():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=1.8, depth_coefficient=2.6) for p in params]
        return params

    @staticmethod
    def B7():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=2.0, depth_coefficient=3.1) for p in params]
        return params


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


def drop_connect(inputs, p, training):
    """
    Drop connect implementation.
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args: EfficientNetBlockArgs, abn_block: ABN):
        super().__init__()
        self.has_se = block_args.se_reduction is not None
        self.id_skip = block_args.id_skip
        self.expand_ratio = block_args.expand_ratio
        self.stride = block_args.stride
        inp = block_args.in_channels
        oup = block_args.in_channels * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.abn0 = abn_block(oup)
        self.depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=block_args.kernel_size, padding=block_args.kernel_size // 2, stride=block_args.stride, bias=False)
        self.abn1 = abn_block(oup)
        if self.has_se:
            se_channels = max(1, inp // block_args.se_reduction)
            self.se_block = SpatialGate2d(oup, squeeze_channels=se_channels)
        self.project_conv = nn.Conv2d(in_channels=oup, out_channels=block_args.out_channels, kernel_size=1, bias=False)
        self.abn2 = abn_block(block_args.out_channels)
        self.input_filters = block_args.in_channels
        self.output_filters = block_args.out_channels
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self.expand_ratio != 1:
            x = self.abn0(self.expand_conv(inputs))
        x = self.abn1(self.depthwise_conv(x))
        if self.has_se:
            x = self.se_block(x)
        x = self.abn2(self.project_conv(x))
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNetStem(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, abn_block: ABN):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.abn = abn_block(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class EfficientNetEncoder(EncoderModule):

    @staticmethod
    def build_layer(block_args: List[EfficientNetBlockArgs], abn_block: ABN):
        blocks = []
        for block_index, cfg in enumerate(block_args):
            module = []
            module.append(('mbconv_0', MBConvBlock(cfg, abn_block)))
            if cfg.num_repeat > 1:
                cfg = cfg.copy()
                cfg.stride = 1
                cfg.in_channels = cfg.out_channels
                for i in range(cfg.num_repeat - 1):
                    module.append((f'mbconv_{i + 1}', MBConvBlock(cfg, abn_block)))
            module = nn.Sequential(OrderedDict(module))
            blocks.append((f'block_{block_index}', module))
        return nn.Sequential(OrderedDict(blocks))

    def __init__(self, encoder_config: List[EfficientNetBlockArgs], in_channels=3, activation=ACT_SWISH, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        blocks2layers = [[encoder_config[0], encoder_config[1]], [encoder_config[2]], [encoder_config[3], encoder_config[4]], [encoder_config[5], encoder_config[6]]]
        filters = [encoder_config[0].in_channels] + [cfg[-1].out_channels for cfg in blocks2layers]
        strides = [2, 4, 8, 16, 32]
        abn_block = partial(ABN, activation=activation)
        super().__init__(filters, strides, layers)
        self.stem = EfficientNetStem(in_channels, encoder_config[0].in_channels, abn_block)
        self.layers = nn.ModuleList([self.build_layer(cfg, abn_block) for cfg in blocks2layers])

    def forward(self, x: torch.Tensor) ->List[torch.Tensor]:
        x = self.stem(x)
        output_features = [x]
        for layer in self.layers:
            output = layer(x)
            output_features.append(output)
            x = output
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.stem.conv = make_n_channel_input(self.stem.conv, input_channels, mode)
        return self


class EfficientNetB0Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B0(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB1Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B1(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB2Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B2(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB3Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B3(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB4Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B4(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB5Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B5(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB6Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B6(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB7Encoder(EfficientNetEncoder):

    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B7(), in_channels=in_channels, activation=activation, layers=layers)


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

    def __init__(self, depth: int, input_features: int, features, increase=0, activation=nn.ReLU, repeats=1, pooling_block=nn.MaxPool2d):
        super(HGBlock, self).__init__()
        nf = features + increase
        self.down = pooling_block(kernel_size=2, padding=0, stride=2)
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
            self.low2 = HGBlock(depth - 1, nf, nf, increase=increase, activation=activation)
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

    def __repr__(self):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.layer0.conv0 = make_n_channel_input(self.layer0.conv0, input_channels, mode)
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
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return self._output_strides

    @property
    def output_filters(self):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
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

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
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

    def change_input_channels(self, input_channels: int, mode='auto'):
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

    def change_input_channels(self, input_channels: int, mode='auto'):
        self.stem.conv_bn_relu_1.conv = make_n_channel_input(self.stem.conv_bn_relu_1.conv, input_channels, mode)


class FPNFuse(nn.Module):

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[Tensor]):
        layers = []
        dst_size = features[0].size()[-2:]
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
        dst_size = features[0].size()[-2:]
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
        return (input - self.mean) * self.std


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
        super(GlobalMaxPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


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
    """
    """

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


def bilinear_upsample_initializer(x):
    cc = x.size(2) // 2
    cr = x.size(3) // 2
    for i in range(x.size(2)):
        for j in range(x.size(3)):
            x[..., i, j] = hypot(cc - i, cr - j)
    y = 1 - x / x.sum(dim=(2, 3), keepdim=True)
    y = y / y.sum(dim=(2, 3), keepdim=True)
    return y


def icnr_init(tensor: torch.Tensor, upscale_factor=2, initializer=nn.init.kaiming_normal):
    """Fill the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        initializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / upscale_factor ** 2)] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = initializer(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)
    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


class DepthToSpaceUpsample2d(nn.Module):
    """
    NOTE: This block is not fully ready yet. Need to figure out how to correctly initialize
    default weights to have bilinear upsample identical to OpenCV results

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, features: int, scale_factor: int=2):
        super().__init__()
        self.n = 2 ** scale_factor
        self.conv = nn.Conv2d(features, features * self.n, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.data = icnr_init(self.conv.weight, upscale_factor=scale_factor, initializer=bilinear_upsample_initializer)
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


class DeconvolutionUpsample2d(nn.Module):

    def __init__(self, in_channels: int, n=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=2)

    def forward(self, x: Tensor) ->Tensor:
        return self.conv(x)


class ResidualDeconvolutionUpsample2d(nn.Module):

    def __init__(self, in_channels, scale_factor=2, n=4):
        if scale_factor != 2:
            raise NotImplementedError('Scale factor other than 2 is not implemented')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=scale_factor, output_padding=1)
        self.residual = BilinearAdditiveUpsample2d(in_channels, scale_factor=scale_factor, n=n)

    def forward(self, x: Tensor) ->Tensor:
        residual_up = self.residual(x)
        return self.conv(x, output_size=residual_up.size()) + residual_up


class LossModule(nn.Module):

    def __init__(self, output_key: str, target_key: str, loss_fn):
        super().__init__()
        self.output_key = output_key
        self.target_ley = target_key
        self.loss_fn = loss_fn

    def forward(self, outputs, targets):
        return self.loss_fn(outputs[self.output_key], targets[self.target_ley])


class LossWrapper(nn.Module):
    """
    A wrapper module around model that computes one or many loss functions and extends output dictionary with
    their values. The point of this wrapper is that loss computed on each GPU node in parallel.

    Usage:
    >>> from catalyst.dl import SupervisedRunner
    >>> runner = SupervisedRunner(input_key=None, output_key=None, device="cuda")
    >>> runner._default_experiment = ParallelLossSupervisedExperiment
    >>> loss_modules = {
    >>>     "my_loss": LossModule(
    >>>         output_key="logits",
    >>>         target_key="targets",
    >>>         loss_fn=nn.BCEWithLogitsLoss(),
    >>>     )}
    >>> loss_callback = PassthroughCriterionCallback("my_loss")
    >>> runner.train(
    >>>     callbacks=[loss_callback, ...]
    >>>     model=LossWrapper(model, "image", loss_modules),
    >>>     ...)

    Note, that SupervisedRunner adds default CriterionCallback
    """

    def __init__(self, model: nn.Module, input_key: str, losses: Dict[str, LossModule]):
        super().__init__()
        self.model = model
        self.input_key = input_key
        self.loss_names = list(losses.keys())
        self.losses = nn.ModuleList([losses[key] for key in self.loss_names])

    def forward(self, **input: Dict[str, Tensor]) ->Dict[str, Tensor]:
        output: Dict[str, Tensor] = self.model(input[self.input_key])
        for output_loss_key, loss in zip(self.loss_names, self.losses):
            output[output_loss_key] = loss(output, input)
        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()


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

