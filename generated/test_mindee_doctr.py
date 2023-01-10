import sys
_module = sys.modules[__name__]
del sys
verify_pr_labels = _module
config = _module
main = _module
detection = _module
kie = _module
ocr = _module
recognition = _module
schemas = _module
vision = _module
conftest = _module
test_detection = _module
test_kie = _module
test_ocr = _module
test_recognition = _module
app = _module
pytorch = _module
tensorflow = _module
conf = _module
doctr = _module
datasets = _module
cord = _module
base = _module
pytorch = _module
doc_artefacts = _module
funsd = _module
generator = _module
pytorch = _module
ic03 = _module
ic13 = _module
iiit5k = _module
imgur5k = _module
loader = _module
mjsynth = _module
sroie = _module
svhn = _module
svt = _module
synthtext = _module
utils = _module
vocabs = _module
file_utils = _module
io = _module
elements = _module
html = _module
image = _module
pytorch = _module
pdf = _module
reader = _module
models = _module
_utils = _module
artefacts = _module
barcode = _module
face = _module
builder = _module
classification = _module
magc_resnet = _module
pytorch = _module
mobilenet = _module
pytorch = _module
predictor = _module
pytorch = _module
resnet = _module
pytorch = _module
vgg = _module
pytorch = _module
vit = _module
pytorch = _module
zoo = _module
core = _module
pytorch = _module
differentiable_binarization = _module
pytorch = _module
linknet = _module
pytorch = _module
pytorch = _module
factory = _module
hub = _module
kie_predictor = _module
pytorch = _module
modules = _module
transformer = _module
pytorch = _module
vision_transformer = _module
pytorch = _module
obj_detection = _module
faster_rcnn = _module
pytorch = _module
pytorch = _module
preprocessor = _module
pytorch = _module
tensorflow = _module
crnn = _module
pytorch = _module
master = _module
pytorch = _module
pytorch = _module
sar = _module
pytorch = _module
vitstr = _module
pytorch = _module
pytorch = _module
transforms = _module
functional = _module
pytorch = _module
base = _module
pytorch = _module
common_types = _module
data = _module
fonts = _module
geometry = _module
metrics = _module
multithreading = _module
repr = _module
visualization = _module
latency_pytorch = _module
latency_tensorflow = _module
train_pytorch = _module
train_tensorflow = _module
evaluate_pytorch = _module
evaluate_tensorflow = _module
latency_pytorch = _module
train_pytorch = _module
latency_pytorch = _module
train_pytorch = _module
evaluate_pytorch = _module
latency_pytorch = _module
train_pytorch = _module
analyze = _module
collect_env = _module
detect_artefacts = _module
detect_text = _module
evaluate = _module
evaluate_kie = _module
setup = _module
test_core = _module
test_datasets = _module
test_datasets_utils = _module
test_io = _module
test_io_elements = _module
test_models = _module
test_models_artefacts = _module
test_models_builder = _module
test_models_detection = _module
test_models_recognition_predictor = _module
test_models_recognition_utils = _module
test_transforms = _module
test_utils_data = _module
test_utils_fonts = _module
test_utils_geometry = _module
test_utils_metrics = _module
test_utils_multithreading = _module
test_utils_visualization = _module
test_datasets_pt = _module
test_file_utils_pt = _module
test_io_image_pt = _module
test_models_classification_pt = _module
test_models_detection_pt = _module
test_models_factory = _module
test_models_obj_detection_pt = _module
test_models_preprocessor_pt = _module
test_models_recognition_pt = _module
test_models_utils_pt = _module
test_models_zoo_pt = _module
test_transforms_pt = _module
test_datasets_loader_tf = _module
test_datasets_tf = _module
test_file_utils_tf = _module
test_io_image_tf = _module
test_models_classification_tf = _module
test_models_detection_tf = _module
test_models_preprocessor_tf = _module
test_models_recognition_tf = _module
test_models_utils_tf = _module
test_models_zoo_tf = _module
test_transforms_tf = _module

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


import matplotlib.pyplot as plt


import numpy as np


import torch


from copy import deepcopy


from typing import Any


from typing import List


from typing import Tuple


from torch.utils.data._utils.collate import default_collate


import logging


from torchvision.transforms.functional import to_tensor


import math


from functools import partial


from typing import Dict


from typing import Optional


from torch import nn


from torchvision.models import mobilenetv3


from typing import Union


from typing import Callable


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import ResNet as TVResNet


from torchvision.models.resnet import resnet18 as tv_resnet18


from torchvision.models.resnet import resnet34 as tv_resnet34


from torchvision.models.resnet import resnet50 as tv_resnet50


from torchvision.models import vgg as tv_vgg


from torch import Tensor


from torch.nn.functional import max_pool2d


from torch.nn import functional as F


from torchvision.models import resnet34


from torchvision.models import resnet50


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.ops.deform_conv import DeformConv2d


from torchvision.models.detection import FasterRCNN


from torchvision.models.detection import faster_rcnn


from torchvision.transforms import functional as F


from torchvision.transforms import transforms as T


import tensorflow as tf


from itertools import groupby


from typing import Sequence


import random


from torch.nn.functional import pad


import time


from torch.nn.functional import cross_entropy


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiplicativeLR


from torch.optim.lr_scheduler import OneCycleLR


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torchvision.transforms import ColorJitter


from torchvision.transforms import Compose


from torchvision.transforms import GaussianBlur


from torchvision.transforms import Grayscale


from torchvision.transforms import InterpolationMode


from torchvision.transforms import Normalize


from torchvision.transforms import RandomRotation


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


import re


from collections import namedtuple


class MAGC(nn.Module):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        attn_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(self, inplanes: int, headers: int=8, attn_scale: bool=False, ratio: float=0.0625, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.headers = headers
        self.inplanes = inplanes
        self.attn_scale = attn_scale
        self.planes = int(inplanes * ratio)
        self.single_header_inplanes = int(inplanes / headers)
        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.transform = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        batch, _, height, width = inputs.size()
        x = inputs.view(batch * self.headers, self.single_header_inplanes, height, width)
        shortcut = x
        shortcut = shortcut.view(batch * self.headers, self.single_header_inplanes, height * width)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch * self.headers, -1)
        if self.attn_scale and self.headers > 1:
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)
        context_mask = self.softmax(context_mask)
        context = (shortcut * context_mask.unsqueeze(1)).sum(-1)
        context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        transformed = self.transform(context)
        return inputs + transformed


def _addindent(s_, num_spaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ' + line) for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class NestedObject:
    _children_names: List[str]

    def extra_repr(self) ->str:
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        if hasattr(self, '_children_names'):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ',\n'.join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f'\n{child_str},', 2) + '\n'
                    child_str = f'[{child_str}]'
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append('(' + key + '): ' + child_str)
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class Resize(T.Resize):

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation=F.InterpolationMode.BILINEAR, preserve_aspect_ratio: bool=False, symmetric_pad: bool=False) ->None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError('size should be either a tuple, a list or an int')

    def forward(self, img: torch.Tensor, target: Optional[np.ndarray]=None) ->Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]
        if not self.preserve_aspect_ratio or target_ratio == actual_ratio and isinstance(self.size, (tuple, list)):
            if target is not None:
                return super().forward(img), target
            return super().forward(img)
        else:
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = self.size[0], max(int(self.size[0] / actual_ratio), 1)
                else:
                    tmp_size = max(int(self.size[1] * actual_ratio), 1), self.size[1]
            elif isinstance(self.size, int):
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = max(int(self.size * actual_ratio), 1), self.size
                else:
                    tmp_size = self.size, max(int(self.size / actual_ratio), 1)
            img = F.resize(img, tmp_size, self.interpolation)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                _pad = 0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2]
                if self.symmetric_pad:
                    half_pad = math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2)
                    _pad = half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1]
                img = pad(img, _pad)
            if target is not None:
                if self.preserve_aspect_ratio:
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError
                return img, target
            return img

    def __repr__(self) ->str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f', preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}'
        return f'{self.__class__.__name__}({_repr})'


ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}


class PreProcessor(NestedObject):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
    """
    _children_names: List[str] = ['resize', 'normalize']

    def __init__(self, output_size: Tuple[int, int], batch_size: int, mean: Tuple[float, float, float]=(0.5, 0.5, 0.5), std: Tuple[float, float, float]=(1.0, 1.0, 1.0), fp16: bool=False, **kwargs: Any) ->None:
        self.batch_size = batch_size
        self.resize = Resize(output_size, **kwargs)
        self.normalize = Normalize(mean, std)

    def batch_inputs(self, samples: List[tf.Tensor]) ->List[tf.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples (tf.Tensor)

        Returns:
            list of batched samples
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [tf.stack(samples[idx * self.batch_size:min((idx + 1) * self.batch_size, len(samples))], axis=0) for idx in range(int(num_batches))]
        return batches

    def sample_transforms(self, x: Union[np.ndarray, tf.Tensor]) ->tf.Tensor:
        if x.ndim != 3:
            raise AssertionError('expected list of 3D Tensors')
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError('unsupported data type for numpy.ndarray')
            x = tf.convert_to_tensor(x)
        elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
            raise TypeError('unsupported data type for torch.Tensor')
        if x.dtype == tf.uint8:
            x = tf.image.convert_image_dtype(x, dtype=tf.float32)
        x = self.resize(x)
        return x

    def __call__(self, x: Union[tf.Tensor, np.ndarray, List[Union[tf.Tensor, np.ndarray]]]) ->List[tf.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)
        Returns:
            list of page batches
        """
        if isinstance(x, (np.ndarray, tf.Tensor)):
            if x.ndim != 4:
                raise AssertionError('expected 4D Tensor')
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError('unsupported data type for numpy.ndarray')
                x = tf.convert_to_tensor(x)
            elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
                raise TypeError('unsupported data type for torch.Tensor')
            if x.dtype == tf.uint8:
                x = tf.image.convert_image_dtype(x, dtype=tf.float32)
            if (x.shape[1], x.shape[2]) != self.resize.output_size:
                x = tf.image.resize(x, self.resize.output_size, method=self.resize.method)
            batches = [x]
        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, tf.Tensor)) for sample in x):
            samples = list(multithread_exec(self.sample_transforms, x))
            batches = self.batch_inputs(samples)
        else:
            raise TypeError(f'invalid input type: {type(x)}')
        batches = list(multithread_exec(self.normalize, batches))
        return batches


class CropOrientationPredictor(nn.Module):
    """Implements an object able to detect the reading direction of a text box.
    4 possible orientations: 0, 90, 180, 270 degrees counter clockwise.

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
    """

    def __init__(self, pre_processor: PreProcessor, model: nn.Module) ->None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, crops: List[Union[np.ndarray, torch.Tensor]]) ->List[int]:
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError('incorrect input shape: all pages are expected to be multi-channel 2D images.')
        processed_batches = self.pre_processor(crops)
        _device = next(self.model.parameters()).device
        predicted_batches = [self.model(batch) for batch in processed_batches]
        predicted_batches = [out_batch.argmax(dim=1).cpu().detach().numpy() for out_batch in predicted_batches]
        return [int(pred) for batch in predicted_batches for pred in batch]


def conv_sequence_pt(in_channels: int, out_channels: int, relu: bool=False, bn: bool=False, **kwargs: Any) ->List[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added

    Returns:
        list of layers
    """
    kwargs['bias'] = kwargs.get('bias', not bn)
    conv_seq: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, **kwargs)]
    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))
    if relu:
        conv_seq.append(nn.ReLU(inplace=True))
    return conv_seq


def resnet_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) ->List[nn.Module]:
    _layers: List[nn.Module] = []
    in_chan = in_channels
    s = stride
    for _ in range(num_blocks):
        downsample = None
        if in_chan != out_channels:
            downsample = nn.Sequential(*conv_sequence_pt(in_chan, out_channels, False, True, kernel_size=1, stride=s))
        _layers.append(BasicBlock(in_chan, out_channels, stride=s, downsample=downsample))
        in_chan = out_channels
        s = 1
    return _layers


class ResNet(nn.Sequential):
    """Implements a ResNet-31 architecture from `"Show, Attend and Read:A Simple and Strong Baseline for Irregular
    Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        origin_stem: whether to use the orginal ResNet stem or ResNet-31's
        stem_channels: number of output channels of the stem convolutions
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
    """

    def __init__(self, num_blocks: List[int], output_channels: List[int], stage_stride: List[int], stage_conv: List[bool], stage_pooling: List[Optional[Tuple[int, int]]], origin_stem: bool=True, stem_channels: int=64, attn_module: Optional[Callable[[int], nn.Module]]=None, include_top: bool=True, num_classes: int=1000, cfg: Optional[Dict[str, Any]]=None) ->None:
        _layers: List[nn.Module]
        if origin_stem:
            _layers = [*conv_sequence_pt(3, stem_channels, True, True, kernel_size=7, padding=3, stride=2), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            _layers = [*conv_sequence_pt(3, stem_channels // 2, True, True, kernel_size=3, padding=1), *conv_sequence_pt(stem_channels // 2, stem_channels, True, True, kernel_size=3, padding=1), nn.MaxPool2d(2)]
        in_chans = [stem_channels] + output_channels[:-1]
        for n_blocks, in_chan, out_chan, stride, conv, pool in zip(num_blocks, in_chans, output_channels, stage_stride, stage_conv, stage_pooling):
            _stage = resnet_stage(in_chan, out_chan, n_blocks, stride)
            if attn_module is not None:
                _stage.append(attn_module(out_chan))
            if conv:
                _stage.extend(conv_sequence_pt(out_chan, out_chan, True, True, kernel_size=3, padding=1))
            if pool is not None:
                _stage.append(nn.MaxPool2d(pool))
            _layers.append(nn.Sequential(*_stage))
        if include_top:
            _layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(output_channels[-1], num_classes, bias=True)])
        super().__init__(*_layers)
        self.cfg = cfg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ClassifierHead(nn.Module):
    """Classifier head for Vision Transformer

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
    """

    def __init__(self, in_channels: int, num_classes: int) ->None:
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.head(x[:, 0])


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention"""
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = torch.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, num_heads: int, d_model: int, dropout: float=0.1) ->None:
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) ->torch.Tensor:
        batch_size = query.size(0)
        query, key, value = [linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for linear, x in zip(self.linear_layers, (query, key, value))]
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model: int, ffd: int, dropout: float=0.1, activation_fct: Callable[[Any], Any]=nn.ReLU()) ->None:
        super().__init__(nn.Linear(d_model, ffd), activation_fct, nn.Dropout(p=dropout), nn.Linear(ffd, d_model), nn.Dropout(p=dropout))


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(self, num_layers: int, num_heads: int, d_model: int, dff: int, dropout: float, activation_fct: Callable[[Any], Any]=nn.ReLU()) ->None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-05)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.position_feed_forward = nn.ModuleList([PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        output = x
        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm_output(output)


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(self, input_shape: Tuple[int, int, int], embed_dim: int) ->None:
        super().__init__()
        channels, height, width = input_shape
        self.patch_size = height // (height // 8), width // (width // 8)
        self.grid_size = self.patch_size[0], self.patch_size[1]
        self.num_patches = self.patch_size[0] * self.patch_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.proj = nn.Linear(channels * self.patch_size[0] * self.patch_size[1], embed_dim)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) ->torch.Tensor:
        """
        100 % borrowed from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.positions.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.positions
        class_pos_embed = self.positions[:, 0]
        patch_pos_embed = self.positions[:, 1:]
        dim = embeddings.shape[-1]
        h0 = float(height // self.patch_size[0])
        w0 = float(width // self.patch_size[1])
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)), mode='bilinear', align_corners=False, recompute_scale_factor=True)
        assert int(h0) == patch_pos_embed.shape[-2], "height of interpolated patch embedding doesn't match"
        assert int(w0) == patch_pos_embed.shape[-1], "width of interpolated patch embedding doesn't match"
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, 'Image height must be divisible by patch height'
        assert W % self.patch_size[1] == 0, 'Image width must be divisible by patch width'
        x = x.reshape(B, C, H // self.patch_size[0], self.patch_size[0], W // self.patch_size[1], self.patch_size[1])
        patches = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2, 4)
        patches = self.proj(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, patches], dim=1)
        embeddings += self.interpolate_pos_encoding(embeddings, H, W)
        return embeddings


class VisionTransformer(nn.Sequential):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        input_shape: size of the input image
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    def __init__(self, d_model: int, num_layers: int, num_heads: int, ffd_ratio: int, input_shape: Tuple[int, int, int]=(3, 32, 32), dropout: float=0.0, num_classes: int=1000, include_top: bool=True, cfg: Optional[Dict[str, Any]]=None) ->None:
        _layers: List[nn.Module] = [PatchEmbedding(input_shape, d_model), EncoderBlock(num_layers, num_heads, d_model, d_model * ffd_ratio, dropout, nn.GELU())]
        if include_top:
            _layers.append(ClassifierHead(d_model, num_classes))
        super().__init__(*_layers)
        self.cfg = cfg


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels: List[int], out_channels: int, deform_conv: bool=False) ->None:
        super().__init__()
        out_chans = out_channels // len(in_channels)
        conv_layer = DeformConv2d if deform_conv else nn.Conv2d
        self.in_branches = nn.ModuleList([nn.Sequential(conv_layer(chans, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for idx, chans in enumerate(in_channels)])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_branches = nn.ModuleList([nn.Sequential(conv_layer(out_channels, out_chans, 3, padding=1, bias=False), nn.BatchNorm2d(out_chans), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2 ** idx, mode='bilinear', align_corners=True)) for idx, chans in enumerate(in_channels)])

    def forward(self, x: List[torch.Tensor]) ->torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        _x: List[torch.Tensor] = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: List[torch.Tensor] = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]
        return torch.cat(out, dim=1)


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(self, box_thresh: float=0.5, bin_thresh: float=0.5, assume_straight_pages: bool=True) ->None:
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)

    def extra_repr(self) ->str:
        return f'bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}'

    @staticmethod
    def box_score(pred: np.ndarray, points: np.ndarray, assume_straight_pages: bool=True) ->float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]
        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin:ymax + 1, xmin:xmax + 1].mean()
        else:
            mask: np.ndarray = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray) ->np.ndarray:
        raise NotImplementedError

    def __call__(self, proba_map) ->List[List[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W, C)

        Returns:
            list of N class predictions (for each input sample), where each class predictions is a list of C tensors
        of shape (*, 5) or (*, 6)
        """
        if proba_map.ndim != 4:
            raise AssertionError(f'arg `proba_map` is expected to be 4-dimensional, got {proba_map.ndim}.')
        bin_map = [[cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel) for idx in range(proba_map.shape[-1])] for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)]
        return [[self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])] for pmaps, bmaps in zip(proba_map, bin_map)]


Point2D = Tuple[float, float]


Polygon = List[Point2D]


class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
        unclip ratio: ratio used to unshrink polygons
        min_size_box: minimal length (pix) to keep a box
        max_candidates: maximum boxes to consider in a single page
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """

    def __init__(self, box_thresh: float=0.1, bin_thresh: float=0.3, assume_straight_pages: bool=True) ->None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.5 if assume_straight_pages else 2.2

    def polygon_to_box(self, points: np.ndarray) ->np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            _points = [_points[idx]]
        expanded_points: np.ndarray = np.asarray(_points)
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray) ->np.ndarray:
        """Compute boxes from a bitmap/pred_map

        Args:
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        min_size_box = 1 + int(height / 512)
        boxes: List[Union[np.ndarray, List[float]]] = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
                continue
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: np.ndarray = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)
            if score < self.box_thresh:
                continue
            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
            if self.assume_straight_pages:
                if _box is None or _box[2] < min_size_box or _box[3] < min_size_box:
                    continue
            elif np.linalg.norm(_box[2, :] - _box[0, :], axis=-1) < min_size_box:
                continue
            if self.assume_straight_pages:
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                if not isinstance(_box, np.ndarray) and _box.shape == (4, 2):
                    raise AssertionError('When assume straight pages is false a box is a (4, 2) array (polygon)')
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(_box)
        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 4, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """
    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 3
    assume_straight_pages: bool = True

    @staticmethod
    def compute_distance(xs: np.ndarray, ys: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float=1e-07) ->float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment

        Returns:
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2) + eps)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result

    def draw_thresh_map(self, polygon: np.ndarray, canvas: np.ndarray, mask: np.ndarray) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
            polygon : array of coord., to draw the boundary of the polygon
            canvas : threshold map to fill with polygons
            mask : mask for training on threshold polygons
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError('polygon should be a 2 dimensional array of coords')
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon: np.ndarray = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        xs: np.ndarray = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys: np.ndarray = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=polygon.dtype)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(1 - distance_map[ymin_valid - ymin:ymax_valid - ymin + 1, xmin_valid - xmin:xmax_valid - xmin + 1], canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        return polygon, canvas, mask

    def build_target(self, target: List[Dict[str, np.ndarray]], output_shape: Tuple[int, int, int, int], channels_last: bool=True) ->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")
        input_dtype = next(iter(target[0].values())).dtype if len(target) > 0 else np.float32
        if channels_last:
            h, w = output_shape[1:-1]
            target_shape = output_shape[0], output_shape[-1], h, w
        else:
            h, w = output_shape[-2:]
            target_shape = output_shape
        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones(target_shape, dtype=bool)
        thresh_target: np.ndarray = np.zeros(target_shape, dtype=np.float32)
        thresh_mask: np.ndarray = np.ones(target_shape, dtype=np.uint8)
        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([abs_boxes[:, [0, 1]], abs_boxes[:, [0, 3]], abs_boxes[:, [2, 3]], abs_boxes[:, [2, 1]]], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                for box, box_size, poly in zip(abs_boxes, boxes_size, polys):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrinked.astype(np.int32)], 1)
                    poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx] = self.draw_thresh_map(poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx])
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
            thresh_target = thresh_target.transpose((0, 2, 3, 1))
            thresh_mask = thresh_mask.transpose((0, 2, 3, 1))
        thresh_target = thresh_target.astype(input_dtype) * (self.thresh_max - self.thresh_min) + self.thresh_min
        seg_target = seg_target.astype(input_dtype)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(input_dtype)
        thresh_mask = thresh_mask.astype(bool)
        return seg_target, seg_mask, thresh_target, thresh_mask


class LinkNetFPN(nn.Module):

    def __init__(self, layer_shapes: List[Tuple[int, int, int]]) ->None:
        super().__init__()
        strides = [(1 if in_shape[-1] == out_shape[-1] else 2) for in_shape, out_shape in zip(layer_shapes[:-1], layer_shapes[1:])]
        chans = [shape[0] for shape in layer_shapes]
        _decoder_layers = [self.decoder_block(ochan, ichan, stride) for ichan, ochan, stride in zip(chans[:-1], chans[1:], strides)]
        self.decoders = nn.ModuleList(_decoder_layers)

    @staticmethod
    def decoder_block(in_chan: int, out_chan: int, stride: int) ->nn.Sequential:
        """Creates a LinkNet decoder block"""
        mid_chan = in_chan // 4
        return nn.Sequential(nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False), nn.BatchNorm2d(mid_chan), nn.ReLU(inplace=True), nn.ConvTranspose2d(mid_chan, mid_chan, 3, padding=1, output_padding=stride - 1, stride=stride, bias=False), nn.BatchNorm2d(mid_chan), nn.ReLU(inplace=True), nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True))

    def forward(self, feats: List[torch.Tensor]) ->torch.Tensor:
        out = feats[-1]
        for decoder, fmap in zip(self.decoders[::-1], feats[:-1][::-1]):
            out = decoder(out) + fmap
        out = self.decoders[0](out)
        return out


class LinkNetPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
        bin_thresh: threshold used to binzarized p_map at inference time
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: whether the inputs were expected to have horizontal text elements
    """

    def __init__(self, bin_thresh: float=0.1, box_thresh: float=0.1, assume_straight_pages: bool=True) ->None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.2

    def polygon_to_box(self, points: np.ndarray) ->np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            _points = [_points[idx]]
        expanded_points: np.ndarray = np.asarray(_points)
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray) ->np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
            np tensor boxes for the bitmap, each box is a 6-element list
                containing x, y, w, h, alpha, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes: List[Union[np.ndarray, List[float]]] = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: np.ndarray = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)
            if score < self.box_thresh:
                continue
            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
            if self.assume_straight_pages:
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(_box)
        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 4, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.cfg = cfg


class _LinkNet(BaseModel):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        out_chan: number of channels for the output
    """
    min_size_box: int = 3
    assume_straight_pages: bool = True
    shrink_ratio = 0.5

    def build_target(self, target: List[Dict[str, np.ndarray]], output_shape: Tuple[int, int, int], channels_last: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        """Build the target, and it's mask to be used from loss computation.

        Args:
            target: target coming from dataset
            output_shape: shape of the output of the model without batch_size
            channels_last: whether channels are last or not

        Returns:
            the new formatted target and the mask
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")
        h: int
        w: int
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = len(target), num_classes, h, w
        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones(target_shape, dtype=bool)
        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([abs_boxes[:, [0, 1]], abs_boxes[:, [0, 3]], abs_boxes[:, [2, 3]], abs_boxes[:, [2, 1]]], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3] + 1, box[0]:box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrunken.astype(np.int32)], 1)
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
        return seg_target, seg_mask


class DetectionPredictor(nn.Module):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    def __init__(self, pre_processor: PreProcessor, model: nn.Module) ->None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, pages: List[Union[np.ndarray, torch.Tensor]], **kwargs: Any) ->List[np.ndarray]:
        if any(page.ndim != 3 for page in pages):
            raise ValueError('incorrect input shape: all pages are expected to be multi-channel 2D images.')
        processed_batches = self.pre_processor(pages)
        _device = next(self.model.parameters()).device
        predicted_batches = [self.model(batch, return_preds=True, **kwargs)['preds'] for batch in processed_batches]
        return [pred for batch in predicted_batches for pred in batch]


class Element(NestedObject):
    """Implements an abstract document element with exporting and text rendering capabilities"""
    _children_names: List[str] = []
    _exported_keys: List[str] = []

    def __init__(self, **kwargs: Any) ->None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) ->Dict[str, Any]:
        """Exports the object into a nested dict format"""
        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for children_name in self._children_names:
            if children_name in ['predictions']:
                export_dict[children_name] = {k: [item.export() for item in c] for k, c in getattr(self, children_name).items()}
            else:
                export_dict[children_name] = [c.export() for c in getattr(self, children_name)]
        return export_dict

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        raise NotImplementedError

    def render(self) ->str:
        raise NotImplementedError


BoundingBox = Tuple[Point2D, Point2D]


class Artefact(Element):
    """Implements a non-textual element

    Args:
        artefact_type: the type of artefact
        confidence: the confidence of the type prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size.
    """
    _exported_keys: List[str] = ['geometry', 'type', 'confidence']
    _children_names: List[str] = []

    def __init__(self, artefact_type: str, confidence: float, geometry: BoundingBox) ->None:
        super().__init__()
        self.geometry = geometry
        self.type = artefact_type
        self.confidence = confidence

    def render(self) ->str:
        """Renders the full text of the element"""
        return f'[{self.type.upper()}]'

    def extra_repr(self) ->str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Word(Element):
    """Implements a word element

    Args:
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
        the page's size
    """
    _exported_keys: List[str] = ['value', 'confidence', 'geometry']
    _children_names: List[str] = []

    def __init__(self, value: str, confidence: float, geometry: Union[BoundingBox, np.ndarray]) ->None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry

    def render(self) ->str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) ->str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


def resolve_enclosing_bbox(bboxes: Union[List[BoundingBox], np.ndarray]) ->Union[BoundingBox, np.ndarray]:
    """Compute enclosing bbox either from:

    - an array of boxes: (*, 5), where boxes have this shape:
    (xmin, ymin, xmax, ymax, score)

    - a list of BoundingBox

    Return a (1, 5) array (enclosing boxarray), or a BoundingBox
    """
    if isinstance(bboxes, np.ndarray):
        xmin, ymin, xmax, ymax, score = np.split(bboxes, 5, axis=1)
        return np.array([xmin.min(), ymin.min(), xmax.max(), ymax.max(), score.mean()])
    else:
        x, y = zip(*[point for box in bboxes for point in box])
        return (min(x), min(y)), (max(x), max(y))


def resolve_enclosing_rbbox(rbboxes: List[np.ndarray], intermed_size: int=1024) ->np.ndarray:
    cloud: np.ndarray = np.concatenate(rbboxes, axis=0)
    cloud *= intermed_size
    rect = cv2.minAreaRect(cloud.astype(np.int32))
    return cv2.boxPoints(rect) / intermed_size


class Line(Element):
    """Implements a line element as a collection of words

    Args:
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """
    _exported_keys: List[str] = ['geometry']
    _children_names: List[str] = ['words']
    words: List[Word] = []

    def __init__(self, words: List[Word], geometry: Optional[Union[BoundingBox, np.ndarray]]=None) ->None:
        if geometry is None:
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].geometry) == 4 else resolve_enclosing_bbox
            geometry = box_resolution_fn([w.geometry for w in words])
        super().__init__(words=words)
        self.geometry = geometry

    def render(self) ->str:
        """Renders the full text of the element"""
        return ' '.join(w.render() for w in self.words)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'words': [Word.from_dict(_dict) for _dict in save_dict['words']]})
        return cls(**kwargs)


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
        lines: list of line elements
        artefacts: list of artefacts
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """
    _exported_keys: List[str] = ['geometry']
    _children_names: List[str] = ['lines', 'artefacts']
    lines: List[Line] = []
    artefacts: List[Artefact] = []

    def __init__(self, lines: List[Line]=[], artefacts: List[Artefact]=[], geometry: Optional[Union[BoundingBox, np.ndarray]]=None) ->None:
        if geometry is None:
            line_boxes = [word.geometry for line in lines for word in line.words]
            artefact_boxes = [artefact.geometry for artefact in artefacts]
            box_resolution_fn = resolve_enclosing_rbbox if isinstance(lines[0].geometry, np.ndarray) else resolve_enclosing_bbox
            geometry = box_resolution_fn(line_boxes + artefact_boxes)
        super().__init__(lines=lines, artefacts=artefacts)
        self.geometry = geometry

    def render(self, line_break: str='\n') ->str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'lines': [Line.from_dict(_dict) for _dict in save_dict['lines']], 'artefacts': [Artefact.from_dict(_dict) for _dict in save_dict['artefacts']]})
        return cls(**kwargs)


def synthesize_page(page: Dict[str, Any], draw_proba: bool=False, font_family: Optional[str]=None) ->np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Return:
        the synthesized page
    """
    h, w = page['dimensions']
    response = 255 * np.ones((h, w, 3), dtype=np.int32)
    for block in page['blocks']:
        for line in block['lines']:
            for word in line['words']:
                (xmin, ymin), (xmax, ymax) = word['geometry']
                xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
                ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
                font = get_font(font_family, int(0.75 * (ymax - ymin)))
                img = Image.new('RGB', (xmax - xmin, ymax - ymin), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                try:
                    d.text((0, 0), word['value'], font=font, fill=(0, 0, 0))
                except UnicodeEncodeError:
                    d.text((0, 0), unidecode(word['value']), font=font, fill=(0, 0, 0))
                if draw_proba:
                    p = int(255 * word['confidence'])
                    mask = np.where(np.array(img) == 0, 1, 0)
                    proba: np.ndarray = np.array([255 - p, 0, p])
                    color = mask * proba[np.newaxis, np.newaxis, :]
                    white_mask = 255 * (1 - mask)
                    img = color + white_mask
                response[ymin:ymax, xmin:xmax, :] = np.array(img)
    return response


Polygon4P = Tuple[Point2D, Point2D, Point2D, Point2D]


def merge_strings(a: str, b: str, dil_factor: float) ->str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
        A merged character sequence.

    Example::
        >>> from doctr.model.recognition.utils import merge_sequences
        >>> merge_sequences('abcd', 'cdefgh', 1.4)
        'abcdefgh'
        >>> merge_sequences('abcdi', 'cdefgh', 1.4)
        'abcdefgh'
    """
    seq_len = min(len(a), len(b))
    if seq_len == 0:
        return b if len(a) == 0 else b
    min_score, index = 1.0, 0
    scores = [(levenshtein(a[-i:], b[:i], processor=None) / i) for i in range(1, seq_len + 1)]
    if len(scores) > 1 and (scores[0], scores[1]) == (0, 0):
        n_overlap = round(len(b) * (dil_factor - 1) / dil_factor)
        n_zeros = sum(val == 0 for val in scores)
        min_score, index = 0, min(n_zeros, n_overlap)
    else:
        for i, score in enumerate(scores):
            if score < min_score:
                min_score, index = score, i + 1
    if index == 0:
        return a + b
    return a[:-1] + b[index - 1:]


def merge_multi_strings(seq_list: List[str], dil_factor: float) ->str:
    """Recursively merges consecutive string sequences with overlapping characters.

    Args:
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
        A merged character sequence

    Example::
        >>> from doctr.model.recognition.utils import merge_multi_sequences
        >>> merge_multi_sequences(['abc', 'bcdef', 'difghi', 'aijkl'], 1.4)
        'abcdefghijkl'
    """

    def _recursive_merge(a: str, seq_list: List[str], dil_factor: float) ->str:
        if len(seq_list) == 1:
            return merge_strings(a, seq_list[0], dil_factor)
        return _recursive_merge(merge_strings(a, seq_list[0], dil_factor), seq_list[1:], dil_factor)
    return _recursive_merge('', seq_list, dil_factor)


def remap_preds(preds: List[Tuple[str, float]], crop_map: List[Union[int, Tuple[int, int]]], dilation: float) ->List[Tuple[str, float]]:
    remapped_out = []
    for _idx in crop_map:
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            vals, probs = zip(*preds[_idx[0]:_idx[1]])
            remapped_out.append((merge_multi_strings(vals, dilation), min(probs)))
    return remapped_out


def split_crops(crops: List[np.ndarray], max_ratio: float, target_ratio: int, dilation: float, channels_last: bool=True) ->Tuple[List[np.ndarray], List[Union[int, Tuple[int, int]]], bool]:
    """Chunk crops horizontally to match a given aspect ratio

    Args:
        crops: list of numpy array of shape (H, W, 3) if channels_last or (3, H, W) otherwise
        max_ratio: the maximum aspect ratio that won't trigger the chunk
        target_ratio: when crops are chunked, they will be chunked to match this aspect ratio
        dilation: the width dilation of final chunks (to provide some overlaps)
        channels_last: whether the numpy array has dimensions in channels last order

    Returns:
        a tuple with the new crops, their mapping, and a boolean specifying whether any remap is required
    """
    _remap_required = False
    crop_map: List[Union[int, Tuple[int, int]]] = []
    new_crops: List[np.ndarray] = []
    for crop in crops:
        h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
        aspect_ratio = w / h
        if aspect_ratio > max_ratio:
            num_subcrops = int(aspect_ratio // target_ratio)
            width = dilation * w / num_subcrops
            centers = [(w / num_subcrops * (1 / 2 + idx)) for idx in range(num_subcrops)]
            if channels_last:
                _crops = [crop[:, max(0, int(round(center - width / 2))):min(w - 1, int(round(center + width / 2))), :] for center in centers]
            else:
                _crops = [crop[:, :, max(0, int(round(center - width / 2))):min(w - 1, int(round(center + width / 2)))] for center in centers]
            _crops = [crop for crop in _crops if all(s > 0 for s in crop.shape)]
            crop_map.append((len(new_crops), len(new_crops) + len(_crops)))
            new_crops.extend(_crops)
            _remap_required = True
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)
    return new_crops, crop_map, _remap_required


class RecognitionPredictor(nn.Module):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    def __init__(self, pre_processor: PreProcessor, model: nn.Module, split_wide_crops: bool=True) ->None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8
        self.dil_factor = 1.4
        self.target_ar = 6

    @torch.no_grad()
    def forward(self, crops: Sequence[Union[np.ndarray, torch.Tensor]], **kwargs: Any) ->List[Tuple[str, float]]:
        if len(crops) == 0:
            return []
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError('incorrect input shape: all crops are expected to be multi-channel 2D images.')
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(crops, self.critical_ar, self.target_ar, self.dil_factor, isinstance(crops[0], np.ndarray))
            if remapped:
                crops = new_crops
        processed_batches = self.pre_processor(crops)
        _device = next(self.model.parameters()).device
        raw = [self.model(batch, return_preds=True, **kwargs)['preds'] for batch in processed_batches]
        out = [charseq for batch in raw for charseq in batch]
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.dil_factor)
        return out


def estimate_page_angle(polys: np.ndarray) ->float:
    """Takes a batch of rotated previously ORIENTED polys (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees
    """
    xleft = polys[:, 0, 0] + polys[:, 3, 0]
    yleft = polys[:, 0, 1] + polys[:, 3, 1]
    xright = polys[:, 1, 0] + polys[:, 2, 0]
    yright = polys[:, 1, 1] + polys[:, 2, 1]
    return float(np.median(np.arctan((yleft - yright) / (xright - xleft))) * 180 / np.pi)


def remap_boxes(loc_preds: np.ndarray, orig_shape: Tuple[int, int], dest_shape: Tuple[int, int]) ->np.ndarray:
    """Remaps a batch of rotated locpred (N, 4, 2) expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes, but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.

    Args:
        loc_preds: (N, 4, 2) array of RELATIVE loc_preds
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
        A batch of rotated loc_preds (N, 4, 2) expressed in the destination referencial
    """
    if len(dest_shape) != 2:
        raise ValueError(f'Mask length should be 2, was found at: {len(dest_shape)}')
    if len(orig_shape) != 2:
        raise ValueError(f'Image_shape length should be 2, was found at: {len(orig_shape)}')
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    mboxes[:, :, 0] = (loc_preds[:, :, 0] * orig_width + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, :, 1] = (loc_preds[:, :, 1] * orig_height + (dest_height - orig_height) / 2) / dest_height
    return mboxes


def rotate_boxes(loc_preds: np.ndarray, angle: float, orig_shape: Tuple[int, int], min_angle: float=1.0, target_shape: Optional[Tuple[int, int]]=None) ->np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c) or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)

    Args:
        loc_preds: (N, 5) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes

    Returns:
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """
    _boxes = loc_preds.copy()
    if _boxes.ndim == 2:
        _boxes = np.stack([_boxes[:, [0, 1]], _boxes[:, [2, 1]], _boxes[:, [2, 3]], _boxes[:, [0, 3]]], axis=1)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    angle_rad = angle * np.pi / 180.0
    rotation_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=_boxes.dtype)
    points: np.ndarray = np.stack((_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1)
    image_center = orig_shape[1] / 2, orig_shape[0] / 2
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes: np.ndarray = np.stack((rotated_points[:, :, 0] / orig_shape[1], rotated_points[:, :, 1] / orig_shape[0]), axis=-1)
    if target_shape is not None:
        rotated_boxes = remap_boxes(rotated_boxes, orig_shape=orig_shape, dest_shape=target_shape)
    return rotated_boxes


class Prediction(Word):
    """Implements a prediction element"""

    def render(self) ->str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) ->str:
        return f"value='{self.value}', confidence={self.confidence:.2}, bounding_box={self.geometry}"


def synthesize_kie_page(page: Dict[str, Any], draw_proba: bool=False, font_family: Optional[str]=None) ->np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Return:
        the synthesized page
    """
    h, w = page['dimensions']
    response = 255 * np.ones((h, w, 3), dtype=np.int32)
    for predictions in page['predictions'].values():
        for prediction in predictions:
            (xmin, ymin), (xmax, ymax) = prediction['geometry']
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
            font = get_font(font_family, int(0.75 * (ymax - ymin)))
            img = Image.new('RGB', (xmax - xmin, ymax - ymin), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            try:
                d.text((0, 0), prediction['value'], font=font, fill=(0, 0, 0))
            except UnicodeEncodeError:
                d.text((0, 0), unidecode(prediction['value']), font=font, fill=(0, 0, 0))
            if draw_proba:
                p = int(255 * prediction['confidence'])
                mask = np.where(np.array(img) == 0, 1, 0)
                proba: np.ndarray = np.array([255 - p, 0, p])
                color = mask * proba[np.newaxis, np.newaxis, :]
                white_mask = 255 * (1 - mask)
                img = color + white_mask
            response[ymin:ymax, xmin:xmax, :] = np.array(img)
    return response


def get_colors(num_colors: int) ->List[Tuple[float, float, float]]:
    """Generate num_colors color for matplotlib

    Args:
        num_colors: number of colors to generate

    Returns:
        colors: list of generated colors
    """
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def extract_crops(img: np.ndarray, boxes: np.ndarray, channels_last: bool=True) ->List[np.ndarray]:
    """Created cropped images from list of bounding boxes
    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError('boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)')
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        _boxes[2:] += 1
    if channels_last:
        return deepcopy([img[box[1]:box[3], box[0]:box[2]] for box in _boxes])
    return deepcopy([img[:, box[1]:box[3], box[0]:box[2]] for box in _boxes])


def extract_rcrops(img: np.ndarray, polys: np.ndarray, dtype=np.float32, channels_last: bool=True) ->List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes
    Args:
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError('polys are expected to be quadrilateral, of shape (N, 4, 2)')
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, :, 0] *= width
        _boxes[:, :, 1] *= height
    src_pts = _boxes[:, :3].astype(np.float32)
    d1 = np.linalg.norm(src_pts[:, 0] - src_pts[:, 1], axis=-1)
    d2 = np.linalg.norm(src_pts[:, 1] - src_pts[:, 2], axis=-1)
    dst_pts = np.zeros((_boxes.shape[0], 3, 2), dtype=dtype)
    dst_pts[:, 1, 0] = dst_pts[:, 2, 0] = d1 - 1
    dst_pts[:, 2, 1] = d2 - 1
    crops = [cv2.warpAffine(img if channels_last else img.transpose(1, 2, 0), cv2.getAffineTransform(src_pts[idx], dst_pts[idx]), (int(d1[idx]), int(d2[idx]))) for idx in range(_boxes.shape[0])]
    return crops


def rectify_crops(crops: List[np.ndarray], orientations: List[int]) ->List[np.ndarray]:
    """Rotate each crop of the list according to the predicted orientation:
    0: already straight, no rotation
    1: 90 ccw, rotate 3 times ccw
    2: 180, rotate 2 times ccw
    3: 270 ccw, rotate 1 time ccw
    """
    orientations = [(4 - pred if pred != 0 else 0) for pred in orientations]
    return [(crop if orientation == 0 else np.rot90(crop, orientation)) for orientation, crop in zip(orientations, crops)] if len(orientations) > 0 else []


def rectify_loc_preds(page_loc_preds: np.ndarray, orientations: List[int]) ->Optional[np.ndarray]:
    """Orient the quadrangle (Polygon4P) according to the predicted orientation,
    so that the points are in this order: top L, top R, bot R, bot L if the crop is readable
    """
    return np.stack([np.roll(page_loc_pred, orientation, axis=0) for orientation, page_loc_pred in zip(orientations, page_loc_preds)], axis=0) if len(orientations) > 0 else None


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        kwargs: keyword args of `DocumentBuilder`
    """
    crop_orientation_predictor: Optional[CropOrientationPredictor]

    def __init__(self, assume_straight_pages: bool=True, straighten_pages: bool=False, preserve_aspect_ratio: bool=True, symmetric_pad: bool=True, **kwargs: Any) ->None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.crop_orientation_predictor = None if assume_straight_pages else crop_orientation_predictor(pretrained=True)
        self.doc_builder = DocumentBuilder(**kwargs)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    @staticmethod
    def _generate_crops(pages: List[np.ndarray], loc_preds: List[np.ndarray], channels_last: bool, assume_straight_pages: bool=False) ->List[List[np.ndarray]]:
        extraction_fn = extract_crops if assume_straight_pages else extract_rcrops
        crops = [extraction_fn(page, _boxes[:, :4], channels_last=channels_last) for page, _boxes in zip(pages, loc_preds)]
        return crops

    @staticmethod
    def _prepare_crops(pages: List[np.ndarray], loc_preds: List[np.ndarray], channels_last: bool, assume_straight_pages: bool=False) ->Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        crops = _OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages)
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [[crop for crop, _kept in zip(page_crops, page_kept) if _kept] for page_crops, page_kept in zip(crops, is_kept)]
        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]
        return crops, loc_preds

    def _rectify_crops(self, crops: List[List[np.ndarray]], loc_preds: List[np.ndarray]) ->Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        orientations = [self.crop_orientation_predictor(page_crops) for page_crops in crops]
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [(rectify_loc_preds(page_loc_preds, orientation) if len(page_loc_preds) > 0 else page_loc_preds) for page_loc_preds, orientation in zip(loc_preds, orientations)]
        return rect_crops, rect_loc_preds

    def _remove_padding(self, pages: List[np.ndarray], loc_preds: List[np.ndarray]) ->List[np.ndarray]:
        if self.preserve_aspect_ratio:
            rectified_preds = []
            for page, loc_pred in zip(pages, loc_preds):
                h, w = page.shape[0], page.shape[1]
                if h > w:
                    if self.symmetric_pad:
                        if self.assume_straight_pages:
                            loc_pred[:, [0, 2]] = np.clip((loc_pred[:, [0, 2]] - 0.5) * h / w + 0.5, 0, 1)
                        else:
                            loc_pred[:, :, 0] = np.clip((loc_pred[:, :, 0] - 0.5) * h / w + 0.5, 0, 1)
                    elif self.assume_straight_pages:
                        loc_pred[:, [0, 2]] *= h / w
                    else:
                        loc_pred[:, :, 0] *= h / w
                elif w > h:
                    if self.symmetric_pad:
                        if self.assume_straight_pages:
                            loc_pred[:, [1, 3]] = np.clip((loc_pred[:, [1, 3]] - 0.5) * w / h + 0.5, 0, 1)
                        else:
                            loc_pred[:, :, 1] = np.clip((loc_pred[:, :, 1] - 0.5) * w / h + 0.5, 0, 1)
                    elif self.assume_straight_pages:
                        loc_pred[:, [1, 3]] *= w / h
                    else:
                        loc_pred[:, :, 1] *= w / h
                rectified_preds.append(loc_pred)
            return rectified_preds
        return loc_preds

    @staticmethod
    def _process_predictions(loc_preds: List[np.ndarray], word_preds: List[Tuple[str, float]]) ->Tuple[List[np.ndarray], List[List[Tuple[str, float]]]]:
        text_preds = []
        if len(loc_preds) > 0:
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx:_idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]
        return loc_preds, text_preds


class _KIEPredictor(_OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        kwargs: keyword args of `DocumentBuilder`
    """
    crop_orientation_predictor: Optional[CropOrientationPredictor]

    def __init__(self, assume_straight_pages: bool=True, straighten_pages: bool=False, preserve_aspect_ratio: bool=True, symmetric_pad: bool=True, **kwargs: Any) ->None:
        super().__init__(assume_straight_pages, straighten_pages, preserve_aspect_ratio, symmetric_pad, **kwargs)
        self.doc_builder: KIEDocumentBuilder = KIEDocumentBuilder(**kwargs)


def get_max_width_length_ratio(contour: np.ndarray) ->float:
    """Get the maximum shape ratio of a contour.

    Args:
        contour: the contour from cv2.findContour

    Returns: the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def estimate_orientation(img: np.ndarray, n_ct: int=50, ratio_threshold_for_lines: float=5) ->float:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

    Args:
        img: the img to analyze
        n_ct: the number of contours used for the orientation estimation
        ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines

    Returns:
        the angle of the general document orientation
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    thresh = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = img.shape[:2]
    k_x = max(1, floor(w / 100))
    k_y = max(1, floor(h / 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=get_max_width_length_ratio, reverse=True)
    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h > ratio_threshold_for_lines:
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:
            angles.append(angle - 90)
    if len(angles) == 0:
        return 0
    else:
        return -median_low(angles)


def get_language(text: str) ->Tuple[str, float]:
    """Get languages of a text using langdetect model.
    Get the language with the highest probability or no language if only a few words or a low probability
    Args:
        text (str): text
    Returns:
        The detected language in ISO 639 code and confidence score
    """
    try:
        lang = detect_langs(text.lower())[0]
    except LangDetectException:
        return 'unknown', 0.0
    if len(text) <= 1 or len(text) <= 5 and lang.prob <= 0.2:
        return 'unknown', 0.0
    return lang.lang, lang.prob


def invert_data_structure(x: Union[List[Dict[str, Any]], Dict[str, List[Any]]]) ->Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Invert a List of Dict of elements to a Dict of list of elements and the other way around

    Args:
        x: a list of dictionaries with the same keys or a dictionary of lists of the same length

    Returns:
        dictionary of list when x is a list of dictionaries or a list of dictionaries when x is dictionary of lists
    """
    if isinstance(x, dict):
        assert len(set([len(v) for v in x.values()])) == 1, 'All the lists in the dictionnary should have the same length.'
        return [dict(zip(x, t)) for t in zip(*x.values())]
    elif isinstance(x, list):
        return {k: [dic[k] for dic in x] for k in x[0]}
    else:
        raise TypeError(f'Expected input to be either a dict or a list, got {type(input)} instead.')


def rotate_abs_points(points: np.ndarray, angle: float=0.0) ->np.ndarray:
    """Rotate points counter-clockwise.
    Points: array of size (N, 2)
    """
    angle_rad = angle * np.pi / 180.0
    rotation_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=points.dtype)
    return np.matmul(points, rotation_mat.T)


def compute_expanded_shape(img_shape: Tuple[int, int], angle: float) ->Tuple[int, int]:
    """Compute the shape of an expanded rotated image

    Args:
        img_shape: the height and width of the image
        angle: angle between -90 and +90 degrees

    Returns:
        the height and width of the rotated image
    """
    points: np.ndarray = np.array([[img_shape[1] / 2, img_shape[0] / 2], [-img_shape[1] / 2, img_shape[0] / 2]])
    rotated_points = rotate_abs_points(points, angle)
    wh_shape = 2 * np.abs(rotated_points).max(axis=0)
    return wh_shape[1], wh_shape[0]


def rotate_image(image: np.ndarray, angle: float, expand: bool=False, preserve_origin_shape: bool=False) ->np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation
        preserve_origin_shape: if expand is set to True, resizes the final output to the original image size

    Returns:
        Rotated array, padded by 0 by default.
    """
    exp_img: np.ndarray
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:2], angle)
        h_pad, w_pad = int(max(0, ceil(exp_shape[0] - image.shape[0]))), int(max(0, ceil(exp_shape[1] - image.shape[1])))
        exp_img = np.pad(image, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
    else:
        exp_img = image
    height, width = exp_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rot_img = cv2.warpAffine(exp_img, rot_mat, (width, height))
    if expand:
        if image.shape[0] / image.shape[1] != rot_img.shape[0] / rot_img.shape[1]:
            if rot_img.shape[0] / rot_img.shape[1] > image.shape[0] / image.shape[1]:
                h_pad, w_pad = 0, int(rot_img.shape[0] * image.shape[1] / image.shape[0] - rot_img.shape[1])
            else:
                h_pad, w_pad = int(rot_img.shape[1] * image.shape[0] / image.shape[1] - rot_img.shape[0]), 0
            rot_img = np.pad(rot_img, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
        if preserve_origin_shape:
            rot_img = cv2.resize(rot_img, image.shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)
    return rot_img


class PositionalEncoding(nn.Module):
    """Compute positional encoding"""

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000) ->None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x: embeddings (batch, max_len, d_model)

        Returns:
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Decoder(nn.Module):
    """Transformer Decoder"""

    def __init__(self, num_layers: int, num_heads: int, d_model: int, vocab_size: int, dropout: float=0.2, dff: int=2048, maximum_position_encoding: int=50) ->None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_masked_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-05)
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, maximum_position_encoding)
        self.attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.source_attention = nn.ModuleList([MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)])
        self.position_feed_forward = nn.ModuleList([PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, source_mask: Optional[torch.Tensor]=None, target_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        pos_enc_tgt = self.positional_encoding(tgt)
        output = pos_enc_tgt
        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, target_mask))
            normed_output = self.layer_norm_masked_attention(output)
            output = output + self.dropout(self.source_attention[i](normed_output, memory, memory, source_mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm_output(output)


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: str) ->None:
        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    def extra_repr(self) ->str:
        return f'vocab_size={len(self.vocab)}'


def encode_string(input_string: str, vocab: str) ->List[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A list encoding the input_string"""
    try:
        return list(map(vocab.index, input_string))
    except ValueError:
        raise ValueError("some characters cannot be found in 'vocab'")


def encode_sequences(sequences: List[str], vocab: str, target_size: Optional[int]=None, eos: int=-1, sos: Optional[int]=None, pad: Optional[int]=None, dynamic_seq_length: bool=False, **kwargs: Any) ->np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD
        dynamic_seq_length: if `target_size` is specified, uses it as upper bound and enables dynamic sequence size

    Returns:
        the padded encoded data as a tensor
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")
    if not isinstance(target_size, int) or dynamic_seq_length:
        max_length = max(len(w) for w in sequences) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)
    if isinstance(pad, int):
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        default_symbol = pad
    else:
        default_symbol = eos
    encoded_data: np.ndarray = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)
    for idx, seq in enumerate(map(partial(encode_string, vocab=vocab), sequences)):
        if isinstance(pad, int):
            seq.append(eos)
        encoded_data[idx, :min(len(seq), target_size)] = seq[:min(len(seq), target_size)]
    if isinstance(sos, int):
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos
    return encoded_data


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""
    vocab: str
    max_length: int

    def build_target(self, gts: List[str]) ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class CRNN(RecognitionModel, nn.Module):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
    """
    _children_names: List[str] = ['feat_extractor', 'decoder', 'linear', 'postprocessor']

    def __init__(self, feature_extractor: nn.Module, vocab: str, rnn_units: int=128, input_shape: Tuple[int, int, int]=(3, 32, 128), exportable: bool=False, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.exportable = exportable
        self.feat_extractor = feature_extractor
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        lstm_in = out_shape[1] * out_shape[2]
        self.feat_extractor.train()
        self.decoder = nn.LSTM(input_size=lstm_in, hidden_size=rnn_units, batch_first=True, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)
        self.postprocessor = CTCPostProcessor(vocab=vocab)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def compute_loss(self, model_output: torch.Tensor, target: List[str]) ->torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(probs, torch.from_numpy(gt), input_length, torch.tensor(seq_len, dtype=torch.int), len(self.vocab), zero_infinity=True)
        return ctc_loss

    def forward(self, x: torch.Tensor, target: Optional[List[str]]=None, return_model_output: bool=False, return_preds: bool=False) ->Dict[str, Any]:
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        features = self.feat_extractor(x)
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)
        out: Dict[str, Any] = {}
        if self.exportable:
            out['logits'] = logits
            return out
        if return_model_output:
            out['out_map'] = logits
        if target is None or return_preds:
            out['preds'] = self.postprocessor(logits)
        if target is not None:
            out['loss'] = self.compute_loss(logits, target)
        return out


class _MASTERPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: str) ->None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ['<eos>'] + ['<sos>'] + ['<pad>']


class MASTERPostProcessor(_MASTERPostProcessor):
    """Post processor for MASTER architectures"""

    def __call__(self, logits: torch.Tensor) ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        probs = probs.min(dim=1).values.detach().cpu()
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.cpu().numpy()]
        return list(zip(word_values, probs.numpy().tolist()))


class _MASTER:
    vocab: str
    max_length: int

    def build_target(self, gts: List[str]) ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab), sos=len(self.vocab) + 1, pad=len(self.vocab) + 2)
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class MASTER(_MASTER, nn.Module):
    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/wenwenyu/MASTER-pytorch>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        dropout: dropout probability of the decoder
        input_shape: size of the image inputs
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor: nn.Module, vocab: str, d_model: int=512, dff: int=2048, num_heads: int=8, num_layers: int=3, max_length: int=50, dropout: float=0.2, input_shape: Tuple[int, int, int]=(3, 32, 128), exportable: bool=False, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.exportable = exportable
        self.max_length = max_length
        self.d_model = d_model
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)
        self.feat_extractor = feature_extractor
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len=input_shape[1] * input_shape[2])
        self.decoder = Decoder(num_layers=num_layers, d_model=self.d_model, num_heads=num_heads, vocab_size=self.vocab_size + 3, dff=dff, dropout=dropout, maximum_position_encoding=self.max_length)
        self.linear = nn.Linear(self.d_model, self.vocab_size + 3)
        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)
        for n, m in self.named_modules():
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_source_and_target_mask(self, source: torch.Tensor, target: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        target_pad_mask = (target != self.vocab_size + 2).unsqueeze(1).unsqueeze(1)
        target_length = target.size(1)
        target_sub_mask = torch.tril(torch.ones((target_length, target_length), device=source.device), diagonal=0)
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        target_mask = target_pad_mask & target_sub_mask
        return source_mask, target_mask.int()

    @staticmethod
    def compute_loss(model_output: torch.Tensor, gt: torch.Tensor, seq_len: torch.Tensor) ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction='none')
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()

    def forward(self, x: torch.Tensor, target: Optional[List[str]]=None, return_model_output: bool=False, return_preds: bool=False) ->Dict[str, Any]:
        """Call function for training

        Args:
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Returns:
            A dictionnary containing eventually loss, logits and predictions.
        """
        features = self.feat_extractor(x)['features']
        b, c, h, w = features.shape
        features = features.view(b, c, h * w).permute((0, 2, 1))
        encoded = self.positional_encoding(features)
        out: Dict[str, Any] = {}
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
            source_mask, target_mask = self.make_source_and_target_mask(encoded, gt)
            output = self.decoder(gt, encoded, source_mask, target_mask)
            logits = self.linear(output)
        else:
            logits = self.decode(encoded)
        if self.exportable:
            out['logits'] = logits
            return out
        if target is not None:
            out['loss'] = self.compute_loss(logits, gt, seq_len)
        if return_model_output:
            out['out_map'] = logits
        if return_preds:
            out['preds'] = self.postprocessor(logits)
        return out

    def decode(self, encoded: torch.Tensor) ->torch.Tensor:
        """Decode function for prediction

        Args:
            encoded: input tensor

        Return:
            A Tuple of torch.Tensor: predictions, logits
        """
        b = encoded.size(0)
        ys = torch.full((b, self.max_length), self.vocab_size + 2, dtype=torch.long, device=encoded.device)
        ys[:, 0] = self.vocab_size + 1
        for i in range(self.max_length - 1):
            source_mask, target_mask = self.make_source_and_target_mask(encoded, ys)
            output = self.decoder(ys, encoded, source_mask, target_mask)
            logits = self.linear(output)
            prob = torch.softmax(logits, dim=-1)
            next_token = torch.max(prob, dim=-1).indices
            ys[:, i + 1] = next_token[:, i]
        return logits


class SAREncoder(nn.Module):

    def __init__(self, in_feats: int, rnn_units: int, dropout_prob: float=0.0) ->None:
        super().__init__()
        self.rnn = nn.LSTM(in_feats, rnn_units, 2, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(rnn_units, rnn_units)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        encoded = self.rnn(x)[0]
        return self.linear(encoded[:, -1, :])


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: int, state_chans: int, attention_units: int) ->None:
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, kernel_size=3, padding=1)
        self.state_conv = nn.Conv2d(state_chans, attention_units, kernel_size=1, bias=False)
        self.attention_projector = nn.Conv2d(attention_units, 1, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor, hidden_state: torch.Tensor) ->torch.Tensor:
        H_f, W_f = features.shape[2:]
        feat_projection = self.feat_conv(features)
        hidden_state = hidden_state.view(hidden_state.size(0), hidden_state.size(1), 1, 1)
        state_projection = self.state_conv(hidden_state)
        state_projection = state_projection.expand(-1, -1, H_f, W_f)
        attention_weights = torch.tanh(feat_projection + state_projection)
        attention_weights = self.attention_projector(attention_weights)
        B, C, H, W = attention_weights.size()
        attention_weights = torch.softmax(attention_weights.view(B, -1), dim=-1).view(B, C, H, W)
        return (features * attention_weights).sum(dim=(2, 3))


class SARDecoder(nn.Module):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units

    """

    def __init__(self, rnn_units: int, max_length: int, vocab_size: int, embedding_units: int, attention_units: int, feat_chans: int=512, dropout_prob: float=0.0) ->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed = nn.Linear(self.vocab_size + 1, embedding_units)
        self.embed_tgt = nn.Embedding(embedding_units, self.vocab_size + 1)
        self.attention_module = AttentionModule(feat_chans, rnn_units, attention_units)
        self.lstm_cell = nn.LSTMCell(rnn_units, rnn_units)
        self.output_dense = nn.Linear(2 * rnn_units, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features: torch.Tensor, holistic: torch.Tensor, gt: Optional[torch.Tensor]=None) ->torch.Tensor:
        if gt is not None:
            gt_embedding = self.embed_tgt(gt)
        logits_list: List[torch.Tensor] = []
        for t in range(self.max_length + 1):
            if t == 0:
                hidden_state_init = cell_state_init = torch.zeros(features.size(0), features.size(1), device=features.device)
                hidden_state, cell_state = hidden_state_init, cell_state_init
                prev_symbol = holistic
            elif t == 1:
                prev_symbol = torch.zeros(features.size(0), self.vocab_size + 1, device=features.device)
                prev_symbol = self.embed(prev_symbol)
            elif gt is not None:
                prev_symbol = self.embed(gt_embedding[:, t - 2])
            else:
                index = logits_list[t - 1].argmax(-1)
                prev_symbol = prev_symbol.scatter_(1, index.unsqueeze(1), 1)
            hidden_state_init, cell_state_init = self.lstm_cell(prev_symbol, (hidden_state_init, cell_state_init))
            hidden_state, cell_state = self.lstm_cell(hidden_state_init, (hidden_state, cell_state))
            glimpse = self.attention_module(features, hidden_state)
            logits = torch.cat([hidden_state, glimpse], dim=1)
            logits = self.dropout(logits)
            logits_list.append(self.output_dense(logits))
        return torch.stack(logits_list[1:]).permute(1, 0, 2)


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, logits: torch.Tensor) ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        probs = probs.min(dim=1).values.detach().cpu()
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.detach().cpu().numpy()]
        return list(zip(word_values, probs.numpy().tolist()))


class SAR(nn.Module, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor, vocab: str, rnn_units: int=512, embedding_units: int=512, attention_units: int=512, max_length: int=30, dropout_prob: float=0.0, input_shape: Tuple[int, int, int]=(3, 32, 128), exportable: bool=False, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 1
        self.feat_extractor = feature_extractor
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape)))['features'].shape
        self.feat_extractor.train()
        self.encoder = SAREncoder(out_shape[1], rnn_units, dropout_prob)
        self.decoder = SARDecoder(rnn_units, self.max_length, len(self.vocab), embedding_units, attention_units, dropout_prob=dropout_prob)
        self.postprocessor = SARPostProcessor(vocab=vocab)

    def forward(self, x: torch.Tensor, target: Optional[List[str]]=None, return_model_output: bool=False, return_preds: bool=False) ->Dict[str, Any]:
        features = self.feat_extractor(x)['features']
        pooled_features = features.max(dim=-2).values
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()
        encoded = self.encoder(pooled_features)
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
        if self.training and target is None:
            raise ValueError('Need to provide labels during training for teacher forcing')
        decoded_features = self.decoder(features, encoded, gt=None if target is None else gt)
        out: Dict[str, Any] = {}
        if self.exportable:
            out['logits'] = decoded_features
            return out
        if return_model_output:
            out['out_map'] = decoded_features
        if target is None or return_preds:
            out['preds'] = self.postprocessor(decoded_features)
        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)
        return out

    @staticmethod
    def compute_loss(model_output: torch.Tensor, gt: torch.Tensor, seq_len: torch.Tensor) ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction='none')
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()


class _ViTSTRPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: str) ->None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ['<eos>', '<sos>', '<pad>']


class ViTSTRPostProcessor(_ViTSTRPostProcessor):
    """Post processor for ViTSTR architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, logits: torch.Tensor) ->List[Tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        probs = probs.min(dim=1).values.detach().cpu()
        word_values = [''.join(self._embedding[idx] for idx in encoded_seq).split('<eos>')[0] for encoded_seq in out_idxs.cpu().numpy()]
        return list(zip(word_values, probs.numpy().tolist()))


class _ViTSTR:
    vocab: str
    max_length: int

    def build_target(self, gts: List[str]) ->Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab), sos=len(self.vocab) + 1, pad=len(self.vocab) + 2)
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class ViTSTR(_ViTSTR, nn.Module):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(self, feature_extractor, vocab: str, embedding_units: int, max_length: int=25, input_shape: Tuple[int, int, int]=(3, 32, 128), exportable: bool=False, cfg: Optional[Dict[str, Any]]=None) ->None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 3
        self.feat_extractor = feature_extractor
        self.head = nn.Linear(embedding_units, len(self.vocab) + 3)
        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def forward(self, x: torch.Tensor, target: Optional[List[str]]=None, return_model_output: bool=False, return_preds: bool=False) ->Dict[str, Any]:
        features = self.feat_extractor(x)['features']
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt), torch.tensor(_seq_len)
            gt, seq_len = gt, seq_len
        if self.training and target is None:
            raise ValueError('Need to provide labels during training')
        features = features[:, :self.max_length + 1]
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 3)
        decoded_features = logits[:, 1:]
        out: Dict[str, Any] = {}
        if self.exportable:
            out['logits'] = decoded_features
            return out
        if return_model_output:
            out['out_map'] = decoded_features
        if target is None or return_preds:
            out['preds'] = self.postprocessor(decoded_features)
        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)
        return out

    @staticmethod
    def compute_loss(model_output: torch.Tensor, gt: torch.Tensor, seq_len: torch.Tensor) ->torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction='none')
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len
        return ce_loss.mean()


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian Noise to the input tensor

    >>> import torch
    >>> from doctr.transforms import GaussianNoise
    >>> transfo = GaussianNoise(0., 1.)
    >>> out = transfo(torch.rand((3, 224, 224)))

    Args:
        mean : mean of the gaussian distribution
        std : std of the gaussian distribution
    """

    def __init__(self, mean: float=0.0, std: float=1.0) ->None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            return (x + 255 * noise).round().clamp(0, 255)
        else:
            return (x + noise).clamp(0, 1)

    def extra_repr(self) ->str:
        return f'mean={self.mean}, std={self.std}'


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) ->torch.Tensor:
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


def expand_line(line: np.ndarray, target_shape: Tuple[int, int]) ->Tuple[float, float]:
    """Expands a 2-point line, so that the first is on the edge. In other terms, we extend the line in
    the same direction until we meet one of the edges.

    Args:
        line: array of shape (2, 2) of the point supposed to be on one edge, and the shadow tip.
        target_shape: the desired mask shape

    Returns:
        2D coordinates of the first point once we extended the line (on one of the edges)
    """
    if any(coord == 0 or coord == size for coord, size in zip(line[0], target_shape[::-1])):
        return line[0]
    _tmp = line[1] - line[0]
    _direction = _tmp > 0
    _flat = _tmp == 0
    if _tmp[0] == 0:
        solutions = [(line[0, 0], 0), (line[0, 0], target_shape[0])]
    elif _tmp[1] == 0:
        solutions = [(0, line[0, 1]), (target_shape[1], line[0, 1])]
    else:
        alpha = _tmp[1] / _tmp[0]
        beta = line[1, 1] - alpha * line[1, 0]
        solutions = [(0, beta), (-beta / alpha, 0), (target_shape[1], alpha * target_shape[1] + beta), ((target_shape[0] - beta) / alpha, target_shape[0])]
    for point in solutions:
        if any(val < 0 or val > size for val, size in zip(point, target_shape[::-1])):
            continue
        if all(val == ref if _same else val < ref if _dir else val > ref for val, ref, _dir, _same in zip(point, line[1], _direction, _flat)):
            return point
    raise ValueError


def rotate_abs_geoms(geoms: np.ndarray, angle: float, img_shape: Tuple[int, int], expand: bool=True) ->np.ndarray:
    """Rotate a batch of bounding boxes or polygons by an angle around the
    image center.

    Args:
        boxes: (N, 4) or (N, 4, 2) array of ABSOLUTE coordinate boxes
        angle: anti-clockwise rotation angle in degrees
        img_shape: the height and width of the image
        expand: whether the image should be padded to avoid information loss

    Returns:
        A batch of rotated polygons (N, 4, 2)
    """
    polys = np.stack([geoms[:, [0, 1]], geoms[:, [2, 1]], geoms[:, [2, 3]], geoms[:, [0, 3]]], axis=1) if geoms.ndim == 2 else geoms
    polys = polys.astype(np.float32)
    polys[..., 0] -= img_shape[1] / 2
    polys[..., 1] = img_shape[0] / 2 - polys[..., 1]
    rotated_polys = rotate_abs_points(polys.reshape(-1, 2), angle).reshape(-1, 4, 2)
    target_shape = compute_expanded_shape(img_shape, angle) if expand else img_shape
    rotated_polys[..., 0] = (rotated_polys[..., 0] + target_shape[1] / 2).clip(0, target_shape[1])
    rotated_polys[..., 1] = (target_shape[0] / 2 - rotated_polys[..., 1]).clip(0, target_shape[0])
    return rotated_polys


def create_shadow_mask(target_shape: Tuple[int, int], min_base_width=0.3, max_tip_width=0.5, max_tip_height=0.3) ->np.ndarray:
    """Creates a random shadow mask

    Args:
        target_shape: the target shape (H, W)
        min_base_width: the relative minimum shadow base width
        max_tip_width: the relative maximum shadow tip width
        max_tip_height: the relative maximum shadow tip height

    Returns:
        a numpy ndarray of shape (H, W, 1) with values in the range [0, 1]
    """
    _params = np.random.rand(6)
    base_width = min_base_width + (1 - min_base_width) * _params[0]
    base_center = base_width / 2 + (1 - base_width) * _params[1]
    tip_width = min(_params[2] * base_width * target_shape[0] / target_shape[1], max_tip_width)
    tip_center = tip_width / 2 + (1 - tip_width) * _params[3]
    tip_height = _params[4] * max_tip_height
    tip_mid = tip_height / 2 + (1 - tip_height) * _params[5]
    _order = tip_center < base_center
    contour: np.ndarray = np.array([[base_center - base_width / 2, 0], [base_center + base_width / 2, 0], [tip_center + tip_width / 2, tip_mid + tip_height / 2 if _order else tip_mid - tip_height / 2], [tip_center - tip_width / 2, tip_mid - tip_height / 2 if _order else tip_mid + tip_height / 2]], dtype=np.float32)
    abs_contour: np.ndarray = np.stack((contour[:, 0] * target_shape[1], contour[:, 1] * target_shape[0]), axis=-1).round().astype(np.int32)
    _params = np.random.rand(1)
    rotated_contour = rotate_abs_geoms(abs_contour[None, ...], 360 * _params[0], target_shape, expand=False)[0].round().astype(np.int32)
    quad_idx = int(_params[0] / 0.25)
    if quad_idx % 2 == 0:
        intensity_mask = np.repeat(np.arange(target_shape[0])[:, None], target_shape[1], axis=1) / (target_shape[0] - 1)
        if quad_idx == 0:
            intensity_mask = 1 - intensity_mask
    else:
        intensity_mask = np.repeat(np.arange(target_shape[1])[None, :], target_shape[0], axis=0) / (target_shape[1] - 1)
        if quad_idx == 1:
            intensity_mask = 1 - intensity_mask
    final_contour = rotated_contour.copy()
    final_contour[0] = expand_line(final_contour[[0, 3]], target_shape)
    final_contour[1] = expand_line(final_contour[[1, 2]], target_shape)
    if not np.any(final_contour[0] == final_contour[1]):
        corner_x = 0 if max(final_contour[0, 0], final_contour[1, 0]) < target_shape[1] else target_shape[1]
        corner_y = 0 if max(final_contour[0, 1], final_contour[1, 1]) < target_shape[0] else target_shape[0]
        corner: np.ndarray = np.array([corner_x, corner_y])
        final_contour = np.concatenate((final_contour[:1], corner[None, ...], final_contour[1:]), axis=0)
    mask: np.ndarray = np.zeros((*target_shape, 1), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [final_contour], (255,), lineType=cv2.LINE_AA)[..., 0]
    return (mask / 255).astype(np.float32).clip(0, 1) * intensity_mask.astype(np.float32)


def random_shadow(img: torch.Tensor, opacity_range: Tuple[float, float], **kwargs) ->torch.Tensor:
    """Crop and image and associated bboxes

    Args:
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow

    Returns:
        shaded image
    """
    shadow_mask = create_shadow_mask(img.shape[1:], **kwargs)
    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - torch.from_numpy(shadow_mask[None, ...])
    k = 7 + 2 * int(4 * np.random.rand(1))
    sigma = np.random.uniform(0.5, 5.0)
    shadow_tensor = F.gaussian_blur(shadow_tensor, k, sigma=[sigma, sigma])
    return opacity * shadow_tensor * img + (1 - opacity) * img


class RandomShadow(torch.nn.Module):
    """Adds random shade to the input image

    >>> import torch
    >>> from doctr.transforms import RandomShadow
    >>> transfo = RandomShadow((0., 1.))
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
        opacity_range : minimum and maximum opacity of the shade
    """

    def __init__(self, opacity_range: Optional[Tuple[float, float]]=None) ->None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        try:
            if x.dtype == torch.uint8:
                return (255 * random_shadow(x.to(dtype=torch.float32) / 255, self.opacity_range)).round().clip(0, 255)
            else:
                return random_shadow(x, self.opacity_range).clip(0, 1)
        except ValueError:
            return x

    def extra_repr(self) ->str:
        return f'opacity_range={self.opacity_range}'


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionModule,
     lambda: ([], {'feat_chans': 4, 'state_chans': 4, 'attention_units': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 1, 1])], {}),
     True),
    (ChannelShuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClassifierHead,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderBlock,
     lambda: ([], {'num_layers': 1, 'num_heads': 4, 'd_model': 4, 'dff': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GaussianNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'num_heads': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'ffd': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resize,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SAREncoder,
     lambda: ([], {'in_feats': 4, 'rnn_units': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_mindee_doctr(_paritybench_base):
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

