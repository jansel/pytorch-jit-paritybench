import sys
_module = sys.modules[__name__]
del sys
conf = _module
hubconf = _module
setup = _module
test_data_pipeline = _module
test_models = _module
test_models_anchor_utils = _module
test_models_common = _module
test_models_transform = _module
test_models_yolov5 = _module
test_relay = _module
test_runtime_ort = _module
test_trainer = _module
test_utils = _module
test_v5 = _module
trace_model = _module
convert_txt_to_json = _module
convert_yolov5_to_yolort = _module
eval_metric = _module
export_model = _module
run_clang_format = _module
yolort = _module
data = _module
_helper = _module
builtin_meta = _module
coco = _module
coco_eval = _module
data_module = _module
distributed = _module
transforms = _module
voc = _module
models = _module
_checkpoint = _module
_utils = _module
anchor_utils = _module
backbone_utils = _module
box_head = _module
darknet = _module
darknetv4 = _module
darknetv6 = _module
path_aggregation_network = _module
transform = _module
transformer = _module
yolo = _module
yolo_lite = _module
yolov5 = _module
relay = _module
head_helper = _module
ir_visualizer = _module
logits_decoder = _module
trace_wrapper = _module
trt_graphsurgeon = _module
trt_inference = _module
runtime = _module
ort_helper = _module
transform = _module
trt_helper = _module
y_onnxruntime = _module
y_tensorrt = _module
trainer = _module
lightning_task = _module
utils = _module
annotations_converter = _module
dependency = _module
hooks = _module
image_utils = _module
logger = _module
visualizer = _module
v5 = _module
helper = _module
common = _module
experimental = _module
yolo = _module
activations = _module
augmentations = _module
autoanchor = _module
callbacks = _module
datasets = _module
downloads = _module
general = _module
loss = _module
metrics = _module
plots = _module
torch_utils = _module

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


import numpy as np


import torch


from torch import Tensor


import warnings


import copy


from torchvision.io import read_image


from torch.jit._trace import TopLevelTracedModule


import time


import torchvision


from typing import Tuple


import logging


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


import torch.utils.data


from torch.utils.data.dataset import Dataset


from typing import Dict


from torch import nn


from torchvision.transforms import functional as F


from torchvision.transforms import transforms as T


from functools import reduce


import math


from torchvision.models._utils import IntermediateLayerGetter


import torch.nn.functional as F


from torchvision.ops import box_convert


from torchvision.ops import boxes as box_ops


from typing import cast


from torchvision.models import mobilenet


from torchvision.models.detection.backbone_utils import _validate_trainable_layers


from torchvision.ops import misc as misc_nn_ops


from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


import random


from collections import OrderedDict


from typing import Union


from collections import namedtuple


from torchvision.ops import box_iou


from typing import Mapping


from typing import Sequence


from typing import Type


from typing import Iterable


import matplotlib.pyplot as plt


from torchvision.ops.boxes import box_convert


from collections import defaultdict


from collections import deque


import torch.distributed as dist


from copy import copy


import pandas as pd


from torch.cuda import amp


from torch.nn import functional as F


from copy import deepcopy


import torch.nn as nn


from torch.hub import download_url_to_file


import re


import matplotlib


class ToTensor(nn.Module):

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


class PILToTensor(nn.Module):

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):

    def __init__(self, dtype: torch.dtype) ->None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCrop(nn.Module):

    def __init__(self, min_scale: float=0.3, max_scale: float=1.0, min_aspect_ratio: float=0.5, max_aspect_ratio: float=2.0, sampler_options: Optional[List[float]]=None, trials: int=40):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")
        if isinstance(image, Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f'image should be 2/3 dimensional. Got {image.ndimension()} dimensions.')
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        orig_w, orig_h = get_image_size(image)
        while True:
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:
                return image, target
            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    continue
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue
                cx = 0.5 * (target['boxes'][:, 0] + target['boxes'][:, 2])
                cy = 0.5 * (target['boxes'][:, 1] + target['boxes'][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue
                boxes = target['boxes'][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device))
                if ious.max() < min_jaccard_overlap:
                    continue
                target['boxes'] = boxes
                target['labels'] = target['labels'][is_within_crop_area]
                target['boxes'][:, 0::2] -= left
                target['boxes'][:, 1::2] -= top
                target['boxes'][:, 0::2].clamp_(min=0, max=new_w)
                target['boxes'][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)
                return image, target


class RandomZoomOut(nn.Module):

    def __init__(self, fill: Optional[List[float]]=None, side_range: Tuple[float, float]=(1.0, 4.0), p: float=0.5):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f'Invalid canvas side range provided {side_range}.')
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil: bool) ->int:
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f'image should be 2/3 dimensional. Got {image.ndimension()} dimensions.')
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        if torch.rand(1) >= self.p:
            return image, target
        orig_w, orig_h = get_image_size(image)
        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)
        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))
        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, Tensor):
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., top + orig_h:, :] = image[..., :, left + orig_w:] = v
        if target is not None:
            target['boxes'][:, 0::2] += left
            target['boxes'][:, 1::2] += top
        return image, target


class RandomPhotometricDistort(nn.Module):

    def __init__(self, contrast: Tuple[float]=(0.5, 1.5), saturation: Tuple[float]=(0.5, 1.5), hue: Tuple[float]=(-0.05, 0.05), brightness: Tuple[float]=(0.875, 1.125), p: float=0.5):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f'image should be 2/3 dimensional. Got {image.ndimension()} dimensions.')
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        r = torch.rand(7)
        if r[0] < self.p:
            image = self._brightness(image)
        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)
        if r[3] < self.p:
            image = self._saturation(image)
        if r[4] < self.p:
            image = self._hue(image)
        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)
        if r[6] < self.p:
            channels = get_image_num_channels(image)
            permutation = torch.randperm(channels)
            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)
        return image, target


class ModelWrapper(nn.Module):

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AnchorGenerator(nn.Module):

    def __init__(self, strides: List[int], anchor_grids: List[List[float]]):
        super().__init__()
        assert len(strides) == len(anchor_grids)
        self.strides = strides
        self.anchor_grids = anchor_grids
        self.num_layers = len(anchor_grids)
        self.num_anchors = len(anchor_grids[0]) // 2

    def _generate_grids(self, grid_sizes: List[List[int]], dtype: torch.dtype=torch.float32, device: torch.device=torch.device('cpu')) ->List[Tensor]:
        grids = []
        for height, width in grid_sizes:
            widths = torch.arange(width, dtype=torch.int32, device=device)
            heights = torch.arange(height, dtype=torch.int32, device=device)
            shift_y, shift_x = torch.meshgrid(heights, widths)
            grid = torch.stack((shift_x, shift_y), 2).expand((1, self.num_anchors, height, width, 2))
            grids.append(grid)
        return grids

    def _generate_shifts(self, grid_sizes: List[List[int]], dtype: torch.dtype=torch.float32, device: torch.device=torch.device('cpu')) ->List[Tensor]:
        anchors = torch.as_tensor(self.anchor_grids, dtype=torch.float32, device=device)
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device)
        anchors = anchors.view(self.num_layers, -1, 2) / strides.view(-1, 1, 1)
        shifts = []
        for i, (height, width) in enumerate(grid_sizes):
            shift = (anchors[i].clone() * self.strides[i]).view((1, self.num_anchors, 1, 1, 2)).expand((1, self.num_anchors, height, width, 2)).contiguous()
            shifts.append(shift)
        return shifts

    def forward(self, feature_maps: List[Tensor]) ->Tuple[List[Tensor], List[Tensor]]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        grids = self._generate_grids(grid_sizes, dtype=dtype, device=device)
        shifts = self._generate_shifts(grid_sizes, dtype=dtype, device=device)
        return grids, shifts


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):
    """
    Standard convolution

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        p (Optional[int]): padding
        g (int): groups
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version='r4.0'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if version == 'r4.0':
            self.act = nn.SiLU() if act else act if isinstance(act, nn.Module) else nn.Identity()
        elif version == 'r3.1':
            self.act = nn.Hardswish() if act else act if isinstance(act, nn.Module) else nn.Identity()
        else:
            raise NotImplementedError(f"Currently doesn't support version {version}.")

    def forward(self, x: Tensor) ->Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: Tensor) ->Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    Standard bottleneck

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, version='r4.0'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, version=version)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        n (int): number
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, version='r3.1')
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, version='r3.1')
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, version='r3.1') for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        n (int): number
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
        version (str): Module version released by ultralytics. Possible values
            are ["r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, version='r4.0'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c1, c_, 1, 1, version=version)
        self.cv3 = Conv(2 * c_, c2, 1, version=version)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, version=version) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


_block = {'r3.1': BottleneckCSP, 'r4.0': C3}


class IntermediateLevelP6(nn.Module):
    """
    This module is used to generate intermediate P6 block to the PAN.

    Args:
        x (List[Tensor]): the original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the PAN
    """

    def __init__(self, depth_multiple: float, in_channel: int, out_channel: int, version: str='r4.0'):
        super().__init__()
        block = _block[version]
        depth_gain = max(round(3 * depth_multiple), 1)
        self.p6 = nn.Sequential(Conv(in_channel, out_channel, k=3, s=2, version=version), block(out_channel, out_channel, n=depth_gain))

    def forward(self, x: List[Tensor]) ->List[Tensor]:
        x.append(self.p6(x[-1]))
        return x


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13), version='r4.0'):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, version=version)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class PathAggregationNetwork(nn.Module):
    """
    Module that adds a PAN from on top of a set of feature maps. This is based on
    `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/abs/1803.01534>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the PAN will be added.

    Args:
        in_channels (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the PAN representation
        version (str): ultralytics release version: ["r3.1", "r4.0", "r6.0"]

    Examples:

        >>> m = PathAggregationNetwork()
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 128, 52, 44)
        >>> x['feat2'] = torch.rand(1, 256, 26, 22)
        >>> x['feat3'] = torch.rand(1, 512, 13, 11)
        >>> # compute the PAN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        [('feat0', torch.Size([1, 128, 52, 44])),
         ('feat2', torch.Size([1, 256, 26, 22])),
         ('feat3', torch.Size([1, 512, 13, 11]))]
    """

    def __init__(self, in_channels: List[int], depth_multiple: float, version: str='r4.0', block: Optional[Callable[..., nn.Module]]=None, use_p6: bool=False):
        super().__init__()
        module_version = 'r4.0' if version == 'r6.0' else version
        if use_p6:
            assert len(in_channels) == 4, 'Length of in channels should be 4.'
            intermediate_blocks = IntermediateLevelP6(depth_multiple, in_channels[2], in_channels[3], version=module_version)
        else:
            assert len(in_channels) == 3, 'Length of in channels should be 3.'
            intermediate_blocks = None
        self.intermediate_blocks = intermediate_blocks
        if block is None:
            block = _block[module_version]
        depth_gain = max(round(3 * depth_multiple), 1)
        if version == 'r6.0':
            init_block = SPP(in_channels[-1], in_channels[-1], k=(5, 9, 13))
        elif version in ['r3.1', 'r4.0']:
            init_block = block(in_channels[-1], in_channels[-1], n=depth_gain, shortcut=False)
        else:
            raise NotImplementedError(f'Version {version} is not implemented yet.')
        inner_blocks = [init_block]
        if use_p6:
            in_channel = in_channels[1] + in_channels[-1]
            inner_blocks_p6 = [Conv(in_channels[-1], in_channels[2], 1, 1, version=module_version), nn.Upsample(scale_factor=2), block(in_channel, in_channels[2], n=depth_gain, shortcut=False)]
            inner_blocks.extend(inner_blocks_p6)
        inner_blocks.extend([Conv(in_channels[2], in_channels[1], 1, 1, version=module_version), nn.Upsample(scale_factor=2), block(in_channels[-1], in_channels[1], n=depth_gain, shortcut=False), Conv(in_channels[1], in_channels[0], 1, 1, version=module_version), nn.Upsample(scale_factor=2)])
        self.inner_blocks = nn.ModuleList(inner_blocks)
        layer_blocks = [block(in_channels[1], in_channels[0], n=depth_gain, shortcut=False), Conv(in_channels[0], in_channels[0], 3, 2, version=module_version), block(in_channels[1], in_channels[1], n=depth_gain, shortcut=False), Conv(in_channels[1], in_channels[1], 3, 2, version=module_version), block(in_channels[-1], in_channels[2], n=depth_gain, shortcut=False)]
        if use_p6:
            in_channel = in_channels[1] + in_channels[-1]
            layer_blocks_p6 = [Conv(in_channels[2], in_channels[2], 3, 2, version=module_version), block(in_channel, in_channels[-1], n=depth_gain, shortcut=False)]
            layer_blocks.extend(layer_blocks_p6)
        self.layer_blocks = nn.ModuleList(layer_blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) ->Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) ->Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) ->List[Tensor]:
        """
        Computes the PAN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after PAN layers.
                They are ordered from highest resolution first.
        """
        x = list(x.values())
        if self.intermediate_blocks is not None:
            x = self.intermediate_blocks(x)
        num_features = len(x)
        inners = []
        last_inner = x[-1]
        for idx in range(num_features - 1):
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx)
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx + 1)
            inners.insert(0, last_inner)
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx + 2)
            last_inner = torch.cat([last_inner, x[num_features - idx - 2]], dim=1)
        inners.insert(0, last_inner)
        results = []
        last_inner = self.get_result_from_layer_blocks(inners[0], 0)
        results.append(last_inner)
        for idx in range(num_features - 1):
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 1)
            last_inner = torch.cat([last_inner, inners[idx + 1]], dim=1)
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 2)
            results.append(last_inner)
        return results


class BackboneWithPAN(nn.Module):
    """
    Adds a PAN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        depth_multiple (float): depth multiplier
        version (str): Module version released by ultralytics: ["r3.1", "r4.0", "r6.0"].
        use_p6 (bool): Whether to use P6 layers.

    Attributes:
        out_channels (int): the number of channels in the PAN
    """

    def __init__(self, backbone, return_layers, in_channels_list, depth_multiple, version, use_p6=False):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.pan = PathAggregationNetwork(in_channels_list, depth_multiple, version=version, use_p6=use_p6)
        self.out_channels = in_channels_list

    def forward(self, x):
        x = self.body(x)
        x = self.pan(x)
        return x


class YOLOHead(nn.Module):
    """
    A regression and classification head for use in YOLO.

    Args:
        in_channels (List[int]): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        strides (List[int]): number of strides of the anchors
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels: List[int], num_anchors: int, strides: List[int], num_classes: int):
        super().__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels] * len(strides)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.strides = strides
        head_blocks = nn.ModuleList(nn.Conv2d(ch, self.num_outputs * self.num_anchors, 1) for ch in in_channels)
        for mi, s in zip(head_blocks, self.strides):
            b = mi.bias.view(self.num_anchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)
        self.head = head_blocks

    def get_result_from_head(self, features: Tensor, idx: int) ->Tensor:
        """
        This is equivalent to self.head[idx](features),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.head:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = features
        for module in self.head:
            if i == idx:
                out = module(features)
            i += 1
        return out

    def forward(self, x: List[Tensor]) ->List[Tensor]:
        all_pred_logits = []
        for i, features in enumerate(x):
            pred_logits = self.get_result_from_head(features, i)
            N, _, H, W = pred_logits.shape
            pred_logits = pred_logits.view(N, self.num_anchors, -1, H, W)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()
            all_pred_logits.append(pred_logits)
        return all_pred_logits


class SetCriterion(nn.Module):
    """
    This class computes the loss for YOLOv5.

    Args:
        num_anchors (int): The number of anchors.
        num_classes (int): The number of output classes of the model.
        fl_gamma (float): focal loss gamma (efficientDet default gamma=1.5). Default: 0.0.
        box_gain (float): box loss gain. Default: 0.05.
        cls_gain (float): class loss gain. Default: 0.5.
        cls_pos (float): cls BCELoss positive_weight. Default: 1.0.
        obj_gain (float): obj loss gain (scale with pixels). Default: 1.0.
        obj_pos (float): obj BCELoss positive_weight. Default: 1.0.
        anchor_thresh (float): anchor-multiple threshold. Default: 4.0.
        label_smoothing (float): Label smoothing epsilon. Default: 0.0.
        auto_balance (bool): Auto balance. Default: False.
    """

    def __init__(self, strides: List[int], anchor_grids: List[List[float]], num_classes: int, fl_gamma: float=0.0, box_gain: float=0.05, cls_gain: float=0.5, cls_pos: float=1.0, obj_gain: float=1.0, obj_pos: float=1.0, anchor_thresh: float=4.0, label_smoothing: float=0.0, auto_balance: bool=False) ->None:
        super().__init__()
        assert len(strides) == len(anchor_grids)
        self.num_classes = num_classes
        self.strides = strides
        self.anchor_grids = anchor_grids
        self.num_anchors = len(anchor_grids[0]) // 2
        self.balance = [4.0, 1.0, 0.4]
        self.ssi = 0
        self.sort_obj_iou = False
        self.cls_pos = cls_pos
        self.obj_pos = obj_pos
        smooth_bce = det_utils.smooth_binary_cross_entropy(eps=label_smoothing)
        self.smooth_pos = smooth_bce[0]
        self.smooth_neg = smooth_bce[1]
        self.gr = 1.0
        self.auto_balance = auto_balance
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.anchor_thresh = anchor_thresh

    def forward(self, targets: Tensor, head_outputs: List[Tensor]) ->Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            targets (Tensor): list of dicts, such that len(targets) == batch_size. The
                expected keys in each dict depends on the losses applied, see each loss' doc
            head_outputs (List[Tensor]): dict of tensors, see the output specification
                of the model for the format
        """
        device = targets.device
        anchor_grids = torch.as_tensor(self.anchor_grids, dtype=torch.float32, device=device).view(self.num_anchors, -1, 2)
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).view(-1, 1, 1)
        anchor_grids /= strides
        target_cls, target_box, indices, anchors = self.build_targets(targets, head_outputs, anchor_grids)
        pos_weight_cls = torch.as_tensor([self.cls_pos], device=device)
        pos_weight_obj = torch.as_tensor([self.obj_pos], device=device)
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        for i, pred_logits in enumerate(head_outputs):
            b, a, gj, gi = indices[i]
            target_obj = torch.zeros_like(pred_logits[..., 0], device=device)
            num_targets = b.shape[0]
            if num_targets > 0:
                pred_logits_subset = pred_logits[b, a, gj, gi]
                pred_box = det_utils.encode_single(pred_logits_subset, anchors[i])
                iou = det_utils.bbox_iou(pred_box.T, target_box[i], x1y1x2y2=False)
                loss_box += (1.0 - iou).mean()
                score_iou = iou.detach().clamp(0)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id]
                    score_iou = score_iou[sort_id]
                target_obj[b, a, gj, gi] = 1.0 - self.gr + self.gr * score_iou
                if self.num_classes > 1:
                    t = torch.full_like(pred_logits_subset[:, 5:], self.smooth_neg, device=device)
                    t[torch.arange(num_targets), target_cls[i]] = self.smooth_pos
                    loss_cls += F.binary_cross_entropy_with_logits(pred_logits_subset[:, 5:], t, pos_weight=pos_weight_cls)
            obji = F.binary_cross_entropy_with_logits(pred_logits[..., 4], target_obj, pos_weight=pos_weight_obj)
            loss_obj += obji * self.balance[i]
            if self.auto_balance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.auto_balance:
            self.balance = [(x / self.balance[self.ssi]) for x in self.balance]
        loss_box *= self.box_gain
        loss_obj *= self.obj_gain
        loss_cls *= self.cls_gain
        return {'cls_logits': loss_cls, 'bbox_regression': loss_box, 'objectness': loss_obj}

    def build_targets(self, targets: Tensor, head_outputs: List[Tensor], anchor_grids: Tensor) ->Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor, Tensor, Tensor]], List[Tensor]]:
        device = targets.device
        num_anchors = self.num_anchors
        num_targets = targets.shape[0]
        gain = torch.ones(7, device=device)
        ai = torch.arange(num_anchors, device=device).float().view(num_anchors, 1).repeat(1, num_targets)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)
        g_bias = 0.5
        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * g_bias
        target_cls, target_box, anch = [], [], []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for i in range(num_anchors):
            anchors = anchor_grids[i]
            gain[2:6] = torch.tensor(head_outputs[i].shape)[[3, 2, 3, 2]]
            targets_with_gain = targets * gain
            if num_targets > 0:
                r = targets_with_gain[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_thresh
                targets_with_gain = targets_with_gain[j]
                gxy = targets_with_gain[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                idx_jk = ((gxy % 1.0 < g_bias) & (gxy > 1.0)).T
                idx_lm = ((gxi % 1.0 < g_bias) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(idx_jk[0]), idx_jk[0], idx_jk[1], idx_lm[0], idx_lm[1]))
                targets_with_gain = targets_with_gain.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                targets_with_gain = targets[0]
                offsets = torch.tensor(0, device=device)
            idx_bc = targets_with_gain[:, :2].long().T
            gxy = targets_with_gain[:, 2:4]
            gwh = targets_with_gain[:, 4:6]
            gij = (gxy - offsets).long()
            idx_gij = gij.T
            a = targets_with_gain[:, 6].long()
            indices.append((idx_bc[0], a, idx_gij[1].clamp_(0, gain[3] - 1), idx_gij[0].clamp_(0, gain[2] - 1)))
            target_box.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            target_cls.append(idx_bc[1])
        return target_cls, target_box, indices, anch


def _concat_pred_logits(head_outputs: List[Tensor], grids: List[Tensor], shifts: List[Tensor], strides: Tensor) ->Tensor:
    batch_size, _, _, _, K = head_outputs[0].shape
    all_pred_logits = []
    for head_output, grid, shift, stride in zip(head_outputs, grids, shifts, strides):
        head_feature = torch.sigmoid(head_output)
        pred_xy, pred_wh = det_utils.decode_single(head_feature[..., :4], grid, shift, stride)
        pred_logits = torch.cat((pred_xy, pred_wh, head_feature[..., 4:]), dim=-1)
        all_pred_logits.append(pred_logits.view(batch_size, -1, K))
    all_pred_logits = torch.cat(all_pred_logits, dim=1)
    return all_pred_logits


def _decode_pred_logits(pred_logits: Tensor):
    """
    Decode the prediction logit from the PostPrecess.
    """
    scores = pred_logits[..., 5:] * pred_logits[..., 4:5]
    boxes = box_convert(pred_logits[..., :4], in_fmt='cxcywh', out_fmt='xyxy')
    return boxes, scores


class PostProcess(nn.Module):
    """
    Performs Non-Maximum Suppression (NMS) on inference results

    Args:
        strides (List[int]): Strides of the AnchorGenerator.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
    """

    def __init__(self, strides: List[int], score_thresh: float, nms_thresh: float, detections_per_img: int) ->None:
        super().__init__()
        self.strides = strides
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(self, head_outputs: List[Tensor], grids: List[Tensor], shifts: List[Tensor]) ->List[Dict[str, Tensor]]:
        """
        Perform the computation. At test time, postprocess_detections is the final layer of YOLO.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions for both confidence
        score and locations.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            grids (List[Tensor]): Anchor grids.
            shifts (List[Tensor]): Anchor shifts.
        """
        batch_size = head_outputs[0].shape[0]
        device = head_outputs[0].device
        dtype = head_outputs[0].dtype
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device)
        all_pred_logits = _concat_pred_logits(head_outputs, grids, shifts, strides)
        detections: List[Dict[str, Tensor]] = []
        for idx in range(batch_size):
            pred_logits = all_pred_logits[idx]
            boxes, scores = _decode_pred_logits(pred_logits)
            inds, labels = torch.where(scores > self.score_thresh)
            boxes, scores = boxes[inds], scores[inds, labels]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            detections.append({'scores': scores, 'labels': labels, 'boxes': boxes})
        return detections


def focus_transform(x: Tensor) ->Tensor:
    """x(b,c,w,h) -> y(b,4c,w/2,h/2)"""
    y = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    return y


class Focus(nn.Module):
    """
    Focus wh information into c-space

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        p (Optional[int]): padding
        g (int): groups
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version='r4.0'):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, version=version)

    def forward(self, x: Tensor) ->Tensor:
        y = focus_transform(x)
        out = self.conv(y)
        return out


def _make_divisible(v: float, divisor: int, min_value: Optional[int]=None) ->int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DarkNetV4(nn.Module):
    """
    DarkNetV4 main class

    Args:
        depth_multiple (float): Depth multiplier
        width_multiple (float): Width multiplier - adjusts number of channels
            in each layer by this amount
        version (str): Module version released by ultralytics, set to r4.0.
        block: Module specifying inverted residual building block for darknet
        stages_repeats (Optional[List[int]]): List of repeats number in the stages.
        stages_out_channels (Optional[List[int]]): List of channels number in the stages.
        num_classes (int): Number of classes
        round_nearest (int): Round the number of channels in each layer to be
            a multiple of this number. Set to 1 to turn off rounding
        last_channel (int): Number of the last channel
    """

    def __init__(self, depth_multiple: float, width_multiple: float, version: str='r4.0', block: Optional[Callable[..., nn.Module]]=None, stages_repeats: Optional[List[int]]=None, stages_out_channels: Optional[List[int]]=None, num_classes: int=1000, round_nearest: int=8, last_channel: int=1024) ->None:
        super().__init__()
        assert version in ['r3.1', 'r4.0'], ('Currently the module version used in DarkNetV4 is r3.1 or r4.0',)
        if block is None:
            block = _block[version]
        input_channel = 64
        if stages_repeats is None:
            stages_repeats = [3, 9, 9]
        if stages_out_channels is None:
            stages_out_channels = [128, 256, 512]
        layers: List[nn.Module] = []
        out_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Focus(3, out_channel, k=3, version=version))
        input_channel = out_channel
        for depth_gain, out_channel in zip(stages_repeats, stages_out_channels):
            depth_gain = max(round(depth_gain * depth_multiple), 1)
            out_channel = _make_divisible(out_channel * width_multiple, round_nearest)
            layers.append(Conv(input_channel, out_channel, k=3, s=2, version=version))
            layers.append(block(out_channel, out_channel, n=depth_gain))
            input_channel = out_channel
        last_channel = _make_divisible(last_channel * width_multiple, round_nearest)
        layers.append(Conv(input_channel, last_channel, k=3, s=2, version=version))
        layers.append(SPP(last_channel, last_channel, k=(5, 9, 13), version=version))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(last_channel, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


class DarkNetV6(nn.Module):
    """
    DarkNetV6 main class.

    Args:
        depth_multiple (float): Depth multiplier
        width_multiple (float): Width multiplier - adjusts number of channels
            in each layer by this amount
        version (str): Module version released by ultralytics, set to r4.0.
        block: Module specifying inverted residual building block for darknet
        stages_repeats (Optional[List[int]]): List of repeats number in the stages.
        stages_out_channels (Optional[List[int]]): List of channels number in the stages.
        num_classes (int): Number of classes
        round_nearest (int): Round the number of channels in each layer to be
            a multiple of this number. Set to 1 to turn off rounding
        last_channel (int): Number of the last channel
    """

    def __init__(self, depth_multiple: float, width_multiple: float, version: str='r4.0', block: Optional[Callable[..., nn.Module]]=None, stages_repeats: Optional[List[int]]=None, stages_out_channels: Optional[List[int]]=None, num_classes: int=1000, round_nearest: int=8, last_channel: int=1024) ->None:
        super().__init__()
        assert version == 'r4.0', 'Currently the module version used in DarkNetV6 is r4.0.'
        if block is None:
            block = C3
        input_channel = 64
        if stages_repeats is None:
            stages_repeats = [3, 6, 9]
        if stages_out_channels is None:
            stages_out_channels = [128, 256, 512]
        layers: List[nn.Module] = []
        out_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Conv(3, out_channel, k=6, s=2, p=2, version=version))
        input_channel = out_channel
        for depth_gain, out_channel in zip(stages_repeats, stages_out_channels):
            depth_gain = max(round(depth_gain * depth_multiple), 1)
            out_channel = _make_divisible(out_channel * width_multiple, round_nearest)
            layers.append(Conv(input_channel, out_channel, k=3, s=2, version=version))
            layers.append(block(out_channel, out_channel, n=depth_gain))
            input_channel = out_channel
        last_channel = _make_divisible(last_channel * width_multiple, round_nearest)
        layers.append(Conv(input_channel, last_channel, k=3, s=2, version=version))
        depth_gain = max(round(3 * depth_multiple), 1)
        layers.append(block(last_channel, last_channel, n=depth_gain))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(last_channel, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


def is_module_available(*modules: str) ->bool:
    """Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.
    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]
    if not missing:

        def decorator(func):
            return func
    else:
        req = f'module: {missing[0]}' if len(missing) == 1 else f'modules: {missing}'

        def decorator(func):

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f'{func.__module__}.{func.__name__} requires {req}')
            return wrapped
    return decorator


class YOLOTransform:

    def __init__(self, height: int, width: int, *, size_divisible: int=32, fixed_shape: Optional[Tuple[int, int]]=None, fill_color: Tuple[int, int, int]=(114, 114, 114), device: torch.device=torch.device('cpu')) ->None:
        self.height = height
        self.width = width
        self.size_divisible = size_divisible
        self.fixed_shape = fixed_shape
        self.fill_color = fill_color
        self.device = device

    def __call__(self, images):
        if isinstance(images, str):
            images = [images]
        images_info = [self.read_one_img(image) for image in images]
        images = [info[0].transpose([2, 0, 1]) for info in images_info]
        ratios = [info[1] for info in images_info]
        whs = [info[2] for info in images_info]
        return self.batch_images(images), ratios, whs

    def batch_images(self, images: List[np.ndarray]) ->Tensor:
        images = np.stack(images, 0)
        images = np.ascontiguousarray(images)
        images = images.astype(np.float32)
        images /= 255.0
        return torch.from_numpy(images)

    def read_one_img(self, image: str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, dwh = self.resize(image)
        return image, ratio, dwh

    def resize(self, image: np.ndarray):
        new_shape = self.height, self.width
        color = self.fill_color
        auto = not self.fixed_shape
        size_divisible = self.size_divisible
        return letterbox(image, new_shape, color=color, auto=auto, stride=size_divisible)


class TransformerLayer(nn.Module):
    """
    Transformer layer <https://arxiv.org/abs/2010.11929>.
    Remove the LayerNorm layers for better performance

    Args:
        c (int): number of channels
        num_heads: number of heads
    """

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """
    Vision Transformer <https://arxiv.org/abs/2010.11929>.

    Args:
        c1 (int): number of input channels
        c2 (int): number of output channels
        num_heads: number of heads
        num_layers: number of layers
    """

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2, version='r4.0')
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e, version='r4.0')
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class TransformerAttentionNetwork(PathAggregationNetwork):

    def __init__(self, in_channels_list: List[int], depth_multiple: float, version: str='r4.0', block: Optional[Callable[..., nn.Module]]=None):
        super().__init__(in_channels_list, depth_multiple, version=version, block=block)
        assert len(in_channels_list) == 3, 'Currently only supports length 3.'
        assert version == 'r4.0', 'Currently only supports version r4.0.'
        if block is None:
            block = C3
        depth_gain = max(round(3 * depth_multiple), 1)
        inner_blocks = [C3TR(in_channels_list[2], in_channels_list[2], n=depth_gain, shortcut=False), Conv(in_channels_list[2], in_channels_list[1], 1, 1, version=version), nn.Upsample(scale_factor=2), block(in_channels_list[2], in_channels_list[1], n=depth_gain, shortcut=False), Conv(in_channels_list[1], in_channels_list[0], 1, 1, version=version), nn.Upsample(scale_factor=2)]
        self.inner_blocks = nn.ModuleList(inner_blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True


class BackboneWithTAN(BackboneWithPAN):
    """
    Adds a TAN on top of a model.
    """

    def __init__(self, backbone, return_layers, in_channels_list, depth_multiple):
        super().__init__(backbone, return_layers, in_channels_list, depth_multiple, 'r4.0')
        self.pan = TransformerAttentionNetwork(in_channels_list, depth_multiple, version='r4.0')


def darknet_pan_backbone(backbone_name: str, depth_multiple: float, width_multiple: float, pretrained: Optional[bool]=False, returned_layers: Optional[List[int]]=None, version: str='r6.0', use_p6: bool=False):
    """
    Constructs a specified DarkNet backbone with PAN on top. Freezes the specified number of
    layers in the backbone.

    Examples:

        >>> from models.backbone_utils import darknet_pan_backbone
        >>> backbone = darknet_pan_backbone("darknet_s_r4_0")
        >>> # get some dummy image
        >>> x = torch.rand(1, 3, 64, 64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        [('0', torch.Size([1, 128, 8, 8])),
         ('1', torch.Size([1, 256, 4, 4])),
         ('2', torch.Size([1, 512, 2, 2]))]

    Args:
        backbone_name (string): darknet architecture. Possible values are "darknet_s_r3_1",
            "darknet_m_r3_1", "darknet_l_r3_1", "darknet_s_r4_0", "darknet_m_r4_0",
            "darknet_l_r4_0", "darknet_s_r6_0", "darknet_m_r6_0", and "darknet_l_r6_0".
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        use_p6 (bool): Whether to use P6 layers.
    """
    assert version in ['r3.1', 'r4.0', 'r6.0'], "Currently only supports version 'r3.1', 'r4.0' and 'r6.0'."
    last_channel = 768 if use_p6 else 1024
    backbone = darknet.__dict__[backbone_name](pretrained=pretrained, last_channel=last_channel).features
    if returned_layers is None:
        returned_layers = [4, 6, 8]
    return_layers = {str(k): str(i) for i, k in enumerate(returned_layers)}
    grow_widths = [256, 512, 768, 1024] if use_p6 else [256, 512, 1024]
    in_channels_list = [int(gw * width_multiple) for gw in grow_widths]
    return BackboneWithPAN(backbone, return_layers, in_channels_list, depth_multiple, version, use_p6=use_p6)


def get_yolov5_size(depth_multiple, width_multiple):
    if depth_multiple == 0.33 and width_multiple == 0.25:
        return 'n'
    if depth_multiple == 0.33 and width_multiple == 0.5:
        return 's'
    if depth_multiple == 0.67 and width_multiple == 0.75:
        return 'm'
    if depth_multiple == 1.0 and width_multiple == 1.0:
        return 'l'
    if depth_multiple == 1.33 and width_multiple == 1.25:
        return 'x'
    raise NotImplementedError(f"Currently does't support architecture with depth: {depth_multiple} and {width_multiple}, fell free to create a ticket labeled enhancement to us")


def obtain_module_sequential(state_dict):
    if isinstance(state_dict, nn.Sequential):
        return state_dict
    else:
        return obtain_module_sequential(state_dict.model)


def rgetattr(obj, attr, *args):
    """
    Nested version of getattr.
    Ref: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr)


class CheckpointConverter:
    """
    Update checkpoint from ultralytics yolov5.
    """

    def __init__(self, depth_multiple: float, width_multiple: float, inner_block_maps: Optional[Dict[str, str]]=None, layer_block_maps: Optional[Dict[str, str]]=None, p6_block_maps: Optional[Dict[str, str]]=None, strides: Optional[List[int]]=None, anchor_grids: Optional[List[List[float]]]=None, head_ind: int=24, head_name: str='m', num_classes: int=80, version: str='r6.0', use_p6: bool=False) ->None:
        if inner_block_maps is None:
            inner_block_maps = {'0': '9', '1': '10', '3': '13', '4': '14'}
        self.inner_block_maps = inner_block_maps
        if layer_block_maps is None:
            layer_block_maps = {'0': '17', '1': '18', '2': '20', '3': '21', '4': '23'}
        self.layer_block_maps = layer_block_maps
        self.p6_block_maps = p6_block_maps
        self.head_ind = head_ind
        self.head_name = head_name
        yolov5_size = get_yolov5_size(depth_multiple, width_multiple)
        backbone_name = f"darknet_{yolov5_size}_{version.replace('.', '_')}"
        backbone = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6)
        num_anchors = len(anchor_grids[0]) // 2
        head = YOLOHead(backbone.out_channels, num_anchors, strides, num_classes)
        self.model = ModelWrapper(backbone, head)

    def updating(self, state_dict):
        state_dict = obtain_module_sequential(state_dict)
        for name, params in self.model.backbone.body.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, None))
        for name, buffers in self.model.backbone.body.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, None))
        if self.p6_block_maps is not None:
            for name, params in self.model.backbone.pan.intermediate_blocks.p6.named_parameters():
                params.data.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))
            for name, buffers in self.model.backbone.pan.intermediate_blocks.p6.named_buffers():
                buffers.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))
        for name, params in self.model.backbone.pan.inner_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))
        for name, buffers in self.model.backbone.pan.inner_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))
        for name, params in self.model.backbone.pan.layer_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))
        for name, buffers in self.model.backbone.pan.layer_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))
        for name, params in self.model.head.named_parameters():
            params.data.copy_(self.attach_parameters_heads(state_dict, name))
        for name, buffers in self.model.head.named_buffers():
            buffers.copy_(self.attach_parameters_heads(state_dict, name))

    @staticmethod
    def attach_parameters_block(state_dict, name, block_maps=None):
        keys = name.split('.')
        ind = int(block_maps[keys[0]]) if block_maps else int(keys[0])
        return rgetattr(state_dict[ind], keys[1:])

    def attach_parameters_heads(self, state_dict, name):
        keys = name.split('.')
        ind = int(keys[1])
        return rgetattr(getattr(state_dict[self.head_ind], self.head_name)[ind], keys[2:])


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def safe_download(file, url, url2=None, min_bytes=1.0, error_msg='', hash_prefix=None):
    """
    Attempts to download file from url or url2, checks
    and removes incomplete downloads < min_bytes
    """
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:
        None
        download_url_to_file(url, str(file), hash_prefix=hash_prefix)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        file.unlink(missing_ok=True)
        None
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink(missing_ok=True)
            None
        None


def attempt_download(file, repo='ultralytics/yolov5', hash_prefix=None):
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            name = name.split('?')[0]
            safe_download(file=name, url=url, min_bytes=100000.0, hash_prefix=hash_prefix)
            return name
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']
        except Exception as e:
            None
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception as e:
                None
                tag = 'v6.0'
        if name in assets:
            safe_download(file, url=f'https://github.com/{repo}/releases/download/{tag}/{name}', min_bytes=100000.0, error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')
    return str(file)


def load_yolov5_model(checkpoint_path: str, fuse: bool=False):
    """
    Creates a specified YOLOv5 model.

    Note:
        Currently this tool is mainly used to load the checkpoints trained by yolov5
        with support for versions v3.1, v4.0 (v5.0) and v6.0 (v6.1). In addition it is
        available for inference with AutoShape attached for versions v6.0 (v6.1).

    Args:
        checkpoint_path (str): path of the YOLOv5 model, i.e. 'yolov5s.pt'
        fuse (bool): fuse model Conv2d() + BatchNorm2d() layers. Default: False

    Returns:
        YOLOv5 pytorch model
    """
    with add_yolov5_context():
        ckpt = torch.load(attempt_download(checkpoint_path), map_location=torch.device('cpu'))
        if fuse:
            model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
        else:
            model = ckpt['ema' if ckpt.get('ema') else 'model'].float().eval()
        for sub_module in model.modules():
            if isinstance(sub_module, Detect):
                if not isinstance(sub_module.anchor_grid, list):
                    delattr(sub_module, 'anchor_grid')
                    setattr(sub_module, 'anchor_grid', [torch.zeros(1)] * sub_module.nl)
            if isinstance(sub_module, nn.Upsample) and not hasattr(sub_module, 'recompute_scale_factor'):
                sub_module.recompute_scale_factor = None
        return model


def load_from_ultralytics(checkpoint_path: str, version: str='r6.0'):
    """
    Allows the user to load model state file from the checkpoint trained from
    the ultralytics/yolov5.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
    """
    if version not in ['r3.1', 'r4.0', 'r6.0']:
        raise NotImplementedError(f'Currently does not support version: {version}. Feel free to file an issue labeled enhancement to us.')
    checkpoint_yolov5 = load_yolov5_model(checkpoint_path)
    num_classes = checkpoint_yolov5.yaml['nc']
    strides = checkpoint_yolov5.stride
    num_anchors = checkpoint_yolov5.model[-1].anchors.shape[1]
    anchor_grids = (checkpoint_yolov5.model[-1].anchors * checkpoint_yolov5.model[-1].stride.view(-1, 1, 1)).reshape(1, -1, 2 * num_anchors).tolist()[0]
    depth_multiple = checkpoint_yolov5.yaml['depth_multiple']
    width_multiple = checkpoint_yolov5.yaml['width_multiple']
    use_p6 = False
    if len(strides) == 4:
        use_p6 = True
    if use_p6:
        inner_block_maps = {'0': '11', '1': '12', '3': '15', '4': '16', '6': '19', '7': '20'}
        layer_block_maps = {'0': '23', '1': '24', '2': '26', '3': '27', '4': '29', '5': '30', '6': '32'}
        p6_block_maps = {'0': '9', '1': '10'}
        head_ind = 33
        head_name = 'm'
    else:
        inner_block_maps = {'0': '9', '1': '10', '3': '13', '4': '14'}
        layer_block_maps = {'0': '17', '1': '18', '2': '20', '3': '21', '4': '23'}
        p6_block_maps = None
        head_ind = 24
        head_name = 'm'
    convert_yolo_checkpoint = CheckpointConverter(depth_multiple, width_multiple, inner_block_maps=inner_block_maps, layer_block_maps=layer_block_maps, p6_block_maps=p6_block_maps, strides=strides, anchor_grids=anchor_grids, head_ind=head_ind, head_name=head_name, num_classes=num_classes, version=version, use_p6=use_p6)
    convert_yolo_checkpoint.updating(checkpoint_yolov5)
    state_dict = convert_yolo_checkpoint.model.half().state_dict()
    size = get_yolov5_size(depth_multiple, width_multiple)
    return {'num_classes': num_classes, 'depth_multiple': depth_multiple, 'width_multiple': width_multiple, 'strides': strides, 'anchor_grids': anchor_grids, 'use_p6': use_p6, 'size': size, 'state_dict': state_dict}


class YOLO(nn.Module):
    """
    Implements YOLO series model.

    The input to the model is expected to be a batched tensors, of shape ``[N, C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    """

    def __init__(self, backbone: nn.Module, num_classes: int, strides: Optional[List[int]]=None, anchor_grids: Optional[List[List[float]]]=None, anchor_generator: Optional[nn.Module]=None, head: Optional[nn.Module]=None, criterion: Optional[Callable[..., Dict[str, Tensor]]]=None, score_thresh: float=0.005, nms_thresh: float=0.45, detections_per_img: int=300, post_process: Optional[nn.Module]=None):
        super().__init__()
        if not hasattr(backbone, 'out_channels'):
            raise ValueError('backbone should contain an attribute out_channels specifying the number of output channels (assumed to be the same for all the levels)')
        self.backbone = backbone
        if strides is None:
            strides: List[int] = [8, 16, 32]
        if anchor_grids is None:
            anchor_grids: List[List[float]] = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        if anchor_generator is None:
            anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.anchor_generator = anchor_generator
        if criterion is None:
            criterion = SetCriterion(strides, anchor_grids, num_classes)
        self.compute_loss = criterion
        if head is None:
            head = YOLOHead(backbone.out_channels, anchor_generator.num_anchors, anchor_generator.strides, num_classes)
        self.head = head
        if post_process is None:
            post_process = PostProcess(anchor_generator.strides, score_thresh, nms_thresh, detections_per_img)
        self.post_process = post_process
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) ->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses
        return detections

    def forward(self, samples: Tensor, targets: Optional[Tensor]=None) ->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            samples (NestedTensor): Expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        features = self.backbone(samples)
        head_outputs = self.head(features)
        grids, shifts = self.anchor_generator(features)
        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None
            losses = self.compute_loss(targets, head_outputs)
        else:
            detections = self.post_process(head_outputs, grids, shifts)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('YOLO always returns a (Losses, Detections) tuple in scripting.')
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    @classmethod
    def load_from_yolov5(cls, checkpoint_path: str, score_thresh: float=0.25, nms_thresh: float=0.45, version: str='r6.0', post_process: Optional[nn.Module]=None):
        """
        Load model state from the checkpoint trained by YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            version (str): upstream version released by the ultralytics/yolov5, Possible
                values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        """
        model_info = load_from_ultralytics(checkpoint_path, version=version)
        backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
        depth_multiple = model_info['depth_multiple']
        width_multiple = model_info['width_multiple']
        use_p6 = model_info['use_p6']
        backbone = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6)
        model = cls(backbone, model_info['num_classes'], strides=model_info['strides'], anchor_grids=model_info['anchor_grids'], score_thresh=score_thresh, nms_thresh=nms_thresh, post_process=post_process)
        model.load_state_dict(model_info['state_dict'])
        return model


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone: nn.Module, return_layers: Dict[str, str], in_channels_list: List[int], out_channels: int, extra_blocks: Optional[ExtraFPNBlock]=None) ->None:
        super().__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return list(x.values())


@torch.jit.unused
def _get_shape_onnx(image: Tensor) ->Tensor:
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]


def contains_any_tensor(value: Any, dtype: Type=Tensor) ->bool:
    """
    Determine whether or not a list contains any Type
    """
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False


class YOLOv5(nn.Module):
    """
    Wrapping the pre-processing (`LetterBox`) into the YOLO models.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size that maintains the aspect ratio before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        Demo pipeline for YOLOv5 Inference.

        .. code-block:: python

            from yolort.models import YOLOv5

            # Load the yolov5s version 6.0 models
            arch = 'yolov5_darknet_pan_s_r60'
            model = YOLOv5(arch=arch, pretrained=True, score_thresh=0.35)
            model = model.eval()

            # Perform inference on an image file
            predictions = model.predict('bus.jpg')
            # Perform inference on a list of image files
            predictions2 = model.predict(['bus.jpg', 'zidane.jpg'])

        We also support loading the custom checkpoints trained from ultralytics/yolov5

        .. code-block:: python

            from yolort.models import YOLOv5

            # Your trained checkpoint from ultralytics
            checkpoint_path = 'yolov5n.pt'
            model = YOLOv5.load_from_yolov5(checkpoint_path, score_thresh=0.35)
            model = model.eval()

            # Perform inference on an image file
            predictions = model.predict('bus.jpg')

    Args:
        arch (string): YOLO model architecture. Default: None
        model (nn.Module): YOLO model. Default: None
        num_classes (int): number of output classes of the model (doesn't including
            background). Default: 80
        pretrained (bool): If true, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(self, arch: Optional[str]=None, model: Optional[nn.Module]=None, num_classes: int=80, pretrained: bool=False, progress: bool=True, size: Tuple[int, int]=(640, 640), size_divisible: int=32, fixed_shape: Optional[Tuple[int, int]]=None, fill_color: int=114, **kwargs: Any) ->None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        if model is None:
            model = yolo.__dict__[arch](pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)
        self.model = model
        self.transform = YOLOTransform(size[0], size[1], size_divisible=size_divisible, fixed_shape=fixed_shape, fill_color=fill_color)
        self._has_warned = False

    def forward(self, inputs: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]=None) ->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        original_image_sizes: List[Tuple[int, int]] = []
        if not self.training:
            for img in inputs:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
        samples, targets = self.transform(inputs, targets)
        outputs = self.model(samples.tensors, targets=targets)
        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if torch.jit.is_scripting():
                losses = outputs[0]
            else:
                losses = outputs
        else:
            if torch.jit.is_scripting():
                result = outputs[1]
            else:
                result = outputs
            if torchvision._is_tracing():
                im_shape = _get_shape_onnx(samples.tensors)
            else:
                im_shape = torch.tensor(samples.tensors.shape[-2:])
            detections = self.transform.postprocess(result, im_shape, original_image_sizes)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('YOLOv5 always returns a (Losses, Detections) tuple in scripting.')
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) ->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses
        return detections

    @torch.no_grad()
    def predict(self, x: Any, image_loader: Optional[Callable]=None) ->List[Dict[str, Tensor]]:
        """
        Predict function for raw data or processed data

        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self.forward(images)

    def default_loader(self, img_path: str) ->Tensor:
        """
        Default loader of read a image path.

        Args:
            img_path (str): a image path

        Returns:
            Tensor, processed tensor for prediction.
        """
        return read_image(img_path) / 255.0

    def collate_images(self, samples: Any, image_loader: Callable) ->List[Tensor]:
        """
        Prepare source samples for inference.

        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - Tensor or List[Tensor]: a tensor or list of tensors.

        Returns:
            List[Tensor], The processed image samples.
        """
        p = next(self.parameters())
        if isinstance(samples, Tensor):
            return [samples.type_as(p)]
        if contains_any_tensor(samples):
            return [sample.type_as(p) for sample in samples]
        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample).type_as(p)
                outputs.append(output)
            return outputs
        raise NotImplementedError(f"The type of the sample is {type(samples)}, we currently don't support it now, the samples should be either a tensor, list of tensors, a image path or list of image paths.")

    @classmethod
    def load_from_yolov5(cls, checkpoint_path: str, *, size: Tuple[int, int]=(640, 640), size_divisible: int=32, fixed_shape: Optional[Tuple[int, int]]=None, fill_color: int=114, **kwargs: Any):
        """
        Load custom checkpoints trained from YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
                Default: (640, 640)
            size_divisible (int): stride of the models. Default: 32
            fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
                the image will be padded to shape `fixed_shape` if specified. Instead the image will
                be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
                is divisible by `size_divisible` if it is not specified. Default: None
            fill_color (int): fill value for padding. Default: 114
        """
        model = YOLO.load_from_yolov5(checkpoint_path, **kwargs)
        yolov5 = cls(model=model, size=size, size_divisible=size_divisible, fixed_shape=fixed_shape, fill_color=fill_color)
        return yolov5


class NonMaxSupressionOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        """
        Symbolic method to export an NonMaxSupression ONNX models.

        Args:
            boxes (Tensor): An input tensor with shape [num_batches, spatial_dimension, 4].
                have been multiplied original size here.
            scores (Tensor): An input tensor with shape [num_batches, num_classes, spatial_dimension].
                only one class score here.
            detections_per_class (Tensor, optional): Integer representing the maximum number of
                boxes to be selected per batch per class. It is a scalar.
            iou_thresh (Tensor, optional): Float representing the threshold for deciding whether
                boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
            score_thresh (Tensor, optional): Float representing the threshold for deciding when to
                remove boxes based on score. It is a scalar.

        Returns:
            Tensor(int64): selected indices from the boxes tensor. [num_selected_indices, 3],
                the selected index format is [batch_index, class_index, box_index].
        """
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        return g.op('NonMaxSuppression', boxes, scores, detections_per_class, iou_thresh, score_thresh)


class FakePostProcess(nn.Module):
    """
    Fake PostProcess used to export an ONNX models containing NMS for ONNX Runtime and OpenVINO.

    Args:
        iou_thresh (float, optional): NMS threshold used for postprocessing the detections.
            Default to 0.45
        score_thresh (float, optional): Score threshold used for postprocessing the detections.
            Default to 0.35
        detections_per_img (int, optional): Number of best detections to keep after NMS.
            Default to 100
        export_type (str, optional): Export onnx backend support onnxruntime and openvino
    """

    def __init__(self, iou_thresh: float=0.45, score_thresh: float=0.35, detections_per_img: int=100, export_type='onnxruntime'):
        super().__init__()
        self.detections_per_img = detections_per_img
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.export_type = export_type
        self.nms_func = NonMaxSupressionOp.apply

    def forward(self, x: Tensor):
        device = x.device
        boxes, scores = _decode_pred_logits(x)
        scores, classes = scores.max(2, keepdim=True)
        scores_t = scores.transpose(1, 2).contiguous()
        detections_per_img = torch.tensor([self.detections_per_img])
        iou_thresh = torch.tensor([self.iou_thresh])
        score_thresh = torch.tensor([self.score_thresh])
        selected_indices = self.nms_func(boxes, scores_t, detections_per_img, iou_thresh, score_thresh)
        i, k = selected_indices[:, 0], selected_indices[:, 2]
        if self.export_type == 'openvino':
            i, k = i[i >= 0], k = k[k >= 0]
        boxes_keep = boxes[i, k, :]
        classes_keep = classes[i, k, :]
        scores_keep = scores[i, k, :]
        i = i.unsqueeze(1)
        i = i.float()
        classes_keep = classes_keep.float()
        out = torch.concat([i, boxes_keep, classes_keep, scores_keep], 1)
        return out


class FakeYOLO(nn.Module):
    """
    Fake YOLO used to export an ONNX models for ONNX Runtime and OpenVINO.
    """

    def __init__(self, model: nn.Module, iou_thresh: float=0.45, score_thresh: float=0.35, detections_per_img: int=100):
        super().__init__()
        self.model = model
        self.post_process = FakePostProcess(iou_thresh=iou_thresh, score_thresh=score_thresh, detections_per_img=detections_per_img)

    def forward(self, x):
        x = self.model(x)
        out = self.post_process(x)
        return out


class LogitsDecoder(nn.Module):
    """
    This is a simplified version of post-processing module, we manually remove
    the ``torchvision::ops::nms``, and it will be used later in the procedure for
    exporting the ONNX Graph to YOLOTRTModule or others.
    """

    def __init__(self, strides: List[int]) ->None:
        """
        Args:
            strides (List[int]): Strides of the AnchorGenerator.
        """
        super().__init__()
        self.strides = strides

    def forward(self, head_outputs: List[Tensor], grids: List[Tensor], shifts: List[Tensor]) ->Tuple[Tensor, Tensor]:
        """
        Just concat the predict logits, ignore the original ``torchvision::nms`` module
        from original ``yolort.models.box_head.PostProcess``.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            grids (List[Tensor]): Anchor grids.
            shifts (List[Tensor]): Anchor shifts.
        """
        batch_size = head_outputs[0].shape[0]
        device = head_outputs[0].device
        dtype = head_outputs[0].dtype
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device)
        all_pred_logits = _concat_pred_logits(head_outputs, grids, shifts, strides)
        bbox_regression = []
        pred_scores = []
        for idx in range(batch_size):
            pred_logits = all_pred_logits[idx]
            boxes, scores = _decode_pred_logits(pred_logits)
            bbox_regression.append(boxes)
            pred_scores.append(scores)
        boxes = torch.stack(bbox_regression)
        scores = torch.stack(pred_scores)
        return boxes, scores


def dict_to_tuple(out_dict: Dict[str, Tensor]) ->Tuple:
    """
    Convert the model output dictionary to tuple format.
    """
    if 'masks' in out_dict.keys():
        return out_dict['boxes'], out_dict['scores'], out_dict['labels'], out_dict['masks']
    return out_dict['boxes'], out_dict['scores'], out_dict['labels']


class TraceWrapper(nn.Module):
    """
    This is a wrapper for `torch.jit.trace`, as there are some scenarios
    where `torch.jit.script` support is limited.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return dict_to_tuple(out[0])


class FeatureExtractor(nn.Module):

    def __init__(self, model: nn.Module, return_layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.return_layers = return_layers
        self._features = {layer: torch.empty(0) for layer in return_layers}
        for layer_id in return_layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) ->Callable:

        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, images: Tensor, targets: Tensor) ->Dict[str, Tensor]:
        _ = self.model(images, targets)
        return self._features


class DWConv(Conv):
    """
    Depth-wise convolution class.

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, act=True, version='r4.0'):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act, version=version)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    """

    def __init__(self, c1, c2, k=5, version='r4.0'):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, version=version)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus2(nn.Module):
    """
    Is the Focus layer equivalent to a simple Conv layer?
    https://github.com/ultralytics/yolov5/issues/4825#issue-998038464

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        p (Optional[int]): padding
        g (int): groups
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version='r4.0'):
        super().__init__()
        self.register_buffer('filter1', torch.tensor([[1, 0], [0, 0]]).float().expand(3, 1, 2, 2))
        self.register_buffer('filter2', torch.tensor([[0, 0], [1, 0]]).float().expand(3, 1, 2, 2))
        self.register_buffer('filter3', torch.tensor([[0, 1], [0, 0]]).float().expand(3, 1, 2, 2))
        self.register_buffer('filter4', torch.tensor([[0, 0], [0, 1]]).float().expand(3, 1, 2, 2))
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, version=version)

    def forward(self, x: Tensor) ->Tensor:
        conv1 = F.conv2d(x, self.filter1, stride=2, groups=3)
        conv2 = F.conv2d(x, self.filter2, stride=2, groups=3)
        conv3 = F.conv2d(x, self.filter3, stride=2, groups=3)
        conv4 = F.conv2d(x, self.filter4, stride=2, groups=3)
        return self.conv(torch.cat([conv1, conv2, conv3, conv4], dim=1))


class Concat(nn.Module):

    def __init__(self, dimension: int=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[Tensor]) ->Tensor:
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x
        return torch.cat(prev_features, self.d)


class Flatten(nn.Module):

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class C3SPP(C3):

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s ** 2, h * s, w * s)


def is_writeable(dir, test=False):
    """
    Return True if directory has write permissions, test opening a file
    with write permissions if test=True
    """
    if test:
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):
                pass
            file.unlink()
            return True
        except OSError:
            return False
    else:
        return os.access(dir, os.R_OK)


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    """
    Return path of user configuration directory. Prefer environment
    variable if exists. Make dir if required.
    """
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}
        path = Path.home() / cfg.get(platform.system(), '')
        path = (path if is_writeable(path) else Path('/tmp')) / dir
    path.mkdir(exist_ok=True)
    return path


def check_font(font='Arial.ttf', size=10):
    font = Path(font)
    font = font if font.exists() else CONFIG_DIR / font.name
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:
        None
        url = 'https://ultralytics.com/assets/' + font.name
        None
        torch.hub.download_url_to_file(url, str(font), progress=False)


def is_ascii(s=''):
    """
    Is string composed of all ASCII (no UTF) characters?
    (note str().isascii() introduced in python 3.7)
    """
    s = str(s)
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    return re.search('[一-\u9fff]', s)


def set_logging(name=None, verbose=True):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format='%(message)s', level=logging.INFO if verbose and rank in (-1, 0) else logging.WARNING)
    return logging.getLogger(name)


class Colors:

    def __init__(self):
        hex = 'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def colorstr(*input):
    """
    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code,
    i.e.  colorstr('blue', 'hello world')
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m', 'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m', 'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m', 'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m', 'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m', 'underline': '\x1b[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment file or directory path.
    i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f'{path}{sep}*')
        matches = [re.search(f'%s{sep}(\\d+)' % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f'{path}{sep}{n}{suffix}')
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """
    Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    max_wh = 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if time.time() - t > time_limit:
            None
            break
    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class AutoShape(nn.Module):
    """
    YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs.
    Includes preprocessing, inference and NMS
    """
    conf = 0.25
    iou = 0.45
    classes = None
    multi_label = False
    max_det = 1000

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())
        self.model = model.eval()

    def _apply(self, fn):
        """
        Apply to(), cpu(), cuda(), half() to model tensors that
        are not parameters or registered buffers
        """
        self = super()._apply(fn)
        m = self.model.model[-1]
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        """
        Inference from various sources. For height=640, width=1280, RGB images example inputs are:
            - file: imgs = 'data/images/zidane.jpg'  # str or PosixPath
            - URI: = 'https://ultralytics.com/images/zidane.jpg'
            - OpenCV: = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
            - PIL: = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
            - numpy: = np.zeros((640,1280,3))  # HWC
            - torch: = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
            - multiple: = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
        """
        t = [time_sync()]
        p = next(self.model.parameters())
        if isinstance(imgs, Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.type_as(p), augment, profile)
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, (str, Path)):
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = size / max(s)
            shape1.append([(y * g) for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).type_as(p) / 255
        t.append(time_sync())
        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)
            t.append(time_sync())
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, multi_label=self.multi_label, max_det=self.max_det)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class CrossConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super().__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)
        return y, None


PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    """
    Check anchor order against stride order for YOLOv5 Detect() module m,
    and correct if necessary
    """
    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def fuse_conv_and_bn(conv, bn):
    """
    Fuse convolution and batchnorm layers
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        None
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            None
    try:
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1000000000.0 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)
    except (ImportError, Exception):
        fs = ''
    LOGGER.info(f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


def parse_model(d, ch):
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """
    Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    """
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = int(h * ratio), int(w * ratio)
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class SiLU(nn.Module):
    """
    Export-friendly version of nn.SiLU(). Starting with PyTorch 1.8,
    this operator supports exporting to ONNX, and there is also a
    build-in implementation of it on TVM.

    Ref: <https://arxiv.org/pdf/1606.08415.pdf>
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """
    Export-friendly version of nn.Hardswish(). Starting with PyTorch 1.8,
    this operator supports exporting to ONNX, and currently this module
    is only used for TVM.
    """

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (C3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (C3SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classify,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Concat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Contract,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNetV4,
     lambda: ([], {'depth_multiple': 1, 'width_multiple': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (DarkNetV6,
     lambda: ([], {'depth_multiple': 1, 'width_multiple': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Expand,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hardswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MixConv2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QFocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RandomZoomOut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sum,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (YOLOHead,
     lambda: ([], {'in_channels': [4, 4], 'num_anchors': 4, 'strides': [4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_zhiqwang_yolov5_rt_stack(_paritybench_base):
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

