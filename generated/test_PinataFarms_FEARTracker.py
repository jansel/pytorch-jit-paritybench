import sys
_module = sys.modules[__name__]
del sys
demo_video = _module
evaluate = _module
coreml_convert = _module
coreml_utils = _module
macs_params = _module
model_training = _module
constants = _module
dataset = _module
aug = _module
box_coder = _module
siam_dataset = _module
track_sampling = _module
tracking_dataset = _module
utils = _module
metrics = _module
dataset_aware_metric = _module
tracking = _module
blocks = _module
fear_net = _module
tracker = _module
base_tracker = _module
fear_tracker = _module
train = _module
base_lightning_model = _module
callbacks = _module
fear_lightning_model = _module
loss = _module
trainer = _module
hydra = _module
logger = _module
utils = _module

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


from typing import Optional


from typing import List


import torch


import copy


from functools import partial


from typing import Dict


from typing import Any


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


from abc import abstractmethod


from abc import ABC


from collections import namedtuple


from typing import Union


from typing import Tuple


import random


from torch.utils.data.dataloader import default_collate


from collections import defaultdict


from typing import Callable


import torch.nn


from torch import Tensor


import torch.nn as nn


from collections import deque


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from typing import Iterable


from torchvision.ops import box_convert


from torchvision.ops import box_iou


from torch import nn


from torch.types import Device


class CoreMLTrackingWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred['TARGET_REGRESSION_LABEL_KEY'], pred['TARGET_CLASSIFICATION_KEY']]


class ProfileTrackingWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred['TARGET_REGRESSION_LABEL_KEY'], pred['TARGET_CLASSIFICATION_KEY']]


class Encoder(nn.Module):

    def __init__(self, pretrained: bool=True) ->None:
        super().__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        self.encoder_channels = {'layer0': 352, 'layer1': 112, 'layer2': 32, 'layer3': 24, 'layer4': 16}

    def _load_model(self) ->Any:
        model_name = 'fbnet_c'
        model = fbnet(model_name, pretrained=self.pretrained)
        return model

    def _get_stages(self) ->List[Any]:
        stages = [self.model.backbone.stages[:2], self.model.backbone.stages[2:5], self.model.backbone.stages[5:9], self.model.backbone.stages[9:18], self.model.backbone.stages[18:23]]
        return stages

    def forward(self, x: Any) ->List[Any]:
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class SepConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=0, dilation: Union[int, Tuple[int, int]]=1, bias: bool=True) ->None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdjustLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, crop_rate: int=4):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.size_threshold = 20
        self.crop_rate = crop_rate

    def forward(self, x):
        x_ori = self.downsample(x)
        adjust = x_ori
        return adjust


class MatrixMobile(nn.Module):
    """
    Encode backbone feature
    """

    def __init__(self, in_channels, out_channels, conv_block: str='regular'):
        super().__init__()
        self.matrix11_s = nn.Sequential(SepConv(in_channels, out_channels, kernel_size=3, bias=False, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, z, x):
        return z.reshape(z.size(0), z.size(1), -1), self.matrix11_s(x)


class MobileCorrelation(nn.Module):
    """
    Mobile Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int=64, conv_block: str='regular'):
        super().__init__()
        self.enc = nn.Sequential(SepConv(num_channels + num_corr_channels, num_channels, kernel_size=3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True))

    def forward(self, z, x):
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([x, s], dim=1)
        s = self.enc(s)
        return s


class BoxTower(nn.Module):
    """
    Box Tower for FCOS regression
    """

    def __init__(self, towernum: int=4, conv_block: str='regular', inchannels: int=512, outchannels: int=256, mobile: bool=False):
        super().__init__()
        tower = []
        cls_tower = []
        self.cls_encode = MatrixMobile(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        self.reg_encode = MatrixMobile(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        self.cls_dw = MobileCorrelation(num_channels=outchannels, conv_block=conv_block)
        self.reg_dw = MobileCorrelation(num_channels=outchannels, conv_block=conv_block)
        for i in range(towernum):
            tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())
        for i in range(towernum):
            cls_tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())
        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.bbox_pred = SepConv(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = SepConv(outchannels, 1, kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, search, kernel, update=None):
        if update is None:
            cls_z, cls_x = self.cls_encode(kernel, search)
        else:
            cls_z, cls_x = self.cls_encode(update, search)
        reg_z, reg_x = self.reg_encode(kernel, search)
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)
        x_reg = self.bbox_tower(reg_dw)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)
        c = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(c)
        return x, cls, cls_dw, x_reg


TARGET_CLASSIFICATION_KEY = 'TARGET_CLASSIFICATION_KEY'


TARGET_REGRESSION_LABEL_KEY = 'TARGET_REGRESSION_LABEL_KEY'


@torch.no_grad()
def make_grid(score_size: int, total_stride: int, instance_size: int) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Each element of feature map on input search image
    :return: H*W*2 (position for each element)
    """
    x, y = np.meshgrid(np.arange(0, score_size) - np.floor(float(score_size // 2)), np.arange(0, score_size) - np.floor(float(score_size // 2)))
    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    grid_x = torch.from_numpy(grid_x[np.newaxis, :, :])
    grid_y = torch.from_numpy(grid_y[np.newaxis, :, :])
    return grid_x, grid_y


class FEARNet(nn.Module):

    def __init__(self, backbone, img_size: int, pretrained: bool=True, score_size: int=25, adjust_channels: int=256, total_stride: int=8, instance_size: int=255, towernum: int=4, max_layer: int=3, crop_template_features: bool=True, conv_block: str='regular', mobile: bool=False, **kwargs) ->None:
        max_layer2name = {(3): 'layer2', (4): 'layer1'}
        assert max_layer in max_layer2name
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.neck = AdjustLayer(in_channels=self.encoder.encoder_channels[max_layer2name[max_layer]], out_channels=adjust_channels)
        self.connect_model = BoxTower(inchannels=adjust_channels, outchannels=adjust_channels, towernum=towernum, conv_block=conv_block, mobile=mobile)
        self.search_size = img_size
        self.score_size = score_size
        self.total_stride = total_stride
        self.instance_size = instance_size
        self.size = 1
        self.max_layer = max_layer
        self.crop_template_features = crop_template_features
        self.grid_x = torch.empty(0)
        self.grid_y = torch.empty(0)
        self.features = None
        self.grids(self.size)

    def feature_extractor(self, x: torch.Tensor) ->torch.Tensor:
        for stage in self.encoder.stages[:self.max_layer]:
            x = stage(x)
        return x

    def get_features(self, crop: torch.Tensor) ->torch.Tensor:
        features = self.feature_extractor(crop)
        features = self.neck(features)
        return features

    def grids(self, size: int) ->None:
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        grid_x, grid_y = make_grid(self.score_size, self.total_stride, self.instance_size)
        self.grid_x, self.grid_y = grid_x.unsqueeze(0).repeat(size, 1, 1, 1), grid_y.unsqueeze(0).repeat(size, 1, 1, 1)

    def connector(self, template_features: torch.Tensor, search_features: torch.Tensor) ->Dict[str, torch.Tensor]:
        bbox_pred, cls_pred, _, _ = self.connect_model(search_features, template_features)
        return {TARGET_REGRESSION_LABEL_KEY: bbox_pred, TARGET_CLASSIFICATION_KEY: cls_pred}

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) ->Dict[str, torch.Tensor]:
        template, search = x
        self.size = search.size(0)
        template_features = self.get_features(template)
        search_features = self.get_features(search)
        return self.connector(template_features=template_features, search_features=search_features)

    def track(self, search: torch.Tensor, template_features: torch.Tensor) ->Dict[str, torch.Tensor]:
        search_features = self.get_features(search)
        return self.connector(template_features=template_features, search_features=search_features)


def calc_iou(reg_target: torch.Tensor, pred: torch.Tensor, smooth: float=1.0) ->torch.Tensor:
    target_area = (reg_target[..., 0] + reg_target[..., 2]) * (reg_target[..., 1] + reg_target[..., 3])
    pred_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])
    w_intersect = torch.min(pred[..., 0], reg_target[..., 0]) + torch.min(pred[..., 2], reg_target[..., 2])
    h_intersect = torch.min(pred[..., 3], reg_target[..., 3]) + torch.min(pred[..., 1], reg_target[..., 1])
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    return (area_intersect + smooth) / (area_union + smooth)


class BoxLoss(nn.Module):
    """
    BBOX Loss: optimizes IoU of bounding boxes
    Original implentation:
    losses = -torch.log(calc_iou(reg_target=target, pred=pred)) was computationally unstable
    those was replaced with: 1 - IoU
    """

    def __init__(self) ->None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]=None) ->torch.Tensor:
        losses = 1 - calc_iou(reg_target=target, pred=pred)
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()


TARGET_REGRESSION_WEIGHT_KEY = 'TARGET_REGRESSION_WEIGHT_KEY'


class FEARLoss(nn.Module):

    def __init__(self, coeffs: Dict[str, float]):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = BoxLoss()
        self.coeffs = coeffs

    def _regression_loss(self, bbox_pred: torch.Tensor, reg_target: torch.Tensor, reg_weight: torch.Tensor) ->torch.Tensor:
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)
        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]
        loss = self.regression_loss(bbox_pred_flatten, reg_target_flatten)
        return loss

    def _weighted_cls_loss(self, pred: torch.Tensor, label: torch.Tensor, select: torch.Tensor) ->torch.Tensor:
        if len(select.size()) == 0:
            return torch.Tensor([0])
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.classification_loss(pred, label)

    def _classification_loss(self, pred: torch.Tensor, label: torch.Tensor) ->torch.Tensor:
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze()
        neg = label.data.eq(0).nonzero().squeeze()
        loss_pos = self._weighted_cls_loss(pred, label, pos)
        loss_neg = self._weighted_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def forward(self, outputs: Dict[str, torch.Tensor], gt: Dict[str, Any]) ->Dict[str, Any]:
        regression_loss = self._regression_loss(bbox_pred=outputs[TARGET_REGRESSION_LABEL_KEY], reg_target=gt[TARGET_REGRESSION_LABEL_KEY], reg_weight=gt[TARGET_REGRESSION_WEIGHT_KEY])
        classification_loss = self._classification_loss(pred=outputs[TARGET_CLASSIFICATION_KEY], label=gt[TARGET_CLASSIFICATION_KEY])
        return {TARGET_CLASSIFICATION_KEY: classification_loss * self.coeffs[TARGET_CLASSIFICATION_KEY], TARGET_REGRESSION_LABEL_KEY: regression_loss * self.coeffs[TARGET_REGRESSION_LABEL_KEY]}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdjustLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BoxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatrixMobile,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_PinataFarms_FEARTracker(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

