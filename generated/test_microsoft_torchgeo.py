import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
conf = _module
evaluate = _module
find_optimal_hyperparams = _module
plot_bar_chart = _module
plot_dataloader_benchmark = _module
plot_percentage_benchmark = _module
run_benchmarks_experiments = _module
run_chesapeake_cvpr_experiments = _module
run_cowc_experiments = _module
run_cowc_seed_experiments = _module
run_landcoverai_experiments = _module
run_landcoverai_seed_experiments = _module
run_resisc45_experiments = _module
run_so2sat_byol_experiments = _module
run_so2sat_experiments = _module
run_so2sat_seed_experiments = _module
test_chesapeakecvpr_models = _module
tests = _module
data = _module
split = _module
datamodules = _module
test_chesapeake = _module
test_fair1m = _module
test_inria = _module
test_loveda = _module
test_nasa_marine_debris = _module
test_oscd = _module
test_potsdam = _module
test_usavars = _module
test_utils = _module
test_vaihingen = _module
test_xview2 = _module
datasets = _module
test_advance = _module
test_agb_live_woody_density = _module
test_astergdem = _module
test_benin_cashews = _module
test_bigearthnet = _module
test_cbf = _module
test_cdl = _module
test_chesapeake = _module
test_cloud_cover = _module
test_cms_mangrove_canopy = _module
test_cowc = _module
test_cv4a_kenya_crop_type = _module
test_cyclone = _module
test_deepglobelandcover = _module
test_dfc2022 = _module
test_eddmaps = _module
test_enviroatlas = _module
test_esri2020 = _module
test_etci2021 = _module
test_eudem = _module
test_eurosat = _module
test_fair1m = _module
test_forestdamage = _module
test_gbif = _module
test_geo = _module
test_gid15 = _module
test_globbiomass = _module
test_idtrees = _module
test_inaturalist = _module
test_inria = _module
test_landcoverai = _module
test_landsat = _module
test_levircd = _module
test_loveda = _module
test_millionaid = _module
test_naip = _module
test_nasa_marine_debris = _module
test_nwpu = _module
test_openbuildings = _module
test_oscd = _module
test_patternnet = _module
test_potsdam = _module
test_reforestree = _module
test_resisc45 = _module
test_seco = _module
test_sen12ms = _module
test_sentinel = _module
test_so2sat = _module
test_spacenet = _module
test_ucmerced = _module
test_usavars = _module
test_utils = _module
test_vaihingen = _module
test_xview2 = _module
test_zuericrop = _module
losses = _module
test_qr = _module
models = _module
test_changestar = _module
test_farseg = _module
test_fcn = _module
test_fcsiam = _module
test_rcf = _module
test_resnet = _module
samplers = _module
test_batch = _module
test_single = _module
test_train = _module
trainers = _module
conftest = _module
test_byol = _module
test_classification = _module
test_detection = _module
test_regression = _module
test_segmentation = _module
test_utils = _module
transforms = _module
test_indices = _module
test_transforms = _module
torchgeo = _module
bigearthnet = _module
chesapeake = _module
cowc = _module
cyclone = _module
deepglobelandcover = _module
etci2021 = _module
eurosat = _module
fair1m = _module
inria = _module
landcoverai = _module
loveda = _module
naip = _module
nasa_marine_debris = _module
oscd = _module
potsdam = _module
resisc45 = _module
sen12ms = _module
so2sat = _module
spacenet = _module
ucmerced = _module
usavars = _module
utils = _module
vaihingen = _module
xview = _module
advance = _module
agb_live_woody_density = _module
astergdem = _module
benin_cashews = _module
bigearthnet = _module
cbf = _module
cdl = _module
chesapeake = _module
cloud_cover = _module
cms_mangrove_canopy = _module
cowc = _module
cv4a_kenya_crop_type = _module
cyclone = _module
deepglobelandcover = _module
dfc2022 = _module
eddmaps = _module
enviroatlas = _module
esri2020 = _module
etci2021 = _module
eudem = _module
eurosat = _module
fair1m = _module
forestdamage = _module
gbif = _module
geo = _module
gid15 = _module
globbiomass = _module
idtrees = _module
inaturalist = _module
inria = _module
landcoverai = _module
landsat = _module
levircd = _module
loveda = _module
millionaid = _module
nasa_marine_debris = _module
nwpu = _module
openbuildings = _module
oscd = _module
patternnet = _module
potsdam = _module
reforestree = _module
resisc45 = _module
seco = _module
sen12ms = _module
sentinel = _module
so2sat = _module
spacenet = _module
ucmerced = _module
usavars = _module
utils = _module
vaihingen = _module
xview = _module
zuericrop = _module
qr = _module
changestar = _module
farseg = _module
fcn = _module
fcsiam = _module
rcf = _module
resnet = _module
batch = _module
constants = _module
single = _module
utils = _module
byol = _module
classification = _module
detection = _module
regression = _module
segmentation = _module
utils = _module
indices = _module
transforms = _module
train = _module

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


import time


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision.models import resnet34


from typing import Any


from typing import Dict


from typing import Union


from typing import cast


from torch.utils.data import TensorDataset


import matplotlib.pyplot as plt


from torch.utils.data import ConcatDataset


from typing import List


from matplotlib import pyplot as plt


import math


import re


from typing import Tuple


import numpy as np


import itertools


from torch.nn.modules import Module


from typing import Optional


from itertools import product


from typing import Iterator


from collections import OrderedDict


import torchvision


from torch import Tensor


from typing import Type


from torchvision.models import resnet18


from torchvision.transforms import Compose


from typing import Callable


import torch.nn.functional as F


from torch import Generator


from torch.utils.data import random_split


from sklearn.model_selection import GroupShuffleSplit


from torch.utils.data import Subset


from torch.utils.data import Dataset


from torchvision.transforms import Normalize


import torchvision.transforms as T


from torch.utils.data._utils.collate import default_collate


from functools import lru_cache


import abc


from typing import Sequence


from matplotlib.colors import ListedColormap


from matplotlib.figure import Figure


from matplotlib import colors


import matplotlib.patches as patches


import functools


import warnings


from torchvision.datasets import ImageFolder


from torchvision.datasets.folder import default_loader as pil_loader


from typing import overload


from torchvision.ops import clip_boxes_to_image


from torchvision.ops import remove_small_boxes


from torchvision.utils import draw_bounding_boxes


from collections import defaultdict


import copy


import collections


from typing import Iterable


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import download_url


from torchvision.utils import draw_segmentation_masks


from torch.nn.modules import BatchNorm2d


from torch.nn.modules import Conv2d


from torch.nn.modules import Identity


from torch.nn.modules import ModuleList


from torch.nn.modules import ReLU


from torch.nn.modules import Sequential


from torch.nn.modules import Sigmoid


from torch.nn.modules import UpsamplingBilinear2d


from torchvision.models import resnet


from torchvision.ops import FeaturePyramidNetwork as FPN


from torch.hub import load_state_dict_from_url


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import ResNet


from torch.utils.data import Sampler


import random


from torch import optim


from torch.nn.modules import BatchNorm1d


from torch.nn.modules import Linear


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision.models.detection import FasterRCNN


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


from torchvision.models.detection.rpn import AnchorGenerator


from torchvision.ops import MultiScaleRoIAlign


class ClassificationTestModel(Module):

    def __init__(self, in_chans: int=3, num_classes: int=1000, **kwargs: Any) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RegressionTestModel(ClassificationTestModel):

    def __init__(self, **kwargs: Any) ->None:
        super().__init__(in_chans=3, num_classes=1)


class SegmentationTestModel(Module):

    def __init__(self, in_channels: int=3, classes: int=1000, **kwargs: Any) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def forward(self, probs: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Computes the QR (forwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W.
            target: prior probabilities, expected shape B x C x H x W.

        Returns:
            qr loss
        """
        q = probs
        q_bar = q.mean(dim=(0, 2, 3))
        qbar_log_S = (q_bar * torch.log(q_bar)).sum()
        q_log_p = torch.einsum('bcxy,bcxy->bxy', q, torch.log(target)).mean()
        loss = qbar_log_S - q_log_p
        return loss


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def forward(self, probs: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Computes the RQ (backwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W
            target: prior probabilities, expected shape B x C x H x W

        Returns:
            qr loss
        """
        q = probs
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(1e-12).expand_as(q)
        r = F.normalize(z * target, p=1, dim=1)
        loss = torch.einsum('bcxy,bcxy->bxy', r, torch.log(r) - torch.log(q)).mean()
        return loss


class ChangeMixin(Module):
    """This module enables any segmentation model to detect binary change.

    The common usage is to attach this module on a segmentation model without the
    classification head.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2108.07002
    """

    def __init__(self, in_channels: int=128 * 2, inner_channels: int=16, num_convs: int=4, scale_factor: float=4.0):
        """Initializes a new ChangeMixin module.

        Args:
            in_channels: sum of channels of bitemporal feature maps
            inner_channels: number of channels of inner feature maps
            num_convs: number of convolution blocks
            scale_factor: number of upsampling factor
        """
        super().__init__()
        layers: List[Module] = [nn.modules.Sequential(nn.modules.Conv2d(in_channels, inner_channels, 3, 1, 1), nn.modules.BatchNorm2d(inner_channels), nn.modules.ReLU(True))]
        layers += [nn.modules.Sequential(nn.modules.Conv2d(inner_channels, inner_channels, 3, 1, 1), nn.modules.BatchNorm2d(inner_channels), nn.modules.ReLU(True)) for _ in range(num_convs - 1)]
        cls_layer = nn.modules.Conv2d(inner_channels, 1, 3, 1, 1)
        layers.append(cls_layer)
        layers.append(nn.modules.UpsamplingBilinear2d(scale_factor=scale_factor))
        self.convs = nn.modules.Sequential(*layers)

    def forward(self, bi_feature: Tensor) ->List[Tensor]:
        """Forward pass of the model.

        Args:
            bi_feature: input bitemporal feature maps of shape [b, t, c, h, w]

        Returns:
            a list of bidirected output predictions
        """
        batch_size = bi_feature.size(0)
        t1t2 = torch.cat([bi_feature[:, 0, :, :, :], bi_feature[:, 1, :, :, :]], dim=1)
        t2t1 = torch.cat([bi_feature[:, 1, :, :, :], bi_feature[:, 0, :, :, :]], dim=1)
        c1221 = self.convs(torch.cat([t1t2, t2t1], dim=0))
        c12, c21 = torch.split(c1221, batch_size, dim=0)
        return [c12, c21]


class ChangeStar(Module):
    """The base class of the network architecture of ChangeStar.

    ChangeStar is composed of an any segmentation model and a ChangeMixin module.
    This model is mainly used for binary/multi-class change detection under bitemporal
    supervision and single-temporal supervision. It features the property of
    segmentation architecture reusing, which is helpful to integrate advanced dense
    prediction (e.g., semantic segmentation) network architecture into change detection.

    For multi-class change detection, semantic change prediction can be inferred by a
    binary change prediction from the ChangeMixin module and two semantic predictions
    from the Segmentation model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2108.07002
    """

    def __init__(self, dense_feature_extractor: Module, seg_classifier: Module, changemixin: ChangeMixin, inference_mode: str='t1t2') ->None:
        """Initializes a new ChangeStar model.

        Args:
            dense_feature_extractor: module for dense feature extraction, typically a
                semantic segmentation model without semantic segmentation head.
            seg_classifier: semantic segmentation head, typically a convolutional layer
                followed by an upsampling layer.
            changemixin: :class:`torchgeo.models.ChangeMixin` module
            inference_mode: name of inference mode ``'t1t2'`` | ``'t2t1'`` | ``'mean'``.
                ``'t1t2'``: concatenate bitemporal features in the order of t1->t2;
                ``'t2t1'``: concatenate bitemporal features in the order of t2->t1;
                ``'mean'``: the weighted mean of the output of ``'t1t2'`` and ``'t1t2'``
        """
        super().__init__()
        self.dense_feature_extractor = dense_feature_extractor
        self.seg_classifier = seg_classifier
        self.changemixin = changemixin
        if inference_mode not in ['t1t2', 't2t1', 'mean']:
            raise ValueError(f'Unknown inference_mode: {inference_mode}')
        self.inference_mode = inference_mode

    def forward(self, x: Tensor) ->Dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            x: a bitemporal input tensor of shape [B, T, C, H, W]

        Returns:
            a dictionary containing bitemporal semantic segmentation logit and binary
            change detection logit/probability
        """
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        bi_feature = self.dense_feature_extractor(x)
        bi_seg_logit = self.seg_classifier(bi_feature)
        bi_seg_logit = rearrange(bi_seg_logit, '(b t) c h w -> b t c h w', t=t)
        bi_feature = rearrange(bi_feature, '(b t) c h w -> b t c h w', t=t)
        c12, c21 = self.changemixin(bi_feature)
        results: Dict[str, Tensor] = {}
        if not self.training:
            results.update({'bi_seg_logit': bi_seg_logit})
            if self.inference_mode == 't1t2':
                results.update({'change_prob': c12.sigmoid()})
            elif self.inference_mode == 't2t1':
                results.update({'change_prob': c21.sigmoid()})
            elif self.inference_mode == 'mean':
                results.update({'change_prob': torch.stack([c12, c21], dim=0).sigmoid_().mean(dim=0)})
        else:
            results.update({'bi_seg_logit': bi_seg_logit, 'bi_change_logit': torch.stack([c12, c21], dim=1)})
        return results


class _FSRelation(Module):
    """F-S Relation module."""

    def __init__(self, scene_embedding_channels: int, in_channels_list: List[int], out_channels: int) ->None:
        """Initialize the _FSRelation module.

        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
        """
        super().__init__()
        self.scene_encoder = ModuleList([Sequential(Conv2d(scene_embedding_channels, out_channels, 1), ReLU(True), Conv2d(out_channels, out_channels, 1)) for _ in range(len(in_channels_list))])
        self.content_encoders = ModuleList()
        self.feature_reencoders = ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(Sequential(Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)))
            self.feature_reencoders.append(Sequential(Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)))
        self.normalizer = Sigmoid()

    def forward(self, scene_feature: Tensor, features: List[Tensor]) ->List[Tensor]:
        """Forward pass of the model."""
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in zip(scene_feats, content_feats)]
        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]
        refined_feats = [(r * p) for r, p in zip(relations, p_feats)]
        return refined_feats


class _LightWeightDecoder(Module):
    """Light Weight Decoder."""

    def __init__(self, in_channels: int, out_channels: int, num_classes: int, in_feature_output_strides: List[int]=[4, 8, 16, 32], out_feature_output_stride: int=4) ->None:
        """Initialize the _LightWeightDecoder module.

        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different
                levels
            out_feature_output_stride: output stride of output feature maps
        """
        super().__init__()
        self.blocks = ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feature_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(Sequential(*[Sequential(Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False), BatchNorm2d(out_channels), ReLU(inplace=True), UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else Identity()) for idx in range(num_layers)]))
        self.classifier = Sequential(Conv2d(out_channels, num_classes, 3, 1, 1), UpsamplingBilinear2d(scale_factor=4))

    def forward(self, features: List[Tensor]) ->Tensor:
        """Forward pass of the model."""
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)
        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out_feat = self.classifier(out_feat)
        return cast(Tensor, out_feat)


class FarSeg(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature representation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    """

    def __init__(self, backbone: str='resnet50', classes: int=16, backbone_pretrained: bool=True) ->None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        super().__init__()
        if backbone in ['resnet18', 'resnet34']:
            max_channels = 512
        elif backbone in ['resnet50', 'resnet101']:
            max_channels = 2048
        else:
            raise ValueError(f'unknown backbone: {backbone}.')
        kwargs = {}
        if parse(torchvision.__version__) >= parse('0.13'):
            if backbone_pretrained:
                kwargs = {'weights': getattr(torchvision.models, f'ResNet{backbone[6:]}_Weights').DEFAULT}
            else:
                kwargs = {'weights': None}
        else:
            kwargs = {'pretrained': backbone_pretrained}
        self.backbone = getattr(resnet, backbone)(**kwargs)
        self.fpn = FPN(in_channels_list=[(max_channels // 2 ** (3 - i)) for i in range(4)], out_channels=256)
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)

    def forward(self, x: Tensor) ->Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]
        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(OrderedDict({f'c{i + 2}': features[i] for i in range(4)}))
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)
        logit = self.decoder(features)
        return cast(Tensor, logit)


class ChangeStarFarSeg(ChangeStar):
    """The network architecture of ChangeStar(FarSeg).

    ChangeStar(FarSeg) is composed of a FarSeg model and a ChangeMixin module.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2108.07002
    """

    def __init__(self, backbone: str='resnet50', classes: int=1, backbone_pretrained: bool=True) ->None:
        """Initializes a new ChangeStarFarSeg model.

        Args:
            backbone: name of ResNet backbone
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        model = FarSeg(backbone=backbone, classes=classes, backbone_pretrained=backbone_pretrained)
        seg_classifier: Module = model.decoder.classifier
        model.decoder.classifier = nn.modules.Identity()
        super().__init__(dense_feature_extractor=model, seg_classifier=seg_classifier, changemixin=ChangeMixin(in_channels=128 * 2, inner_channels=16, num_convs=4, scale_factor=4.0), inference_mode='t1t2')


class FCN(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int=64) ->None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super().__init__()
        conv1 = nn.modules.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        conv2 = nn.modules.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv3 = nn.modules.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv4 = nn.modules.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv5 = nn.modules.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.backbone = nn.modules.Sequential(conv1, nn.modules.LeakyReLU(inplace=True), conv2, nn.modules.LeakyReLU(inplace=True), conv3, nn.modules.LeakyReLU(inplace=True), conv4, nn.modules.LeakyReLU(inplace=True), conv5, nn.modules.LeakyReLU(inplace=True))
        self.last = nn.modules.Conv2d(num_filters, classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) ->Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x


class RCF(Module):
    """This model extracts random convolutional features (RCFs) from its input.

    RCFs are used in Multi-task Observation using Satellite Imagery & Kitchen Sinks
    (MOSAIKS) method proposed in https://www.nature.com/articles/s41467-021-24638-z.

    .. note::

        This Module is *not* trainable. It is only used as a feature extractor.
    """
    weights: Tensor
    biases: Tensor

    def __init__(self, in_channels: int=4, features: int=16, kernel_size: int=3, bias: float=-1.0, seed: Optional[int]=None) ->None:
        """Initializes the RCF model.

        This is a static model that serves to extract fixed length feature vectors from
        input patches.

        Args:
            in_channels: number of input channels
            features: number of features to compute, must be divisible by 2
            kernel_size: size of the kernel used to compute the RCFs
            bias: bias of the convolutional layer
            seed: random seed used to initialize the convolutional layer

        .. versionadded:: 0.2
           The *seed* parameter.
        """
        super().__init__()
        assert features % 2 == 0
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        self.register_buffer('weights', torch.randn(features // 2, in_channels, kernel_size, kernel_size, requires_grad=False, generator=generator))
        self.register_buffer('biases', torch.zeros(features // 2, requires_grad=False) + bias)

    def forward(self, x: Tensor) ->Tensor:
        """Forward pass of the RCF model.

        Args:
            x: a tensor with shape (B, C, H, W)

        Returns:
            a tensor of size (B, ``self.num_features``)
        """
        x1a = F.relu(F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0), inplace=True)
        x1b = F.relu(-F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0), inplace=False)
        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()
        if len(x1a.shape) == 1:
            output = torch.cat((x1a, x1b), dim=0)
            return output
        else:
            assert len(x1a.shape) == 2
            output = torch.cat((x1a, x1b), dim=1)
            return output


class RandomApply(Module):
    """Applies augmentation function (augm) with probability p."""

    def __init__(self, augm: Callable[[Tensor], Tensor], p: float) ->None:
        """Initialize RandomApply.

        Args:
            augm: augmentation function to apply
            p: probability with which the augmentation function is applied
        """
        super().__init__()
        self.augm = augm
        self.p = p

    def forward(self, x: Tensor) ->Tensor:
        """Applies an augmentation to the input with some probability.

        Args:
            x: a batch of imagery

        Returns
            augmented version of ``x`` with probability ``self.p`` else an un-augmented
                version
        """
        return x if random.random() > self.p else self.augm(x)


class SimCLRAugmentation(Module):
    """A module for applying SimCLR augmentations.

    SimCLR was one of the first papers to show the effectiveness of random data
    augmentation in self-supervised-learning setups. See
    https://arxiv.org/pdf/2002.05709.pdf for more details.
    """

    def __init__(self, image_size: Tuple[int, int]=(256, 256)) ->None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
        """
        super().__init__()
        self.size = image_size
        self.augmentation = Sequential(KorniaTransform.Resize(size=image_size, align_corners=False), K.RandomHorizontalFlip(), RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1), K.RandomResizedCrop(size=image_size))

    def forward(self, x: Tensor) ->Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        return cast(Tensor, self.augmentation(x))


class MLP(Module):
    """MLP used in the BYOL projection head."""

    def __init__(self, dim: int, projection_size: int=256, hidden_size: int=4096) ->None:
        """Initializes the MLP projection head.

        Args:
            dim: size of layer to project
            projection_size: size of the output layer
            hidden_size: size of the hidden layer
        """
        super().__init__()
        self.mlp = Sequential(Linear(dim, hidden_size), BatchNorm1d(hidden_size), ReLU(inplace=True), Linear(hidden_size, projection_size))

    def forward(self, x: Tensor) ->Tensor:
        """Forward pass of the MLP model.

        Args:
            x: batch of imagery

        Returns:
            embedded version of the input
        """
        return cast(Tensor, self.mlp(x))


class BackboneWrapper(Module):
    """Backbone wrapper for joining a model and a projection head.

    When we call .forward() on this module the following steps happen:

    * The input is passed through the base model
    * When the encoding layer is reached a hook is called
    * The output of the encoding layer is passed through the projection head
    * The forward call returns the output of the projection head

    .. versionchanged 0.4: Name changed from *EncoderWrapper* to
        *BackboneWrapper*.
    """

    def __init__(self, model: Module, projection_size: int=256, hidden_size: int=4096, layer: int=-2) ->None:
        """Initializes BackboneWrapper.

        Args:
            model: model to encode
            projection_size: size of the ouput layer of the projector MLP
            hidden_size: size of hidden layer of the projector MLP
            layer: layer from model to project
        """
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer
        self._projector: Optional[Module] = None
        self._projector_dim: Optional[int] = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self) ->Module:
        """Wrapper module for the projector head."""
        assert self._projector_dim is not None
        if self._projector is None:
            self._projector = MLP(self._projector_dim, self.projection_size, self.hidden_size)
        return self._projector

    def _hook(self, module: Any, input: Any, output: Tensor) ->None:
        """Hook to record the activations at the projection layer.

        See the following docs page for more details on hooks:
        https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html

        Args:
            module: the calling module
            input: input to the module this hook was registered to
            output: output from the module this hook was registered to
        """
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)
        self._embedding = output

    def _register_hook(self) ->None:
        """Register a hook for layer that we will extract features from."""
        layer = list(self.model.children())[self.layer]
        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) ->Tensor:
        """Pass through the model, and collect the representation from our forward hook.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        _ = self.model(x)
        return self._encoded


class BYOL(Module):
    """BYOL implementation.

    BYOL contains two identical backbone networks. The first is trained as usual, and
    its weights are updated with each training batch. The second, "target" network,
    is updated using a running average of the first backbone's weights.

    See https://arxiv.org/abs/2006.07733 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(self, model: Module, image_size: Tuple[int, int]=(256, 256), hidden_layer: int=-2, in_channels: int=4, projection_size: int=256, hidden_size: int=4096, augment_fn: Optional[Module]=None, beta: float=0.99, **kwargs: Any) ->None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            hidden_layer: the hidden layer in ``model`` to attach the projection
                head to, can be the name of the layer or index of the layer
            in_channels: number of input channels to the model
            projection_size: size of first layer of the projection MLP
            hidden_size: size of the hidden layer of the projection MLP
            augment_fn: an instance of a module that performs data augmentation
            beta: the speed at which the target backbone is updated using the main
                backbone
        """
        super().__init__()
        self.augment: Module
        if augment_fn is None:
            self.augment = SimCLRAugmentation(image_size)
        else:
            self.augment = augment_fn
        self.beta = beta
        self.in_channels = in_channels
        self.backbone = BackboneWrapper(model, projection_size, hidden_size, layer=hidden_layer)
        self.predictor = MLP(projection_size, projection_size, hidden_size)
        self.target = BackboneWrapper(model, projection_size, hidden_size, layer=hidden_layer)
        self.backbone(torch.zeros(2, self.in_channels, *image_size))

    def forward(self, x: Tensor) ->Tensor:
        """Forward pass of the backbone model through the MLP and prediction head.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.predictor(self.backbone(x)))

    def update_target(self) ->None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.backbone.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


_EPSILON = 1e-10


class AppendNormalizedDifferenceIndex(Module):
    """Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \\text{NDI} = \\frac{A - B}{A + B}

    .. versionadded:: 0.2
    """

    def __init__(self, index_a: int, index_b: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_index(self, band_a: Tensor, band_b: Tensor) ->Tensor:
        """Compute normalized difference index.

        Args:
            band_a: reference band tensor
            band_b: difference band tensor

        Returns:
            the index
        """
        return (band_a - band_b) / (band_a + band_b + _EPSILON)

    def forward(self, sample: Dict[str, Tensor]) ->Dict[str, Tensor]:
        """Compute and append normalized difference index to image.

        Args:
            sample: a sample or batch dict

        Returns:
            the transformed sample
        """
        if 'image' in sample:
            index = self._compute_index(band_a=sample['image'][..., self.index_a, :, :], band_b=sample['image'][..., self.index_b, :, :])
            index = index.unsqueeze(self.dim)
            sample['image'] = torch.cat([sample['image'], index], dim=self.dim)
        return sample


class AppendNBR(AppendNormalizedDifferenceIndex):
    """Normalized Burn Ratio (NBR).

    Computes the following index:

    .. math::

       \\text{NBR} = \\frac{\\text{NIR} - \\text{SWIR}}{\\text{NIR} + \\text{SWIR}}

    If you use this index in your research, please cite the following paper:

    * https://www.sciencebase.gov/catalog/item/4f4e4b20e4b07f02db6abb36

    .. versionadded:: 0.2
    """

    def __init__(self, index_nir: int, index_swir: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_swir)


class AppendNDBI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Built-up Index (NDBI).

    Computes the following index:

    .. math::

       \\text{NDBI} = \\frac{\\text{SWIR} - \\text{NIR}}{\\text{SWIR} + \\text{NIR}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160304987
    """

    def __init__(self, index_swir: int, index_nir: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_swir, index_b=index_nir)


class AppendNDSI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Snow Index (NDSI).

    Computes the following index:

    .. math::

       \\text{NDSI} = \\frac{\\text{G} - \\text{SWIR}}{\\text{G} + \\text{SWIR}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.1994.399618
    """

    def __init__(self, index_green: int, index_swir: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__(index_a=index_green, index_b=index_swir)


class AppendNDVI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Vegetation Index (NDVI).

    Computes the following index:

    .. math::

       \\text{NDVI} = \\frac{\\text{NIR} - \\text{R}}{\\text{NIR} + \\text{R}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_nir: int, index_red: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_red: index of the Red band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_red)


class AppendNDWI(AppendNormalizedDifferenceIndex):
    """Normalized Difference Water Index (NDWI).

    Computes the following index:

    .. math::

       \\text{NDWI} = \\frac{\\text{G} - \\text{NIR}}{\\text{G} + \\text{NIR}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431169608948714
    """

    def __init__(self, index_green: int, index_nir: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_green, index_b=index_nir)


class AppendSWI(AppendNormalizedDifferenceIndex):
    """Standardized Water-Level Index (SWI).

    Computes the following index:

    .. math::

       \\text{SWI} = \\frac{\\text{VRE1} - \\text{SWIR2}}{\\text{VRE1} + \\text{SWIR2}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.3390/w13121647

    .. versionadded:: 0.3
    """

    def __init__(self, index_vre1: int, index_swir2: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_vre1: index of the VRE1 band, e.g. B5 in Sentinel 2 imagery
            index_swir2: index of the SWIR2 band, e.g. B11 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_vre1, index_b=index_swir2)


class AppendGNDVI(AppendNormalizedDifferenceIndex):
    """Green Normalized Difference Vegetation Index (GNDVI).

    Computes the following index:

    .. math::

       \\text{GNDVI} = \\frac{\\text{NIR} - \\text{G}}{\\text{NIR} + \\text{G}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.2134/agronj2001.933583x

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, e.g. B3 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green)


class AppendBNDVI(AppendNormalizedDifferenceIndex):
    """Blue Normalized Difference Vegetation Index (BNDVI).

    Computes the following index:

    .. math::

       \\text{BNDVI} = \\frac{\\text{NIR} - \\text{B}}{\\text{NIR} + \\text{B}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_blue: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_blue: index of the Blue band, e.g. B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_blue)


class AppendNDRE(AppendNormalizedDifferenceIndex):
    """Normalized Difference Red Edge Vegetation Index (NDRE).

    Computes the following index:

    .. math::

       \\text{NDRE} = \\frac{\\text{NIR} - \\text{VRE1}}{\\text{NIR} + \\text{VRE1}}

    If you use this index in your research, please cite the following paper:

    * https://agris.fao.org/agris-search/search.do?recordID=US201300795763

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_vre1: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_vre1: index of the Red Edge band, B5 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_vre1)


class AppendTriBandNormalizedDifferenceIndex(Module):
    """Append normalized difference index involving 3 bands as channel to image tensor.

    Computes the following index:

    .. math::

       \\text{NDI} = \\frac{A - (B + C)}{A + (B + C)}

    .. versionadded:: 0.3
    """

    def __init__(self, index_a: int, index_b: int, index_c: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index of component 1
            index_c: difference band channel index of component 2
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b
        self.index_c = index_c

    def _compute_index(self, band_a: Tensor, band_b: Tensor, band_c: Tensor) ->Tensor:
        """Compute tri-band normalized difference index.

        Args:
            band_a: reference band tensor
            band_b: difference band tensor component 1
            band_c: difference band tensor component 2

        Returns:
            the index
        """
        return (band_a - (band_b + band_c)) / (band_a + band_b + band_c + _EPSILON)

    def forward(self, sample: Dict[str, Tensor]) ->Dict[str, Tensor]:
        """Compute and append tri-band normalized difference index to image.

        Args:
            sample: a sample or batch dict

        Returns:
            the transformed sample
        """
        if 'image' in sample:
            index = self._compute_index(band_a=sample['image'][..., self.index_a, :, :], band_b=sample['image'][..., self.index_b, :, :], band_c=sample['image'][..., self.index_c, :, :])
            index = index.unsqueeze(self.dim)
            sample['image'] = torch.cat([sample['image'], index], dim=self.dim)
        return sample


class AppendGRNDVI(AppendTriBandNormalizedDifferenceIndex):
    """Green-Red Normalized Difference Vegetation Index (GRNDVI).

    Computes the following index:

    .. math::

       \\text{GRNDVI} =
           \\frac{\\text{NIR} - (\\text{G} + \\text{R})}{\\text{NIR} + (\\text{G} + \\text{R})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int, index_red: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, B3 in Sentinel 2 imagery
            index_red: index of the Red band, B4 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_red)


class AppendGBNDVI(AppendTriBandNormalizedDifferenceIndex):
    """Green-Blue Normalized Difference Vegetation Index (GBNDVI).

    Computes the following index:

    .. math::

       \\text{GBNDVI} =
           \\frac{\\text{NIR} - (\\text{G} + \\text{B})}{\\text{NIR} + (\\text{G} + \\text{B})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_green: int, index_blue: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_green: index of the Green band, B3 in Sentinel 2 imagery
            index_blue: index of the Blue band, B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_blue)


class AppendRBNDVI(AppendTriBandNormalizedDifferenceIndex):
    """Red-Blue Normalized Difference Vegetation Index (RBNDVI).

    Computes the following index:

    .. math::

       \\text{RBNDVI} =
           \\frac{\\text{NIR} - (\\text{R} + \\text{B})}{\\text{NIR} + (\\text{R} + \\text{B})}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/S1672-6308(07)60027-4

    .. versionadded:: 0.3
    """

    def __init__(self, index_nir: int, index_red: int, index_blue: int) ->None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the NIR band, e.g. B8 in Sentinel 2 imagery
            index_red: index of the Red band, B4 in Sentinel 2 imagery
            index_blue: index of the Blue band, B2 in Sentinel 2 imagery
        """
        super().__init__(index_a=index_nir, index_b=index_red, index_c=index_blue)


class AugmentationSequential(Module):
    """Wrapper around kornia AugmentationSequential to handle input dicts."""

    def __init__(self, *args: Module, data_keys: List[str]) ->None:
        """Initialize a new augmentation sequential instance.

        Args:
            *args: Sequence of kornia augmentations
            data_keys: List of inputs to augment (e.g. ["image", "mask", "boxes"])
        """
        super().__init__()
        self.data_keys = data_keys
        keys = []
        for key in data_keys:
            if key == 'image':
                keys.append('input')
            elif key == 'boxes':
                keys.append('bbox')
            else:
                keys.append(key)
        self.augs = K.AugmentationSequential(*args, data_keys=keys)

    def forward(self, sample: Dict[str, Tensor]) ->Dict[str, Tensor]:
        """Perform augmentations and update data dict.

        Args:
            sample: the input

        Returns:
            the augmented input
        """
        if 'mask' in self.data_keys:
            mask_dtype = sample['mask'].dtype
            sample['mask'] = sample['mask']
        if 'boxes' in self.data_keys:
            boxes_dtype = sample['boxes'].dtype
            sample['boxes'] = sample['boxes']
        inputs = [sample[k] for k in self.data_keys]
        outputs_list: Union[Tensor, List[Tensor]] = self.augs(*inputs)
        outputs_list = outputs_list if isinstance(outputs_list, list) else [outputs_list]
        outputs: Dict[str, Tensor] = {k: v for k, v in zip(self.data_keys, outputs_list)}
        sample.update(outputs)
        if 'mask' in self.data_keys:
            sample['mask'] = sample['mask']
        if 'boxes' in self.data_keys:
            sample['boxes'] = sample['boxes']
        return sample


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassificationTestModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FCN,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (QRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RCF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RQLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RandomApply,
     lambda: ([], {'augm': _mock_layer(), 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RegressionTestModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SegmentationTestModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_microsoft_torchgeo(_paritybench_base):
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

