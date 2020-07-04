import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
cifar_simple = _module
experiment = _module
model = _module
main = _module
recognizer = _module
dataset = _module
executors = _module
infer = _module
valid = _module
model = _module
executor = _module
mlcomp = _module
__main__ = _module
__version__ = _module
contrib = _module
catalyst = _module
callbacks = _module
inference = _module
optim = _module
cosineanneal = _module
register = _module
complex = _module
criterion = _module
ce = _module
ring = _module
triplet = _module
classify = _module
segment = _module
video = _module
metrics = _module
dice = _module
efficientnet = _module
pretrained = _module
segmentation_model_pytorch = _module
timm = _module
resnext3d = _module
r2plus1_util = _module
resnext3d = _module
resnext3d_block = _module
resnext3d_stage = _module
resnext3d_stem = _module
sampler = _module
balanced = _module
distributed = _module
hard_negative = _module
scripts = _module
split = _module
search = _module
grid = _module
segmentation = _module
base = _module
encoder_decoder = _module
model = _module
common = _module
blocks = _module
deeplabv3 = _module
aspp = _module
backbone = _module
drn = _module
mobilenet = _module
resnet = _module
xception = _module
decoder = _module
deeplab = _module
encoders = _module
_preprocessing = _module
densenet = _module
dpn = _module
inceptionresnetv2 = _module
senet = _module
vgg = _module
fpn = _module
decoder = _module
model = _module
linknet = _module
decoder = _module
model = _module
pspnet = _module
decoder = _module
model = _module
unet = _module
decoder = _module
model = _module
frame = _module
torch = _module
layers = _module
tensors = _module
transform = _module
albumentations = _module
rle = _module
tta = _module
db = _module
core = _module
options = _module
enums = _module
models = _module
auxilary = _module
computer = _module
dag = _module
dag_storage = _module
docker = _module
file = _module
log = _module
memory = _module
project = _module
report = _module
space = _module
step = _module
task = _module
providers = _module
auxiliary = _module
img = _module
layout = _module
series = _module
task_synced = _module
report_info = _module
f1 = _module
img_classify = _module
img_segment = _module
info = _module
item = _module
metric = _module
precision_recall = _module
signals = _module
tests = _module
test_project = _module
migration = _module
manage = _module
server = _module
back = _module
app = _module
create_dags = _module
copy = _module
model_add = _module
model_start = _module
pipe = _module
standard = _module
supervisor = _module
utils = _module
config = _module
describe = _module
io = _module
logging = _module
misc = _module
plot = _module
req = _module
schedule = _module
worker = _module
bash = _module
catalyst_ = _module
click = _module
kaggle = _module
model = _module
reports = _module
classification = _module
segmenation = _module
storage = _module
sync = _module
tasks = _module
setup = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


from torch.nn.functional import nll_loss


from torch.nn.functional import log_softmax


import torch


from torch.nn.parameter import Parameter


from torch.nn.modules.loss import CrossEntropyLoss


from torch import nn


import logging


from collections import OrderedDict


from typing import Tuple


import numpy as np


from torch.nn.functional import cross_entropy


from torch.utils.data import Sampler


import math


import torch.utils.model_zoo as model_zoo


import re


from torchvision.models.densenet import DenseNet


from torchvision.models.vgg import VGG


from torchvision.models.vgg import make_layers


from torch.jit import ScriptModule


class LabelSmoothingCrossEntropy(CrossEntropyLoss):

    def __init__(self, eps: float=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        nl = nll_loss(log_preds, target, reduction=self.reduction)
        return loss * self.eps / c + (1 - self.eps) * nl


class RingLoss(nn.Module):

    def __init__(self, type='auto', loss_weight=1.0, softmax_loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01.
            For 'auto', try with 1.0.

        Source: https://github.com/Paralysis/ringloss
        """
        super().__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type
        self.softmax = CrossEntropyLoss()
        self.softmax_loss_weight = softmax_loss_weight

    def forward(self, x, y):
        softmax = self.softmax(x, y).mul_(self.softmax_loss_weight)
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0:
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1':
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self
                .loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self
                .loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto':
            diff = x.sub(self.radius.expand_as(x)) / x.mean().detach().clamp(
                min=0.5)
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else:
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return softmax + ringloss


class EfficientNet(nn.Module):

    def __init__(self, variant, num_classes, pretrained=True, activation=None):
        super().__init__()
        if 'efficientnet' not in variant:
            variant = f'efficientnet-{variant}'
        if pretrained:
            model = _EfficientNet.from_pretrained(variant, num_classes=
                num_classes)
        else:
            model = _EfficientNet.from_name(variant, {'num_classes':
                num_classes})
        self.model = model
        self.model._fc = nn.Sequential(LambdaLayer(lambda x: x.unsqueeze_(0
            )), nn.AdaptiveAvgPool1d(self.model._fc.in_features),
            LambdaLayer(lambda x: x.squeeze_(0).view(x.size(0), -1)), self.
            model._fc)
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        res = self.model(x)
        if isinstance(res, tuple):
            res = res[0]
        if self.activation:
            res = self.activation(res)
        return res


class Pretrained(nn.Module):

    def __init__(self, variant, num_classes, pretrained=True, activation=None):
        super().__init__()
        params = {'num_classes': 1000}
        if not pretrained:
            params['pretrained'] = None
        model = pretrainedmodels.__dict__[variant](**params)
        self.model = model
        linear = self.model.last_linear
        if isinstance(linear, nn.Linear):
            self.model.last_linear = nn.Linear(model.last_linear.
                in_features, num_classes)
            self.model.last_linear.in_channels = linear.in_features
        elif isinstance(linear, nn.Conv2d):
            self.model.last_linear = nn.Conv2d(linear.in_channels,
                num_classes, kernel_size=linear.kernel_size, bias=True)
            self.model.last_linear.in_features = linear.in_channels
        self.model.last_linear = nn.Sequential(LambdaLayer(lambda x: x.
            unsqueeze_(0)), nn.AdaptiveAvgPool1d(self.model.last_linear.
            in_channels), LambdaLayer(lambda x: x.squeeze_(0).view(x.size(0
            ), -1)), self.model.last_linear)
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        res = self.model(x)
        if isinstance(res, tuple):
            res = res[0]
        if self.activation:
            res = self.activation(res)
        return res


class SegmentationModelPytorch(nn.Module):

    def __init__(self, arch: str, encoder: str, num_classes: int=1,
        encoder_weights: str='imagenet', activation=None, **kwargs):
        super().__init__()
        model = getattr(smb, arch)
        self.model = model(encoder_name=encoder, classes=num_classes,
            encoder_weights=encoder_weights, activation=activation, **kwargs)

    def forward(self, x):
        res = self.model.forward(x)
        if self.model.activation:
            res = self.model.activation(res)
        return res


class Timm(nn.Module):

    def __init__(self, variant, num_classes, pretrained=True, activation=None):
        super().__init__()
        model = timm.create_model(variant, pretrained=pretrained,
            num_classes=num_classes)
        self.model = model
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        res = self.model(x)
        if isinstance(res, tuple):
            res = res[0]
        if self.activation:
            res = self.activation(res)
        return res


class FullyConvolutionalLinear(nn.Module):

    def __init__(self, dim_in, num_classes):
        super(FullyConvolutionalLinear, self).__init__()
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        return x


def r2plus1_unit(dim_in, dim_out, temporal_stride, spatial_stride, groups,
    inplace_relu, bn_eps, bn_mmt, dim_mid=None):
    """
    Implementation of `R(2+1)D unit <https://arxiv.org/abs/1711.11248>`_.
    Decompose one 3D conv into one 2D spatial conv and one 1D temporal conv.
    Choose the middle dimensionality so that the total No. of parameters
    in 2D spatial conv and 1D temporal conv is unchanged.

    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temporal_stride (int): the temporal stride of the bottleneck.
        spatial_stride (int): the spatial_stride of the bottleneck.
        groups (int): number of groups for the convolution.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        bn_eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dim_mid (Optional[int]): If not None, use the provided channel dimension
            for the output of the 2D spatial conv. If None, compute the output
            channel dimension of the 2D spatial conv so that the total No. of
            model parameters remains unchanged.
    """
    if dim_mid is None:
        dim_mid = int(dim_out * dim_in * 3 * 3 * 3 / (dim_in * 3 * 3 + 
            dim_out * 3))
        logging.info('dim_in: %d, dim_out: %d. Set dim_mid to %d' % (dim_in,
            dim_out, dim_mid))
    conv_middle = nn.Conv3d(dim_in, dim_mid, [1, 3, 3], stride=[1,
        spatial_stride, spatial_stride], padding=[0, 1, 1], groups=groups,
        bias=False)
    conv_middle_bn = nn.BatchNorm3d(dim_mid, eps=bn_eps, momentum=bn_mmt)
    conv_middle_relu = nn.ReLU(inplace=inplace_relu)
    conv = nn.Conv3d(dim_mid, dim_out, [3, 1, 1], stride=[temporal_stride, 
        1, 1], padding=[1, 0, 0], groups=groups, bias=False)
    return nn.Sequential(conv_middle, conv_middle_bn, conv_middle_relu, conv)


class ResNeXt3D(torch.nn.Module):
    """
    Implementation of:
        1. Conventional `post-activated 3D ResNe(X)t <https://arxiv.org/
        abs/1812.03982>`_.

        2. `Pre-activated 3D ResNe(X)t <https://arxiv.org/abs/1811.12814>`_.
        The model consists of one stem, a number of stages, and one or multiple
        heads that are attached to different blocks in the stage.
    """

    def __init__(self, input_planes: int=3, skip_transformation_type: str=
        'postactivated_shortcut', residual_transformation_type: str=
        'basic_transformation', num_blocks: list=(2, 2, 2, 2), stem_name:
        str='resnext3d_stem', stem_planes: int=64, stem_temporal_kernel:
        int=3, stem_spatial_kernel: int=7, stem_maxpool: bool=False,
        stage_planes: int=64, stage_temporal_kernel_basis: list=([3], [3],
        [3], [3]), temporal_conv_1x1: list=(False, False, False, False),
        stage_temporal_stride: list=(1, 2, 2, 2), stage_spatial_stride:
        list=(1, 2, 2, 2), num_groups: int=1, width_per_group: int=64,
        zero_init_residual_transform: bool=False, in_plane: int=512,
        num_classes: int=2):
        """
        Args:
            input_planes (int): the channel dimension of the input.
                Normally 3 is used for rgb input.
            skip_transformation_type (str): the type of skip transformation.
                residual_transformation_type (str):
                the type of residual transformation.
            num_blocks (list): list of the number of blocks in stages.
            stem_name (str): name of model stem.
            stem_planes (int): the output dimension
                of the convolution in the model stem.
            stem_temporal_kernel (int): the temporal kernel
                size of the convolution
                in the model stem.
            stem_spatial_kernel (int): the spatial kernel size
                of the convolution in the model stem.
            stem_maxpool (bool): If true, perform max pooling.
            stage_planes (int): the output channel dimension
                of the 1st residual stage
            stage_temporal_kernel_basis (list): Basis of temporal kernel
                sizes for each of the stage.
            temporal_conv_1x1 (bool): Only useful for BottleneckTransformation.
                In a pathaway, if True, do temporal convolution
                in the first 1x1
                Conv3d. Otherwise, do it in the second 3x3 Conv3d.
            stage_temporal_stride (int): the temporal stride of the residual
                transformation.
            stage_spatial_stride (int): the spatial stride of the the residual
                transformation.
            num_groups (int): number of groups for the convolution.
                num_groups = 1 is for standard ResNet like networks, and
                num_groups > 1 is for ResNeXt like networks.
            width_per_group (int): Number of channels per group in 2nd (group)
                conv in the residual transformation in the first stage
            zero_init_residual_transform (bool): if true, the weight of last
                operation, which could be either BatchNorm3D in post-activated
                transformation or Conv3D in pre-activated transformation,
                in the residual transformation is initialized to zero
            pool_size: for fully convolution layer
            in_plane: for fully convolution layer
            num_classes: number of classes
        """
        super().__init__()
        num_stages = len(num_blocks)
        out_planes = [(stage_planes * 2 ** i) for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [(num_groups * width_per_group * 2 ** i) for i in
            range(num_stages)]
        self.stem = model_stems[stem_name](stem_temporal_kernel,
            stem_spatial_kernel, input_planes, stem_planes, stem_maxpool)
        stages = []
        for s in range(num_stages):
            stage = ResStage(s + 1, [in_planes[s]], [out_planes[s]], [
                inner_planes[s]], [stage_temporal_kernel_basis[s]], [
                temporal_conv_1x1[s]], [stage_temporal_stride[s]], [
                stage_spatial_stride[s]], [num_blocks[s]], [num_groups],
                skip_transformation_type, residual_transformation_type,
                disable_pre_activation=s == 0, final_stage=s == num_stages - 1)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_residual_transform)
        self.final_avgpool = nn.AdaptiveAvgPool1d(in_plane)
        self.head_fcl = FullyConvolutionalLinear(in_plane, num_classes)

    def _init_parameter(self, zero_init_residual_transform):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if hasattr(m, 'final_transform_op'
                    ) and m.final_transform_op and zero_init_residual_transform:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if hasattr(m, 'final_transform_op'
                    ) and m.final_transform_op and zero_init_residual_transform:
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor(B, T, W, H, C)
        """
        out = self.stem([x])
        out = self.stages(out)[0]
        out = out.view((out.shape[0], 1, -1))
        out = self.final_avgpool(out)
        out = self.head_fcl(out)
        return out


class BasicTransformation(nn.Module):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        groups, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            groups (int): number of groups for the convolution.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BasicTransformation, self).__init__()
        self._construct_model(dim_in, dim_out, temporal_stride,
            spatial_stride, groups, inplace_relu, bn_eps, bn_mmt)

    def _construct_model(self, dim_in, dim_out, temporal_stride,
        spatial_stride, groups, inplace_relu, bn_eps, bn_mmt):
        branch2a = nn.Conv3d(dim_in, dim_out, [3, 3, 3], stride=[
            temporal_stride, spatial_stride, spatial_stride], padding=[1, 1,
            1], groups=groups, bias=False)
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        branch2b = nn.Conv3d(dim_out, dim_out, [3, 3, 3], stride=[1, 1, 1],
            padding=[1, 1, 1], groups=groups, bias=False)
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True
        self.transform = nn.Sequential(branch2a, branch2a_bn, branch2a_relu,
            branch2b, branch2b_bn)

    def forward(self, x):
        return self.transform(x)


class PostactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        num_groups, dim_inner, temporal_kernel_size=3, temporal_conv_1x1=
        True, spatial_stride_1x1=False, inplace_relu=True, bn_eps=1e-05,
        bn_mmt=0.1, **kwargs):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): if True, do temporal convolution in the fist
                1x1 Conv3d. Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            num_groups (int): number of groups for the convolution.
            dim_inner (int): the inner dimension of the block.
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            spatial_stride_1x1 (bool): if True, apply spatial_stride to 1x1 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(PostactivatedBottleneckTransformation, self).__init__()
        temporal_kernel_size_1x1, temporal_kernel_size_3x3 = (
            temporal_kernel_size, 1) if temporal_conv_1x1 else (1,
            temporal_kernel_size)
        str1x1, str3x3 = (spatial_stride, 1) if spatial_stride_1x1 else (1,
            spatial_stride)
        self.branch2a = nn.Conv3d(dim_in, dim_inner, kernel_size=[
            temporal_kernel_size_1x1, 1, 1], stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0], bias=False)
        self.branch2a_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=
            bn_mmt)
        self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(dim_inner, dim_inner, [
            temporal_kernel_size_3x3, 3, 3], stride=[temporal_stride,
            str3x3, str3x3], padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups, bias=False)
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=
            bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1],
            stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
        self.branch2c_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_bn.final_transform_op = True

    def forward(self, x):
        x = self.branch2a(x)
        x = self.branch2a_bn(x)
        x = self.branch2a_relu(x)
        x = self.branch2b(x)
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)
        x = self.branch2c(x)
        x = self.branch2c_bn(x)
        return x


class PreactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation with pre-activation, which includes BatchNorm3D
        and ReLu. Conv3D kernsl are Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel (https://arxiv.org/abs/1603.05027).
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        num_groups, dim_inner, temporal_kernel_size=3, temporal_conv_1x1=
        True, spatial_stride_1x1=False, inplace_relu=True, bn_eps=1e-05,
        bn_mmt=0.1, disable_pre_activation=False, **kwargs):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): if True, do temporal convolution in the fist
                1x1 Conv3d. Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            num_groups (int): number of groups for the convolution.
            dim_inner (int): the inner dimension of the block.
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            spatial_stride_1x1 (bool): if True, apply spatial_stride to 1x1 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            disable_pre_activation (bool): If true, disable pre activation,
                including BatchNorm3D and ReLU.
        """
        super(PreactivatedBottleneckTransformation, self).__init__()
        temporal_kernel_size_1x1, temporal_kernel_size_3x3 = (
            temporal_kernel_size, 1) if temporal_conv_1x1 else (1,
            temporal_kernel_size)
        str1x1, str3x3 = (spatial_stride, 1) if spatial_stride_1x1 else (1,
            spatial_stride)
        self.disable_pre_activation = disable_pre_activation
        if not disable_pre_activation:
            self.branch2a_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=
                bn_mmt)
            self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2a = nn.Conv3d(dim_in, dim_inner, kernel_size=[
            temporal_kernel_size_1x1, 1, 1], stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0], bias=False)
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=
            bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(dim_inner, dim_inner, [
            temporal_kernel_size_3x3, 3, 3], stride=[temporal_stride,
            str3x3, str3x3], padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups, bias=False)
        self.branch2c_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=
            bn_mmt)
        self.branch2c_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1],
            stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
        self.branch2c.final_transform_op = True

    def forward(self, x):
        if not self.disable_pre_activation:
            x = self.branch2a_bn(x)
            x = self.branch2a_relu(x)
        x = self.branch2a(x)
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)
        x = self.branch2b(x)
        x = self.branch2c_bn(x)
        x = self.branch2c_relu(x)
        x = self.branch2c(x)
        return x


class PostactivatedShortcutTransformation(nn.Module):
    """
    Skip connection used in ResNet3D model.
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        bn_eps=1e-05, bn_mmt=0.1, **kwargs):
        super(PostactivatedShortcutTransformation, self).__init__()
        assert dim_in != dim_out or spatial_stride != 1 or temporal_stride != 1
        self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[
            temporal_stride, spatial_stride, spatial_stride], padding=0,
            bias=False)
        self.branch1_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)

    def forward(self, x):
        return self.branch1_bn(self.branch1(x))


class PreactivatedShortcutTransformation(nn.Module):
    """
    Skip connection with pre-activation, which includes BatchNorm3D and ReLU,
        in ResNet3D model (https://arxiv.org/abs/1603.05027).
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, disable_pre_activation
        =False, **kwargs):
        super(PreactivatedShortcutTransformation, self).__init__()
        assert dim_in != dim_out or spatial_stride != 1 or temporal_stride != 1
        if not disable_pre_activation:
            self.branch1_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=
                bn_mmt)
            self.branch1_relu = nn.ReLU(inplace=inplace_relu)
        self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[
            temporal_stride, spatial_stride, spatial_stride], padding=0,
            bias=False)

    def forward(self, x):
        if hasattr(self, 'branch1_bn') and hasattr(self, 'branch1_relu'):
            x = self.branch1_relu(self.branch1_bn(x))
        x = self.branch1(x)
        return x


class BasicR2Plus1DTransformation(BasicTransformation):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride,
        groups, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            groups (int): number of groups for the convolution.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BasicR2Plus1DTransformation, self).__init__(dim_in, dim_out,
            temporal_stride, spatial_stride, groups, inplace_relu=
            inplace_relu, bn_eps=bn_eps, bn_mmt=bn_mmt)

    def _construct_model(self, dim_in, dim_out, temporal_stride,
        spatial_stride, groups, inplace_relu, bn_eps, bn_mmt):
        branch2a = r2plus1_unit(dim_in, dim_out, temporal_stride,
            spatial_stride, groups, inplace_relu, bn_eps, bn_mmt)
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        branch2b = r2plus1_unit(dim_out, dim_out, 1, 1, groups,
            inplace_relu, bn_eps, bn_mmt)
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True
        self.transform = nn.Sequential(branch2a, branch2a_bn, branch2a_relu,
            branch2b, branch2b_bn)


residual_transformations = {'basic_r2plus1d_transformation':
    BasicR2Plus1DTransformation, 'basic_transformation':
    BasicTransformation, 'postactivated_bottleneck_transformation':
    PostactivatedBottleneckTransformation,
    'preactivated_bottleneck_transformation':
    PreactivatedBottleneckTransformation}


skip_transformations = {'postactivated_shortcut':
    PostactivatedShortcutTransformation, 'preactivated_shortcut':
    PreactivatedShortcutTransformation}


class ResBlock(nn.Module):
    """
    Residual block with skip connection.
    """

    def __init__(self, dim_in, dim_out, dim_inner, temporal_kernel_size,
        temporal_conv_1x1, temporal_stride, spatial_stride,
        skip_transformation_type, residual_transformation_type, num_groups=
        1, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1,
        disable_pre_activation=False):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            dim_inner (int): the inner dimension of the block.
            temporal_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            temporal_conv_1x1 (bool): Only useful for PostactivatedBottleneckTransformation.
                if True, do temporal convolution in the fist 1x1 Conv3d.
                Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (int): the temporal stride of the bottleneck.
            spatial_stride (int): the spatial_stride of the bottleneck.
            stride (int): the stride of the bottleneck.
            skip_transformation_type (str): the type of skip transformation
            residual_transformation_type (str): the type of residual transformation
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            disable_pre_activation (bool): If true, disable the preactivation,
                which includes BatchNorm3D and ReLU.
        """
        super(ResBlock, self).__init__()
        assert skip_transformation_type in skip_transformations, 'unknown skip transformation: %s' % skip_transformation_type
        if dim_in != dim_out or spatial_stride != 1 or temporal_stride != 1:
            self.skip = skip_transformations[skip_transformation_type](dim_in,
                dim_out, temporal_stride, spatial_stride, bn_eps=bn_eps,
                bn_mmt=bn_mmt, disable_pre_activation=disable_pre_activation)
        assert residual_transformation_type in residual_transformations, 'unknown residual transformation: %s' % residual_transformation_type
        self.residual = residual_transformations[residual_transformation_type](
            dim_in, dim_out, temporal_stride, spatial_stride, num_groups,
            dim_inner, temporal_kernel_size=temporal_kernel_size,
            temporal_conv_1x1=temporal_conv_1x1, disable_pre_activation=
            disable_pre_activation)
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        if hasattr(self, 'skip'):
            x = self.skip(x) + self.residual(x)
        else:
            x = x + self.residual(x)
        x = self.relu(x)
        return x


class ResStageBase(nn.Module):

    def __init__(self, stage_idx, dim_in, dim_out, dim_inner,
        temporal_kernel_basis, temporal_conv_1x1, temporal_stride,
        spatial_stride, num_blocks, num_groups):
        super(ResStageBase, self).__init__()
        assert len({len(dim_in), len(dim_out), len(temporal_kernel_basis),
            len(temporal_conv_1x1), len(temporal_stride), len(
            spatial_stride), len(num_blocks), len(dim_inner), len(num_groups)}
            ) == 1
        self.stage_idx = stage_idx
        self.num_blocks = num_blocks
        self.num_pathways = len(self.num_blocks)
        self.temporal_kernel_sizes = [(temporal_kernel_basis[i] *
            num_blocks[i])[:num_blocks[i]] for i in range(len(
            temporal_kernel_basis))]

    def _block_name(self, pathway_idx, stage_idx, block_idx):
        return 'pathway{}-stage{}-block{}'.format(pathway_idx, stage_idx,
            block_idx)

    def _pathway_name(self, pathway_idx):
        return 'pathway{}'.format(pathway_idx)

    def forward(self, inputs):
        output = []
        for p in range(self.num_pathways):
            x = inputs[p]
            pathway_module = getattr(self, self._pathway_name(p))
            output.append(pathway_module(x))
        return output


class ResNeXt3DStemSinglePathway(nn.Module):
    """
    ResNe(X)t 3D basic stem module. Assume a single pathway.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, maxpool=
        True, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            maxpool (bool): If true, perform max pooling.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(ResNeXt3DStemSinglePathway, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt
        self.maxpool = maxpool
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(dim_in, dim_out, self.kernel, stride=self.
            stride, padding=self.padding, bias=False)
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1,
                2, 2], padding=[0, 1, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.pool_layer(x)
        return x


class ResNeXt3DStemMultiPathway(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding,
        inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, maxpool=(True,)):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            bn_eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            maxpool (iterable): At training time, when crop size is 224 x 224, do max
                pooling. When crop size is 112 x 112, skip max pooling.
                Default value is a (True,)
        """
        super(ResNeXt3DStemMultiPathway, self).__init__()
        assert len({len(dim_in), len(dim_out), len(kernel), len(stride),
            len(padding)}) == 1, 'Input pathway dimensions are not consistent.'
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt
        self.maxpool = maxpool
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        assert type(dim_in) == list
        assert all(dim > 0 for dim in dim_in)
        assert type(dim_out) == list
        assert all(dim > 0 for dim in dim_out)
        self.blocks = {}
        for p in range(len(dim_in)):
            stem = ResNeXt3DStemSinglePathway(dim_in[p], dim_out[p], self.
                kernel[p], self.stride[p], self.padding[p], inplace_relu=
                self.inplace_relu, bn_eps=self.bn_eps, bn_mmt=self.bn_mmt,
                maxpool=self.maxpool[p])
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem

    def _stem_name(self, path_idx):
        return 'stem-path{}'.format(path_idx)

    def forward(self, x):
        assert len(x
            ) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(
            self.num_pathways)
        for p in range(len(x)):
            stem_name = self._stem_name(p)
            x[p] = self.blocks[stem_name](x[p])
        return x


class ResNeXt3DStem(nn.Module):

    def __init__(self, temporal_kernel, spatial_kernel, input_planes,
        stem_planes, maxpool):
        super(ResNeXt3DStem, self).__init__()
        self._construct_stem(temporal_kernel, spatial_kernel, input_planes,
            stem_planes, maxpool)

    def _construct_stem(self, temporal_kernel, spatial_kernel, input_planes,
        stem_planes, maxpool):
        self.stem = ResNeXt3DStemMultiPathway([input_planes], [stem_planes],
            [[temporal_kernel, spatial_kernel, spatial_kernel]], [[1, 2, 2]
            ], [[temporal_kernel // 2, spatial_kernel // 2, spatial_kernel //
            2]], maxpool=[maxpool])

    def forward(self, x):
        return self.stem(x)


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Conv2dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
        stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding, bias=not use_batchnorm), nn.ReLU(
            inplace=True)]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SCSEModule(nn.Module):

    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch //
            re, 1), nn.ReLU(inplace=True), nn.Conv2d(ch // re, ch, 1), nn.
            Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
        BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=dilation, bias
            =False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=
            dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1],
            dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2],
            dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3],
            dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False), BatchNorm(
            256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
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


class DRN(nn.Module):

    def __init__(self, block, layers, arch='D', channels=(16, 32, 64, 128, 
        256, 512, 512, 512), BatchNorm=None):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]
        self.arch = arch
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[
                0], stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[
                1], stride=2, BatchNorm=BatchNorm)
        elif arch == 'D':
            self.layer0 = nn.Sequential(nn.Conv2d(3, channels[0],
                kernel_size=7, stride=1, padding=3, bias=False), BatchNorm(
                channels[0]), nn.ReLU(inplace=True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0],
                stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._make_conv_layers(channels[1], layers[1],
                stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, channels[2], layers[2],
            stride=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, channels[3], layers[3],
            stride=2, BatchNorm=BatchNorm)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
            dilation=2, new_level=False, BatchNorm=BatchNorm)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block,
            channels[5], layers[5], dilation=4, new_level=False, BatchNorm=
            BatchNorm)
        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(
                BasicBlock, channels[6], layers[6], dilation=2, new_level=
                False, residual=False, BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else self._make_layer(
                BasicBlock, channels[7], layers[7], dilation=1, new_level=
                False, residual=False, BatchNorm=BatchNorm)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(
                channels[6], layers[6], dilation=2, BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(
                channels[7], layers[7], dilation=1, BatchNorm=BatchNorm)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        new_level=True, residual=True, BatchNorm=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if
            new_level else dilation, dilation), residual=residual,
            BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                dilation=(dilation, dilation), BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1,
        BatchNorm=None):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(self.inplanes, channels, kernel_size=
                3, stride=stride if i == 0 else 1, padding=dilation, bias=
                False, dilation=dilation), BatchNorm(channels), nn.ReLU(
                inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        low_level_feat = x
        x = self.layer4(x)
        x = self.layer5(x)
        if self.layer6 is not None:
            x = self.layer6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)
        return x, low_level_feat


class DRN_A(nn.Module):

    def __init__(self, block, layers, BatchNorm=None):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=
            BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, BatchNorm=BatchNorm)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation), BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(
                hidden_dim, oup, 1, 1, 0, 1, 1, bias=False), BatchNorm(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1,
                bias=False), BatchNorm(hidden_dim), nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation,
                groups=hidden_dim, bias=False), BatchNorm(hidden_dim), nn.
                ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1,
                bias=False), BatchNorm(oup))

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1.0,
        pretrained=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        current_stride *= 2
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, stride, dilation, t, BatchNorm))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()
        if pretrained:
            self._load_pretrained_model()
        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True
        ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides
            [0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=
            strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=
            strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=
            strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation,
            downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1,
        BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[
            0] * dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=
                blocks[i] * dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,
            dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1
            .dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
        BatchNorm=None, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=
                False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation,
                BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation,
                BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation,
                BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=
                BatchNorm))
            rep.append(BatchNorm(planes))
        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=
                BatchNorm))
            rep.append(BatchNorm(planes))
        if not start_with_relu:
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, output_stride, BatchNorm, pretrained=True):
        super(AlignedXception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)
        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm,
            start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm,
            start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            BatchNorm=BatchNorm, start_with_relu=True, grow_first=True,
            is_last=True)
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=True)
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=
            exit_block_dilations[0], BatchNorm=BatchNorm, start_with_relu=
            True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=
            exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
            )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class Decoder(nn.Module):

    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1, bias=False), BatchNorm(256), nn.ReLU(), nn
            .Dropout(0.5), nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1, bias=False), BatchNorm(256), nn.ReLU(), nn.Dropout(
            0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode=
            'bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)


class DeepLab(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
        freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.
                    BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.
                    BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class Conv3x3GNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3,
            3), stride=1, padding=1, bias=False), nn.GroupNorm(32,
            out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        return x


class FPNBlock(nn.Module):

    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels,
            kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(
            n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels,
                    upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class TransposeX2(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, **
        batchnorm_params):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=4, stride=2, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(Conv2dReLU(in_channels, in_channels // 4,
            kernel_size=1, use_batchnorm=use_batchnorm), TransposeX2(
            in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm
            ), Conv2dReLU(in_channels // 4, out_channels, kernel_size=1,
            use_batchnorm=use_batchnorm))

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


def _upsample(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class PyramidStage(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True
        ):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(
            pool_size, pool_size)), Conv2dReLU(in_channels, out_channels, (
            1, 1), use_batchnorm=use_bathcnorm))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = _upsample(x, size=(h, w))
        return x


class PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()
        self.stages = nn.ModuleList([PyramidStage(in_channels, in_channels //
            len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes])

    def forward(self, x):
        xs = [stage(x) for stage in self.stages] + [x]
        x = torch.cat(xs, dim=1)
        return x


class AUXModule(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True,
        attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)
        self.block = nn.Sequential(Conv2dReLU(in_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1,
            use_batchnorm=use_batchnorm))

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.block(x)
        x = self.attention2(x)
        return x


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class _ForwardOverrideModel(nn.Module):
    """
    Model that calls specified method instead of forward

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method = method_name

    def forward(self, *args, **kwargs):
        args = args[0][self.method]
        if isinstance(args, dict):
            kwargs = args
            args = ()
        return getattr(self.model, self.method)(*args, **kwargs)


class _TracingModelWrapper(nn.Module):
    """
    Wrapper that traces model with batch instead of calling it

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.method_name = method_name
        self.model = model
        self.tracing_result: ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)
        example_inputs = {self.method_name: kwargs if len(kwargs) > 0 else args
            }
        self.tracing_result = torch.jit.trace(method_model, example_inputs=
            example_inputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lightforever_mlcomp(_paritybench_base):
    pass
    def test_000(self):
        self._check(AUXModule(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicR2Plus1DTransformation(*[], **{'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'groups': 1}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_002(self):
        self._check(BasicTransformation(*[], **{'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'groups': 1}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_003(self):
        self._check(Conv2dReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(FullyConvolutionalLinear(*[], **{'dim_in': 4, 'num_classes': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'dilation': 1, 'expand_ratio': 4, 'BatchNorm': _mock_layer}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(LambdaLayer(*[], **{'lambd': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(PSPModule(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(PostactivatedBottleneckTransformation(*[], **{'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'num_groups': 1, 'dim_inner': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_009(self):
        self._check(PostactivatedShortcutTransformation(*[], **{'dim_in': 1, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1}), [torch.rand([4, 1, 64, 64, 64])], {})

    @_fails_compile()
    def test_010(self):
        self._check(PyramidStage(*[], **{'in_channels': 4, 'out_channels': 4, 'pool_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(ResNeXt3DStemSinglePathway(*[], **{'dim_in': 4, 'dim_out': 4, 'kernel': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_012(self):
        self._check(SCSEModule(*[], **{'ch': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_013(self):
        self._check(TransposeX2(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

