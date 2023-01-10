import sys
_module = sys.modules[__name__]
del sys
conf = _module
__about__ = _module
pl_bolts = _module
callbacks = _module
byol_updates = _module
data_monitor = _module
knn_online = _module
printing = _module
sparseml = _module
ssl_online = _module
torch_ort = _module
variational = _module
verification = _module
base = _module
batch_gradient = _module
vision = _module
confused_logit = _module
image_generation = _module
sr_image_logger = _module
datamodules = _module
async_dataloader = _module
binary_emnist_datamodule = _module
binary_mnist_datamodule = _module
cifar10_datamodule = _module
cityscapes_datamodule = _module
emnist_datamodule = _module
experience_source = _module
fashion_mnist_datamodule = _module
imagenet_datamodule = _module
kitti_datamodule = _module
mnist_datamodule = _module
sklearn_datamodule = _module
sr_datamodule = _module
ssl_imagenet_datamodule = _module
stl10_datamodule = _module
vision_datamodule = _module
vocdetection_datamodule = _module
datasets = _module
array_dataset = _module
base_dataset = _module
cifar10_dataset = _module
concat_dataset = _module
dummy_dataset = _module
emnist_dataset = _module
imagenet_dataset = _module
kitti_dataset = _module
mnist_dataset = _module
sr_celeba_dataset = _module
sr_dataset_mixin = _module
sr_mnist_dataset = _module
sr_stl10_dataset = _module
ssl_amdim_datasets = _module
utils = _module
losses = _module
object_detection = _module
rl = _module
self_supervised_learning = _module
metrics = _module
aggregation = _module
object_detection = _module
models = _module
autoencoders = _module
basic_ae = _module
basic_ae_module = _module
basic_vae = _module
basic_vae_module = _module
components = _module
detection = _module
_supported_models = _module
torchvision_backbones = _module
faster_rcnn = _module
backbones = _module
faster_rcnn_module = _module
retinanet = _module
backbones = _module
retinanet_module = _module
yolo = _module
yolo_config = _module
yolo_layers = _module
yolo_module = _module
gans = _module
basic = _module
basic_gan_module = _module
components = _module
dcgan = _module
components = _module
dcgan_module = _module
pix2pix = _module
components = _module
pix2pix_module = _module
srgan = _module
components = _module
srgan_module = _module
srresnet_module = _module
mnist_module = _module
regression = _module
linear_regression = _module
logistic_regression = _module
advantage_actor_critic_model = _module
common = _module
agents = _module
cli = _module
distributions = _module
gym_wrappers = _module
memory = _module
networks = _module
double_dqn_model = _module
dqn_model = _module
dueling_dqn_model = _module
noisy_dqn_model = _module
per_dqn_model = _module
ppo_model = _module
reinforce_model = _module
sac_model = _module
vanilla_policy_gradient_model = _module
self_supervised = _module
amdim = _module
amdim_module = _module
datasets = _module
networks = _module
transforms = _module
byol = _module
byol_module = _module
models = _module
cpc = _module
cpc_finetuner = _module
cpc_module = _module
networks = _module
evaluator = _module
moco = _module
moco2_module = _module
resnets = _module
simclr = _module
simclr_finetuner = _module
simclr_module = _module
simsiam = _module
simsiam_module = _module
ssl_finetuner = _module
swav = _module
loss = _module
swav_finetuner = _module
swav_module = _module
swav_resnet = _module
image_gpt = _module
gpt2 = _module
igpt_module = _module
pixel_cnn = _module
segmentation = _module
unet = _module
optimizers = _module
lars = _module
lr_scheduler = _module
dataset_normalizations = _module
ssl_transforms = _module
utils = _module
arguments = _module
pretrained_weights = _module
self_supervised = _module
semi_supervised = _module
shaping = _module
stability = _module
types = _module
warnings = _module
setup = _module
tests = _module
test_data_monitor = _module
test_info_callbacks = _module
test_ort = _module
test_param_update_callbacks = _module
test_sparseml = _module
test_variational_callbacks = _module
test_base = _module
test_batch_gradient = _module
conftest = _module
test_dataloader = _module
test_datamodules = _module
test_experience_sources = _module
test_sklearn_dataloaders = _module
test_array_dataset = _module
test_base_dataset = _module
test_datasets = _module
test_utils = _module
helpers = _module
boring_model = _module
test_object_detection = _module
test_rl_loss = _module
test_aggregation = _module
test_object_detection = _module
integration = _module
test_gans = _module
unit = _module
test_basic_components = _module
test_actor_critic_models = _module
test_policy_models = _module
test_value_models = _module
test_scripts = _module
test_a2c = _module
test_agents = _module
test_memory = _module
test_ppo = _module
test_reinforce = _module
test_sac = _module
test_vpg = _module
test_wrappers = _module
test_models = _module
test_resnets = _module
test_ssl_scripts = _module
test_transforms = _module
test_autoencoders = _module
test_classic_ml = _module
test_detection = _module
test_mnist_templates = _module
test_vision = _module
test_lr_scheduler = _module
test_utilities = _module
test_normalizations = _module
test_transforms = _module
test_arguments = _module
test_self_supervised = _module
test_semi_supervised = _module

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


import inspect


import math


from typing import Sequence


from typing import Union


import torch.nn as nn


from torch import Tensor


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


import numpy as np


import torch


from torch import nn


from torch.nn import Module


from torch.utils.hooks import RemovableHandle


from typing import Tuple


from torch.nn import functional as F


from torch.optim import Optimizer


from abc import abstractmethod


from copy import deepcopy


from typing import Callable


from typing import Iterable


from typing import Type


import torch.nn.functional as F


import collections.abc as container_abcs


import re


from queue import Queue


from torch._six import string_classes


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from abc import ABC


from collections import deque


from collections import namedtuple


from typing import Iterator


from torch.utils.data import IterableDataset


from torch.utils.data.dataset import random_split


from torch.utils.data import random_split


import logging


from torchvision.ops import box_iou


from torchvision.ops import generalized_box_iou


from warnings import warn


from torch import optim


from torch.optim import Adam


from torch.optim.optimizer import Optimizer


from torch.nn.functional import softmax


from collections import OrderedDict


import collections


from torch import FloatTensor


from torch.distributions import Categorical


from torch.distributions import Normal


from torch.nn.functional import log_softmax


from torch.utils.model_zoo import load_url as load_state_dict_from_url


from torch import distributed as dist


from torch.optim.optimizer import required


import warnings


from torch.optim.lr_scheduler import _LRScheduler


import uuid


from torchvision import transforms as transform_lib


import torch.testing


from torch.utils.data import Subset


from torch.utils.data.dataloader import DataLoader


import functools


from torch.optim import SGD


from collections import Counter


class UnderReviewWarning(Warning):
    pass


def _create_full_message(message: str) ->str:
    return f'{message} The compatibility with other Lightning projects is not guaranteed and API may change at any time. The API and functionality may change without warning in future releases. More details: https://lightning-bolts.readthedocs.io/en/latest/stability.html'


def _create_docstring_message(docstring: str, message: str) ->str:
    rst_warning = '.. warning:: ' + _create_full_message(message)
    if docstring is None:
        return rst_warning
    return rst_warning + '\n\n    ' + docstring


def _raise_review_warning(message: str, stacklevel: int=6) ->None:
    rank_zero_warn(_create_full_message(message), stacklevel=stacklevel, category=UnderReviewWarning)


def under_review():
    """The under_review decorator is used to indicate that a particular feature is not properly reviewed and tested yet.
    A callable or type that has been marked as under_review will give a ``UnderReviewWarning`` when it is called or
    instantiated. This designation should be used following the description given in :ref:`stability`.
    Args:
        message: The message to include in the warning.
    Examples
    ________
    >>> from pytest import warns
    >>> from pl_bolts.utils.stability import under_review, UnderReviewWarning
    >>> @under_review()
    ... class MyExperimentalFeature:
    ...     pass
    ...
    >>> with warns(UnderReviewWarning, match="The feature MyExperimentalFeature is currently marked under review."):
    ...     MyExperimentalFeature()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ...
    <...>
    """

    def decorator(cls_or_callable: Union[Callable, Type], feature_name: Optional[str]=None, was_class: bool=False):
        if feature_name is None:
            feature_name = cls_or_callable.__qualname__
        message = f'The feature {feature_name} is currently marked under review.'
        filterwarnings('once', message, UnderReviewWarning)
        if inspect.isclass(cls_or_callable):
            cls_or_callable.__init__ = decorator(cls_or_callable.__init__, feature_name=cls_or_callable.__qualname__, was_class=True)
            cls_or_callable.__doc__ = _create_docstring_message(cls_or_callable.__doc__, message)
            return cls_or_callable

        @functools.wraps(cls_or_callable)
        def wrapper(*args, **kwargs):
            _raise_review_warning(message)
            return cls_or_callable(*args, **kwargs)
        if not was_class:
            wrapper.__doc__ = _create_docstring_message(cls_or_callable.__doc__, message)
        return wrapper
    return decorator


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Discriminator(nn.Module):

    def __init__(self, img_shape, hidden_dim=1024):
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) ->None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.gen = nn.Sequential(self._make_gen_block(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0), self._make_gen_block(feature_maps * 8, feature_maps * 4), self._make_gen_block(feature_maps * 4, feature_maps * 2), self._make_gen_block(feature_maps * 2, feature_maps), self._make_gen_block(feature_maps, image_channels, last_block=True))

    @staticmethod
    def _make_gen_block(in_channels: int, out_channels: int, kernel_size: int=4, stride: int=2, padding: int=1, bias: bool=False, last_block: bool=False) ->nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            gen_block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.Tanh())
        return gen_block

    def forward(self, noise: Tensor) ->Tensor:
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):

    def __init__(self, feature_maps: int, image_channels: int) ->None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = nn.Sequential(self._make_disc_block(image_channels, feature_maps, batch_norm=False), self._make_disc_block(feature_maps, feature_maps * 2), self._make_disc_block(feature_maps * 2, feature_maps * 4), self._make_disc_block(feature_maps * 4, feature_maps * 8), self._make_disc_block(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True))

    @staticmethod
    def _make_disc_block(in_channels: int, out_channels: int, kernel_size: int=4, stride: int=2, padding: int=1, bias: bool=False, batch_norm: bool=True, last_block: bool=False) ->nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(), nn.LeakyReLU(0.2, inplace=True))
        else:
            disc_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.Sigmoid())
        return disc_block

    def forward(self, x: Tensor) ->Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)


class MLP(nn.Module):
    """MLP architecture used as projectors in online and target networks and predictors in the online network.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 2048.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4096.
        output_dim (int, optional): Output dimension. Defaults to 256.

    Note:
        Default values for input, hidden, and output dimensions are based on values used in BYOL.
    """

    def __init__(self, input_dim: int=2048, hidden_dim: int=4096, output_dim: int=256) ->None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim, bias=True))

    def forward(self, x: Tensor) ->Tensor:
        return self.model(x)


class SiameseArm(nn.Module):
    """SiameseArm consolidates the encoder and projector networks of BYOL's symmetric architecture into a single
    class.

    Args:
        encoder (Union[str, nn.Module], optional): Online and target network encoder architecture.
            Defaults to "resnet50".
        encoder_out_dim (int, optional): Output dimension of encoder. Defaults to 2048.
        projector_hidden_dim (int, optional): Online and target network projector network hidden dimension.
            Defaults to 4096.
        projector_out_dim (int, optional): Online and target network projector network output dimension.
            Defaults to 256.
    """

    def __init__(self, encoder: Union[str, nn.Module]='resnet50', encoder_out_dim: int=2048, projector_hidden_dim: int=4096, projector_out_dim: int=256) ->None:
        super().__init__()
        if isinstance(encoder, str):
            self.encoder = torchvision_ssl_encoder(encoder)
        else:
            self.encoder = encoder
        self.projector = MLP(encoder_out_dim, projector_hidden_dim, projector_out_dim)

    def forward(self, x: Tensor) ->Tuple[Tensor, Tensor]:
        y = self.encoder(x)[0]
        z = self.projector(y)
        return y, z

    def encode(self, x: Tensor) ->Tensor:
        """Returns the encoded representation of a view. This method does not calculate the projection as in the
        forward method.

        Args:
            x (Tensor): sample to be encoded
        """
        return self.encoder(x)[0]


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, nmb_prototypes):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, widen=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, normalize=False, output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False, first_conv=True, maxpool1=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        num_out_filters = width_per_group * widen
        if first_conv:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        if maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2norm = normalize
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(nn.Linear(num_out_filters * block.expansion, hidden_mlp), nn.BatchNorm1d(hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim))
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.eval_mode:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in inputs]), return_counts=True)[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])
            if 'cuda' in str(self.conv1.weight.device):
                _out = self.forward_backbone(_out)
            else:
                _out = self.forward_backbone(_out)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class SWAVLoss(nn.Module):

    def __init__(self, temperature: float, crops_for_assign: tuple, nmb_crops: tuple, sinkhorn_iterations: int, epsilon: float, gpus: int, num_nodes: int):
        """Implementation for SWAV loss function.

        Args:
            temperature:  loss temperature
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            sinkhorn_iterations: iterations for sinkhorn normalization
            epsilon: epsilon val for swav assignments
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes:  num_nodes: number of nodes to train on
        """
        super().__init__()
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.softmax = nn.Softmax(dim=1)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.nmb_crops = nmb_crops
        self.gpus = gpus
        self.num_nodes = num_nodes
        if self.gpus * self.num_nodes > 1:
            self.assignment_fn = self.distributed_sinkhorn
        else:
            self.assignment_fn = self.sinkhorn

    def forward(self, output: torch.Tensor, embedding: torch.Tensor, prototype_weights: torch.Tensor, batch_size: int, queue: Optional[torch.Tensor]=None, use_queue: bool=False) ->Tuple[int, Optional[torch.Tensor], bool]:
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[batch_size * crop_id:batch_size * (crop_id + 1)]
                if queue is not None:
                    if use_queue or not torch.all(queue[i, -1, :] == 0):
                        use_queue = True
                        out = torch.cat((torch.mm(queue[i], prototype_weights.t()), out))
                    queue[i, batch_size:] = self.queue[i, :-batch_size].clone()
                    queue[i, :batch_size] = embedding[crop_id * batch_size:(crop_id + 1) * batch_size]
                q = torch.exp(out / self.epsilon).t()
                q = self.assignment_fn(q, self.sinkhorn_iterations)[-batch_size:]
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[batch_size * v:batch_size * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss, queue, use_queue

    def sinkhorn(self, Q: torch.Tensor, nmb_iters: int) ->torch.Tensor:
        """Implementation of Sinkhorn clustering."""
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            K, B = Q.shape
            if self.gpus > 0:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B
            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q: torch.Tensor, nmb_iters: int) ->torch.Tensor:
        """Implementation of Distributed Sinkhorn."""
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q
            if self.gpus > 0:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)
            for _ in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) ->Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x: Tensor) ->Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool=False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, in_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) ->Tensor:
        x1 = self.upsample(x1)
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Pytorch Lightning implementation of U-Net.

    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
    """

    def __init__(self, num_classes: int, input_channels: int=3, num_layers: int=5, features_start: int=64, bilinear: bool=False):
        if num_layers < 1:
            raise ValueError(f'num_layers = {num_layers}, expected: num_layers > 0')
        super().__init__()
        self.num_layers = num_layers
        layers = [DoubleConv(input_channels, features_start)]
        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2
        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) ->Tensor:
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class SubModule(nn.Module):

    def __init__(self, inp, out):
        super().__init__()
        self.sub_layer = nn.Linear(inp, out)

    def forward(self, *args, **kwargs):
        return self.sub_layer(*args, **kwargs)


class ModuleDataMonitorModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(12, 5)
        self.layer2 = SubModule(5, 2)

    def forward(self, x):
        x = x.flatten(1)
        self.layer1_input = x
        x = self.layer1(x)
        self.layer1_output = x
        x = torch.relu(x + 1)
        self.layer2_input = x
        x = self.layer2(x)
        self.layer2_output = x
        x = torch.relu(x - 2)
        return x


class PyTorchModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 2)

    def forward(self, *args):
        return args


class TemplateModel(nn.Module):

    def __init__(self, mix_data=False):
        """Base model for testing.

        The setting ``mix_data=True`` simulates a wrong implementation.
        """
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(10)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        if self.mix_data:
            x = x.view(10, -1).permute(1, 0).view(-1, 10)
        else:
            x = x.view(-1, 10)
        return self.linear(self.bn(x))


class MultipleInputModel(TemplateModel):
    """Base model for testing verification when forward accepts multiple arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = torch.rand(10, 5, 2), torch.rand(10, 5, 2)

    def forward(self, x, y, some_kwarg=True):
        out = super().forward(x) + super().forward(y)
        return out


class MultipleOutputModel(TemplateModel):
    """Base model for testing verification when forward has multiple outputs."""

    def forward(self, x):
        out = super().forward(x)
        return None, out, out, False


class DictInputDictOutputModel(TemplateModel):
    """Base model for testing verification when forward has a collection of outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = {'w': 42, 'x': {'a': torch.rand(3, 5, 2)}, 'y': torch.rand(3, 1, 5, 2), 'z': torch.tensor(2)}

    def forward(self, y, x, z, w):
        out1 = super().forward(x['a'])
        out2 = super().forward(y)
        out3 = out1 + out2
        out = {(1): out1, (2): out2, (3): [out1, out3]}
        return out


class BatchNormModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm0 = nn.BatchNorm1d(2)
        self.batch_norm1 = nn.BatchNorm1d(3)
        self.instance_norm = nn.InstanceNorm1d(4)


class SchedulerTestNet(torch.nn.Module):
    """adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DCGANDiscriminator,
     lambda: ([], {'feature_maps': 4, 'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (DCGANGenerator,
     lambda: ([], {'latent_dim': 4, 'feature_maps': 4, 'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {'img_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiPrototypes,
     lambda: ([], {'output_dim': 4, 'nmb_prototypes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultipleInputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10]), torch.rand([4, 10])], {}),
     False),
    (MultipleOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10])], {}),
     False),
    (PyTorchModel,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (SchedulerTestNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (SubModule,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_Lightning_AI_lightning_bolts(_paritybench_base):
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

