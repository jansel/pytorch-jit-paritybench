import sys
_module = sys.modules[__name__]
del sys
classy_train = _module
classy_vision = _module
dataset = _module
classy_cifar = _module
classy_dataset = _module
classy_hmdb51 = _module
classy_kinetics400 = _module
classy_synthetic_image = _module
classy_synthetic_image_streaming = _module
classy_synthetic_video = _module
classy_ucf101 = _module
classy_video_dataset = _module
core = _module
list_dataset = _module
random_image_datasets = _module
random_video_datasets = _module
dataloader_async_gpu_wrapper = _module
dataloader_limit_wrapper = _module
dataloader_skip_none_wrapper = _module
dataloader_wrapper = _module
image_path_dataset = _module
transforms = _module
autoaugment = _module
classy_transform = _module
lighting_transform = _module
mixup = _module
util = _module
util_video = _module
distributed = _module
launch_ray = _module
generic = _module
debug = _module
distributed_util = _module
opts = _module
pdb = _module
perf_stats = _module
profiler = _module
registry_utils = _module
util = _module
visualize = _module
heads = _module
classy_head = _module
fully_connected_head = _module
fully_convolutional_linear_head = _module
identity_head = _module
vision_transformer_head = _module
hooks = _module
checkpoint_hook = _module
classy_hook = _module
constants = _module
exponential_moving_average_model_hook = _module
loss_lr_meter_logging_hook = _module
model_complexity_hook = _module
model_tensorboard_hook = _module
output_csv_hook = _module
precise_batch_norm_hook = _module
profiler_hook = _module
progress_bar_hook = _module
tensorboard_plot_hook = _module
torchscript_hook = _module
visdom_hook = _module
hub = _module
classy_hub_interface = _module
hydra = _module
conf = _module
losses = _module
barron_loss = _module
classy_loss = _module
label_smoothing_loss = _module
multi_output_sum_loss = _module
soft_target_cross_entropy_loss = _module
sum_arbitrary_loss = _module
meters = _module
accuracy_meter = _module
classy_meter = _module
precision_meter = _module
recall_meter = _module
video_accuracy_meter = _module
video_meter = _module
models = _module
anynet = _module
classy_block = _module
classy_model = _module
densenet = _module
efficientnet = _module
lecun_normal_init = _module
mlp = _module
r2plus1_util = _module
regnet = _module
resnet = _module
resnext = _module
resnext3d = _module
resnext3d_block = _module
resnext3d_stage = _module
resnext3d_stem = _module
squeeze_and_excitation_layer = _module
vision_transformer = _module
optim = _module
adam = _module
adamw = _module
adamw_mt = _module
classy_optimizer = _module
param_scheduler = _module
classy_vision_param_scheduler = _module
composite_scheduler = _module
fvcore_schedulers = _module
rmsprop = _module
rmsprop_tf = _module
sgd = _module
zero = _module
tasks = _module
classification_task = _module
classy_task = _module
fine_tuning_task = _module
datasets = _module
my_dataset = _module
my_loss = _module
my_model = _module
trainer = _module
classy_trainer = _module
distributed_trainer = _module
local_trainer = _module
hubconf = _module
classy_vision_path = _module
parse_sphinx = _module
parse_tutorials = _module
setup = _module
test = _module
api_test = _module
classy_vision_head_test = _module
dataloader_async_gpu_wrapper_test = _module
dataset_classy_dataset_test = _module
dataset_classy_video_dataset_test = _module
dataset_dataloader_limit_wrapper_test = _module
dataset_image_path_dataset_test = _module
dataset_transforms_autoaugment_test = _module
dataset_transforms_lighting_transform_test = _module
dataset_transforms_mixup_test = _module
dataset_transforms_test = _module
dataset_transforms_util_test = _module
dataset_transforms_util_video_test = _module
config_utils = _module
hook_test_utils = _module
merge_dataset = _module
meter_test_utils = _module
optim_test_util = _module
utils = _module
generic_distributed_util_test = _module
generic_profiler_test = _module
generic_util_test = _module
heads_fully_connected_head_test = _module
heads_fully_convolutional_linear_head_test = _module
heads_vision_transformer_head_test = _module
hooks_checkpoint_hook_test = _module
hooks_classy_hook_test = _module
hooks_exponential_moving_average_model_hook_test = _module
hooks_loss_lr_meter_logging_hook_test = _module
hooks_output_csv_hook_test = _module
hooks_precise_batch_norm_hook_test = _module
hooks_profiler_hook_test = _module
hooks_torchscript_hook_test = _module
hub_classy_hub_interface_test = _module
losses_barron_loss_test = _module
losses_generic_utils_test = _module
losses_label_smoothing_cross_entropy_loss_test = _module
losses_multi_output_sum_loss_test = _module
losses_soft_target_cross_entropy_loss_test = _module
losses_sum_arbitrary_loss_test = _module
losses_test = _module
hooks_model_complexity_hook_test = _module
hooks_model_tensorboard_hook_test = _module
hooks_progress_bar_hook_test = _module
hooks_tensorboard_plot_hook_test = _module
hooks_visdom_hook_test = _module
models_classy_vision_model_test = _module
optim_adamw_mt_test = _module
optim_adamw_test = _module
tasks_classification_task_amp_test = _module
meters_accuracy_meter_test = _module
meters_precision_meter_test = _module
meters_recall_meter_test = _module
meters_video_accuracy_meter_test = _module
models_classy_block_stateless_test = _module
models_classy_block_test = _module
models_classy_model_test = _module
models_densenet_test = _module
models_efficientnet_test = _module
models_mlp_test = _module
models_regnet_test = _module
models_resnext3d_test = _module
models_resnext_test = _module
models_vision_transformer_test = _module
optim_adam_test = _module
optim_param_scheduler_composite_test = _module
optim_param_scheduler_constant_test = _module
optim_param_scheduler_cosine_test = _module
optim_param_scheduler_linear_test = _module
optim_param_scheduler_multi_step_test = _module
optim_param_scheduler_polynomial_test = _module
optim_param_scheduler_step_test = _module
optim_param_scheduler_step_with_fixed_gamma_test = _module
optim_param_scheduler_test = _module
optim_rmsprop_test = _module
optim_rmsprop_tf_test = _module
optim_sgd_test = _module
optim_sharded_sgd_test = _module
suites = _module
tasks_classification_task_test = _module
tasks_fine_tuning_task_test = _module
test_generic_utils_test = _module
trainer_distributed_trainer_test = _module
trainer_local_trainer_test = _module

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


import logging


import torch


from torchvision import set_image_backend


from torchvision import set_video_backend


from typing import Any


from typing import Callable


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import Union


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torchvision.datasets.hmdb51 import HMDB51


from typing import List


from torchvision.datasets.kinetics import Kinetics


from torchvision.datasets.ucf101 import UCF101


from torch.utils.data import Sampler


from torchvision import get_video_backend


from torchvision.datasets.samplers.clip_sampler import DistributedSampler


from torchvision.datasets.samplers.clip_sampler import RandomClipSampler


from torchvision.datasets.samplers.clip_sampler import UniformClipSampler


from typing import Iterable


from typing import Iterator


import collections.abc as abc


import math


from typing import Tuple


import numpy as np


from torch.distributions.beta import Beta


import collections


import torchvision.transforms as transforms


import random


import torchvision.transforms._transforms_video as transforms_video


from collections import defaultdict


from collections import deque


from time import perf_counter


from typing import Mapping


from torch.cuda import Event as CudaEvent


import torch.nn as nn


from torch.cuda import cudart


import collections.abc


import time


from functools import partial


import torch.nn.modules as nn


from collections.abc import Sequence


import copy


from collections import OrderedDict


from typing import Collection


import itertools


from itertools import accumulate


import torch.nn.modules.loss as torch_losses


import torch.nn.functional as F


from torch import Tensor


from enum import auto


from enum import Enum


import types


from typing import NamedTuple


import warnings


import re


import torch.optim


from abc import ABC


from abc import abstractmethod


from torch.optim import Optimizer


import torch.distributed as dist


import enum


from torch.distributed import broadcast


import torchvision.models as models


import functools


from torch.utils.data import Dataset


from torchvision import transforms


import numpy


import queue


from functools import wraps


from itertools import product


from torch.multiprocessing import Event


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


from torchvision import models


from torch.utils.tensorboard import SummaryWriter


import torchvision.models


from torch.nn.modules.loss import CrossEntropyLoss


class ClassyHead(nn.Module):
    """
    Base class for heads that can be attached to :class:`ClassyModel`.

    A head is a regular :class:`torch.nn.Module` that can be attached to a
    pretrained model. This enables a form of transfer learning: utilizing a
    model trained for one dataset to extract features that can be used for
    other problems. A head must be attached to a :class:`models.ClassyBlock`
    within a :class:`models.ClassyModel`.
    """

    def __init__(self, unique_id: Optional[str]=None, num_classes: Optional[int]=None):
        """
        Constructs a ClassyHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head.
        """
        super().__init__()
        self.unique_id = unique_id or self.__class__.__name__
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ClassyHead':
        """Instantiates a ClassyHead from a configuration.

        Args:
            config: A configuration for the ClassyHead.

        Returns:
            A ClassyHead instance.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Performs inference on the head.

        This is a regular PyTorch method, refer to :class:`torch.nn.Module` for
        more details
        """
        raise NotImplementedError


NORMALIZE_L2 = 'l2'


RELU_IN_PLACE = True


def get_torch_version():
    """Get the torch version as [major, minor].

    All comparisons must be done with the two version values. Revisions are not
    supported.
    """
    version_list = torch.__version__.split('.')[:2]
    return [int(version_str) for version_str in version_list]


def is_pos_int(number: int) ->bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


HEAD_CLASS_NAMES = set()


HEAD_CLASS_NAMES_TB = {}


HEAD_REGISTRY = {}


HEAD_REGISTRY_TB = {}


def register_head(name, bypass_checks=False):
    """Registers a ClassyHead subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyHead from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyHead subclass, like this:

    .. code-block:: python

      @register_head("my_head")
      class MyHead(ClassyHead):
          ...

    To instantiate a head from a configuration file, see
    :func:`build_head`."""

    def register_head_cls(cls):
        if not bypass_checks:
            if name in HEAD_REGISTRY:
                msg = 'Cannot register duplicate head ({}). Already registered at \n{}\n'
                raise ValueError(msg.format(name, HEAD_REGISTRY_TB[name]))
            if not issubclass(cls, ClassyHead):
                raise ValueError('Head ({}: {}) must extend ClassyHead'.format(name, cls.__name__))
            if cls.__name__ in HEAD_CLASS_NAMES:
                msg = 'Cannot register head with duplicate class name({}).' + 'Previously registered at \n{}\n'
                raise ValueError(msg.format(cls.__name__, HEAD_CLASS_NAMES_TB[cls.__name__]))
        tb = ''.join(traceback.format_stack())
        HEAD_REGISTRY[name] = cls
        HEAD_CLASS_NAMES.add(cls.__name__)
        HEAD_REGISTRY_TB[name] = tb
        HEAD_CLASS_NAMES_TB[cls.__name__] = tb
        return cls
    return register_head_cls


class FullyConnectedHead(ClassyHead):
    """This head defines a 2d average pooling layer
    (:class:`torch.nn.AdaptiveAvgPool2d`) followed by a fully connected
    layer (:class:`torch.nn.Linear`).
    """

    def __init__(self, unique_id: str, num_classes: Optional[int], in_plane: int, conv_planes: Optional[int]=None, activation: Optional[nn.Module]=None, zero_init_bias: bool=False, normalize_inputs: Optional[str]=None):
        """Constructor for FullyConnectedHead

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head. If None, then the fully
                connected layer is not applied.
            in_plane: Input size for the fully connected layer.
            conv_planes: If specified, applies a 1x1 convolutional layer to the input
                before passing it to the average pooling layer. The convolution is also
                followed by a BatchNorm and an activation.
            activation: The activation to be applied after the convolutional layer.
                Unused if `conv_planes` is not specified.
            zero_init_bias: Zero initialize the bias
            normalize_inputs: If specified, normalize the inputs after performing
                average pooling using the specified method. Supports "l2" normalization.
        """
        super().__init__(unique_id, num_classes)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        if conv_planes is not None and activation is None:
            raise TypeError('activation cannot be None if conv_planes is specified')
        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(f'Unsupported value for normalize_inputs: {normalize_inputs}')
        self.conv = nn.Conv2d(in_plane, conv_planes, kernel_size=1, bias=False) if conv_planes else None
        self.bn = nn.BatchNorm2d(conv_planes) if conv_planes else None
        self.activation = activation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(in_plane if conv_planes is None else conv_planes, num_classes)
        self.normalize_inputs = normalize_inputs
        if zero_init_bias:
            self.fc.bias.data.zero_()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'FullyConnectedHead':
        """Instantiates a FullyConnectedHead from a configuration.

        Args:
            config: A configuration for a FullyConnectedHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConnectedHead instance.
        """
        num_classes = config.get('num_classes', None)
        in_plane = config['in_plane']
        silu = None if get_torch_version() < [1, 7] else nn.SiLU()
        activation = {'relu': nn.ReLU(RELU_IN_PLACE), 'silu': silu}[config.get('activation', 'relu')]
        if activation is None:
            raise RuntimeError('SiLU activation is only supported since PyTorch 1.7')
        return cls(config['unique_id'], num_classes, in_plane, conv_planes=config.get('conv_planes', None), activation=activation, zero_init_bias=config.get('zero_init_bias', False), normalize_inputs=config.get('normalize_inputs', None))

    def forward(self, x):
        out = x
        if self.conv is not None:
            out = self.activation(self.bn(self.conv(x)))
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                out = nn.functional.normalize(out, p=2.0, dim=1)
        if self.fc is not None:
            out = self.fc(out)
        return out


class FullyConvolutionalLinear(nn.Module):

    def __init__(self, dim_in, num_classes, act_func='softmax'):
        super(FullyConvolutionalLinear, self).__init__()
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_func == 'identity':
            self.act = nn.Identity()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(act_func))

    def forward(self, x):
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.flatten(start_dim=1)
        return x


class FullyConvolutionalLinearHead(ClassyHead):
    """
    This head defines a 3d average pooling layer (:class:`torch.nn.AvgPool3d` or
    :class:`torch.nn.AdaptiveAvgPool3d` if pool_size is None) followed by a fully
    convolutional linear layer. This layer performs a fully-connected projection
    during training, when the input size is 1x1x1.
    It performs a convolutional projection during testing when the input size
    is larger than 1x1x1.
    """

    def __init__(self, unique_id: str, num_classes: int, in_plane: int, pool_size: Optional[List[int]], activation_func: str, use_dropout: Optional[bool]=None, dropout_ratio: float=0.5):
        """
        Constructor for FullyConvolutionalLinearHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head.
            in_plane: Input size for the fully connected layer.
            pool_size: Optional kernel size for the 3d pooling layer. If None, use
                :class:`torch.nn.AdaptiveAvgPool3d` with output size (1, 1, 1).
            activation_func: activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            use_dropout: Whether to apply dropout after the pooling layer.
            dropout_ratio: dropout ratio.
        """
        super().__init__(unique_id, num_classes)
        if pool_size is not None:
            self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        else:
            self.final_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_ratio)
        self.head_fcl = FullyConvolutionalLinear(in_plane, num_classes, act_func=activation_func)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'FullyConvolutionalLinearHead':
        """Instantiates a FullyConvolutionalLinearHead from a configuration.

        Args:
            config: A configuration for a FullyConvolutionalLinearHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConvolutionalLinearHead instance.
        """
        required_args = ['in_plane', 'num_classes']
        for arg in required_args:
            assert arg in config, 'argument %s is required' % arg
        config.update({'activation_func': config.get('activation_func', 'softmax')})
        config.update({'use_dropout': config.get('use_dropout', False)})
        pool_size = config.get('pool_size', None)
        if pool_size is not None:
            assert isinstance(pool_size, Sequence) and len(pool_size) == 3
            for pool_size_dim in pool_size:
                assert is_pos_int(pool_size_dim)
        assert is_pos_int(config['in_plane'])
        assert is_pos_int(config['num_classes'])
        num_classes = config.get('num_classes', None)
        in_plane = config['in_plane']
        return cls(config['unique_id'], num_classes, in_plane, pool_size, config['activation_func'], config['use_dropout'], config.get('dropout_ratio', 0.5))

    def forward(self, x):
        out = self.final_avgpool(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        out = self.head_fcl(out)
        return out


def lecun_normal_init(tensor, fan_in):
    if get_torch_version() >= [1, 7]:
        trunc_normal_ = nn.init.trunc_normal_
    else:

        def trunc_normal_(tensor: Tensor, mean: float=0.0, std: float=1.0, a: float=-2.0, b: float=2.0) ->Tensor:

            def norm_cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            if mean < a - 2 * std or mean > b + 2 * std:
                warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.0))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor
    trunc_normal_(tensor, std=math.sqrt(1 / fan_in))


class VisionTransformerHead(ClassyHead):

    def __init__(self, unique_id: str, in_plane: int, num_classes: Optional[int]=None, hidden_dim: Optional[int]=None, normalize_inputs: Optional[str]=None):
        """
        Args:
            unique_id: A unique identifier for the head
            in_plane: Input size for the fully connected layer
            num_classes: Number of output classes for the head
            hidden_dim: If not None, a hidden layer with the specific dimension is added
            normalize_inputs: If specified, normalize the inputs using the specified
                method. Supports "l2" normalization.
        """
        super().__init__(unique_id, num_classes)
        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(f'Unsupported value for normalize_inputs: {normalize_inputs}')
        if num_classes is None:
            layers = []
        elif hidden_dim is None:
            layers = [('head', nn.Linear(in_plane, num_classes))]
        else:
            layers = [('pre_logits', nn.Linear(in_plane, hidden_dim)), ('act', nn.Tanh()), ('head', nn.Linear(hidden_dim, num_classes))]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.normalize_inputs = normalize_inputs
        self.init_weights()

    def init_weights(self):
        if hasattr(self.layers, 'pre_logits'):
            lecun_normal_init(self.layers.pre_logits.weight, fan_in=self.layers.pre_logits.in_features)
            nn.init.zeros_(self.layers.pre_logits.bias)
        if hasattr(self.layers, 'head'):
            nn.init.zeros_(self.layers.head.weight)
            nn.init.zeros_(self.layers.head.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        return cls(**config)

    def forward(self, x):
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                x = nn.functional.normalize(x, p=2.0, dim=1)
        return self.layers(x)


class ClassyLoss(nn.Module):
    """
    Base class to calculate the loss during training.

    This implementation of :class:`torch.nn.Module` allows building
    the loss object from a configuration file.
    """

    def __init__(self):
        """
        Constructor for ClassyLoss.
        """
        super(ClassyLoss, self).__init__()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ClassyLoss':
        """Instantiates a ClassyLoss from a configuration.

        Args:
            config: A configuration for a ClassyLoss.

        Returns:
            A ClassyLoss instance.
        """
        raise NotImplementedError()

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.

        Refer to :class:`torch.nn.Module` for more details.
        """
        raise NotImplementedError

    def get_classy_state(self) ->Dict[str, Any]:
        """Get the state of the ClassyLoss.

        The returned state is used for checkpointing. Note that most losses are
        stateless and do not need to save any state.

        Returns:
            A state dictionary containing the state of the loss.
        """
        return self.state_dict()

    def set_classy_state(self, state: Dict[str, Any]) ->None:
        """Set the state of the ClassyLoss.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the loss from a checkpoint. Note
        that most losses are stateless and do not need to load any state.
        """
        return self.load_state_dict(state)

    def has_learned_parameters(self) ->bool:
        """Does this loss have learned parameters?"""
        return any(param.requires_grad for param in self.parameters(recurse=True))


LOSS_REGISTRY = {}


def log_class_usage(component_type, klass):
    """This function is used to log the usage of different Classy components."""
    identifier = 'ClassyVision'
    if klass and hasattr(klass, '__name__'):
        identifier += f'.{component_type}.{klass.__name__}'
    torch._C._log_api_usage_once(identifier)


def build_loss(config):
    """Builds a ClassyLoss from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_loss",
    "foo": "bar"}` will find a class that was registered as "my_loss"
    (see :func:`register_loss`) and call .from_config on it.

    In addition to losses registered with :func:`register_loss`, we also
    support instantiating losses available in the `torch.nn.modules.loss <https:
    //pytorch.org/docs/stable/nn.html#loss-functions>`_
    module. Any keys in the config will get expanded to parameters of the loss
    constructor. For instance, the following call will instantiate a
    `torch.nn.modules.CrossEntropyLoss <https://pytorch.org/docs/stable/
    nn.html#torch.nn.CrossEntropyLoss>`_:

    .. code-block:: python

     build_loss({"name": "CrossEntropyLoss", "reduction": "sum"})
    """
    assert 'name' in config, f'name not provided for loss: {config}'
    name = config['name']
    args = copy.deepcopy(config)
    del args['name']
    if 'weight' in args and args['weight'] is not None:
        args['weight'] = torch.tensor(args['weight'], dtype=torch.float)
    if name in LOSS_REGISTRY:
        loss = LOSS_REGISTRY[name].from_config(config)
    else:
        assert hasattr(torch_losses, name), f"{name} isn't a registered loss, nor is it available in torch.nn.modules.loss"
        loss = getattr(torch_losses, name)(**args)
    log_class_usage('Loss', loss.__class__)
    return loss


LOSS_CLASS_NAMES = set()


LOSS_CLASS_NAMES_TB = {}


LOSS_REGISTRY_TB = {}


def register_loss(name, bypass_checks=False):
    """Registers a ClassyLoss subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    ClassyLoss from a configuration file, even if the class itself is not
    part of the Classy Vision framework. To use it, apply this decorator to a
    ClassyLoss subclass, like this:

    .. code-block:: python

     @register_loss("my_loss")
     class MyLoss(ClassyLoss):
          ...

    To instantiate a loss from a configuration file, see
    :func:`build_loss`."""

    def register_loss_cls(cls):
        if not bypass_checks:
            if name in LOSS_REGISTRY:
                msg = 'Cannot register duplicate loss ({}). Already registered at \n{}\n'
                raise ValueError(msg.format(name, LOSS_REGISTRY_TB[name]))
            if not issubclass(cls, ClassyLoss):
                raise ValueError('Loss ({}: {}) must extend ClassyLoss'.format(name, cls.__name__))
        tb = ''.join(traceback.format_stack())
        LOSS_REGISTRY[name] = cls
        LOSS_CLASS_NAMES.add(cls.__name__)
        LOSS_REGISTRY_TB[name] = tb
        LOSS_CLASS_NAMES_TB[cls.__name__] = tb
        return cls
    return register_loss_cls


class MultiOutputSumLoss(ClassyLoss):
    """
    Applies the provided loss to the list of outputs (or single output) and sums
    up the losses.
    """

    def __init__(self, loss) ->None:
        super().__init__()
        self._loss = loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'MultiOutputSumLoss':
        """Instantiates a MultiOutputSumLoss from a configuration.

        Args:
            config: A configuration for a MultiOutpuSumLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MultiOutputSumLoss instance.
        """
        assert type(config['loss']) == dict, 'loss must be a dict containing a configuration for a registered loss'
        return cls(loss=build_loss(config['loss']))

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]
        loss = 0
        for pred in output:
            loss += self._loss(pred, target)
        return loss


def convert_to_one_hot(targets: torch.Tensor, classes) ->torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    assert torch.max(targets).item() < classes, 'Class Index must be less than number of classes'
    one_hot_targets = torch.zeros((targets.shape[0], classes), dtype=torch.long, device=targets.device)
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


class SoftTargetCrossEntropyLoss(ClassyLoss):

    def __init__(self, ignore_index=-100, reduction='mean', normalize_targets=True):
        """Intializer for the soft target cross-entropy loss loss.
        This allows the targets for the cross entropy loss to be multilabel

        Args:
            ignore_index: sample should be ignored for loss if the class is this value
            reduction: specifies reduction to apply to the output
            normalize_targets: whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample
        """
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        assert isinstance(normalize_targets, bool)
        self._normalize_targets = normalize_targets
        if self._reduction not in ['none', 'mean']:
            raise NotImplementedError('reduction type "{}" not implemented'.format(self._reduction))
        self._eps = torch.finfo(torch.float32).eps

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'SoftTargetCrossEntropyLoss':
        """Instantiates a SoftTargetCrossEntropyLoss from a configuration.

        Args:
            config: A configuration for a SoftTargetCrossEntropyLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SoftTargetCrossEntropyLoss instance.
        """
        return cls(ignore_index=config.get('ignore_index', -100), reduction=config.get('reduction', 'mean'), normalize_targets=config.get('normalize_targets', True))

    def forward(self, output, target):
        """for N examples and C classes
        - output: N x C these are raw outputs (without softmax/sigmoid)
        - target: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        if target.ndim == 1:
            assert output.shape[0] == target.shape[0], 'SoftTargetCrossEntropyLoss requires output and target to have same batch size'
            target = convert_to_one_hot(target.view(-1, 1), output.shape[1])
        assert output.shape == target.shape, f'SoftTargetCrossEntropyLoss requires output and target to be same shape: {output.shape} != {target.shape}'
        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()
        if self._normalize_targets:
            valid_targets /= self._eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(output, -1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        if self._reduction == 'mean':
            loss = per_sample_loss.sum() / torch.sum(torch.sum(valid_mask, -1) > 0).clamp(min=1)
        elif self._reduction == 'none':
            loss = per_sample_loss
        return loss


class SumArbitraryLoss(ClassyLoss):
    """
    Sums a collection of (weighted) torch.nn losses.

    NOTE: this applies all the losses to the same output and does not support
    taking a list of outputs as input.
    """

    def __init__(self, losses: List[ClassyLoss], weights: Optional[Tensor]=None) ->None:
        super().__init__()
        if weights is None:
            weights = torch.ones(len(losses))
        self.losses = losses
        self.weights = weights

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'SumArbitraryLoss':
        """Instantiates a SumArbitraryLoss from a configuration.

        Args:
            config: A configuration for a SumArbitraryLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SumArbitraryLoss instance.
        """
        assert type(config['losses']) == list and len(config['losses']) > 0, 'losses must be a list of registered losses with length > 0'
        assert type(config['weights']) == list and len(config['weights']) == len(config['losses']), 'weights must be None or a list and have same length as losses'
        loss_modules = []
        for loss_config in config['losses']:
            loss_modules.append(build_loss(loss_config))
        assert all(isinstance(loss_module, ClassyLoss) for loss_module in loss_modules), 'All losses must be registered, valid ClassyLosses'
        return cls(losses=loss_modules, weights=config.get('weights', None))

    def forward(self, prediction, target):
        for idx, loss in enumerate(self.losses):
            current_loss = loss(prediction, target)
            if idx == 0:
                total_loss = current_loss
            else:
                total_loss = total_loss.add(self.weights[idx], current_loss)
        return total_loss


class BasicTransform(nn.Sequential):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation, nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False))
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 2


class ResStemCifar(nn.Sequential):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, width_in: int, width_out: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class ResStemIN(nn.Sequential):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, width_in: int, width_out: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation, nn.MaxPool2d(3, stride=2, padding=1))
        self.depth = 3


class SimpleStemIN(nn.Sequential):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, width_in: int, width_out: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class VanillaBlock(nn.Sequential):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module, *args, **kwargs):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.b = nn.Sequential(nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module, *args, **kwargs):
        super().__init__()
        self.proj_block = width_in != width_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(width_in, width_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BasicTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation)
        self.activation = activation
        self.depth = self.f.depth

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class SqueezeAndExcitationLayer(nn.Module):
    """Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, in_planes, reduction_ratio: Optional[int]=16, reduced_planes: Optional[int]=None, activation: Optional[nn.Module]=None):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        assert bool(reduction_ratio) != bool(reduced_planes)
        if activation is None:
            activation = nn.ReLU()
        reduced_planes = in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        self.excitation = nn.Sequential(nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True), activation, nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module, group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]):
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        self.a = nn.Sequential(nn.Conv2d(width_in, w_b, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.b = nn.Sequential(nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False), nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum), activation)
        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeAndExcitationLayer(in_planes=w_b, reduction_ratio=None, reduced_planes=width_se_out, activation=activation)
        self.c = nn.Conv2d(w_b, width_out, 1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 3 if not se_ratio else 4


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module, group_width: int=1, bottleneck_multiplier: float=1.0, se_ratio: Optional[float]=None):
        super().__init__()
        self.proj_block = width_in != width_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(width_in, width_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BottleneckTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation, group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation
        self.depth = self.f.depth

    def forward(self, x, *args):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class ResBottleneckLinearBlock(nn.Module):
    """Residual linear bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: int, width_out: int, stride: int, bn_epsilon: float, bn_momentum: float, activation: nn.Module, group_width: int=1, bottleneck_multiplier: float=4.0, se_ratio: Optional[float]=None):
        super().__init__()
        self.has_skip = width_in == width_out and stride == 1
        self.f = BottleneckTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation, group_width, bottleneck_multiplier, se_ratio)
        self.depth = self.f.depth

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: nn.Module, activation: nn.Module, group_width: int, bottleneck_multiplier: float, params: 'AnyNetParams', stage_index: int=0):
        super().__init__()
        self.stage_depth = 0
        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, params.bn_epsilon, params.bn_momentum, activation, group_width, bottleneck_multiplier, params.se_ratio)
            self.stage_depth += block.depth
            self.add_module(f'block{stage_index}-{i}', block)


class ClassyBlock(nn.Module):
    """
    This is a thin wrapper for head execution, which records the output of
    wrapped module for executing the heads forked from this module.
    """

    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.output = torch.zeros(0)
        self._module = module
        self._is_output_stateless = os.environ.get('CLASSY_BLOCK_STATELESS') == '1'

    def wrapped_module(self):
        return self._module

    def forward(self, input):
        if hasattr(self, '_is_output_stateless'):
            if self._is_output_stateless:
                return self._module(input)
        output = self._module(input)
        self.output = output
        return output


class _ClassyModelMethod:
    """Class to override ClassyModel method calls to ensure the wrapper is returned.

    This helps override calls like model.cuda() which return self, to return the
    wrapper instead of the underlying classy_model.
    """

    def __init__(self, wrapper, classy_method):
        self.wrapper = wrapper
        self.classy_method = classy_method

    def __call__(self, *args, **kwargs):
        ret_val = self.classy_method(*args, **kwargs)
        if ret_val is self.wrapper.classy_model:
            ret_val = self.wrapper
        return ret_val


class ClassyModelWrapper:
    """Base ClassyModel wrapper class.

    This class acts as a thin pass through wrapper which lets users modify the behavior
    of ClassyModels, such as changing the return output of the forward() call.
    This wrapper acts as a ClassyModel by itself and the underlying model can be
    accessed by the `classy_model` attribute.
    """

    def __init__(self, classy_model):
        self.classy_model = classy_model

    def __getattr__(self, name):
        if name != 'classy_model' and hasattr(self, 'classy_model'):
            attr = getattr(self.classy_model, name)
            if isinstance(attr, types.MethodType):
                attr = _ClassyModelMethod(self, attr)
            return attr
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name not in ['classy_model', 'forward'] and hasattr(self, 'classy_model'):
            setattr(self.classy_model, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name != 'classy_model' and hasattr(self, 'classy_model'):
            delattr(self.classy_model, name)
        else:
            return super().__delattr__(name)

    def forward(self, *args, **kwargs):
        return self.classy_model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'Classy {type(self.classy_model)}:\n{self.classy_model.__repr__()}'

    @property
    def __class__(self):
        return self.classy_model.__class__


class ClassyModelHeadExecutorWrapper(ClassyModelWrapper):
    """Wrapper which changes the forward to also execute and return head output."""

    def forward(self, *args, **kwargs):
        out = self.classy_model(*args, **kwargs)
        if len(self._heads) == 0:
            return out
        head_outputs = self.execute_heads()
        if len(head_outputs) == 1:
            return list(head_outputs.values())[0]
        else:
            return head_outputs


class _ClassyModelMeta(type):
    """Metaclass to return a ClassyModel instance wrapped by a ClassyModelWrapper."""

    def __call__(cls, *args, **kwargs):
        """Override the __call__ function for the metaclass.

        This is called when a new instance of a class with this class as its metaclass
        is initialized. For example -

        .. code-block:: python
          class MyClass(metaclass=_ClassyModelMeta):
              wrapper_cls = MyWrapper

          my_class_instance = MyClass()  # returned instance will be a MyWrapper
        """
        classy_model = super().__call__(*args, **kwargs)
        wrapper_cls = cls.wrapper_cls
        if wrapper_cls is not None:
            classy_model = wrapper_cls(classy_model)
        return classy_model


class ClassyModel(nn.Module, metaclass=_ClassyModelMeta):
    """Base class for models in classy vision.

    A model refers either to a specific architecture (e.g. ResNet50) or a
    family of architectures (e.g. ResNet). Models can take arguments in the
    constructor in order to configure different behavior (e.g.
    hyperparameters).  Classy Models must implement :func:`from_config` in
    order to allow instantiation from a configuration file. Like regular
    PyTorch models, Classy Models must also implement :func:`forward`, where
    the bulk of the inference logic lives.

    Classy Models also have some advanced functionality for production
    fine-tuning systems. For example, we allow users to train a trunk
    model and then attach heads to the model via the attachable
    blocks.  Making your model support the trunk-heads paradigm is
    completely optional.

    NOTE: Advanced users can modify the behavior of their implemented models by
        specifying the `wrapper_cls` class attribute, which should be a class
        derived from :class:`ClassyModelWrapper` (see the documentation for that class
        for more information). Users can set it to `None` to skip wrapping their model
        and to make their model torchscriptable. This is set to
        :class:`ClassyModelHeadExecutorWrapper` by default.
    """
    wrapper_cls = ClassyModelHeadExecutorWrapper
    _attachable_block_names: List[str]
    __jit_unused_properties__ = ['attachable_block_names', 'head_outputs']

    def __init__(self):
        """Constructor for ClassyModel."""
        super().__init__()
        self._attachable_blocks = {}
        self._attachable_block_names = []
        self._heads = nn.ModuleDict()
        self._head_outputs = {}
        log_class_usage('Model', self.__class__)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ClassyModel':
        """Instantiates a ClassyModel from a configuration.

        Args:
            config: A configuration for the ClassyModel.

        Returns:
            A ClassyModel instance.
        """
        raise NotImplementedError

    @classmethod
    def from_model(cls, model: nn.Module, input_shape: Optional[Tuple]=None, model_depth: Optional[int]=None):
        """Converts an :class:`nn.Module` to a `ClassyModel`.

        Args:
            model: The model to convert
            For the remaining args, look at the corresponding properties of ClassyModel

        Returns:
            A ClassyModel instance.
        """
        return _ClassyModelAdapter(model, input_shape=input_shape, model_depth=model_depth)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = build_model(checkpoint['input_args']['config']['model'])
        model.set_classy_state(checkpoint['classy_state_dict']['base_model'])
        return model

    def get_classy_state(self, deep_copy=False):
        """Get the state of the ClassyModel.

        The returned state is used for checkpointing.

        NOTE: For advanced users, the structure of the returned dict is -
            `{"model": {"trunk": trunk_state, "heads": heads_state}}`.
            The trunk state is the state of the model when no heads are attached.

        Args:
            deep_copy: If True, creates a deep copy of the state Dict. Otherwise, the
                returned Dict's state will be tied to the object's.

        Returns:
            A state dictionary containing the state of the model.
        """
        attached_heads = self.get_heads()
        self.clear_heads()
        trunk_state_dict = self.state_dict()
        self.set_heads(attached_heads)
        head_state_dict = {}
        for block, heads in attached_heads.items():
            head_state_dict[block] = {head.unique_id: head.state_dict() for head in heads}
        model_state_dict = {'model': {'trunk': trunk_state_dict, 'heads': head_state_dict}}
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        return model_state_dict

    def load_head_states(self, state, strict=True):
        """Load only the state (weights) of the heads.

        For a trunk-heads model, this function allows the user to
        only update the head state of the model. Useful for attaching
        fine-tuned heads to a pre-trained trunk.

        Args:
            state (Dict): Contains the classy model state under key "model"

        """
        for block_name, head_states in state['model']['heads'].items():
            for head_name, head_state in head_states.items():
                self._heads[block_name][head_name].load_state_dict(head_state, strict)

    def set_classy_state(self, state, strict=True):
        """Set the state of the ClassyModel.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the model from a checkpoint.
        """
        self.load_head_states(state, strict)
        attached_heads = self.get_heads()
        self.clear_heads()
        self.load_state_dict(state['model']['trunk'], strict)
        self.set_heads(attached_heads)

    def forward(self, x):
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        raise NotImplementedError

    def extract_features(self, x):
        """
        Extract features from the model.

        Derived classes can implement this method to extract the features before
        applying the final fully connected layer.
        """
        return self.forward(x)

    def _build_attachable_block(self, name, module):
        """
        Add a wrapper to the module to allow to attach heads to the module.
        """
        if name in self._attachable_blocks:
            raise ValueError('Found duplicated block name {}'.format(name))
        block = ClassyBlock(name, module)
        self._attachable_blocks[name] = block
        self._attachable_block_names.append(name)
        return block

    @property
    def attachable_block_names(self):
        """
        Return names of all attachable blocks.
        """
        return self._attachable_block_names

    def clear_heads(self):
        self._heads.clear()
        self._head_outputs.clear()
        self._strip_classy_blocks(self)
        self._attachable_blocks = {}
        self._attachable_block_names = []

    def _strip_classy_blocks(self, module):
        for name, child_module in module.named_children():
            if isinstance(child_module, ClassyBlock):
                module.add_module(name, child_module.wrapped_module())
            self._strip_classy_blocks(child_module)

    def _make_module_attachable(self, module, module_name):
        found = False
        for name, child_module in module.named_children():
            if name == module_name:
                module.add_module(name, self._build_attachable_block(name, child_module))
                found = True
            found_in_child = self._make_module_attachable(child_module, module_name)
            found = found or found_in_child
        return found

    def set_heads(self, heads: Dict[str, List[ClassyHead]]):
        """Attach all the heads to corresponding blocks.

        A head is expected to be a ClassyHead object. For more
        details, see :class:`classy_vision.heads.ClassyHead`.

        Args:
            heads (Dict): a mapping between attachable block name
                and a list of heads attached to that block. For
                example, if you have two different teams that want to
                attach two different heads for downstream classifiers to
                the 15th block, then they would use:

                .. code-block:: python

                  heads = {"block15":
                      [classifier_head1, classifier_head2]
                  }
        """
        self.clear_heads()
        head_ids = set()
        for block_name, block_heads in heads.items():
            if not self._make_module_attachable(self, block_name):
                raise KeyError(f'{block_name} not found in the model')
            for head in block_heads:
                if head.unique_id in head_ids:
                    raise ValueError('head id {} already exists'.format(head.unique_id))
                head_ids.add(head.unique_id)
            self._heads[block_name] = nn.ModuleDict({head.unique_id: head for head in block_heads})

    def get_heads(self):
        """Returns the heads on the model

        Function returns the heads a dictionary of block names to
        `nn.Modules <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_
        attached to that block.

        """
        return {block_name: list(heads.values()) for block_name, heads in self._heads.items()}

    @property
    def head_outputs(self):
        """Return outputs of all heads in the format of Dict[head_id, output]

        Head outputs are cached during a forward pass.
        """
        return self._head_outputs.copy()

    def get_block_outputs(self) ->Dict[str, torch.Tensor]:
        outputs = {}
        for name, block in self._attachable_blocks.items():
            outputs[name] = block.output
        return outputs

    def execute_heads(self) ->Dict[str, torch.Tensor]:
        block_outs = self.get_block_outputs()
        outputs = {}
        for block_name, heads in self._heads.items():
            for head in heads.values():
                outputs[head.unique_id] = head(block_outs[block_name])
        self._head_outputs = outputs
        return outputs

    @property
    def input_shape(self):
        """Returns the input shape that the model can accept, excluding the batch dimension.

        By default it returns (3, 224, 224).
        """
        return 3, 224, 224


INPLACE = True


class _DenseLayer(nn.Sequential):
    """Single layer of a DenseNet."""

    def __init__(self, in_planes, growth_rate=32, expansion=4, use_se=False, se_reduction_ratio=16):
        assert is_pos_int(in_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)
        super(_DenseLayer, self).__init__()
        intermediate = expansion * growth_rate
        self.add_module('norm-1', nn.BatchNorm2d(in_planes))
        self.add_module('relu-1', nn.ReLU(inplace=INPLACE))
        self.add_module('conv-1', nn.Conv2d(in_planes, intermediate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm-2', nn.BatchNorm2d(intermediate))
        self.add_module('relu-2', nn.ReLU(inplace=INPLACE))
        self.add_module('conv-2', nn.Conv2d(intermediate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        if use_se:
            self.add_module('se', SqueezeAndExcitationLayer(growth_rate, reduction_ratio=se_reduction_ratio))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    """
    Transition layer to reduce spatial resolution.
    """

    def __init__(self, in_planes, out_planes, reduction=2):
        assert is_pos_int(in_planes)
        assert is_pos_int(out_planes)
        assert is_pos_int(reduction)
        super(_Transition, self).__init__()
        self.add_module('pool-norm', nn.BatchNorm2d(in_planes))
        self.add_module('pool-relu', nn.ReLU(inplace=INPLACE))
        self.add_module('pool-conv', nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        self.add_module('pool-pool', nn.AvgPool2d(kernel_size=reduction, stride=reduction))


MODEL_CLASS_NAMES = set()


MODEL_CLASS_NAMES_TB = {}


MODEL_REGISTRY = {}


MODEL_REGISTRY_TB = {}


def register_model(name, bypass_checks=False):
    """Registers a :class:`ClassyModel` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`ClassyModel` from a configuration file, even if the class itself is
    not part of the Classy Vision framework. To use it, apply this decorator to
    a ClassyModel subclass, like this:

    .. code-block:: python

      @register_model('resnet')
      class ResidualNet(ClassyModel):
         ...

    To instantiate a model from a configuration file, see
    :func:`build_model`."""

    def register_model_cls(cls):
        if not bypass_checks:
            if name in MODEL_REGISTRY:
                msg = 'Cannot register duplicate model ({}). Already registered at \n{}\n'
                raise ValueError(msg.format(name, MODEL_REGISTRY_TB[name]))
            if not issubclass(cls, ClassyModel):
                raise ValueError('Model ({}: {}) must extend ClassyModel'.format(name, cls.__name__))
            if cls.__name__ in MODEL_CLASS_NAMES:
                msg = 'Cannot register model with duplicate class name({}).' + 'Previously registered at \n{}\n'
                raise ValueError(msg.format(cls.__name__, MODEL_CLASS_NAMES_TB[cls.__name__]))
        tb = ''.join(traceback.format_stack())
        MODEL_REGISTRY[name] = cls
        MODEL_CLASS_NAMES.add(cls.__name__)
        MODEL_REGISTRY_TB[name] = tb
        MODEL_CLASS_NAMES_TB[cls.__name__] = tb
        return cls
    return register_model_cls


class DenseNet(ClassyModel):

    def __init__(self, num_blocks, num_classes, init_planes, growth_rate, expansion, small_input, final_bn_relu, use_se=False, se_reduction_ratio=16):
        """
        Implementation of a standard densely connected network (DenseNet).

        Contains the following attachable blocks:
            block{block_idx}-{idx}: This is the output of each dense block,
                indexed by the block index and the index of the dense layer
            transition-{idx}: This is the output of the transition layers
            trunk_output: The final output of the `DenseNet`. This is
                where a `fully_connected` head is normally attached.

        Args:
            small_input: set to `True` for 32x32 sized image inputs.
            final_bn_relu: set to `False` to exclude the final batchnorm and
                ReLU layers. These settings are useful when training Siamese
                networks.
            use_se: Enable squeeze and excitation
            se_reduction_ratio: The reduction ratio to apply in the excitation
                stage. Only used if `use_se` is `True`.
        """
        super().__init__()
        assert isinstance(num_blocks, Sequence)
        assert all(is_pos_int(b) for b in num_blocks)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(init_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)
        assert type(small_input) == bool
        self._num_classes = num_classes
        self.num_blocks = num_blocks
        self.small_input = small_input
        if self.small_input:
            self.initial_block = nn.Sequential(nn.Conv2d(3, init_planes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(init_planes), nn.ReLU(inplace=INPLACE))
        else:
            self.initial_block = nn.Sequential(nn.Conv2d(3, init_planes, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(init_planes), nn.ReLU(inplace=INPLACE), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        num_planes = init_planes
        blocks = nn.Sequential()
        for idx, num_layers in enumerate(num_blocks):
            block = self._make_dense_block(num_layers, num_planes, idx, growth_rate=growth_rate, expansion=expansion, use_se=use_se, se_reduction_ratio=se_reduction_ratio)
            blocks.add_module(f'block_{idx}', block)
            num_planes = num_planes + num_layers * growth_rate
            if idx != len(num_blocks) - 1:
                trans = _Transition(num_planes, num_planes // 2)
                blocks.add_module(f'transition-{idx}', trans)
                num_planes = num_planes // 2
        blocks.add_module('trunk_output', self._make_trunk_output_block(num_planes, final_bn_relu))
        self.features = blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_trunk_output_block(self, num_planes, final_bn_relu):
        layers = nn.Sequential()
        if final_bn_relu:
            layers.add_module('norm-final', nn.BatchNorm2d(num_planes))
            layers.add_module('relu-final', nn.ReLU(inplace=INPLACE))
        return layers

    def _make_dense_block(self, num_layers, in_planes, block_idx, growth_rate=32, expansion=4, use_se=False, se_reduction_ratio=16):
        assert is_pos_int(in_planes)
        assert is_pos_int(growth_rate)
        assert is_pos_int(expansion)
        layers = OrderedDict()
        for idx in range(num_layers):
            layers[f'block{block_idx}-{idx}'] = _DenseLayer(in_planes + idx * growth_rate, growth_rate=growth_rate, expansion=expansion, use_se=use_se, se_reduction_ratio=se_reduction_ratio)
        return nn.Sequential(layers)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'DenseNet':
        """Instantiates a DenseNet from a configuration.

        Args:
            config: A configuration for a DenseNet.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A DenseNet instance.
        """
        assert 'num_blocks' in config
        config = {'num_blocks': config['num_blocks'], 'num_classes': config.get('num_classes'), 'init_planes': config.get('init_planes', 64), 'growth_rate': config.get('growth_rate', 32), 'expansion': config.get('expansion', 4), 'small_input': config.get('small_input', False), 'final_bn_relu': config.get('final_bn_relu', True), 'use_se': config.get('use_se', False), 'se_reduction_ratio': config.get('se_reduction_ratio', 16)}
        return cls(**config)

    def forward(self, x):
        out = self.initial_block(x)
        out = self.features(out)
        return out


def drop_connect(inputs, is_training, drop_connect_rate):
    """
    Apply drop connect to random inputs in a batch.
    """
    if not is_training:
        return inputs
    keep_prob = 1 - drop_connect_rate
    batch_size = inputs.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    outputs = inputs / keep_prob * binary_tensor
    return outputs


def get_same_padding_for_kernel_size(kernel_size):
    """
    Returns the required padding for "same" style convolutions
    """
    if kernel_size % 2 == 0:
        raise ValueError(f'Only odd sized kernels are supported, got {kernel_size}')
    return (kernel_size - 1) // 2


def swish(x):
    """
    Swish activation function.
    """
    return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(self, input_filters: int, output_filters: int, expand_ratio: float, kernel_size: int, stride: int, se_ratio: float, id_skip: bool, use_se: bool, bn_momentum: float, bn_epsilon: float):
        assert se_ratio is None or 0 < se_ratio <= 1
        super().__init__()
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.has_se = use_se and se_ratio is not None
        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.id_skip = id_skip
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.relu_fn = swish if get_torch_version() < [1, 7] else nn.SiLU()
        self.depth = 0
        expanded_filters = input_filters * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=input_filters, out_channels=expanded_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn0 = nn.BatchNorm2d(num_features=expanded_filters, momentum=self.bn_momentum, eps=self.bn_epsilon)
            self.depth += 1
        self.depthwise_conv = nn.Conv2d(in_channels=expanded_filters, out_channels=expanded_filters, groups=expanded_filters, kernel_size=kernel_size, stride=stride, padding=get_same_padding_for_kernel_size(kernel_size), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=expanded_filters, momentum=self.bn_momentum, eps=self.bn_epsilon)
        self.depth += 1
        if self.has_se:
            num_reduced_filters = max(1, int(input_filters * se_ratio))
            self.se_reduce = nn.Conv2d(in_channels=expanded_filters, out_channels=num_reduced_filters, kernel_size=1, stride=1, padding=0, bias=True)
            self.se_expand = nn.Conv2d(in_channels=num_reduced_filters, out_channels=expanded_filters, kernel_size=1, stride=1, padding=0, bias=True)
            self.depth += 2
        self.project_conv = nn.Conv2d(in_channels=expanded_filters, out_channels=output_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=output_filters, momentum=self.bn_momentum, eps=self.bn_epsilon)
        self.depth += 1

    def forward(self, inputs, drop_connect_rate=None):
        if self.expand_ratio != 1:
            x = self.relu_fn(self.bn0(self.expand_conv(inputs)))
        else:
            x = inputs
        x = self.relu_fn(self.bn1(self.depthwise_conv(x)))
        if self.has_se:
            x_squeezed = self.se_avgpool(x)
            x_expanded = self.se_expand(self.relu_fn(self.se_reduce(x_squeezed)))
            x = torch.sigmoid(x_expanded) * x
        x = self.bn2(self.project_conv(x))
        if self.id_skip:
            if self.stride == 1 and self.input_filters == self.output_filters:
                if drop_connect_rate:
                    x = drop_connect(x, self.training, drop_connect_rate)
                x = x + inputs
        return x


class BlockParams(NamedTuple):
    num_repeat: int
    kernel_size: int
    stride: int
    expand_ratio: float
    input_filters: int
    output_filters: int
    se_ratio: float
    id_skip: bool


BLOCK_PARAMS = [BlockParams(1, 3, 1, 1, 32, 16, 0.25, True), BlockParams(2, 3, 2, 6, 16, 24, 0.25, True), BlockParams(2, 5, 2, 6, 24, 40, 0.25, True), BlockParams(3, 3, 2, 6, 40, 80, 0.25, True), BlockParams(3, 5, 1, 6, 80, 112, 0.25, True), BlockParams(4, 5, 2, 6, 112, 192, 0.25, True), BlockParams(1, 3, 1, 6, 192, 320, 0.25, True)]


class EfficientNetParams(NamedTuple):
    width_coefficient: float
    depth_coefficient: float
    resolution: int
    dropout_rate: float


MODEL_PARAMS = {'B0': EfficientNetParams(1.0, 1.0, 224, 0.2), 'B1': EfficientNetParams(1.0, 1.1, 240, 0.2), 'B2': EfficientNetParams(1.1, 1.2, 260, 0.3), 'B3': EfficientNetParams(1.2, 1.4, 300, 0.3), 'B4': EfficientNetParams(1.4, 1.8, 380, 0.4), 'B5': EfficientNetParams(1.6, 2.2, 456, 0.4), 'B6': EfficientNetParams(1.8, 2.6, 528, 0.5), 'B7': EfficientNetParams(2.0, 3.1, 600, 0.5)}


def scale_depth(num_repeats, depth_coefficient):
    """
    Calculates the scaled number of repeats based on the depth coefficient.
    """
    if not depth_coefficient:
        return num_repeats
    return int(math.ceil(depth_coefficient * num_repeats))


def scale_width(num_filters, width_coefficient, width_divisor, min_width):
    """
    Calculates the scaled number of filters based on the width coefficient and
    rounds the result by the width divisor.
    """
    if not width_coefficient:
        return num_filters
    num_filters *= width_coefficient
    min_width = min_width or width_divisor
    new_filters = max(min_width, int(num_filters + width_divisor / 2) // width_divisor * width_divisor)
    if new_filters < 0.9 * num_filters:
        new_filters += width_divisor
    return int(new_filters)


class EfficientNet(ClassyModel):
    """
    Implementation of EfficientNet, https://arxiv.org/pdf/1905.11946.pdf
    References:
        https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
        https://github.com/lukemelas/EfficientNet-PyTorch

    NOTE: the original implementation uses the names depth_divisor and min_depth
          to refer to the number of channels, which is confusing, since the paper
          refers to the channel dimension as width. We use the width_divisor and
          min_width names instead.
    """

    def __init__(self, num_classes: int, model_params: EfficientNetParams, bn_momentum: float, bn_epsilon: float, width_divisor: int, min_width: Optional[int], drop_connect_rate: float, use_se: bool):
        super().__init__()
        self.num_classes = num_classes
        self.image_resolution = model_params.resolution
        self.relu_fn = swish
        width_coefficient = model_params.width_coefficient
        depth_coefficient = model_params.depth_coefficient
        self.drop_connect_rate = drop_connect_rate
        in_channels = 3
        out_channels = 32
        out_channels = scale_width(out_channels, width_coefficient, width_divisor, min_width)
        self.conv_stem = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_momentum, eps=bn_epsilon)
        blocks = OrderedDict()
        for block_idx, block_params in enumerate(BLOCK_PARAMS):
            assert block_params.num_repeat > 0, 'num_repeat has to be > 0'
            block_params = block_params._replace(input_filters=scale_width(block_params.input_filters, width_coefficient, width_divisor, min_width), output_filters=scale_width(block_params.output_filters, width_coefficient, width_divisor, min_width), num_repeat=scale_depth(block_params.num_repeat, depth_coefficient))
            block_name = f'block{block_idx}-0'
            blocks[block_name] = MBConvBlock(block_params.input_filters, block_params.output_filters, block_params.expand_ratio, block_params.kernel_size, block_params.stride, block_params.se_ratio, block_params.id_skip, use_se, bn_momentum, bn_epsilon)
            if block_params.num_repeat > 1:
                block_params = block_params._replace(input_filters=block_params.output_filters, stride=1)
            for i in range(1, block_params.num_repeat):
                block_name = f'block{block_idx}-{i}'
                blocks[block_name] = MBConvBlock(block_params.input_filters, block_params.output_filters, block_params.expand_ratio, block_params.kernel_size, block_params.stride, block_params.se_ratio, block_params.id_skip, use_se, bn_momentum, bn_epsilon)
        self.blocks = nn.Sequential(blocks)
        in_channels = block_params.output_filters
        out_channels = 1280
        out_channels = scale_width(out_channels, width_coefficient, width_divisor, min_width)
        self.conv_head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_momentum, eps=bn_epsilon)
        self.trunk_output = nn.Identity()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)
        if model_params.dropout_rate > 0:
            self.dropout = nn.Dropout(p=model_params.dropout_rate)
        else:
            self.dropout = None
        self.init_weights()

    @classmethod
    def from_config(cls, config):
        """Instantiates an EfficientNet from a configuration.

        Args:
            config: A configuration for an EfficientNet.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt instance.
        """
        config = copy.deepcopy(config)
        del config['name']
        if 'heads' in config:
            del config['heads']
        if 'model_name' in config:
            assert config['model_name'] in MODEL_PARAMS, f"Unknown model_name: {config['model_name']}"
            model_params = MODEL_PARAMS[config['model_name']]
            del config['model_name']
        else:
            assert 'model_params' in config, 'Need either model_name or model_params'
            model_params = EfficientNetParams(**config['model_params'])
        config['model_params'] = model_params
        return cls(**config)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_height, kernel_width = module.kernel_size
                out_channels = module.out_channels
                fan_out = kernel_height * kernel_width * out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init_range = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        outputs = self.relu_fn(self.bn0(self.conv_stem(inputs)))
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            outputs = block(outputs, drop_connect_rate=drop_connect_rate)
        outputs = self.relu_fn(self.bn1(self.conv_head(outputs)))
        outputs = self.trunk_output(outputs)
        outputs = self.avg_pooling(outputs).view(outputs.size(0), -1)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs

    @property
    def input_shape(self):
        return 3, self.image_resolution, self.image_resolution


class _EfficientNet(EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(bn_momentum=0.01, bn_epsilon=0.001, drop_connect_rate=0.2, num_classes=1000, width_divisor=8, min_width=None, use_se=True, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'EfficientNet':
        config = copy.deepcopy(config)
        config.pop('name')
        if 'heads' in config:
            config.pop('heads')
        return cls(**config)


class EfficientNetB0(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B0'])


class EfficientNetB1(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B1'])


class EfficientNetB2(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B2'])


class EfficientNetB3(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B3'])


class EfficientNetB4(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B4'])


class EfficientNetB5(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B5'])


class EfficientNetB6(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B6'])


class EfficientNetB7(_EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(model_params=MODEL_PARAMS['B7'])


class MLP(ClassyModel):
    """MLP model using ReLU. Useful for testing on CPUs."""

    def __init__(self, input_dim, output_dim, hidden_dims, dropout, first_dropout, use_batchnorm, first_batchnorm):
        super().__init__()
        layers = []
        assert not first_batchnorm or use_batchnorm
        self._num_inputs = input_dim
        self._num_classes = output_dim
        self._model_depth = len(hidden_dims) + 1
        if dropout > 0 and first_dropout:
            layers.append(nn.Dropout(p=dropout))
        if use_batchnorm and first_batchnorm:
            layers.append(nn.BatchNorm1d(input_dim))
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'MLP':
        """Instantiates a MLP from a configuration.

        Args:
            config: A configuration for a MLP.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MLP instance.
        """
        assert (key in config for key in ['input_dim', 'output_dim', 'hidden_dims'])
        output_dim = config['output_dim']
        return cls(input_dim=config['input_dim'], output_dim=output_dim, hidden_dims=config['hidden_dims'], dropout=config.get('dropout', 0), first_dropout=config.get('first_dropout', False), use_batchnorm=config.get('use_batchnorm', False), first_batchnorm=config.get('first_batchnorm', False))

    def forward(self, x):
        batchsize_per_replica = x.shape[0]
        out = x.view(batchsize_per_replica, -1)
        out = self.mlp(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """helper function for constructing 1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def is_pos_int_tuple(t: Tuple) ->bool:
    """
    Returns True if a tuple contains positive integers
    """
    return type(t) == tuple and all(is_pos_int(n) for n in t)


class GenericLayer(nn.Module):
    """
    Parent class for 2-layer (BasicLayer) and 3-layer (BottleneckLayer)
    bottleneck layer class
    """

    def __init__(self, convolutional_block, in_planes, out_planes, stride=1, mid_planes_and_cardinality=None, reduction=4, final_bn_relu=True, use_se=False, se_reduction_ratio=16):
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(reduction)
        super(GenericLayer, self).__init__()
        self.convolutional_block = convolutional_block
        self.final_bn_relu = final_bn_relu
        if final_bn_relu:
            self.bn = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=INPLACE)
        self.downsample = None
        if stride != 1 and stride != (1, 1) or in_planes != out_planes:
            self.downsample = nn.Sequential(conv1x1(in_planes, out_planes, stride=stride), nn.BatchNorm2d(out_planes))
        self.se = SqueezeAndExcitationLayer(out_planes, reduction_ratio=se_reduction_ratio) if use_se else None

    def forward(self, x):
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        out = self.convolutional_block(x)
        if self.final_bn_relu:
            out = self.bn(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        if self.final_bn_relu:
            out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """helper function for constructing 3x3 grouped convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


class BasicLayer(GenericLayer):
    """
    ResNeXt layer with `in_planes` input planes and `out_planes`
    output planes.
    """

    def __init__(self, in_planes, out_planes, stride=1, mid_planes_and_cardinality=None, reduction=1, final_bn_relu=True, use_se=False, se_reduction_ratio=16):
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(reduction)
        convolutional_block = nn.Sequential(conv3x3(in_planes, out_planes, stride=stride), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=INPLACE), conv3x3(out_planes, out_planes))
        super().__init__(convolutional_block, in_planes, out_planes, stride=stride, reduction=reduction, final_bn_relu=final_bn_relu, use_se=use_se, se_reduction_ratio=se_reduction_ratio)


class BottleneckLayer(GenericLayer):
    """
    ResNeXt bottleneck layer with `in_planes` input planes, `out_planes`
    output planes, and a bottleneck `reduction`.
    """

    def __init__(self, in_planes, out_planes, stride=1, mid_planes_and_cardinality=None, reduction=4, final_bn_relu=True, use_se=False, se_reduction_ratio=16):
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(reduction)
        bottleneck_planes = int(math.ceil(out_planes / reduction))
        cardinality = 1
        if mid_planes_and_cardinality is not None:
            mid_planes, cardinality = mid_planes_and_cardinality
            bottleneck_planes = mid_planes * cardinality
        convolutional_block = nn.Sequential(conv1x1(in_planes, bottleneck_planes), nn.BatchNorm2d(bottleneck_planes), nn.ReLU(inplace=INPLACE), conv3x3(bottleneck_planes, bottleneck_planes, stride=stride, groups=cardinality), nn.BatchNorm2d(bottleneck_planes), nn.ReLU(inplace=INPLACE), conv1x1(bottleneck_planes, out_planes))
        super(BottleneckLayer, self).__init__(convolutional_block, in_planes, out_planes, stride=stride, reduction=reduction, final_bn_relu=final_bn_relu, use_se=use_se, se_reduction_ratio=se_reduction_ratio)


class SmallInputInitialBlock(nn.Module):
    """
    ResNeXt initial block for small input with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(conv3x3(3, init_planes, stride=1), nn.BatchNorm2d(init_planes), nn.ReLU(inplace=INPLACE))

    def forward(self, x):
        return self._module(x)


class InitialBlock(nn.Module):
    """
    ResNeXt initial block with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(nn.Conv2d(3, init_planes, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(init_planes), nn.ReLU(inplace=INPLACE), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self._module(x)


VERSION = 0.2


class ResNeXt(ClassyModel):
    __jit_unused_properties__ = ClassyModel.__jit_unused_properties__ + ['model_depth']

    def __init__(self, num_blocks, init_planes: int=64, reduction: int=4, small_input: bool=False, zero_init_bn_residuals: bool=False, base_width_and_cardinality: Optional[Sequence]=None, basic_layer: bool=False, final_bn_relu: bool=True, use_se: bool=False, se_reduction_ratio: int=16):
        """
        Implementation of `ResNeXt <https://arxiv.org/pdf/1611.05431.pdf>`_.

        Args:
            small_input: set to `True` for 32x32 sized image inputs.
            final_bn_relu: set to `False` to exclude the final batchnorm and
                ReLU layers. These settings are useful when training Siamese
                networks.
            use_se: Enable squeeze and excitation
            se_reduction_ratio: The reduction ratio to apply in the excitation
                stage. Only used if `use_se` is `True`.
        """
        super().__init__()
        assert isinstance(num_blocks, Sequence)
        assert all(is_pos_int(n) for n in num_blocks)
        assert is_pos_int(init_planes) and is_pos_int(reduction)
        assert type(small_input) == bool
        assert type(zero_init_bn_residuals) == bool, 'zero_init_bn_residuals must be a boolean, set to true if gamma of last             BN of residual block should be initialized to 0.0, false for 1.0'
        assert base_width_and_cardinality is None or isinstance(base_width_and_cardinality, Sequence) and len(base_width_and_cardinality) == 2 and is_pos_int(base_width_and_cardinality[0]) and is_pos_int(base_width_and_cardinality[1])
        assert isinstance(use_se, bool), 'use_se has to be a boolean'
        self.num_blocks = num_blocks
        self.small_input = small_input
        self._make_initial_block(small_input, init_planes, basic_layer)
        out_planes = [(init_planes * 2 ** i * reduction) for i in range(len(num_blocks))]
        in_planes = [init_planes] + out_planes[:-1]
        blocks = []
        for idx in range(len(out_planes)):
            mid_planes_and_cardinality = None
            if base_width_and_cardinality is not None:
                w, c = base_width_and_cardinality
                mid_planes_and_cardinality = w * 2 ** idx, c
            new_block = self._make_resolution_block(in_planes[idx], out_planes[idx], idx, num_blocks[idx], stride=1 if idx == 0 else 2, mid_planes_and_cardinality=mid_planes_and_cardinality, reduction=reduction, final_bn_relu=final_bn_relu or idx != len(out_planes) - 1, use_se=use_se, se_reduction_ratio=se_reduction_ratio)
            blocks.append(new_block)
        self.blocks = nn.Sequential(*blocks)
        self.out_planes = out_planes[-1]
        self._num_classes = out_planes
        self._initialize_weights(zero_init_bn_residuals)

    def _initialize_weights(self, zero_init_bn_residuals):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_bn_residuals:
            for m in self.modules():
                if isinstance(m, GenericLayer):
                    if hasattr(m, 'bn'):
                        nn.init.constant_(m.bn.weight, 0)

    def _make_initial_block(self, small_input, init_planes, basic_layer):
        if small_input:
            self.initial_block = SmallInputInitialBlock(init_planes)
            self.layer_type = BasicLayer
        else:
            self.initial_block = InitialBlock(init_planes)
            self.layer_type = BasicLayer if basic_layer else BottleneckLayer

    def _make_resolution_block(self, in_planes, out_planes, resolution_idx, num_blocks, stride=1, mid_planes_and_cardinality=None, reduction=4, final_bn_relu=True, use_se=False, se_reduction_ratio=16):
        blocks = OrderedDict()
        for idx in range(num_blocks):
            block_name = 'block{}-{}'.format(resolution_idx, idx)
            blocks[block_name] = self.layer_type(in_planes if idx == 0 else out_planes, out_planes, stride=stride if idx == 0 else 1, mid_planes_and_cardinality=mid_planes_and_cardinality, reduction=reduction, final_bn_relu=final_bn_relu or idx != num_blocks - 1, use_se=use_se, se_reduction_ratio=se_reduction_ratio)
        return nn.Sequential(blocks)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ResNeXt':
        """Instantiates a ResNeXt from a configuration.

        Args:
            config: A configuration for a ResNeXt.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt instance.
        """
        assert 'num_blocks' in config
        basic_layer = config.get('basic_layer', False)
        config = {'num_blocks': config['num_blocks'], 'init_planes': config.get('init_planes', 64), 'reduction': config.get('reduction', 1 if basic_layer else 4), 'base_width_and_cardinality': config.get('base_width_and_cardinality'), 'small_input': config.get('small_input', False), 'basic_layer': basic_layer, 'final_bn_relu': config.get('final_bn_relu', True), 'zero_init_bn_residuals': config.get('zero_init_bn_residuals', False), 'use_se': config.get('use_se', False), 'se_reduction_ratio': config.get('se_reduction_ratio', 16)}
        return cls(**config)

    def forward(self, x):
        out = self.initial_block(x)
        out = self.blocks(out)
        return out

    def _convert_model_state(self, state):
        """Convert model state from the old implementation to the current format.

        Updates the state dict in place and returns True if the state dict was updated.
        """
        pattern = 'blocks\\.(?P<block_id_0>[0-9]*)\\.(?P<block_id_1>[0-9]*)\\._module\\.'
        repl = 'blocks.\\g<block_id_0>.block\\g<block_id_0>-\\g<block_id_1>.'
        trunk_dict = state['model']['trunk']
        new_trunk_dict = {}
        replaced_keys = False
        for key, value in trunk_dict.items():
            new_key = re.sub(pattern, repl, key)
            if new_key != key:
                replaced_keys = True
            new_trunk_dict[new_key] = value
        state['model']['trunk'] = new_trunk_dict
        state['version'] = VERSION
        return replaced_keys

    def get_classy_state(self, deep_copy=False):
        state = super().get_classy_state(deep_copy=deep_copy)
        state['version'] = VERSION
        return state

    def set_classy_state(self, state, strict=True):
        version = state.get('version')
        if version is None:
            if not self._convert_model_state(state):
                raise RuntimeError('ResNeXt state conversion failed')
            message = 'Provided state dict is from an old implementation of ResNeXt. This has been deprecated and will be removed soon.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        elif version != VERSION:
            raise ValueError(f'Unsupported ResNeXt version: {version}. Expected: {VERSION}')
        super().set_classy_state(state, strict)


class _ResNeXt(ResNeXt):

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ResNeXt':
        config = copy.deepcopy(config)
        config.pop('name')
        if 'heads' in config:
            config.pop('heads')
        return cls(**config)


class ResNet18(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[2, 2, 2, 2], basic_layer=True, zero_init_bn_residuals=True, reduction=1, **kwargs)


class ResNet34(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 4, 6, 3], basic_layer=True, zero_init_bn_residuals=True, reduction=1, **kwargs)


class ResNet50(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 4, 6, 3], basic_layer=False, zero_init_bn_residuals=True, **kwargs)


class ResNet101(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 4, 23, 3], basic_layer=False, zero_init_bn_residuals=True, **kwargs)


class ResNet152(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 8, 36, 3], basic_layer=False, zero_init_bn_residuals=True, **kwargs)


class ResNeXt50(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 4, 6, 3], basic_layer=False, zero_init_bn_residuals=True, base_width_and_cardinality=(4, 32), **kwargs)


class ResNeXt101(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 4, 23, 3], basic_layer=False, zero_init_bn_residuals=True, base_width_and_cardinality=(4, 32), **kwargs)


class ResNeXt152(_ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(num_blocks=[3, 8, 36, 3], basic_layer=False, zero_init_bn_residuals=True, base_width_and_cardinality=(4, 32), **kwargs)


def is_pos_int_list(l: List) ->bool:
    """
    Returns True if a list contains positive integers
    """
    return type(l) == list and all(is_pos_int(n) for n in l)


class ResNeXt3DStemSinglePathway(nn.Module):
    """
    ResNe(X)t 3D basic stem module. Assume a single pathway.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, maxpool=True, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1):
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
        self.conv = nn.Conv3d(dim_in, dim_out, self.kernel, stride=self.stride, padding=self.padding, bias=False)
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.pool_layer(x)
        return x


def r2plus1_unit(dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu, bn_eps, bn_mmt, dim_mid=None):
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
        dim_mid = int(dim_out * dim_in * 3 * 3 * 3 / (dim_in * 3 * 3 + dim_out * 3))
        logging.info('dim_in: %d, dim_out: %d. Set dim_mid to %d' % (dim_in, dim_out, dim_mid))
    conv_middle = nn.Conv3d(dim_in, dim_mid, [1, 3, 3], stride=[1, spatial_stride, spatial_stride], padding=[0, 1, 1], groups=groups, bias=False)
    conv_middle_bn = nn.BatchNorm3d(dim_mid, eps=bn_eps, momentum=bn_mmt)
    conv_middle_relu = nn.ReLU(inplace=inplace_relu)
    conv = nn.Conv3d(dim_mid, dim_out, [3, 1, 1], stride=[temporal_stride, 1, 1], padding=[1, 0, 0], groups=groups, bias=False)
    return nn.Sequential(conv_middle, conv_middle_bn, conv_middle_relu, conv)


class R2Plus1DStemSinglePathway(ResNeXt3DStemSinglePathway):
    """
    R(2+1)D basic stem module. Assume a single pathway.
    Performs spatial convolution, temporal convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, maxpool=True, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1):
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
        super(R2Plus1DStemSinglePathway, self).__init__(dim_in, dim_out, kernel, stride, padding, maxpool=maxpool, inplace_relu=inplace_relu, bn_eps=bn_eps, bn_mmt=bn_mmt)

    def _construct_stem(self, dim_in, dim_out):
        assert self.stride[1] == self.stride[2], 'Only support identical height stride and width stride'
        self.conv = r2plus1_unit(dim_in, dim_out, self.stride[0], self.stride[1], 1, self.inplace_relu, self.bn_eps, self.bn_mmt, dim_mid=45)
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])


class ResNeXt3DStemMultiPathway(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, maxpool=(True,)):
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
        assert len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)}) == 1, 'Input pathway dimensions are not consistent.'
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
        assert isinstance(dim_in, Sequence)
        assert all(dim > 0 for dim in dim_in)
        assert isinstance(dim_out, Sequence)
        assert all(dim > 0 for dim in dim_out)
        self.blocks = {}
        for p in range(len(dim_in)):
            stem = ResNeXt3DStemSinglePathway(dim_in[p], dim_out[p], self.kernel[p], self.stride[p], self.padding[p], inplace_relu=self.inplace_relu, bn_eps=self.bn_eps, bn_mmt=self.bn_mmt, maxpool=self.maxpool[p])
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem

    def _stem_name(self, path_idx):
        return 'stem-path{}'.format(path_idx)

    def forward(self, x):
        assert len(x) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(self.num_pathways)
        for p in range(len(x)):
            stem_name = self._stem_name(p)
            x[p] = self.blocks[stem_name](x[p])
        return x


class R2Plus1DStemMultiPathway(ResNeXt3DStemMultiPathway):
    """
    Video R(2+1)D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, maxpool=(True,)):
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
        super(R2Plus1DStemMultiPathway, self).__init__(dim_in, dim_out, kernel, stride, padding, inplace_relu=inplace_relu, bn_eps=bn_eps, bn_mmt=bn_mmt, maxpool=maxpool)

    def _construct_stem(self, dim_in, dim_out):
        assert isinstance(dim_in, Sequence)
        assert all(dim > 0 for dim in dim_in)
        assert isinstance(dim_out, Sequence)
        assert all(dim > 0 for dim in dim_out)
        self.blocks = {}
        for p in range(len(dim_in)):
            stem = R2Plus1DStemSinglePathway(dim_in[p], dim_out[p], self.kernel[p], self.stride[p], self.padding[p], inplace_relu=self.inplace_relu, bn_eps=self.bn_eps, bn_mmt=self.bn_mmt, maxpool=self.maxpool[p])
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem


class ResNeXt3DStem(nn.Module):

    def __init__(self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool):
        super(ResNeXt3DStem, self).__init__()
        self._construct_stem(temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool)

    def _construct_stem(self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool):
        self.stem = ResNeXt3DStemMultiPathway([input_planes], [stem_planes], [[temporal_kernel, spatial_kernel, spatial_kernel]], [[1, 2, 2]], [[temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]], maxpool=[maxpool])

    def forward(self, x):
        return self.stem(x)


class R2Plus1DStem(ResNeXt3DStem):

    def __init__(self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool):
        super(R2Plus1DStem, self).__init__(temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool)

    def _construct_stem(self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool):
        self.stem = R2Plus1DStemMultiPathway([input_planes], [stem_planes], [[temporal_kernel, spatial_kernel, spatial_kernel]], [[1, 2, 2]], [[temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]], maxpool=[maxpool])


model_stems = {'r2plus1d_stem': R2Plus1DStem, 'resnext3d_stem': ResNeXt3DStem}


class ResNeXt3DBase(ClassyModel):

    def __init__(self, input_key, input_planes, clip_crop_size, frames_per_clip, num_blocks, stem_name, stem_planes, stem_temporal_kernel, stem_spatial_kernel, stem_maxpool):
        """
        ResNeXt3DBase implements everything in ResNeXt3D model except the
        construction of 4 stages. See more details in ResNeXt3D.
        """
        super(ResNeXt3DBase, self).__init__()
        self._input_key = input_key
        self.input_planes = input_planes
        self.clip_crop_size = clip_crop_size
        self.frames_per_clip = frames_per_clip
        self.num_blocks = num_blocks
        assert stem_name in model_stems, 'unknown stem: %s' % stem_name
        self.stem = model_stems[stem_name](stem_temporal_kernel, stem_spatial_kernel, input_planes, stem_planes, stem_maxpool)

    @staticmethod
    def _parse_config(config):
        ret_config = {}
        required_args = ['input_planes', 'clip_crop_size', 'skip_transformation_type', 'residual_transformation_type', 'frames_per_clip', 'num_blocks']
        for arg in required_args:
            assert arg in config, 'resnext3d model requires argument %s' % arg
            ret_config[arg] = config[arg]
        ret_config.update({'input_key': config.get('input_key', None), 'stem_name': config.get('stem_name', 'resnext3d_stem'), 'stem_planes': config.get('stem_planes', 64), 'stem_temporal_kernel': config.get('stem_temporal_kernel', 3), 'stem_spatial_kernel': config.get('stem_spatial_kernel', 7), 'stem_maxpool': config.get('stem_maxpool', False)})
        ret_config.update({'stage_planes': config.get('stage_planes', 256), 'stage_temporal_kernel_basis': config.get('stage_temporal_kernel_basis', [[3], [3], [3], [3]]), 'temporal_conv_1x1': config.get('temporal_conv_1x1', [False, False, False, False]), 'stage_temporal_stride': config.get('stage_temporal_stride', [1, 2, 2, 2]), 'stage_spatial_stride': config.get('stage_spatial_stride', [1, 2, 2, 2]), 'num_groups': config.get('num_groups', 1), 'width_per_group': config.get('width_per_group', 64)})
        ret_config.update({'zero_init_residual_transform': config.get('zero_init_residual_transform', False)})
        assert is_pos_int_list(ret_config['num_blocks'])
        assert is_pos_int(ret_config['stem_planes'])
        assert is_pos_int(ret_config['stem_temporal_kernel'])
        assert is_pos_int(ret_config['stem_spatial_kernel'])
        assert type(ret_config['stem_maxpool']) == bool
        assert is_pos_int(ret_config['stage_planes'])
        assert isinstance(ret_config['stage_temporal_kernel_basis'], Sequence)
        assert all(is_pos_int_list(l) for l in ret_config['stage_temporal_kernel_basis'])
        assert isinstance(ret_config['temporal_conv_1x1'], Sequence)
        assert is_pos_int_list(ret_config['stage_temporal_stride'])
        assert is_pos_int_list(ret_config['stage_spatial_stride'])
        assert is_pos_int(ret_config['num_groups'])
        assert is_pos_int(ret_config['width_per_group'])
        return ret_config

    def _init_parameter(self, zero_init_residual_transform):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if hasattr(m, 'final_transform_op') and m.final_transform_op and zero_init_residual_transform:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if hasattr(m, 'final_transform_op') and m.final_transform_op and zero_init_residual_transform:
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def set_classy_state(self, state, strict=True):
        self.load_head_states(state)
        attached_heads = self.get_heads()
        self.clear_heads()
        current_state = self.state_dict()
        for name, weight_src in state['model']['trunk'].items():
            if name not in current_state:
                logging.warning(f'weight {name} is not found in current ResNeXt3D state')
                continue
            weight_tgt = current_state[name]
            assert weight_src.dim() == weight_tgt.dim(), 'weight of source- and target 3D convolution should have same dimension'
            if weight_src.dim() == 5 and weight_src.shape[2] == 1 and weight_tgt.shape[2] > 1:
                assert weight_src.shape[-2:] == weight_tgt.shape[-2:] and weight_src.shape[:2] == weight_tgt.shape[:2], 'weight shapes of source- and target 3D convolution mismatch'
                weight_src_inflated = weight_src.repeat(1, 1, weight_tgt.shape[2], 1, 1) / weight_tgt.shape[2]
                weight_src = weight_src_inflated
            else:
                assert all(weight_src.size(d) == weight_tgt.size(d) for d in range(weight_src.dim())), 'the shapes of source and target weight mismatch: %s Vs %s' % (str(weight_src.size()), str(weight_tgt.size()))
            current_state[name] = weight_src.clone()
        self.load_state_dict(current_state, strict=strict)
        self.set_heads(attached_heads)

    def forward(self, x):
        """
        Args:
            x (dict or torch.Tensor): video input.
                When its type is dict, the dataset is a video dataset, and its
                content is like {"video": torch.tensor, "audio": torch.tensor}.
                When its type is torch.Tensor, the dataset is an image dataset.
        """
        assert isinstance(x, dict) or isinstance(x, torch.Tensor), 'x must be either a dictionary or a torch.Tensor'
        if isinstance(x, dict):
            assert self._input_key is not None and self._input_key in x, 'input key (%s) not in the input' % self._input_key
            x = x[self._input_key]
        else:
            assert self._input_key is None, 'when input of forward pass is a tensor, input key should not be set'
            assert x.dim() == 4 or x.dim() == 5, 'tensor x must be 4D/5D tensor'
            if x.dim() == 4:
                x = torch.unsqueeze(x, 2)
        out = self.stem([x])
        out = self.stages(out)
        return out

    @property
    def input_shape(self):
        """
        Shape of video model input can vary in the following cases
        - At training stage, input are video frame croppings of fixed size.
        - At test stage, input are original video frames to support Fully Convolutional
            evaluation and its size can vary video by video
        """
        return self.input_planes, self.frames_per_clip, self.clip_crop_size, self.clip_crop_size

    @property
    def input_key(self):
        return self._input_key


class BasicTransformation(nn.Module):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
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
        self._construct_model(dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu, bn_eps, bn_mmt)

    def _construct_model(self, dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu, bn_eps, bn_mmt):
        branch2a = nn.Conv3d(dim_in, dim_out, [3, 3, 3], stride=[temporal_stride, spatial_stride, spatial_stride], padding=[1, 1, 1], groups=groups, bias=False)
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        branch2b = nn.Conv3d(dim_out, dim_out, [3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], groups=groups, bias=False)
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True
        self.transform = nn.Sequential(branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn)

    def forward(self, x):
        return self.transform(x)


class BasicR2Plus1DTransformation(BasicTransformation):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
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
        super(BasicR2Plus1DTransformation, self).__init__(dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu=inplace_relu, bn_eps=bn_eps, bn_mmt=bn_mmt)

    def _construct_model(self, dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu, bn_eps, bn_mmt):
        branch2a = r2plus1_unit(dim_in, dim_out, temporal_stride, spatial_stride, groups, inplace_relu, bn_eps, bn_mmt)
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        branch2b = r2plus1_unit(dim_out, dim_out, 1, 1, groups, inplace_relu, bn_eps, bn_mmt)
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True
        self.transform = nn.Sequential(branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn)


class PostactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, num_groups, dim_inner, temporal_kernel_size=3, temporal_conv_1x1=True, spatial_stride_1x1=False, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
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
        temporal_kernel_size_1x1, temporal_kernel_size_3x3 = (temporal_kernel_size, 1) if temporal_conv_1x1 else (1, temporal_kernel_size)
        str1x1, str3x3 = (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        self.branch2a = nn.Conv3d(dim_in, dim_inner, kernel_size=[temporal_kernel_size_1x1, 1, 1], stride=[1, str1x1, str1x1], padding=[temporal_kernel_size_1x1 // 2, 0, 0], bias=False)
        self.branch2a_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(dim_inner, dim_inner, [temporal_kernel_size_3x3, 3, 3], stride=[temporal_stride, str3x3, str3x3], padding=[temporal_kernel_size_3x3 // 2, 1, 1], groups=num_groups, bias=False)
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
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

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, num_groups, dim_inner, temporal_kernel_size=3, temporal_conv_1x1=True, spatial_stride_1x1=False, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, disable_pre_activation=False, **kwargs):
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
        temporal_kernel_size_1x1, temporal_kernel_size_3x3 = (temporal_kernel_size, 1) if temporal_conv_1x1 else (1, temporal_kernel_size)
        str1x1, str3x3 = (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        self.disable_pre_activation = disable_pre_activation
        if not disable_pre_activation:
            self.branch2a_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2a = nn.Conv3d(dim_in, dim_inner, kernel_size=[temporal_kernel_size_1x1, 1, 1], stride=[1, str1x1, str1x1], padding=[temporal_kernel_size_1x1 // 2, 0, 0], bias=False)
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(dim_inner, dim_inner, [temporal_kernel_size_3x3, 3, 3], stride=[temporal_stride, str3x3, str3x3], padding=[temporal_kernel_size_3x3 // 2, 1, 1], groups=num_groups, bias=False)
        self.branch2c_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
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


residual_transformations = {'basic_r2plus1d_transformation': BasicR2Plus1DTransformation, 'basic_transformation': BasicTransformation, 'postactivated_bottleneck_transformation': PostactivatedBottleneckTransformation, 'preactivated_bottleneck_transformation': PreactivatedBottleneckTransformation}


class PostactivatedShortcutTransformation(nn.Module):
    """
    Skip connection used in ResNet3D model.
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, bn_eps=1e-05, bn_mmt=0.1, **kwargs):
        super(PostactivatedShortcutTransformation, self).__init__()
        assert dim_in != dim_out or spatial_stride != 1 or temporal_stride != 1
        self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[temporal_stride, spatial_stride, spatial_stride], padding=0, bias=False)
        self.branch1_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)

    def forward(self, x):
        return self.branch1_bn(self.branch1(x))


class PreactivatedShortcutTransformation(nn.Module):
    """
    Skip connection with pre-activation, which includes BatchNorm3D and ReLU,
        in ResNet3D model (https://arxiv.org/abs/1603.05027).
    """

    def __init__(self, dim_in, dim_out, temporal_stride, spatial_stride, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, disable_pre_activation=False, **kwargs):
        super(PreactivatedShortcutTransformation, self).__init__()
        assert dim_in != dim_out or spatial_stride != 1 or temporal_stride != 1
        if not disable_pre_activation:
            self.branch1_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch1_relu = nn.ReLU(inplace=inplace_relu)
        self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[temporal_stride, spatial_stride, spatial_stride], padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'branch1_bn') and hasattr(self, 'branch1_relu'):
            x = self.branch1_relu(self.branch1_bn(x))
        x = self.branch1(x)
        return x


skip_transformations = {'postactivated_shortcut': PostactivatedShortcutTransformation, 'preactivated_shortcut': PreactivatedShortcutTransformation}


class ResBlock(nn.Module):
    """
    Residual block with skip connection.
    """

    def __init__(self, dim_in, dim_out, dim_inner, temporal_kernel_size, temporal_conv_1x1, temporal_stride, spatial_stride, skip_transformation_type, residual_transformation_type, num_groups=1, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, disable_pre_activation=False):
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
            self.skip = skip_transformations[skip_transformation_type](dim_in, dim_out, temporal_stride, spatial_stride, bn_eps=bn_eps, bn_mmt=bn_mmt, disable_pre_activation=disable_pre_activation)
        assert residual_transformation_type in residual_transformations, 'unknown residual transformation: %s' % residual_transformation_type
        self.residual = residual_transformations[residual_transformation_type](dim_in, dim_out, temporal_stride, spatial_stride, num_groups, dim_inner, temporal_kernel_size=temporal_kernel_size, temporal_conv_1x1=temporal_conv_1x1, disable_pre_activation=disable_pre_activation)
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        if hasattr(self, 'skip'):
            x = self.skip(x) + self.residual(x)
        else:
            x = x + self.residual(x)
        x = self.relu(x)
        return x


class ResStageBase(nn.Module):

    def __init__(self, stage_idx, dim_in, dim_out, dim_inner, temporal_kernel_basis, temporal_conv_1x1, temporal_stride, spatial_stride, num_blocks, num_groups):
        super(ResStageBase, self).__init__()
        assert len({len(dim_in), len(dim_out), len(temporal_kernel_basis), len(temporal_conv_1x1), len(temporal_stride), len(spatial_stride), len(num_blocks), len(dim_inner), len(num_groups)}) == 1
        self.stage_idx = stage_idx
        self.num_blocks = num_blocks
        self.num_pathways = len(self.num_blocks)
        self.temporal_kernel_sizes = [(temporal_kernel_basis[i] * num_blocks[i])[:num_blocks[i]] for i in range(len(temporal_kernel_basis))]

    def _block_name(self, pathway_idx, stage_idx, block_idx):
        return 'pathway{}-stage{}-block{}'.format(pathway_idx, stage_idx, block_idx)

    def _pathway_name(self, pathway_idx):
        return 'pathway{}'.format(pathway_idx)

    def forward(self, inputs):
        output = []
        for p in range(self.num_pathways):
            x = inputs[p]
            pathway_module = getattr(self, self._pathway_name(p))
            output.append(pathway_module(x))
        return output


class ResStage(ResStageBase):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, SlowOnly), and multi-pathway (SlowFast) cases.
        More details can be found here:
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, stage_idx, dim_in, dim_out, dim_inner, temporal_kernel_basis, temporal_conv_1x1, temporal_stride, spatial_stride, num_blocks, num_groups, skip_transformation_type, residual_transformation_type, inplace_relu=True, bn_eps=1e-05, bn_mmt=0.1, disable_pre_activation=False, final_stage=False):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            stage_idx (int): integer index of stage.
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_inner (list): list of the p inner channel dimensions of the
                input.
                Different channel dimensions control the input dimension of
                different pathways.
            temporal_kernel_basis (list): Basis of temporal kernel sizes for each of
                the stage.
            temporal_conv_1x1 (list): Only useful for BottleneckBlock.
                In a pathaway, if True, do temporal convolution in the fist 1x1 Conv3d.
                Otherwise, do it in the second 3x3 Conv3d
            temporal_stride (list): the temporal stride of the bottleneck.
            spatial_stride (list): the spatial_stride of the bottleneck.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            skip_transformation_type (str): the type of skip transformation
            residual_transformation_type (str): the type of residual transformation
            disable_pre_activation (bool): If true, disable the preactivation,
                which includes BatchNorm3D and ReLU.
            final_stage (bool): If true, this is the last stage in the model.
        """
        super(ResStage, self).__init__(stage_idx, dim_in, dim_out, dim_inner, temporal_kernel_basis, temporal_conv_1x1, temporal_stride, spatial_stride, num_blocks, num_groups)
        for p in range(self.num_pathways):
            blocks = []
            for i in range(self.num_blocks[p]):
                block_disable_pre_activation = True if disable_pre_activation and i == 0 else False
                res_block = ResBlock(dim_in[p] if i == 0 else dim_out[p], dim_out[p], dim_inner[p], self.temporal_kernel_sizes[p][i], temporal_conv_1x1[p], temporal_stride[p] if i == 0 else 1, spatial_stride[p] if i == 0 else 1, skip_transformation_type, residual_transformation_type, num_groups=num_groups[p], inplace_relu=inplace_relu, bn_eps=bn_eps, bn_mmt=bn_mmt, disable_pre_activation=block_disable_pre_activation)
                block_name = self._block_name(p, stage_idx, i)
                blocks.append((block_name, res_block))
            if final_stage and residual_transformation_type == 'preactivated_bottleneck_transformation':
                activate_bn = nn.BatchNorm3d(dim_out[p])
                activate_relu = nn.ReLU(inplace=True)
                activate_bn_name = '-'.join([block_name, 'bn'])
                activate_relu_name = '-'.join([block_name, 'relu'])
                blocks.append((activate_bn_name, activate_bn))
                blocks.append((activate_relu_name, activate_relu))
            self.add_module(self._pathway_name(p), nn.Sequential(OrderedDict(blocks)))


class ResNeXt3D(ResNeXt3DBase):
    """
    Implementation of:
        1. Conventional `post-activated 3D ResNe(X)t <https://arxiv.org/
        abs/1812.03982>`_.

        2. `Pre-activated 3D ResNe(X)t <https://arxiv.org/abs/1811.12814>`_.
        The model consists of one stem, a number of stages, and one or multiple
        heads that are attached to different blocks in the stage.
    """

    def __init__(self, input_key, input_planes, clip_crop_size, skip_transformation_type, residual_transformation_type, frames_per_clip, num_blocks, stem_name, stem_planes, stem_temporal_kernel, stem_spatial_kernel, stem_maxpool, stage_planes, stage_temporal_kernel_basis, temporal_conv_1x1, stage_temporal_stride, stage_spatial_stride, num_groups, width_per_group, zero_init_residual_transform):
        """
        Args:
            input_key (str): a key that can index into model input that is
                of dict type.
            input_planes (int): the channel dimension of the input. Normally 3 is used
                for rgb input.
            clip_crop_size (int): spatial cropping size of video clip at train time.
            skip_transformation_type (str): the type of skip transformation.
            residual_transformation_type (str): the type of residual transformation.
            frames_per_clip (int): Number of frames in a video clip.
            num_blocks (list): list of the number of blocks in stages.
            stem_name (str): name of model stem.
            stem_planes (int): the output dimension of the convolution in the model
                stem.
            stem_temporal_kernel (int): the temporal kernel size of the convolution
                in the model stem.
            stem_spatial_kernel (int): the spatial kernel size of the convolution
                in the model stem.
            stem_maxpool (bool): If true, perform max pooling.
            stage_planes (int): the output channel dimension of the 1st residual stage
            stage_temporal_kernel_basis (list): Basis of temporal kernel sizes for
                each of the stage.
            temporal_conv_1x1 (bool): Only useful for BottleneckTransformation.
                In a pathaway, if True, do temporal convolution in the first 1x1
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
                transformation or Conv3D in pre-activated transformation, in the
                residual transformation is initialized to zero
        """
        super(ResNeXt3D, self).__init__(input_key, input_planes, clip_crop_size, frames_per_clip, num_blocks, stem_name, stem_planes, stem_temporal_kernel, stem_spatial_kernel, stem_maxpool)
        num_stages = len(num_blocks)
        out_planes = [(stage_planes * 2 ** i) for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [(num_groups * width_per_group * 2 ** i) for i in range(num_stages)]
        stages = []
        for s in range(num_stages):
            stage = ResStage(s + 1, [in_planes[s]], [out_planes[s]], [inner_planes[s]], [stage_temporal_kernel_basis[s]], [temporal_conv_1x1[s]], [stage_temporal_stride[s]], [stage_spatial_stride[s]], [num_blocks[s]], [num_groups], skip_transformation_type, residual_transformation_type, disable_pre_activation=s == 0, final_stage=s == num_stages - 1)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_residual_transform)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'ResNeXt3D':
        """Instantiates a ResNeXt3D from a configuration.

        Args:
            config: A configuration for a ResNeXt3D.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A ResNeXt3D instance.
        """
        ret_config = ResNeXt3D._parse_config(config)
        return cls(**ret_config)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-06)
        nn.init.normal_(self.linear_2.bias, std=1e-06)


LayerNorm = partial(nn.LayerNorm, eps=1e-06)


class EncoderBlock(nn.Module):
    """Transformer encoder block.

    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.

    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        self.num_heads = num_heads

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    def flops(self, x):
        flops = 0
        seq_len, batch_size, hidden_dim = x.shape
        num_elems = x.numel() // batch_size
        flops += num_elems * 6
        flops += 3 * seq_len * (hidden_dim + 1) * hidden_dim
        flops += hidden_dim * seq_len
        flops += hidden_dim * seq_len * seq_len
        flops += self.num_heads * seq_len * seq_len
        flops += hidden_dim * seq_len * seq_len
        flops += seq_len * (hidden_dim + 1) * hidden_dim
        mlp_dim = self.mlp.linear_1.out_features
        flops += seq_len * (hidden_dim + 1) * mlp_dim
        flops += seq_len * mlp_dim
        flops += seq_len * (mlp_dim + 1) * hidden_dim
        return flops * batch_size

    def activations(self, out, x):
        activations = 0
        seq_len, batch_size, hidden_dim = x.shape
        activations += 3 * seq_len * hidden_dim
        activations += self.num_heads * seq_len * seq_len
        activations += hidden_dim * seq_len
        activations += hidden_dim * seq_len
        mlp_dim = self.mlp.linear_1.out_features
        activations += seq_len * mlp_dim
        activations += seq_len * hidden_dim
        return activations


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for i in range(num_layers):
            layers.append((f'layer_{i}', EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate)))
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding
        return self.ln(self.layers(self.dropout(x)))


class ConvStemLayer(NamedTuple):
    kernel: int
    stride: int
    out_channels: int


class VisionTransformer(ClassyModel):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout_rate=0, attention_dropout_rate=0, classifier='token', conv_stem_layers: Union[List[ConvStemLayer], List[Dict], None]=None):
        super().__init__()
        assert image_size % patch_size == 0, 'Input shape indivisible by patch size'
        assert classifier in ['token', 'gap'], 'Unexpected classifier mode'
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.classifier = classifier
        input_channels = 3
        self.conv_stem_layers = conv_stem_layers
        if conv_stem_layers is None:
            self.conv_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        else:
            prev_channels = input_channels
            self.conv_proj = nn.Sequential()
            for i, conv_stem_layer in enumerate(conv_stem_layers):
                if isinstance(conv_stem_layer, Mapping):
                    conv_stem_layer = ConvStemLayer(**conv_stem_layer)
                kernel = conv_stem_layer.kernel
                stride = conv_stem_layer.stride
                out_channels = conv_stem_layer.out_channels
                padding = get_same_padding_for_kernel_size(kernel)
                self.conv_proj.add_module(f'conv_{i}', nn.Conv2d(prev_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False))
                self.conv_proj.add_module(f'bn_{i}', nn.BatchNorm2d(out_channels))
                self.conv_proj.add_module(f'relu_{i}', nn.ReLU())
                prev_channels = out_channels
            self.conv_proj.add_module(f'conv_{i + 1}', nn.Conv2d(prev_channels, hidden_dim, kernel_size=1))
        seq_length = (image_size // patch_size) ** 2
        if self.classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1
        self.encoder = Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate)
        self.trunk_output = nn.Identity()
        self.seq_length = seq_length
        self.init_weights()

    def init_weights(self):
        if self.conv_stem_layers is None:
            lecun_normal_init(self.conv_proj.weight, fan_in=self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1])
            nn.init.zeros_(self.conv_proj.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config.pop('name')
        config.pop('heads', None)
        return cls(**config)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, 'Unexpected input shape'
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(2, 0, 1)
        if self.classifier == 'token':
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)
        x = self.encoder(x)
        if self.classifier == 'token':
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)
        return self.trunk_output(x)

    def set_classy_state(self, state, strict=True):
        pos_embedding = state['model']['trunk']['encoder.pos_embedding']
        seq_length, n, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f'Unexpected position embedding shape: {pos_embedding.shape}')
        if hidden_dim != self.hidden_dim:
            raise ValueError(f'Position embedding hidden_dim incorrect: {hidden_dim}, expected: {self.hidden_dim}')
        new_seq_length = self.seq_length
        if new_seq_length != seq_length:
            if self.classifier == 'token':
                seq_length -= 1
                new_seq_length -= 1
                pos_embedding_token = pos_embedding[:1, :, :]
                pos_embedding_img = pos_embedding[1:, :, :]
            else:
                pos_embedding_token = pos_embedding[:0, :, :]
                pos_embedding_img = pos_embedding
            pos_embedding_img = pos_embedding_img.permute(1, 2, 0)
            seq_length_1d = int(math.sqrt(seq_length))
            assert seq_length_1d * seq_length_1d == seq_length, 'seq_length is not a perfect square'
            logging.info(f'Interpolating the position embeddings from image {seq_length_1d * self.patch_size} to size {self.image_size}')
            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = self.image_size // self.patch_size
            new_pos_embedding_img = torch.nn.functional.interpolate(pos_embedding_img, size=new_seq_length_1d, mode='bicubic', align_corners=True)
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.permute(2, 0, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=0)
            state['model']['trunk']['encoder.pos_embedding'] = new_pos_embedding
        super().set_classy_state(state, strict=strict)

    @property
    def input_shape(self):
        return 3, self.image_size, self.image_size


class ViTB32(VisionTransformer):

    def __init__(self, image_size=224, dropout_rate=0, attention_dropout_rate=0, classifier='token'):
        super().__init__(image_size=image_size, patch_size=32, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, classifier=classifier)


class ViTB16(VisionTransformer):

    def __init__(self, image_size=224, dropout_rate=0, attention_dropout_rate=0, classifier='token'):
        super().__init__(image_size=image_size, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, classifier=classifier)


class ViTL32(VisionTransformer):

    def __init__(self, image_size=224, dropout_rate=0, attention_dropout_rate=0, classifier='token'):
        super().__init__(image_size=image_size, patch_size=32, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, classifier=classifier)


class ViTL16(VisionTransformer):

    def __init__(self, image_size=224, dropout_rate=0, attention_dropout_rate=0, classifier='token'):
        super().__init__(image_size=image_size, patch_size=16, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, classifier=classifier)


class ViTH14(VisionTransformer):

    def __init__(self, image_size=224, dropout_rate=0, attention_dropout_rate=0, classifier='token'):
        super().__init__(image_size=image_size, patch_size=14, num_layers=32, num_heads=16, hidden_dim=1280, mlp_dim=5120, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, classifier=classifier)


class MyLoss(ClassyLoss):

    def forward(self, input, target):
        labels = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy(input, labels)

    @classmethod
    def from_config(cls, config):
        return cls()


class MyModel(ClassyModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.AdaptiveAvgPool2d((20, 20)), nn.Flatten(1), nn.Linear(3 * 20 * 20, 2), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls()


class TestModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x):
        return x + 1

    def flops(self, x):
        return x.numel()


class TestConvModule(nn.Conv2d):

    def __init__(self):
        super().__init__(2, 3, (4, 4), bias=False)
        self.linear = nn.Linear(4, 5, bias=False)

    def forward(self, x):
        return x

    def activations(self, x, out):
        return out.numel()

    def flops(self, x):
        return 0


class TestModuleWithTwoArguments(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

    def flops(self, x1, x2):
        return x1.numel()


class TestModuleDoubleValue(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10, bias=False)
        self.add = TestModuleWithTwoArguments()

    def forward(self, x):
        x = self.linear(x)
        return self.add(x, x)


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def extract_features(self, x):
        return torch.cat([x, x], dim=1)


class TestModel2(nn.Module):

    def __init__(self):
        super().__init__()
        conv_module = nn.Conv2d(3, 3, (2, 2), bias=False)
        self.seq_1 = nn.Sequential(conv_module)
        self.seq_2 = nn.Sequential(conv_module)

    def forward(self, x):
        return self.seq_1(x) + self.seq_2(x)


class TestModuleWithoutFlops(nn.Module):

    def forward(self, x):
        return x


class TestModuleWithFlops(nn.Module):
    _flops = 1234

    def __init__(self):
        super().__init__()
        self.mod = TestModuleWithoutFlops()
        self.conv = nn.Conv2d(3, 3, (2, 2))

    def forward(self, x):
        return self.conv(x)

    def flops(self, x):
        return self._flops


class MockLoss1(ClassyLoss):

    def forward(self, pred, target):
        return torch.tensor(1.0)

    @classmethod
    def from_config(cls, config):
        return cls()


class MockLoss2(ClassyLoss):

    def forward(self, pred, target):
        return torch.tensor(2.0)

    @classmethod
    def from_config(cls, config):
        return cls()


class MockLoss3(ClassyLoss):

    def forward(self, pred, target):
        return torch.tensor(3.0)

    @classmethod
    def from_config(cls, config):
        return cls()


class MyTestModel(ClassyModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 10)

    def forward(self, x):
        return self.linear2(self.linear(x))

    @classmethod
    def from_config(cls, config):
        return cls()


class MyTestModel2(ClassyModel):

    def forward(self, x):
        return x + 1

    @property
    def input_shape(self):
        return 1, 2, 3


class TestStatefulLoss(ClassyLoss):

    def __init__(self, in_plane):
        super(TestStatefulLoss, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(in_plane, 2))
        torch.nn.init.xavier_normal(self.alpha)

    @classmethod
    def from_config(cls, config) ->'TestStatefulLoss':
        return cls(in_plane=config['in_plane'])

    def forward(self, output, target):
        value = output.matmul(self.alpha)
        loss = torch.mean(torch.abs(value))
        return loss


class SimpleModel(ClassyModel):

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(5.0), requires_grad=True)

    def forward(self, x):
        return x + self.param

    @classmethod
    def from_config(cls):
        return cls()


class SimpleLoss(nn.Module):

    def forward(self, x, y):
        return x.pow(2).mean()


class BatchNormCrossEntropyLoss(ClassyLoss):
    """A special loss containing a BatchNorm module"""

    def __init__(self, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_classes)
        self.fc = nn.Linear(num_classes, num_classes)
        self.xent = CrossEntropyLoss()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) ->'BatchNormCrossEntropyLoss':
        assert 'num_classes' in config
        return cls(config['num_classes'])

    def forward(self, x, target):
        return self.xent(self.fc(self.bn(x)), target)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicLayer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicR2Plus1DTransformation,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicTransformation,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BatchNormCrossEntropyLoss,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (BottleneckLayer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FullyConnectedHead,
     lambda: ([], {'unique_id': 4, 'num_classes': 4, 'in_plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GenericLayer,
     lambda: ([], {'convolutional_block': _mock_layer(), 'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InitialBlock,
     lambda: ([], {'init_planes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'hidden_dims': [4, 4], 'dropout': 0.5, 'first_dropout': 0.5, 'use_batchnorm': 4, 'first_batchnorm': 4}),
     lambda: ([], {'x': torch.rand([4, 4])}),
     False),
    (MockLoss1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MockLoss2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MockLoss3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiOutputSumLoss,
     lambda: ([], {'loss': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MyTestModel2,
     lambda: ([], {}),
     lambda: ([], {'x': 4}),
     False),
    (PostactivatedBottleneckTransformation,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'num_groups': 1, 'dim_inner': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (PostactivatedShortcutTransformation,
     lambda: ([], {'dim_in': 1, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (PreactivatedBottleneckTransformation,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temporal_stride': 1, 'spatial_stride': 1, 'num_groups': 1, 'dim_inner': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ResNeXt3DStemSinglePathway,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SimpleLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmallInputInitialBlock,
     lambda: ([], {'init_planes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SoftTargetCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TestConvModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestModel2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestModuleWithFlops,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestModuleWithTwoArguments,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestModuleWithoutFlops,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestStatefulLoss,
     lambda: ([], {'in_plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (VisionTransformerHead,
     lambda: ([], {'unique_id': 4, 'in_plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DenseLayer,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_ClassyVision(_paritybench_base):
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

