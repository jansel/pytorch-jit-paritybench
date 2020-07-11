import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
parity_modules = _module
test_parity = _module
conf = _module
pl_examples = _module
basic_examples = _module
cpu_template = _module
gpu_template = _module
multi_node_ddp2_demo = _module
multi_node_ddp_demo = _module
domain_templates = _module
computer_vision_fine_tuning = _module
generative_adversarial_net = _module
imagenet = _module
reinforce_learn_Qnet = _module
semantic_segmentation = _module
models = _module
lightning_template = _module
unet = _module
pytorch_lightning = _module
callbacks = _module
base = _module
early_stopping = _module
gradient_accumulation_scheduler = _module
lr_logger = _module
model_checkpoint = _module
progress = _module
core = _module
decorators = _module
grads = _module
hooks = _module
lightning = _module
memory = _module
saving = _module
loggers = _module
base = _module
comet = _module
mlflow = _module
neptune = _module
tensorboard = _module
test_tube = _module
trains = _module
wandb = _module
logging = _module
metrics = _module
converters = _module
metric = _module
sklearn = _module
utils = _module
overrides = _module
data_parallel = _module
profiler = _module
profilers = _module
trainer = _module
auto_mix_precision = _module
callback_config = _module
callback_hook = _module
data_loading = _module
deprecated_api = _module
distrib_data_parallel = _module
distrib_parts = _module
evaluation_loop = _module
ignored_warnings = _module
logging = _module
lr_finder = _module
model_hooks = _module
optimizers = _module
seed = _module
supporters = _module
trainer = _module
training_io = _module
training_loop = _module
training_tricks = _module
utilities = _module
apply_func = _module
device_dtype_mixin = _module
distributed = _module
exceptions = _module
io = _module
memory = _module
parsing = _module
setup = _module
tests = _module
dataloaders = _module
datasets = _module
mixins = _module
model_optimizers = _module
model_template = _module
model_test_dataloaders = _module
model_test_epoch_ends = _module
model_test_steps = _module
model_train_dataloaders = _module
model_train_steps = _module
model_utilities = _module
model_valid_dataloaders = _module
model_valid_epoch_ends = _module
model_valid_steps = _module
models = _module
utils = _module
test_callbacks = _module
test_lr = _module
test_progress_bar = _module
collect_env_details = _module
conftest = _module
test_all = _module
test_base = _module
test_comet = _module
test_mlflow = _module
test_neptune = _module
test_tensorboard = _module
test_trains = _module
test_wandb = _module
test_converters = _module
test_metrics = _module
test_sklearn_metrics = _module
train_default_model = _module
test_amp = _module
test_cpu = _module
test_gpu = _module
test_grad_norm = _module
test_hooks = _module
test_horovod = _module
test_hparams = _module
test_restore = _module
test_deprecated = _module
test_profiler = _module
test_checks = _module
test_dataloaders = _module
test_lr_finder = _module
test_optimizers = _module
test_trainer = _module
test_trainer_cli = _module
test_trainer_tricks = _module
test_apply_func = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import time


import numpy as np


import inspect


from collections import OrderedDict


from typing import Optional


from typing import Generator


from typing import Union


from torch.nn import Module


from torch import optim


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.optimizer import Optimizer


from torchvision import models


from torchvision import transforms


from torchvision.datasets import ImageFolder


from torchvision.datasets.utils import download_and_extract_archive


import torchvision


import torchvision.transforms as transforms


from torchvision.datasets import MNIST


import random


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


import torchvision.models as models


from typing import Tuple


from typing import List


from collections import deque


from collections import namedtuple


from torch.optim import Optimizer


from torch.utils.data.dataset import IterableDataset


import re


from typing import Dict


from typing import Any


from torch import Tensor


import collections


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import Sequence


import torch.distributed as torch_distrib


from torch.nn.parallel import DistributedDataParallel


import functools


from typing import Iterable


from typing import Mapping


from torch import is_tensor


from warnings import warn


from torch.utils.tensorboard import SummaryWriter


import numbers


from torch.utils.data._utils.collate import np_str_obj_array_pattern


import torch.distributed


from torch.utils.data._utils.collate import default_convert


import itertools


from itertools import chain


from torch.cuda._utils import _get_device_index


from torch.nn import DataParallel


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from time import sleep


from torch.optim.lr_scheduler import _LRScheduler


from typing import Type


import torch.multiprocessing as mp


import math


from collections import Mapping


from collections import Sequence


import logging


import numpy


from functools import wraps


from functools import partial


import torch.distributed as dist


from sklearn.metrics import accuracy_score


from sklearn.metrics import average_precision_score


from sklearn.metrics import auc


from sklearn.metrics import confusion_matrix


from sklearn.metrics import f1_score


from sklearn.metrics import fbeta_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import roc_curve


from sklearn.metrics import roc_auc_score


import logging as log


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Subset


import types


class Generator(nn.Module):

    def __init__(self, latent_dim: tuple, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self, img_shape: tuple):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class DQN(nn.Module):
    """
    Simple MLP network

    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x.float())


class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool=False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, in_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597

    Parameters:
        num_classes: Number of output classes required (default 19 for KITTI dataset)
        num_layers: Number of layers in each side of U-net
        features_start: Number of features in first layer
        bilinear: Whether to use bilinear interpolation or transposed
            convolutions for upsampling.
    """

    def __init__(self, num_classes: int=19, num_layers: int=5, features_start: int=64, bilinear: bool=False):
        super().__init__()
        self.num_layers = num_layers
        layers = [DoubleConv(3, features_start)]
        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2
        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class GradInformation(Module):

    def grad_norm(self, norm_type: Union[float, int, str]) ->Dict[str, float]:
        """Compute each parameter's gradient's norm and their overall norm.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.

        Return:
            norms: The dictionary of p-norms of each parameter's gradient and
                a special entry for the total p-norm of the gradients viewed
                as a single vector.
        """
        norm_type = float(norm_type)
        norms, all_norms = {}, []
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            param_norm = float(p.grad.data.norm(norm_type))
            norms[f'grad_{norm_type}_norm_{name}'] = round(param_norm, 3)
            all_norms.append(param_norm)
        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        norms[f'grad_{norm_type}_norm_total'] = round(total_norm, 3)
        return norms


def apply_to_collection(data: Any, dtype: Union[type, tuple], function: Callable, *args, **kwargs) ->Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        the resulting collection

    """
    elem_type = type(data)
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)
    elif isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])
    return data


def move_data_to_device(batch: Any, device: torch.device):
    """
    Transfers a collection of tensors to the given device.

    Args:
        batch: A tensor or collection of tensors. See :func:`apply_to_collection`
            for a list of supported collection types.
        device: The device to which tensors should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`
    """

    def to(tensor):
        return tensor
    return apply_to_collection(batch, dtype=torch.Tensor, function=to)


class ModelHooks(Module):

    def on_sanity_check_start(self):
        """
        Called before starting evaluation.

        Warning:
            Deprecated. Will be removed in v0.9.0.
        """

    def on_train_start(self) ->None:
        """
        Called at the beginning of training before sanity check.
        """

    def on_train_end(self) ->None:
        """
        Called at the end of training before logger experiment is closed.
        """

    def on_batch_start(self, batch: Any) ->None:
        """
        Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
        """

    def on_batch_end(self) ->None:
        """
        Called in the training loop after the batch.
        """

    def on_epoch_start(self) ->None:
        """
        Called in the training loop at the very beginning of the epoch.
        """

    def on_epoch_end(self) ->None:
        """
        Called in the training loop at the very end of the epoch.
        """

    def on_pre_performance_check(self) ->None:
        """
        Called at the very beginning of the validation loop.
        """

    def on_post_performance_check(self) ->None:
        """
        Called at the very end of the validation loop.
        """

    def on_before_zero_grad(self, optimizer: Optimizer) ->None:
        """
        Called after optimizer.step() and before optimizer.zero_grad().

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        This is where it is called::

            for optimizer in optimizers:
                optimizer.step()
                model.on_before_zero_grad(optimizer) # < ---- called here
                optimizer.zero_grad

        Args:
            optimizer: The optimizer for which grads should be zeroed.
        """

    def on_after_backward(self) ->None:
        """
        Called in the training loop after loss.backward() and before optimizers do anything.
        This is the ideal place to inspect or log gradient information.

        Example::

            def on_after_backward(self):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    params = self.state_dict()
                    for k, v in params.items():
                        grads = v
                        name = k
                        self.logger.experiment.add_histogram(tag=name, values=grads,
                                                             global_step=self.trainer.global_step)

        """

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) ->None:
        """
        Override backward with your own implementation if you need to.

        Args:
            trainer: Pointer to the trainer
            loss: Loss is already scaled by accumulated grads
            optimizer: Current optimizer being used
            optimizer_idx: Index of the current optimizer being used

        Called to perform backward step.
        Feel free to override as needed.

        The loss passed in has already been scaled for accumulated gradients if requested.

        Example::

            def backward(self, use_amp, loss, optimizer):
                if use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        """
        if trainer.precision == 16:
            if trainer.on_tpu:
                return
            if self.trainer.use_native_amp:
                self.trainer.scaler.scale(loss).backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()

    def transfer_batch_to_device(self, batch: Any, device: torch.device) ->Any:
        """
        Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors
        wrapped in a custom data structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`
        - ``torchtext.data.Batch`` (COMING SOON)

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Example::

            def transfer_batch_to_device(self, batch, device)
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                else:
                    batch = super().transfer_batch_to_device(data, device)
                return batch

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.

        Returns:
            A reference to the data on the new device.

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            The :class:`~pytorch_lightning.trainer.trainer.Trainer` already takes care of splitting the
            batch and determines the target devices.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
            - :func:`~pytorch_lightning.utilities.apply_func.apply_to_collection`
        """
        return move_data_to_device(batch, device)


class AttributeDict(dict):
    """Extended dictionary accesisable with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(mew_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "mew_key": 42
    "my-key":  3.14
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        if not len(self):
            return ''
        max_key_length = max([len(str(k)) for k in self])
        tmp_name = '{:' + str(max_key_length + 3) + 's} {}'
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = '\n'.join(rows)
        return out


class DeviceDtypeModuleMixin(Module):
    _device: ...
    _dtype: Union[str, torch.dtype]

    @property
    def dtype(self) ->Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]):
        raise RuntimeError('Cannot set the dtype explicitly. Please use module.to(new_dtype).')

    @property
    def device(self) ->Union[str, torch.device]:
        return self._device

    @device.setter
    def device(self, new_device: Union[str, torch.device]):
        raise RuntimeError('Cannot set the device explicitly. Please use module.to(new_device).')

    def to(self, *args, **kwargs) ->Module:
        """Moves and/or casts the parameters and buffers.

        This can be called as
        .. function:: to(device=None, dtype=None, non_blocking=False)
        .. function:: to(dtype, non_blocking=False)
        .. function:: to(tensor, non_blocking=False)
        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.
        See below for examples.

        Note:
            This method modifies the module in-place.

        Args:
            device: the desired device of the parameters
                and buffers in this module
            dtype: the desired floating point type of
                the floating point parameters and buffers in this module
            tensor: Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self

        Example::
            >>> class ExampleModule(DeviceDtypeModuleMixin):
            ...     def __init__(self, weight: torch.Tensor):
            ...         super().__init__()
            ...         self.register_buffer('weight', weight)
            >>> _ = torch.manual_seed(0)
            >>> module = ExampleModule(torch.rand(3, 4))
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]])
            >>> module.to(torch.double)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float64)
            >>> cpu = torch.device('cpu')
            >>> module.to(cpu, dtype=torch.half, non_blocking=True)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)
            >>> module.to(cpu)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)
        """
        out = torch._C._nn._parse_to(*args, **kwargs)
        device = out[0]
        dtype = out[1]
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return super()

    def cuda(self, device: Optional[int]=None) ->Module:
        """Moves all model parameters and buffers to the GPU.
        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device: if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        self._device = torch.device('cuda', index=device)
        return super()

    def cpu(self) ->Module:
        """Moves all model parameters and buffers to the CPU.
        Returns:
            Module: self
        """
        self._device = torch.device('cpu')
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) ->Module:
        """Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        self._dtype = dst_type
        return super().type(dst_type=dst_type)

    def float(self) ->Module:
        """Casts all floating point parameters and buffers to float datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.float
        return super().float()

    def double(self) ->Module:
        """Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.double
        return super().double()

    def half(self) ->Module:
        """Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.half
        return super().half()


def _find_tensors(obj):
    """
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class LightningDistributedDataParallel(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def forward(self, *inputs, **kwargs):
        self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        elif self.module.training:
            output = self.module.training_step(*inputs, **kwargs)
        elif self.module.testing:
            output = self.module.test_step(*inputs, **kwargs)
        else:
            output = self.module.validation_step(*inputs, **kwargs)
        if torch.is_grad_enabled():
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output


class MisconfigurationException(Exception):
    pass


def convert(val: str) ->Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        log.debug(err)
        return val


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
    return wrapped_fn


rank_zero_warn = rank_zero_only(_warn)


def load_hparams_from_tags_csv(tags_csv: str) ->Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = './testing-hparams.csv'
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)
    """
    if not os.path.isfile(tags_csv):
        rank_zero_warn(f'Missing Tags: {tags_csv}.', RuntimeWarning)
        return {}
    with open(tags_csv) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}
    return tags


def load_hparams_from_yaml(config_yaml: str) ->Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    if not os.path.isfile(config_yaml):
        rank_zero_warn(f'Missing Tags: {config_yaml}.', RuntimeWarning)
        return {}
    with open(config_yaml) as fp:
        tags = yaml.load(fp, Loader=yaml.SafeLoader)
    return tags


class ModelIO(object):
    CHECKPOINT_KEY_HYPER_PARAMS = 'hyper_parameters'
    CHECKPOINT_NAME_HYPER_PARAMS = 'hparams_name'

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, map_location=None):
        """
        Warning:
            Deprecated in version 0.7.0. You should use :meth:`load_from_checkpoint` instead.
            Will be removed in v0.9.0.
        """
        rank_zero_warn('`load_from_metrics` method has been unified with `load_from_checkpoint` in v0.7.0. The deprecated method will be removed in v0.9.0.', DeprecationWarning)
        return cls.load_from_checkpoint(weights_path, tags_csv=tags_csv, map_location=map_location)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, *args, map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]]=None, hparams_file: Optional[str]=None, tags_csv: Optional[str]=None, **kwargs):
        """
        Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
        it stores the arguments passed to `__init__`  in the checkpoint under `module_arguments`

        Any arguments specified through \\*args and \\*\\*kwargs will override args stored in `hparams`.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL.
            args: Any positional args needed to init the model.
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a .yaml file with hierarchical structure
                as in this example::

                    drop_prob: 0.2
                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a .yaml file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningModule` for use.

                If your model's `hparams` argument is :class:`~argparse.Namespace`
                and .yaml file has hierarchical structure, you need to refactor your model to treat
                `hparams` as :class:`~dict`.

                .csv files are acceptable here till v0.9.0, see tags_csv argument for detailed usage.
            tags_csv:
                .. warning:: .. deprecated:: 0.7.6

                    `tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0.

                Optional path to a .csv file with two columns (key, value)
                as in this example::

                    key,value
                    drop_prob,0.2
                    batch_size,32

                Use this method to pass in a .csv file with the hparams you'd like to use.
            hparam_overrides: A dictionary with keys to override in the hparams
            kwargs: Any keyword args needed to init the model.

        Return:
            :class:`LightningModule` with loaded weights and hyperparameters (if available).

        Example:
            .. code-block:: python

                # load weights without mapping ...
                MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

                # or load weights mapping all weights from GPU 1 to GPU 0 ...
                map_location = {'cuda:1':'cuda:0'}
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    map_location=map_location
                )

                # or load weights and hyperparameters from separate files.
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    hparams_file='/path/to/hparams_file.yaml'
                )

                # override some of the params with new values
                MyLightningModule.load_from_checkpoint(
                    PATH,
                    num_layers=128,
                    pretrained_ckpt_path: NEW_PATH,
                )

                # predict
                pretrained_model.eval()
                pretrained_model.freeze()
                y_hat = pretrained_model(x)
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        if tags_csv is not None:
            hparams_file = tags_csv
            rank_zero_warn('`tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0', DeprecationWarning)
        if hparams_file is not None:
            extension = hparams_file.split('.')[-1]
            if extension.lower() in 'csv':
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ('yml', 'yaml'):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')
            hparams['on_gpu'] = False
            checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS] = hparams
        checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS].update(kwargs)
        model = cls._load_model_state(checkpoint, *args, **kwargs)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], *args, **kwargs):
        if cls.CHECKPOINT_KEY_HYPER_PARAMS in checkpoint:
            model_args = checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS]
            args_name = checkpoint.get(cls.CHECKPOINT_NAME_HYPER_PARAMS)
            init_args_name = inspect.signature(cls).parameters.keys()
            if args_name == 'kwargs':
                cls_kwargs = {k: v for k, v in model_args.items() if k in init_args_name}
                kwargs.update(**cls_kwargs)
            elif args_name:
                if args_name in init_args_name:
                    kwargs.update({args_name: model_args})
            else:
                args = (model_args,) + args
        model = cls(*args, **kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)
        return model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) ->None:
        """
        Do something with the checkpoint.
        Gives model a chance to load something before ``state_dict`` is restored.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) ->None:
        """
        Give the model a chance to add something to the checkpoint.
        ``state_dict`` is already there.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    def on_hpc_save(self, checkpoint: Dict[str, Any]) ->None:
        """
        Hook to do whatever you need right before Slurm manager saves the model.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    def on_hpc_load(self, checkpoint: Dict[str, Any]) ->None:
        """
        Hook to do whatever you need right before Slurm manager loads the model.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """


def _format_summary_table(*cols) ->str:
    """
    Takes in a number of arrays, each specifying a column in
    the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted.
    """
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)
    counter = list(map(str, list(range(n_rows))))
    counter_len = max([len(c) for c in counter])
    length = []
    for c in cols:
        str_l = len(c[0])
        for a in c[1]:
            if isinstance(a, np.ndarray):
                array_string = '[' + ', '.join([str(j) for j in a]) + ']'
                str_l = max(len(array_string), str_l)
            else:
                str_l = max(len(a), str_l)
        length.append(str_l)
    s = '{:<{}}'
    full_length = sum(length) + 3 * n_cols
    header = [s.format(' ', counter_len)] + [s.format(c[0], l) for c, l in zip(cols, length)]
    summary = ' | '.join(header) + '\n' + '-' * full_length
    for i in range(n_rows):
        line = s.format(counter[i], counter_len)
        for c, l in zip(cols, length):
            if isinstance(c[1][i], np.ndarray):
                array_string = '[' + ', '.join([str(j) for j in c[1][i]]) + ']'
                line += ' | ' + array_string + ' ' * (l - len(array_string))
            else:
                line += ' | ' + s.format(c[1][i], l)
        summary += '\n' + line
    return summary


def get_human_readable_count(number: int) ->str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = [' ', 'K', 'M', 'B', 'T']
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * 10 ** shift
    index = num_groups - 1
    return f'{int(number):,d} {labels[index]}'


class ModelSummary(object):

    def __init__(self, model: 'pl.LightningModule', mode: str='full'):
        """ Generates summaries of model layers and dimensions. """
        self.model = model
        self.mode = mode
        self.in_sizes = []
        self.out_sizes = []
        self.summarize()

    def __str__(self):
        return self.summary.__str__()

    def __repr__(self):
        return self.summary.__str__()

    def named_modules(self) ->List[Tuple[str, Module]]:
        if self.mode == 'full':
            mods = self.model.named_modules()
            mods = list(mods)[1:]
        elif self.mode == 'top':
            mods = self.model.named_children()
        else:
            mods = []
        return list(mods)

    def get_variable_sizes(self) ->None:
        """ Run sample input through each layer to get output sizes. """
        mods = self.named_modules()
        in_sizes = []
        out_sizes = []
        input_ = self.model.example_input_array
        if self.model.on_gpu:
            device = next(self.model.parameters()).get_device()
            if isinstance(input_, (list, tuple)):
                input_ = [(input_i if torch.is_tensor(input_i) else input_i) for input_i in input_]
            else:
                input_ = input_
        if self.model.trainer.use_amp:
            if isinstance(input_, (list, tuple)):
                input_ = [(input_i.half() if torch.is_tensor(input_i) else input_i) for input_i in input_]
            else:
                input_ = input_.half()
        with torch.no_grad():
            for _, m in mods:
                if isinstance(input_, (list, tuple)):
                    out = m(*input_)
                else:
                    out = m(input_)
                if isinstance(input_, (list, tuple)):
                    in_size = []
                    for x in input_:
                        if isinstance(x, list):
                            in_size.append(len(x))
                        else:
                            in_size.append(x.size())
                else:
                    in_size = np.array(input_.size())
                in_sizes.append(in_size)
                if isinstance(out, (list, tuple)):
                    out_size = np.asarray([x.size() for x in out])
                else:
                    out_size = np.array(out.size())
                out_sizes.append(out_size)
                input_ = out
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        assert len(in_sizes) == len(out_sizes)

    def get_layer_names(self) ->None:
        """ Collect Layer Names """
        mods = self.named_modules()
        names = []
        layers = []
        for name, m in mods:
            names += [name]
            layers += [str(m.__class__)]
        layer_types = [x.split('.')[-1][:-2] for x in layers]
        self.layer_names = names
        self.layer_types = layer_types

    def get_parameter_sizes(self) ->None:
        """ Get sizes of all parameters in `model`. """
        mods = self.named_modules()
        sizes = []
        for _, m in mods:
            p = list(m.parameters())
            modsz = [np.array(param.size()) for param in p]
            sizes.append(modsz)
        self.param_sizes = sizes

    def get_parameter_nums(self) ->None:
        """ Get number of parameters in each layer. """
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums

    def make_summary(self) ->None:
        """
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        """
        arrays = [['Name', self.layer_names], ['Type', self.layer_types], ['Params', list(map(get_human_readable_count, self.param_nums))]]
        if self.model.example_input_array is not None:
            arrays.append(['In sizes', self.in_sizes])
            arrays.append(['Out sizes', self.out_sizes])
        self.summary = _format_summary_table(*arrays)

    def summarize(self) ->None:
        self.get_layer_names()
        self.get_parameter_sizes()
        self.get_parameter_nums()
        if self.model.example_input_array is not None:
            self.get_variable_sizes()
        self.make_summary()


PRIMITIVE_TYPES = bool, int, float, str


def get_init_args(frame) ->dict:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' not in local_vars:
        return
    cls = local_vars['__class__']
    spec = inspect.getfullargspec(cls.__init__)
    init_parameters = inspect.signature(cls.__init__).parameters
    self_identifier = spec.args[0]
    varargs_identifier = spec.varargs
    kwargs_identifier = spec.varkw
    exclude_argnames = varargs_identifier, kwargs_identifier, self_identifier, '__class__', 'frame', 'frame_args'
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    local_args.update(local_args.get(kwargs_identifier, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    return local_args


def collect_init_args(frame, path_args: list, inside: bool=False) ->list:
    """
    Recursively collects the arguments passed to the child constructors in the inheritance tree.

    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon

    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' in local_vars:
        local_args = get_init_args(frame)
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True)
    elif not inside:
        return collect_init_args(frame.f_back, path_args, inside)
    else:
        return path_args


class Metric(ABC, DeviceDtypeModuleMixin, Module):
    """
    Abstract base class for metric implementation.

    Should be used to implement metrics that
    1. Return multiple Outputs
    2. Handle their own DDP sync
    """

    def __init__(self, name: str):
        """
        Args:
            name: the metric's name

        """
        super().__init__()
        self.name = name
        self._dtype = torch.get_default_dtype()
        self._device = torch.device('cpu')

    @abstractmethod
    def forward(self, *args, **kwargs) ->torch.Tensor:
        """
        Implements the actual metric computation.

        Returns:
            metric value

        """
        raise NotImplementedError


def _apply_to_outputs(func_to_apply, *dec_args, **dec_kwargs):

    def decorator_fn(function_to_decorate):

        def new_func(*args, **kwargs):
            result = function_to_decorate(*args, **kwargs)
            return func_to_apply(result, *dec_args, **dec_kwargs)
        return new_func
    return decorator_fn


def _sync_ddp(result: Union[torch.Tensor], group: Any=torch.distributed.group.WORLD, reduce_op: torch.distributed.ReduceOp=torch.distributed.ReduceOp.SUM) ->torch.Tensor:
    """
    Function to reduce the tensors from several ddp processes to one master process

    Args:
        result: the value to sync and reduce (typically tensor or number)
        device: the device to put the synced and reduced value to
        dtype: the datatype to convert the synced and reduced value to
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum

    Returns:
        reduced value

    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier(group=group)
        torch.distributed.all_reduce(result, op=reduce_op, group=group, async_op=False)
    return result


def _apply_to_inputs(func_to_apply, *dec_args, **dec_kwargs):

    def decorator_fn(func_to_decorate):

        def new_func(*args, **kwargs):
            args = func_to_apply(args, *dec_args, **dec_kwargs)
            kwargs = func_to_apply(kwargs, *dec_args, **dec_kwargs)
            return func_to_decorate(*args, **kwargs)
        return new_func
    return decorator_fn


def _convert_to_tensor(data: Any) ->Any:
    """
    Maps all kind of collections and numbers to tensors

    Args:
        data: the data to convert to tensor

    Returns:
        the converted data

    """
    if isinstance(data, numbers.Number):
        return torch.tensor([data])
    else:
        return default_convert(data)


def _tensor_metric_conversion(func_to_decorate):
    func_convert_inputs = _apply_to_inputs(_convert_to_tensor)(func_to_decorate)
    return _apply_to_outputs(_convert_to_tensor)(func_convert_inputs)


def tensor_metric(group: Any=torch.distributed.group.WORLD, reduce_op: torch.distributed.ReduceOp=torch.distributed.ReduceOp.SUM):

    def decorator_fn(func_to_decorate):
        return _apply_to_outputs(apply_to_collection, torch.Tensor, _sync_ddp, group=group, reduce_op=reduce_op)(_tensor_metric_conversion(func_to_decorate))
    return decorator_fn


class TensorMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs and outputs will be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str, reduce_group: Optional[Any]=None, reduce_op: Optional[Any]=None):
        """

        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(name)
        self._orig_call = tensor_metric(group=reduce_group, reduce_op=reduce_op)(super().__call__)

    def __call__(self, *args, **kwargs) ->torch.Tensor:

        def _to_device_dtype(x: torch.Tensor) ->torch.Tensor:
            return x
        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor, _to_device_dtype)


def _convert_to_numpy(data: Union[torch.Tensor, np.ndarray, numbers.Number]) ->np.ndarray:
    """
    converts all tensors and numpy arrays to numpy arrays
    Args:
        data: the tensor or array to convert to numpy

    Returns:
        the resulting numpy array

    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, numbers.Number):
        return np.array([data])
    return data


def _numpy_metric_conversion(func_to_decorate):
    func_convert_inputs = _apply_to_inputs(apply_to_collection, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)(func_to_decorate)
    func_convert_in_out = _apply_to_outputs(_convert_to_tensor)(func_convert_inputs)
    return func_convert_in_out


def numpy_metric(group: Any=torch.distributed.group.WORLD, reduce_op: torch.distributed.ReduceOp=torch.distributed.ReduceOp.SUM):

    def decorator_fn(func_to_decorate):
        return _apply_to_outputs(apply_to_collection, torch.Tensor, _sync_ddp, group=group, reduce_op=reduce_op)(_numpy_metric_conversion(func_to_decorate))
    return decorator_fn


class NumpyMetric(Metric):
    """
    Base class for metric implementation operating on numpy arrays.
    All inputs will be casted to numpy if necessary and all outputs will
    be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str, reduce_group: Optional[Any]=None, reduce_op: Optional[Any]=None):
        """

        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(name)
        self._orig_call = numpy_metric(group=reduce_group, reduce_op=reduce_op)(super().__call__)

    def __call__(self, *args, **kwargs) ->torch.Tensor:

        def _to_device_dtype(x: torch.Tensor) ->torch.Tensor:
            return x
        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor, _to_device_dtype)


class SklearnMetric(NumpyMetric):
    """
    Bridge between PyTorch Lightning and scikit-learn metrics

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code

    Note:
        The order of targets and predictions may be different from the order typically used in PyTorch
    """

    def __init__(self, metric_name: str, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM, **kwargs):
        """
        Args:
            metric_name: the metric name to import and compute from scikit-learn.metrics
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
            **kwargs: additonal keyword arguments (will be forwarded to metric call)
        """
        super().__init__(name=metric_name, reduce_group=reduce_group, reduce_op=reduce_op)
        self.metric_kwargs = kwargs
        lightning_logger.debug(f'Metric {self.__class__.__name__} is using Sklearn as backend, meaning that every metric call will cause a GPU synchronization, which may slow down your code')

    @property
    def metric_fn(self):
        import sklearn.metrics
        return getattr(sklearn.metrics, self.name)

    def forward(self, *args, **kwargs) ->Union[np.ndarray, int, float]:
        """
        Carries the actual metric computation

        Args:
            *args: Positional arguments forwarded to metric call (should be already converted to numpy)
            **kwargs: keyword arguments forwarded to metric call (should be already converted to numpy)

        Return:
            the metric value (will be converted to tensor by baseclass)

        """
        return self.metric_fn(*args, **kwargs, **self.metric_kwargs)


class Accuracy(SklearnMetric):
    """
    Calculates the Accuracy Score

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code
    """

    def __init__(self, normalize: bool=True, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            normalize: If ``False``, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(metric_name='accuracy_score', reduce_group=reduce_group, reduce_op=reduce_op, normalize=normalize)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->float:
        """
        Computes the accuracy

        Args:
            y_pred: the array containing the predictions (already in categorical form)
            y_true: the array containing the targets (in categorical form)
            sample_weight:  Sample weights.

        Return:
            Accuracy Score

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class AUC(SklearnMetric):
    """
    Calculates the Area Under the Curve using the trapoezoidal rule

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code
    """

    def __init__(self, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(metric_name='auc', reduce_group=reduce_group, reduce_op=reduce_op)

    def forward(self, x: np.ndarray, y: np.ndarray) ->float:
        """
        Computes the AUC

        Args:
            x: x coordinates.
            y: y coordinates.

        Return:
            AUC calculated with trapezoidal rule

        """
        return super().forward(x=x, y=y)


class AveragePrecision(SklearnMetric):
    """
    Calculates the average precision (AP) score.
    """

    def __init__(self, average: Optional[str]='macro', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            average: If None, the scores for each class are returned. Otherwise, this determines the type of
                averaging performed on the data:

                * If 'micro': Calculate metrics globally by considering each element of the label indicator
                  matrix as a label.
                * If 'macro': Calculate metrics for each label, and find their unweighted mean.
                  This does not take label imbalance into account.
                * If 'weighted': Calculate metrics for each label, and find their average, weighted by
                  support (the number of true instances for each label).
                * If 'samples': Calculate metrics for each instance, and find their average.

            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('average_precision_score', reduce_group=reduce_group, reduce_op=reduce_op, average=average)

    def forward(self, y_score: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->float:
        """
        Args:
            y_score: Target scores, can either be probability estimates of the positive class,
                confidence values, or binary decisions.
            y_true: True binary labels in binary label indicators.
            sample_weight: Sample weights.

        Return:
            average precision score
        """
        return super().forward(y_score=y_score, y_true=y_true, sample_weight=sample_weight)


class ConfusionMatrix(SklearnMetric):
    """
    Compute confusion matrix to evaluate the accuracy of a classification
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.
    """

    def __init__(self, labels: Optional[Sequence]=None, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If none is given, those that appear at least once
                in ``y_true`` or ``y_pred`` are used in sorted order.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('confusion_matrix', reduce_group=reduce_group, reduce_op=reduce_op, labels=labels)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) ->np.ndarray:
        """
        Args:
            y_pred: Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.

        Return:
            Confusion matrix (array of shape [n_classes, n_classes])

        """
        return super().forward(y_pred=y_pred, y_true=y_true)


class F1(SklearnMetric):
    """
    Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is:

    .. math::

        F_1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}

    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    References
        - [1] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(self, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: Optional[str]='binary', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
                  Calculate metrics for each instance, and find their average (only
                  meaningful for multilabel classification where this differs from
                  :func:`accuracy_score`).

                Note that if ``pos_label`` is given in binary classification with
                `average != 'binary'`, only that positive class is reported. This
                behavior is deprecated and will change in version 0.18.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('f1_score', reduce_group=reduce_group, reduce_op=reduce_op, labels=labels, pos_label=pos_label, average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class FBeta(SklearnMetric):
    """
    Compute the F-beta score. The `beta` parameter determines the weight of precision in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> inf``
    only recall).

    References:
        - [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
          Modern Information Retrieval. Addison Wesley, pp. 327-328.
        - [2] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(self, beta: float, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: Optional[str]='binary', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            beta: Weight of precision in harmonic mean.
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
                  Calculate metrics for each instance, and find their average (only
                  meaningful for multilabel classification where this differs from
                  :func:`accuracy_score`).

                Note that if ``pos_label`` is given in binary classification with
                `average != 'binary'`, only that positive class is reported. This
                behavior is deprecated and will change in version 0.18.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('fbeta_score', reduce_group=reduce_group, reduce_op=reduce_op, beta=beta, labels=labels, pos_label=pos_label, average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.


        Return:
            FBeta score of the positive class in binary classification or weighted
            average of the FBeta scores of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class Precision(SklearnMetric):
    """
    Compute the precision
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The best value is 1 and the worst value is 0.
    """

    def __init__(self, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: Optional[str]='binary', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
                  Calculate metrics for each instance, and find their average (only
                  meaningful for multilabel classification where this differs from
                  :func:`accuracy_score`).

                Note that if ``pos_label`` is given in binary classification with
                `average != 'binary'`, only that positive class is reported. This
                behavior is deprecated and will change in version 0.18.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('precision_score', reduce_group=reduce_group, reduce_op=reduce_op, labels=labels, pos_label=pos_label, average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class Recall(SklearnMetric):
    """
    Compute the recall
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    """

    def __init__(self, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: Optional[str]='binary', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
                  Calculate metrics for each instance, and find their average (only
                  meaningful for multilabel classification where this differs from
                  :func:`accuracy_score`).

                Note that if ``pos_label`` is given in binary classification with
                `average != 'binary'`, only that positive class is reported. This
                behavior is deprecated and will change in version 0.18.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('recall_score', reduce_group=reduce_group, reduce_op=reduce_op, labels=labels, pos_label=pos_label, average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Recall of the positive class in binary classification or weighted
            average of the recall of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class PrecisionRecallCurve(SklearnMetric):
    """
    Compute precision-recall pairs for different probability thresholds

    Note:
        This implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    x axis.
    """

    def __init__(self, pos_label: Union[str, int]=1, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            pos_label: The class to report if ``average='binary'``.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('precision_recall_curve', reduce_group=reduce_group, reduce_op=reduce_op, pos_label=pos_label)

    def forward(self, probas_pred: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            probas_pred : Estimated probabilities or decision function.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Returns:
            precision:
                Precision values such that element i is the precision of
                predictions with score >= thresholds[i] and the last element is 1.
            recall:
                Decreasing recall values such that element i is the recall of
                predictions with score >= thresholds[i] and the last element is 0.
            thresholds:
                Increasing thresholds on the decision function used to compute
                precision and recall.

        """
        return np.array(super().forward(probas_pred=probas_pred, y_true=y_true, sample_weight=sample_weight)[:2])


class ROC(SklearnMetric):
    """
    Compute Receiver operating characteristic (ROC)

    Note:
        this implementation is restricted to the binary classification task.
    """

    def __init__(self, pos_label: Union[str, int]=1, reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            pos_labels: The class to report if ``average='binary'``.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.

        References:
            - [1] `Wikipedia entry for the Receiver operating characteristic
              <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
        """
        super().__init__('roc_curve', reduce_group=reduce_group, reduce_op=reduce_op, pos_label=pos_label)

    def forward(self, y_score: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->Union[np.ndarray, float]:
        """
        Args:
            y_score : Target scores, can either be probability estimates of the positive
                class or confidence values.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Returns:
            fpr:
                Increasing false positive rates such that element i is the false
                positive rate of predictions with score >= thresholds[i].
            tpr:
                Increasing true positive rates such that element i is the true
                positive rate of predictions with score >= thresholds[i].
            thresholds:
                Decreasing thresholds on the decision function used to compute
                fpr and tpr. `thresholds[0]` represents no instances being predicted
                and is arbitrarily set to `max(y_score) + 1`.

        """
        return np.array(super().forward(y_score=y_score, y_true=y_true, sample_weight=sample_weight)[:2])


class AUROC(SklearnMetric):
    """
    Compute Area Under the Curve (AUC) from prediction scores

    Note:
        this implementation is restricted to the binary classification task
        or multilabel classification task in label indicator format.
    """

    def __init__(self, average: Optional[str]='macro', reduce_group: Any=torch.distributed.group.WORLD, reduce_op: Any=torch.distributed.ReduceOp.SUM):
        """
        Args:
            average: If None, the scores for each class are returned. Otherwise, this determines the type of
                averaging performed on the data:

                * If 'micro': Calculate metrics globally by considering each element of the label indicator
                  matrix as a label.
                * If 'macro': Calculate metrics for each label, and find their unweighted mean.
                  This does not take label imbalance into account.
                * If 'weighted': Calculate metrics for each label, and find their average, weighted by
                  support (the number of true instances for each label).
                * If 'samples': Calculate metrics for each instance, and find their average.

            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('roc_auc_score', reduce_group=reduce_group, reduce_op=reduce_op, average=average)

    def forward(self, y_score: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->float:
        """
        Args:
            y_score: Target scores, can either be probability estimates of the positive class,
                confidence values, or binary decisions.
            y_true: True binary labels in binary label indicators.
            sample_weight: Sample weights.

        Return:
            Area Under Receiver Operating Characteristic Curve
        """
        return super().forward(y_score=y_score, y_true=y_true, sample_weight=sample_weight)


class LightningDataParallel(DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError('module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}'.format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            if self.module.training:
                return self.module.training_step(*inputs[0], **kwargs[0])
            if self.module.testing:
                return self.module.test_step(*inputs[0], **kwargs[0])
            return self.module.validation_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class ConfigureOptimizersPool(ABC):

    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers__empty(self):
        return None

    def configure_optimizers__lbfgs(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers__multiple_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer1, optimizer2

    def configure_optimizers__single_scheduler(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def configure_optimizers__multiple_schedulers(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    def configure_optimizers__mixed_scheduling(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 4, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
        return [optimizer1, optimizer2], [{'scheduler': lr_scheduler1, 'interval': 'step'}, lr_scheduler2]

    def configure_optimizers__reduce_lr_on_plateau(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [lr_scheduler]

    def configure_optimizers__param_groups(self):
        param_groups = [{'params': list(self.parameters())[:2], 'lr': self.learning_rate * 0.1}, {'params': list(self.parameters())[2:], 'lr': self.learning_rate}]
        optimizer = optim.Adam(param_groups)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]


class ModelTemplateData:
    hparams: ...

    def dataloader(self, train):
        dataset = TrialMNIST(root=self.data_root, train=train, download=True)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=3, shuffle=train)
        return loader


class ModelTemplateUtils:

    def get_output_metric(self, output, name):
        if isinstance(output, dict):
            val = output[name]
        else:
            val = sum(out[name] for out in output) / len(output)
        return val


class CustomInfDataloader:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= 50:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)


class TestDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def test_dataloader__empty(self):
        return None

    def test_dataloader__multiple(self):
        return [self.dataloader(train=False), self.dataloader(train=False)]


class TestEpochEndVariations(ABC):

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = self.get_output_metric(output, 'test_loss')
            if self.trainer.use_dp:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss
            test_acc = self.get_output_metric(output, 'test_acc')
            if self.trainer.use_dp:
                test_acc = torch.mean(test_acc)
            test_acc_mean += test_acc
        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        metrics_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        result = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return result

    def test_epoch_end__multiple_dataloaders(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        test_loss_mean = 0
        test_acc_mean = 0
        i = 0
        for dl_output in outputs:
            for output in dl_output:
                test_loss = output['test_loss']
                if self.trainer.use_dp:
                    test_loss = torch.mean(test_loss)
                test_loss_mean += test_loss
                test_acc = output['test_acc']
                if self.trainer.use_dp:
                    test_acc = torch.mean(test_acc)
                test_acc_mean += test_acc
                i += 1
        test_loss_mean /= i
        test_acc_mean /= i
        tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        result = {'progress_bar': tqdm_dict}
        return result


class TestStepVariations(ABC):
    """
    Houses all variations of test steps
    """

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss_test = self.loss(y, y_hat)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)
        test_acc = test_acc.type_as(x)
        if batch_idx % 1 == 0:
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc})
            return output
        if batch_idx % 2 == 0:
            return test_acc
        if batch_idx % 3 == 0:
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc, 'test_dic': {'test_loss_a': loss_test}})
            return output

    def test_step__multiple_dataloaders(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss_test = self.loss(y, y_hat)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)
        test_acc = test_acc.type_as(x)
        if batch_idx % 1 == 0:
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc})
            return output
        if batch_idx % 2 == 0:
            return test_acc
        if batch_idx % 3 == 0:
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc, 'test_dic': {'test_loss_a': loss_test}})
            return output
        if batch_idx % 5 == 0:
            output = OrderedDict({f'test_loss_{dataloader_idx}': loss_test, f'test_acc_{dataloader_idx}': test_acc})
            return output

    def test_step__empty(self, batch, batch_idx, *args, **kwargs):
        return {}


class TrainDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def train_dataloader(self):
        return self.dataloader(train=True)

    def train_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=True))

    def train_dataloader__zero_length(self):
        dataloader = self.dataloader(train=True)
        dataloader.dataset.data = dataloader.dataset.data[:0]
        dataloader.dataset.targets = dataloader.dataset.targets[:0]
        return dataloader


class TrainingStepVariations(ABC):
    """
    Houses all variations of training steps
    """
    test_step_inf_loss = float('inf')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss_val = self.loss(y, y_hat)
        output = OrderedDict({'loss': loss_val, 'progress_bar': {'some_val': loss_val * loss_val}, 'log': {'train_some_val': loss_val * loss_val}})
        return output

    def training_step__inf_loss(self, batch, batch_idx, optimizer_idx=None):
        output = self.training_step(batch, batch_idx, optimizer_idx)
        if batch_idx == self.test_step_inf_loss:
            if isinstance(output, dict):
                output['loss'] *= torch.tensor(math.inf)
            else:
                output /= 0
        return output


class ValDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def val_dataloader(self):
        return self.dataloader(train=False)

    def val_dataloader__multiple(self):
        return [self.dataloader(train=False), self.dataloader(train=False)]

    def val_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))


class ValidationEpochEndVariations(ABC):
    """
    Houses all variations of validation_epoch_end steps
    """

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """

        def _mean(res, key):
            return torch.stack([(x[key] if isinstance(x, dict) else _mean(x, key)) for x in res]).mean()
        val_loss_mean = _mean(outputs, 'val_loss')
        val_acc_mean = _mean(outputs, 'val_acc')
        metrics_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results

    def validation_epoch_end_multiple_dataloaders(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """

        def _mean(res, key):
            return torch.stack([x[key] for x in res]).mean()
        pbar = {}
        logs = {}
        for dl_output_list in outputs:
            output_keys = dl_output_list[0].keys()
            output_keys = [x for x in output_keys if 'val_' in x]
            for key in output_keys:
                metric_out = _mean(dl_output_list, key)
                pbar[key] = metric_out
                logs[key] = metric_out
        results = {'progress_bar': pbar, 'log': logs}
        return results


class ValidationStepVariations(ABC):
    """
    Houses all variations of validation steps
    """

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss_val = self.loss(y, y_hat)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc).type_as(x)
        output = OrderedDict({'val_loss': loss_val, 'val_acc': val_acc, 'test_dic': {'val_loss_a': loss_val}})
        return output

    def validation_step__multiple_dataloaders(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss_val = self.loss(y, y_hat)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc).type_as(x)
        output = OrderedDict({f'val_loss_{dataloader_idx}': loss_val, f'val_acc_{dataloader_idx}': val_acc})
        return output


class DummyTensorMetric(TensorMetric):

    def __init__(self):
        super().__init__('dummy')

    def forward(self, input1, input2):
        assert isinstance(input1, torch.Tensor)
        assert isinstance(input2, torch.Tensor)
        return 1.0


class DummyNumpyMetric(NumpyMetric):

    def __init__(self):
        super().__init__('dummy')

    def forward(self, input1, input2):
        assert isinstance(input1, np.ndarray)
        assert isinstance(input2, np.ndarray)
        return 1.0


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DQN,
     lambda: ([], {'obs_size': 4, 'n_actions': 4}),
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
    (DummyNumpyMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyTensorMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LightningDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_PyTorchLightning_pytorch_lightning(_paritybench_base):
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

