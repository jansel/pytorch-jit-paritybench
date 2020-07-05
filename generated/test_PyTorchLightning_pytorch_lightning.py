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
lr_finder = _module
model_hooks = _module
optimizers = _module
seed = _module
supporters = _module
training_io = _module
training_loop = _module
training_tricks = _module
utilities = _module
apply_func = _module
device_dtype_mixin = _module
distributed = _module
exceptions = _module
io = _module
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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


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


import numpy as np


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


from typing import Dict


from typing import Any


from torch import Tensor


import collections


import inspect


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import Sequence


import torch.distributed as torch_distrib


from torch.nn.parallel import DistributedDataParallel


import functools


from typing import Iterable


from typing import Mapping


import torch.distributed


import itertools


from itertools import chain


from torch.cuda._utils import _get_device_index


from torch.nn import DataParallel


import math


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Subset


import types


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
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

    def __init__(self, img_shape):
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
        return tensor.to(device, non_blocking=True)
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
                input_ = [(input_i.cuda(device) if torch.is_tensor(input_i) else input_i) for input_i in input_]
            else:
                input_ = input_.cuda(device)
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

