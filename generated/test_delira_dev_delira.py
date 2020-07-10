import sys
_module = sys.modules[__name__]
del sys
delira = _module
_backends = _module
_debug_mode = _module
_version = _module
data_loading = _module
augmenter = _module
data_loader = _module
data_manager = _module
dataset = _module
load_utils = _module
numba_transform = _module
sampler = _module
abstract = _module
batch = _module
random = _module
sequential = _module
weighted = _module
io = _module
chainer = _module
sklearn = _module
tf = _module
logging = _module
base_backend = _module
base_logger = _module
logging_context = _module
registry = _module
tensorboard_backend = _module
visdom_backend = _module
writer_backend = _module
models = _module
abstract_network = _module
backends = _module
abstract_network = _module
data_parallel = _module
tf_eager = _module
tf_graph = _module
abstract_network = _module
data_parallel = _module
utils = _module
torchscript = _module
abstract_network = _module
training = _module
backends = _module
experiment = _module
trainer = _module
experiment = _module
trainer = _module
utils = _module
experiment = _module
trainer = _module
base_experiment = _module
base_trainer = _module
callbacks = _module
abstract_callback = _module
early_stopping = _module
logging_callback = _module
pytorch_schedulers = _module
losses = _module
metrics = _module
predictor = _module
codecs = _module
config = _module
context_managers = _module
decorators = _module
dict_reductions = _module
messenger = _module
path = _module
time = _module
tensorboard_backend = _module
conf = _module
setup = _module
tests = _module
test_augmenters = _module
test_data_loader = _module
test_data_manager = _module
test_dataset = _module
test_numba_transforms = _module
test_sampler = _module
test_chainer = _module
test_sklearn = _module
test_tf = _module
test_torch = _module
test_logging_frequency = _module
test_logging_outside_trainer = _module
test_single_threaded_logging = _module
test_torch = _module
test_abstract_models = _module
test_tf_eager = _module
test_tf_graph = _module
test_torch = _module
test_torchscript = _module
test_losses_torch = _module
test_metrics = _module
test_codecs = _module
test_config = _module
test_messenger = _module
versioneer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from collections import OrderedDict


from queue import Queue


import abc


import numpy as np


from functools import partial


import typing


import warnings


from functools import wraps


import re


from copy import deepcopy


from sklearn.metrics import mean_absolute_error


class AbstractNetwork(object):
    """
    Abstract class all networks should be derived from

    """
    _init_kwargs = {}

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Init function to register init kwargs (should be called from all
        subclasses)

        Parameters
        ----------
        **kwargs
            keyword arguments (will be registered to `self.init_kwargs`)

        """
        super().__init__()
        for key, val in kwargs.items():
            self._init_kwargs[key] = val

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        AbstractMethod to specify that each model should be able to be called
        for predictions

        Parameters
        ----------
        *args :
            Positional arguments
        **kwargs :
            Keyword Arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict, iter_num: int, fold=0, **kwargs):
        """
        Function which handles prediction from batch, logging, loss calculation
        and optimizer step

        Parameters
        ----------
        model : :class:`AbstractNetwork`
            model to forward data through
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary containing all optimizers to perform parameter update
        losses : dict
            Functions or classes to calculate losses
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        kwargs : dict
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Converts a numpy batch of data and labels to suitable datatype and
        pushes them to correct devices

        Parameters
        ----------
        batch : dict
            dictionary containing the batch (must have keys 'data' and 'label'
        input_device :
            device for network inputs
        output_device :
            device for network outputs

        Returns
        -------
        dict
            dictionary containing all necessary data in right format and type
            and on the correct device

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @property
    def init_kwargs(self):
        """
        Returns all arguments registered as init kwargs

        Returns
        -------
        dict
            init kwargs

        """
        return self._init_kwargs


class AbstractPyTorchNetwork(AbstractNetwork, torch.nn.Module):
    """
    Abstract Class for PyTorch Networks

    See Also
    --------
    `torch.nn.Module`
    :class:`AbstractNetwork`

    """

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            keyword arguments (are passed to :class:`AbstractNetwork`'s `
            __init__ to register them as init kwargs

        """
        torch.nn.Module.__init__(self)
        AbstractNetwork.__init__(self, **kwargs)

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        Forward inputs through module (defines module behavior)
        Parameters
        ----------
        inputs : list
            inputs of arbitrary type and number

        Returns
        -------
        Any
            result: module results of arbitrary type and number

        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """
        Calls Forward method

        Parameters
        ----------
        *args :
            positional arguments (passed to `forward`)
        **kwargs :
            keyword arguments (passed to `forward`)

        Returns
        -------
        Any
            result: module results of arbitrary type and number

        """
        return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device

        """
        return_dict = {'data': torch.from_numpy(batch['data']).to(input_device)}
        for key, vals in batch.items():
            if key == 'data':
                continue
            return_dict[key] = torch.from_numpy(vals).to(output_device)
        return return_dict

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict, iter_num: int, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the losses to calculate errors
            (gradients from different losses will be accumulated)
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions as numpy array

        """
        loss_vals = {}
        total_loss = 0
        with torch.enable_grad():
            inputs = data_dict['data']
            preds = model(inputs)
            for key, crit_fn in losses.items():
                _loss_val = crit_fn(preds['pred'], data_dict['label'])
                loss_vals[key] = _loss_val.item()
                total_loss += _loss_val
            optimizers['default'].zero_grad()
            with scale_loss(total_loss, optimizers['default']) as scaled_loss:
                scaled_loss.backward()
            optimizers['default'].step()
        return loss_vals, {k: v.detach() for k, v in preds.items()}


class DataParallelPyTorchNetwork(AbstractPyTorchNetwork, torch.nn.DataParallel):
    """
    A Wrapper around a :class:`AbstractPyTorchNetwork` instance to
    implement parallel training by splitting the batches
    """

    def __init__(self, module: AbstractPyTorchNetwork, device_ids=None, output_device=None, dim=0):
        """

        Parameters
        ----------
        module : :class:`AbstractPyTorchNetwork`
            the module to wrap (will be replicated on all devices)
        device_ids : list
            a list containing the devices to use (either as strings or as
            :class:`chainer.backend.Device`).
        output_device : str or :class:`chainer.backend.Device`
            The output device
            Make sure, your labels are also on this device
            for loss calculation!
            If not specified, the second device of ``devices`` will be used
            for output gathering.
        dim : int
            the index of the batchdimension (usually 0, but can become
            e.g. 1 in NLP tasks)

        """
        AbstractPyTorchNetwork.__init__(self)
        torch.nn.DataParallel.__init__(self, module, device_ids, output_device, dim)

    def forward(self, *args, **kwargs):
        """
        Scatters the inputs (both positional and keyword arguments) across
        all devices, feeds them through model replicas and re-builds
        batches on output device

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        Any
            combined output from all scattered models

        """
        return torch.nn.DataParallel.forward(*args, **kwargs)

    @property
    def closure(self):
        return self.module.closure

    @property
    def prepare_batch(self):
        return self.module.prepare_batch


class AbstractTorchScriptNetwork(AbstractNetwork, torch.jit.ScriptModule):
    """
    Abstract Interface Class for TorchScript Networks. For more information
    have a look at https://pytorch.org/docs/stable/jit.html#torchscript

    Warnings
    --------
    In addition to the here defined API, a forward function must be
    implemented and decorated with ``@torch.jit.script_method``

    """

    @abc.abstractmethod
    def __init__(self, optimize=True, **kwargs):
        """

        Parameters
        ----------
        optimize : bool
            whether to optimize the network graph or not; default: True
        **kwargs :
            additional keyword arguments
            (passed to :class:`AbstractNetwork`)
        """
        torch.jit.ScriptModule.__init__(self, optimize=optimize)
        AbstractNetwork.__init__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Calls Forward method

        Parameters
        ----------
        *args :
            positional arguments (passed to `forward`)
        **kwargs :
            keyword arguments (passed to `forward`)

        Returns
        -------
        Any
            result: module results of arbitrary type and number

        """
        return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device

        """
        return_dict = {'data': torch.from_numpy(batch['data']).to(input_device)}
        for key, vals in batch.items():
            if key == 'data':
                continue
            return_dict[key] = torch.from_numpy(vals).to(output_device)
        return return_dict

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict, iter_num: int, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`AbstractTorchScriptNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the losses to calculate errors
            (gradients from different losses will be accumulated)
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions as numpy array

        """
        loss_vals = {}
        total_loss = 0
        with torch.enable_grad():
            inputs = data_dict['data']
            preds = model(inputs)
            for key, crit_fn in losses.items():
                _loss_val = crit_fn(preds['pred'], data_dict['label'])
                loss_vals[key] = _loss_val.item()
                total_loss += _loss_val
            optimizers['default'].zero_grad()
            total_loss.backward()
            optimizers['default'].step()
        return loss_vals, {k: v.detach() for k, v in preds.items()}

