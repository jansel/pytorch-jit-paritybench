import sys
_module = sys.modules[__name__]
del sys
conf = _module
basic_mnist_classification = _module
basic_random_binary_classification = _module
basic_random_classification = _module
basic_random_classification_with_model_bundle = _module
basic_random_regression = _module
basic_random_regression_with_model_bundle = _module
poutyne_experiment = _module
poutyne_model = _module
pure_pytorch = _module
train_with_alert_callback = _module
train_with_custom_metrics = _module
train_with_metrics = _module
poutyne = _module
framework = _module
callbacks = _module
_utils = _module
best_model_restore = _module
checkpoint = _module
clip_grad = _module
color_formatting = _module
delay = _module
earlystopping = _module
gradient_logger = _module
gradient_tracker = _module
lambda_ = _module
logger = _module
lr_scheduler = _module
mlflow_logger = _module
notification = _module
periodic = _module
policies = _module
progress = _module
progress_bar = _module
terminate_on_nan = _module
wandb_logger = _module
experiment = _module
iterators = _module
metrics = _module
base = _module
decomposable = _module
metric_argument_indexing = _module
metrics_registering = _module
predefined = _module
accuracy = _module
fscores = _module
sklearn_metrics = _module
pytorch_loss_functions_registering = _module
utils = _module
model = _module
model_bundle = _module
optimizers = _module
layers = _module
utils = _module
plotting = _module
utils = _module
warning_manager = _module
setup = _module
callback_interface = _module
example_callbacks = _module
example_experiment = _module
example_metrics = _module
example_model = _module
example_pytorch = _module
long_pytorch_part1 = _module
long_pytorch_part2 = _module
long_pytorch_to_poutyne = _module
long_pytorch_to_poutyne_shorten = _module
long_pytorch_training_code = _module
tests = _module
context = _module
test_best_model_restore = _module
test_callback_list = _module
test_checkpoint = _module
test_color_formatting = _module
test_delay = _module
test_earlystopping = _module
test_gradient_logger = _module
test_lambda = _module
test_logger = _module
test_lr_scheduler = _module
test_lr_scheduler_checkpoint = _module
test_mlflow_logger = _module
test_notification = _module
test_optimizer_checkpoint = _module
test_periodic = _module
test_policies = _module
test_progress = _module
test_progress_bar = _module
test_terminate_on_nan = _module
test_tracker = _module
test_wandb_logger = _module
test_checkpointing = _module
test_experiment = _module
test_is_better_than = _module
test_tasks = _module
test_batch_metrics = _module
test_fscores = _module
test_metrics_model_integration = _module
test_sklearn_metrics = _module
base = _module
test_dict_output = _module
test_model = _module
test_model_optimizer = _module
test_model_progress = _module
test_multi_dict_io = _module
test_multi_gpu = _module
test_multi_input = _module
test_multi_io = _module
test_multi_output = _module
tools = _module
test_plotting = _module
test_utils = _module
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


import torch


import torch.nn as nn


from torch.utils.data import random_split


from torchvision.datasets import MNIST


from torchvision.transforms import ToTensor


import numpy as np


from sklearn.metrics import r2_score


import torch.optim as optim


from torch.utils.data.dataloader import DataLoader


from typing import Dict


import warnings


from typing import IO


from typing import Iterable


from typing import Union


from torch.nn.utils import clip_grad_norm_


from torch.nn.utils import clip_grad_value_


from typing import List


from typing import Optional


from typing import TextIO


from typing import Tuple


import inspect


from typing import BinaryIO


import torch.optim.lr_scheduler


from torch.optim import Optimizer


from abc import ABC


from abc import abstractmethod


from typing import Mapping


from typing import Callable


import torch.nn.functional as F


from collections import defaultdict


from typing import Any


from torch.nn.utils.rnn import PackedSequence


from torch.utils.data import DataLoader


import random


import numbers


from torch.utils.data import Dataset


from torch import nn


from torch.utils.data import TensorDataset


from typing import Sequence


import math


import pandas as pd


import numpy


from itertools import repeat


from math import ceil


from collections import OrderedDict


from copy import deepcopy


from torch.nn.utils.rnn import pad_sequence


from torch.nn.utils.rnn import pack_padded_sequence


class Metric(ABC, nn.Module):
    """
    The abstract class representing a metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    def forward(self, y_pred, y_true):
        """
        Update the current state of the metric and return the metric for the current batch. This method has to
        be implemented if the metric is used as a **batch metric**. If used as an epoch metric, it does not need to be
        implemented.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.

        Returns:
            The value of the metric for the current batch.
        """
        raise NotImplementedError

    def update(self, y_pred, y_true) ->None:
        """
        Update the current state of the metric. This method has to be implemented if the metric is used as an **epoch
        metric**. If used as a batch metric, it does not need to be implemented.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """
        Compute and return the metric. Should not modify the state of metric.
        """
        pass

    @abstractmethod
    def reset(self) ->None:
        """
        The information kept for the computation of the metric is cleaned so that a new epoch can be done.
        """
        pass


warning_settings = {'batch_size': 'warn'}


def get_batch_size(*values):
    """
    This method infers the batch size of a batch. Here is the inferring algorithm used to compute the
    batch size. The values are tested in order at each step of the inferring algorithm. If one
    step succeed for one of the values, the algorithm stops.

    - Step 1: if a value is a tensor or a Numpy array, then the ``len()`` is returned.
    - Step 2: if a value is a list or a tuple, then the ``len()`` of the first element is returned
      if it is a tensor or a Numpy array.
    - Step 3: if a value is a dict, then the value for the key ``'batch_size'`` is returned if it
      is of integral type.
    - Step 4: if a value is a dict, then the ``len()`` of the first element of ``.values()`` is
      returned if it is a tensor or a Numpy array.

    If inferring the batch size is not possible, the batch size is set to 1 and a warning is raised.
    To disable this warning, set

    .. code-block:: python

        from poutyne import warning_settings

        warning_settings['batch_size'] = 'ignore'



    Args:
        values: The values used for inferring the batch size.
    """

    def is_torch_or_numpy(v):
        return torch.is_tensor(v) or isinstance(v, np.ndarray)
    for v in values:
        if is_torch_or_numpy(v):
            return len(v)
    for v in values:
        if isinstance(v, (tuple, list)):
            if is_torch_or_numpy(v[0]):
                return len(v[0])
    for v in values:
        if isinstance(v, dict):
            if 'batch_size' in v and isinstance(v['batch_size'], numbers.Integral):
                return v['batch_size']
    for v in values:
        if isinstance(v, dict):
            first_value = list(v.values())[0]
            if is_torch_or_numpy(first_value):
                return len(first_value)
    if warning_settings['batch_size'] == 'warn':
        warnings.warn("Inferring the batch size is not possible. Hence, the batch size is set to 1. To disable this warning, set\nfrom poutyne import warning_settings\nwarning_settings['batch_size'] = 'ignore'\n\nHere is the inferring algorithm used to compute the batch size. The values are tested in order at each step of the inferring algorithm. If one step succeed for one of the values, the algorithm stops.\n\nStep 1: if a value is a tensor or a Numpy array, then the 'len()' is returned.\nStep 2: if a value is a list or a tuple, then the 'len()' of the first element is returned if it is a tensor or a Numpy array.\nStep 3: if a value is a dict, then the value for the key 'batch_size' is returned if it is of integral type.\nStep 4: if a value is a dict, then the 'len()' of the first element of '.values()' is returned if it is a tensor or a Numpy array.\n")
    return 1


class DecomposableMetric(Metric):

    def __init__(self, func, names):
        super().__init__()
        self.func = func
        self.names = [names] if isinstance(names, str) else names
        self.__name__ = self.names
        self.reset()

    def forward(self, y_pred, y_true):
        return self._update(y_pred, y_true)

    def update(self, y_pred, y_true):
        self._update(y_pred, y_true)

    def _update(self, y_pred, y_true):
        output = self.func(y_pred, y_true)
        np_output = self._output_to_array(output)
        batch_size = get_batch_size(y_pred, y_true)
        self.output_sums += np_output * batch_size
        self.size += batch_size
        return output

    def _output_to_array(self, output):
        if (torch.is_tensor(output) or isinstance(output, np.ndarray)) and len(output.shape) == 0:
            values = [float(output)]
        elif isinstance(output, Mapping):
            values = [float(output[name]) for name in self.names]
        elif isinstance(output, Iterable):
            values = [float(metric) for metric in output]
        else:
            values = [float(output)]
        return np.array(values)

    def compute(self):
        return self.output_sums / self.size

    def reset(self) ->None:
        self.output_sums = np.zeros(len(self.names))
        self.size = 0


class BatchMetric(ABC, nn.Module):

    def __init__(self, reduction: str='mean'):
        super().__init__()
        REDUCTIONS = ['none', 'mean', 'sum']
        if reduction not in REDUCTIONS:
            raise ValueError(f'Reduction is not in {REDUCTIONS}')
        self.reduction = reduction


def _get_registering_decorator(register_function):

    def decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
            register_function(args[0])
            return args[0]

        def register(func):
            register_function(func, args, **kwargs)
            return func
        return register
    return decorator


def clean_metric_func_name(name):
    name = name.lower()
    name = name[:-4] if name.endswith('loss') else name
    name = name.replace('_', '')
    return name


metric_funcs_dict = {}


def do_register_metric_func(func, names=None, unique_name=None):
    names = [func.__name__] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_metric_func_name(name) for name in names]
    if unique_name is None:
        update = {name: func for name in names}
    else:
        update = {name: (unique_name, func) for name in names}
    metric_funcs_dict.update(update)
    return names


register_metric_func = _get_registering_decorator(do_register_metric_func)


@register_metric_func('acc', 'accuracy')
def acc(y_pred, y_true, *, ignore_index=-100, reduction='mean'):
    """
    Computes the accuracy.

    This is a functional version of :class:`~poutyne.Accuracy`.

    See :class:`~poutyne.Accuracy` for details.
    """
    y_pred = y_pred.argmax(1)
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = (y_pred == y_true).float() * weights
    if reduction in ['mean', 'sum']:
        acc_pred = acc_pred.sum()
    if reduction == 'mean':
        acc_pred = acc_pred / num_labels
    return acc_pred * 100


class Accuracy(BatchMetric):
    """
    This metric computes the accuracy using a similar interface to
    :class:`~torch.nn.CrossEntropyLoss`.

    Args:
        ignore_index (int): Specifies a target value that is ignored and does not contribute
            to the accuracy. (Default value = -100)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.


    Possible string name:
        - ``'acc'``
        - ``'accuracy'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'acc'``
        - Validation: ``'val_acc'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of
          `K`-dimensional accuracy.
        - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of
          K-dimensional accuracy.
        - Output: The accuracy.
    """

    def __init__(self, *, ignore_index: int=-100, reduction: str='mean'):
        super().__init__(reduction)
        self.__name__ = 'acc'
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        return acc(y_pred, y_true, ignore_index=self.ignore_index, reduction=self.reduction)


@register_metric_func('binacc', 'binaryacc', 'binaryaccuracy')
def bin_acc(y_pred, y_true, *, threshold=0.0, reduction='mean'):
    """
    Computes the binary accuracy.

    This is a functional version of :class:`~poutyne.BinaryAccuracy`.

    See :class:`~poutyne.BinaryAccuracy` for details.
    """
    y_pred = (y_pred > threshold).float()
    acc_pred = (y_pred == y_true).float()
    if reduction == 'mean':
        acc_pred = acc_pred.mean()
    elif reduction == 'sum':
        acc_pred = acc_pred.sum()
    return acc_pred * 100


class BinaryAccuracy(BatchMetric):
    """
    This metric computes the accuracy using a similar interface to
    :class:`~torch.nn.BCEWithLogitsLoss`.

    Args:
        threshold (float): the threshold for class :math:`1`. Default value is ``0.``, that is a
            probability of ``sigmoid(0.) = 0.5``.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Possible string name:
        - ``'bin_acc'``
        - ``'binary_acc'``
        - ``'binary_accuracy'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_acc'``
        - Validation: ``'val_bin_acc'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: The binary accuracy.
    """

    def __init__(self, *, threshold: float=0.0, reduction: str='mean'):
        super().__init__(reduction)
        self.__name__ = 'bin_acc'
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        return bin_acc(y_pred, y_true, threshold=self.threshold, reduction=self.reduction)


def topk(y_pred, y_true, k, *, ignore_index=-100, reduction='mean'):
    """
    Computes the top-k accuracy.

    This is a functional version of :class:`~poutyne.TopKAccuracy`.

    See :class:`~poutyne.TopKAccuracy` for details.
    """
    topk_pred = y_pred.topk(k, dim=1)[1]
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    topk_acc = (y_true.unsqueeze(1) == topk_pred).any(1).float() * weights
    if reduction in ['mean', 'sum']:
        topk_acc = topk_acc.sum()
    if reduction == 'mean':
        topk_acc = topk_acc / num_labels
    return topk_acc * 100


class TopKAccuracy(BatchMetric):
    """
    This metric computes the top-k accuracy using a similar interface to
    :class:`~torch.nn.CrossEntropyLoss`.

    Args:
        k (int): Specifies the value of ``k`` in the top-k accuracy.
        ignore_index (int): Specifies a target value that is ignored and does not contribute
            to the top-k accuracy. (Default value = -100)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.


    Possible string name:
        - ``'top{k}'``
        - ``'top{k}_acc'``
        - ``'top{k}_accuracy'``

        for ``{k}`` from 1 to 10, 20, 30, ..., 100.

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'top{k}'``
        - Validation: ``'val_top{k}'``

        where ``{k}`` is replaced by the value of parameter ``k``.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of
          `K`-dimensional top-k accuracy.
        - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of
          K-dimensional top-k accuracy.
        - Output: The top-k accuracy.
    """

    def __init__(self, k: int, *, ignore_index: int=-100, reduction: str='mean'):
        super().__init__(reduction)
        self.__name__ = f'top{k}'
        self.k = k
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        return topk(y_pred, y_true, self.k, ignore_index=self.ignore_index, reduction=self.reduction)


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result
    result[mask] = 0.0
    return result


class FBeta(Metric):
    """
    The source code of this class is under the Apache v2 License and was copied from
    the AllenNLP project and has been modified.

    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    ``F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)``

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'{metric}_{average}'``
        - Validation: ``'val_{metric}_{average}'``

        where ``{metric}`` and ``{average}`` are replaced by the value of their
        respective parameters.

    Args:
        metric (Optional[str]): One of {'fscore', 'precision', 'recall'}.
            Whether to return the F-score, the precision or the recall. When not
            provided, all three metrics are returned. (Default value = None)
        average (Union[str, int]): One of {'micro' (default), 'macro', label_number}
            If the argument is of type integer, the score for this class (the label number) is calculated.
            Otherwise, this determines the type of averaging performed on the data:

            ``'binary'``:
                Calculate metrics with regard to a single class identified by the
                `pos_label` argument. This is equivalent to `average=pos_label` except
                that the binary mode is enforced, i.e. an exception will be raised if
                there are more than two prediction scores.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.

            (Default value = 'macro')
        beta (float):
            The strength of recall versus precision in the F-score. (Default value = 1.0)
        pos_label (int):
            The class with respect to which the metric is computed when ``average == 'binary'``. Otherwise, this
            argument has no effect. (Default value = 1)
        ignore_index (int): Specifies a target value that is ignored. This also works in combination with
            a mask if provided. (Default value = -100)
        threshold (float): Threshold for when there is a single score for each prediction. If a sigmoid output is used,
            this should be between 0 and 1. A suggested value would be 0.5. If a logits output is used, the threshold
            would be between -inf and inf. The suggested default value is 0 as to give a probability of 0.5 if a sigmoid
            output were used. (Default = 0)
        names (Optional[Union[str, List[str]]]): The names associated to the metrics. It is a string when
            a single metric is requested. It is a list of 3 strings if all metrics are requested.
            (Default value = None)
    """

    def __init__(self, *, metric: Optional[str]=None, average: Union[str, int]='macro', beta: float=1.0, pos_label: int=1, ignore_index: int=-100, threshold: float=0.0, names: Optional[Union[str, List[str]]]=None) ->None:
        super().__init__()
        self.metric_options = 'fscore', 'precision', 'recall'
        if metric is not None and metric not in self.metric_options:
            raise ValueError(f'`metric` has to be one of {self.metric_options}.')
        if metric in ('precision', 'recall') and beta != 1.0:
            warnings.warn(f'The use of the `beta` argument is useless with {repr(metric)}.')
        average_options = 'binary', 'micro', 'macro'
        if average not in average_options and not isinstance(average, int):
            raise ValueError(f'`average` has to be one of {average_options} or an integer.')
        if beta <= 0:
            raise ValueError('`beta` should be >0 in the F-beta score.')
        self._metric = metric
        self._average = average if average in average_options else None
        self._label = None
        if isinstance(average, int):
            self._label = average
        elif average == 'binary':
            self._label = pos_label
        self._beta = beta
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.__name__ = self._get_names(names)
        self.register_buffer('_true_positive_sum', None)
        self.register_buffer('_total_sum', None)
        self.register_buffer('_pred_sum', None)
        self.register_buffer('_true_sum', None)

    def _get_name(self, metric):
        name = metric
        if self._average is not None:
            name += '_' + self._average
        if self._label is not None:
            name += '_' + str(self._label)
        return name

    def _get_names(self, names):
        if self._metric is None:
            default_name = list(map(self._get_name, self.metric_options))
        else:
            default_name = self._get_name(self._metric)
        if names is not None:
            self._validate_supplied_names(names, default_name)
            return names
        return default_name

    def _validate_supplied_names(self, names, default_name):
        names_list = [names] if isinstance(names, str) else names
        default_name = [default_name] if isinstance(default_name, str) else default_name
        if len(names_list) != len(default_name):
            raise ValueError(f"`names` should contain names for the following metrics: {', '.join(default_name)}.")

    def forward(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->Union[float, Tuple[float]]:
        """
        Update the confusion matrix for calculating the F-score and compute the metrics for the current batch. See
        :meth:`FBeta.compute` for details on the return value.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, num_classes, ...).
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.

        Returns:
            A float if a single metric is set in the ``__init__`` or a tuple of floats (f-score, precision, recall) if
            all metrics are requested.
        """
        true_positive_sum, pred_sum, true_sum = self._update(y_pred, y_true)
        return self._compute(true_positive_sum, pred_sum, true_sum)

    def update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->None:
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, num_classes, ...).
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.
        """
        self._update(y_pred, y_true)

    def _update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->None:
        if isinstance(y_true, tuple):
            y_true, mask = y_true
            mask = mask.bool()
        else:
            mask = torch.ones_like(y_true).bool()
        if self.ignore_index is not None:
            mask *= y_true != self.ignore_index
        if y_pred.shape[0] == 1:
            y_pred, y_true, mask = y_pred.squeeze().unsqueeze(0), y_true.squeeze().unsqueeze(0), mask.squeeze().unsqueeze(0)
        else:
            y_pred, y_true, mask = y_pred.squeeze(), y_true.squeeze(), mask.squeeze()
        num_classes = 2
        if y_pred.shape != y_true.shape:
            num_classes = y_pred.size(1)
        if (y_true >= num_classes).any():
            raise ValueError(f'A gold label passed to FBetaMeasure contains an id >= {num_classes}, the number of classes.')
        if self._average == 'binary' and num_classes > 2:
            raise ValueError('When `average` is binary, the number of prediction scores must be 2.')
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
            self._true_sum = torch.zeros(num_classes, device=y_pred.device)
            self._pred_sum = torch.zeros(num_classes, device=y_pred.device)
            self._total_sum = torch.zeros(num_classes, device=y_pred.device)
        y_true = y_true.float()
        if y_pred.shape != y_true.shape:
            argmax_y_pred = y_pred.argmax(1).float()
        else:
            argmax_y_pred = (y_pred > self.threshold).float()
        true_positives = (y_true == argmax_y_pred) * mask
        true_positives_bins = y_true[true_positives]
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()
        pred_bins = argmax_y_pred[mask].long()
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=y_pred.device)
        y_true_bins = y_true[mask].long()
        if y_true.shape[0] != 0:
            true_sum = torch.bincount(y_true_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=y_pred.device)
        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum()
        return true_positive_sum, pred_sum, true_sum

    def compute(self) ->Union[float, Tuple[float]]:
        """
        Returns either a float if a single metric is set in the ``__init__`` or a tuple
        of floats (f-score, precision, recall) if all metrics are requested.
        """
        if self._true_positive_sum is None:
            raise RuntimeError('You never call this metric before.')
        return self._compute(self._true_positive_sum, self._pred_sum, self._true_sum)

    def _compute(self, tp_sum, pred_sum, true_sum):
        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()
        beta2 = self._beta ** 2
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0
        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        if self._label is not None:
            precision = precision[self._label]
            recall = recall[self._label]
            fscore = fscore[self._label]
        if self._metric is None:
            return [fscore.item(), precision.item(), recall.item()]
        if self._metric == 'fscore':
            return fscore.item()
        if self._metric == 'precision':
            return precision.item()
        return recall.item()

    def reset(self) ->None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None


def _raise_invalid_use_of_beta(name, kwargs):
    if 'beta' in kwargs:
        raise ValueError(f'The use of the `beta` argument with {name} is invalid. Use FBeta instead.')


pattern1 = re.compile('(.)([A-Z][a-z]+)')


pattern2 = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(name):
    """
    Convert CamelCase to snake_case.

    From https://stackoverflow.com/a/1176023
    """
    name = pattern1.sub('\\1_\\2', name)
    return pattern2.sub('\\1_\\2', name).lower()


def clean_metric_class_name(name):
    name = name.lower()
    name = name[:-5] if name.endswith('score') else name
    name = name.replace('_', '')
    return name


metric_classes_dict = {}


def do_register_metric_class(clz, names=None, unique_name=None):
    names = [camel_to_snake(clz.__name__)] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_metric_class_name(name) for name in names]
    if unique_name is None:
        update = {name: clz for name in names}
    else:
        update = {name: (unique_name, clz) for name in names}
    metric_classes_dict.update(update)
    return names


register_metric_class = _get_registering_decorator(do_register_metric_class)


class F1(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'fscore'`` and ``beta == 1``.

    Possible string name:
        - ``'f1'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'fscore_{average}'``
        - Validation: ``'val_fscore_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        _raise_invalid_use_of_beta('F1', kwargs)
        super().__init__(metric='fscore', **kwargs)


class Precision(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'precision'``.

    Possible string name:
        - ``'precision'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'precision_{average}'``
        - Validation: ``'val_precision_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(metric='precision', **kwargs)


class Recall(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'recall'``.

    Possible string name:
        - ``'recall'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'recall_{average}'``
        - Validation: ``'val_recall_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(metric='recall', **kwargs)


class BinaryF1(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'fscore'``, ``average='binary'`` and ``beta == 1``.

    Possible string name:
        - ``'binary_f1'``
        - ``'bin_f1'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_fscore'``
        - Validation: ``'val_bin_fscore'``
    """

    def __init__(self, **kwargs):
        _raise_invalid_use_of_beta('BinaryF1', kwargs)
        kwargs = {'names': 'bin_fscore', **kwargs}
        super().__init__(metric='fscore', average='binary', **kwargs)


class BinaryPrecision(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'precision'`` and ``average='binary'``.

    Possible string name:
        - ``'binary_precision'``
        - ``'bin_precision'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_precision'``
        - Validation: ``'val_bin_precision'``
    """

    def __init__(self, **kwargs):
        kwargs = {'names': 'bin_precision', **kwargs}
        super().__init__(metric='precision', average='binary', **kwargs)


class BinaryRecall(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'recall'`` and ``average='binary'``.

    Possible string name:
        - ``'binary_recall'``
        - ``'bin_recall'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_recall'``
        - Validation: ``'val_bin_recall'``
    """

    def __init__(self, **kwargs):
        kwargs = {'names': 'bin_recall', **kwargs}
        super().__init__(metric='recall', average='binary', **kwargs)


class SKLearnMetrics(Metric):
    """
    Wrap metrics with Scikit-learn-like interface
    (``metric(y_true, y_pred, sample_weight=sample_weight, **kwargs)``).
    The ``SKLearnMetrics`` object has to keep in memory the ground truths and
    predictions so that in can compute the metric at the end.

    Example:
        .. code-block:: python

            from sklearn.metrics import roc_auc_score, average_precision_score
            from poutyne import SKLearnMetrics
            my_epoch_metric = SKLearnMetrics([roc_auc_score, average_precision_score])

    Args:
        funcs (Union[Callable, List[Callable]]): A metric or a list of metrics with a
            scikit-learn-like interface.
        kwargs (Optional[Union[dict, List[dict]]]): Optional dictionary of list of dictionaries
            corresponding to keyword arguments to pass to each corresponding metric.
            (Default value = None)
        names (Optional[Union[str, List[str]]]): Optional string or list of strings corresponding to
            the names given to the metrics. By default, the names are the names of the functions.
    """

    def __init__(self, funcs: Union[Callable, List[Callable]], kwargs: Optional[Union[dict, List[dict]]]=None, names: Optional[Union[str, List[str]]]=None) ->None:
        super().__init__()
        self.funcs = funcs if isinstance(funcs, (list, tuple)) else [funcs]
        self.kwargs = self._validate_kwargs(kwargs)
        self.__name__ = self._validate_names(names)
        self.reset()

    def _validate_kwargs(self, kwargs):
        if kwargs is not None:
            kwargs = kwargs if isinstance(kwargs, (list, tuple)) else [kwargs]
            if kwargs is not None and len(self.funcs) != len(kwargs):
                raise ValueError('`kwargs` has to have the same length as `funcs` when provided')
        else:
            kwargs = [{}] * len(self.funcs) if kwargs is None else kwargs
        return kwargs

    def _validate_names(self, names):
        if names is not None:
            names = names if isinstance(names, (list, tuple)) else [names]
            if len(self.funcs) != len(names):
                raise ValueError('`names` has to have the same length as `funcs` when provided')
        else:
            names = [func.__name__ for func in self.funcs]
        return names

    def forward(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->None:
        """
        Accumulate the predictions, ground truths and sample weights if any, and compute the metric for the current
        batch.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of the shape expected by
                the metric functions passed to the class.
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of ground truths of the shape expected by
                the metric functions passed to the class.
                It can also be a tuple with two tensors, the first being the
                ground truths and the second corresponding the ``sample_weight``
                argument passed to the metric functions in Scikit-Learn.
        """
        y_pred, y_true, sample_weight = self._update(y_pred, y_true)
        return self._compute(y_true, y_pred, sample_weight)

    def update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->None:
        """
        Accumulate the predictions, ground truths and sample weights if any.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of the shape expected by
                the metric functions passed to the class.
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of ground truths of the shape expected by
                the metric functions passed to the class.
                It can also be a tuple with two tensors, the first being the
                ground truths and the second corresponding the ``sample_weight``
                argument passed to the metric functions in Scikit-Learn.
        """
        self._update(y_pred, y_true)

    def _update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->None:
        y_pred = y_pred.cpu().numpy()
        self.y_pred_list.append(y_pred)
        sample_weight = None
        if isinstance(y_true, (tuple, list)):
            y_true, sample_weight = y_true
            sample_weight = sample_weight.cpu().numpy()
            self.sample_weight_list.append(sample_weight)
        y_true = y_true.cpu().numpy()
        self.y_true_list.append(y_true)
        return y_pred, y_true, sample_weight

    def compute(self) ->Dict:
        """
        Returns the metrics as a dictionary with the names as keys.
        """
        sample_weight = None
        if len(self.sample_weight_list) != 0:
            sample_weight = np.concatenate(self.sample_weight_list)
        y_pred = np.concatenate(self.y_pred_list)
        y_true = np.concatenate(self.y_true_list)
        return self._compute(y_true, y_pred, sample_weight)

    def _compute(self, y_true, y_pred, sample_weight):
        return {name: func(y_true, y_pred, sample_weight=sample_weight, **kwargs) for name, func, kwargs in zip(self.__name__, self.funcs, self.kwargs)}

    def reset(self) ->None:
        self.y_true_list = []
        self.y_pred_list = []
        self.sample_weight_list = []


class Lambda(nn.Module):
    """
    Applies a function to the input tensor.

    Args:
        func (Callable[[~torch.Tensor], ~torch.Tensor]): The function to apply.

    Example:

        .. code-block:: python

            # Alternate version to the ``nn.Flatten`` module.
            my_flatten = Lambda(lambda x: x.flatten(1))

    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *x):
        return self.func(*x)


class MnistLogistic(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb.matmul(self.weights) + self.bias


class EarlyStoppingDummyMetric(Metric):
    __name__ = 'dummy'

    def __init__(self, values) ->None:
        super().__init__()
        self.values = values
        self.current_epoch = None

    def update(self, y_pred, y_true):
        pass

    def compute(self):
        return self.values[self.current_epoch - 1]

    def reset(self) ->None:
        pass


class ConstMetric(Metric):

    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

    def compute(self):
        return self.value

    def reset(self):
        pass


class SomeMetricName(ConstMetric):

    def compute(self):
        return torch.FloatTensor([self.value])

    def reset(self):
        pass


class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super().__init__()
        inputs = []
        for _ in range(num_input):
            inputs.append(nn.Linear(1, 1))
        self.inputs = nn.ModuleList(inputs)
        outputs = []
        for _ in range(num_output):
            outputs.append(nn.Linear(num_input, 1))
        self.outputs = nn.ModuleList(outputs)

    def forward(self, *x):
        inp_to_cat = []
        for i, inp in enumerate(self.inputs):
            inp_to_cat.append(inp(x[i]))
        inp_cat = torch.cat(inp_to_cat, dim=1)
        outputs = []
        for out in self.outputs:
            outputs.append(out(inp_cat))
        outputs = outputs if len(outputs) > 1 else outputs[0]
        return outputs


class DictOutputModel(nn.Module):
    """Model to test multiple dictionnary output"""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1, 1)
        self.output1 = nn.Linear(1, 1)
        self.output2 = nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.output1(self.input(x))
        out2 = self.output2(self.input(x))
        return {'out1': out1, 'out2': out2}


class DictIOModel(nn.Module):
    """Model to test multiple dict input/output"""

    def __init__(self, input_keys, output_keys):
        super().__init__()
        assert len(input_keys) == len(output_keys)
        inputs = {k: nn.Linear(1, 1) for k in input_keys}
        self.inputs = nn.ModuleDict(inputs)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, x):
        return {out_k: self.inputs[in_k](x[in_k]) for in_k, out_k in zip(self.input_keys, self.output_keys)}


some_constant_metric_value = 3


class SomeConstantMetric(Metric):

    def update(self, y_pred, y_true):
        pass

    def compute(self):
        return torch.FloatTensor([some_constant_metric_value])

    def reset(self):
        pass


class SomeMetric(Metric):

    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def update(self, y_pred, y_true):
        self.increment += 1

    def compute(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass


class SomeBatchMetric(Metric):

    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def forward(self, y_pred, y_true):
        self.increment += 1
        return self.increment

    def compute(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Accuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryAccuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryF1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryPrecision,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryRecall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (F1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FBeta,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Precision,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Recall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SomeBatchMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TopKAccuracy,
     lambda: ([], {'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_GRAAL_Research_poutyne(_paritybench_base):
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

