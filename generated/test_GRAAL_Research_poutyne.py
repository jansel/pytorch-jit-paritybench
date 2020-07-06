import sys
_module = sys.modules[__name__]
del sys
conf = _module
poutyne = _module
framework = _module
callbacks = _module
_utils = _module
best_model_restore = _module
checkpoint = _module
clip_grad = _module
delay = _module
earlystopping = _module
logger = _module
lr_scheduler = _module
periodic = _module
policies = _module
progress = _module
terminate_on_nan = _module
exceptions = _module
experiment = _module
iterators = _module
metrics = _module
epoch_metrics = _module
base = _module
fscores = _module
metrics = _module
utils = _module
model = _module
optimizers = _module
warning_manager = _module
layers = _module
utils = _module
utils = _module
version = _module
setup = _module
tests = _module
context = _module
test_best_model_restore = _module
test_checkpoint = _module
test_delay = _module
test_earlystopping = _module
test_logger = _module
test_lr_scheduler = _module
test_lr_scheduler_checkpoint = _module
test_optimizer_checkpoint = _module
test_policies = _module
test_fscores = _module
test_model = _module
test_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.nn.utils import clip_grad_norm_


from torch.nn.utils import clip_grad_value_


import inspect


import torch.optim.lr_scheduler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import warnings


import numpy as np


import torch


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


import torch.nn.functional as F


from collections import defaultdict


from torch.utils.data import DataLoader


import torch.optim as optim


import random


from torch.utils.data import Dataset


import numpy


from math import ceil


class EpochMetric(ABC, nn.Module):
    """
    The abstract class representing a epoch metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        To define the behavior of the metric when called.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        Compute and return the metric.
        """
        pass


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


class FBeta(EpochMetric):
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

    Args:
        metric (str): One of {'fscore', 'precision', 'recall'}.
            Wheter to return the F-score, the precision or the recall. (Default value = 'fscore')
        average (Union[str, int]): One of {'micro' (default), 'macro', label_number}
            If the argument is of type integer, the score for this class (the label number) is calculated.
            Otherwise, this determines the type of averaging performed on the data:

            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.

            (Default value = 'micro')

        beta (float):
            The strength of recall versus precision in the F-score. (Default value = 1.0)
    """

    def __init__(self, metric: str='fscore', average: str='micro', beta: float=1.0) ->None:
        super().__init__()
        metric_options = 'fscore', 'precision', 'recall'
        if metric not in metric_options:
            raise ValueError('`metric` has to be one of {}.'.format(metric_options))
        average_options = 'micro', 'macro'
        if average not in average_options and not isinstance(average, int):
            raise ValueError('`average` has to be one of {} or an integer.'.format(average_options))
        if beta <= 0:
            raise ValueError('`beta` should be >0 in the F-beta score.')
        self._metric = metric
        self._average = average if average in average_options else None
        self._label = average if isinstance(average, int) else None
        self._beta = beta
        if self._average is not None:
            self.__name__ = self._metric + '_' + self._average
        else:
            self.__name__ = self._metric + '_' + str(self._label)
        self.register_buffer('_true_positive_sum', None)
        self.register_buffer('_total_sum', None)
        self.register_buffer('_pred_sum', None)
        self.register_buffer('_true_sum', None)

    def forward(self, y_pred, y_true):
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_pred : Predictions of the model.
            y_true : A tensor of the gold labels. Can also be a tuple of gold_label and a mask.
        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, ..., num_classes).
            y_true Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.
        """
        mask = None
        if isinstance(y_true, tuple):
            y_true, mask = y_true
        num_classes = y_pred.size(1)
        if (y_true >= num_classes).any():
            raise ValueError('A gold label passed to FBetaMeasure contains an id >= {}, the number of classes.'.format(num_classes))
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
            self._true_sum = torch.zeros(num_classes, device=y_pred.device)
            self._pred_sum = torch.zeros(num_classes, device=y_pred.device)
            self._total_sum = torch.zeros(num_classes, device=y_pred.device)
        if mask is None:
            mask = torch.ones_like(y_true)
        mask = mask
        y_true = y_true.float()
        argmax_y_pred = y_pred.max(dim=1)[1].float()
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

    def get_metric(self):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]

        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        if self._true_positive_sum is None:
            raise RuntimeError('You never call this metric before.')
        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum
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
        self.reset()
        if self._label is not None:
            precision = precision[self._label]
            recall = recall[self._label]
            fscore = fscore[self._label]
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


class F1(FBeta):
    """
    Alias class for FBeta where ``metric == 'fscore'`` and ``beta == 1``.
    """

    def __init__(self, average='micro'):
        super().__init__(metric='fscore', average=average, beta=1)


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
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SomeEpochMetric(EpochMetric):

    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def forward(self, y_pred, y_true):
        self.increment += 1

    def get_metric(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value


some_constant_epoch_metric_value = 3


class SomeConstantEpochMetric(EpochMetric):

    def forward(self, y_pred, y_true):
        pass

    def get_metric(self):
        return torch.FloatTensor([some_constant_epoch_metric_value])


class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super(MultiIOModel, self).__init__()
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
        super(DictOutputModel, self).__init__()
        self.input = nn.Linear(1, 1)
        self.output1 = nn.Linear(1, 1)
        self.output2 = nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.output1(self.input(x))
        out2 = self.output2(self.input(x))
        return {'out1': out1, 'out2': out2}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DictOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([1, 1])], {}),
     True),
    (Lambda,
     lambda: ([], {'func': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SomeConstantEpochMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SomeEpochMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
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

