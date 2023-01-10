import sys
_module = sys.modules[__name__]
del sys
conf = _module
lenet_cifar10 = _module
lenet_mnist = _module
resnet_cifar10 = _module
resnet_cifar100 = _module
classification_cifar10_cnn = _module
classification_mnist_tree_ensemble = _module
fast_geometric_ensemble_cifar10_resnet18 = _module
regression_YearPredictionMSD_mlp = _module
snapshot_ensemble_cifar10_resnet18 = _module
setup = _module
torchensemble = _module
_base = _module
_constants = _module
adversarial_training = _module
bagging = _module
fast_geometric = _module
fusion = _module
gradient_boosting = _module
snapshot_ensemble = _module
soft_gradient_boosting = _module
test_adversarial_training = _module
test_all_models = _module
test_all_models_multi_input = _module
test_fixed_dataloder = _module
test_logging = _module
test_neural_tree_ensemble = _module
test_operator = _module
test_set_optimizer = _module
test_set_scheduler = _module
test_tb_logging = _module
test_training_params = _module
utils = _module
dataloder = _module
io = _module
logging = _module
operator = _module
set_module = _module
voting = _module

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


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import numbers


from torch.nn import functional as F


from sklearn.preprocessing import scale


from sklearn.datasets import load_svmlight_file


from torch.utils.data import TensorDataset


import abc


import copy


import logging


import warnings


import numpy as np


import math


from torch.optim.lr_scheduler import LambdaLR


from numpy.testing import assert_array_equal


from numpy.testing import assert_array_almost_equal


from typing import List


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


_tb_logger = None


def get_tb_logger():
    return _tb_logger


class BaseModule(nn.Module):
    """Base class for all ensembles.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def __init__(self, estimator, n_estimators, estimator_args=None, cuda=True, n_jobs=None):
        super(BaseModule, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        if estimator_args and not isinstance(estimator, type):
            msg = 'The input `estimator_args` will have no effect since `estimator` is already an object after instantiation.'
            warnings.warn(msg, RuntimeWarning)
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()
        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def __len__(self):
        """
        Return the number of base estimators in the ensemble. The real number
        of base estimators may not match `self.n_estimators` because of the
        early stopping stage in several ensembles such as Gradient Boosting.
        """
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the `index`-th base estimator in the ensemble."""
        return self.estimators_[index]

    @abc.abstractmethod
    def _decide_n_outputs(self, train_loader):
        """Decide the number of outputs according to the `train_loader`."""

    def _make_estimator(self):
        """Make and configure a copy of `self.base_estimator_`."""
        if not isinstance(self.base_estimator_, type):
            estimator = copy.deepcopy(self.base_estimator_)
        elif self.estimator_args is None:
            estimator = self.base_estimator_()
        else:
            estimator = self.base_estimator_(**self.estimator_args)
        return estimator

    def _validate_parameters(self, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""
        if not epochs > 0:
            msg = 'The number of training epochs should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))
        if not log_interval > 0:
            msg = 'The number of batches to wait before printing the training status should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

    def set_criterion(self, criterion):
        """Set the training criterion."""
        self._criterion = criterion

    def set_optimizer(self, optimizer_name, **kwargs):
        """Set the parameter optimizer."""
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    def set_scheduler(self, scheduler_name, **kwargs):
        """Set the learning rate scheduler."""
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @abc.abstractmethod
    def forward(self, *x):
        """
        Implementation on the data forwarding in the ensemble. Notice
        that the input ``x`` should be a data batch instead of a standalone
        data loader that contains many data batches.
        """

    @abc.abstractmethod
    def fit(self, train_loader, epochs=100, log_interval=100, test_loader=None, save_model=True, save_dir=None):
        """
        Implementation on the training stage of the ensemble.
        """

    @torch.no_grad()
    def predict(self, *x):
        """Docstrings decorated by downstream ensembles."""
        self.eval()
        x_device = []
        for data in x:
            if isinstance(data, torch.Tensor):
                x_device.append(data)
            elif isinstance(data, np.ndarray):
                x_device.append(torch.Tensor(data))
            else:
                msg = 'The type of input X should be one of {{torch.Tensor, np.ndarray}}.'
                raise ValueError(msg)
        pred = self.forward(*x_device)
        pred = pred.cpu()
        return pred


class BaseTree(nn.Module):
    """Fast implementation of soft decision tree in PyTorch, copied from:
    `https://github.com/xuyxu/Soft-Decision-Tree/blob/master/SDT.py`
    """

    def __init__(self, input_dim, output_dim, depth=5, lamda=0.001, cuda=False):
        super(BaseTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self._validate_parameters()
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth
        self.penalty_list = [(self.lamda * 2 ** -depth) for depth in range(0, self.depth)]
        self.inner_nodes = nn.Sequential(nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False), nn.Sigmoid())
        self.leaf_nodes = nn.Linear(self.leaf_node_num_, self.output_dim, bias=False)

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""
        batch_size = X.size()[0]
        X = self._data_augment(X)
        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0)
        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        mu = _mu.view(batch_size, self.leaf_node_num_)
        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """Compute the regularization term for internal nodes"""
        penalty = torch.tensor(0.0)
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))
        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) / torch.sum(_mu[:, node // 2], dim=0)
            coeff = self.penalty_list[layer_idx]
            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1)
        X = torch.cat((bias, X), 1)
        return X

    def _validate_parameters(self):
        if not self.depth > 0:
            msg = 'The tree depth should be strictly positive, but got {}instead.'
            raise ValueError(msg.format(self.depth))
        if not self.lamda >= 0:
            msg = 'The coefficient of the regularization term should not be negative, but got {} instead.'
            raise ValueError(msg.format(self.lamda))


class BaseTreeEnsemble(BaseModule):

    def __init__(self, n_estimators=10, depth=5, lamda=0.001, cuda=False, n_jobs=None):
        super(BaseModule, self).__init__()
        self.base_estimator_ = BaseTree
        self.n_estimators = n_estimators
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()
        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _decidce_n_inputs(self, train_loader):
        """Decide the input dimension according to the `train_loader`."""
        for _, elem in enumerate(train_loader):
            data = elem[0]
            n_samples = data.size(0)
            data = data.view(n_samples, -1)
            return data.size(1)

    def _make_estimator(self):
        """Make and configure a soft decision tree."""
        estimator = BaseTree(input_dim=self.n_inputs, output_dim=self.n_outputs, depth=self.depth, lamda=self.lamda, cuda=self.device == torch.device('cuda'))
        return estimator


def split_data_target(element, device, logger=None):
    """Split elements in dataloader according to pre-defined rules."""
    if not (isinstance(element, list) or isinstance(element, tuple)):
        msg = 'Invalid dataloader, please check if the input dataloder is valid.'
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    if len(element) == 2:
        data, target = element[0], element[1]
        return [data], target
    elif len(element) > 2:
        data, target = element[:-1], element[-1]
        data_device = [tensor for tensor in data]
        return data_device, target
    else:
        msg = 'The input dataloader should at least contain two tensors - data and target.'
        if logger:
            logger.error(msg)
        raise ValueError(msg)


class BaseClassifier(BaseModule):
    """Base class for all ensemble classifiers.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def _decide_n_outputs(self, train_loader):
        """
        Decide the number of outputs according to the `train_loader`.
        The number of outputs equals the number of distinct classes for
        classifiers.
        """
        if hasattr(train_loader.dataset, 'classes'):
            n_outputs = len(train_loader.dataset.classes)
        else:
            labels = []
            for _, elem in enumerate(train_loader):
                _, target = split_data_target(elem, self.device)
                labels.append(target)
            labels = torch.unique(torch.cat(labels))
            n_outputs = labels.size(0)
        return n_outputs

    @torch.no_grad()
    def evaluate(self, test_loader, return_loss=False):
        """Docstrings decorated by downstream models."""
        self.eval()
        correct = 0
        total = 0
        loss = 0.0
        for _, elem in enumerate(test_loader):
            data, target = split_data_target(elem, self.device)
            output = self.forward(*data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            loss += self._criterion(output, target)
        acc = 100 * correct / total
        loss /= len(test_loader)
        if return_loss:
            return acc, float(loss)
        return acc


class BaseRegressor(BaseModule):
    """Base class for all ensemble regressors.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def _decide_n_outputs(self, train_loader):
        """
        Decide the number of outputs according to the `train_loader`.
        The number of outputs equals the number of target variables for
        regressors (e.g., `1` in univariate regression).
        """
        for _, elem in enumerate(train_loader):
            _, target = split_data_target(elem, self.device)
            if len(target.size()) == 1:
                n_outputs = 1
            else:
                n_outputs = target.size(1)
            break
        return n_outputs

    @torch.no_grad()
    def evaluate(self, test_loader):
        """Docstrings decorated by downstream ensembles."""
        self.eval()
        loss = 0.0
        for _, elem in enumerate(test_loader):
            data, target = split_data_target(elem, self.device)
            output = self.forward(*data)
            loss += self._criterion(output, target)
        return float(loss) / len(test_loader)


class _BaseAdversarialTraining(BaseModule):

    def _validate_parameters(self, epochs, epsilon, log_interval):
        """Validate hyper-parameters on training the ensemble."""
        if not epochs > 0:
            msg = 'The number of training epochs = {} should be strictly positive.'
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))
        if not 0 < epsilon <= 1:
            msg = 'The step used to generate adversarial samples in FGSM should be in the range (0, 1], but got {} instead.'
            self.logger.error(msg.format(epsilon))
            raise ValueError(msg.format(epsilon))
        if not log_interval > 0:
            msg = 'The number of batches to wait before printting the training status should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A data loader that contains the training data.
    epochs : int, default=100
        The number of training epochs per base estimator.
    use_reduction_sum : bool, default=True
        Whether to set ``reduction="sum"`` for the internal mean squared
        error used to fit each base estimator.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A data loader that contains the evaluating data.

        - If ``None``, no validation is conducted after each base
          estimator being trained.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each base estimator being trained.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble containing
          ``n_estimators`` base estimators will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _adversarial_training_model_doc(header, item='fit'):
    """
    Decorator on obtaining documentation for different adversarial training
    models.
    """

    def get_doc(item):
        """Return selected item"""
        __doc = {'fit': __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


def _parallel_fit_per_epoch(train_loader, estimator, cur_lr, optimizer, criterion, idx, epoch, log_interval, device, is_classification):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    if cur_lr:
        set_module.update_lr(optimizer, cur_lr)
    for batch_idx, elem in enumerate(train_loader):
        data, target = io.split_data_target(elem, device)
        batch_size = data[0].size(0)
        optimizer.zero_grad()
        output = estimator(*data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if is_classification:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
                msg = 'Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f} | Correct: {:d}/{:d}'
                None
            else:
                msg = 'Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}'
                None
    return estimator, optimizer, loss


def torchensemble_model_doc(header='', item='model'):
    """
    A decorator on obtaining documentation for different methods in the
    ensemble. This decorator is modified from `sklearn.py` in XGBoost.

    Parameters
    ----------
    header: string
       Introduction to the decorated class or method.
    item : string
       Type of the docstring item.
    """

    def get_doc(item):
        """Return the selected item."""
        __doc = {'model': const.__model_doc, 'seq_model': const.__seq_model_doc, 'tree_ensmeble_model': const.__tree_ensemble_doc, 'fit': const.__fit_doc, 'predict': const.__predict_doc, 'set_optimizer': const.__set_optimizer_doc, 'set_scheduler': const.__set_scheduler_doc, 'set_criterion': const.__set_criterion_doc, 'classifier_forward': const.__classification_forward_doc, 'classifier_evaluate': const.__classification_evaluate_doc, 'regressor_forward': const.__regression_forward_doc, 'regressor_evaluate': const.__regression_evaluate_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


def _get_bagging_dataloaders(original_dataloader, n_estimators):
    dataset = original_dataloader.dataset
    dataloaders = []
    for i in range(n_estimators):
        indices = torch.randint(high=len(dataset), size=(len(dataset),), dtype=torch.int64)
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=original_dataloader.batch_size, num_workers=original_dataloader.num_workers, collate_fn=original_dataloader.collate_fn, shuffle=True)
        dataloaders.append(dataloader)
    return dataloaders


def _fast_geometric_model_doc(header, item='fit'):
    """
    Decorator on obtaining documentation for different fast geometric models.
    """

    def get_doc(item):
        """Return selected item"""
        __doc = {'fit': __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


class _BaseGradientBoosting(BaseModule):

    def __init__(self, estimator, n_estimators, estimator_args=None, shrinkage_rate=1.0, cuda=True):
        super(BaseModule, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        if estimator_args and not isinstance(estimator, type):
            msg = 'The input `estimator_args` will have no effect since `estimator` is already an object after instantiation.'
            warnings.warn(msg, RuntimeWarning)
        self.shrinkage_rate = shrinkage_rate
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()
        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _validate_parameters(self, epochs, log_interval, early_stopping_rounds):
        """Validate hyper-parameters on training the ensemble."""
        if not epochs > 0:
            msg = 'The number of training epochs = {} should be strictly positive.'
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))
        if not log_interval > 0:
            msg = 'The number of batches to wait before printting the training status should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))
        if not early_stopping_rounds >= 1:
            msg = 'The number of tolerant rounds before triggering the early stopping should at least be 1, but got {} instead.'
            self.logger.error(msg.format(early_stopping_rounds))
            raise ValueError(msg.format(early_stopping_rounds))
        if not 0 < self.shrinkage_rate <= 1:
            msg = 'The shrinkage rate should be in the range (0, 1], but got {} instead.'
            self.logger.error(msg.format(self.shrinkage_rate))
            raise ValueError(msg.format(self.shrinkage_rate))

    @abc.abstractmethod
    def _handle_early_stopping(self, test_loader, est_idx):
        """Decide whether to trigger the internal counter on early stopping."""

    def _staged_forward(self, est_idx, *x):
        """
        Return the accumulated outputs from the first `est_idx+1` base
        estimators.
        """
        if est_idx >= self.n_estimators:
            msg = 'est_idx = {} should be an integer smaller than the number of base estimators = {}.'
            self.logger.error(msg.format(est_idx, self.n_estimators))
            raise ValueError(msg.format(est_idx, self.n_estimators))
        outputs = [estimator(*x) for estimator in self.estimators_[:est_idx + 1]]
        out = op.sum_with_multiplicative(outputs, self.shrinkage_rate)
        return out

    def fit(self, train_loader, epochs=100, use_reduction_sum=True, log_interval=100, test_loader=None, early_stopping_rounds=2, save_model=True, save_dir=None):
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval, early_stopping_rounds)
        self.n_outputs = self._decide_n_outputs(train_loader)
        criterion = nn.MSELoss(reduction='sum') if use_reduction_sum else nn.MSELoss()
        n_counter = 0
        for est_idx, estimator in enumerate(self.estimators_):
            learner_optimizer = set_module.set_optimizer(estimator, self.optimizer_name, **self.optimizer_args)
            if self.use_scheduler_:
                learner_scheduler = set_module.set_scheduler(learner_optimizer, self.scheduler_name, **self.scheduler_args)
            estimator.train()
            total_iters = 0
            for epoch in range(epochs):
                for batch_idx, elem in enumerate(train_loader):
                    data, target = io.split_data_target(elem, self.device)
                    residual = self._pseudo_residual(est_idx, target, *data)
                    output = estimator(*data)
                    loss = criterion(output, residual)
                    learner_optimizer.zero_grad()
                    loss.backward()
                    learner_optimizer.step()
                    if batch_idx % log_interval == 0:
                        msg = 'Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d} | RegLoss: {:.5f}'
                        self.logger.info(msg.format(est_idx, epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar('gradient_boosting/Est_{}/Train_Loss'.format(est_idx), loss, total_iters)
                    total_iters += 1
                if self.use_scheduler_:
                    if self.scheduler_name == 'ReduceLROnPlateau':
                        learner_scheduler.step(loss)
                    else:
                        learner_scheduler.step()
            if test_loader:
                flag, test_metric_val = self._handle_early_stopping(test_loader, est_idx)
                if flag:
                    n_counter += 1
                    msg = 'Early stopping counter: {} out of {}'
                    self.logger.info(msg.format(n_counter, early_stopping_rounds))
                    if n_counter == early_stopping_rounds:
                        msg = 'Handling early stopping...'
                        self.logger.info(msg)
                        offset = est_idx - n_counter
                        self.estimators_ = self.estimators_[:offset + 1]
                        self.n_estimators = len(self.estimators_)
                        break
                else:
                    n_counter = 0
        msg = 'The optimal number of base estimators: {}'
        self.logger.info(msg.format(len(self.estimators_)))
        if save_model:
            io.save(self, save_dir, self.logger)


__model_doc = """
    Parameters
    ----------
    estimator : torch.nn.Module
        The class or object of your base estimator.

        - If :obj:`class`, it should inherit from :mod:`torch.nn.Module`.
        - If :obj:`object`, it should be instantiated from a class inherited
          from :mod:`torch.nn.Module`.
    n_estimators : int
        The number of base estimators in the ensemble.
    estimator_args : dict, default=None
        The dictionary of hyper-parameters used to instantiate base
        estimators. This parameter will have no effect if ``estimator`` is a
        base estimator object after instantiation.
    shrinkage_rate : float, default=1
        The shrinkage rate used in gradient boosting.
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.
    n_jobs : int, default=None
        The number of workers for training the ensemble. This input
        argument is used for parallel ensemble methods such as
        :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
        than ``1`` enables ``n_jobs`` base estimators to be trained
        simultaneously.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        An internal container that stores all fitted base estimators.
"""


def _gradient_boosting_model_doc(header, item='model'):
    """
    Decorator on obtaining documentation for different gradient boosting
    models.
    """

    def get_doc(item):
        """Return the selected item"""
        __doc = {'model': __model_doc, 'fit': __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


class _BaseSnapshotEnsemble(BaseModule):

    def __init__(self, estimator, n_estimators, estimator_args=None, cuda=True):
        super(BaseModule, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        if estimator_args and not isinstance(estimator, type):
            msg = 'The input `estimator_args` will have no effect since `estimator` is already an object after instantiation.'
            warnings.warn(msg, RuntimeWarning)
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()
        self.estimators_ = nn.ModuleList()

    def _validate_parameters(self, lr_clip, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""
        if lr_clip:
            if not (isinstance(lr_clip, list) or isinstance(lr_clip, tuple)):
                msg = 'lr_clip should be a list or tuple with two elements.'
                self.logger.error(msg)
                raise ValueError(msg)
            if len(lr_clip) != 2:
                msg = 'lr_clip should only have two elements, one for lower bound, and another for upper bound.'
                self.logger.error(msg)
                raise ValueError(msg)
            if not lr_clip[0] < lr_clip[1]:
                msg = 'The first element = {} should be smaller than the second element = {} in lr_clip.'
                self.logger.error(msg.format(lr_clip[0], lr_clip[1]))
                raise ValueError(msg.format(lr_clip[0], lr_clip[1]))
        if not epochs > 0:
            msg = 'The number of training epochs = {} should be strictly positive.'
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))
        if not log_interval > 0:
            msg = 'The number of batches to wait before printting the training status should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))
        if not epochs % self.n_estimators == 0:
            msg = 'The number of training epochs = {} should be a multiple of n_estimators = {}.'
            self.logger.error(msg.format(epochs, self.n_estimators))
            raise ValueError(msg.format(epochs, self.n_estimators))

    def _forward(self, *x):
        """
        Implementation on the internal data forwarding in snapshot ensemble.
        """
        results = [estimator(*x) for estimator in self.estimators_]
        output = op.average(results)
        return output

    def _clip_lr(self, optimizer, lr_clip):
        """Clip the learning rate of the optimizer according to `lr_clip`."""
        if not lr_clip:
            return optimizer
        for param_group in optimizer.param_groups:
            if param_group['lr'] < lr_clip[0]:
                param_group['lr'] = lr_clip[0]
            if param_group['lr'] > lr_clip[1]:
                param_group['lr'] = lr_clip[1]
        return optimizer

    def _set_scheduler(self, optimizer, n_iters):
        """
        Set the learning rate scheduler for snapshot ensemble.
        Please refer to the equation (2) in original paper for details.
        """
        T_M = math.ceil(n_iters / self.n_estimators)
        lr_lambda = lambda iteration: 0.5 * (torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler

    def set_scheduler(self, scheduler_name, **kwargs):
        msg = 'The learning rate scheduler for Snapshot Ensemble will be automatically set. Calling this function has no effect on the training stage of Snapshot Ensemble.'
        warnings.warn(msg, RuntimeWarning)


def _snapshot_ensemble_model_doc(header, item='fit'):
    """
    Decorator on obtaining documentation for different snapshot ensemble
    models.
    """

    def get_doc(item):
        """Return selected item"""
        __doc = {'fit': __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


def _parallel_compute_pseudo_residual(output, target, estimator_idx, shrinkage_rate, n_outputs, is_classification):
    """
    Compute pseudo residuals in soft gradient boosting for each base estimator
    in a parallel fashion.
    """
    accumulated_output = torch.zeros_like(output[0], device=output[0].device)
    for i in range(estimator_idx):
        accumulated_output += shrinkage_rate * output[i]
    if is_classification:
        residual = op.pseudo_residual_classification(target, accumulated_output, n_outputs)
    else:
        residual = op.pseudo_residual_regression(target, accumulated_output)
    return residual


class _BaseSoftGradientBoosting(BaseModule):

    def __init__(self, estimator, n_estimators, estimator_args=None, shrinkage_rate=1.0, cuda=True, n_jobs=None):
        super(BaseModule, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        if estimator_args and not isinstance(estimator, type):
            msg = 'The input `estimator_args` will have no effect since `estimator` is already an object after instantiation.'
            warnings.warn(msg, RuntimeWarning)
        self.shrinkage_rate = shrinkage_rate
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()
        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _validate_parameters(self, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""
        if not epochs > 0:
            msg = 'The number of training epochs = {} should be strictly positive.'
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))
        if not log_interval > 0:
            msg = 'The number of batches to wait before printting the training status should be strictly positive, but got {} instead.'
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))
        if not 0 < self.shrinkage_rate <= 1:
            msg = 'The shrinkage rate should be in the range (0, 1], but got {} instead.'
            self.logger.error(msg.format(self.shrinkage_rate))
            raise ValueError(msg.format(self.shrinkage_rate))

    @abc.abstractmethod
    def _evaluate_during_fit(self, test_loader, epoch):
        """Evaluate the ensemble after each training epoch."""

    def fit(self, train_loader, epochs=100, use_reduction_sum=True, log_interval=100, test_loader=None, save_model=True, save_dir=None):
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)
        criterion = nn.MSELoss(reduction='sum') if use_reduction_sum else nn.MSELoss()
        total_iters = 0
        optimizer = set_module.set_optimizer(self, self.optimizer_name, **self.optimizer_args)
        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(optimizer, self.scheduler_name, **self.scheduler_args)
        for epoch in range(epochs):
            self.train()
            for batch_idx, elem in enumerate(train_loader):
                data, target = io.split_data_target(elem, self.device)
                output = [estimator(*data) for estimator in self.estimators_]
                rets = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_compute_pseudo_residual)(output, target, i, self.shrinkage_rate, self.n_outputs, self.is_classification) for i in range(self.n_estimators))
                loss = torch.tensor(0.0, device=self.device)
                for idx, estimator in enumerate(self.estimators_):
                    loss += criterion(output[idx], rets[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = 'Epoch: {:03d} | Batch: {:03d} | RegLoss: {:.5f}'
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar('sGBM/Train_Loss', loss, total_iters)
                total_iters += 1
            if test_loader:
                flag, test_metric_val = self._evaluate_during_fit(test_loader, epoch)
                if save_model and flag:
                    io.save(self, save_dir, self.logger)
            if self.use_scheduler_:
                if self.scheduler_name == 'ReduceLROnPlateau':
                    if test_loader:
                        scheduler.step(test_metric_val)
                    else:
                        scheduler.step(loss)
                else:
                    scheduler.step()
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)


def _soft_gradient_boosting_model_doc(header, item='model'):
    """
    Decorator on obtaining documentation for different gradient boosting
    models.
    """

    def get_doc(item):
        """Return the selected item"""
        __doc = {'model': __model_doc, 'fit': __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + '\n\n']
        doc.extend(get_doc(item))
        cls.__doc__ = ''.join(doc)
        return cls
    return adddoc


class MLP_clf(nn.Module):

    def __init__(self):
        super(MLP_clf, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


class MLP_reg(nn.Module):

    def __init__(self):
        super(MLP_reg, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseClassifier,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (BaseModule,
     lambda: ([], {'estimator': _mock_layer(), 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (BaseRegressor,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (BaseTree,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BaseTreeEnsemble,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeNet5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (_BaseAdversarialTraining,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (_BaseGradientBoosting,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (_BaseSnapshotEnsemble,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
    (_BaseSoftGradientBoosting,
     lambda: ([], {'estimator': 4, 'n_estimators': 4}),
     lambda: ([], {}),
     False),
]

class Test_TorchEnsemble_Community_Ensemble_Pytorch(_paritybench_base):
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

