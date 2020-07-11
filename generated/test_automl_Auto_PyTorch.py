import sys
_module = sys.modules[__name__]
del sys
autoPyTorch = _module
components = _module
ensembles = _module
abstract_ensemble = _module
ensemble_selection = _module
lr_scheduler = _module
lr_schedulers = _module
metrics = _module
additional_logs = _module
balanced_accuracy = _module
pac_score = _module
standard_metrics = _module
networks = _module
activations = _module
base_net = _module
feature = _module
embedding = _module
mlpnet = _module
resnet = _module
shapedmlpnet = _module
shapedresnet = _module
image = _module
convnet = _module
darts = _module
darts_worker = _module
genotypes = _module
model = _module
operations = _module
utils = _module
densenet = _module
densenet_flexible = _module
mobilenet = _module
resnet = _module
resnet152 = _module
conv2d_helpers = _module
mobilenet_utils = _module
shakedrop = _module
shakeshakeblock = _module
utils = _module
initialization = _module
optimizer = _module
optimizer = _module
preprocessing = _module
feature_preprocessing = _module
fast_ica = _module
kernel_pca = _module
kitchen_sinks = _module
nystroem = _module
polynomial_features = _module
power_transformer = _module
truncated_svd = _module
image_preprocessing = _module
archive = _module
augmentation_transforms = _module
transforms = _module
loss_weight_strategies = _module
preprocessor_base = _module
resampling = _module
random = _module
smote = _module
target_size_strategies = _module
resampling_base = _module
regularization = _module
mixup = _module
shake = _module
training = _module
base_training = _module
budget_types = _module
early_stopping = _module
base_training = _module
checkpoints = _module
load_specific = _module
save_load = _module
lr_scheduling = _module
mixup = _module
trainer = _module
trainer = _module
core = _module
api = _module
autonet_classes = _module
autonet_feature_classification = _module
autonet_feature_data = _module
autonet_feature_multilabel = _module
autonet_feature_regression = _module
autonet_image_classification = _module
autonet_image_classification_multiple_datasets = _module
autonet_image_data = _module
ensemble = _module
hpbandster_extensions = _module
bohb_ext = _module
hyperband_ext = _module
run_with_time = _module
presets = _module
feature_classification = _module
feature_multilabel = _module
feature_regression = _module
image_classification = _module
image_classification_multiple_datasets = _module
worker = _module
worker_no_timelimit = _module
data_management = _module
data_converter = _module
data_loader = _module
data_manager = _module
data_reader = _module
image_loader = _module
pipeline = _module
base = _module
node = _module
pipeline_node = _module
sub_pipeline_node = _module
nodes = _module
autonet_settings = _module
create_dataloader = _module
create_dataset_info = _module
cross_validation = _module
embedding_selector = _module
autonet_settings_no_shuffle = _module
create_image_dataloader = _module
cross_validation_indices = _module
image_augmentation = _module
image_dataset_reader = _module
loss_module_selector_indices = _module
multiple_datasets = _module
network_selector_datasetinfo = _module
optimization_algorithm_no_timelimit = _module
simple_scheduler_selector = _module
simple_train_node = _module
single_dataset = _module
imputation = _module
initialization_selector = _module
log_functions_selector = _module
loss_module_selector = _module
lr_scheduler_selector = _module
metric_selector = _module
network_selector = _module
normalization_strategy_selector = _module
one_hot_encoding = _module
optimization_algorithm = _module
optimizer_selector = _module
preprocessor_selector = _module
resampling_strategy_selector = _module
train_node = _module
benchmarking = _module
benchmark = _module
benchmark_pipeline = _module
apply_user_updates = _module
benchmark_settings = _module
create_autonet = _module
fit_autonet = _module
for_autonet_config = _module
for_instance = _module
for_run = _module
prepare_result_folder = _module
read_instance_data = _module
save_ensemble_logs = _module
save_results = _module
set_autonet_config = _module
set_ensemble_config = _module
visualization_pipeline = _module
collect_trajectories = _module
get_additional_trajectories = _module
get_ensemble_trajectories = _module
get_run_trajectories = _module
plot_summary = _module
plot_trajectories = _module
read_instance_info = _module
visualization_settings = _module
config = _module
config_condition = _module
config_file_parser = _module
config_option = _module
config_space_hyperparameter = _module
configspace_wrapper = _module
hyperparameter_search_space_update = _module
loggers = _module
mem_test_thread = _module
modify_config_space = _module
modules = _module
thread_read_write = _module
examples = _module
classification = _module
modify_pipeline = _module
regression = _module
advanced_classification = _module
classification_test = _module
openml_task = _module
build_singularity_container = _module
recompute_ensemble_performance = _module
run_benchmark = _module
run_benchmark_cluster = _module
run_benchmark_cluster_condensed = _module
visualize_benchmark = _module
setup = _module
test = _module
test_pipeline = _module
test_cross_validation = _module
test_imputation = _module
test_initialization = _module
test_log_selector = _module
test_loss_selector = _module
test_lr_scheduler_selector = _module
test_metric_selector = _module
test_network_selector = _module
test_normalization_strategy_selector = _module
test_optimization_algorithm = _module
test_optimizer_selector = _module
test_resampling_strategy_selector = _module

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


import numpy as np


import math


import torch


import torch.optim.lr_scheduler as lr_scheduler


from torch.optim import Optimizer


import sklearn.metrics as metrics


import torch.nn as nn


import inspect


from collections import OrderedDict


import torchvision.transforms as transforms


from torch.autograd import Variable


import re


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import logging


from copy import deepcopy


from torch.autograd import Function


import torch.optim as optim


import warnings


from torchvision.transforms import *


import random


import time


import scipy.sparse


import copy


import torchvision


from sklearn.datasets import make_regression


from sklearn.datasets import make_multilabel_classification


from enum import Enum


import torch.utils.data as data


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.utils.data.dataset import Subset


from torch.utils.data.sampler import SubsetRandomSampler


import pandas as pd


from sklearn.model_selection import BaseCrossValidator


from torch.utils.data import Dataset


from torchvision import datasets


from torchvision import models


from torchvision import transforms


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import StratifiedShuffleSplit


import torchvision.models as models


from sklearn.impute import SimpleImputer


from sklearn.compose import ColumnTransformer


from torch.nn.modules.loss import _Loss


from sklearn.model_selection import KFold


from numpy.testing import assert_array_equal


from torch.nn import Linear


from numpy.testing import assert_array_almost_equal


from sklearn.preprocessing import MinMaxScaler


class BaseNet(nn.Module):
    """ Parent class for all Networks"""

    def __init__(self, config, in_features, out_features, final_activation):
        """
        Initialize the BaseNet.
        """
        super(BaseNet, self).__init__()
        self.layers = nn.Sequential()
        self.config = config
        self.n_feats = in_features
        self.n_classes = out_features
        self.epochs_trained = 0
        self.budget_trained = 0
        self.stopped_early = False
        self.last_compute_result = None
        self.logs = []
        self.num_epochs_no_progress = 0
        self.current_best_epoch_performance = None
        self.best_parameters = None
        self.final_activation = final_activation

    def forward(self, x):
        x = self.layers(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def snapshot(self):
        self.best_parameters = OrderedDict({key: value.cpu().clone() for key, value in self.state_dict().items()})

    def load_snapshot(self):
        if self.best_parameters is not None:
            self.load_state_dict(self.best_parameters)

    @staticmethod
    def get_config_space():
        return ConfigSpace.ConfigurationSpace()


class BaseFeatureNet(BaseNet):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for cagtegorical features"""

    def __init__(self, config, in_features, out_features, embedding, final_activation):
        """
        Initialize the BaseFeatureNet.
        """
        super(BaseFeatureNet, self).__init__(config, in_features, out_features, final_activation)
        self.embedding = embedding

    def forward(self, x):
        x = self.embedding(x)
        return super(BaseFeatureNet, self).forward(x)


class BaseImageNet(BaseNet):

    def __init__(self, config, in_features, out_features, final_activation):
        super(BaseImageNet, self).__init__(config, in_features, out_features, final_activation)
        if len(in_features) == 2:
            self.channels = 1
            self.iw = in_features[0]
            self.ih = in_features[1]
        if len(in_features) == 3:
            self.channels = in_features[0]
            self.iw = in_features[1]
            self.ih = in_features[2]


def get_hyperparameter(hyper_type, name, value_range, log=False):
    if isinstance(value_range, tuple) and len(value_range) == 2 and isinstance(value_range[1], bool) and isinstance(value_range[0], (tuple, list)):
        value_range, log = value_range
    if len(value_range) == 0:
        raise ValueError(name + ': The range has to contain at least one element')
    if len(value_range) == 1:
        return CSH.Constant(name, int(value_range[0]) if isinstance(value_range[0], bool) else value_range[0])
    if len(value_range) == 2 and value_range[0] == value_range[1]:
        return CSH.Constant(name, int(value_range[0]) if isinstance(value_range[0], bool) else value_range[0])
    if hyper_type == CSH.CategoricalHyperparameter:
        return CSH.CategoricalHyperparameter(name, value_range)
    if hyper_type == CSH.UniformFloatHyperparameter:
        assert len(value_range) == 2, 'Float HP range update for %s is specified by the two upper and lower values. %s given.' % (name, len(value_range))
        return CSH.UniformFloatHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    if hyper_type == CSH.UniformIntegerHyperparameter:
        assert len(value_range) == 2, 'Int HP range update for %s is specified by the two upper and lower values. %s given.' % (name, len(value_range))
        return CSH.UniformIntegerHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    raise ValueError('Unknown type: %s for hp %s' % (hyper_type, name))


class LearnedEntityEmbedding(nn.Module):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for cagtegorical features"""

    def __init__(self, config, in_features, one_hot_encoder):
        """
        Initialize the BaseFeatureNet.
        
        Arguments:
            config: The configuration sampled by the hyperparameter optimizer
            in_features: the number of features of the dataset
            one_hot_encoder: OneHot encoder, that is used to encode X
        """
        super(LearnedEntityEmbedding, self).__init__()
        self.config = config
        self.n_feats = in_features
        self.one_hot_encoder = one_hot_encoder
        self.num_numerical = len([f for f in one_hot_encoder.categorical_features if not f])
        self.num_input_features = [len(c) for c in one_hot_encoder.categories_]
        self.embed_features = [(num_in >= config['min_unique_values_for_embedding']) for num_in in self.num_input_features]
        self.num_output_dimensions = [(config['dimension_reduction_' + str(i)] * num_in) for i, num_in in enumerate(self.num_input_features)]
        self.num_output_dimensions = [int(np.clip(num_out, 1, num_in - 1)) for num_out, num_in in zip(self.num_output_dimensions, self.num_input_features)]
        self.num_output_dimensions = [(num_out if embed else num_in) for num_out, embed, num_in in zip(self.num_output_dimensions, self.embed_features, self.num_input_features)]
        self.num_out_feats = self.num_numerical + sum(self.num_output_dimensions)
        self.ee_layers = self._create_ee_layers(in_features)

    def forward(self, x):
        concat_seq = []
        last_concat = 0
        x_pointer = 0
        layer_pointer = 0
        for num_in, embed in zip(self.num_input_features, self.embed_features):
            if not embed:
                x_pointer += 1
                continue
            if x_pointer > last_concat:
                concat_seq.append(x[:, last_concat:x_pointer])
            categorical_feature_slice = x[:, x_pointer:x_pointer + num_in]
            concat_seq.append(self.ee_layers[layer_pointer](categorical_feature_slice))
            layer_pointer += 1
            x_pointer += num_in
            last_concat = x_pointer
        concat_seq.append(x[:, last_concat:])
        return torch.cat(concat_seq, dim=1)

    def _create_ee_layers(self, in_features):
        layers = nn.ModuleList()
        for i, (num_in, embed, num_out) in enumerate(zip(self.num_input_features, self.embed_features, self.num_output_dimensions)):
            if not embed:
                continue
            layers.append(nn.Linear(num_in, num_out))
        return layers

    @staticmethod
    def get_config_space(categorical_features=None, min_unique_values_for_embedding=((3, 300), True), dimension_reduction=(0, 1), **kwargs):
        if categorical_features is None or not any(categorical_features):
            return CS.ConfigurationSpace()
        cs = CS.ConfigurationSpace()
        min_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'min_unique_values_for_embedding', min_unique_values_for_embedding)
        cs.add_hyperparameter(min_hp)
        for i in range(len([x for x in categorical_features if x])):
            ee_dimensions_hp = get_hyperparameter(CSH.UniformFloatHyperparameter, 'dimension_reduction_' + str(i), kwargs.pop('dimension_reduction_' + str(i), dimension_reduction))
            cs.add_hyperparameter(ee_dimensions_hp)
        assert len(kwargs) == 0, 'Invalid hyperparameter updates for learned embedding: %s' % str(kwargs)
        return cs


class NoEmbedding(nn.Module):

    def __init__(self, config, in_features, one_hot_encoder):
        super(NoEmbedding, self).__init__()
        self.config = config
        self.n_feats = in_features
        self.num_out_feats = self.n_feats

    def forward(self, x):
        return x

    @staticmethod
    def get_config_space(*args, **kwargs):
        return CS.ConfigurationSpace()


def add_hyperparameter(cs, hyper_type, name, value_range, log=False):
    return cs.add_hyperparameter(get_hyperparameter(hyper_type, name, value_range, log))


class MlpNet(BaseFeatureNet):
    activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

    def __init__(self, config, in_features, out_features, embedding, final_activation=None):
        super(MlpNet, self).__init__(config, in_features, out_features, embedding, final_activation)
        self.activation = self.activations[config['activation']]
        self.layers = self._build_net(self.n_feats, self.n_classes)

    def _build_net(self, in_features, out_features):
        layers = list()
        self._add_layer(layers, in_features, self.config['num_units_1'], 1)
        for i in range(2, self.config['num_layers'] + 1):
            self._add_layer(layers, self.config['num_units_%d' % (i - 1)], self.config['num_units_%d' % i], i)
        layers.append(nn.Linear(self.config['num_units_%d' % self.config['num_layers']], out_features))
        return nn.Sequential(*layers)

    def _add_layer(self, layers, in_features, out_features, layer_id):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(self.activation())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.config['dropout_%d' % layer_id]))

    @staticmethod
    def get_config_space(num_layers=((1, 15), False), num_units=((10, 1024), True), activation=('sigmoid', 'tanh', 'relu'), dropout=(0.0, 0.8), use_dropout=(True, False), **kwargs):
        cs = CS.ConfigurationSpace()
        num_layers_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'num_layers', num_layers)
        cs.add_hyperparameter(num_layers_hp)
        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, 'use_dropout', use_dropout)
        for i in range(1, num_layers[0][1] + 1):
            n_units_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'num_units_%d' % i, kwargs.pop('num_units_%d' % i, num_units))
            cs.add_hyperparameter(n_units_hp)
            if i > num_layers[0][0]:
                cs.add_condition(CS.GreaterThanCondition(n_units_hp, num_layers_hp, i - 1))
            if True in use_dropout:
                dropout_hp = get_hyperparameter(CSH.UniformFloatHyperparameter, 'dropout_%d' % i, kwargs.pop('dropout_%d' % i, dropout))
                cs.add_hyperparameter(dropout_hp)
                dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout_hp, True)
                if i > num_layers[0][0]:
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_layers_hp, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'activation', activation)
        assert len(kwargs) == 0, 'Invalid hyperparameter updates for mlpnet: %s' % str(kwargs)
        return cs


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(BaseImageNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ShakeDropFunction(Function):

    @staticmethod
    def forward(ctx, x, alpha, beta, bl):
        ctx.save_for_backward(x, alpha, beta, bl)
        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, bl = ctx.saved_variables
        grad_x = grad_alpha = grad_beta = grad_bl = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)
        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_drop_get_bl(block_index, min_prob_no_shake, num_blocks, is_training, is_cuda):
    pl = 1 - (block_index + 1) / num_blocks * (1 - min_prob_no_shake)
    if not is_training:
        bl = torch.tensor(1.0) if random.random() <= pl else torch.tensor(0.0)
    if is_training:
        bl = torch.tensor(pl)
    if is_cuda:
        bl = bl
    return bl


def shake_get_alpha_beta(is_training, is_cuda):
    if is_training:
        result = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
        return result if not is_cuda else (result[0], result[1])
    alpha = torch.rand(1)
    beta = torch.rand(1)
    if is_cuda:
        alpha = alpha
        beta = beta
    return alpha, beta


class ShakeShakeFunction(Function):

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)
        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ResBlock(nn.Module):

    def __init__(self, config, in_features, out_features, block_index, dropout, activation):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation
        self.shortcut = None
        self.start_norm = None
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(nn.BatchNorm1d(in_features), self.activation())
        self.block_index = block_index
        self.num_blocks = self.config['blocks_per_group'] * self.config['num_groups']
        self.layers = self._build_block(in_features, out_features)
        if config['use_shake_shake']:
            self.shake_shake_layers = self._build_block(in_features, out_features)

    def _build_block(self, in_features, out_features):
        layers = list()
        if self.start_norm == None:
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        if self.shortcut != None:
            x = self.start_norm(x)
            residual = self.shortcut(x)
        if self.config['use_shake_shake']:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)
        if self.config['use_shake_drop']:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(self.block_index, 1 - self.config['max_shake_drop_probability'], self.num_blocks, self.training, x.is_cuda)
            x = shake_drop(x, alpha, beta, bl)
        x = x + residual
        return x


def get_shaped_neuron_counts(shape, in_feat, out_feat, max_neurons, layer_count):
    counts = []
    if layer_count <= 0:
        return counts
    if layer_count == 1:
        counts.append(out_feat)
        return counts
    max_neurons = max(in_feat, max_neurons)
    if shape == 'brick':
        for _ in range(layer_count - 1):
            counts.append(max_neurons)
        counts.append(out_feat)
    if shape == 'triangle':
        previous = in_feat
        step_size = int((max_neurons - previous) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous + step_size
            counts.append(previous)
        counts.append(max_neurons)
        counts.append(out_feat)
    if shape == 'funnel':
        previous = max_neurons
        counts.append(previous)
        step_size = int((previous - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous - step_size
            counts.append(previous)
        counts.append(out_feat)
    if shape == 'long_funnel':
        brick_layer = int(layer_count / 2)
        funnel_layer = layer_count - brick_layer
        counts.extend(get_shaped_neuron_counts('brick', in_feat, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts('funnel', in_feat, out_feat, max_neurons, funnel_layer))
        if len(counts) != layer_count:
            None
    if shape == 'diamond':
        triangle_layer = int(layer_count / 2) + 1
        funnel_layer = layer_count - triangle_layer
        counts.extend(get_shaped_neuron_counts('triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        remove_triangle_layer = len(counts) > 1
        if remove_triangle_layer:
            counts = counts[0:-2]
        counts.extend(get_shaped_neuron_counts('funnel', max_neurons, out_feat, max_neurons, funnel_layer + (2 if remove_triangle_layer else 0)))
        if len(counts) != layer_count:
            None
    if shape == 'hexagon':
        triangle_layer = int(layer_count / 3) + 1
        funnel_layer = triangle_layer
        brick_layer = layer_count - triangle_layer - funnel_layer
        counts.extend(get_shaped_neuron_counts('triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        counts.extend(get_shaped_neuron_counts('brick', max_neurons, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts('funnel', max_neurons, out_feat, max_neurons, funnel_layer))
        if len(counts) != layer_count:
            None
    if shape == 'stairs':
        previous = max_neurons
        counts.append(previous)
        if layer_count % 2 == 1:
            counts.append(previous)
        step_size = 2 * int((max_neurons - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(int(layer_count / 2 - 1)):
            previous = previous - step_size
            counts.append(previous)
            counts.append(previous)
        counts.append(out_feat)
        if len(counts) != layer_count:
            None
    return counts


class ShapedMlpNet(MlpNet):

    def __init__(self, *args, **kwargs):
        super(ShapedMlpNet, self).__init__(*args, **kwargs)

    def _build_net(self, in_features, out_features):
        layers = list()
        neuron_counts = get_shaped_neuron_counts(self.config['mlp_shape'], in_features, out_features, self.config['max_units'], self.config['num_layers'])
        if self.config['use_dropout']:
            dropout_shape = get_shaped_neuron_counts(self.config['dropout_shape'], 0, 0, 1000, self.config['num_layers'])
        previous = in_features
        for i in range(self.config['num_layers'] - 1):
            if i >= len(neuron_counts):
                break
            dropout = dropout_shape[i] / 1000 * self.config['max_dropout'] if self.config['use_dropout'] else 0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]
        layers.append(nn.Linear(previous, out_features))
        return nn.Sequential(*layers)

    def _add_layer(self, layers, in_features, out_features, dropout):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(self.activation())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(dropout))

    @staticmethod
    def get_config_space(num_layers=(1, 15), max_units=((10, 1024), True), activation=('sigmoid', 'tanh', 'relu'), mlp_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'), dropout_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'), max_dropout=(0, 0.8), use_dropout=(True, False)):
        cs = CS.ConfigurationSpace()
        mlp_shape_hp = get_hyperparameter(CSH.CategoricalHyperparameter, 'mlp_shape', mlp_shape)
        cs.add_hyperparameter(mlp_shape_hp)
        num_layers_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'num_layers', num_layers)
        cs.add_hyperparameter(num_layers_hp)
        max_units_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter, 'max_units', max_units)
        cs.add_hyperparameter(max_units_hp)
        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, 'use_dropout', use_dropout)
        if True in use_dropout:
            dropout_shape_hp = add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'dropout_shape', dropout_shape)
            max_dropout_hp = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'max_dropout', max_dropout)
            cs.add_condition(CS.EqualsCondition(dropout_shape_hp, use_dropout_hp, True))
            cs.add_condition(CS.EqualsCondition(max_dropout_hp, use_dropout_hp, True))
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'activation', activation)
        return cs


class ConvNet(BaseImageNet):

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):
        super(ConvNet, self).__init__(config, in_features, out_features, final_activation)
        self.layers = self._build_net(self.n_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.last_layer(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def _build_net(self, out_features):
        layers = list()
        init_filter = self.config['conv_init_filters']
        self._add_layer(layers, self.channels, init_filter, 1)
        cw, ch = self._get_layer_size(self.iw, self.ih)
        self.dense_size = init_filter * cw * ch
        None
        for i in range(2, self.config['num_layers'] + 1):
            cw, ch = self._get_layer_size(cw, ch)
            if cw == 0 or ch == 0:
                None
                break
            self._add_layer(layers, init_filter, init_filter * 2, i)
            init_filter *= 2
            self.dense_size = init_filter * cw * ch
            None
        self.last_layer = nn.Linear(self.dense_size, out_features)
        nw = nn.Sequential(*layers)
        return nw

    def _get_layer_size(self, w, h):
        cw = (w - self.config['conv_kernel_size'] + 2 * self.config['conv_kernel_padding']) // self.config['conv_kernel_stride'] + 1
        ch = (h - self.config['conv_kernel_size'] + 2 * self.config['conv_kernel_padding']) // self.config['conv_kernel_stride'] + 1
        cw, ch = cw // self.config['pool_size'], ch // self.config['pool_size']
        return cw, ch

    def _add_layer(self, layers, in_filters, out_filters, layer_id):
        layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=self.config['conv_kernel_size'], stride=self.config['conv_kernel_stride'], padding=self.config['conv_kernel_padding']))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(self.activation())
        layers.append(nn.MaxPool2d(kernel_size=self.config['pool_size'], stride=self.config['pool_size']))

    @staticmethod
    def get_config_space(user_updates=None):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('activation', ['relu']))
        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=2, upper=5)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_init_filters', lower=16, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_size', lower=2, upper=5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_stride', lower=1, upper=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_padding', lower=2, upper=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('pool_size', lower=2, upper=3))
        return cs


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine), 'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C, affine=affine))}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        try:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        except:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(BaseImageNet):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class NetworkImageNet(BaseImageNet):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        self._layers = layers
        self._auxiliary = auxiliary
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


PRIMITIVES = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']


def generate_genotype(gene_function):

    @wraps(gene_function)
    def wrapper(config=None, steps=4):
        concat = range(2, 6)
        gene_normal, gene_reduce = gene_function(config, steps).values()
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
        return genotype
    return wrapper


@generate_genotype
def get_gene_from_config(config, steps=4):
    gene = {'normal': [], 'reduce': []}
    for cell_type in gene.keys():
        first_edge = config['edge_{}_0'.format(cell_type)], 0
        second_edge = config['edge_{}_1'.format(cell_type)], 1
        gene[cell_type].append(first_edge)
        gene[cell_type].append(second_edge)
    for i, offset in zip(range(3, steps + 2), [2, 5, 9]):
        for cell_type in gene.keys():
            input_nodes = config['inputs_node_{}_{}'.format(cell_type, i)].split('_')
            for node in input_nodes:
                edge = config['edge_{}_{}'.format(cell_type, int(node) + offset)], int(node)
                gene[cell_type].append(edge)
    return gene


class DARTSImageNet(NetworkCIFAR):

    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(NetworkCIFAR, self).__init__(config, in_features, out_features, final_activation)
        self.drop_path_prob = config['drop_path_prob']
        topology = {key: config[key] for key in config if 'edge' in key or 'inputs_node' in key}
        genotype = get_gene_from_config(topology)
        super(DARTSImageNet, self).__init__(config['init_channels'], out_features, config['layers'], config['auxiliary'], genotype)

    def forward(self, x):
        x = super(DARTSImageNet, self).forward(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    @staticmethod
    def get_config_space(**kwargs):
        return DARTSWorker.get_config_space()


all_activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu': nn.LeakyReLU, 'selu': nn.SELU, 'rrelu': nn.RReLU, 'tanhshrink': nn.Tanhshrink, 'hardtanh': nn.Hardtanh, 'elu': nn.ELU, 'prelu': nn.PReLU}


def get_activation(name, inplace=False):
    if name not in all_activations:
        raise ValueError('Activation ' + str(name) + ' not defined')
    activation = all_activations[name]
    activation_kwargs = {'inplace': True} if 'inplace' in inspect.getargspec(activation)[0] else dict()
    return activation(**activation_kwargs)


class _DenseLayer(nn.Sequential):

    def __init__(self, nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(nChannels))
        self.add_module('relu1', get_activation(activation, inplace=True))
        if bottleneck:
            self.add_module('conv1', nn.Conv2d(nChannels, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
            nChannels = 4 * growth_rate
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
            self.add_module('norm2', nn.BatchNorm2d(nChannels))
            self.add_module('relu2', get_activation(activation, inplace=True))
        self.add_module('conv2', nn.Conv2d(nChannels, growth_rate, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False))
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, N, nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation):
        super(_DenseBlock, self).__init__()
        for i in range(N):
            self.add_module('denselayer%d' % (i + 1), _DenseLayer(nChannels, growth_rate, drop_rate, bottleneck, kernel_size, activation))
            nChannels += growth_rate


class Reshape(nn.Module):

    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, x):
        return x.reshape(-1, self.size)


class _Transition(nn.Sequential):

    def __init__(self, nChannels, nOutChannels, drop_rate, last, pool_size, kernel_size, stride, padding, activation):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(nChannels))
        self.add_module('relu', get_activation(activation, inplace=True))
        if last:
            self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))
            self.add_module('reshape', Reshape(nChannels))
        else:
            self.add_module('conv', nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, bias=False))
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))
            self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))


class DenseNet(BaseImageNet):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):
        super(DenseNet, self).__init__(config, in_features, out_features, final_activation)
        growth_rate = config['growth_rate']
        block_config = [config['layer_in_block_%d' % (i + 1)] for i in range(config['blocks'])]
        num_init_features = 2 * growth_rate
        bn_size = 4
        drop_rate = config['dropout'] if config['use_dropout'] else 0
        num_classes = self.n_classes
        image_size, min_image_size = min(self.iw, self.ih), 1
        import math
        division_steps = math.floor(math.log2(image_size) - math.log2(min_image_size) - 1e-05) + 1
        if division_steps > len(block_config) + 1:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(self.channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
            division_steps -= 2
        else:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(self.channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, pool_size=2 if i > len(block_config) - division_steps else 1)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('last_norm', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        self.layers = nn.Sequential(self.features)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        if not self.training and self.final_activation is not None:
            out = self.final_activation(out)
        return out

    @staticmethod
    def get_config_space(growth_rate_range=(12, 40), nr_blocks=(3, 4), layer_range=([1, 12], [6, 24], [12, 64], [12, 64]), num_init_features=(32, 128), **kwargs):
        cs = CS.ConfigurationSpace()
        growth_rate_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'growth_rate', growth_rate_range)
        cs.add_hyperparameter(growth_rate_hp)
        blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'blocks', nr_blocks)
        cs.add_hyperparameter(blocks_hp)
        use_dropout = add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'use_dropout', [True, False])
        dropout = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'dropout', [0.0, 1.0])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))
        if type(nr_blocks[0]) == int:
            min_blocks = nr_blocks[0]
            max_blocks = nr_blocks[1]
        else:
            min_blocks = nr_blocks[0][0]
            max_blocks = nr_blocks[0][1]
        for i in range(1, max_blocks + 1):
            layer_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'layer_in_block_%d' % i, layer_range[i - 1])
            cs.add_hyperparameter(layer_hp)
            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))
        return cs


logger = logging.getLogger('autonet')


class PrintNode(nn.Module):

    def __init__(self, msg):
        super(PrintNode, self).__init__()
        self.msg = msg

    def forward(self, x):
        logger.debug(self.msg)
        return x


def _get_out_size(in_size, kernel_size, stride, padding):
    return int(math.floor((in_size - kernel_size + 2 * padding) / stride + 1))


def get_layer_params(in_size, out_size, kernel_size):
    kernel_size = int(kernel_size)
    stride = int(max(1, math.ceil((in_size - kernel_size) / (out_size - 1)) if out_size > 1 else 1))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, 0)
    required_padding = stride / 2 * (in_size - cur_out_size)
    cur_padding = int(math.ceil(required_padding))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, cur_padding)
    if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
        return cur_out_size, kernel_size, stride, cur_padding
    cur_padding = int(math.floor(required_padding))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, cur_padding)
    if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
        return cur_out_size, kernel_size, stride, cur_padding
    if stride > 1:
        stride = int(stride - 1)
        cur_padding = 0
        cur_out_size = int(_get_out_size(in_size, kernel_size, stride, cur_padding))
        if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
            return cur_out_size, kernel_size, stride, cur_padding
    if kernel_size % 2 == 0 and out_size == in_size:
        return get_layer_params(in_size, out_size, kernel_size + 1)
    raise Exception('Could not find padding and stride to reduce ' + str(in_size) + ' to ' + str(out_size) + ' using kernel ' + str(kernel_size))


class DenseNetFlexible(BaseImageNet):

    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):
        super(DenseNetFlexible, self).__init__(config, in_features, out_features, final_activation)
        growth_rate = config['growth_rate']
        bottleneck = config['bottleneck']
        channel_reduction = config['channel_reduction']
        in_size = min(self.iw, self.ih)
        out_size = max(1, in_size * config['last_image_size'])
        size_reduction = math.pow(in_size / out_size, 1 / (config['blocks'] + 1))
        nChannels = 2 * growth_rate
        self.features = nn.Sequential()
        sizes = [max(1, round(in_size / math.pow(size_reduction, i + 1))) for i in range(config['blocks'] + 2)]
        in_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[0], config['first_conv_kernel'])
        self.features.add_module('conv0', nn.Conv2d(self.channels, nChannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(nChannels))
        self.features.add_module('activ0', get_activation(config['first_activation'], inplace=True))
        in_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[1], config['first_pool_kernel'])
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        nOutChannels = nChannels
        for i in range(1, config['blocks'] + 1):
            nChannels = nOutChannels
            drop_rate = config['dropout_%d' % i] if config['use_dropout'] else 0
            block = _DenseBlock(N=config['layer_in_block_%d' % i], nChannels=nChannels, bottleneck=bottleneck, growth_rate=growth_rate, drop_rate=drop_rate, kernel_size=config['conv_kernel_%d' % i], activation=config['activation_%d' % i])
            self.features.add_module('denseblock%d' % i, block)
            nChannels = nChannels + config['layer_in_block_%d' % i] * growth_rate
            nOutChannels = max(1, math.floor(nChannels * channel_reduction))
            out_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[i + 1], config['pool_kernel_%d' % i])
            transition = _Transition(nChannels=nChannels, nOutChannels=nOutChannels, drop_rate=drop_rate, last=i == config['blocks'], pool_size=in_size, kernel_size=kernel_size, stride=stride, padding=padding, activation=config['activation_%d' % i])
            in_size = out_size
            self.features.add_module('trans%d' % i, transition)
        self.classifier = nn.Linear(nChannels, out_features)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.matrix_init(m.weight, config['conv_init'])
            elif isinstance(m, nn.BatchNorm2d):
                self.matrix_init(m.weight, config['batchnorm_weight_init'])
                self.matrix_init(m.bias, config['batchnorm_bias_init'])
            elif isinstance(m, nn.Linear):
                self.matrix_init(m.bias, config['linear_bias_init'])
        self.layers = nn.Sequential(self.features)

    def matrix_init(self, matrix, init_type):
        if init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(matrix)
        elif init_type == 'constant_0':
            nn.init.constant_(matrix, 0)
        elif init_type == 'constant_1':
            nn.init.constant_(matrix, 1)
        elif init_type == 'constant_05':
            nn.init.constant_(matrix, 0.5)
        elif init_type == 'random':
            return
        else:
            raise ValueError('Init type ' + init_type + ' is not supported')

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        if not self.training and self.final_activation is not None:
            out = self.final_activation(out)
        return out

    @staticmethod
    def get_config_space(growth_rate_range=(5, 128), nr_blocks=(1, 5), kernel_range=(2, 7), layer_range=(5, 50), activations=all_activations.keys(), conv_init=('random', 'kaiming_normal', 'constant_0', 'constant_1', 'constant_05'), batchnorm_weight_init=('random', 'constant_0', 'constant_1', 'constant_05'), batchnorm_bias_init=('random', 'constant_0', 'constant_1', 'constant_05'), linear_bias_init=('random', 'constant_0', 'constant_1', 'constant_05'), **kwargs):
        cs = CS.ConfigurationSpace()
        growth_rate_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'growth_rate', growth_rate_range)
        first_conv_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'first_conv_kernel', kernel_range)
        first_pool_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'first_pool_kernel', kernel_range)
        conv_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'conv_init', conv_init)
        batchnorm_weight_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'batchnorm_weight_init', batchnorm_weight_init)
        batchnorm_bias_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'batchnorm_bias_init', batchnorm_bias_init)
        linear_bias_init_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'linear_bias_init', linear_bias_init)
        first_activation_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'first_activation', set(activations).intersection(all_activations))
        blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'blocks', nr_blocks)
        cs.add_hyperparameter(growth_rate_hp)
        cs.add_hyperparameter(first_conv_kernel_hp)
        cs.add_hyperparameter(first_pool_kernel_hp)
        cs.add_hyperparameter(conv_init_hp)
        cs.add_hyperparameter(batchnorm_weight_init_hp)
        cs.add_hyperparameter(batchnorm_bias_init_hp)
        cs.add_hyperparameter(linear_bias_init_hp)
        cs.add_hyperparameter(first_activation_hp)
        cs.add_hyperparameter(blocks_hp)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'channel_reduction', [0.1, 0.9])
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'last_image_size', [0, 1])
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'bottleneck', [True, False])
        use_dropout = add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'use_dropout', [True, False])
        if type(nr_blocks[0]) == int:
            min_blocks = nr_blocks[0]
            max_blocks = nr_blocks[1]
        else:
            min_blocks = nr_blocks[0][0]
            max_blocks = nr_blocks[0][1]
        for i in range(1, max_blocks + 1):
            layer_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'layer_in_block_%d' % i, layer_range)
            pool_kernel_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'pool_kernel_%d' % i, kernel_range)
            activation_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'activation_%d' % i, set(activations).intersection(all_activations))
            cs.add_hyperparameter(layer_hp)
            cs.add_hyperparameter(pool_kernel_hp)
            cs.add_hyperparameter(activation_hp)
            dropout = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'dropout_%d' % i, [0.0, 1.0])
            conv_kernel = add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'conv_kernel_%d' % i, [3, 5, 7])
            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(conv_kernel, blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(pool_kernel_hp, blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(activation_hp, blocks_hp, i - 1))
                cs.add_condition(CS.AndConjunction(CS.EqualsCondition(dropout, use_dropout, True), CS.GreaterThanCondition(dropout, blocks_hp, i - 1)))
            else:
                cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))
        return cs


class Arch_Encoder:
    """ Encode block definition string
    Encodes a list of config space (dicts) through a string notation of arguments for further usage with _decode_architecure and timm.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block hyperpar dict as coming from MobileNet class
    Returns:
        Architecture encoded as string for further usage with _decode_architecure and timm.
    """

    def __init__(self, block_types, nr_sub_blocks, kernel_sizes, strides, output_filters, se_ratios, skip_connections, expansion_rates=0):
        self.block_types = block_types
        self.nr_sub_blocks = nr_sub_blocks
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.expansion_rates = expansion_rates
        self.output_filters = output_filters
        self.se_ratios = se_ratios
        self.skip_connections = skip_connections
        self.arch_encoded = [[''] for ind in range(len(self.block_types))]
        self._encode_architecture()

    def _encode_architecture(self):
        encoding_functions = [self._get_encoded_blocks, self._get_encoded_nr_sub_bocks, self._get_encoded_kernel_sizes, self._get_encoded_strides, self._get_encoded_expansion_rates, self._get_encoded_output_filters, self._get_encoded_se_ratios, self._get_encoded_skip_connections]
        for func in encoding_functions:
            return_val = func()
            self._add_specifications(return_val)

    def _add_specifications(self, arguments):
        for ind, arg in enumerate(arguments):
            if len(self.arch_encoded[ind][0]) != 0 and arg != '' and not self.arch_encoded[ind][0].endswith('_'):
                self.arch_encoded[ind][0] = self.arch_encoded[ind][0] + '_'
            self.arch_encoded[ind][0] = self.arch_encoded[ind][0] + arg

    def _get_encoded_blocks(self):
        block_type_dict = {'inverted_residual': 'ir', 'dwise_sep_conv': 'ds', 'conv_bn_act': 'cn'}
        block_type_list = self._dict_to_list(self.block_types)
        return [block_type_dict[item] for item in block_type_list]

    def _get_encoded_nr_sub_bocks(self):
        nr_sub_blocks_dict = dict([(i, 'r' + str(i)) for i in range(10)])
        nr_sub_blocks_list = self._dict_to_list(self.nr_sub_blocks)
        return [nr_sub_blocks_dict[item] for item in nr_sub_blocks_list]

    def _get_encoded_kernel_sizes(self):
        kernel_sizes_dict = dict([(i, 'k' + str(i)) for i in range(10)])
        kernel_sizes_list = self._dict_to_list(self.kernel_sizes)
        return [kernel_sizes_dict[item] for item in kernel_sizes_list]

    def _get_encoded_strides(self):
        strides_dict = dict([(i, 's' + str(i)) for i in range(10)])
        strides_list = self._dict_to_list(self.strides)
        return [strides_dict[item] for item in strides_list]

    def _get_encoded_expansion_rates(self):
        if self.expansion_rates == 0:
            exp_list = ['e1', 'e6', 'e6', 'e6', 'e6', 'e6', 'e6']
            return exp_list[0:len(self.block_types)]
        else:
            expansion_rates_dict = dict([(i, 'e' + str(i)) for i in range(10)])
            expansion_rates_list = self._dict_to_list(self.expansion_rates)
            return [expansion_rates_dict[item] for item in expansion_rates_list]

    def _get_encoded_output_filters(self):
        output_filters_dict = dict([(i, 'c' + str(i)) for i in range(5000)])
        output_filters_list = self._dict_to_list(self.output_filters)
        return [output_filters_dict[item] for item in output_filters_list]

    def _get_encoded_se_ratios(self):
        se_ratios_dict = {(0): '', (0.25): 'se0.25'}
        se_ratios_list = self._dict_to_list(self.se_ratios)
        return [se_ratios_dict[item] for item in se_ratios_list]

    def _get_encoded_skip_connections(self):
        skip_connections_dict = {(True): '', (False): 'no_skip'}
        skip_connections_list = self._dict_to_list(self.skip_connections)
        return [skip_connections_dict[item] for item in skip_connections_list]

    def _dict_to_list(self, input_dict):
        output_list = []
        dict_len = len(input_dict)
        for ind in range(dict_len):
            output_list.append(input_dict['Group_' + str(ind + 1)])
        return output_list

    def get_encoded_architecture(self):
        return self.arch_encoded


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg'):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                assert False, 'Invalid pool type: %s' % pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.output_size) + ', pool_type=' + self.pool_type + ')'


_BN_EPS_PT_DEFAULT = 1e-05


_BN_MOMENTUM_PT_DEFAULT = 0.01


_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if _is_static_pad(kernel_size, **kwargs):
                padding = _get_padding(kernel_size, **kwargs)
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
            else:
                return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            self.add_module(str(idx), conv2d_pad(in_ch, out_ch, k, stride=stride, padding=padding, dilation=d, groups=conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x


def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs
    if isinstance(kernel_size, list):
        return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)


class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, pad_type='', act_fn=F.relu, bn_args=_BN_ARGS_PT):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        return x


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def drop_connect(inputs, training=False, drop_connect_rate=0.0):
    """Apply drop connect."""
    if not training:
        return inputs
    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()
    output = inputs.div(keep_prob) * random_tensor
    return output


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, pad_type='', act_fn=F.relu, noskip=False, pw_kernel_size=1, pw_act=False, se_ratio=0.0, se_gate_fn=sigmoid, bn_args=_BN_ARGS_PT, drop_connect_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        self.conv_dw = select_conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)
        if self.has_se:
            self.se = SqueezeExcite(in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)
        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        if self.has_se:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x, inplace=True)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, pad_type='', act_fn=F.relu, noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_reduce_mid=False, se_gate_fn=sigmoid, shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.0):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)
        self.shuffle_type = shuffle_type
        if shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = ChannelShuffle(len(exp_kernel_size))
        self.conv_dw = select_conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        if self.shuffle_type == 'mid':
            x = self.shuffle(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)
        if self.has_se:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class _BlockBuilder:
    """ Build Trunk Blocks
    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py
    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None, pad_type='', act_fn=None, se_gate_fn=sigmoid, se_reduce_mid=False, bn_args=_BN_ARGS_PT, drop_connect_rate=0.0, verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_fn = act_fn
        self.se_gate_fn = se_gate_fn
        self.se_reduce_mid = se_reduce_mid
        self.bn_args = bn_args
        self.drop_connect_rate = drop_connect_rate
        self.verbose = verbose
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return _round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        ba['bn_args'] = self.bn_args
        ba['pad_type'] = self.pad_type
        ba['act_fn'] = ba['act_fn'] if ba['act_fn'] is not None else self.act_fn
        assert ba['act_fn'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(self.block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(self.block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(self.block_idx, str(ba)))
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']
        return block

    def _make_stack(self, stack_args):
        blocks = []
        for i, ba in enumerate(stack_args):
            if self.verbose:
                logging.info(' Block: {}'.format(i))
            if i >= 1:
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(block_args))
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        for stack_idx, stack in enumerate(block_args):
            if self.verbose:
                logging.info('Stack: {}'.format(stack_idx))
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


_DEBUG = False


def _initialize_weight_default(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')


def _initialize_weight_goog(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    channel_min = channel_min or divisor
    new_channels = max(int(channels + divisor / 2) // divisor * divisor, channel_min)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


class GenEfficientNet(nn.Module):
    """ Generic EfficientNet
    An implementation of efficient network architectures, in many cases mobile optimized networks:
      * MobileNet-V1
      * MobileNet-V2
      * MobileNet-V3
      * MnasNet A1, B1, and small
      * FBNet A, B, and C
      * ChamNet (arch details are murky)
      * Single-Path NAS Pixel1
      * EfficientNet B0-B7
      * MixNet S, M, L
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280, channel_multiplier=1.0, channel_divisor=8, channel_min=None, pad_type='', act_fn=F.relu, drop_rate=0.0, drop_connect_rate=0.0, se_gate_fn=sigmoid, se_reduce_mid=False, bn_args=_BN_ARGS_PT, global_pool='avg', head_conv='default', weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features
        stem_size = _round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **bn_args)
        in_chs = stem_size
        builder = _BlockBuilder(channel_multiplier, channel_divisor, channel_min, pad_type, act_fn, se_gate_fn, se_reduce_mid, bn_args, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs
        if not head_conv or head_conv == 'none':
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.efficient_head = head_conv == 'efficient'
            self.conv_head = select_conv2d(in_chs, self.num_features, 1, padding=pad_type)
            self.bn2 = None if self.efficient_head else nn.BatchNorm2d(self.num_features, **bn_args)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)
        for m in self.modules():
            if weight_init == 'goog':
                _initialize_weight_goog(m)
            else:
                _initialize_weight_default(m)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.classifier
        if num_classes:
            self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.classifier = None

    def forward_features(self, x, pool=True):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks(x)
        if self.efficient_head:
            x = self.global_pool(x)
            x = self.conv_head(x)
            x = self.act_fn(x, inplace=True)
            if pool:
                x = x.view(x.size(0), -1)
        else:
            if self.conv_head is not None:
                x = self.conv_head(x)
                x = self.bn2(x)
            x = self.act_fn(x, inplace=True)
            if pool:
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]


def hard_swish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x + 3.0) / 6.0)
    else:
        return x * F.relu6(x + 3.0) / 6.0


def swish(x, inplace=False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x * x.sigmoid()


def _decode_block_str(block_str, depth_multiplier=1.0):
    """ Decode block definition string
    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            key = op[0]
            v = op[1:]
            if v == 're':
                value = F.relu
            elif v == 'r6':
                value = F.relu6
            elif v == 'hs':
                value = hard_swish
            elif v == 'sw':
                value = swish
            else:
                continue
            options[key] = value
        else:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
    act_fn = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    num_repeat = int(options['r'])
    if block_type == 'ir':
        block_args = dict(block_type=block_type, dw_kernel_size=_parse_ksize(options['k']), exp_kernel_size=exp_kernel_size, pw_kernel_size=pw_kernel_size, out_chs=int(options['c']), exp_ratio=float(options['e']), se_ratio=float(options['se']) if 'se' in options else None, stride=int(options['s']), act_fn=act_fn, noskip=noskip)
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(block_type=block_type, dw_kernel_size=_parse_ksize(options['k']), pw_kernel_size=pw_kernel_size, out_chs=int(options['c']), se_ratio=float(options['se']) if 'se' in options else None, stride=int(options['s']), act_fn=act_fn, pw_act=block_type == 'dsa', noskip=block_type == 'dsa' or noskip)
    elif block_type == 'cn':
        block_args = dict(block_type=block_type, kernel_size=int(options['k']), out_chs=int(options['c']), stride=int(options['s']), act_fn=act_fn)
    else:
        assert False, 'Unknown block type (%s)' % block_type
    num_repeat = int(math.ceil(num_repeat * depth_multiplier))
    return [deepcopy(block_args) for _ in range(num_repeat)]


def _decode_arch_def(arch_def, depth_multiplier=1.0):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            stack_args.extend(_decode_block_str(block_str, depth_multiplier))
        arch_args.append(stack_args)
    return arch_args


_BN_EPS_TF_DEFAULT = 0.001


_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99


_BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)


def _resolve_bn_args(kwargs):
    bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


class MobileNet(BaseImageNet):
    """
    Implements a search space as in MnasNet (https://arxiv.org/abs/1807.11626) using inverted residuals.
    """

    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(MobileNet, self).__init__(config, in_features, out_features, final_activation)
        nn.Module.config = config
        self.final_activation = final_activation
        self.nr_main_blocks = config['nr_main_blocks']
        self.initial_filters = config['initial_filters']
        self.nr_sub_blocks = dict([('Group_%d' % (i + 1), config['nr_sub_blocks_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.op_types = dict([('Group_%d' % (i + 1), config['op_type_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.kernel_sizes = dict([('Group_%d' % (i + 1), config['kernel_size_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.strides = dict([('Group_%d' % (i + 1), config['stride_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.output_filters = dict([('Group_%d' % (i + 1), config['out_filters_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.skip_cons = dict([('Group_%d' % (i + 1), config['skip_con_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        self.se_ratios = dict([('Group_%d' % (i + 1), config['se_ratio_%i' % (i + 1)]) for i in range(self.nr_main_blocks)])
        encoder = Arch_Encoder(block_types=self.op_types, nr_sub_blocks=self.nr_sub_blocks, kernel_sizes=self.kernel_sizes, strides=self.strides, expansion_rates=0, output_filters=self.output_filters, se_ratios=self.se_ratios, skip_connections=self.skip_cons)
        arch_enc = encoder.get_encoded_architecture()
        kwargs['bn_momentum'] = 0.01
        self.model = GenEfficientNet(_decode_arch_def(arch_enc, depth_multiplier=1.0), num_classes=out_features, stem_size=self.initial_filters, channel_multiplier=1.0, num_features=_round_channels(1280, 1.0, 8, None), bn_args=_resolve_bn_args(kwargs), act_fn=swish, drop_connect_rate=0.2, drop_rate=0.2, **kwargs)

        def _cfg(url='', **kwargs):
            return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7), 'crop_pct': 0.875, 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'first_conv': 'conv_stem', 'classifier': 'classifier', **kwargs}
        self.model.default_cfg = _cfg(url='', input_size=in_features, pool_size=(10, 10), crop_pct=0.904, num_classes=out_features)

    def forward(self, x):
        x = self.model(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    @staticmethod
    def get_config_space(nr_main_blocks=[3, 7], initial_filters=([8, 32], True), nr_sub_blocks=([1, 4], False), op_types=['inverted_residual', 'dwise_sep_conv'], kernel_sizes=[3, 5], strides=[1, 2], output_filters=[[12, 16, 20], [18, 24, 30], [24, 32, 40], [48, 64, 80], [72, 96, 120], [120, 160, 200], [240, 320, 400]], skip_connection=[True, False], se_ratios=[0, 0.25], **kwargs):
        cs = CS.ConfigurationSpace()
        main_blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'nr_main_blocks', nr_main_blocks)
        initial_filters_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'initial_filters', initial_filters)
        cs.add_hyperparameter(main_blocks_hp)
        cs.add_hyperparameter(initial_filters_hp)
        if type(nr_main_blocks[0]) == int:
            min_blocks = nr_main_blocks[0]
            max_blocks = nr_main_blocks[1]
        else:
            min_blocks = nr_main_blocks[0][0]
            max_blocks = nr_main_blocks[0][1]
        for i in range(1, max_blocks + 1):
            sub_blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'nr_sub_blocks_%d' % i, nr_sub_blocks)
            op_type_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'op_type_%d' % i, op_types)
            kernel_size_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'kernel_size_%d' % i, kernel_sizes)
            stride_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'stride_%d' % i, strides)
            out_filters_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'out_filters_%d' % i, output_filters[i - 1])
            se_ratio_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'se_ratio_%d' % i, se_ratios)
            cs.add_hyperparameter(sub_blocks_hp)
            cs.add_hyperparameter(op_type_hp)
            cs.add_hyperparameter(kernel_size_hp)
            cs.add_hyperparameter(stride_hp)
            cs.add_hyperparameter(out_filters_hp)
            cs.add_hyperparameter(se_ratio_hp)
            skip_con = cs.add_hyperparameter(CSH.CategoricalHyperparameter('skip_con_%d' % i, [True, False]))
            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(sub_blocks_hp, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(op_type_hp, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(kernel_size_hp, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(stride_hp, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(out_filters_hp, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(skip_con, main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(se_ratio_hp, main_blocks_hp, i - 1))
        return cs


class SkipConnection(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection, self).__init__()
        self.s1 = nn.Sequential()
        self.s1.add_module('Skip_1_AvgPool', nn.AvgPool2d(1, stride=stride))
        self.s1.add_module('Skip_1_Conv', nn.Conv2d(in_channels, int(out_channels / 2), kernel_size=1, stride=1, padding=0, bias=False))
        self.s2 = nn.Sequential()
        self.s2.add_module('Skip_2_AvgPool', nn.AvgPool2d(1, stride=stride))
        self.s2.add_module('Skip_2_Conv', nn.Conv2d(in_channels, int(out_channels / 2) if out_channels % 2 == 0 else int(out_channels / 2) + 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(x, inplace=False)
        out1 = self.s1(out1)
        out2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        out2 = self.s2(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.batch_norm(out)
        return out


class ResidualBranch(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride, branch_index):
        super(ResidualBranch, self).__init__()
        self.residual_branch = nn.Sequential()
        self.residual_branch.add_module('Branch_{}:ReLU_1'.format(branch_index), nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'.format(branch_index), nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=round(filter_size / 3), bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'.format(branch_index), nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'.format(branch_index), nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'.format(branch_index), nn.Conv2d(out_channels, out_channels, kernel_size=filter_size, stride=1, padding=round(filter_size / 3), bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'.format(branch_index), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


class ResidualGroup(nn.Module):

    def __init__(self, block, n_input_plane, n_output_plane, n_blocks, filter_size, res_branches, stride, shake_config):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks
        self.group.add_module('Block_1', block(n_input_plane, n_output_plane, filter_size, res_branches, stride=stride, shake_config=shake_config))
        for block_index in range(2, n_blocks + 1):
            block_name = 'Block_{}'.format(block_index)
            self.group.add_module(block_name, block(n_output_plane, n_output_plane, filter_size, res_branches, stride=1, shake_config=shake_config))

    def forward(self, x):
        return self.group(x)


class ResNet152(ResNet):

    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(ResNet, self).__init__(config, in_features, out_features, final_activation)
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes=out_features)

    def forward(self, x):
        x = super(ResNet152, self).forward(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AdaptiveCatAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BaseNet,
     lambda: ([], {'config': _mock_config(), 'in_features': 4, 'out_features': 4, 'final_activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBnAct,
     lambda: ([], {'in_chs': 4, 'out_chs': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseSeparableConv,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GenEfficientNet,
     lambda: ([], {'block_args': _mock_config()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MixedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoEmbedding,
     lambda: ([], {'config': _mock_config(), 'in_features': 4, 'one_hot_encoder': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PrintNode,
     lambda: ([], {'msg': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'config': _mock_config(blocks_per_group=4, num_groups=1, use_dropout=0.5, use_shake_shake=4, use_shake_drop=4, max_shake_drop_probability=4), 'in_features': 4, 'out_features': 4, 'block_index': 4, 'dropout': 0.5, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ResNet152,
     lambda: ([], {'config': _mock_config(), 'in_features': [4, 4], 'out_features': 4, 'final_activation': _mock_layer()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Reshape,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBranch,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'filter_size': 4, 'stride': 1, 'branch_index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelectAdaptivePool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipConnection,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExcite,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_automl_Auto_PyTorch(_paritybench_base):
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

