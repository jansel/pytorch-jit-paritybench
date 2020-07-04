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
checkpoints = _module
load_specific = _module
save_load = _module
lr_scheduling = _module
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


import inspect


from collections import OrderedDict


import torch


import numpy as np


import re


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import math


import logging


from torch.autograd import Variable


from copy import deepcopy


from torch.autograd import Function


import time


import random


import scipy.sparse


import copy


import torchvision.models as models


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.nn.modules.loss import _Loss


from torch.nn import Linear


import torch.optim.lr_scheduler as lr_scheduler


import torch.optim as optim


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
        self.best_parameters = OrderedDict({key: value.cpu().clone() for 
            key, value in self.state_dict().items()})

    def load_snapshot(self):
        if self.best_parameters is not None:
            self.load_state_dict(self.best_parameters)

    @staticmethod
    def get_config_space():
        return ConfigSpace.ConfigurationSpace()


def get_hyperparameter(hyper_type, name, value_range, log=False):
    if isinstance(value_range, tuple) and len(value_range) == 2 and isinstance(
        value_range[1], bool) and isinstance(value_range[0], (tuple, list)):
        value_range, log = value_range
    if len(value_range) == 0:
        raise ValueError(name +
            ': The range has to contain at least one element')
    if len(value_range) == 1:
        return CSH.Constant(name, int(value_range[0]) if isinstance(
            value_range[0], bool) else value_range[0])
    if len(value_range) == 2 and value_range[0] == value_range[1]:
        return CSH.Constant(name, int(value_range[0]) if isinstance(
            value_range[0], bool) else value_range[0])
    if hyper_type == CSH.CategoricalHyperparameter:
        return CSH.CategoricalHyperparameter(name, value_range)
    if hyper_type == CSH.UniformFloatHyperparameter:
        assert len(value_range
            ) == 2, 'Float HP range update for %s is specified by the two upper and lower values. %s given.' % (
            name, len(value_range))
        return CSH.UniformFloatHyperparameter(name, lower=value_range[0],
            upper=value_range[1], log=log)
    if hyper_type == CSH.UniformIntegerHyperparameter:
        assert len(value_range
            ) == 2, 'Int HP range update for %s is specified by the two upper and lower values. %s given.' % (
            name, len(value_range))
        return CSH.UniformIntegerHyperparameter(name, lower=value_range[0],
            upper=value_range[1], log=log)
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
        self.num_numerical = len([f for f in one_hot_encoder.
            categorical_features if not f])
        self.num_input_features = [len(c) for c in one_hot_encoder.categories_]
        self.embed_features = [(num_in >= config[
            'min_unique_values_for_embedding']) for num_in in self.
            num_input_features]
        self.num_output_dimensions = [(config['dimension_reduction_' + str(
            i)] * num_in) for i, num_in in enumerate(self.num_input_features)]
        self.num_output_dimensions = [int(np.clip(num_out, 1, num_in - 1)) for
            num_out, num_in in zip(self.num_output_dimensions, self.
            num_input_features)]
        self.num_output_dimensions = [(num_out if embed else num_in) for 
            num_out, embed, num_in in zip(self.num_output_dimensions, self.
            embed_features, self.num_input_features)]
        self.num_out_feats = self.num_numerical + sum(self.
            num_output_dimensions)
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
            concat_seq.append(self.ee_layers[layer_pointer](
                categorical_feature_slice))
            layer_pointer += 1
            x_pointer += num_in
            last_concat = x_pointer
        concat_seq.append(x[:, last_concat:])
        return torch.cat(concat_seq, dim=1)

    def _create_ee_layers(self, in_features):
        layers = nn.ModuleList()
        for i, (num_in, embed, num_out) in enumerate(zip(self.
            num_input_features, self.embed_features, self.
            num_output_dimensions)):
            if not embed:
                continue
            layers.append(nn.Linear(num_in, num_out))
        return layers

    @staticmethod
    def get_config_space(categorical_features=None,
        min_unique_values_for_embedding=((3, 300), True),
        dimension_reduction=(0, 1), **kwargs):
        if categorical_features is None or not any(categorical_features):
            return CS.ConfigurationSpace()
        cs = CS.ConfigurationSpace()
        min_hp = get_hyperparameter(CSH.UniformIntegerHyperparameter,
            'min_unique_values_for_embedding', min_unique_values_for_embedding)
        cs.add_hyperparameter(min_hp)
        for i in range(len([x for x in categorical_features if x])):
            ee_dimensions_hp = get_hyperparameter(CSH.
                UniformFloatHyperparameter, 'dimension_reduction_' + str(i),
                kwargs.pop('dimension_reduction_' + str(i),
                dimension_reduction))
            cs.add_hyperparameter(ee_dimensions_hp)
        assert len(kwargs
            ) == 0, 'Invalid hyperparameter updates for learned embedding: %s' % str(
            kwargs)
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


class ShakeDrop(Function):

    @staticmethod
    def forward(ctx, x, alpha, beta, death_rate, is_train):
        gate = (torch.rand(1) > death_rate).numpy()
        ctx.gate = gate
        ctx.save_for_backward(x, alpha, beta)
        if is_train:
            if not gate:
                y = alpha * x
            else:
                y = x
        else:
            y = x.mul(1 - death_rate * 1.0)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta = ctx.saved_variables
        grad_x1 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            if not ctx.gate:
                grad_x = grad_output * beta
            else:
                grad_x = grad_output
        return grad_x, grad_alpha, grad_beta, None, None


shake_drop = ShakeDrop.apply


def shake_drop_get_bl(block_index, min_prob_no_shake, num_blocks,
    is_training, is_cuda):
    pl = 1 - (block_index + 1) / num_blocks * (1 - min_prob_no_shake)
    if not is_training:
        bl = torch.tensor(1.0) if random.random() <= pl else torch.tensor(0.0)
    if is_training:
        bl = torch.tensor(pl)
    if is_cuda:
        bl = bl.cuda()
    return bl


def shake_get_alpha_beta(is_training, is_cuda):
    if is_training:
        result = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
        return result if not is_cuda else (result[0].cuda(), result[1].cuda())
    alpha = torch.rand(1)
    beta = torch.rand(1)
    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
    return alpha, beta


class ShakeShakeBlock(Function):

    @staticmethod
    def forward(ctx, alpha, beta, *args):
        ctx.save_for_backward(beta)
        y = sum(alpha[i] * args[i] for i in range(len(args)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        beta = ctx.saved_variables
        grad_x = [(beta[0][i] * grad_output) for i in range(beta[0].shape[0])]
        return None, None, *grad_x


shake_shake = ShakeShakeBlock.apply


class ResBlock(nn.Module):

    def __init__(self, config, in_features, out_features, block_index,
        dropout, activation):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation
        self.shortcut = None
        self.start_norm = None
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(nn.BatchNorm1d(in_features),
                self.activation())
        self.block_index = block_index
        self.num_blocks = self.config['blocks_per_group'] * self.config[
            'num_groups']
        self.layers = self._build_block(in_features, out_features)
        if config['use_shake_shake']:
            self.shake_shake_layers = self._build_block(in_features,
                out_features)

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
            bl = shake_drop_get_bl(self.block_index, 1 - self.config[
                'max_shake_drop_probability'], self.num_blocks, self.
                training, x.is_cuda)
            x = shake_drop(x, alpha, beta, bl)
        x = x + residual
        return x


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda
    C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
    count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine: nn.
    MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C,
    stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C,
    affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C,
    3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine:
    SepConv(C, C, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C,
    stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2,
    affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C,
    5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C, stride,
    affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 7),
    stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C, C, (7, 1),
    stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C,
    affine=affine))}


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        try:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).
                bernoulli_(keep_prob))
        except:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).
                bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
        reduction_prev):
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
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(
            inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in,
            C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(
            C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1,
            padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn
            .ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=
            kernel_size, stride=1, padding=padding, groups=C_in, bias=False
            ), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, pool_size):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=
            pool_size))


logger = logging.getLogger('autonet')


class PrintNode(nn.Module):

    def __init__(self, msg):
        super(PrintNode, self).__init__()
        self.msg = msg

    def forward(self, x):
        logger.debug(self.msg)
        return x


all_activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU, 'selu': nn.SELU, 'rrelu': nn.RReLU,
    'tanhshrink': nn.Tanhshrink, 'hardtanh': nn.Hardtanh, 'elu': nn.ELU,
    'prelu': nn.PReLU}


def get_activation(name, inplace=False):
    if name not in all_activations:
        raise ValueError('Activation ' + str(name) + ' not defined')
    activation = all_activations[name]
    activation_kwargs = {'inplace': True} if 'inplace' in inspect.getargspec(
        activation)[0] else dict()
    return activation(**activation_kwargs)


class _DenseLayer(nn.Sequential):

    def __init__(self, nChannels, growth_rate, drop_rate, bottleneck,
        kernel_size, activation):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(nChannels))
        self.add_module('relu1', get_activation(activation, inplace=True))
        if bottleneck:
            self.add_module('conv1', nn.Conv2d(nChannels, 4 * growth_rate,
                kernel_size=1, stride=1, bias=False))
            nChannels = 4 * growth_rate
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True)
                    )
            self.add_module('norm2', nn.BatchNorm2d(nChannels))
            self.add_module('relu2', get_activation(activation, inplace=True))
        self.add_module('conv2', nn.Conv2d(nChannels, growth_rate,
            kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1
            ) / 2), bias=False))
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, N, nChannels, growth_rate, drop_rate, bottleneck,
        kernel_size, activation):
        super(_DenseBlock, self).__init__()
        for i in range(N):
            self.add_module('denselayer%d' % (i + 1), _DenseLayer(nChannels,
                growth_rate, drop_rate, bottleneck, kernel_size, activation))
            nChannels += growth_rate


class _Transition(nn.Sequential):

    def __init__(self, nChannels, nOutChannels, drop_rate, last, pool_size,
        kernel_size, stride, padding, activation):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(nChannels))
        self.add_module('relu', get_activation(activation, inplace=True))
        if last:
            self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size,
                stride=pool_size))
            self.add_module('reshape', Reshape(nChannels))
        else:
            self.add_module('conv', nn.Conv2d(nChannels, nOutChannels,
                kernel_size=1, stride=1, bias=False))
            if drop_rate > 0:
                self.add_module('drop', nn.Dropout2d(p=drop_rate, inplace=True)
                    )
            self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size,
                stride=stride, padding=padding))


class SkipConnection(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection, self).__init__()
        self.s1 = nn.Sequential()
        self.s1.add_module('Skip_1_AvgPool', nn.AvgPool2d(1, stride=stride))
        self.s1.add_module('Skip_1_Conv', nn.Conv2d(in_channels, int(
            out_channels / 2), kernel_size=1, stride=1, padding=0, bias=False))
        self.s2 = nn.Sequential()
        self.s2.add_module('Skip_2_AvgPool', nn.AvgPool2d(1, stride=stride))
        self.s2.add_module('Skip_2_Conv', nn.Conv2d(in_channels, int(
            out_channels / 2) if out_channels % 2 == 0 else int(
            out_channels / 2) + 1, kernel_size=1, stride=1, padding=0, bias
            =False))
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

    def __init__(self, in_channels, out_channels, filter_size, stride,
        branch_index):
        super(ResidualBranch, self).__init__()
        self.residual_branch = nn.Sequential()
        self.residual_branch.add_module('Branch_{}:ReLU_1'.format(
            branch_index), nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'.format(
            branch_index), nn.Conv2d(in_channels, out_channels, kernel_size
            =filter_size, stride=stride, padding=round(filter_size / 3),
            bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'.format(
            branch_index), nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'.format(
            branch_index), nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'.format(
            branch_index), nn.Conv2d(out_channels, out_channels,
            kernel_size=filter_size, stride=1, padding=round(filter_size / 
            3), bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'.format(
            branch_index), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


def generate_alpha_beta(num_branches, batch_size, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config
    if forward_shake and not shake_image:
        alpha = torch.rand(num_branches)
    elif forward_shake and shake_image:
        alpha = torch.rand(num_branches, batch_size).view(num_branches,
            batch_size, 1, 1, 1)
    else:
        alpha = torch.ones(num_branches)
    if backward_shake and not shake_image:
        beta = torch.rand(num_branches)
    elif backward_shake and shake_image:
        beta = torch.rand(num_branches, batch_size).view(num_branches,
            batch_size, 1, 1, 1)
    else:
        beta = torch.ones(num_branches)
    alpha = torch.nn.Softmax(0)(Variable(alpha))
    beta = torch.nn.Softmax(0)(Variable(beta))
    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
    return alpha, beta


def generate_alpha_beta_single(tensor_size, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config
    if forward_shake and not shake_image:
        alpha = torch.rand(tensor_size).mul(2).add(-1)
    elif forward_shake and shake_image:
        alpha = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
        alpha.mul_(2).add_(-1)
    else:
        alpha = torch.FloatTensor([0.5])
    if backward_shake and not shake_image:
        beta = torch.rand(tensor_size)
    elif backward_shake and shake_image:
        beta = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])
    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
    return Variable(alpha), Variable(beta)


class BasicBlock(nn.Module):

    def __init__(self, n_input_plane, n_output_plane, filter_size,
        res_branches, stride, shake_config):
        super(BasicBlock, self).__init__()
        self.shake_config = shake_config
        self.branches = nn.ModuleList([ResidualBranch(n_input_plane,
            n_output_plane, filter_size, stride, branch + 1) for branch in
            range(res_branches)])
        self.skip = nn.Sequential()
        if n_input_plane != n_output_plane or stride != 1:
            self.skip.add_module('Skip_connection', SkipConnection(
                n_input_plane, n_output_plane, stride))

    def forward(self, x):
        if len(self.branches) == 1:
            out = self.branches[0](x)
            if self.config.apply_shakeDrop:
                alpha, beta = generate_alpha_beta_single(out.size(), self.
                    shake_config if self.training else (False, False, False
                    ), x.is_cuda)
                out = shake_drop(out, alpha, beta, self.config.death_rate,
                    self.training)
        elif self.config.apply_shakeShake:
            alpha, beta = generate_alpha_beta(len(self.branches), x.size(0),
                self.shake_config if self.training else (False, False, 
                False), x.is_cuda)
            branches = [self.branches[i](x) for i in range(len(self.branches))]
            out = shake_shake(alpha, beta, *branches)
        else:
            out = sum([self.branches[i](x) for i in range(len(self.branches))])
        return out + self.skip(x)


class ResidualGroup(nn.Module):

    def __init__(self, block, n_input_plane, n_output_plane, n_blocks,
        filter_size, res_branches, stride, shake_config):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks
        self.group.add_module('Block_1', block(n_input_plane,
            n_output_plane, filter_size, res_branches, stride=stride,
            shake_config=shake_config))
        for block_index in range(2, n_blocks + 1):
            block_name = 'Block_{}'.format(block_index)
            self.group.add_module(block_name, block(n_output_plane,
                n_output_plane, filter_size, res_branches, stride=1,
                shake_config=shake_config))

    def forward(self, x):
        return self.group(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


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


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


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
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=
                    padding, **kwargs)
            else:
                return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding,
                **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **
            kwargs)


class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [
            kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits,
            out_splits)):
            d = 1
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            self.add_module(str(idx), conv2d_pad(in_ch, out_ch, k, stride=
                stride, padding=padding, dilation=d, groups=conv_groups, **
                kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x


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
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.
            output_size) + ', pool_type=' + self.pool_type + ')'


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(
            g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid
        ):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.
            size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


_BN_EPS_PT_DEFAULT = 1e-05


_BN_MOMENTUM_PT_DEFAULT = 0.01


_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)


def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs
    if isinstance(kernel_size, list):
        return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs
            )


class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, pad_type='',
        act_fn=F.relu, bn_args=_BN_ARGS_PT):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=
            stride, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        return x


def drop_connect(inputs, training=False, drop_connect_rate=0.0):
    """Apply drop connect."""
    if not training:
        return inputs
    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1),
        dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()
    output = inputs.div(keep_prob) * random_tensor
    return output


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        pad_type='', act_fn=F.relu, noskip=False, pw_kernel_size=1, pw_act=
        False, se_ratio=0.0, se_gate_fn=sigmoid, bn_args=_BN_ARGS_PT,
        drop_connect_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        self.conv_dw = select_conv2d(in_chs, in_chs, dw_kernel_size, stride
            =stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)
        if self.has_se:
            self.se = SqueezeExcite(in_chs, reduce_chs=max(1, int(in_chs *
                se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)
        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size,
            padding=pad_type)
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


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        pad_type='', act_fn=F.relu, noskip=False, exp_ratio=1.0,
        exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_reduce_mid=
        False, se_gate_fn=sigmoid, shuffle_type=None, bn_args=_BN_ARGS_PT,
        drop_connect_rate=0.0):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size,
            padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)
        self.shuffle_type = shuffle_type
        if shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = ChannelShuffle(len(exp_kernel_size))
        self.conv_dw = select_conv2d(mid_chs, mid_chs, dw_kernel_size,
            stride=stride, padding=pad_type, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(
                se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size,
            padding=pad_type)
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

    def __init__(self, channel_multiplier=1.0, channel_divisor=8,
        channel_min=None, pad_type='', act_fn=None, se_gate_fn=sigmoid,
        se_reduce_mid=False, bn_args=_BN_ARGS_PT, drop_connect_rate=0.0,
        verbose=False):
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
        return _round_channels(chs, self.channel_multiplier, self.
            channel_divisor, self.channel_min)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        ba['bn_args'] = self.bn_args
        ba['pad_type'] = self.pad_type
        ba['act_fn'] = ba['act_fn'] if ba['act_fn'
            ] is not None else self.act_fn
        assert ba['act_fn'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'
                ] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(self.
                    block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_connect_rate'
                ] = self.drop_connect_rate * self.block_idx / self.block_count
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(
                    self.block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(self.
                    block_idx, str(ba)))
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
            logging.info('Building model trunk with %d stages...' % len(
                block_args))
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
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear'
            )


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
    new_channels = max(int(channels + divisor / 2) // divisor * divisor,
        channel_min)
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

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=
        32, num_features=1280, channel_multiplier=1.0, channel_divisor=8,
        channel_min=None, pad_type='', act_fn=F.relu, drop_rate=0.0,
        drop_connect_rate=0.0, se_gate_fn=sigmoid, se_reduce_mid=False,
        bn_args=_BN_ARGS_PT, global_pool='avg', head_conv='default',
        weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features
        stem_size = _round_channels(stem_size, channel_multiplier,
            channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **bn_args)
        in_chs = stem_size
        builder = _BlockBuilder(channel_multiplier, channel_divisor,
            channel_min, pad_type, act_fn, se_gate_fn, se_reduce_mid,
            bn_args, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs
        if not head_conv or head_conv == 'none':
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.efficient_head = head_conv == 'efficient'
            self.conv_head = select_conv2d(in_chs, self.num_features, 1,
                padding=pad_type)
            self.bn2 = None if self.efficient_head else nn.BatchNorm2d(self
                .num_features, **bn_args)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), self.num_classes)
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
            self.classifier = nn.Linear(self.num_features * self.
                global_pool.feat_mult(), num_classes)
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


class Reshape(nn.Module):

    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, x):
        return x.reshape(-1, self.size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_automl_Auto_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AdaptiveAvgMaxPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(AdaptiveCatAvgMaxPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BaseNet(*[], **{'config': _mock_config(), 'in_features': 4, 'out_features': 4, 'final_activation': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Conv2dSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ConvBnAct(*[], **{'in_chs': 4, 'out_chs': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(DepthwiseSeparableConv(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(DilConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(FactorizedReduce(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(GenEfficientNet(*[], **{'block_args': _mock_config()}), [torch.rand([4, 3, 64, 64])], {})

    def test_011(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(InvertedResidual(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(MixedConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(NoEmbedding(*[], **{'config': _mock_config(), 'in_features': 4, 'one_hot_encoder': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(PrintNode(*[], **{'msg': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(ReLUConvBN(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(Reshape(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(ResidualBranch(*[], **{'in_channels': 4, 'out_channels': 4, 'filter_size': 4, 'stride': 1, 'branch_index': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(SelectAdaptivePool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(SepConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(SkipConnection(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_022(self):
        self._check(SqueezeExcite(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_023(self):
        self._check(Zero(*[], **{'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

