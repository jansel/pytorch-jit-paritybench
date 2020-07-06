import sys
_module = sys.modules[__name__]
del sys
conf = _module
eval = _module
find_neighbour_dist = _module
forward_scripts = _module
forward = _module
find_env = _module
find_runs = _module
descriptor_matcher = _module
fpfh = _module
misc = _module
save_feature = _module
test = _module
mock_models = _module
mockdatasets = _module
test_api = _module
test_basedataset = _module
test_basemodel = _module
test_batch = _module
test_bn_scheduler = _module
test_boxstuff = _module
test_confusionMatrix = _module
test_dataset_factory = _module
test_filter = _module
test_fps = _module
test_grid_sampling = _module
test_ind_tracker = _module
test_interpolateop = _module
test_kpconv = _module
test_losses = _module
test_lr_scheduler = _module
test_make_pair = _module
test_model_checkpoint = _module
test_models = _module
test_modules = _module
test_msdata = _module
test_msdatapair = _module
test_normal = _module
test_random_sphere = _module
test_registration_metrics = _module
test_registration_tracker = _module
test_resolver = _module
test_sampler = _module
test_samplers = _module
test_sampling_strategy = _module
test_segmentationtracker = _module
test_shapenetforward = _module
test_shapenetparttracker = _module
test_sphere_sampling = _module
test_to_sparse = _module
test_transform = _module
test_unwrapped_unet_base = _module
test_visualization = _module
utils = _module
torch_points3d = _module
applications = _module
kpconv = _module
modelfactory = _module
models = _module
pointnet2 = _module
rsconv = _module
core = _module
base_conv = _module
dense = _module
message_passing = _module
partial_dense = _module
common_modules = _module
base_modules = _module
dense_modules = _module
spatial_transform = _module
data_transform = _module
feature_augment = _module
features = _module
filters = _module
grid_transform = _module
inference_transforms = _module
sparse_transforms = _module
transforms = _module
initializer = _module
initializer = _module
losses = _module
dirichlet_loss = _module
huber_loss = _module
losses = _module
metric_losses = _module
regularizer = _module
regularizers = _module
schedulers = _module
bn_schedulers = _module
lr_schedulers = _module
spatial_ops = _module
interpolate = _module
neighbour_finder = _module
sampling = _module
datasets = _module
base_dataset = _module
batch = _module
classification = _module
modelnet = _module
dataset_factory = _module
multiscale_data = _module
object_detection = _module
box_data = _module
scannet = _module
base3dmatch = _module
base_siamese_dataset = _module
basetest = _module
detector = _module
fusion = _module
general3dmatch = _module
pair = _module
test3dmatch = _module
utils = _module
samplers = _module
segmentation = _module
shapenet = _module
s3dis = _module
scannet = _module
shapenet = _module
metrics = _module
base_tracker = _module
box_detection = _module
ap = _module
classification_tracker = _module
colored_tqdm = _module
confusion_matrix = _module
meters = _module
model_checkpoint = _module
object_detection_tracker = _module
registration_metrics = _module
registration_tracker = _module
s3dis_tracker = _module
scannet_segmentation_tracker = _module
segmentation_helpers = _module
segmentation_tracker = _module
shapenet_part_tracker = _module
base_architectures = _module
backbone = _module
unet = _module
base_model = _module
model_factory = _module
model_interface = _module
votenet = _module
base = _module
kpconv = _module
minkowski = _module
pointnet = _module
pointnet2 = _module
base = _module
kpconv = _module
minkowski = _module
pointcnn = _module
pointnet = _module
pointnet2 = _module
randlanet = _module
rsconv = _module
KPConv = _module
blocks = _module
convolution_ops = _module
kernel_utils = _module
kernels = _module
losses = _module
plyutils = _module
MinkowskiEngine = _module
common = _module
modules = _module
networks = _module
res16unet = _module
resunet = _module
PointCNN = _module
modules = _module
PointNet = _module
modules = _module
RSConv = _module
dense = _module
message_passing = _module
RandLANet = _module
modules = _module
VoteNet = _module
loss_helper = _module
proposal_module = _module
votenet_results = _module
voting_module = _module
dense = _module
box_utils = _module
colors = _module
config = _module
debugging_vars = _module
enums = _module
geometry = _module
mock = _module
activation_resolver = _module
model_definition_resolver = _module
resolver_utils = _module
running_stats = _module
timer = _module
transform_utils = _module
visualization = _module
experiment_manager = _module
visualizer = _module
train = _module

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


import torch


import logging


import numpy as np


import re


from collections import namedtuple


from collections import defaultdict


from torch import nn


from torch.nn import Sequential


from torch.nn import Linear as Lin


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn import BatchNorm1d as BN


from torch.nn import Dropout


import numpy.testing as npt


import numpy.matlib


import torch.testing as tt


import copy


from itertools import combinations


import queue


from abc import abstractmethod


from typing import *


from torch.nn.parameter import Parameter


import torch.nn as nn


from torch.nn import Linear


import random


from typing import List


from typing import Optional


import itertools


import math


from torch.nn import functional as F


from sklearn.neighbors import NearestNeighbors


from sklearn.neighbors import KDTree


from functools import partial


import numpy


import scipy


import torch.nn.functional as F


from torch.nn import init


from typing import Any


from collections import OrderedDict


from torch.autograd import Variable


from torch.optim import lr_scheduler


from torch.optim.lr_scheduler import LambdaLR


from abc import ABC


from typing import Union


from typing import cast


import collections


from torch.utils.data import Sampler


from itertools import repeat


from itertools import product


from typing import Dict


from torch.utils.tensorboard import SummaryWriter


from typing import Tuple


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn import Sequential as Seq


from torch.nn import Identity


from enum import Enum


from math import ceil


from torch.nn import Sequential as S


from torch.nn import Linear as L


from torch.nn import ELU


from torch.nn import Conv1d


from scipy.spatial import ConvexHull


import torch.nn


from numpy.lib import recfunctions as rfn


import time


class COLORS:
    """[This class is used to color the bash shell by using {} {} {} with 'COLORS.{}, text, COLORS.END_TOKEN']
    """
    TRAIN_COLOR = '\x1b[0;92m'
    VAL_COLOR = '\x1b[0;94m'
    TEST_COLOR = '\x1b[0;93m'
    BEST_COLOR = '\x1b[0;92m'
    END_TOKEN = '\x1b[0m)'
    END_NO_TOKEN = '\x1b[0m'
    Black = '\x1b[0;30m'
    Red = '\x1b[0;31m'
    Green = '\x1b[0;32m'
    Yellow = '\x1b[0;33m'
    Blue = '\x1b[0;34m'
    Purple = '\x1b[0;35m'
    Cyan = '\x1b[0;36m'
    White = '\x1b[0;37m'
    BBlack = '\x1b[1;30m'
    BRed = '\x1b[1;31m'
    BGreen = '\x1b[1;32m'
    BYellow = '\x1b[1;33m'
    BBlue = '\x1b[1;34m'
    BPurple = '\x1b[1;35m'
    BCyan = '\x1b[1;36m'
    BWhite = '\x1b[1;37m'
    UBlack = '\x1b[4;30m'
    URed = '\x1b[4;31m'
    UGreen = '\x1b[4;32m'
    UYellow = '\x1b[4;33m'
    UBlue = '\x1b[4;34m'
    UPurple = '\x1b[4;35m'
    UCyan = '\x1b[4;36m'
    UWhite = '\x1b[4;37m'
    On_Black = '\x1b[40m'
    On_Red = '\x1b[41m'
    On_Green = '\x1b[42m'
    On_Yellow = '\x1b[43m'
    On_Blue = '\x1b[44m'
    On_Purple = '\x1b[45m'
    On_Cyan = '\x1b[46m'
    On_White = '\x1b[47m'
    IBlack = '\x1b[0;90m'
    IRed = '\x1b[0;91m'
    IGreen = '\x1b[0;92m'
    IYellow = '\x1b[0;93m'
    IBlue = '\x1b[0;94m'
    IPurple = '\x1b[0;95m'
    ICyan = '\x1b[0;96m'
    IWhite = '\x1b[0;97m'
    BIBlack = '\x1b[1;90m'
    BIRed = '\x1b[1;91m'
    BIGreen = '\x1b[1;92m'
    BIYellow = '\x1b[1;93m'
    BIBlue = '\x1b[1;94m'
    BIPurple = '\x1b[1;95m'
    BICyan = '\x1b[1;96m'
    BIWhite = '\x1b[1;97m'
    On_IBlack = '\x1b[0;100m'
    On_IRed = '\x1b[0;101m'
    On_IGreen = '\x1b[0;102m'
    On_IYellow = '\x1b[0;103m'
    On_IBlue = '\x1b[0;104m'
    On_IPurple = '\x1b[10;95m'
    On_ICyan = '\x1b[0;106m'
    On_IWhite = '\x1b[0;107m'


class ConvolutionFormat(enum.Enum):
    DENSE = 'dense'
    PARTIAL_DENSE = 'partial_dense'
    MESSAGE_PASSING = 'message_passing'
    SPARSE = 'sparse'


class ConvolutionFormatFactory:

    @staticmethod
    def check_is_dense_format(conv_type):
        if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower() or conv_type.lower() == ConvolutionFormat.MESSAGE_PASSING.value.lower() or conv_type.lower() == ConvolutionFormat.SPARSE.value.lower():
            return False
        elif conv_type.lower() == ConvolutionFormat.DENSE.value.lower():
            return True
        else:
            raise NotImplementedError('Conv type {} not supported'.format(conv_type))


def from_data_list_token(data_list, follow_batch=[]):
    """ This is pretty a copy paste of the from data list of pytorch geometric
    batch object with the difference that indexes that are negative are not incremented
    """
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys
    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}
    for key in keys:
        batch[key] = []
    for key in follow_batch:
        batch['{}_batch'.format(key)] = []
    cumsum = {key: (0) for key in keys}
    batch.batch = []
    for i, data in enumerate(data_list):
        for key in data.keys:
            item = data[key]
            if torch.is_tensor(item) and item.dtype != torch.bool and cumsum[key] > 0:
                mask = item >= 0
                item[mask] = item[mask] + cumsum[key]
            if torch.is_tensor(item):
                size = item.size(data.__cat_dim__(key, data[key]))
            else:
                size = 1
            batch.__slices__[key].append(size + batch.__slices__[key][-1])
            cumsum[key] += data.__inc__(key, item)
            batch[key].append(item)
            if key in follow_batch:
                item = torch.full((size,), i, dtype=torch.long)
                batch['{}_batch'.format(key)].append(item)
        num_nodes = data.num_nodes
        if num_nodes is not None:
            item = torch.full((num_nodes,), i, dtype=torch.long)
            batch.batch.append(item)
    if num_nodes is None:
        batch.batch = None
    for key in batch.keys:
        item = batch[key][0]
        if torch.is_tensor(item):
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, item))
        elif isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        else:
            raise ValueError('Unsupported attribute type {} : {}'.format(type(item), item))
    if torch_geometric.is_debug_enabled():
        batch.debug()
    return batch.contiguous()


def explode_transform(transforms):
    """ Returns a flattened list of transform
    Arguments:
        transforms {[list | T.Compose]} -- Contains list of transform to be added

    Returns:
        [list] -- [List of transforms]
    """
    out = []
    if transforms is not None:
        if isinstance(transforms, Compose):
            out = copy.deepcopy(transforms.transforms)
        elif isinstance(transforms, list):
            out = copy.deepcopy(transforms)
        else:
            raise Exception('transforms should be provided either within a list or a Compose')
    return out


class FCompose(object):
    """
    allow to compose different filters using the boolean operation

    Parameter
    ---------
    list_filter: list
        list of different filter functions we want to apply
    boolean_operation: function, optional
        boolean function to compose the filter (take a pair and return a boolean)
    """

    def __init__(self, list_filter, boolean_operation=np.logical_and):
        self.list_filter = list_filter
        self.boolean_operation = boolean_operation

    def __call__(self, data):
        assert len(self.list_filter) > 0
        res = self.list_filter[0](data)
        for filter_fn in self.list_filter:
            res = self.boolean_operation(res, filter_fn(data))
        return res

    def __repr__(self):
        rep = '{}(['.format(self.__class__.__name__)
        for filt in self.list_filter:
            rep = rep + filt.__repr__() + ', '
        rep = rep + '])'
        return rep


_custom_transforms = sys.modules[__name__]


_torch_geometric_transforms = sys.modules['torch_geometric.transforms']


def instantiate_transform(transform_option, attr='transform'):
    """ Creates a transform from an OmegaConf dict such as
    transform: GridSampling3D
        params:
            size: 0.01
    """
    tr_name = getattr(transform_option, attr, None)
    try:
        tr_params = transform_option.params
    except KeyError:
        tr_params = None
    try:
        lparams = transform_option.lparams
    except KeyError:
        lparams = None
    cls = getattr(_custom_transforms, tr_name, None)
    if not cls:
        cls = getattr(_torch_geometric_transforms, tr_name, None)
        if not cls:
            raise ValueError('Transform %s is nowhere to be found' % tr_name)
    if tr_params and lparams:
        return cls(*lparams, **tr_params)
    if tr_params:
        return cls(**tr_params)
    if lparams:
        return cls(*lparams)
    return cls()


def instantiate_filters(filter_options):
    filters = []
    for filt in filter_options:
        filters.append(instantiate_transform(filt, 'filter'))
    return FCompose(filters)


def instantiate_transforms(transform_options):
    """ Creates a torch_geometric composite transform from an OmegaConf list such as
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))
    return T.Compose(transforms)


log = logging.getLogger(__name__)


class BaseFactory:

    def __init__(self, module_name_down, module_name_up, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == 'UP':
            return getattr(self.modules_lib, self.module_name_up, None)
        else:
            return getattr(self.modules_lib, self.module_name_down, None)


class BaseInternalLossModule(torch.nn.Module):
    """ABC for modules which have internal loss(es)
    """

    @abstractmethod
    def get_internal_losses(self) ->Dict[str, Any]:
        pass


class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError


class ElasticNetRegularizer(_Regularizer):
    """
    Elastic Net Regularizer
    """

    def __init__(self, model, lambda_reg=0.01, alpha_reg=0.01):
        super(ElasticNetRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * ((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=param_weights) + self.alpha_reg * ElasticNetRegularizer.__add_l1(var=param_weights))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * ((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value) + self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value))
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """

    def __init__(self, model, lambda_reg=0.01):
        super(L1Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight') and '1.weight' not in model_param_name and 'bn' not in model_param_name:
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
    """

    def __init__(self, model, lambda_reg=0.01):
        super(L2Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight') and '1.weight' not in model_param_name and 'bn' not in model_param_name:
                reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class RegularizerTypes(Enum):
    L1 = L1Regularizer
    L2 = L2Regularizer
    ELASTIC = ElasticNetRegularizer


class SchedulerUpdateOn(enum.Enum):
    ON_EPOCH = 'on_epoch'
    ON_NUM_BATCH = 'on_num_batch'
    ON_NUM_SAMPLE = 'on_num_sample'


def colored_print(color, msg):
    None


def set_bn_momentum_default(bn_momentum):
    """
    This function return a function which will assign `bn_momentum` to every module instance within `BATCH_NORM_MODULES`.
    """

    def fn(m):
        if isinstance(m, BATCH_NORM_MODULES):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):

    def __init__(self, model, bn_lambda, update_scheduler_on, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))
        self.model = model
        self.setter = setter
        self.bn_lambda = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch
        self._scheduler_opt = None
        self._update_scheduler_on = update_scheduler_on

    @property
    def update_scheduler_on(self):
        return self._update_scheduler_on

    @property
    def scheduler_opt(self):
        return self._scheduler_opt

    @scheduler_opt.setter
    def scheduler_opt(self, scheduler_opt):
        self._scheduler_opt = scheduler_opt

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        current_momemtum = self.bn_lambda(epoch)
        if not hasattr(self, 'current_momemtum'):
            self._current_momemtum = current_momemtum
        elif self._current_momemtum != current_momemtum:
            self._current_momemtum = current_momemtum
            log.info('Setting batchnorm momentum at {}'.format(current_momemtum))
        self.model.apply(self.setter(current_momemtum))

    def state_dict(self):
        return {'current_momemtum': self.bn_lambda(self.last_epoch), 'last_epoch': self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.current_momemtum = state_dict['current_momemtum']

    def __repr__(self):
        return '{}(base_momentum: {}, update_scheduler_on={})'.format(self.__class__.__name__, self.bn_lambda(self.last_epoch), self._update_scheduler_on)


def instantiate_bn_scheduler(model, bn_scheduler_opt):
    """Return a batch normalization scheduler
    Parameters:
        model          -- the nn network
        bn_scheduler_opt (option class) -- dict containing all the params to build the scheduler　
                              opt.bn_policy is the name of learning rate policy: lambda_rule | step | plateau | cosine
                              opt.params contains the scheduler_params to construct the scheduler
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    update_scheduler_on = bn_scheduler_opt.update_scheduler_on
    bn_scheduler_params = bn_scheduler_opt.params
    if bn_scheduler_opt.bn_policy == 'step_decay':
        bn_lambda = lambda e: max(bn_scheduler_params.bn_momentum * bn_scheduler_params.bn_decay ** int(e // bn_scheduler_params.decay_step), bn_scheduler_params.bn_clip)
    else:
        return NotImplementedError('bn_policy [%s] is not implemented', bn_scheduler_opt.bn_policy)
    bn_scheduler = BNMomentumScheduler(model, bn_lambda, update_scheduler_on)
    bn_scheduler.scheduler_opt = bn_scheduler_opt
    return bn_scheduler


_custom_losses = sys.modules['torch_points3d.core.losses.losses']


_torch_metric_learning_losses = sys.modules['pytorch_metric_learning.losses']


_torch_metric_learning_miners = sys.modules['pytorch_metric_learning.miners']


def instantiate_loss_or_miner(option, mode='loss'):
    """
    create a loss from an OmegaConf dict such as
    TripletMarginLoss.
    params:
        margin=0.1
    It can also instantiate a miner to better learn a loss
    """
    class_ = getattr(option, 'class', None)
    try:
        params = option.params
    except KeyError:
        params = None
    try:
        lparams = option.lparams
    except KeyError:
        lparams = None
    if 'loss' in mode:
        cls = getattr(_custom_losses, class_, None)
        if not cls:
            cls = getattr(_torch_metric_learning_losses, class_, None)
            if not cls:
                raise ValueError('loss %s is nowhere to be found' % class_)
    elif mode == 'miner':
        cls = getattr(_torch_metric_learning_miners, class_, None)
        if not cls:
            raise ValueError('miner %s is nowhere to be found' % class_)
    else:
        raise NotImplementedError('Cannot instantiate this mode {}'.format(mode))
    if params and lparams:
        return cls(*lparams, **params)
    if params:
        return cls(**params)
    if lparams:
        return cls(*params)
    return cls()


class LRScheduler:

    def __init__(self, scheduler, scheduler_params, update_scheduler_on):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._update_scheduler_on = update_scheduler_on

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return self._scheduler._scheduler_opt

    def __repr__(self):
        return '{}({}, update_scheduler_on={})'.format(self._scheduler.__class__.__name__, self._scheduler_params, self._update_scheduler_on)

    def step(self, *args, **kwargs):
        self._scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


_custom_lr_scheduler = sys.modules[__name__]


def collect_params(params, update_scheduler_on):
    """
    This function enable to handle if params contains on_epoch and on_iter or not.
    """
    on_epoch_params = params.get('on_epoch')
    on_batch_params = params.get('on_num_batch')
    on_sample_params = params.get('on_num_sample')

    def check_params(params):
        if params is not None:
            return params
        else:
            raise Exception("The lr_scheduler doesn't have policy {}. Options: {}".format(update_scheduler_on, SchedulerUpdateOn))
    if on_epoch_params or on_batch_params or on_sample_params:
        if update_scheduler_on == SchedulerUpdateOn.ON_EPOCH.value:
            return check_params(on_epoch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_BATCH.value:
            return check_params(on_batch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_SAMPLE.value:
            return check_params(on_sample_params)
        else:
            raise Exception("The provided update_scheduler_on {} isn't within {}".format(update_scheduler_on, SchedulerUpdateOn))
    else:
        return params


def instantiate_scheduler(optimizer, scheduler_opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        scheduler_opt (option class) -- dict containing all the params to build the scheduler　
                              opt.lr_policy is the name of learning rate policy: lambda_rule | step | plateau | cosine
                              opt.params contains the scheduler_params to construct the scheduler
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    update_scheduler_on = scheduler_opt.update_scheduler_on
    scheduler_cls_name = getattr(scheduler_opt, 'class')
    scheduler_params = collect_params(scheduler_opt.params, update_scheduler_on)
    try:
        scheduler_cls = getattr(lr_scheduler, scheduler_cls_name)
    except:
        scheduler_cls = getattr(_custom_lr_scheduler, scheduler_cls_name)
        log.info('Created custom lr scheduler')
    if scheduler_cls_name.lower() == 'ReduceLROnPlateau'.lower():
        raise NotImplementedError('This scheduler is not fully supported yet')
    scheduler = scheduler_cls(optimizer, **scheduler_params)
    setattr(scheduler, '_scheduler_opt', scheduler_opt)
    return LRScheduler(scheduler, scheduler_params, update_scheduler_on)


SPECIAL_NAMES = ['radius', 'max_num_neighbors', 'block_names']


def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)


class ConvMockDown(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data.append(self.kwargs['down_conv_nn'])
        if self.test_precompute:
            assert kwargs['precomputed'] is not None
        return data


class InnerMock(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, data, *args, **kwargs):
        data.append('inner')
        return data


class ConvMockUp(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data = data[0].copy()
        data.append(self.kwargs['up_conv_nn'])
        if self.test_precompute:
            assert kwargs['precomputed'] is not None
        return data


class Conv2D(Seq):

    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias))
        if bn:
            self.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.append(activation)


class MLP2D(Seq):

    def __init__(self, channels, bias=False, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        for i in range(len(channels) - 1):
            self.append(Conv2D(channels[i], channels[i + 1], bn=bn, bias=bias, activation=activation))


class GlobalDenseBaseModule(torch.nn.Module):

    def __init__(self, nn, aggr='max', bn=True, activation=torch.nn.LeakyReLU(negative_slope=0.01), **kwargs):
        super(GlobalDenseBaseModule, self).__init__()
        self.nn = MLP2D(nn, bn=bn, activation=activation, bias=False)
        if aggr.lower() not in ['mean', 'max']:
            raise Exception('The aggregation provided is unrecognized {}'.format(aggr))
        self._aggr = aggr.lower()

    @property
    def nb_params(self):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

    def forward(self, data, **kwargs):
        x, pos = data.x, data.pos
        pos_flipped = pos.transpose(1, 2).contiguous()
        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))
        if self._aggr == 'max':
            x = x.squeeze(-1).max(-1)[0]
        elif self._aggr == 'mean':
            x = x.squeeze(-1).mean(-1)
        else:
            raise NotImplementedError('The following aggregation {} is not recognized'.format(self._aggr))
        pos = None
        x = x.unsqueeze(-1)
        return Data(x=x, pos=pos)

    def __repr__(self):
        return '{}: {} (aggr={}, {})'.format(self.__class__.__name__, self.nb_params, self._aggr, self.nn)


class BaseModule(nn.Module):
    """ Base module class with some basic additions to the pytorch Module class
    """

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params


class FastBatchNorm1d(BaseModule):

    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum)

    def _forward_dense(self, x):
        return self.batch_norm(x)

    def _forward_sparse(self, x):
        """ Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze()

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError('Non supported number of dimensions {}'.format(x.dim()))


def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i - 1], channels[i], bias=bias), FastBatchNorm1d(channels[i], momentum=bn_momentum), activation) for i in range(1, len(channels))])


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


class GlobalBaseModule(torch.nn.Module):

    def __init__(self, nn, aggr='max', *args, **kwargs):
        super(GlobalBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == 'max' else global_mean_pool

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=batch.device)
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseResnetBlock(torch.nn.Module):

    def __init__(self, indim, outdim, convdim):
        """
            indim: size of x at the input
            outdim: desired size of x at the output
            convdim: size of x following convolution
        """
        torch.nn.Module.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.convdim = convdim
        self.features_downsample_nn = MLP([self.indim, self.outdim // 4])
        self.features_upsample_nn = MLP([self.convdim, self.outdim])
        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])
        self.activation = ReLU()

    @property
    @abstractmethod
    def convs(self):
        pass

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x = data.x
        shortcut = x
        x = self.features_downsample_nn(x)
        data = self.convs(data)
        x = data.x
        idx = data.idx
        x = self.features_upsample_nn(x)
        if idx is not None:
            shortcut = shortcut[idx]
        shortcut = self.shortcut_feature_resize_nn(shortcut)
        x = shortcut + x
        batch_obj.x = x
        batch_obj.pos = data.pos
        batch_obj.batch = data.batch
        copy_from_to(data, batch_obj)
        return batch_obj


class GlobalPartialDenseBaseModule(torch.nn.Module):

    def __init__(self, nn, aggr='max', *args, **kwargs):
        super(GlobalPartialDenseBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == 'max' else global_mean_pool

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=x.device)
        copy_from_to(data, batch_obj)
        return batch_obj


class Identity(BaseModule):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, data):
        return data


def weight_variable(shape):
    initial = torch.empty(shape, dtype=torch.float)
    torch.nn.init.xavier_normal_(initial)
    return initial


class UnaryConv(BaseModule):

    def __init__(self, kernel_shape):
        """
        1x1 convolution on point cloud (we can even call it a mini pointnet)
        """
        super(UnaryConv, self).__init__()
        self.weight = Parameter(weight_variable(kernel_shape))

    def forward(self, features):
        """
        features(Torch Tensor): size N x d d is the size of inputs
        """
        return torch.matmul(features, self.weight)

    def __repr__(self):
        return 'UnaryConv {}'.format(self.weight.shape)


class MultiHeadClassifier(BaseModule):
    """ Allows segregated segmentation in case the category of an object is known. This is the case in ShapeNet
    for example.

        Arguments:
            in_features -- size of the input channel
            cat_to_seg {[type]} -- category to segment maps for example:
                {
                    'Airplane': [0,1,2],
                    'Table': [3,4]
                }

        Keyword Arguments:
            dropout_proba  (default: {0.5})
            bn_momentum  -- batch norm momentum (default: {0.1})
        """

    def __init__(self, in_features, cat_to_seg, dropout_proba=0.5, bn_momentum=0.1):
        super().__init__()
        self._cat_to_seg = {}
        self._num_categories = len(cat_to_seg)
        self._max_seg_count = 0
        self._max_seg = 0
        self._shifts = torch.zeros((self._num_categories,), dtype=torch.long)
        for i, seg in enumerate(cat_to_seg.values()):
            self._max_seg_count = max(self._max_seg_count, len(seg))
            self._max_seg = max(self._max_seg, max(seg))
            self._shifts[i] = min(seg)
            self._cat_to_seg[i] = seg
        self.channel_rasing = MLP([in_features, self._num_categories * in_features], bn_momentum=bn_momentum, bias=False)
        if dropout_proba:
            self.channel_rasing.add_module('Dropout', nn.Dropout(p=dropout_proba))
        self.classifier = UnaryConv((self._num_categories, in_features, self._max_seg_count))
        self._bias = Parameter(torch.zeros(self._max_seg_count))

    def forward(self, features, category_labels, **kwargs):
        assert features.dim() == 2
        self._shifts = self._shifts
        in_dim = features.shape[-1]
        features = self.channel_rasing(features)
        features = features.reshape((-1, self._num_categories, in_dim))
        features = features.transpose(0, 1)
        features = self.classifier(features) + self._bias
        ind = category_labels.unsqueeze(-1).repeat(1, 1, features.shape[-1]).long()
        logits = features.gather(0, ind).squeeze(0)
        softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        output = torch.zeros(logits.shape[0], self._max_seg + 1)
        cats_in_batch = torch.unique(category_labels)
        for cat in cats_in_batch:
            cat_mask = category_labels == cat
            seg_indices = self._cat_to_seg[cat.item()]
            probs = softmax[(cat_mask), :len(seg_indices)]
            output[(cat_mask), seg_indices[0]:seg_indices[-1] + 1] = probs
        return output


class Seq(nn.Sequential):

    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1


class Conv1D(Seq):

    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias))
        if bn:
            self.append(nn.BatchNorm1d(out_channels))
        if activation:
            self.append(activation)


class BaseLinearTransformSTNkD(torch.nn.Module):
    """STN which learns a k-dimensional linear transformation

    Arguments:
        nn (torch.nn.Module) -- module which takes feat_x as input and regresses it to a global feature used to calculate the transform
        nn_feat_size -- the size of the global feature
        k -- the size of trans_x
        batch_size -- the number of examples per batch
    """

    def __init__(self, nn, nn_feat_size, k=3, batch_size=1):
        super().__init__()
        self.nn = nn
        self.k = k
        self.batch_size = batch_size
        self.fc_layer = Linear(nn_feat_size, k * k)
        torch.nn.init.constant_(self.fc_layer.weight, 0)
        torch.nn.init.constant_(self.fc_layer.bias, 0)
        self.identity = torch.eye(k).view(1, k * k).repeat(batch_size, 1)

    def forward(self, feat_x, trans_x, batch):
        """
            Learns and applies a linear transformation to trans_x based on feat_x.
            feat_x and trans_x may be the same or different.
        """
        global_feature = self.nn(feat_x, batch)
        trans = self.fc_layer(global_feature)
        trans = trans + self.identity
        trans = trans.view(-1, self.k, self.k)
        self.trans = trans
        batch_x = trans_x.view(trans_x.shape[0], 1, trans_x.shape[1])
        x_transformed = torch.bmm(batch_x, trans[batch])
        return x_transformed.view(len(trans_x), trans_x.shape[1])

    def get_orthogonal_regularization_loss(self):
        loss = torch.mean(torch.norm(torch.bmm(self.trans, self.trans.transpose(2, 1)) - self.identity.view(-1, self.k, self.k), dim=(1, 2)))
        return loss


_MAX_NEIGHBOURS = 32


def _variance_estimator_dense(r, pos, f):
    nei_idx = tp.ball_query(r, _MAX_NEIGHBOURS, pos, pos, sort=True)[0].reshape(pos.shape[0], -1).long()
    f_neighboors = f.gather(1, nei_idx).reshape(f.shape[0], f.shape[1], -1)
    gradient = (f.unsqueeze(-1).repeat(1, 1, f_neighboors.shape[-1]) - f_neighboors) ** 2
    return gradient.sum(-1)


def _dirichlet_dense(r, pos, f, aggr):
    variances = _variance_estimator_dense(r, pos, f)
    return 1 / 2.0 * aggr(variances)


def _variance_estimator_sparse(r, pos, f, batch_idx):
    with torch.no_grad():
        assign_index = radius(pos, pos, r, batch_x=batch_idx, batch_y=batch_idx)
        y_idx, x_idx = assign_index
        grad_f = (f[x_idx] - f[y_idx]) ** 2
    y = scatter_add(grad_f, y_idx, dim=0, dim_size=pos.size(0))
    return y


def _dirichlet_sparse(r, pos, f, batch_idx, aggr):
    variances = _variance_estimator_sparse(r, pos, f, batch_idx)
    return 1 / 2.0 * aggr(variances)


def dirichlet_loss(r, pos, f, batch_idx=None, aggr=torch.mean):
    """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
    Arguments:
        r -- Radius for the beighbour search
        pos -- [N,3] (or [B,N,3] for dense format)  location of each point
        f -- [N] (or [B,N] for dense format)  Value of a function at each points
        batch_idx -- [N] Batch id of each point (Only for sparse format)
        aggr -- aggregation function for the final loss value
    """
    if batch_idx is None:
        assert f.dim() == 2 and pos.dim() == 3
        return _dirichlet_dense(r, pos, f, aggr)
    else:
        assert f.dim() == 1 and pos.dim() == 2
        return _dirichlet_sparse(r, pos, f, batch_idx, aggr)


class DirichletLoss(torch.nn.Module):
    """ L2 norm of the gradient estimated as the average change of a field value f
    accross neighbouring points within a radius r
    """

    def __init__(self, r, aggr=torch.mean):
        super().__init__()
        self._r = r
        self._aggr = aggr

    def forward(self, pos, f, batch_idx=None):
        """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
        Arguments:
            pos -- [N,3] (or [B,N,3] for dense format)  location of each point
            f -- [N] (or [B,N] for dense format)  Value of a function at each points
            batch_idx -- [N] Batch id of each point (Only for sparse format)
        """
        return dirichlet_loss(self._r, pos, f, batch_idx=batch_idx, aggr=self._aggr)


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


class HuberLoss(torch.nn.Module):

    def __init__(self, delta=0.1):
        super().__init__()
        self._delta = delta

    def forward(self, error):
        return huber_loss(error, self._delta)


class LossAnnealer(torch.nn.modules.loss._Loss):
    """
    This class will be used to perform annealing between two losses
    """

    def __init__(self, args):
        super(LossAnnealer, self).__init__()
        self._coeff = 0.5
        self.normalized_loss = True

    def forward(self, loss_1, loss_2, **kwargs):
        annealing_alpha = kwargs.get('annealing_alpha', None)
        if annealing_alpha is None:
            return self._coeff * loss_1 + (1 - self._coeff) * loss_2
        else:
            return (1 - annealing_alpha) * loss_1 + annealing_alpha * loss_2


class FocalLoss(torch.nn.modules.loss._Loss):

    def __init__(self, gamma: float=2, alphas: Any=None, size_average: bool=True, normalized: bool=True):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alphas = alphas
        self.size_average = size_average
        self.normalized = normalized

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, -1, target.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self._alphas is not None:
            at = self._alphas.gather(0, target)
            logpt = logpt * Variable(at)
        if self.normalized:
            sum_ = 1 / torch.sum((1 - pt) ** self._gamma)
        else:
            sum_ = 1
        loss = -1 * sum_ * (1 - pt) ** self._gamma * logpt
        return loss.sum()


class WrapperKLDivLoss(torch.nn.modules.loss._Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WrapperKLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, label_vec=None, segm_size=None):
        label_vec = Variable(label_vec).float() / segm_size.unsqueeze(-1).float()
        input = F.log_softmax(input, dim=-1)
        loss = torch.nn.modules.loss.KLDivLoss()(input, label_vec)
        return loss


class LossFactory(torch.nn.modules.loss._Loss):

    def __init__(self, loss, dbinfo):
        super(LossFactory, self).__init__()
        self._loss = loss
        self.special_args = {}
        self.search_for_args = []
        if self._loss == 'cross_entropy':
            self._loss_func = nn.functional.cross_entropy
            self.special_args = {'weight': dbinfo['class_weights']}
        elif self._loss == 'focal_loss':
            self._loss_func = FocalLoss(alphas=dbinfo['class_weights'])
        elif self._loss == 'KLDivLoss':
            self._loss_func = WrapperKLDivLoss()
            self.search_for_args = ['segm_size', 'label_vec']
        else:
            raise NotImplementedError

    def forward(self, input, target, **kwargs):
        added_arguments = OrderedDict()
        for key in self.search_for_args:
            added_arguments[key] = kwargs.get(key, None)
        input, target = filter_valid(input, target)
        return self._loss_func(input, target, **added_arguments, **self.special_args)


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)
    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, (d)] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-07)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


class ContrastiveHardestNegativeLoss(nn.Module):
    """
    Compute contrastive loss between positive pairs and mine negative pairs which are not in the intersection of the two point clouds (taken from https://github.com/chrischoy/FCGF)
    Let :math:`(f_i, f^{+}_i)_{i=1 \\dots N}` set of positive_pairs and :math:`(f_i, f^{-}_i)_{i=1 \\dots M}` a set of negative pairs
    The loss is computed as:
    .. math::
        L = \\frac{1}{N^2} \\sum_{i=1}^N \\sum_{j=1}^N [d^{+}_{ij} - \\lambda_+]_+ + \\frac{1}{M} \\sum_{i=1}^M [\\lambda_{-} - d^{-}_i]_+

    where:
    .. math::
        d^{+}_{ij} = ||f_{i} - f^{+}_{j}||

    and
    .. math::
        d^{-}_{i} = \\min_{j}(||f_{i} - f^{-}_{j}||)

    In this loss, we only mine the negatives
    Parameters
    ----------

    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    num_pos:
        number of positive pairs
    num_hn_samples:
        number of negative point we mine.
    """

    def __init__(self, pos_thresh, neg_thresh, num_pos=5192, num_hn_samples=2048):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.num_pos = num_pos
        self.num_hn_samples = num_hn_samples

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs, thresh=None):
        """
        Generate negative pairs
        """
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, self.num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, self.num_hn_samples), replace=False)
        if N_pos_pairs > self.num_pos:
            pos_sel = np.random.choice(N_pos_pairs, self.num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs
        subF0, subF1 = F0[sel0], F1[sel1]
        pos_ind0 = sample_pos_pairs[:, (0)].long()
        pos_ind1 = sample_pos_pairs[:, (1)].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')
        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        pos_keys = _hash(positive_pairs, hash_seed)
        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)
        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def forward(self, F0, F1, matches, xyz0=None, xyz1=None):
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(F0, F1, matches.detach().cpu())
        return pos_loss + neg_loss


class BatchHardContrastiveLoss(nn.Module):
    """
        apply contrastive loss but mine the negative sample in the batch.
    apply a mask if the distance between negative pair is too close.
    Parameters
    ----------
    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    min_dist:
        minimum distance to be in the negative sample
    """

    def __init__(self, pos_thresh, neg_thresh, min_dist=0.15):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.min_dist = min_dist

    def forward(self, F0, F1, positive_pairs, xyz0=None, xyz1=None):
        posF0 = F0[positive_pairs[:, (0)]]
        posF1 = F1[positive_pairs[:, (1)]]
        subxyz0 = xyz0[positive_pairs[:, (0)]]
        false_negative = pdist(subxyz0, subxyz0, dist_type='L2') > self.min_dist
        furthest_pos, _ = (posF0 - posF1).pow(2).max(1)
        neg_loss = F.relu(self.neg_thresh - (posF0[0] - posF1[false_negative[0]]).pow(2).sum(1).min()).pow(2) / len(posF0)
        for i in range(1, len(posF0)):
            neg_loss += F.relu(self.neg_thresh - (posF0[i] - posF1[false_negative[i]]).pow(2).sum(1).min()).pow(2) / len(posF0)
        pos_loss = F.relu(furthest_pos - self.pos_thresh).pow(2)
        return pos_loss.mean() + neg_loss.mean()


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """

    def get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def __init__(self, args_up=None, args_down=None, args_innermost=None, modules_lib=None, submodule=None, outermost=False, innermost=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            args_up -- arguments for up convs
            args_down -- arguments for down convs
            args_innermost -- arguments for innermost
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if innermost:
            assert outermost == False
            module_name = self.get_from_kwargs(args_innermost, 'module_name')
            inner_module_cls = getattr(modules_lib, module_name)
            self.inner = inner_module_cls(**args_innermost)
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            self.up = upconv_cls(**args_up)
        else:
            downconv_cls = self.get_from_kwargs(args_down, 'down_conv_cls')
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            downconv = downconv_cls(**args_down)
            upconv = upconv_cls(**args_up)
            self.down = downconv
            self.submodule = submodule
            self.up = upconv

    def forward(self, data, **kwargs):
        if self.innermost:
            data_out = self.inner(data, **kwargs)
            data = data_out, data
            return self.up(data, **kwargs)
        else:
            data_out = self.down(data, **kwargs)
            data_out2 = self.submodule(data_out, **kwargs)
            data = data_out2, data
            return self.up(data, **kwargs)


def concatenate_pair_ind(list_data_source, list_data_target):
    """
    for a list of pair of indices batched, change the index it refers to wrt the batch index
    Parameters
    ----------
    list_data_source: list[Data]
    list_data_target: list[Data]
    Returns
    -------
    torch.Tensor
        indices of y corrected wrt batch indices


    """
    assert len(list_data_source) == len(list_data_target)
    assert hasattr(list_data_source[0], 'pair_ind')
    list_pair_ind = []
    cum_size = torch.zeros(2)
    for i in range(len(list_data_source)):
        size = torch.tensor([len(list_data_source[i].pos), len(list_data_target[i].pos)])
        list_pair_ind.append(list_data_source[i].pair_ind + cum_size)
        cum_size = cum_size + size
    return torch.cat(list_pair_ind, 0)


class MiniPointNet(torch.nn.Module):

    def __init__(self, local_nn, global_nn, aggr='max', return_local_out=False):
        super().__init__()
        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn) if global_nn else None
        self.g_pool = global_max_pool if aggr == 'max' else global_mean_pool
        self.return_local_out = return_local_out

    def forward(self, x, batch):
        y = x = self.local_nn(x)
        x = self.g_pool(x, batch)
        if self.global_nn:
            x = self.global_nn(x)
        if self.return_local_out:
            return x, y
        return x

    def forward_embedding(self, pos, batch):
        global_feat, local_feat = self.forward(pos, batch)
        indices = batch.unsqueeze(-1).repeat((1, global_feat.shape[-1]))
        gathered_global_feat = torch.gather(global_feat, 0, indices)
        x = torch.cat([local_feat, gathered_global_feat], -1)
        return x


class PointNetSTN3D(BaseLinearTransformSTNkD):

    def __init__(self, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], 3, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)


class PointNetSTNkD(BaseLinearTransformSTNkD, BaseInternalLossModule):

    def __init__(self, k=64, local_nn=[64, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], k, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)

    def get_internal_losses(self):
        return {'orthogonal_regularization_loss': self.get_orthogonal_regularization_loss()}


class PointNetSeg(torch.nn.Module):

    def __init__(self, input_stn_local_nn=[3, 64, 128, 1024], input_stn_global_nn=[1024, 512, 256], local_nn_1=[3, 64, 64], feat_stn_k=64, feat_stn_local_nn=[64, 64, 128, 1024], feat_stn_global_nn=[1024, 512, 256], local_nn_2=[64, 64, 128, 1024], seg_nn=[1088, 512, 256, 128, 4], batch_size=1, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.input_stn = PointNetSTN3D(input_stn_local_nn, input_stn_global_nn, batch_size)
        self.local_nn_1 = MLP(local_nn_1)
        self.feat_stn = PointNetSTNkD(feat_stn_k, feat_stn_local_nn, feat_stn_global_nn, batch_size)
        self.local_nn_2 = MLP(local_nn_2)
        self.seg_nn = MLP(seg_nn)
        self._use_scatter_pooling = True

    def set_scatter_pooling(self, use_scatter_pooling):
        self._use_scatter_pooling = use_scatter_pooling

    def func_global_max_pooling(self, x3, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x3, batch)
        else:
            global_feature = x3.max(1)
            return global_feature[0]

    def forward(self, x, batch):
        x = self.input_stn(x, batch)
        x = self.local_nn_1(x)
        x_feat_trans = self.feat_stn(x, batch)
        x3 = self.local_nn_2(x_feat_trans)
        global_feature = self.func_global_max_pooling(x3, batch)
        feat_concat = torch.cat([x_feat_trans, global_feature[batch]], dim=1)
        out = self.seg_nn(feat_concat)
        return out


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """
    KEY = 'origin_id'

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data
        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data


_INTEGER_LABEL_KEYS = ['y', 'instance_labels']


def group_data(data, cluster=None, unique_pos_indices=None, mode='last', skip_keys=[]):
    """ Group data based on indices in cluster.
    The option ``mode`` controls how data gets agregated within each cluster.

    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    """
    assert mode in ['mean', 'last']
    if mode == 'mean' and cluster is None:
        raise ValueError('In mean mode the cluster argument needs to be specified')
    if mode == 'last' and unique_pos_indices is None:
        raise ValueError('In last mode the unique_pos_indices argument needs to be specified')
    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search('edge', key)):
            raise ValueError('Edges not supported. Wrong data type.')
        if key in skip_keys:
            continue
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == 'last' or key == 'batch' or key == SaveOriginalPosId.KEY:
                data[key] = item[unique_pos_indices]
            elif mode == 'mean':
                if key in _INTEGER_LABEL_KEYS:
                    item_min = item.min()
                    item = F.one_hot(item - item_min)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1) + item_min
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
    return data


def shuffle_data(data):
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if torch.is_tensor(item) and num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


class GridSampling3D:
    """ Clusters points into voxels with size :attr:`size`.
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
            If True, it will convert the points into their associated sparse coordinates within the grid.     mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a cell will be averaged
        If mode is `last`, one random points per cell will be selected with its associated features
    """

    def __init__(self, size, quantize_coords=False, mode='mean', verbose=False):
        self._grid_size = size
        self._quantize_coords = quantize_coords
        self._mode = mode
        if verbose:
            log.warning('If you need to keep track of the position of your points, use SaveOriginalPosId transform before using GridSampling3D')
            if self._mode == 'last':
                log.warning("The tensors within data will be shuffled each time this transform is applied. Be careful that if an attribute doesn't have the size of num_points, it won't be shuffled")

    def _process(self, data):
        if self._mode == 'last':
            data = shuffle_data(data)
        coords = (data.pos / self._grid_size).int()
        if 'batch' not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)
        skip_keys = []
        if self._quantize_coords:
            skip_keys.append('pos')
        data = group_data(data, cluster, unique_pos_indices, mode=self._mode, skip_keys=skip_keys)
        if self._quantize_coords:
            data.pos = coords[unique_pos_indices]
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return '{}(grid_size={}, quantize_coords={}, mode={})'.format(self.__class__.__name__, self._grid_size, self._quantize_coords, self._mode)


def radius_gaussian(sq_r, sig, eps=1e-09):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def KPConv_deform_ops(query_points, support_points, neighbors_indices, features, K_points, offsets, modulations, K_values, KP_extent, KP_influence, aggregation_mode):
    """
    This function creates a graph of operations to define Deformable Kernel Point Convolution in tensorflow. See
    KPConv_deformable function above for a description of each parameter
    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n0_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param offsets:             [n_points, n_kpoints, dim]
    :param modulations:         [n_points, n_kpoints] or None
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string in ('closest', 'sum') - whether to sum influences, or only keep the closest

    :return features, square_distances, deformed_K_points
    """
    n_kp = int(K_points.shape[0])
    shadow_ind = support_points.shape[0]
    shadow_point = torch.ones_like(support_points[:1, :]) * 1000000.0
    support_points = torch.cat([support_points, shadow_point], axis=0)
    neighbors = support_points[neighbors_indices]
    neighbors = neighbors - query_points.unsqueeze(1)
    deformed_K_points = torch.add(offsets, K_points)
    neighbors = neighbors.unsqueeze(2)
    neighbors = neighbors.repeat([1, 1, n_kp, 1])
    differences = neighbors - deformed_K_points.unsqueeze(1)
    sq_distances = torch.sum(differences ** 2, axis=3)
    in_range = (sq_distances < KP_extent ** 2).any(2)
    new_max_neighb = torch.max(torch.sum(in_range, axis=1))
    new_neighb_bool, new_neighb_inds = torch.topk(in_range, k=new_max_neighb)
    new_neighbors_indices = neighbors_indices.gather(1, new_neighb_inds)
    new_neighb_inds_sq = new_neighb_inds.unsqueeze(-1)
    new_sq_distances = sq_distances.gather(1, new_neighb_inds_sq.repeat((1, 1, sq_distances.shape[-1])))
    new_neighbors_indices *= new_neighb_bool
    new_neighbors_indices += (1 - new_neighb_bool) * shadow_ind
    if KP_influence == 'constant':
        all_weights = new_sq_distances < KP_extent ** 2
        all_weights = all_weights.permute(0, 2, 1)
    elif KP_influence == 'linear':
        all_weights = torch.relu(1 - torch.sqrt(new_sq_distances) / KP_extent)
        all_weights = all_weights.permute(0, 2, 1)
    elif KP_influence == 'gaussian':
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(new_sq_distances, sigma)
        all_weights = all_weights.permute(0, 2, 1)
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')
    if aggregation_mode == 'closest':
        neighbors_1nn = torch.argmin(new_sq_distances, axis=2, output_type=torch.long)
        all_weights *= torch.zeros_like(all_weights, dtype=torch.float32).scatter_(1, neighbors_1nn, 1)
    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")
    features = torch.cat([features, torch.zeros_like(features[:1, :])], axis=0)
    neighborhood_features = features[new_neighbors_indices]
    weighted_features = torch.matmul(all_weights, neighborhood_features)
    if modulations is not None:
        weighted_features *= modulations.unsqueeze(2)
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)
    output_features = torch.sum(kernel_outputs, axis=0)
    return output_features, sq_distances, deformed_K_points


def gather(x, idx, method=2):
    """
    https://github.com/pytorch/pytorch/issues/15245
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    idx[idx == -1] = x.shape[0] - 1
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def KPConv_ops(query_points, support_points, neighbors_indices, features, K_points, K_values, KP_extent, KP_influence, aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """
    int(K_points.shape[0])
    shadow_point = torch.ones_like(support_points[:1, :]) * 1000000.0
    support_points = torch.cat([support_points, shadow_point], dim=0)
    neighbors = gather(support_points, neighbors_indices)
    neighbors = neighbors - query_points.unsqueeze(1)
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points
    sq_distances = torch.sum(differences ** 2, dim=3)
    if KP_influence == 'constant':
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(2, 1)
    elif KP_influence == 'linear':
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent, min=0.0)
        all_weights = all_weights.transpose(2, 1)
    elif KP_influence == 'gaussian':
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')
    if aggregation_mode == 'closest':
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(neighbors_1nn, K_points.shape[0]), 1, 2)
    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")
    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)
    neighborhood_features = gather(features, neighbors_indices)
    weighted_features = torch.matmul(all_weights, neighborhood_features)
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)
    output_features = torch.sum(kernel_outputs, dim=0)
    return output_features


def add_ones(query_points, x, add_one):
    if add_one:
        ones = torch.ones(query_points.shape[0], dtype=torch.float).unsqueeze(-1)
        if x is not None:
            x = torch.cat([ones, x], dim=-1)
        else:
            x = ones
    return x


def fitting_loss(sq_distance, radius):
    """ KPConv fitting loss. For each query point it ensures that at least one neighboor is
    close to each kernel point

    Arguments:
        sq_distance - For each querry point, from all neighboors to all KP points [N_querry, N_neighboors, N_KPoints]
        radius - Radius of the convolution
    """
    kpmin = sq_distance.min(dim=1)[0]
    normalised_kpmin = kpmin / radius ** 2
    return torch.mean(normalised_kpmin)


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', ratio=1.0, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    radius0 = 1
    diameter0 = 2
    moving_factor = 0.01
    continuous_moving_decay = 0.9995
    thresh = 1e-05
    clip = 0.05 * radius0
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[(d2 < 0.5 * radius0 * radius0), :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))
    if fixed == 'center':
        kernel_points[:, (0), :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, (1), (-1)] += 2 * radius0 / 3
        kernel_points[:, (2), (-1)] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-06)
        inter_grads = np.sum(inter_grads, axis=1)
        circle_grads = 10 * kernel_points
        gradients = inter_grads + circle_grads
        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[(iter), :] = np.max(gradients_norms, axis=1)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)
        if fixed == 'center':
            moving_dists[:, (0)] = 0
        if fixed == 'verticals':
            moving_dists[:, (0)] = 0
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-06, -1)
        if verbose:
            log.info('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[(0), :, (0)], kernel_points[(0), :, (1)], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            log.info(moving_factor)
        moving_factor *= continuous_moving_decay
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])
    return kernel_points * radius, saved_gradient_norms


def makedirs(path):
    """
    taken from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/makedirs.py
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


ply_dtypes = dict([(b'int8', 'i1'), (b'char', 'i1'), (b'uint8', 'u1'), (b'uchar', 'u1'), (b'int16', 'i2'), (b'short', 'i2'), (b'uint16', 'u2'), (b'ushort', 'u2'), (b'int32', 'i4'), (b'int', 'i4'), (b'uint32', 'u4'), (b'uint', 'u4'), (b'float32', 'f4'), (b'float', 'f4'), (b'float64', 'f8'), (b'double', 'f8')])


def parse_header(plyfile, ext):
    line = []
    properties = []
    num_points = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element' in line:
            line = line.split()
            num_points = int(line[2])
        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
    return num_points, properties


def parse_mesh_header(plyfile, ext):
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])
        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])
        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)
    return num_points, num_faces, vertex_properties


valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """
    with open(filename, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')
        fmt = plyfile.readline().split()[1].decode()
        if fmt == 'ascii':
            raise ValueError('The file is not binary')
        ext = valid_formats[fmt]
        if triangular_mesh:
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
            face_properties = [('k', ext + 'u1'), ('v1', ext + 'i4'), ('v2', ext + 'i4'), ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]
        else:
            num_points, properties = parse_header(plyfile, ext)
            data = np.fromfile(plyfile, dtype=properties, count=num_points)
    return data


def header_properties(field_list, field_names):
    lines = []
    lines.append('element vertex %d' % field_list[0].shape[0])
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1
    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """
    field_list = list(field_list) if type(field_list) == list or type(field_list) == tuple else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            log.info('fields have more than 2 dimensions')
            return False
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        log.info('wrong field dimensions')
        return False
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        log.info('wrong number of field names')
        return False
    if not filename.endswith('.ply'):
        filename += '.ply'
    with open(filename, 'w') as plyfile:
        header = ['ply']
        header.append('format binary_' + sys.byteorder + '_endian 1.0')
        header.extend(header_properties(field_list, field_names))
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')
        header.append('end_header')
        for line in header:
            plyfile.write('%s\n' % line)
    with open(filename, 'ab') as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1
        data.tofile(plyfile)
        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, (0)]
            data['1'] = triangular_faces[:, (1)]
            data['2'] = triangular_faces[:, (2)]
            data.tofile(plyfile)
    return True


def load_kernels(radius, num_kpoints, num_kernels, dimension, fixed):
    num_tries = 100
    kernel_dir = join(DIR, 'kernels/dispositions')
    if not exists(kernel_dir):
        makedirs(kernel_dir)
    if dimension == 3:
        kernel_file = join(kernel_dir, 'k_{:03d}_{:s}.ply'.format(num_kpoints, fixed))
    elif dimension == 2:
        kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_2D.ply'.format(num_kpoints, fixed))
    else:
        raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))
    if not exists(kernel_file):
        kernel_points, grad_norms = kernel_point_optimization_debug(1.0, num_kpoints, num_kernels=num_tries, dimension=dimension, fixed=fixed, verbose=0)
        best_k = np.argmin(grad_norms[(-1), :])
        original_kernel = kernel_points[(best_k), :, :]
        write_ply(kernel_file, original_kernel, ['x', 'y', 'z'])
    else:
        data = read_ply(kernel_file)
        original_kernel = np.vstack((data['x'], data['y'], data['z'])).T
    if dimension == 2:
        return original_kernel
    if fixed == 'verticals':
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, (0), (0)] = c
        R[:, (1), (1)] = c
        R[:, (2), (2)] = 1
        R[:, (0), (1)] = s
        R[:, (1), (0)] = -s
        original_kernel = radius * np.expand_dims(original_kernel, 0)
        kernels = np.matmul(original_kernel, R)
    else:
        u = np.ones((num_kernels, 3))
        v = np.ones((num_kernels, 3))
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        while np.any(wrongs):
            new_u = np.random.rand(num_kernels, 3) * 2 - 1
            new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-09, -1)
            u[(wrongs), :] = new_u[(wrongs), :]
            new_v = np.random.rand(num_kernels, 3) * 2 - 1
            new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-09, -1)
            v[(wrongs), :] = new_v[(wrongs), :]
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
        v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-09, -1)
        w = np.cross(u, v)
        R = np.stack((u, v, w), axis=-1)
        original_kernel = radius * np.expand_dims(original_kernel, 0)
        kernels = np.matmul(original_kernel, R)
        kernels = kernels
        kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)
    return kernels


def permissive_loss(deformed_kpoints, radius):
    """This loss is responsible to penalize deformed_kpoints to
    move outside from the radius defined for the convolution
    """
    norm_deformed_normalized = torch.norm(deformed_kpoints, p=2, dim=-1) / float(radius)
    permissive_loss = torch.mean(norm_deformed_normalized[norm_deformed_normalized > 1.0])
    return permissive_loss


def repulsion_loss(deformed_kpoints, radius):
    """ Ensures that the deformed points within the kernel remain equidistant

    Arguments:
        deformed_kpoints - deformed points for each query point
        radius - Radius of the kernel
    """
    deformed_kpoints / float(radius)
    n_points = deformed_kpoints.shape[1]
    repulsive_loss = 0
    for i in range(n_points):
        with torch.no_grad():
            other_points = torch.cat([deformed_kpoints[:, :i, :], deformed_kpoints[:, i + 1:, :]], dim=1)
        distances = torch.sqrt(torch.sum((other_points - deformed_kpoints[:, i:i + 1, :]) ** 2, dim=-1))
        repulsion_force = torch.sum(torch.pow(torch.relu(1.5 - distances), 2), dim=1)
        repulsive_loss += torch.mean(repulsion_force)
    return repulsive_loss


class KPConvDeformableLayer(BaseInternalLossModule):
    """
    apply the deformable kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    modulated = False :   If deformable conv should be modulated
    """
    PERMISSIVE_LOSS_KEY = 'permissive_loss'
    FITTING_LOSS_KEY = 'fitting_loss'
    REPULSION_LOSS_KEY = 'repulsion_loss'
    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(self, num_inputs, num_outputs, point_influence, n_kernel_points=15, fixed='center', KP_influence='linear', aggregation_mode='sum', dimension=3, modulated=False, loss_mode='fitting', add_one=False):
        super(KPConvDeformableLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs
        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated
        self.internal_losses = {self.PERMISSIVE_LOSS_KEY: 0.0, self.FITTING_LOSS_KEY: 0.0, self.REPULSION_LOSS_KEY: 0.0}
        self.loss_mode = loss_mode
        K_points_numpy = load_kernels(self.kernel_radius, n_kernel_points, num_kernels=1, dimension=dimension, fixed=fixed)
        self.K_points = Parameter(torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))), requires_grad=False)
        if modulated:
            offset_dim = (dimension + 1) * self.n_kernel_points
        else:
            offset_dim = dimension * self.n_kernel_points
        offset_weights = torch.empty([n_kernel_points, self.num_inputs, offset_dim], dtype=torch.float)
        torch.nn.init.xavier_normal_(offset_weights)
        self.offset_weights = Parameter(offset_weights)
        self.offset_bias = Parameter(torch.zeros(offset_dim, dtype=torch.float))
        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        x = add_ones(support_points, x, self.add_one)
        offset_feat = KPConv_ops(query_points, support_points, neighbors, x, self.K_points, self.offset_weights, self.point_influence, self.KP_influence, self.aggregation_mode) + self.offset_bias
        points_dim = query_points.shape[-1]
        if self.modulated:
            offsets = offset_feat[:, :points_dim * self.n_kernel_points]
            offsets = offsets.reshape((-1, self.n_kernel_points, points_dim))
            modulations = 2 * torch.nn.functional.sigmoid(offset_feat[:, points_dim * self.n_kernel_points:])
        else:
            offsets = offset_feat.reshape((-1, self.n_kernel_points, points_dim))
            modulations = None
        offsets *= self.point_influence
        new_feat, sq_distances, K_points_deformed = KPConv_deform_ops(query_points, support_points, neighbors, x, self.K_points, offsets, modulations, self.weight, self.point_influence, self.KP_influence, self.aggregation_mode)
        if self.loss_mode == 'fitting':
            self.internal_losses[self.FITTING_LOSS_KEY] = fitting_loss(sq_distances, self.kernel_radius)
            self.internal_losses[self.REPULSION_LOSS_KEY] = repulsion_loss(K_points_deformed, self.point_influence)
        elif self.loss_mode == 'permissive':
            self.internal_losses[self.PERMISSIVE_LOSS_KEY] = permissive_loss(K_points_deformed, self.kernel_radius)
        else:
            raise NotImplementedError('Loss mode %s not recognised. Only permissive and fitting are valid' % self.loss_mode)
        return new_feat

    def get_internal_losses(self):
        return self.internal_losses

    def __repr__(self):
        return 'KPConvDeformableLayer(InF: %i, OutF: %i, kernel_pts: %i, radius: %.2f, KP_influence: %s)' % (self.num_inputs, self.num_outputs, self.n_kernel_points, self.kernel_radius, self.KP_influence)


class KPConvLayer(torch.nn.Module):
    """
    apply the kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    """
    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(self, num_inputs, num_outputs, point_influence, n_kernel_points=15, fixed='center', KP_influence='linear', aggregation_mode='sum', dimension=3, add_one=False):
        super(KPConvLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs
        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode
        K_points_numpy = load_kernels(self.kernel_radius, n_kernel_points, num_kernels=1, dimension=dimension, fixed=fixed)
        self.K_points = Parameter(torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))), requires_grad=False)
        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        x = add_ones(support_points, x, self.add_one)
        new_feat = KPConv_ops(query_points, support_points, neighbors, x, self.K_points, self.weight, self.point_influence, self.KP_influence, self.aggregation_mode)
        return new_feat

    def __repr__(self):
        return 'KPConvLayer(InF: %i, OutF: %i, kernel_pts: %i, radius: %.2f, KP_influence: %s, Add_one: %s)' % (self.num_inputs, self.num_outputs, self.n_kernel_points, self.kernel_radius, self.KP_influence, self.add_one)


class BaseNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x, batch_y):
        return self.find_neighbours(x, y, batch_x, batch_y)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x, batch_y):
        pass

    def __repr__(self):
        return str(self.__class__.__name__) + ' ' + str(self.__dict__)


class RadiusNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, radius: float, max_num_neighbors: int=64, conv_type=ConvolutionFormat.MESSAGE_PASSING.value):
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._conv_type = conv_type.lower()

    def find_neighbours(self, x, y, batch_x=None, batch_y=None):
        if self._conv_type == ConvolutionFormat.MESSAGE_PASSING.value:
            return radius(x, y, self._radius, batch_x, batch_y, max_num_neighbors=self._max_num_neighbors)
        elif self._conv_type == ConvolutionFormat.DENSE.value or ConvolutionFormat.PARTIAL_DENSE.value:
            return tp.ball_query(self._radius, self._max_num_neighbors, x, y, mode=self._conv_type, batch_x=batch_x, batch_y=batch_y)[0]
        else:
            raise NotImplementedError


class SimpleBlock(BaseModule):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """
    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    DEFORMABLE_DENSITY = 5.0
    RIGID_DENSITY = 2.5

    def __init__(self, down_conv_nn=None, grid_size=None, prev_grid_size=None, sigma=1.0, max_num_neighbors=16, activation=torch.nn.LeakyReLU(negative_slope=0.1), bn_momentum=0.02, bn=FastBatchNorm1d, deformable=False, add_one=False, **kwargs):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        if deformable:
            density_parameter = self.DEFORMABLE_DENSITY
            self.kp_conv = KPConvDeformableLayer(num_inputs, num_outputs, point_influence=prev_grid_size * sigma, add_one=add_one)
        else:
            density_parameter = self.RIGID_DENSITY
            self.kp_conv = KPConvLayer(num_inputs, num_outputs, point_influence=prev_grid_size * sigma, add_one=add_one)
        search_radius = density_parameter * sigma * prev_grid_size
        self.neighbour_finder = RadiusNeighbourFinder(search_radius, max_num_neighbors, conv_type=self.CONV_TYPE)
        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation
        is_strided = prev_grid_size != grid_size
        if is_strided:
            self.sampler = GridSampling3D(grid_size)
        else:
            self.sampler = None

    def forward(self, data, precomputed=None, **kwargs):
        if not hasattr(data, 'block_idx'):
            setattr(data, 'block_idx', 0)
        if precomputed:
            query_data = precomputed[data.block_idx]
        elif self.sampler:
            query_data = self.sampler(data.clone())
        else:
            query_data = data.clone()
        if precomputed:
            idx_neighboors = query_data.idx_neighboors
            q_pos = query_data.pos
        else:
            q_pos, q_batch = query_data.pos, query_data.batch
            idx_neighboors = self.neighbour_finder(data.pos, q_pos, batch_x=data.batch, batch_y=q_batch)
            query_data.idx_neighboors = idx_neighboors
        x = self.kp_conv(q_pos, data.pos, idx_neighboors, data.x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        query_data.x = x
        query_data.block_idx = data.block_idx + 1
        return query_data

    def extra_repr(self):
        return 'Nb parameters: {}; {}; {}'.format(self.nb_params, self.sampler, self.neighbour_finder)


class ResnetBBlock(BaseModule):
    """ Resnet block with optional bottleneck activated by default
    Arguments:
        down_conv_nn (len of 2 or 3) :
                        sizes of input, intermediate, output.
                        If length == 2 then intermediate =  num_outputs // 4
        radius : radius of the conv kernel
        sigma :
        density_parameter : density parameter for the kernel
        max_num_neighbors : maximum number of neighboors for the neighboor search
        activation : activation function
        has_bottleneck: wether to use the bottleneck or not
        bn_momentum
        bn : batch norm (can be None -> no batch norm)
        grid_size : size of the grid,
        prev_grid_size : size of the grid at previous step.
                        In case of a strided block, this is different than grid_size
    """
    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value

    def __init__(self, down_conv_nn=None, grid_size=None, prev_grid_size=None, sigma=1, max_num_neighbors=16, activation=torch.nn.LeakyReLU(negative_slope=0.1), has_bottleneck=True, bn_momentum=0.02, bn=FastBatchNorm1d, deformable=False, add_one=False, **kwargs):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2 or len(down_conv_nn) == 3, 'down_conv_nn should be of size 2 or 3'
        if len(down_conv_nn) == 2:
            num_inputs, num_outputs = down_conv_nn
            d_2 = num_outputs // 4
        else:
            num_inputs, d_2, num_outputs = down_conv_nn
        self.is_strided = prev_grid_size != grid_size
        self.has_bottleneck = has_bottleneck
        if self.has_bottleneck:
            kp_size = [d_2, d_2]
        else:
            kp_size = [num_inputs, num_outputs]
        self.kp_conv = SimpleBlock(down_conv_nn=kp_size, grid_size=grid_size, prev_grid_size=prev_grid_size, sigma=sigma, max_num_neighbors=max_num_neighbors, activation=activation, bn_momentum=bn_momentum, bn=bn, deformable=deformable, add_one=add_one)
        if self.has_bottleneck:
            if bn:
                self.unary_1 = torch.nn.Sequential(Lin(num_inputs, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation)
                self.unary_2 = torch.nn.Sequential(Lin(d_2, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum), activation)
            else:
                self.unary_1 = torch.nn.Sequential(Lin(num_inputs, d_2, bias=False), activation)
                self.unary_2 = torch.nn.Sequential(Lin(d_2, num_outputs, bias=False), activation)
        if num_inputs != num_outputs:
            if bn:
                self.shortcut_op = torch.nn.Sequential(Lin(num_inputs, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum))
            else:
                self.shortcut_op = Lin(num_inputs, num_outputs, bias=False)
        else:
            self.shortcut_op = torch.nn.Identity()
        self.activation = activation

    def forward(self, data, precomputed=None, **kwargs):
        """
            data: x, pos, batch_idx and idx_neighbour when the neighboors of each point in pos have already been computed
        """
        output = data.clone()
        shortcut_x = data.x
        if self.has_bottleneck:
            output.x = self.unary_1(output.x)
        output = self.kp_conv(output, precomputed=precomputed)
        if self.has_bottleneck:
            output.x = self.unary_2(output.x)
        if self.is_strided:
            idx_neighboors = output.idx_neighboors
            shortcut_x = torch.cat([shortcut_x, torch.zeros_like(shortcut_x[:1, :])], axis=0)
            neighborhood_features = shortcut_x[idx_neighboors]
            shortcut_x = torch.max(neighborhood_features, dim=1, keepdim=False)[0]
        shortcut = self.shortcut_op(shortcut_x)
        output.x += shortcut
        return output

    @property
    def sampler(self):
        return self.kp_conv.sampler

    @property
    def neighbour_finder(self):
        return self.kp_conv.neighbour_finder

    def extra_repr(self):
        return 'Nb parameters: %i' % self.nb_params


class KPDualBlock(BaseModule):
    """ Dual KPConv block (usually strided + non strided)

    Arguments: Accepted kwargs
        block_names: Name of the blocks to be used as part of this dual block
        down_conv_nn: Size of the convs e.g. [64,128],
        grid_size: Size of the grid for each block,
        prev_grid_size: Size of the grid in the previous KPConv
        has_bottleneck: Wether a block should implement the bottleneck
        max_num_neighbors: Max number of neighboors for the radius search,
        deformable: Is deformable,
        add_one: Add one as a feature,
    """

    def __init__(self, block_names=None, down_conv_nn=None, grid_size=None, prev_grid_size=None, has_bottleneck=None, max_num_neighbors=None, deformable=False, add_one=False, **kwargs):
        super(KPDualBlock, self).__init__()
        assert len(block_names) == len(down_conv_nn)
        self.blocks = torch.nn.ModuleList()
        for i, class_name in enumerate(block_names):
            block_kwargs = {}
            for key, arg in kwargs.items():
                block_kwargs[key] = arg[i] if is_list(arg) else arg
            kpcls = getattr(sys.modules[__name__], class_name)
            block = kpcls(down_conv_nn=down_conv_nn[i], grid_size=grid_size[i], prev_grid_size=prev_grid_size[i], has_bottleneck=has_bottleneck[i], max_num_neighbors=max_num_neighbors[i], deformable=deformable[i] if is_list(deformable) else deformable, add_one=add_one[i] if is_list(add_one) else add_one, **block_kwargs)
            self.blocks.append(block)

    def forward(self, data, precomputed=None, **kwargs):
        for block in self.blocks:
            data = block(data, precomputed=precomputed)
        return data

    @property
    def sampler(self):
        return [b.sampler for b in self.blocks]

    @property
    def neighbour_finder(self):
        return [b.neighbour_finder for b in self.blocks]

    def extra_repr(self):
        return 'Nb parameters: %i' % self.nb_params


class ConvType(Enum):
    """
  Define the kernel region type
  """
    HYPERCUBE = 0, 'HYPERCUBE'
    SPATIAL_HYPERCUBE = 1, 'SPATIAL_HYPERCUBE'
    SPATIO_TEMPORAL_HYPERCUBE = 2, 'SPATIO_TEMPORAL_HYPERCUBE'
    HYPERCROSS = 3, 'HYPERCROSS'
    SPATIAL_HYPERCROSS = 4, 'SPATIAL_HYPERCROSS'
    SPATIO_TEMPORAL_HYPERCROSS = 5, 'SPATIO_TEMPORAL_HYPERCROSS'
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, 'SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS '

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), 'conv_type must be of ConvType'
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        axis_types = [ME.RegionType.HYPERCUBE] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPERCROSS)
    return region_type, axis_types, kernel_size


def conv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)
    return ME.MinkowskiConvolution(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, kernel_generator=kernel_generator, dimension=D)


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(ME.MinkowskiInstanceNorm(n_channels), ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f'Norm type: {norm_type} not supported')


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, bn_momentum=0.1, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, bn_momentum=0.1, D=3):
        super(BottleneckBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16, D=-1):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(ME.MinkowskiLinear(channel, channel // reduction), ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(channel // reduction, channel), ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication(dimension=D)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, reduction=16, D=-1):
        super(SEBasicBlock, self).__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D)
        self.se = SELayer(planes, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SEBasicBlockBN(SEBasicBlock):
    NORM_TYPE = NormType.BATCH_NORM


class SEBasicBlockIN(SEBasicBlock):
    NORM_TYPE = NormType.INSTANCE_NORM


class SEBasicBlockIBN(SEBasicBlock):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class SEBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, D=3, reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D)
        self.se = SELayer(planes * self.expansion, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SEBottleneckBN(SEBottleneck):
    NORM_TYPE = NormType.BATCH_NORM


class SEBottleneckIN(SEBottleneck):
    NORM_TYPE = NormType.INSTANCE_NORM


class SEBottleneckIBN(SEBottleneck):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)
    return ME.MinkowskiSumPooling(kernel_size=kernel_size, stride=stride, dilation=dilation, kernel_generator=kernel_generator, dimension=D)


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class BottleneckIN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BottleneckINBN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


def conv_tr(in_planes, out_planes, kernel_size, upsample_stride=1, dilation=1, bias=False, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(kernel_size, upsample_stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)
    return ME.MinkowskiConvolutionTranspose(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=upsample_stride, dilation=dilation, has_bias=bias, kernel_generator=kernel_generator, dimension=D)


class XConv(torch.nn.Module):
    """The convolutional operator on :math:`\\mathcal{X}`-transformed points
    from the `"PointCNN: Convolution On X-Transformed Points"
    <https://arxiv.org/abs/1801.07791>`_ paper
    .. math::
        \\mathbf{x}^{\\prime}_i = \\mathrm{Conv}\\left(\\mathbf{K},
        \\gamma_{\\mathbf{\\Theta}}(\\mathbf{P}_i - \\mathbf{p}_i) \\times
        \\left( h_\\mathbf{\\Theta}(\\mathbf{P}_i - \\mathbf{p}_i) \\, \\Vert \\,
        \\mathbf{x}_i \\right) \\right),
    where :math:`\\mathbf{K}` and :math:`\\mathbf{P}_i` denote the trainable
    filter and neighboring point positions of :math:`\\mathbf{x}_i`,
    respectively.
    :math:`\\gamma_{\\mathbf{\\Theta}}` and :math:`h_{\\mathbf{\\Theta}}` describe
    neural networks, *i.e.* MLPs, where :math:`h_{\\mathbf{\\Theta}}`
    individually lifts each point into a higher-dimensional space, and
    :math:`\\gamma_{\\mathbf{\\Theta}}` computes the :math:`\\mathcal{X}`-
    transformation matrix based on *all* points in a neighborhood.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Point cloud dimensionality.
        kernel_size (int): Size of the convolving kernel, *i.e.* number of
            neighbors including self-loops.
        hidden_channels (int, optional): Output size of
            :math:`h_{\\mathbf{\\Theta}}`, *i.e.* dimensionality of lifted
            points. If set to :obj:`None`, will be automatically set to
            :obj:`in_channels / 4`. (default: :obj:`None`)
        dilation (int, optional): The factor by which the neighborhood is
            extended, from which :obj:`kernel_size` neighbors are then
            uniformly sampled. Can be interpreted as the dilation rate of
            classical convolutional operators. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_cluster.knn_graph`.
    """

    def __init__(self, in_channels, out_channels, dim, kernel_size, hidden_channels=None, dilation=1, bias=True, **kwargs):
        super(XConv, self).__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.kwargs = kwargs
        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size
        self.mlp1 = S(L(dim, C_delta), ELU(), BN(C_delta), L(C_delta, C_delta), ELU(), BN(C_delta), Reshape(-1, K, C_delta))
        self.mlp2 = S(L(D * K, K ** 2), ELU(), BN(K ** 2), Reshape(-1, K, K), Conv1d(K, K ** 2, K, groups=K), ELU(), BN(K ** 2), Reshape(-1, K, K), Conv1d(K, K ** 2, K, groups=K), BN(K ** 2), Reshape(-1, K, K))
        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in), Reshape(-1, C_in * depth_multiplier), L(C_in * depth_multiplier, C_out, bias=bias))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x, pos, edge_index):
        posFrom, posTo = pos
        (N, D), K = posTo.size(), self.kernel_size
        idxFrom, idxTo = edge_index
        relPos = posTo[idxTo] - posFrom[idxFrom]
        x_star = self.mlp1(relPos)
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[idxFrom].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)
        transform_matrix = self.mlp2(relPos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)
        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)
        out = self.conv(x_transformed)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class RSConvMapper(nn.Module):
    """[This class handles the special mechanism between the msg
        and the features of RSConv]
    """

    def __init__(self, down_conv_nn, use_xyz, bn=True, activation=nn.LeakyReLU(negative_slope=0.01), *args, **kwargs):
        super(RSConvMapper, self).__init__()
        self._down_conv_nn = down_conv_nn
        self._use_xyz = use_xyz
        self.nn = nn.ModuleDict()
        if len(self._down_conv_nn) == 2:
            self._first_layer = True
            f_in, f_intermediate, f_out = self._down_conv_nn[0]
            self.nn['features_nn'] = MLP2D(self._down_conv_nn[1], bn=bn, bias=False)
        else:
            self._first_layer = False
            f_in, f_intermediate, f_out = self._down_conv_nn
        self.nn['mlp_msg'] = MLP2D([f_in, f_intermediate, f_out], bn=bn, bias=False)
        self.nn['norm'] = Sequential(*[nn.BatchNorm2d(f_out), activation])
        self._f_out = f_out

    @property
    def f_out(self):
        return self._f_out

    def forward(self, features, msg):
        """
        features  -- [B, C, num_points, nsamples]
        msg  -- [B, 10, num_points, nsamples]

        The 10 features comes from [distance: 1,
                                    coord_origin:3,
                                    coord_target:3,
                                    delta_origin_target:3]
        """
        msg = self.nn['mlp_msg'](msg)
        if self._first_layer:
            features = self.nn['features_nn'](features)
        return self.nn['norm'](torch.mul(features, msg))


class SharedRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapper: RSConvMapper, radius):
        super(SharedRSConv, self).__init__()
        self._mapper = mapper
        self._radius = radius

    def forward(self, aggr_features, centroids):
        """
        aggr_features  -- [B, 3 + 3 + C, num_points, nsamples]
        centroids  -- [B, 3, num_points, 1]
        """
        abs_coord = aggr_features[:, :3]
        delta_x = aggr_features[:, 3:6]
        features = aggr_features[:, 3:]
        nsample = abs_coord.shape[-1]
        coord_xi = centroids.repeat(1, 1, 1, nsample)
        distance = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((distance, coord_xi, abs_coord, delta_x), dim=1)
        return self._mapper(features, h_xi_xj)

    def __repr__(self):
        return '{}(radius={})'.format(self.__class__.__name__, self._radius)


class OriginalRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapping=None, first_layer=False, radius=None, activation=nn.ReLU(inplace=True)):
        super(OriginalRSConv, self).__init__()
        self.nn = nn.ModuleList()
        self._radius = radius
        self.mapping_func1 = mapping[0]
        self.mapping_func2 = mapping[1]
        self.cr_mapping = mapping[2]
        self.first_layer = first_layer
        if first_layer:
            self.xyz_raising = mapping[3]
            self.bn_xyz_raising = nn.BatchNorm2d(self.xyz_raising.out_channels)
            self.nn.append(self.bn_xyz_raising)
        self.bn_mapping = nn.BatchNorm2d(self.mapping_func1.out_channels)
        self.bn_rsconv = nn.BatchNorm2d(self.cr_mapping.in_channels)
        self.bn_channel_raising = nn.BatchNorm1d(self.cr_mapping.out_channels)
        self.nn.append(self.bn_mapping)
        self.nn.append(self.bn_rsconv)
        self.nn.append(self.bn_channel_raising)
        self.activation = activation

    def forward(self, input):
        x = input[:, 3:, :, :]
        nsample = x.size()[3]
        abs_coord = input[:, 0:3, :, :]
        delta_x = input[:, 3:6, :, :]
        coord_xi = abs_coord[:, :, :, 0:1].repeat(1, 1, 1, nsample)
        h_xi_xj = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((h_xi_xj, coord_xi, abs_coord, delta_x), dim=1)
        h_xi_xj = self.mapping_func2(self.activation(self.bn_mapping(self.mapping_func1(h_xi_xj))))
        if self.first_layer:
            x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        x = F.max_pool2d(self.activation(self.bn_rsconv(torch.mul(h_xi_xj, x))), kernel_size=(1, nsample)).squeeze(3)
        x = self.activation(self.bn_channel_raising(self.cr_mapping(x)))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn.__repr__())


class BaseConvolution(ABC, BaseModule):

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        BaseModule.__init__(self)
        self.sampler = sampler
        self.neighbour_finder = neighbour_finder


class BaseConvolutionDown(BaseConvolution):

    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        super(BaseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)
        self._index = kwargs.get('index', None)

    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        idx = self.sampler(pos, batch)
        row, col = self.neighbour_finder(pos, pos[idx], batch_x=batch, batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        batch_obj.idx = idx
        batch_obj.edge_index = edge_index
        batch_obj.x = self.conv(x, (pos[idx], pos), edge_index, batch)
        batch_obj.pos = pos[idx]
        batch_obj.batch = batch[idx]
        copy_from_to(data, batch_obj)
        return batch_obj


class KNNNeighbourFinder(BaseNeighbourFinder):

    def __init__(self, k):
        self.k = k

    def find_neighbours(self, x, y, batch_x, batch_y):
        return knn(x, y, self.k, batch_x, batch_y)


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if ratio is not None or subsampling_param is not None:
                raise ValueError('Can only specify ratio or num_to_sample or subsampling_param, not several !')
            self._num_to_sample = num_to_sample
        elif ratio is not None:
            self._ratio = ratio
        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param
        else:
            raise Exception('At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, pos, x=None, batch=None):
        return self.sample(pos, batch=batch, x=x)

    def _get_num_to_sample(self, batch_size) ->int:
        if hasattr(self, '_num_to_sample'):
            return self._num_to_sample
        else:
            return math.floor(batch_size * self._ratio)

    def _get_ratio_to_sample(self, batch_size) ->float:
        if hasattr(self, '_ratio'):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, pos, x=None, batch=None):
        pass


class RandomSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch, **kwargs):
        if len(pos.shape) != 2:
            raise ValueError(' This class is for sparse data and expects the pos tensor to be of dimension 2')
        idx = torch.randint(0, pos.shape[0], (self._get_num_to_sample(pos.shape[0]),))
        return idx


class RandlaConv(BaseConvolutionDown):

    def __init__(self, ratio=None, k=None, *args, **kwargs):
        super(RandlaConv, self).__init__(RandomSampler(ratio), KNNNeighbourFinder(k), *args, **kwargs)
        if kwargs.get('index') == 0 and kwargs.get('nb_feature') is not None:
            kwargs['point_pos_nn'][-1] = kwargs.get('nb_feature')
            kwargs['attention_nn'][0] = kwargs['attention_nn'][-1] = kwargs.get('nb_feature') * 2
            kwargs['down_conv_nn'][0] = kwargs.get('nb_feature') * 2
        self._conv = RandlaKernel(*args, global_nn=kwargs['down_conv_nn'], **kwargs)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class DilatedResidualBlock(BaseResnetBlock):

    def __init__(self, indim, outdim, ratio1, ratio2, point_pos_nn1, point_pos_nn2, attention_nn1, attention_nn2, global_nn1, global_nn2, *args, **kwargs):
        if kwargs.get('index') == 0 and kwargs.get('nb_feature') is not None:
            indim = kwargs.get('nb_feature')
        super(DilatedResidualBlock, self).__init__(indim, outdim, outdim)
        self.conv1 = RandlaConv(ratio1, 16, *args, point_pos_nn=point_pos_nn1, attention_nn=attention_nn1, down_conv_nn=global_nn1, **kwargs)
        kwargs['nb_feature'] = None
        self.conv2 = RandlaConv(ratio2, 16, *args, point_pos_nn=point_pos_nn2, attention_nn=attention_nn2, down_conv_nn=global_nn2, **kwargs)

    def convs(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        return data


class RandLANetRes(torch.nn.Module):

    def __init__(self, indim, outdim, ratio, point_pos_nn, attention_nn, down_conv_nn, *args, **kwargs):
        super(RandLANetRes, self).__init__()
        self._conv = DilatedResidualBlock(indim, outdim, ratio[0], ratio[1], point_pos_nn[0], point_pos_nn[1], attention_nn[0], attention_nn[1], down_conv_nn[0], down_conv_nn[1], *args, **kwargs)

    def forward(self, data):
        return self._conv.forward(data)


class BaseMSNeighbourFinder(ABC):

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        return self.find_neighbours(x, y, batch_x=batch_x, batch_y=batch_y, scale_idx=scale_idx)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        pass

    @property
    @abstractmethod
    def num_scales(self):
        pass

    @property
    def dist_meters(self):
        return getattr(self, '_dist_meters', None)


class BaseDenseConvolutionDown(BaseConvolution):
    """ Multiscale convolution down (also supports single scale). Convolution kernel is shared accross the scales

        Arguments:
            sampler  -- Strategy for sampling the input clouds
            neighbour_finder -- Multiscale strategy for finding neighbours
    """
    CONV_TYPE = ConvolutionFormat.DENSE.value

    def __init__(self, sampler, neighbour_finder: BaseMSNeighbourFinder, *args, **kwargs):
        super(BaseDenseConvolutionDown, self).__init__(sampler, neighbour_finder, *args, **kwargs)
        self._index = kwargs.get('index', None)
        self._save_sampling_id = kwargs.get('save_sampling_id', None)

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        """
        raise NotImplementedError

    def forward(self, data, sample_idx=None, **kwargs):
        """
        Parameters
        ----------
        data: Data
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
        sample_idx: Optional[torch.Tensor]
            can be used to shortcut the sampler [B,K]
        """
        x, pos = data.x, data.pos
        if sample_idx:
            idx = sample_idx
        else:
            idx = self.sampler(pos)
        idx = idx.unsqueeze(-1).repeat(1, 1, pos.shape[-1]).long()
        new_pos = pos.gather(1, idx)
        ms_x = []
        for scale_idx in range(self.neighbour_finder.num_scales):
            radius_idx = self.neighbour_finder(pos, new_pos, scale_idx=scale_idx)
            ms_x.append(self.conv(x, pos, new_pos, radius_idx, scale_idx))
        new_x = torch.cat(ms_x, 1)
        new_data = Data(pos=new_pos, x=new_x)
        if self._save_sampling_id:
            setattr(new_data, 'sampling_id_{}'.format(self._index), idx[:, :, (0)])
        return new_data


class DenseFPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, **kwargs):
        """ Sample pos

        Arguments:
            pos -- [B, N, 3]

        Returns:
            indexes -- [B, num_sample]
        """
        if len(pos.shape) != 3:
            raise ValueError(' This class is for dense data and expects the pos tensor to be of dimension 2')
        return tp.furthest_point_sample(pos, self._get_num_to_sample(pos.shape[1]))


DEBUGGING_VARS = {'FIND_NEIGHBOUR_DIST': False}


class DistributionNeighbour(object):

    def __init__(self, radius, bins=1000):
        self._radius = radius
        self._bins = bins
        self._histogram = np.zeros(self._bins)

    def reset(self):
        self._histogram = np.zeros(self._bins)

    @property
    def radius(self):
        return self._radius

    @property
    def histogram(self):
        return self._histogram

    @property
    def histogram_non_zero(self):
        idx = len(self._histogram) - np.cumsum(self._histogram[::-1]).nonzero()[0][0]
        return self._histogram[:idx]

    def add_valid_neighbours(self, points):
        for num_valid in points:
            self._histogram[num_valid] += 1

    def __repr__(self):
        return '{}(radius={}, bins={})'.format(self.__class__.__name__, self._radius, self._bins)


class MultiscaleRadiusNeighbourFinder(BaseMSNeighbourFinder):
    """ Radius search with support for multiscale for sparse graphs

        Arguments:
            radius {Union[float, List[float]]}

        Keyword Arguments:
            max_num_neighbors {Union[int, List[int]]}  (default: {64})

        Raises:
            ValueError: [description]
    """

    def __init__(self, radius: Union[float, List[float]], max_num_neighbors: Union[int, List[int]]=64):
        if DEBUGGING_VARS['FIND_NEIGHBOUR_DIST']:
            if not isinstance(radius, list):
                radius = [radius]
            self._dist_meters = [DistributionNeighbour(r) for r in radius]
            if not isinstance(max_num_neighbors, list):
                max_num_neighbors = [max_num_neighbors]
            max_num_neighbors = [(256) for _ in max_num_neighbors]
        if not is_list(max_num_neighbors) and is_list(radius):
            self._radius = cast(list, radius)
            max_num_neighbors = cast(int, max_num_neighbors)
            self._max_num_neighbors = [max_num_neighbors for i in range(len(self._radius))]
            return
        if not is_list(radius) and is_list(max_num_neighbors):
            self._max_num_neighbors = cast(list, max_num_neighbors)
            radius = cast(int, radius)
            self._radius = [radius for i in range(len(self._max_num_neighbors))]
            return
        if is_list(max_num_neighbors):
            max_num_neighbors = cast(list, max_num_neighbors)
            radius = cast(list, radius)
            if len(max_num_neighbors) != len(radius):
                raise ValueError('Both lists max_num_neighbors and radius should be of the same length')
            self._max_num_neighbors = max_num_neighbors
            self._radius = radius
            return
        self._max_num_neighbors = [cast(int, max_num_neighbors)]
        self._radius = [cast(int, radius)]

    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError('Scale %i is out of bounds %i' % (scale_idx, self.num_scales))
        radius_idx = radius(x, y, self._radius[scale_idx], batch_x, batch_y, max_num_neighbors=self._max_num_neighbors[scale_idx])
        return radius_idx

    @property
    def num_scales(self):
        return len(self._radius)

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        """ Sparse interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, batch_x, batch_y, scale_idx)


class DenseRadiusNeighbourFinder(MultiscaleRadiusNeighbourFinder):
    """ Multiscale radius search for dense graphs
    """

    def find_neighbours(self, x, y, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError('Scale %i is out of bounds %i' % (scale_idx, self.num_scales))
        num_neighbours = self._max_num_neighbors[scale_idx]
        neighbours = tp.ball_query(self._radius[scale_idx], num_neighbours, x, y)[0]
        if DEBUGGING_VARS['FIND_NEIGHBOUR_DIST']:
            for i in range(neighbours.shape[0]):
                start = neighbours[(i), :, (0)]
                valid_neighbours = (neighbours[(i), :, 1:] != start.view((-1, 1)).repeat(1, num_neighbours - 1)).sum(1) + 1
                self._dist_meters[scale_idx].add_valid_neighbours(valid_neighbours)
        return neighbours

    def __call__(self, x, y, scale_idx=0, **kwargs):
        """ Dense interface of the neighboorhood finder
        """
        return self.find_neighbours(x, y, scale_idx)


class PointNetMSGDown(BaseDenseConvolutionDown):

    def __init__(self, npoint=None, radii=None, nsample=None, down_conv_nn=None, bn=True, activation=torch.nn.LeakyReLU(negative_slope=0.01), use_xyz=True, **kwargs):
        assert len(radii) == len(nsample) == len(down_conv_nn)
        super(PointNetMSGDown, self).__init__(DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs)
        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            self.mlps.append(MLP2D(down_conv_nn[i], bn=bn, activation=activation, bias=False))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(new_pos_trans, idx)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)
        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_pos
        return new_features

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        Returns:
            new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
        """
        assert scale_idx < len(self.mlps)
        new_features = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](new_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        new_features = new_features.squeeze(-1)
        return new_features


class BoxData:
    """ Basic data structure to hold a box prediction or ground truth
    if an objectness is provided then it will be treated as a prediction. Else, it is a ground truth box
    """

    def __init__(self, classname, corners3d, objectness=None):
        assert corners3d.shape == (8, 3)
        assert objectness is None or objectness <= 1 and objectness >= 0
        if torch.is_tensor(classname):
            classname = classname.cpu().item()
        self.classname = classname
        if torch.is_tensor(corners3d):
            corners3d = corners3d.cpu().numpy()
        self.corners3d = corners3d
        if torch.is_tensor(objectness):
            objectness = objectness.cpu().item()
        self.objectness = objectness

    @property
    def is_gt(self):
        return self.objectness is not None

    def __repr__(self):
        return '{}: (objectness={})'.format(self.__class__.__name__, self.objectness)


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R


def box_corners_from_param(box_size, heading_angle, center):
    """ Generates box corners from a parameterised box.
    box_size is array(size_x,size_y,size_z), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box corners
    """
    R = euler_angles_to_rotation_matrix(torch.tensor([0.0, 0.0, float(heading_angle)]))
    if torch.is_tensor(box_size):
        box_size = box_size.float()
    l, w, h = box_size
    x_corners = torch.tensor([-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2])
    y_corners = torch.tensor([-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2])
    z_corners = torch.tensor([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    corners_3d = R @ torch.stack([x_corners, y_corners, z_corners])
    corners_3d[(0), :] = corners_3d[(0), :] + center[0]
    corners_3d[(1), :] = corners_3d[(1), :] + center[1]
    corners_3d[(2), :] = corners_3d[(2), :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d


def nms_samecls(boxes, classes, scores, overlap_threshold=0.25):
    """ Returns the list of boxes that are kept after nms.
    A box is suppressed only if it overlaps with
    another box of the same class that has a higher score

    Parameters
    ----------
    boxes : [num_boxes, 6]
        xmin, ymin, zmin, xmax, ymax, zmax
    classes : [num_shapes]
        Class of each box
    scores : [num_shapes,]
        score of each box
    overlap_threshold : float, optional
        [description], by default 0.25
    """
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(classes):
        classes = classes.cpu().numpy()
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    z1 = boxes[:, (2)]
    x2 = boxes[:, (3)]
    y2 = boxes[:, (4)]
    z2 = boxes[:, (5)]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    I = np.argsort(scores)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[I[:last - 1]])
        yy1 = np.maximum(y1[i], y1[I[:last - 1]])
        zz1 = np.maximum(z1[i], z1[I[:last - 1]])
        xx2 = np.minimum(x2[i], x2[I[:last - 1]])
        yy2 = np.minimum(y2[i], y2[I[:last - 1]])
        zz2 = np.minimum(z2[i], z2[I[:last - 1]])
        cls1 = classes[i]
        cls2 = classes[I[:last - 1]]
        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)
        inter = l * w * h
        o = inter / (area[i] + area[I[:last - 1]] - inter)
        o = o * (cls1 == cls2)
        I = np.delete(I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0])))
    return pick


def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)
    dist1, idx1 = torch.min(pc_dist, dim=2)
    dist2, idx2 = torch.min(pc_dist, dim=1)
    return dist1, idx1, dist2, idx2


class ProposalModule(nn.Module):

    def __init__(self, num_class, vote_aggregation_config, num_heading_bin, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__()
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = len(mean_size_arr)
        self.mean_size_arr = nn.Parameter(torch.Tensor(mean_size_arr), requires_grad=False)
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        assert vote_aggregation_config.module_name == 'PointNetMSGDown', 'Proposal Module support only PointNet2 for now'
        params = OmegaConf.to_container(vote_aggregation_config)
        self.vote_aggregation = PointNetMSGDown(**params)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + self.num_size_cluster * 4 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, data):
        """
        Args:
            pos: (B,N,3)
            features: (B,C,N)
            seed_pos (B,N,3)
        Returns:
            VoteNetResults
        """
        if data.pos.dim() != 3:
            raise ValueError('This method only supports dense convolutions for now')
        if self.sampling == 'seed_fps':
            sample_idx = tp.furthest_point_sample(data.seed_pos, self.num_proposal)
        else:
            raise ValueError('Unknown sampling strategy: %s. Exiting!' % self.sampling)
        data_features = self.vote_aggregation(data, sampled_idx=sample_idx)
        x = F.relu(self.bn1(self.conv1(data_features.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return VoteNetResults.from_logits(data.seed_inds, data.pos, data.seed_pos, data_features.pos, x, self.num_class, self.num_heading_bin, self.mean_size_arr)


class VotingModule(nn.Module):

    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, data):
        """ Votes for centres using a PN++ like architecture
        Returns
        -------
        data:
            - pos: position of the vote (centre of the box)
            - x: feature of the vote (original feature + processed feature)
            - seed_pos: position of the original point
        """
        if data.pos.dim() != 3:
            raise ValueError('This method only supports dense convolutions for now')
        batch_size = data.pos.shape[0]
        num_points = data.pos.shape[1]
        num_votes = num_points * self.vote_factor
        x = F.relu(self.bn1(self.conv1(data.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).view(batch_size, num_points, self.vote_factor, 3 + self.out_dim)
        offset = x[:, :, :, 0:3]
        vote_pos = data.pos.unsqueeze(2) + offset
        vote_pos = vote_pos.contiguous().view(batch_size, num_votes, 3)
        res_x = x[:, :, :, 3:]
        vote_x = data.x.transpose(2, 1).unsqueeze(2) + res_x
        vote_x = vote_x.contiguous().view(batch_size, num_votes, self.out_dim)
        vote_x = vote_x.transpose(2, 1).contiguous()
        return Data(pos=vote_pos, x=vote_x, seed_pos=data.pos)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContrastiveHardestNegativeLoss,
     lambda: ([], {'pos_thresh': 4, 'neg_thresh': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Conv1D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (FastBatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (HuberLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LossAnnealer,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Seq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_nicolas_chaulet_torch_points3d(_paritybench_base):
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

