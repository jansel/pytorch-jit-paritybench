import sys
_module = sys.modules[__name__]
del sys
custom = _module
dataset = _module
forward_proc = _module
loss = _module
optim = _module
registry = _module
text_classification = _module
image_classification = _module
log_visualizer = _module
object_detection = _module
params_extractor = _module
semantic_segmentation = _module
setup = _module
core_test = _module
registry_test = _module
torchdistill = _module
common = _module
constant = _module
file_util = _module
main_util = _module
misc_util = _module
module_util = _module
tensor_util = _module
yaml_util = _module
core = _module
distillation = _module
forward_hook = _module
training = _module
util = _module
datasets = _module
coco = _module
collator = _module
registry = _module
sample_loader = _module
sampler = _module
transform = _module
util = _module
wrapper = _module
eval = _module
classification = _module
coco = _module
losses = _module
custom = _module
registry = _module
single = _module
misc = _module
log = _module
models = _module
adaptation = _module
densenet = _module
resnet = _module
wide_resnet = _module
bottleneck = _module
base = _module
densenet = _module
inception = _module
resnet = _module
detection = _module
rcnn = _module
resnet_backbone = _module
processor = _module
official = _module
registry = _module
special = _module
util = _module
registry = _module
scheduler = _module

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


from torch import nn


from torch.nn import functional


import math


import logging


import time


import numpy as np


import pandas as pd


from torch.backends import cudnn


from torch import distributed as dist


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data._utils.collate import default_collate


from torchvision.models.detection.keypoint_rcnn import KeypointRCNN


from torchvision.models.detection.mask_rcnn import MaskRCNN


from torchvision import models


import random


import torch.distributed as dist


from collections import OrderedDict


from torch.nn import Sequential


from torch.nn import ModuleList


import copy


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import LambdaLR


from collections import abc


from torch._six import string_classes


from torch.nn.parallel.scatter_gather import gather


import torch.utils.data


from torchvision.datasets import CocoDetection


from torchvision.transforms import functional


from types import BuiltinFunctionType


from types import BuiltinMethodType


from types import FunctionType


import torchvision


from collections import defaultdict


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from torch.utils.model_zoo import tqdm


from torchvision import transforms as T


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import functional as F


from torchvision.transforms.functional import InterpolationMode


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import random_split


from torch.utils.data.distributed import DistributedSampler


from torchvision.datasets import PhotoTour


from torchvision.datasets import HMDB51


from torchvision.datasets import UCF101


from torchvision.datasets import Cityscapes


from torchvision.datasets import CocoCaptions


from torchvision.datasets import SBDataset


from torchvision.datasets import VOCSegmentation


from torchvision.datasets import VOCDetection


from torch.utils.data import Dataset


from torch.nn.functional import adaptive_avg_pool2d


from torch.nn.functional import adaptive_max_pool2d


from torch.nn.functional import normalize


from torch.nn.functional import cosine_similarity


from collections import deque


from logging import FileHandler


from logging import Formatter


from typing import Any


from typing import Tuple


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torchvision.models.densenet import _DenseBlock


from torchvision.models.densenet import _Transition


from typing import Type


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import conv1x1


from torchvision.models import densenet169


from torchvision.models import densenet201


from torchvision.models import inception_v3


from torchvision.models import resnet50


from torchvision.models import resnet101


from torchvision.models import resnet152


from torch.hub import load_state_dict_from_url


from torchvision.models.detection.faster_rcnn import FasterRCNN


from torchvision.models.detection.faster_rcnn import model_urls as fasterrcnn_model_urls


from torchvision.models.detection.keypoint_rcnn import model_urls as keypointrcnn_model_urls


from torchvision.models.detection.mask_rcnn import model_urls as maskrcnn_model_urls


from torchvision.ops import MultiScaleRoIAlign


from torchvision.models import resnet


from torchvision.models.detection.backbone_utils import BackboneWithFPN


from torchvision.ops import misc as misc_nn_ops


from collections import namedtuple


from torch.nn import SyncBatchNorm


from torch.jit.annotations import Tuple


from torch.jit.annotations import List


from torch.nn import Module


ORG_LOSS_LIST = list()


SINGLE_LOSS_CLASS_DICT = dict()


def register_org_loss(arg=None, **kwargs):

    def _register_org_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        SINGLE_LOSS_CLASS_DICT[key] = cls
        ORG_LOSS_LIST.append(cls)
        return cls
    if callable(arg):
        return _register_org_loss(arg)
    return _register_org_loss


class KDLoss4Transformer(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha

    def compute_soft_loss(self, student_logits, teacher_logits):
        return super().forward(torch.log_softmax(student_logits / self.temperature, dim=1), torch.softmax(teacher_logits / self.temperature, dim=1))

    def compute_hard_loss(self, logits, positions, ignored_index):
        return functional.cross_entropy(logits, positions, reduction=self.cel_reduction, ignore_index=ignored_index)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = self.compute_soft_loss(student_output.logits, teacher_output.logits)
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = student_output.loss
        return self.alpha * hard_loss + self.beta * self.temperature ** 2 * soft_loss


class SpecialModule(nn.Module):

    def __init__(self):
        super().__init__()

    def post_forward(self, *args, **kwargs):
        pass

    def post_process(self, *args, **kwargs):
        pass


class BaseDatasetWrapper(Dataset):

    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target = self.org_dataset.__getitem__(index)
        return sample, target, dict()

    def __len__(self):
        return len(self.org_dataset)


class CacheableDataset(BaseDatasetWrapper):

    def __init__(self, org_dataset, cache_dir_path, idx2subpath_func=None, ext='.pt'):
        super().__init__(org_dataset)
        self.cache_dir_path = cache_dir_path
        self.idx2subath_func = str if idx2subpath_func is None else idx2subpath_func
        self.ext = ext

    def __getitem__(self, index):
        sample, target, supp_dict = super().__getitem__(index)
        cache_file_path = os.path.join(self.cache_dir_path, self.idx2subath_func(index) + self.ext)
        if file_util.check_if_exists(cache_file_path):
            cached_data = torch.load(cache_file_path)
            supp_dict['cached_data'] = cached_data
        supp_dict['cache_file_path'] = cache_file_path
        return sample, target, supp_dict


def default_idx2subpath(index):
    digits_str = '{:04d}'.format(index)
    return os.path.join(digits_str[-4:], digits_str)


BATCH_SAMPLER_CLASS_DICT = dict()


def get_batch_sampler(class_name, *args, **kwargs):
    if class_name is None:
        return None
    if class_name not in BATCH_SAMPLER_CLASS_DICT and class_name != 'BatchSampler':
        raise ValueError('No batch sampler `{}` registered.'.format(class_name))
    batch_sampler_cls = BATCH_SAMPLER_CLASS_DICT[class_name]
    return batch_sampler_cls(*args, **kwargs)


COLLATE_FUNC_DICT = dict()


def get_collate_func(func_name):
    if func_name is None:
        return None
    elif func_name in COLLATE_FUNC_DICT:
        return COLLATE_FUNC_DICT[func_name]
    raise ValueError('No collate function `{}` registered'.format(func_name))


WRAPPER_CLASS_DICT = dict()


def get_dataset_wrapper(class_name, *args, **kwargs):
    if class_name not in WRAPPER_CLASS_DICT:
        return WRAPPER_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No dataset wrapper `{}` registered.'.format(class_name))


def build_data_loader(dataset, data_loader_config, distributed, accelerator=None):
    num_workers = data_loader_config['num_workers']
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset = get_dataset_wrapper(dataset_wrapper_config['name'], dataset, **dataset_wrapper_config['params'])
    elif cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)
    sampler = DistributedSampler(dataset) if distributed and accelerator is None else RandomSampler(dataset) if data_loader_config.get('random_sample', False) else SequentialSampler(dataset)
    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler = None if batch_sampler_config is None else get_batch_sampler(batch_sampler_config['type'], sampler, **batch_sampler_config['params'])
    collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    drop_last = data_loader_config.get('drop_last', False)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last)
    batch_size = data_loader_config['batch_size']
    pin_memory = data_loader_config.get('pin_memory', True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)


def build_data_loaders(dataset_dict, data_loader_configs, distributed, accelerator=None):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed, accelerator)
        data_loader_list.append(data_loader)
    return data_loader_list


SPECIAL_CLASS_DICT = dict()


def get_special_module(class_name, *args, **kwargs):
    if class_name in SPECIAL_CLASS_DICT:
        return SPECIAL_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No special module `{}` registered'.format(class_name))


def build_special_module(model_config, **kwargs):
    special_model_config = model_config.get('special', dict())
    special_model_type = special_model_config.get('type', None)
    if special_model_type is None:
        return None
    special_model_params_config = special_model_config.get('params', None)
    if special_model_params_config is None:
        special_model_params_config = dict()
    return get_special_module(special_model_type, **kwargs, **special_model_params_config)


def change_device(data, device):
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return elem_type(*(change_device(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(change_device(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: change_device(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [change_device(samples, device) for samples in transposed]
    return data


def check_if_wrapped(model):
    return isinstance(model, (DataParallel, DistributedDataParallel))


def clear_io_dict(model_io_dict):
    for module_io_dict in model_io_dict.values():
        for sub_dict in list(module_io_dict.values()):
            sub_dict.clear()


def extract_io_dict(model_io_dict, target_device):
    uses_cuda = target_device.type == 'cuda'
    gathered_io_dict = dict()
    for module_path, module_io_dict in model_io_dict.items():
        gathered_io_dict[module_path] = dict()
        for io_type in list(module_io_dict.keys()):
            sub_dict = module_io_dict.pop(io_type)
            values = [sub_dict[key] for key in sorted(sub_dict.keys())]
            gathered_obj = gather(values, target_device) if uses_cuda and len(values) > 1 else values[-1]
            gathered_io_dict[module_path][io_type] = gathered_obj
    return gathered_io_dict


def extract_sub_model_output_dict(model_output_dict, index):
    sub_model_output_dict = dict()
    for module_path, sub_model_io_dict in model_output_dict.items():
        tmp_dict = dict()
        for key, value in sub_model_io_dict.items():
            tmp_dict[key] = value[index]
        sub_model_output_dict[module_path] = tmp_dict
    return sub_model_output_dict


def freeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = False


CUSTOM_LOSS_CLASS_DICT = dict()


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('No custom loss `{}` registered'.format(criterion_type))


PROC_FUNC_DICT = dict()


def get_forward_proc_func(func_name):
    if func_name is None:
        return PROC_FUNC_DICT['forward_batch_only']
    elif func_name in PROC_FUNC_DICT:
        return PROC_FUNC_DICT[func_name]
    raise ValueError('No forward process function `{}` registered'.format(func_name))


FUNC2EXTRACT_ORG_OUTPUT_DICT = dict()


def register_func2extract_org_output(arg=None, **kwargs):

    def _register_func2extract_org_output(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__
        FUNC2EXTRACT_ORG_OUTPUT_DICT[key] = func
        return func
    if callable(arg):
        return _register_func2extract_org_output(arg)
    return _register_func2extract_org_output


@register_func2extract_org_output
def extract_simple_org_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        if isinstance(student_outputs, (list, tuple)):
            if uses_teacher_output:
                for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                    org_loss_dict[i] = org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = org_criterion(sub_outputs, targets)
        else:
            org_loss = org_criterion(student_outputs, teacher_outputs, targets) if uses_teacher_output else org_criterion(student_outputs, targets)
            org_loss_dict = {(0): org_loss}
    return org_loss_dict


def get_func2extract_org_output(func_name):
    if func_name is None:
        return extract_simple_org_loss
    elif func_name in FUNC2EXTRACT_ORG_OUTPUT_DICT:
        return FUNC2EXTRACT_ORG_OUTPUT_DICT[func_name]
    raise ValueError('No function to extract original output `{}` registered'.format(func_name))


def_logger = logging.getLogger()


logger = def_logger.getChild(__name__)


def get_module(root_module, module_path):
    module_names = module_path.split('.')
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (DataParallel, DistributedDataParallel)):
                module = module.module
                if not hasattr(module, module_name):
                    if isinstance(module, Sequential) and module_name.lstrip('-').isnumeric():
                        module = module[int(module_name)]
                    else:
                        logger.info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path, type(root_module).__name__))
                else:
                    module = getattr(module, module_name)
            elif isinstance(module, (Sequential, ModuleList)) and module_name.lstrip('-').isnumeric():
                module = module[int(module_name)]
            else:
                logger.info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path, type(root_module).__name__))
                return None
        else:
            module = getattr(module, module_name)
    return module


def get_optimizer(module, optim_type, param_dict=None, filters_params=True, **kwargs):
    if param_dict is None:
        param_dict = dict()
    is_module = isinstance(module, nn.Module)
    lower_optim_type = optim_type.lower()
    if lower_optim_type in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[lower_optim_type]
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            return optim_cls_or_func(updatable_params, **param_dict, **kwargs)
        return optim_cls_or_func(module, **param_dict, **kwargs)
    raise ValueError('No optimizer `{}` registered'.format(optim_type))


def get_scheduler(optimizer, scheduler_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()
    lower_scheduler_type = scheduler_type.lower()
    if lower_scheduler_type in SCHEDULER_DICT:
        return SCHEDULER_DICT[lower_scheduler_type](optimizer, **param_dict, **kwargs)
    raise ValueError('No scheduler `{}` registered'.format(scheduler_type))


def get_loss(loss_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()
    lower_loss_type = loss_type.lower()
    if lower_loss_type in LOSS_DICT:
        return LOSS_DICT[lower_loss_type](**param_dict, **kwargs)
    raise ValueError('No loss `{}` registered'.format(loss_type))


LOSS_WRAPPER_CLASS_DICT = dict()


def register_loss_wrapper(arg=None, **kwargs):

    def _register_loss_wrapper(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        LOSS_WRAPPER_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_loss_wrapper(arg)
    return _register_loss_wrapper


class SimpleLossWrapper(nn.Module):

    def __init__(self, single_loss, params_config):
        super().__init__()
        self.single_loss = single_loss
        input_config = params_config['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io']
        target_config = params_config['target']
        self.is_target_from_teacher = target_config['is_from_teacher']
        self.target_module_path = target_config['module_path']
        self.target_key = target_config['io']

    @staticmethod
    def extract_value(io_dict, path, key):
        return io_dict[path][key]

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict, self.input_module_path, self.input_key)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict, self.target_module_path, self.target_key)
        return self.single_loss(input_batch, target_batch, *args, **kwargs)

    def __str__(self):
        return self.single_loss.__str__()


def get_loss_wrapper(single_loss, params_config, wrapper_config):
    wrapper_type = wrapper_config.get('type', None)
    if wrapper_type is None:
        return SimpleLossWrapper(single_loss, params_config)
    elif wrapper_type in LOSS_WRAPPER_CLASS_DICT:
        return LOSS_WRAPPER_CLASS_DICT[wrapper_type](single_loss, params_config, **wrapper_config.get('params', dict()))
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_type))


def get_single_loss(single_criterion_config, params_config=None):
    loss_type = single_criterion_config['type']
    single_loss = SINGLE_LOSS_CLASS_DICT[loss_type](**single_criterion_config['params']) if loss_type in SINGLE_LOSS_CLASS_DICT else get_loss(loss_type, single_criterion_config['params'])
    if params_config is None:
        return single_loss
    return get_loss_wrapper(single_loss, params_config, params_config.get('wrapper', dict()))


def get_updatable_param_names(module):
    return [name for name, param in module.named_parameters() if param.requires_grad]


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def add_submodule(module, module_path, module_dict):
    module_names = module_path.split('.')
    module_name = module_names.pop(0)
    if len(module_names) == 0:
        if module_name in module_dict:
            raise KeyError('module_name `{}` is already used.'.format(module_name))
        module_dict[module_name] = module
        return
    next_module_path = '.'.join(module_names)
    sub_module_dict = module_dict.get(module_name, None)
    if module_name not in module_dict:
        sub_module_dict = OrderedDict()
        module_dict[module_name] = sub_module_dict
    add_submodule(module, next_module_path, sub_module_dict)


def build_sequential_container(module_dict):
    for key in module_dict.keys():
        value = module_dict[key]
        if isinstance(value, OrderedDict):
            value = build_sequential_container(value)
            module_dict[key] = value
        elif not isinstance(value, Module):
            raise ValueError('module type `{}` is not expected'.format(type(value)))
    return Sequential(module_dict)


ADAPTATION_CLASS_DICT = dict()


MODULE_CLASS_DICT = nn.__dict__


def get_adaptation_module(class_name, *args, **kwargs):
    if class_name in ADAPTATION_CLASS_DICT:
        return ADAPTATION_CLASS_DICT[class_name](*args, **kwargs)
    elif class_name in MODULE_CLASS_DICT:
        return MODULE_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(class_name))


def redesign_model(org_model, model_config, model_label, model_type='original'):
    logger.info('[{} model]'.format(model_label))
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} {} model'.format(model_type, model_label))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))
        isinstance_str = 'instance('
        for frozen_module_path in frozen_module_path_set:
            if frozen_module_path.startswith(isinstance_str) and frozen_module_path.endswith(')'):
                target_cls = nn.__dict__[frozen_module_path[len(isinstance_str):-1]]
                for m in org_model.modules():
                    if isinstance(m, target_cls):
                        freeze_module_params(m)
            else:
                module = get_module(org_model, frozen_module_path)
                freeze_module_params(module)
        return org_model
    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))
    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())
    for frozen_module_path in frozen_module_path_set:
        module = get_module(org_model, frozen_module_path)
        freeze_module_params(module)
    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['type'], **adaptation_config['params'])
        else:
            module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)
        add_submodule(module, module_path, module_dict)
    return build_sequential_container(module_dict)


def extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def get_device_index(data):
    if isinstance(data, torch.Tensor):
        device = data.device
        return 'cpu' if device.type == 'cpu' else device.index
    elif isinstance(data, abc.Mapping):
        for key, data in data.items():
            result = get_device_index(data)
            if result is not None:
                return result
    elif isinstance(data, tuple):
        for d in data:
            result = get_device_index(d)
            if result is not None:
                return result
    elif isinstance(data, abc.Sequence) and not isinstance(data, string_classes):
        for d in data:
            result = get_device_index(d)
            if result is not None:
                return result
    return None


def register_forward_hook_with_dict(module, module_path, requires_input, requires_output, io_dict):
    io_dict[module_path] = dict()

    def forward_hook4input(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]
        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'input' not in sub_io_dict:
            sub_io_dict['input'] = dict()
        sub_io_dict['input'][device_index] = func_input

    def forward_hook4output(self, func_input, func_output):
        if isinstance(func_output, tuple) and len(func_output) == 1:
            func_output = func_output[0]
        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'output' not in sub_io_dict:
            sub_io_dict['output'] = dict()
        sub_io_dict['output'][device_index] = func_output

    def forward_hook4io(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]
        if isinstance(func_output, tuple) and len(func_output) == 1:
            func_output = func_output[0]
        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'input' not in sub_io_dict:
            sub_io_dict['input'] = dict()
        if 'output' not in sub_io_dict:
            sub_io_dict['output'] = dict()
        sub_io_dict['input'][device_index] = func_input
        sub_io_dict['output'][device_index] = func_output
    if requires_input and not requires_output:
        return module.register_forward_hook(forward_hook4input)
    elif not requires_input and requires_output:
        return module.register_forward_hook(forward_hook4output)
    elif requires_input and requires_output:
        return module.register_forward_hook(forward_hook4io)
    raise ValueError('Either requires_input or requires_output should be True')


def set_distillation_box_info(io_dict, module_path, **kwargs):
    io_dict[module_path] = kwargs


def set_hooks(model, unwrapped_org_model, model_config, io_dict):
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list
    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    for target_module_path in input_module_path_set.union(output_module_path_set):
        requires_input = target_module_path in input_module_path_set
        requires_output = target_module_path in output_module_path_set
        set_distillation_box_info(io_dict, target_module_path)
        target_module = extract_module(unwrapped_org_model, model, target_module_path)
        handle = register_forward_hook_with_dict(target_module, target_module_path, requires_input, requires_output, io_dict)
        pair_list.append((target_module_path, handle))
    return pair_list


def tensor2numpy2tensor(data, device):
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return torch.Tensor(data.data.numpy())
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return elem_type(*(tensor2numpy2tensor(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(tensor2numpy2tensor(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: tensor2numpy2tensor(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [tensor2numpy2tensor(samples, device) for samples in transposed]
    return data


def unfreeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = True


def update_io_dict(main_io_dict, new_io_dict):
    for key, module_io_dict in new_io_dict.items():
        for io_type, value in module_io_dict.items():
            if len(value) > 0:
                main_io_dict[key][io_type] = value


def wrap_model(model, model_config, device, device_ids=None, distributed=False, find_unused_parameters=False, any_updatable=True):
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    model
    if wrapper is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper == 'DistributedDataParallel' and distributed and any_updatable:
            model = DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
        elif wrapper in {'DataParallel', 'DistributedDataParallel'}:
            model = DataParallel(model, device_ids=device_ids)
    return model


class DistillationBox(nn.Module):

    def setup_data_loaders(self, train_config):
        train_data_loader_config = train_config.get('train_data_loader', dict())
        if 'requires_supp' not in train_data_loader_config:
            train_data_loader_config['requires_supp'] = True
        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader = build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config], self.distributed, self.accelerator)
        if train_data_loader is not None:
            self.train_data_loader = train_data_loader
        if val_data_loader is not None:
            self.val_data_loader = val_data_loader

    def setup_teacher_student_models(self, teacher_config, student_config):
        unwrapped_org_teacher_model = self.org_teacher_model.module if check_if_wrapped(self.org_teacher_model) else self.org_teacher_model
        unwrapped_org_student_model = self.org_student_model.module if check_if_wrapped(self.org_student_model) else self.org_student_model
        self.target_teacher_pairs.clear()
        self.target_student_pairs.clear()
        teacher_ref_model = unwrapped_org_teacher_model
        student_ref_model = unwrapped_org_student_model
        if len(teacher_config) > 0 or len(teacher_config) == 0 and self.teacher_model is None:
            model_type = 'original'
            special_teacher_model = build_special_module(teacher_config, teacher_model=unwrapped_org_teacher_model, device=self.device, device_ids=self.device_ids, distributed=self.distributed)
            if special_teacher_model is not None:
                teacher_ref_model = special_teacher_model
                model_type = type(teacher_ref_model).__name__
            self.teacher_model = redesign_model(teacher_ref_model, teacher_config, 'teacher', model_type)
        if len(student_config) > 0 or len(student_config) == 0 and self.student_model is None:
            model_type = 'original'
            special_student_model = build_special_module(student_config, student_model=unwrapped_org_student_model, device=self.device, device_ids=self.device_ids, distributed=self.distributed)
            if special_student_model is not None:
                student_ref_model = special_student_model
                model_type = type(student_ref_model).__name__
            self.student_model = redesign_model(student_ref_model, student_config, 'student', model_type)
        self.teacher_any_frozen = len(teacher_config.get('frozen_modules', list())) > 0 or not teacher_config.get('requires_grad', True)
        self.student_any_frozen = len(student_config.get('frozen_modules', list())) > 0 or not student_config.get('requires_grad', True)
        self.target_teacher_pairs.extend(set_hooks(self.teacher_model, teacher_ref_model, teacher_config, self.teacher_io_dict))
        self.target_student_pairs.extend(set_hooks(self.student_model, student_ref_model, student_config, self.student_io_dict))
        self.teacher_forward_proc = get_forward_proc_func(teacher_config.get('forward_proc', None))
        self.student_forward_proc = get_forward_proc_func(student_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        self.org_criterion = None if org_criterion_config is None or len(org_criterion_config) == 0 else get_single_loss(org_criterion_config)
        self.criterion = get_custom_loss(criterion_config)
        logger.info(self.criterion)
        self.uses_teacher_output = self.org_criterion is not None and isinstance(self.org_criterion, tuple(ORG_LOSS_LIST))
        self.extract_org_loss = get_func2extract_org_output(criterion_config.get('func2extract_org_loss', None))

    def setup(self, train_config):
        self.setup_data_loaders(train_config)
        teacher_config = train_config.get('teacher', dict())
        student_config = train_config.get('student', dict())
        self.setup_teacher_student_models(teacher_config, student_config)
        self.setup_loss(train_config)
        self.teacher_updatable = True
        if not teacher_config.get('requires_grad', True):
            logger.info('Freezing the whole teacher model')
            freeze_module_params(self.teacher_model)
            self.teacher_updatable = False
        if not student_config.get('requires_grad', True):
            logger.info('Freezing the whole student model')
            freeze_module_params(self.student_model)
        teacher_unused_parameters = teacher_config.get('find_unused_parameters', self.teacher_any_frozen)
        teacher_any_updatable = len(get_updatable_param_names(self.teacher_model)) > 0
        self.teacher_model = wrap_model(self.teacher_model, teacher_config, self.device, self.device_ids, self.distributed, teacher_unused_parameters, teacher_any_updatable)
        student_unused_parameters = student_config.get('find_unused_parameters', self.student_any_frozen)
        student_any_updatable = len(get_updatable_param_names(self.student_model)) > 0
        self.student_model = wrap_model(self.student_model, student_config, self.device, self.device_ids, self.distributed, student_unused_parameters, student_any_updatable)
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_params_config = optim_config['params']
            if 'lr' in optim_params_config:
                optim_params_config['lr'] *= self.lr_factor
            module_wise_params_configs = optim_config.get('module_wise_params', list())
            if len(module_wise_params_configs) > 0:
                trainable_module_list = list()
                for module_wise_params_config in module_wise_params_configs:
                    module_wise_params_dict = dict()
                    if isinstance(module_wise_params_config.get('params', None), dict):
                        module_wise_params_dict.update(module_wise_params_config['params'])
                    if 'lr' in module_wise_params_dict:
                        module_wise_params_dict['lr'] *= self.lr_factor
                    target_model = self.teacher_model if module_wise_params_config.get('is_teacher', False) else self.student_model
                    module = get_module(target_model, module_wise_params_config['module'])
                    module_wise_params_dict['params'] = module.parameters()
                    trainable_module_list.append(module_wise_params_dict)
            else:
                trainable_module_list = nn.ModuleList([self.student_model])
                if self.teacher_updatable:
                    logger.info('Note that you are training some/all of the modules in the teacher model')
                    trainable_module_list.append(self.teacher_model)
            filters_params = optim_config.get('filters_params', True)
            self.optimizer = get_optimizer(trainable_module_list, optim_config['type'], optim_params_config, filters_params)
            self.optimizer.zero_grad()
            self.max_grad_norm = optim_config.get('max_grad_norm', None)
            self.grad_accum_step = optim_config.get('grad_accum_step', 1)
            optimizer_reset = True
        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
            self.scheduling_step = scheduler_config.get('scheduling_step', 0)
        elif optimizer_reset:
            self.lr_scheduler = None
            self.scheduling_step = None
        if self.accelerator is not None:
            if self.teacher_updatable:
                self.teacher_model, self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader = self.accelerator.prepare(self.teacher_model, self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader)
            else:
                self.teacher_model = self.teacher_model
                if self.accelerator.state.use_fp16:
                    self.teacher_model = self.teacher_model.half()
                self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader = self.accelerator.prepare(self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader)

    def __init__(self, teacher_model, student_model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        super().__init__()
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
        self.accelerator = accelerator
        self.teacher_model = None
        self.student_model = None
        self.teacher_forward_proc, self.student_forward_proc = None, None
        self.target_teacher_pairs, self.target_student_pairs = list(), list()
        self.teacher_io_dict, self.student_io_dict = dict(), dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.org_criterion, self.criterion, self.uses_teacher_output, self.extract_org_loss = None, None, None, None
        self.teacher_updatable, self.teacher_any_frozen, self.student_any_frozen = None, None, None
        self.grad_accum_step = None
        self.max_grad_norm = None
        self.scheduling_step = 0
        self.stage_grad_count = 0
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_process(self, epoch=None, **kwargs):
        clear_io_dict(self.teacher_io_dict)
        clear_io_dict(self.student_io_dict)
        self.teacher_model.eval()
        self.student_model.train()
        if self.distributed:
            self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)

    def get_teacher_output(self, sample_batch, targets, supp_dict):
        if supp_dict is None:
            supp_dict = dict()
        cached_data = supp_dict.get('cached_data', None)
        cache_file_paths = supp_dict.get('cache_file_path', None)
        teacher_outputs = None
        cached_extracted_teacher_output_dict = None
        if cached_data is not None and isinstance(cached_data, dict):
            device = sample_batch.device
            teacher_outputs = cached_data['teacher_outputs']
            cached_extracted_teacher_output_dict = cached_data['extracted_outputs']
            if device.type != 'cpu':
                teacher_outputs = change_device(teacher_outputs, device)
                cached_extracted_teacher_output_dict = change_device(cached_extracted_teacher_output_dict, device)
            if not self.teacher_updatable:
                return teacher_outputs, cached_extracted_teacher_output_dict
        if teacher_outputs is None:
            if self.teacher_updatable:
                teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)
            else:
                with torch.no_grad():
                    teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)
        if cached_extracted_teacher_output_dict is not None:
            if isinstance(self.teacher_model, SpecialModule) or check_if_wrapped(self.teacher_model) and isinstance(self.teacher_model.module, SpecialModule):
                self.teacher_io_dict.update(cached_extracted_teacher_output_dict)
                if isinstance(self.teacher_model, SpecialModule):
                    self.teacher_model.post_forward(self.teacher_io_dict)
            extracted_teacher_io_dict = extract_io_dict(self.teacher_io_dict, self.device)
            return teacher_outputs, extracted_teacher_io_dict
        teacher_io_dict4cache = copy.deepcopy(self.teacher_io_dict) if self.teacher_updatable and isinstance(cache_file_paths, (list, tuple)) is not None else None
        extracted_teacher_io_dict = extract_io_dict(self.teacher_io_dict, self.device)
        if isinstance(self.teacher_model, SpecialModule):
            self.teacher_model.post_forward(extracted_teacher_io_dict)
        update_io_dict(extracted_teacher_io_dict, extract_io_dict(self.teacher_io_dict, self.device))
        if isinstance(cache_file_paths, (list, tuple)):
            if teacher_io_dict4cache is None:
                teacher_io_dict4cache = extracted_teacher_io_dict
            cpu_device = torch.device('cpu')
            for i, (teacher_output, cache_file_path) in enumerate(zip(teacher_outputs.cpu().numpy(), cache_file_paths)):
                sub_dict = extract_sub_model_output_dict(teacher_io_dict4cache, i)
                sub_dict = tensor2numpy2tensor(sub_dict, cpu_device)
                cache_dict = {'teacher_outputs': torch.Tensor(teacher_output), 'extracted_outputs': sub_dict}
                make_parent_dirs(cache_file_path)
                torch.save(cache_dict, cache_file_path)
        return teacher_outputs, extracted_teacher_io_dict

    def forward(self, sample_batch, targets, supp_dict):
        teacher_outputs, extracted_teacher_io_dict = self.get_teacher_output(sample_batch, targets, supp_dict=supp_dict)
        student_outputs = self.student_forward_proc(self.student_model, sample_batch, targets, supp_dict)
        extracted_student_io_dict = extract_io_dict(self.student_io_dict, self.device)
        if isinstance(self.student_model, SpecialModule):
            self.student_model.post_forward(extracted_student_io_dict)
        org_loss_dict = self.extract_org_loss(self.org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output=self.uses_teacher_output, supp_dict=supp_dict)
        update_io_dict(extracted_student_io_dict, extract_io_dict(self.student_io_dict, self.device))
        output_dict = {'teacher': extracted_teacher_io_dict, 'student': extracted_student_io_dict}
        total_loss = self.criterion(output_dict, org_loss_dict, targets)
        return total_loss

    def update_params(self, loss, **kwargs):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = [p for group in self.optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None and self.scheduling_step > 0 and self.stage_grad_count % self.scheduling_step == 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                local_epoch = int(self.stage_grad_count / self.scheduling_step)
                self.lr_scheduler.step(local_epoch)
            else:
                self.lr_scheduler.step()

    def post_process(self, **kwargs):
        if self.lr_scheduler is not None and self.scheduling_step <= 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                epoch = self.lr_scheduler.last_epoch + 1
                self.lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step()
        if isinstance(self.teacher_model, SpecialModule):
            self.teacher_model.post_process()
        if isinstance(self.student_model, SpecialModule):
            self.student_model.post_process()
        if self.distributed:
            dist.barrier()

    def clean_modules(self):
        unfreeze_module_params(self.org_teacher_model)
        unfreeze_module_params(self.org_student_model)
        self.teacher_io_dict.clear()
        self.student_io_dict.clear()
        for _, module_handle in (self.target_teacher_pairs + self.target_student_pairs):
            module_handle.remove()
        self.target_teacher_pairs.clear()
        self.target_student_pairs.clear()


class MultiStagesDistillationBox(DistillationBox):

    def __init__(self, teacher_model, student_model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        stage1_config = train_config['stage1']
        super().__init__(teacher_model, student_model, data_loader_dict, stage1_config, device, device_ids, distributed, lr_factor, accelerator)
        self.train_config = train_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['num_epochs']
        self.num_epochs = sum(train_config[key]['num_epochs'] for key in train_config.keys() if key.startswith('stage'))
        self.current_epoch = 0
        logger.info('Started stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        self.clean_modules()
        self.stage_grad_count = 0
        self.stage_number += 1
        next_stage_config = self.train_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config)
        self.stage_end_epoch += next_stage_config['num_epochs']
        logger.info('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, **kwargs):
        super().post_process(**kwargs)
        self.current_epoch += 1
        if self.current_epoch == self.stage_end_epoch and self.current_epoch < self.num_epochs:
            self.advance_to_next_stage()


class TrainingBox(nn.Module):

    def setup_data_loaders(self, train_config):
        train_data_loader_config = train_config.get('train_data_loader', dict())
        if 'requires_supp' not in train_data_loader_config:
            train_data_loader_config['requires_supp'] = True
        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader = build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config], self.distributed, self.accelerator)
        if train_data_loader is not None:
            self.train_data_loader = train_data_loader
        if val_data_loader is not None:
            self.val_data_loader = val_data_loader

    def setup_model(self, model_config):
        unwrapped_org_model = self.org_model.module if check_if_wrapped(self.org_model) else self.org_model
        self.target_model_pairs.clear()
        ref_model = unwrapped_org_model
        if len(model_config) > 0 or len(model_config) == 0 and self.model is None:
            model_type = 'original'
            special_model = build_special_module(model_config, student_model=unwrapped_org_model, device=self.device, device_ids=self.device_ids, distributed=self.distributed)
            if special_model is not None:
                ref_model = special_model
                model_type = type(ref_model).__name__
            self.model = redesign_model(ref_model, model_config, 'student', model_type)
        self.model_any_frozen = len(model_config.get('frozen_modules', list())) > 0 or not model_config.get('requires_grad', True)
        self.target_model_pairs.extend(set_hooks(self.model, ref_model, model_config, self.model_io_dict))
        self.model_forward_proc = get_forward_proc_func(model_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        self.org_criterion = None if org_criterion_config is None or len(org_criterion_config) == 0 else get_single_loss(org_criterion_config)
        self.criterion = get_custom_loss(criterion_config)
        logger.info(self.criterion)
        self.uses_teacher_output = False
        self.extract_org_loss = get_func2extract_org_output(criterion_config.get('func2extract_org_loss', None))

    def setup(self, train_config):
        self.setup_data_loaders(train_config)
        model_config = train_config.get('model', dict())
        self.setup_model(model_config)
        self.setup_loss(train_config)
        if not model_config.get('requires_grad', True):
            logger.info('Freezing the whole model')
            freeze_module_params(self.model)
        any_updatable = len(get_updatable_param_names(self.model)) > 0
        model_unused_parameters = model_config.get('find_unused_parameters', self.model_any_frozen)
        self.model = wrap_model(self.model, model_config, self.device, self.device_ids, self.distributed, model_unused_parameters, any_updatable)
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_params_config = optim_config['params']
            if 'lr' in optim_params_config:
                optim_params_config['lr'] *= self.lr_factor
            module_wise_params_configs = optim_config.get('module_wise_params', list())
            if len(module_wise_params_configs) > 0:
                trainable_module_list = list()
                for module_wise_params_config in module_wise_params_configs:
                    module_wise_params_dict = dict()
                    if isinstance(module_wise_params_config.get('params', None), dict):
                        module_wise_params_dict.update(module_wise_params_config['params'])
                    if 'lr' in module_wise_params_dict:
                        module_wise_params_dict['lr'] *= self.lr_factor
                    module = get_module(self.model, module_wise_params_config['module'])
                    module_wise_params_dict['params'] = module.parameters()
                    trainable_module_list.append(module_wise_params_dict)
            else:
                trainable_module_list = nn.ModuleList([self.model])
            filters_params = optim_config.get('filters_params', True)
            self.optimizer = get_optimizer(trainable_module_list, optim_config['type'], optim_params_config, filters_params)
            self.optimizer.zero_grad()
            self.max_grad_norm = optim_config.get('max_grad_norm', None)
            self.grad_accum_step = optim_config.get('grad_accum_step', 1)
            optimizer_reset = True
        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
            self.scheduling_step = scheduler_config.get('scheduling_step', 0)
        elif optimizer_reset:
            self.lr_scheduler = None
            self.scheduling_step = None
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_data_loader, self.val_data_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_data_loader, self.val_data_loader)

    def __init__(self, model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        super().__init__()
        self.org_model = model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
        self.accelerator = accelerator
        self.model = None
        self.model_forward_proc = None
        self.target_model_pairs = list()
        self.model_io_dict = dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.org_criterion, self.criterion, self.extract_org_loss = None, None, None
        self.model_any_frozen = None
        self.grad_accum_step = None
        self.max_grad_norm = None
        self.scheduling_step = 0
        self.stage_grad_count = 0
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_process(self, epoch=None, **kwargs):
        clear_io_dict(self.model_io_dict)
        self.model.train()
        if self.distributed:
            self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)

    def forward(self, sample_batch, targets, supp_dict):
        model_outputs = self.model_forward_proc(self.model, sample_batch, targets, supp_dict)
        extracted_model_io_dict = extract_io_dict(self.model_io_dict, self.device)
        if isinstance(self.model, SpecialModule):
            self.model.post_forward(extracted_model_io_dict)
        teacher_outputs = None
        org_loss_dict = self.extract_org_loss(self.org_criterion, model_outputs, teacher_outputs, targets, uses_teacher_output=False, supp_dict=supp_dict)
        update_io_dict(extracted_model_io_dict, extract_io_dict(self.model_io_dict, self.device))
        output_dict = {'student': extracted_model_io_dict, 'teacher': dict()}
        total_loss = self.criterion(output_dict, org_loss_dict, targets)
        return total_loss

    def update_params(self, loss, **kwargs):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = [p for group in self.optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None and self.scheduling_step > 0 and self.stage_grad_count % self.scheduling_step == 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                local_epoch = int(self.stage_grad_count / self.scheduling_step)
                self.lr_scheduler.step(local_epoch)
            else:
                self.lr_scheduler.step()

    def post_process(self, **kwargs):
        if self.lr_scheduler is not None and self.scheduling_step <= 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                epoch = self.lr_scheduler.last_epoch + 1
                self.lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step()
        if isinstance(self.model, SpecialModule):
            self.model.post_process()
        if self.distributed:
            dist.barrier()

    def clean_modules(self):
        unfreeze_module_params(self.org_model)
        self.model_io_dict.clear()
        for _, module_handle in self.target_model_pairs:
            module_handle.remove()
        self.target_model_pairs.clear()


class MultiStagesTrainingBox(TrainingBox):

    def __init__(self, model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        stage1_config = train_config['stage1']
        super().__init__(model, data_loader_dict, stage1_config, device, device_ids, distributed, lr_factor, accelerator)
        self.train_config = train_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['num_epochs']
        self.num_epochs = sum(train_config[key]['num_epochs'] for key in train_config.keys() if key.startswith('stage'))
        self.current_epoch = 0
        logger.info('Started stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        self.clean_modules()
        self.stage_grad_count = 0
        self.stage_number += 1
        next_stage_config = self.train_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config)
        self.stage_end_epoch += next_stage_config['num_epochs']
        logger.info('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, **kwargs):
        super().post_process(**kwargs)
        self.current_epoch += 1
        if self.current_epoch == self.stage_end_epoch and self.current_epoch < self.num_epochs:
            self.advance_to_next_stage()


class CustomLoss(nn.Module):

    def __init__(self, criterion_config):
        super().__init__()
        term_dict = dict()
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            for loss_name, loss_config in sub_terms_config.items():
                sub_criterion_config = loss_config['criterion']
                sub_criterion = get_single_loss(sub_criterion_config, loss_config.get('params', None))
                term_dict[loss_name] = sub_criterion, loss_config['factor']
        self.term_dict = term_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')

    def __str__(self):
        desc = 'Loss = '
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for criterion, factor in self.term_dict.values()])
        return desc


def register_custom_loss(arg=None, **kwargs):

    def _register_custom_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        CUSTOM_LOSS_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_custom_loss(arg)
    return _register_custom_loss


class GeneralizedCustomLoss(CustomLoss):

    def __init__(self, criterion_config):
        super().__init__(criterion_config)
        self.org_loss_factor = criterion_config['org_term'].get('factor', None)

    def forward(self, output_dict, org_loss_dict, targets):
        loss_dict = dict()
        student_output_dict = output_dict['student']
        teacher_output_dict = output_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_output_dict, teacher_output_dict, targets)
        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.org_loss_factor is None or isinstance(self.org_loss_factor, (int, float)) and self.org_loss_factor == 0:
            return sub_total_loss
        if isinstance(self.org_loss_factor, dict):
            org_loss = sum([(self.org_loss_factor[k] * v) for k, v in org_loss_dict.items()])
            return sub_total_loss + org_loss
        return sub_total_loss + self.org_loss_factor * sum(org_loss_dict.values() if len(org_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.org_loss_factor, 'OrgLoss')] if self.org_loss_factor is not None and self.org_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc


def register_single_loss(arg=None, **kwargs):

    def _register_single_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        SINGLE_LOSS_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_single_loss(arg)
    return _register_single_loss


class OrgDictLoss(nn.Module):

    def __init__(self, single_loss, factors, **kwargs):
        super().__init__()
        self.single_loss = get_single_loss(single_loss)
        self.factor_dict = factors

    def forward(self, student_output, targets, *args, **kwargs):
        loss = 0
        for key, input_batch in student_output.items():
            factor = self.factor_dict.get(key, 1)
            loss += factor * self.single_loss(input_batch, targets, *args, **kwargs)
        return loss


class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1), torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return self.alpha * hard_loss + self.beta * self.temperature ** 2 * soft_loss


def extract_feature_map(io_dict, feature_map_config):
    io_type = feature_map_config['io']
    module_path = feature_map_config['path']
    return io_dict[module_path][io_type]


class FSPLoss(nn.Module):
    """
    "A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"
    """

    def __init__(self, fsp_pairs, **kwargs):
        super().__init__()
        self.fsp_pairs = fsp_pairs

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = adaptive_max_pool2d(first_feature_map, (target_h, target_w))
        if second_h > target_h or second_w > target_w:
            second_feature_map = adaptive_max_pool2d(second_feature_map, (target_h, target_w))
        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        fsp_loss = 0
        batch_size = None
        for pair_name, pair_config in self.fsp_pairs.items():
            student_first_feature_map = extract_feature_map(student_io_dict, pair_config['student_first'])
            student_second_feature_map = extract_feature_map(student_io_dict, pair_config['student_second'])
            student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
            teacher_first_feature_map = extract_feature_map(teacher_io_dict, pair_config['teacher_first'])
            teacher_second_feature_map = extract_feature_map(teacher_io_dict, pair_config['teacher_second'])
            teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
            factor = pair_config.get('factor', 1)
            fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
            if batch_size is None:
                batch_size = student_first_feature_map.shape[0]
        return fsp_loss / batch_size


class ATLoss(nn.Module):
    """
    "Paying More Attention to Attention: Improving the Performance of
     Convolutional Neural Networks via Attention Transfer"
    Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
    Discrepancy between Eq. (2) in the paper and the author's implementation
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23
    as partly pointed out at https://github.com/szagoruyko/attention-transfer/issues/34
    To follow the equations in the paper, use mode='paper' in place of 'code'
    """

    def __init__(self, at_pairs, mode='code', **kwargs):
        super().__init__()
        self.at_pairs = at_pairs
        self.mode = mode
        if mode not in ('code', 'paper'):
            raise ValueError('mode `{}` is not expected'.format(mode))

    @staticmethod
    def attention_transfer_paper(feature_map):
        return normalize(feature_map.pow(2).sum(1).flatten(1))

    def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer_paper(student_feature_map)
        at_teacher = self.attention_transfer_paper(teacher_feature_map)
        return torch.norm(at_student - at_teacher, dim=1).sum()

    @staticmethod
    def attention_transfer(feature_map):
        return normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_at_loss(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer(student_feature_map)
        at_teacher = self.attention_transfer(teacher_feature_map)
        return (at_student - at_teacher).pow(2).mean()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        at_loss = 0
        batch_size = None
        for pair_name, pair_config in self.at_pairs.items():
            student_feature_map = extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('factor', 1)
            if self.mode == 'paper':
                at_loss += factor * self.compute_at_loss_paper(student_feature_map, teacher_feature_map)
            else:
                at_loss += factor * self.compute_at_loss(student_feature_map, teacher_feature_map)
            if batch_size is None:
                batch_size = len(student_feature_map)
        return at_loss / batch_size if self.mode == 'paper' else at_loss


class PKTLoss(nn.Module):
    """
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    Refactored https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, eps=1e-07):
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.eps = eps

    def cosine_similarity_loss(self, student_outputs, teacher_outputs):
        norm_s = torch.sqrt(torch.sum(student_outputs ** 2, dim=1, keepdim=True))
        student_outputs = student_outputs / (norm_s + self.eps)
        student_outputs[student_outputs != student_outputs] = 0
        norm_t = torch.sqrt(torch.sum(teacher_outputs ** 2, dim=1, keepdim=True))
        teacher_outputs = teacher_outputs / (norm_t + self.eps)
        teacher_outputs[teacher_outputs != teacher_outputs] = 0
        student_similarity = torch.mm(student_outputs, student_outputs.transpose(0, 1))
        teacher_similarity = torch.mm(teacher_outputs, teacher_outputs.transpose(0, 1))
        student_similarity = (student_similarity + 1.0) / 2.0
        teacher_similarity = (teacher_similarity + 1.0) / 2.0
        student_similarity = student_similarity / torch.sum(student_similarity, dim=1, keepdim=True)
        teacher_similarity = teacher_similarity / torch.sum(teacher_similarity, dim=1, keepdim=True)
        return torch.mean(teacher_similarity * torch.log((teacher_similarity + self.eps) / (student_similarity + self.eps)))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_penultimate_outputs = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_penultimate_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        return self.cosine_similarity_loss(student_penultimate_outputs, teacher_penultimate_outputs)


class FTLoss(nn.Module):
    """
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, p=1, reduction='mean', paraphraser_path='paraphraser', translator_path='translator', **kwargs):
        super().__init__()
        self.norm_p = p
        self.paraphraser_path = paraphraser_path
        self.translator_path = translator_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        paraphraser_flat_outputs = teacher_io_dict[self.paraphraser_path]['output'].flatten(1)
        translator_flat_outputs = student_io_dict[self.translator_path]['output'].flatten(1)
        norm_paraphraser_flat_outputs = paraphraser_flat_outputs / paraphraser_flat_outputs.norm(dim=1).unsqueeze(1)
        norm_translator_flat_outputs = translator_flat_outputs / translator_flat_outputs.norm(dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_translator_flat_outputs, norm_paraphraser_flat_outputs, reduction=self.reduction)
        ft_loss = torch.norm(norm_translator_flat_outputs - norm_paraphraser_flat_outputs, self.norm_p, dim=1)
        return ft_loss.mean() if self.reduction == 'mean' else ft_loss.sum()


class AltActTransferLoss(nn.Module):
    """
    "Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"
    Refactored https://github.com/bhheo/AB_distillation/blob/master/cifar10_AB_distillation.py
    """

    def __init__(self, feature_pairs, margin, reduction, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs
        self.margin = margin
        self.reduction = reduction

    @staticmethod
    def compute_alt_act_transfer_loss(source, target, margin):
        loss = (source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() + (source - margin) ** 2 * ((source <= margin) & (target > 0)).float()
        return torch.abs(loss).sum()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        dab_loss = 0
        batch_size = None
        for pair_name, pair_config in self.feature_pairs.items():
            student_feature_map = extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('factor', 1)
            dab_loss += factor * self.compute_alt_act_transfer_loss(student_feature_map, teacher_feature_map, self.margin)
            if batch_size is None:
                batch_size = student_feature_map.shape[0]
        return dab_loss / batch_size if self.reduction == 'mean' else dab_loss


class RKDLoss(nn.Module):
    """
    "Relational Knowledge Distillation"
    Refactored https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    """

    def __init__(self, student_output_path, teacher_output_path, dist_factor, angle_factor, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.dist_factor = dist_factor
        self.angle_factor = angle_factor
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction=reduction)

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def compute_rkd_distance_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.dist_factor is None or self.dist_factor == 0:
            return 0
        with torch.no_grad():
            t_d = self.pdist(teacher_flat_outputs, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = self.pdist(student_flat_outputs, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        return self.smooth_l1_loss(d, t_d)

    def compute_rkd_angle_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.angle_factor is None or self.angle_factor == 0:
            return 0
        with torch.no_grad():
            td = teacher_flat_outputs.unsqueeze(0) - teacher_flat_outputs.unsqueeze(1)
            norm_td = normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = student_flat_outputs.unsqueeze(0) - student_flat_outputs.unsqueeze(1)
        norm_sd = normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        return self.smooth_l1_loss(s_angle, t_angle)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_flat_outputs = teacher_io_dict[self.teacher_output_path]['output'].flatten(1)
        student_flat_outputs = student_io_dict[self.student_output_path]['output'].flatten(1)
        rkd_distance_loss = self.compute_rkd_distance_loss(teacher_flat_outputs, student_flat_outputs)
        rkd_angle_loss = self.compute_rkd_angle_loss(teacher_flat_outputs, student_flat_outputs)
        return self.dist_factor * rkd_distance_loss + self.angle_factor * rkd_angle_loss


class VIDLoss(nn.Module):
    """
    "Variational Information Distillation for Knowledge Transfer"
    Referred to https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py
    """

    def __init__(self, feature_pairs, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        vid_loss = 0
        for pair_name, pair_config in self.feature_pairs.items():
            pred_mean, pred_var = extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('factor', 1)
            neg_log_prob = 0.5 * ((pred_mean - teacher_feature_map) ** 2 / pred_var + torch.log(pred_var))
            vid_loss += factor * neg_log_prob.mean()
        return vid_loss


class CCKDLoss(nn.Module):
    """
    "Correlation Congruence for Knowledge Distillation"
    Configure KDLoss in a yaml file to meet eq. (7), using GeneralizedCustomLoss
    """

    def __init__(self, student_linear_path, teacher_linear_path, kernel_params, reduction, **kwargs):
        super().__init__()
        self.student_linear_path = student_linear_path
        self.teacher_linear_path = teacher_linear_path
        self.kernel_type = kernel_params['type']
        if self.kernel_type == 'gaussian':
            self.gamma = kernel_params['gamma']
            self.max_p = kernel_params['max_p']
        elif self.kernel_type not in ('bilinear', 'gaussian'):
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))
        self.reduction = reduction

    @staticmethod
    def compute_cc_mat_by_bilinear_pool(linear_outputs):
        return torch.matmul(linear_outputs, torch.t(linear_outputs))

    def compute_cc_mat_by_gaussian_rbf(self, linear_outputs):
        row_list = list()
        for index, linear_output in enumerate(linear_outputs):
            row = 1
            right_term = torch.matmul(linear_output, torch.t(linear_outputs))
            for p in range(1, self.max_p + 1):
                left_term = (2 * self.gamma) ** p / math.factorial(p)
                row += left_term * right_term ** p
            row *= math.exp(-2 * self.gamma)
            row_list.append(row.squeeze(0))
        return torch.stack(row_list)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_path]['output']
        student_linear_outputs = student_io_dict[self.student_linear_path]['output']
        batch_size = teacher_linear_outputs.shape[0]
        if self.kernel_type == 'bilinear':
            teacher_cc = self.compute_cc_mat_by_bilinear_pool(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_bilinear_pool(student_linear_outputs)
        elif self.kernel_type == 'gaussian':
            teacher_cc = self.compute_cc_mat_by_gaussian_rbf(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_gaussian_rbf(student_linear_outputs)
        else:
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))
        cc_loss = torch.dist(student_cc, teacher_cc, 2)
        return cc_loss / batch_size ** 2 if self.reduction == 'batchmean' else cc_loss


class SPKDLoss(nn.Module):
    """
    "Similarity-Preserving Knowledge Distillation"
    """

    def __init__(self, student_output_path, teacher_output_path, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / batch_size ** 2 if self.reduction == 'batchmean' else spkd_loss


class CRDLoss(nn.Module):
    """
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py
    """

    def init_prob_alias(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        k = len(probs)
        self.probs = torch.zeros(k)
        self.alias = torch.zeros(k, dtype=torch.int64)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.probs[kk] = k * prob
            if self.probs[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.probs[large] = self.probs[large] - 1.0 + self.probs[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.probs[last_one] = 1

    def __init__(self, student_norm_module_path, student_empty_module_path, teacher_norm_module_path, input_size, output_size, num_negative_samples, num_samples, temperature=0.07, momentum=0.5, eps=1e-07):
        super().__init__()
        self.student_norm_module_path = student_norm_module_path
        self.student_empty_module_path = student_empty_module_path
        self.teacher_norm_module_path = teacher_norm_module_path
        self.eps = eps
        self.unigrams = torch.ones(output_size)
        self.num_negative_samples = num_negative_samples
        self.num_samples = num_samples
        self.register_buffer('params', torch.tensor([num_negative_samples, temperature, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(input_size / 3)
        self.register_buffer('memory_v1', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.probs, self.alias = None, None
        self.init_prob_alias(self.unigrams)

    def draw(self, n):
        """ Draw n samples from multinomial """
        k = self.alias.size(0)
        kk = torch.zeros(n, dtype=torch.long, device=self.prob.device).random_(0, k)
        prob = self.probs.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj

    def contrast_memory(self, student_embed, teacher_embed, pos_indices, contrast_idx=None):
        param_k = int(self.params[0].item())
        param_t = self.params[1].item()
        z_v1 = self.params[2].item()
        z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        batch_size = student_embed.size(0)
        output_size = self.memory_v1.size(0)
        input_size = self.memory_v1.size(1)
        if contrast_idx is None:
            contrast_idx = self.draw(batch_size * (self.num_negative_samples + 1)).view(batch_size, -1)
            contrast_idx.select(1, 0).copy_(pos_indices.data)
        weight_v1 = torch.index_select(self.memory_v1, 0, contrast_idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batch_size, param_k + 1, input_size)
        out_v2 = torch.bmm(weight_v1, teacher_embed.view(batch_size, input_size, 1))
        out_v2 = torch.exp(torch.div(out_v2, param_t))
        weight_v2 = torch.index_select(self.memory_v2, 0, contrast_idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batch_size, param_k + 1, input_size)
        out_v1 = torch.bmm(weight_v2, student_embed.view(batch_size, input_size, 1))
        out_v1 = torch.exp(torch.div(out_v1, param_t))
        if z_v1 < 0:
            self.params[2] = out_v1.mean() * output_size
            z_v1 = self.params[2].clone().detach().item()
            logger.info('normalization constant z_v1 is set to {:.1f}'.format(z_v1))
        if z_v2 < 0:
            self.params[3] = out_v2.mean() * output_size
            z_v2 = self.params[3].clone().detach().item()
            logger.info('normalization constant z_v2 is set to {:.1f}'.format(z_v2))
        out_v1 = torch.div(out_v1, z_v1).contiguous()
        out_v2 = torch.div(out_v2, z_v2).contiguous()
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, pos_indices.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(student_embed, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, pos_indices, updated_v1)
            ab_pos = torch.index_select(self.memory_v2, 0, pos_indices.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(teacher_embed, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, pos_indices, updated_v2)
        return out_v1, out_v2

    def compute_contrast_loss(self, x):
        batch_size = x.shape[0]
        m = x.size(1) - 1
        pn = 1 / float(self.num_samples)
        p_pos = x.select(1, 0)
        log_d1 = torch.div(p_pos, p_pos.add(m * pn + self.eps)).log_()
        p_neg = x.narrow(1, 1, m)
        log_d0 = torch.div(p_neg.clone().fill_(m * pn), p_neg.add(m * pn + self.eps)).log_()
        loss = -(log_d1.sum(0) + log_d0.view(-1, 1).sum(0)) / batch_size
        return loss

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        """
        pos_idx: the indices of these positive samples in the dataset, size [batch_size]
        contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
        teacher_linear_outputs = teacher_io_dict[self.teacher_norm_module_path]['output']
        student_linear_outputs = student_io_dict[self.student_norm_module_path]['output']
        supp_dict = student_io_dict[self.student_empty_module_path]['input']
        pos_idx, contrast_idx = supp_dict['pos_idx'], supp_dict.get('contrast_idx', None)
        device = student_linear_outputs.device
        pos_idx = pos_idx
        if contrast_idx is not None:
            contrast_idx = contrast_idx
        if device != self.probs.device:
            self.probs
            self.alias
            self
        out_s, out_t = self.contrast_memory(student_linear_outputs, teacher_linear_outputs, pos_idx, contrast_idx)
        student_contrast_loss = self.compute_contrast_loss(out_s)
        teacher_contrast_loss = self.compute_contrast_loss(out_t)
        loss = student_contrast_loss + teacher_contrast_loss
        return loss


class AuxSSKDLoss(nn.CrossEntropyLoss):
    """
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
    """

    def __init__(self, module_path='ss_module', module_io='output', reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.module_path = module_path
        self.module_io = module_io

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        ss_module_outputs = teacher_io_dict[self.module_path][self.module_io]
        device = ss_module_outputs.device
        batch_size = ss_module_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = torch.arange(batch_size) % 4 == 0
        aug_indices = torch.arange(batch_size) % 4 != 0
        normal_rep = ss_module_outputs[normal_indices]
        aug_rep = ss_module_outputs[aug_indices]
        normal_rep = normal_rep.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        cos_similarities = cosine_similarity(aug_rep, normal_rep, dim=1)
        targets = torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        targets = targets[:three_forth_batch_size].long()
        return super().forward(cos_similarities, targets)


class SSKDLoss(nn.Module):
    """
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
    """

    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_ss_module_path, teacher_ss_module_path, kl_temp, ss_temp, tf_temp, ss_ratio, tf_ratio, student_linear_module_io='output', teacher_linear_module_io='output', student_ss_module_io='output', teacher_ss_module_io='output', loss_weights=None, reduction='batchmean', **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        self.kl_temp = kl_temp
        self.ss_temp = ss_temp
        self.tf_temp = tf_temp
        self.ss_ratio = ss_ratio
        self.tf_ratio = tf_ratio
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_ss_module_path = student_ss_module_path
        self.student_ss_module_io = student_ss_module_io
        self.teacher_ss_module_path = teacher_ss_module_path
        self.teacher_ss_module_io = teacher_ss_module_io

    @staticmethod
    def compute_cosine_similarities(ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size):
        normal_feat = ss_module_outputs[normal_indices]
        aug_feat = ss_module_outputs[aug_indices]
        normal_feat = normal_feat.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_feat = aug_feat.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        return cosine_similarity(aug_feat, normal_feat, dim=1)

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        device = student_linear_outputs.device
        batch_size = student_linear_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = torch.arange(batch_size) % 4 == 0
        aug_indices = torch.arange(batch_size) % 4 != 0
        ce_loss = self.cross_entropy_loss(student_linear_outputs[normal_indices], targets)
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs[normal_indices] / self.kl_temp, dim=1), torch.softmax(teacher_linear_outputs[normal_indices] / self.kl_temp, dim=1))
        kl_loss *= self.kl_temp ** 2
        aug_knowledges = torch.softmax(teacher_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        aug_targets = targets.unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long()
        ranks = torch.argsort(aug_knowledges, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.tf_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_tf = torch.sort(indices)[0]
        student_ss_module_outputs = student_io_dict[self.student_ss_module_path][self.student_ss_module_io]
        teacher_ss_module_outputs = teacher_io_dict[self.teacher_ss_module_path][self.teacher_ss_module_io]
        s_cos_similarities = self.compute_cosine_similarities(student_ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = self.compute_cosine_similarities(teacher_ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = t_cos_similarities.detach()
        aug_targets = torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long()
        ranks = torch.argsort(t_cos_similarities, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.ss_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_ss = torch.sort(indices)[0]
        ss_loss = self.kldiv_loss(torch.log_softmax(s_cos_similarities[distill_index_ss] / self.ss_temp, dim=1), torch.softmax(t_cos_similarities[distill_index_ss] / self.ss_temp, dim=1))
        ss_loss *= self.ss_temp ** 2
        log_aug_outputs = torch.log_softmax(student_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        tf_loss = self.kldiv_loss(log_aug_outputs[distill_index_tf], aug_knowledges[distill_index_tf])
        tf_loss *= self.tf_temp ** 2
        total_loss = 0
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, ss_loss, tf_loss]):
            total_loss += loss_weight * loss
        return total_loss


class PADL2Loss(nn.Module):
    """
    "Prime-Aware Adaptive Distillation"
    """

    def __init__(self, student_embed_module_path, teacher_embed_module_path, student_embed_module_io='output', teacher_embed_module_io='output', module_path='var_estimator', module_io='output', eps=1e-06, reduction='sum', **kwargs):
        super().__init__()
        self.student_embed_module_path = student_embed_module_path
        self.teacher_embed_module_path = teacher_embed_module_path
        self.student_embed_module_io = student_embed_module_io
        self.teacher_embed_module_io = teacher_embed_module_io
        self.module_path = module_path
        self.module_io = module_io
        self.eps = eps
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        log_variances = student_io_dict[self.module_path][self.module_io]
        student_embed_outputs = student_io_dict[self.student_embed_module_path][self.student_embed_module_io].flatten(1)
        teacher_embed_outputs = teacher_io_dict[self.teacher_embed_module_path][self.teacher_embed_module_io].flatten(1)
        squared_losses = torch.mean((teacher_embed_outputs - student_embed_outputs) ** 2 / (self.eps + torch.exp(log_variances)) + log_variances, dim=1)
        return squared_losses.mean()


class HierarchicalContextLoss(nn.Module):
    """
    "Distilling Knowledge via Knowledge Review"
    Referred to https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, reduction='mean', kernel_sizes=None, **kwargs):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [4, 2, 1]
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.criteria = nn.MSELoss(reduction=reduction)
        self.kernel_sizes = kernel_sizes

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features, _ = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_features = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        _, _, h, _ = student_features.shape
        loss = self.criteria(student_features, teacher_features)
        weight = 1.0
        total_weight = 1.0
        for k in self.kernel_sizes:
            if k >= h:
                continue
            proc_student_features = adaptive_avg_pool2d(student_features, (k, k))
            proc_teacher_features = adaptive_avg_pool2d(teacher_features, (k, k))
            weight /= 2.0
            loss += weight * self.criteria(proc_student_features, proc_teacher_features)
            total_weight += weight
        return loss / total_weight


class RegularizationLoss(nn.Module):

    def __init__(self, module_path, io_type='output', is_from_teacher=False, p=1, **kwargs):
        super().__init__()
        self.module_path = module_path
        self.io_type = io_type
        self.is_from_teacher = is_from_teacher
        self.norm_p = p

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        io_dict = teacher_io_dict if self.is_from_teacher else student_io_dict
        z = io_dict[self.module_path][self.io_type]
        return z.norm(p=self.norm_p)


class KTALoss(nn.Module):
    """
    "Knowledge Adaptation for Efficient Semantic Segmentation"
    """

    def __init__(self, p=1, q=2, reduction='mean', knowledge_translator_path='paraphraser', feature_adapter_path='feature_adapter', **kwargs):
        super().__init__()
        self.norm_p = p
        self.norm_q = q
        self.knowledge_translator_path = knowledge_translator_path
        self.feature_adapter_path = feature_adapter_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        knowledge_translator_flat_outputs = teacher_io_dict[self.knowledge_translator_path]['output'].flatten(1)
        feature_adapter_flat_outputs = student_io_dict[self.feature_adapter_path]['output'].flatten(1)
        norm_knowledge_translator_flat_outputs = knowledge_translator_flat_outputs / knowledge_translator_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        norm_feature_adapter_flat_outputs = feature_adapter_flat_outputs / feature_adapter_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_feature_adapter_flat_outputs, norm_knowledge_translator_flat_outputs, reduction=self.reduction)
        kta_loss = torch.norm(norm_feature_adapter_flat_outputs - norm_knowledge_translator_flat_outputs, self.norm_p, dim=1)
        return kta_loss.mean() if self.reduction == 'mean' else kta_loss.sum()


class AffinityLoss(nn.Module):
    """
    "Knowledge Adaptation for Efficient Semantic Segmentation"
    """

    def __init__(self, student_module_path, teacher_module_path, student_module_io='output', teacher_module_io='output', reduction='mean', **kwargs):
        super().__init__()
        self.student_module_path = student_module_path
        self.teacher_module_path = teacher_module_path
        self.student_module_io = student_module_io
        self.teacher_module_io = teacher_module_io
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_flat_outputs = student_io_dict[self.student_module_path][self.student_module_io].flatten(2)
        teacher_flat_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io].flatten(2)
        batch_size, ch_size, hw = student_flat_outputs.shape
        student_flat_outputs = student_flat_outputs / student_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        teacher_flat_outputs = teacher_flat_outputs / teacher_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        total_squared_losses = torch.zeros(batch_size)
        for i in range(ch_size):
            total_squared_losses += ((torch.bmm(student_flat_outputs[:, i].unsqueeze(2), student_flat_outputs[:, i].unsqueeze(1)) - torch.bmm(teacher_flat_outputs[:, i].unsqueeze(2), teacher_flat_outputs[:, i].unsqueeze(1))) / hw).norm(p=2, dim=(1, 2))
        return total_squared_losses.mean() if self.reduction == 'mean' else total_squared_losses.sum()


def register_adaptation_module(arg=None, **kwargs):

    def _register_adaptation_module(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        ADAPTATION_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_adaptation_module(arg)
    return _register_adaptation_module


class ConvReg(nn.Sequential):
    """
    Convolutional regression for FitNets used in "Contrastive Representation Distillation" (CRD)
    https://github.com/HobbitLong/RepDistiller/blob/34557d27282c83d49cff08b594944cf9570512bb/models/util.py#L131-L154
    But, hyperparameters are different from the original module due to larger input images in the target datasets
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_relu=True):
        module_dict = OrderedDict()
        module_dict['conv'] = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        module_dict['bn'] = nn.BatchNorm2d(num_output_channels)
        if uses_relu:
            module_dict['relu'] = nn.ReLU(inplace=True)
        super().__init__(module_dict)


class DenseNet4Cifar(nn.Module):
    """DenseNet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, growth_rate: int=32, block_config: Tuple[int, int, int]=(12, 12, 12), num_init_features: int=64, bn_size: int=4, drop_rate: float=0, num_classes: int=10, memory_efficient: bool=False) ->None:
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) ->Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ResNet4Cifar(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock]], layers: List[int], num_classes: int=10, zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int, stride: int=1, dilate: bool=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


class WideBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, k, dropout_p, block, num_classes, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_wide_layer(block, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self._make_wide_layer(block, stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self._make_wide_layer(block, stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


class BottleneckBase(nn.Module):

    def __init__(self, encoder, decoder, compressor=None, decompressor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.decompressor = decompressor

    def forward(self, x):
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)
        if self.decompressor is not None:
            z = self.decompressor(z)
        return self.decoder(z)


MODEL_CLASS_DICT = dict()


def register_model_class(arg=None, **kwargs):

    def _register_model_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        MODEL_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_model_class(arg)
    return _register_model_class


class Bottleneck4DenseNet(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=2)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomDenseNet(nn.Module):

    def __init__(self, bottleneck, short_feature_names, org_densenet):
        super().__init__()
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_features_set = set(short_feature_names)
        if 'classifier' in short_features_set:
            short_features_set.remove('classifier')
        for child_name, child_module in org_densenet.features.named_children():
            if child_name in short_features_set:
                module_dict[child_name] = child_module
        self.features = nn.Sequential(module_dict)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = org_densenet.classifier

    def forward(self, x):
        z = self.features(x)
        z = self.relu(z)
        z = self.adaptive_avgpool(z)
        z = torch.flatten(z, 1)
        return self.classifier(z)


class Bottleneck4Inception3(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 256, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=1)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomInception3(nn.Sequential):

    def __init__(self, bottleneck, short_module_names, org_model):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        child_name_list = list()
        for child_name, child_module in org_model.named_children():
            if child_name in short_module_set:
                if len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_2b_3x3' and child_name == 'Conv2d_3b_1x1':
                    module_dict['maxpool1'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool1')
                elif len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_4a_3x3' and child_name == 'Mixed_5b':
                    module_dict['maxpool2'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool2')
                elif child_name == 'fc':
                    module_dict['adaptive_avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
                    module_dict['dropout'] = nn.Dropout()
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module
                child_name_list.append(child_name)
        super().__init__(module_dict)


class Bottleneck4ResNet(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=1)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomResNet(nn.Sequential):

    def __init__(self, bottleneck, short_module_names, org_resnet):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        for child_name, child_module in org_resnet.named_children():
            if child_name in short_module_set:
                if child_name == 'fc':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module
        super().__init__(module_dict)


class Bottleneck4SmallResNet(BottleneckBase):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """

    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False))
        decoder = nn.Sequential(nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class Bottleneck4LargeResNet(BottleneckBase):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """

    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False))
        decoder = nn.Sequential(nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, kernel_size=2, bias=False), nn.BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


def register_special_module(arg=None, **kwargs):

    def _register_special_module(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__
        SPECIAL_CLASS_DICT[key] = cls
        return cls
    if callable(arg):
        return _register_special_module(arg)
    return _register_special_module


class EmptyModule(SpecialModule):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args


class Paraphraser4FactorTransfer(nn.Module):
    """
    Paraphraser for factor transfer described in the supplementary material of
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    @staticmethod
    def make_tail_modules(num_output_channels, uses_bn):
        leaky_relu = nn.LeakyReLU(0.1)
        if uses_bn:
            return [nn.BatchNorm2d(num_output_channels), leaky_relu]
        return [leaky_relu]

    @classmethod
    def make_enc_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [nn.Conv2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding), *cls.make_tail_modules(num_output_channels, uses_bn)]

    @classmethod
    def make_dec_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [nn.ConvTranspose2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding), *cls.make_tail_modules(num_output_channels, uses_bn)]

    def __init__(self, k, num_input_channels, kernel_size=3, stride=1, padding=1, uses_bn=True):
        super().__init__()
        self.paraphrase_rate = k
        num_enc_output_channels = int(num_input_channels * k)
        self.encoder = nn.Sequential(*self.make_enc_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *self.make_enc_modules(num_input_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn), *self.make_enc_modules(num_enc_output_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn))
        self.decoder = nn.Sequential(*self.make_dec_modules(num_enc_output_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn), *self.make_dec_modules(num_enc_output_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *self.make_dec_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn))

    def forward(self, z):
        if self.training:
            return self.decoder(self.encoder(z))
        return self.encoder(z)


class Translator4FactorTransfer(nn.Sequential):
    """
    Translator for factor transfer described in the supplementary material of
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    Note that "the student translator has the same three convolution layers as the paraphraser"
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, uses_bn=True):
        super().__init__(*Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn), *Paraphraser4FactorTransfer.make_enc_modules(num_output_channels, num_output_channels, kernel_size, stride, padding, uses_bn))


def load_module_ckpt(module, map_location, ckpt_file_path):
    state_dict = torch.load(ckpt_file_path, map_location=map_location)
    if check_if_wrapped(module):
        module.module.load_state_dict(state_dict)
    else:
        module.load_state_dict(state_dict)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_module_ckpt(module, ckpt_file_path):
    if is_main_process():
        make_parent_dirs(ckpt_file_path)
    state_dict = module.module.state_dict() if check_if_wrapped(module) else module.state_dict()
    save_on_master(state_dict, ckpt_file_path)


def get_frozen_param_names(module):
    return [name for name, param in module.named_parameters() if not param.requires_grad]


def wrap_if_distributed(model, device, device_ids, distributed):
    model
    if distributed:
        any_frozen = len(get_frozen_param_names(model)) > 0
        return DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=any_frozen)
    return model


class Teacher4FactorTransfer(SpecialModule):
    """
    Teacher for factor transfer proposed in "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, teacher_model, minimal, input_module_path, paraphraser_params, paraphraser_ckpt, uses_decoder, device, device_ids, distributed, **kwargs):
        super().__init__()
        if minimal is None:
            minimal = dict()
        special_teacher_model = build_special_module(minimal, teacher_model=teacher_model)
        model_type = 'original'
        teacher_ref_model = teacher_model
        if special_teacher_model is not None:
            teacher_ref_model = special_teacher_model
            model_type = type(teacher_ref_model).__name__
        self.teacher_model = redesign_model(teacher_ref_model, minimal, 'teacher', model_type)
        self.input_module_path = input_module_path
        self.paraphraser = wrap_if_distributed(Paraphraser4FactorTransfer(**paraphraser_params), device, device_ids, distributed)
        self.ckpt_file_path = paraphraser_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(self.paraphraser, map_location, self.ckpt_file_path)
        self.uses_decoder = uses_decoder

    def forward(self, *args):
        with torch.no_grad():
            return self.teacher_model(*args)

    def post_forward(self, io_dict):
        if self.uses_decoder and not self.paraphraser.training:
            self.paraphraser.train()
        self.paraphraser(io_dict[self.input_module_path]['output'])

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.paraphraser, self.ckpt_file_path)


class Student4FactorTransfer(SpecialModule):
    """
    Student for factor transfer proposed in "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, student_model, input_module_path, translator_params, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.input_module_path = input_module_path
        self.translator = wrap_if_distributed(Translator4FactorTransfer(**translator_params), device, device_ids, distributed)

    def forward(self, *args):
        return self.student_model(*args)

    def post_forward(self, io_dict):
        self.translator(io_dict[self.input_module_path]['output'])


class Connector4DAB(SpecialModule):
    """
    Connector proposed in "Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"
    """

    @staticmethod
    def build_connector(conv_params_config, bn_params_config=None):
        module_list = [nn.Conv2d(**conv_params_config)]
        if bn_params_config is not None and len(bn_params_config) > 0:
            module_list.append(nn.BatchNorm2d(**bn_params_config))
        return nn.Sequential(*module_list)

    def __init__(self, student_model, connectors, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        io_path_pairs = list()
        self.connector_dict = nn.ModuleDict()
        for connector_key, connector_params in connectors.items():
            connector = self.build_connector(connector_params['conv_params'], connector_params.get('bn_params', None))
            self.connector_dict[connector_key] = wrap_if_distributed(connector, device, device_ids, distributed)
            io_path_pairs.append((connector_key, connector_params['io'], connector_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        for connector_key, io_type, module_path in self.io_path_pairs:
            self.connector_dict[connector_key](io_dict[module_path][io_type])


class Regressor4VID(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, eps, init_pred_var, **kwargs):
        super().__init__()
        self.regressor = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.soft_plus_param = nn.Parameter(np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(out_channels))
        self.eps = eps
        self.init_pred_var = init_pred_var

    def forward(self, student_feature_map):
        pred_mean = self.regressor(student_feature_map)
        pred_var = torch.log(1.0 + torch.exp(self.soft_plus_param)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        return pred_mean, pred_var


class VariationalDistributor4VID(SpecialModule):
    """
    "Variational Information Distillation for Knowledge Transfer"
    """

    def __init__(self, student_model, regressors, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        io_path_pairs = list()
        self.regressor_dict = nn.ModuleDict()
        for regressor_key, regressor_params in regressors.items():
            regressor = Regressor4VID(**regressor_params)
            self.regressor_dict[regressor_key] = wrap_if_distributed(regressor, device, device_ids, distributed)
            io_path_pairs.append((regressor_key, regressor_params['io'], regressor_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        for regressor_key, io_type, module_path in self.io_path_pairs:
            self.regressor_dict[regressor_key](io_dict[module_path][io_type])


class Linear4CCKD(SpecialModule):
    """
    Fully-connected layer to cope with a mismatch of feature representations of teacher and student network for
    "Correlation Congruence for Knowledge Distillation"
    """

    def __init__(self, input_module, linear_params, device, device_ids, distributed, teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        self.linear = wrap_if_distributed(nn.Linear(**linear_params), device, device_ids, distributed)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.linear(flat_outputs)


class Normalizer4CRD(nn.Module):

    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


class Linear4CRD(SpecialModule):
    """
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py
    """

    def __init__(self, input_module_path, linear_params, device, device_ids, distributed, power=2, teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.empty = nn.Sequential()
        self.input_module_path = input_module_path
        linear = nn.Linear(**linear_params)
        self.normalizer = wrap_if_distributed(Normalizer4CRD(linear, power=power), device, device_ids, distributed)

    def forward(self, x, supp_dict):
        self.empty(supp_dict)
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path]['output'], 1)
        self.normalizer(flat_outputs)


class HeadRCNN(SpecialModule):

    def __init__(self, head_rcnn, **kwargs):
        super().__init__()
        tmp_ref_model = kwargs.get('teacher_model', None)
        ref_model = kwargs.get('student_model', tmp_ref_model)
        if ref_model is None:
            raise ValueError('Either student_model or teacher_model has to be given.')
        self.transform = ref_model.transform
        self.seq = redesign_model(ref_model, head_rcnn, 'R-CNN', 'HeadRCNN')

    def forward(self, images, targets=None):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        return self.seq(images.tensors)


class SSWrapper4SSKD(SpecialModule):
    """
    Semi-supervision wrapper for "Knowledge Distillation Meets Self-Supervision"
    """

    def __init__(self, input_module, feat_dim, ss_module_ckpt, device, device_ids, distributed, freezes_ss_module=False, teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        ss_module = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim))
        self.ckpt_file_path = ss_module_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(ss_module, map_location, self.ckpt_file_path)
        self.ss_module = ss_module if is_teacher and freezes_ss_module else wrap_if_distributed(ss_module, device, device_ids, distributed)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.ss_module(flat_outputs)

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.ss_module, self.ckpt_file_path)


class VarianceBranch4PAD(SpecialModule):
    """
    Variance branch wrapper for "Prime-Aware Adaptive Distillation"
    """

    def __init__(self, student_model, input_module, feat_dim, var_estimator_ckpt, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        var_estimator = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim))
        self.ckpt_file_path = var_estimator_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(var_estimator, map_location, self.ckpt_file_path)
        self.var_estimator = wrap_if_distributed(var_estimator, device, device_ids, distributed)

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        embed_outputs = io_dict[self.input_module_path][self.input_module_io].flatten(1)
        self.var_estimator(embed_outputs)

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.var_estimator, self.ckpt_file_path)


class AttentionBasedFusion(nn.Module):
    """
    Attention based fusion module in "Distilling Knowledge via Knowledge Review"
    Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py
    """

    def __init__(self, in_channel, mid_channel, out_channel, uses_attention):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channel))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel))
        self.attention_conv = None if not uses_attention else nn.Sequential(nn.Conv2d(mid_channel * 2, 2, kernel_size=1), nn.Sigmoid())
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None, size=None):
        x = self.conv1(x)
        if self.attention_conv is not None:
            n, _, h, w = x.shape
            y = functional.interpolate(y, (size, size), mode='nearest')
            z = torch.cat([x, y], dim=1)
            z = self.attention_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        y = self.conv2(x)
        return y, x


class Student4KnowledgeReview(SpecialModule):
    """
    Student for knowledge review proposed in "Distilling Knowledge via Knowledge Review"
    Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py
    """

    def __init__(self, student_model, abfs, device, device_ids, distributed, sizes=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        if sizes is None:
            sizes = [1, 7, 14, 28, 56]
        self.sizes = sizes
        abf_list = nn.ModuleList()
        num_abfs = len(abfs)
        io_path_pairs = list()
        for idx, abf_config in enumerate(abfs):
            abf = wrap_if_distributed(AttentionBasedFusion(uses_attention=idx < num_abfs - 1, **abf_config['params']), device, device_ids, distributed)
            abf_list.append(abf)
            io_path_pairs.append((abf_config['io'], abf_config['path']))
        self.abf_modules = abf_list[::-1]
        self.io_path_pairs = io_path_pairs[::-1]

    def forward(self, *args):
        return self.student_model(*args)

    def post_forward(self, io_dict):
        feature_maps = [io_dict[module_path][io_type] for io_type, module_path in self.io_path_pairs]
        out_features, res_features = self.abf_modules[0](feature_maps[0])
        if len(self.sizes) > 1:
            for features, abf, size in zip(feature_maps[1:], self.abf_modules[1:], self.sizes[1:]):
                out_features, res_features = abf(features, res_features, size)


class Student4KTAAD(SpecialModule):
    """
    Student for knowledge translation and adaptation + affinity distillation proposed in
    "Knowledge Adaptation for Efficient Semantic Segmentation"
    """

    def __init__(self, student_model, input_module_path, feature_adapter_params, affinity_adapter_params, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.input_module_path = input_module_path
        feature_adapter = nn.Sequential(nn.Conv2d(**feature_adapter_params['conv']), nn.BatchNorm2d(**feature_adapter_params['bn']), nn.ReLU(**feature_adapter_params['relu']))
        affinity_adapter = nn.Sequential(nn.Conv2d(**affinity_adapter_params['conv']))
        self.feature_adapter = wrap_if_distributed(feature_adapter, device, device_ids, distributed)
        self.affinity_adapter = wrap_if_distributed(affinity_adapter, device, device_ids, distributed)

    def forward(self, *args):
        return self.student_model(*args)

    def post_forward(self, io_dict):
        feature_maps = io_dict[self.input_module_path]['output']
        self.feature_adapter(feature_maps)
        self.affinity_adapter(feature_maps)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bottleneck4DenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Bottleneck4Inception3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Bottleneck4LargeResNet,
     lambda: ([], {'bottleneck_channel': 4}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Bottleneck4ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Bottleneck4SmallResNet,
     lambda: ([], {'bottleneck_channel': 4}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (BottleneckBase,
     lambda: ([], {'encoder': _mock_layer(), 'decoder': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvReg,
     lambda: ([], {'num_input_channels': 4, 'num_output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseNet4Cifar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Normalizer4CRD,
     lambda: ([], {'linear': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Paraphraser4FactorTransfer,
     lambda: ([], {'k': 4, 'num_input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Regressor4VID,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'eps': 4, 'init_pred_var': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Translator4FactorTransfer,
     lambda: ([], {'num_input_channels': 4, 'num_output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WideBasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yoshitomo_matsubara_torchdistill(_paritybench_base):
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

