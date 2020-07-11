import sys
_module = sys.modules[__name__]
del sys
basic_install_test = _module
deepspeed = _module
pt = _module
deepspeed_checkpointing = _module
deepspeed_checkpointing_config = _module
deepspeed_config = _module
deepspeed_config_utils = _module
deepspeed_constants = _module
deepspeed_csr_tensor = _module
deepspeed_dataloader = _module
deepspeed_fused_lamb = _module
deepspeed_launch = _module
deepspeed_light = _module
deepspeed_lr_schedules = _module
deepspeed_run = _module
deepspeed_timer = _module
deepspeed_utils = _module
deepspeed_zero_config = _module
deepspeed_zero_optimizer = _module
fp16_optimizer = _module
fp16_unfused_optimizer = _module
loss_scaler = _module
zero_optimizer_stage1 = _module
zero_utils = _module
conf = _module
setup = _module
BingBertSquad_run_func_test = _module
BingBertSquad_test_common = _module
BingBertSquad = _module
test_e2e_squad = _module
Megatron_GPT2 = _module
run_checkpoint_test = _module
run_func_test = _module
run_perf_baseline = _module
run_perf_test = _module
test_common = _module
run_sanity_check = _module
test_model = _module
common = _module
multi_output_model = _module
simple_model = _module
test_checkpointing = _module
test_config = _module
test_csr = _module
test_dist = _module
test_ds_arguments = _module
test_ds_config = _module
test_dynamic_loss_scale = _module
test_fp16 = _module
test_multi_output_model = _module
test_run = _module

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


import torch.distributed as dist


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


import logging


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


import types


import warnings


from torch.nn.modules import Module


from torch.distributed.distributed_c10d import _get_global_rank


from torch.optim import Optimizer


from typing import Union


from typing import List


import math


import collections


from copy import deepcopy


import torch.cuda


import time


from torch._six import inf


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch.autograd import Variable


from collections import defaultdict


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


from torch.multiprocessing import Process


import numbers


import random


import numpy as np


ADAM_OPTIMIZER = 'adam'


class CSRTensor(object):
    """ Compressed Sparse Row (CSR) Tensor """

    def __init__(self, dense_tensor=None):
        self.orig_dense_tensor = dense_tensor
        if dense_tensor is not None:
            result = torch.sum(dense_tensor, dim=1)
            self.indices = result.nonzero().flatten()
            self.values = dense_tensor[self.indices]
            self.dense_size = list(dense_tensor.size())
        else:
            self.indices = None
            self.values = None
            self.dense_size = None

    @staticmethod
    def type():
        return 'deepspeed.CSRTensor'

    def to_dense(self):
        it = self.indices.unsqueeze(1)
        full_indices = torch.cat([it for _ in range(self.dense_size[1])], dim=1)
        return self.values.new_zeros(self.dense_size).scatter_add_(0, full_indices, self.values)

    def sparse_size(self):
        index_size = list(self.indices.size())
        index_size = index_size[0]
        value_size = list(self.values.size())
        value_size = value_size[0] * value_size[1]
        dense_size = self.dense_size[0] * self.dense_size[1]
        return index_size + value_size, dense_size

    def add(self, b):
        assert self.dense_size == b.dense_size
        self.indices = torch.cat([self.indices, b.indices])
        self.values = torch.cat([self.values, b.values])

    def __str__(self):
        sparse_size, dense_size = self.sparse_size()
        return 'DeepSpeed.CSRTensor(indices_size={}, values_size={}, dense_size={}, device={}, reduction_factor={})'.format(self.indices.size(), self.values.size(), self.dense_size, self.indices.get_device(), dense_size / sparse_size)

    def __repr__(self):
        return self.__str__()


LAMB_OPTIMIZER = 'lamb'


DEEPSPEED_OPTIMIZERS = [ADAM_OPTIMIZER, LAMB_OPTIMIZER]


ACT_CHKPT = 'activation_checkpointing'


ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION = 'contiguous_memory_optimization'


ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT = False


ACT_CHKPT_CPU_CHECKPOINTING = 'cpu_checkpointing'


ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT = False


ACT_CHKPT_NUMBER_CHECKPOINTS = 'number_checkpoints'


ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT = None


ACT_CHKPT_PARTITION_ACTIVATIONS = 'partition_activations'


ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT = False


ACT_CHKPT_PROFILE = 'profile'


ACT_CHKPT_PROFILE_DEFAULT = False


ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY = 'synchronize_checkpoint_boundary'


ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT = False


ACT_CHKPT_DEFAULT = {ACT_CHKPT_PARTITION_ACTIVATIONS: ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT, ACT_CHKPT_NUMBER_CHECKPOINTS: ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION: ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY: ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT, ACT_CHKPT_PROFILE: ACT_CHKPT_PROFILE_DEFAULT, ACT_CHKPT_CPU_CHECKPOINTING: ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT}


def get_scalar_param(param_dict, param_name, param_default_value):
    if param_name in param_dict.keys():
        return param_dict[param_name]
    else:
        return param_default_value


class DeepSpeedActivationCheckpointingConfig(object):

    def __init__(self, param_dict):
        super(DeepSpeedActivationCheckpointingConfig, self).__init__()
        self.partition_activations = None
        self.contiguous_memory_optimization = None
        self.cpu_checkpointing = None
        self.number_checkpoints = None
        self.synchronize_checkpoint_boundary = None
        self.profile = None
        if ACT_CHKPT in param_dict.keys():
            act_chkpt_config_dict = param_dict[ACT_CHKPT]
        else:
            act_chkpt_config_dict = ACT_CHKPT_DEFAULT
        self._initialize(act_chkpt_config_dict)
    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def _initialize(self, act_chkpt_config_dict):
        self.partition_activations = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_PARTITION_ACTIVATIONS, ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT)
        self.contiguous_memory_optimization = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT)
        self.cpu_checkpointing = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_CPU_CHECKPOINTING, ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT)
        self.number_checkpoints = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_NUMBER_CHECKPOINTS, ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT)
        self.profile = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_PROFILE, ACT_CHKPT_PROFILE_DEFAULT)
        self.synchronize_checkpoint_boundary = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT)


ZERO_FORMAT = """
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": [0|1|2],
  "zero_all_gather_size": 200
}
"""


ZERO_OPTIMIZATION = 'zero_optimization'


ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE = 'allgather_bucket_size'


ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT = 500000000


ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED = 'allgather_size'


ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS = 'allgather_partitions'


ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT = True


ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS = 'contiguous_gradients'


ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT = True


ZERO_OPTIMIZATION_DEFAULT = 0


ZERO_OPTIMIZATION_OVERLAP_COMM = 'overlap_comm'


ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT = False


ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE = 'reduce_bucket_size'


ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT = 500000000


ZERO_OPTIMIZATION_REDUCE_SCATTER = 'reduce_scatter'


ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT = True


ZERO_OPTIMIZATION_STAGE = 'stage'


ZERO_OPTIMIZATION_DISABLED = 0


ZERO_OPTIMIZATION_STAGE_DEFAULT = ZERO_OPTIMIZATION_DISABLED


class DeepSpeedZeroConfig(object):

    def __init__(self, param_dict):
        super(DeepSpeedZeroConfig, self).__init__()
        self.stage = None
        self.contiguous_gradients = None
        self.reduce_scatter = None
        self.reduce_bucket_size = None
        self.allgather_partitions = None
        self.allgather_bucket_size = None
        self.overlap_comm = None
        if ZERO_OPTIMIZATION in param_dict.keys():
            zero_config_dict = param_dict[ZERO_OPTIMIZATION]
            if type(zero_config_dict) is bool:
                zero_config_dict = self.read_zero_config_deprecated(param_dict)
        else:
            zero_config_dict = ZERO_OPTIMIZATION_DEFAULT
        self._initialize(zero_config_dict)

    def read_zero_config_deprecated(self, param_dict):
        zero_config_dict = {}
        zero_config_dict[ZERO_OPTIMIZATION_STAGE] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
        if zero_config_dict[ZERO_OPTIMIZATION_STAGE] > 0:
            zero_config_dict[ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE] = get_scalar_param(param_dict, ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED, ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)
        logging.warning('DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'.format(ZERO_FORMAT))
        return zero_config_dict
    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def _initialize(self, zero_config_dict):
        self.stage = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_STAGE, ZERO_OPTIMIZATION_STAGE_DEFAULT)
        self.contiguous_gradients = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS, ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT)
        self.reduce_bucket_size = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE, ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)
        self.reduce_scatter = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_REDUCE_SCATTER, ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)
        self.overlap_comm = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_OVERLAP_COMM, ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT)
        self.allgather_partitions = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS, ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT)
        self.allgather_bucket_size = get_scalar_param(zero_config_dict, ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE, ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)


GRADIENT_ACCUMULATION_STEPS = 'gradient_accumulation_steps'


MAX_GRAD_NORM = 'max_grad_norm'


ZERO_OPTIMIZATION_GRADIENTS = 2


MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_GRADIENTS


TENSOR_CORE_ALIGN_SIZE = 8


TRAIN_MICRO_BATCH_SIZE_PER_GPU = """
TRAIN_MICRO_BATCH_SIZE_PER_GPU is defined in this format:
"train_micro_batch_size_per_gpu": 1
"""


VOCABULARY_SIZE = 'vocabulary_size'


VOCABULARY_SIZE_DEFAULT = None


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError('Duplicate key in DeepSpeed config: %r' % (k,))
        else:
            d[k] = v
    return d


ALLGATHER_SIZE = 'allgather_size'


ALLGATHER_SIZE_DEFAULT = 500000000


def get_allgather_size(param_dict):
    return get_scalar_param(param_dict, ALLGATHER_SIZE, ALLGATHER_SIZE_DEFAULT) if get_scalar_param(param_dict, ALLGATHER_SIZE, ALLGATHER_SIZE_DEFAULT) > 0 else ALLGATHER_SIZE_DEFAULT


FP32_ALLREDUCE = 'fp32_allreduce'


FP32_ALLREDUCE_DEFAULT = False


def get_allreduce_always_fp32(param_dict):
    return get_scalar_param(param_dict, FP32_ALLREDUCE, FP32_ALLREDUCE_DEFAULT)


DISABLE_ALLGATHER = 'disable_allgather'


DISABLE_ALLGATHER_DEFAULT = False


def get_disable_allgather(param_dict):
    return get_scalar_param(param_dict, DISABLE_ALLGATHER, DISABLE_ALLGATHER_DEFAULT)


DUMP_STATE = 'dump_state'


DUMP_STATE_DEFAULT = False


def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


DELAYED_SHIFT = 'delayed_shift'


FP16 = 'fp16'


FP16_HYSTERESIS = 'hysteresis'


FP16_HYSTERESIS_DEFAULT = 2


FP16_INITIAL_SCALE_POWER = 'initial_scale_power'


FP16_INITIAL_SCALE_POWER_DEFAULT = 32


FP16_LOSS_SCALE_WINDOW = 'loss_scale_window'


FP16_LOSS_SCALE_WINDOW_DEFAULT = 1000


FP16_MIN_LOSS_SCALE = 'min_loss_scale'


FP16_MIN_LOSS_SCALE_DEFAULT = 1


INITIAL_LOSS_SCALE = 'init_scale'


MIN_LOSS_SCALE = 'min_scale'


SCALE_WINDOW = 'scale_window'


FP16_ENABLED = 'enabled'


FP16_ENABLED_DEFAULT = False


def get_fp16_enabled(param_dict):
    if FP16 in param_dict.keys():
        return get_scalar_param(param_dict[FP16], FP16_ENABLED, FP16_ENABLED_DEFAULT)
    else:
        return False


def get_dynamic_loss_scale_args(param_dict):
    loss_scale_args = None
    if get_fp16_enabled(param_dict):
        fp16_dict = param_dict[FP16]
        dynamic_loss_args = [FP16_INITIAL_SCALE_POWER, FP16_LOSS_SCALE_WINDOW, FP16_MIN_LOSS_SCALE, FP16_HYSTERESIS]
        if any(arg in list(fp16_dict.keys()) for arg in dynamic_loss_args):
            init_scale = get_scalar_param(fp16_dict, FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
            scale_window = get_scalar_param(fp16_dict, FP16_LOSS_SCALE_WINDOW, FP16_LOSS_SCALE_WINDOW_DEFAULT)
            delayed_shift = get_scalar_param(fp16_dict, FP16_HYSTERESIS, FP16_HYSTERESIS_DEFAULT)
            min_loss_scale = get_scalar_param(fp16_dict, FP16_MIN_LOSS_SCALE, FP16_MIN_LOSS_SCALE_DEFAULT)
            loss_scale_args = {INITIAL_LOSS_SCALE: 2 ** init_scale, SCALE_WINDOW: scale_window, DELAYED_SHIFT: delayed_shift, MIN_LOSS_SCALE: min_loss_scale}
    return loss_scale_args


GRADIENT_ACCUMULATION_STEPS_DEFAULT = None


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict, GRADIENT_ACCUMULATION_STEPS, GRADIENT_ACCUMULATION_STEPS_DEFAULT)


GRADIENT_CLIPPING = 'gradient_clipping'


GRADIENT_CLIPPING_DEFAULT = 0.0


OPTIMIZER = 'optimizer'


OPTIMIZER_PARAMS = 'params'


OPTIMIZER_TYPE_DEFAULT = None


TYPE = 'type'


def get_optimizer_name(param_dict):
    if OPTIMIZER in param_dict.keys() and TYPE in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][TYPE]
    else:
        return OPTIMIZER_TYPE_DEFAULT


def get_optimizer_params(param_dict):
    if get_optimizer_name(param_dict) is not None and OPTIMIZER_PARAMS in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][OPTIMIZER_PARAMS]
    else:
        return None


def get_optimizer_gradient_clipping(param_dict):
    optimizer_params = get_optimizer_params(param_dict)
    if optimizer_params is not None and MAX_GRAD_NORM in optimizer_params.keys():
        return optimizer_params[MAX_GRAD_NORM]
    else:
        return None


def get_gradient_clipping(param_dict):
    grad_clip = get_optimizer_gradient_clipping(param_dict)
    if grad_clip is not None:
        return grad_clip
    else:
        return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


def get_initial_dynamic_scale(param_dict):
    if get_fp16_enabled(param_dict):
        initial_scale_power = get_scalar_param(param_dict[FP16], FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
    else:
        initial_scale_power = FP16_INITIAL_SCALE_POWER_DEFAULT
    return 2 ** initial_scale_power


FP16_LOSS_SCALE = 'loss_scale'


FP16_LOSS_SCALE_DEFAULT = 0


def get_loss_scale(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_LOSS_SCALE, FP16_LOSS_SCALE_DEFAULT)
    else:
        return FP16_LOSS_SCALE_DEFAULT


MEMORY_BREAKDOWN = 'memory_breakdown'


MEMORY_BREAKDOWN_DEFAULT = False


def get_memory_breakdown(param_dict):
    return get_scalar_param(param_dict, MEMORY_BREAKDOWN, MEMORY_BREAKDOWN_DEFAULT)


LEGACY_FUSION = 'legacy_fusion'


LEGACY_FUSION_DEFAULT = False


def get_optimizer_legacy_fusion(param_dict):
    if OPTIMIZER in param_dict.keys() and LEGACY_FUSION in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][LEGACY_FUSION]
    else:
        return LEGACY_FUSION_DEFAULT


PRESCALE_GRADIENTS = 'prescale_gradients'


PRESCALE_GRADIENTS_DEFAULT = False


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


SCHEDULER = 'scheduler'


SCHEDULER_TYPE_DEFAULT = None


def get_scheduler_name(param_dict):
    if SCHEDULER in param_dict.keys() and TYPE in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][TYPE]
    else:
        return SCHEDULER_TYPE_DEFAULT


SCHEDULER_PARAMS = 'params'


def get_scheduler_params(param_dict):
    if get_scheduler_name(param_dict) is not None and SCHEDULER_PARAMS in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][SCHEDULER_PARAMS]
    else:
        return None


SPARSE_GRADIENTS = 'sparse_gradients'


SPARSE_GRADIENTS_DEFAULT = False


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


STEPS_PER_PRINT = 'steps_per_print'


STEPS_PER_PRINT_DEFAULT = 10


def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)


TENSORBOARD = 'tensorboard'


TENSORBOARD_ENABLED = 'enabled'


TENSORBOARD_ENABLED_DEFAULT = False


def get_tensorboard_enabled(param_dict):
    if TENSORBOARD in param_dict.keys():
        return get_scalar_param(param_dict[TENSORBOARD], TENSORBOARD_ENABLED, TENSORBOARD_ENABLED_DEFAULT)
    else:
        return False


TENSORBOARD_JOB_NAME = 'job_name'


TENSORBOARD_JOB_NAME_DEFAULT = 'DeepSpeedJobName'


def get_tensorboard_job_name(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD], TENSORBOARD_JOB_NAME, TENSORBOARD_JOB_NAME_DEFAULT)
    else:
        return TENSORBOARD_JOB_NAME_DEFAULT


TENSORBOARD_OUTPUT_PATH = 'output_path'


TENSORBOARD_OUTPUT_PATH_DEFAULT = ''


def get_tensorboard_output_path(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD], TENSORBOARD_OUTPUT_PATH, TENSORBOARD_OUTPUT_PATH_DEFAULT)
    else:
        return TENSORBOARD_OUTPUT_PATH_DEFAULT


TRAIN_BATCH_SIZE = 'train_batch_size'


TRAIN_BATCH_SIZE_DEFAULT = None


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = None


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(param_dict, TRAIN_MICRO_BATCH_SIZE_PER_GPU, TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)


WALL_CLOCK_BREAKDOWN = 'wall_clock_breakdown'


WALL_CLOCK_BREAKDOWN_DEFAULT = False


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict, WALL_CLOCK_BREAKDOWN, WALL_CLOCK_BREAKDOWN_DEFAULT)


ZERO_ALLOW_UNTESTED_OPTIMIZER = 'zero_allow_untested_optimizer'


ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT = False


def get_zero_allow_untested_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_ALLOW_UNTESTED_OPTIMIZER, ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT)


class DeepSpeedConfig(object):

    def __init__(self, json_file, mpu=None, param_dict=None):
        super(DeepSpeedConfig, self).__init__()
        if param_dict is None:
            self._param_dict = json.load(open(json_file, 'r'), object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            self._param_dict = param_dict
        try:
            self.global_rank = torch.distributed.get_rank()
            if mpu is None:
                self.world_size = torch.distributed.get_world_size()
            else:
                self.world_size = mpu.get_data_parallel_world_size()
        except:
            self.global_rank = 0
            self.world_size = 1
        self._initialize_params(self._param_dict)
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)
        self.disable_allgather = get_disable_allgather(param_dict)
        self.allreduce_always_fp32 = get_allreduce_always_fp32(param_dict)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)
        self.allgather_size = get_allgather_size(param_dict)
        self.zero_config = DeepSpeedZeroConfig(param_dict)
        self.zero_optimization_stage = self.zero_config.stage
        self.zero_enabled = self.zero_optimization_stage > 0
        self.activation_checkpointing_config = DeepSpeedActivationCheckpointingConfig(param_dict)
        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.fp16_enabled = get_fp16_enabled(param_dict)
        self.loss_scale = get_loss_scale(param_dict)
        self.initial_dynamic_scale = get_initial_dynamic_scale(param_dict)
        self.dynamic_loss_scale_args = get_dynamic_loss_scale_args(param_dict)
        self.optimizer_name = get_optimizer_name(param_dict)
        if self.optimizer_name is not None and self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS:
            self.optimizer_name = self.optimizer_name.lower()
        self.optimizer_params = get_optimizer_params(param_dict)
        self.optimizer_legacy_fusion = get_optimizer_legacy_fusion(param_dict)
        self.zero_allow_untested_optimizer = get_zero_allow_untested_optimizer(param_dict)
        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)
        self.wall_clock_breakdown = get_wall_clock_breakdown(param_dict)
        self.memory_breakdown = get_memory_breakdown(param_dict)
        self.tensorboard_enabled = get_tensorboard_enabled(param_dict)
        self.tensorboard_output_path = get_tensorboard_output_path(param_dict)
        self.tensorboard_job_name = get_tensorboard_job_name(param_dict)

    def _batch_assertion(self):
        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps
        assert train_batch > 0, f'Train batch size: {train_batch} has to be greater than 0'
        assert micro_batch > 0, f'Micro batch size per gpu: {micro_batch} has to be greater than 0'
        assert grad_acc > 0, f'Gradient accumulation steps: {grad_acc} has to be greater than 0'
        assert train_batch == micro_batch * grad_acc * self.world_size, f'Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size{train_batch} != {micro_batch} * {grad_acc} * {self.world_size}'

    def _set_batch_related_parameters(self):
        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.world_size
            self.gradient_accumulation_steps = grad_acc
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // self.world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch
        elif micro_batch is not None and grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.world_size
            self.train_batch_size = train_batch_size
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.world_size
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.world_size
            self.gradient_accumulation_steps = 1
        else:
            assert False, 'Either train_batch_size or micro_batch_per_gpu needs to be provided'
        None

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()
        self._do_warning_check()

    def print(self, name):
        None
        for arg in sorted(vars(self)):
            if arg != '_param_dict':
                dots = '.' * (29 - len(arg))
                None
        None

    def _do_error_check(self):
        if self.zero_enabled:
            assert self.fp16_enabled, 'DeepSpeedConfig: ZeRO is only supported if fp16 is enabled'
            assert self.zero_optimization_stage <= MAX_STAGE_ZERO_OPTIMIZATION, 'DeepSpeedConfig: Maximum supported ZeRO stage is {}'.format(MAX_STAGE_ZERO_OPTIMIZATION)
        assert self.train_micro_batch_size_per_gpu, 'DeepSpeedConfig: {} is not defined'.format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)
        assert self.gradient_accumulation_steps, 'DeepSpeedConfig: {} is not defined'.format(GRADIENT_ACCUMULATION_STEPS)

    def _do_warning_check(self):
        fp16_enabled = self.fp16_enabled or self.zero_enabled
        if self.gradient_clipping > 0.0 and not fp16_enabled:
            logging.warning('DeepSpeedConfig: gradient clipping enabled without FP16 enabled.')
        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logging.warning('DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization.'.format(vocabulary_size, TENSOR_CORE_ALIGN_SIZE))
        if self.optimizer_params is not None and MAX_GRAD_NORM in self.optimizer_params.keys() and self.optimizer_params[MAX_GRAD_NORM] > 0:
            if fp16_enabled:
                logging.warning('DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper'.format(MAX_GRAD_NORM, self.optimizer_params[MAX_GRAD_NORM]))
            else:
                logging.warning('DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero'.format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0


class DeepSpeedDataLoader(object):

    def __init__(self, dataset, batch_size, pin_memory, local_rank, tput_timer, collate_fn=None, num_local_io_workers=None, data_sampler=None):
        self.tput_timer = tput_timer
        self.batch_size = batch_size
        if local_rank >= 0:
            if data_sampler is None:
                data_sampler = DistributedSampler(dataset)
            device_count = 1
        else:
            if data_sampler is None:
                data_sampler = RandomSampler(dataset)
            device_count = torch.cuda.device_count()
            batch_size *= device_count
        if num_local_io_workers is None:
            num_local_io_workers = 2 * device_count
        self.num_local_io_workers = num_local_io_workers
        self.data_sampler = data_sampler
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device_count = device_count
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.len = len(self.data_sampler)
        self.data = None

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.tput_timer:
            self.tput_timer.start()
        return next(self.data)

    def _create_dataloader(self):
        if self.collate_fn is None:
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, sampler=self.data_sampler, num_workers=self.num_local_io_workers)
        else:
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, sampler=self.data_sampler, collate_fn=self.collate_fn, num_workers=self.num_local_io_workers)
        self.data = (x for x in self.dataloader)
        return self.dataloader


class DynamicLossScaler:
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.  If ``scale_window`` consecutive iterations take place without an overflow, the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait before increasing the loss scale.
    """

    def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000, min_scale=1, delayed_shift=1, consecutive_hysteresis=False):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

    def has_overflow_serial(self, params):
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        if not hasattr(self, 'min_scale'):
            self.min_scale = 1
        if not hasattr(self, 'delayed_shift'):
            self.delayed_shift = 1
        if not hasattr(self, 'cur_hysteresis'):
            self.cur_hysteresis = 1
        if not hasattr(self, 'consecutive_hysteresis'):
            self.consecutive_hysteresis = True
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class LossScaler:
    """
    Class that manages a static loss scale.  This class is intended to interact with
    :class:`FP16_Optimizer`, and should not be directly manipulated by the user.

    Use of :class:`LossScaler` is enabled via the ``static_loss_scale`` argument to
    :class:`FP16_Optimizer`'s constructor.

    Args:
        scale (float, optional, default=1.0):  The loss scale.
    """

    def __init__(self, scale=1):
        self.cur_scale = scale

    def has_overflow(self, params):
        return False

    def _has_inf_or_nan(x):
        return False

    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = torch.distributed.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        None


def flatten_dense_tensors_aligned(tensor_list, alignment, pg):
    num_elements = 0
    for tensor in tensor_list:
        num_elements = num_elements + tensor.numel()
    remaining = num_elements % alignment
    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
        num_elements = num_elements + elements_to_add
    else:
        padded_tensor_list = tensor_list
    return _flatten_dense_tensors(padded_tensor_list)


def is_model_parallel_parameter(p):
    return hasattr(p, 'model_parallel') and p.model_parallel


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


pg_correctness_test = False


def see_memory_usage(message):
    return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() == 0:
        return
    None
    None
    None
    None
    None
    None


def split_half_float_double(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor']
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


class FP16_DeepSpeedZeroOptimizer(object):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """

    def __init__(self, init_optimizer, timers, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, contiguous_gradients=True, reduce_bucket_size=500000000, allgather_bucket_size=5000000000, dp_process_group=None, reduce_scatter=True, overlap_comm=False, mpu=None, clip_grad=0.0):
        if dist.get_rank() == 0:
            None
            None
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.timers = timers
        self.reduce_scatter = reduce_scatter
        self.overlap_comm = overlap_comm
        self.dp_process_group = dp_process_group
        self.partition_count = dist.get_world_size(group=self.dp_process_group)
        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()
        self.overflow = False
        self.clip_grad = clip_grad
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.parallel_partitioned_fp16_groups = []
        self.single_partition_of_fp32_groups = []
        self.params_not_in_partition = []
        self.params_in_partition = []
        self.first_offset = []
        self.partition_size = []
        partition_id = dist.get_rank(group=self.dp_process_group)
        self.all_reduce_print = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            see_memory_usage(f'Before moving param group {i} to CPU')
            move_to_cpu(self.fp16_groups[i])
            see_memory_usage(f'After moving param group {i} to CPU')
            self.fp16_groups_flat.append(flatten_dense_tensors_aligned(self.fp16_groups[i], dist.get_world_size(group=self.dp_process_group), self.dp_process_group))
            see_memory_usage(f'After flattening and moving param group {i} to GPU')
            if dist.get_rank(group=self.dp_process_group) == 0:
                see_memory_usage(f'After Flattening and after emptying param group {i} cache')
                None
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            data_parallel_partitions = self.get_data_parallel_partitions(self.fp16_groups_flat[i])
            self.parallel_partitioned_fp16_groups.append(data_parallel_partitions)
            self.single_partition_of_fp32_groups.append(self.parallel_partitioned_fp16_groups[i][partition_id].clone().float().detach())
            self.single_partition_of_fp32_groups[i].requires_grad = True
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]
            partition_size = len(self.fp16_groups_flat[i]) / dist.get_world_size(group=self.dp_process_group)
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(self.fp16_groups[i], partition_size, partition_id)
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.reduction_stream = torch.cuda.Stream()
        self.callback_queued = False
        self.param_dict = {}
        self.is_param_in_current_partition = {}
        self.contiguous_gradients = contiguous_gradients
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.param_id = {}
        count = 0
        for i, params_group in enumerate(self.fp16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                count = count + 1
        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True
        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False
        self.param_to_partition_ids = {}
        self.is_partition_reduced = {}
        self.remaining_grads_in_partition = {}
        self.total_grads_in_partition = {}
        self.is_grad_computed = {}
        self.grad_partition_insertion_offset = {}
        self.grad_start_offset = {}
        self.averaged_gradients = {}
        self.first_param_index_in_partition = {}
        self.initialize_gradient_partitioning_data_structures()
        self.reset_partition_gradient_structures()
        self.create_reduce_and_remove_grad_hooks()
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)
            self.dynamic_loss_scale = True
        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0
        see_memory_usage('Before initializing optimizer states')
        self.initialize_optimizer_states()
        see_memory_usage('After initializing optimizer states')
        if dist.get_rank() == 0:
            None
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f'After initializing ZeRO optimizer')
            None

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
            single_grad_partition = torch.zeros(int(self.partition_size[i]), dtype=self.single_partition_of_fp32_groups[i].dtype)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
        self.optimizer.step()
        for group in self.single_partition_of_fp32_groups:
            group.grad = None
        return

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):
        total_partitions = dist.get_world_size(group=self.dp_process_group)
        for i, param_group in enumerate(self.fp16_groups):
            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}
            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(i, param_group, partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f'In ipg_epilogue before reduce_ipg_grads', 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f'In ipg_epilogue after reduce_ipg_grads', 0)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False
        if self.overlap_comm:
            torch.cuda.synchronize()
        for i, _ in enumerate(self.fp16_groups):
            self.averaged_gradients[i] = self.get_flat_partition(self.params_in_partition[i], self.first_offset[i], self.partition_size[i], return_tensor_list=True)
        self._release_ipg_buffers()
        see_memory_usage(f'End ipg_epilogue')

    def reset_partition_gradient_structures(self):
        total_partitions = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.fp16_groups):
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]
                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):

        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1
        partition_size = self.partition_size[i]
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)
        current_index = 0
        first_offset = 0
        for param in param_group:
            param_size = param.numel()
            param_id = self.get_param_id(param)
            if current_index >= start_index and current_index < end_index:
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)
                self.is_grad_computed[i][partition_id][param_id] = False
                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0
            elif start_index > current_index and start_index < current_index + param_size:
                assert first_offset == 0, 'This can happen either zero or only once as this must be the first tensor in the partition'
                first_offset = start_index - current_index
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)
                self.is_grad_computed[i][partition_id][param_id] = False
                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset
            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def create_reduce_and_remove_grad_hooks(self):
        self.grad_accs = []
        for i, param_group in enumerate(self.fp16_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)
                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)
                    wrapper(param, i)

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = 100.0 * elem_count // self.reduce_bucket_size
        see_memory_usage(f'{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}')

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage('In ipg_remove_grads before reduce_ipg_grads', param.numel())
            self.reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage('In ipg_remove_grads after reduce_ipg_grads', param.numel())
        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, f'The parameter {param_id} has already been reduced.             Gradient computed twice for this partition.             Multiple gradient reduction is currently not supported'
        if self.contiguous_gradients:
            new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
            new_grad_tensor.copy_(param.grad.view(-1))
            param.grad.data = new_grad_tensor.data.view_as(param.grad)
        self.elements_in_ipg_bucket += param.numel()
        self.grads_in_ipg_bucket.append(param.grad)
        self.params_in_ipg_bucket.append((i, param, param_id))
        self.report_ipg_memory_usage('End ipg_remove_grads', 0)

    def print_rank_0(self, message):
        if dist.get_rank() == 0:
            None

    def average_tensor(self, tensor):
        if self.overlap_comm:
            torch.cuda.synchronize()
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            if not self.reduce_scatter:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))
                dist.all_reduce(tensor, group=self.dp_process_group)
                return
            rank_and_offsets = []
            curr_size = 0
            prev_id = -1
            for i, param, param_id in self.params_in_ipg_bucket:
                partition_ids = self.param_to_partition_ids[i][param_id]
                partition_size = self.partition_size[i]
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]
                    if idx == len(partition_ids_w_offsets) - 1:
                        numel = param.numel() - offset
                    else:
                        numel = partition_ids_w_offsets[idx + 1][1] - offset
                    if partition_id == prev_id:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = prev_pid, prev_size, prev_numel + numel
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                    curr_size += numel
                    prev_id = partition_id
            tensor.div_(dist.get_world_size(group=self.dp_process_group))
            async_handles = []
            for dst, bucket_offset, numel in rank_and_offsets:
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                dst_rank = _get_global_rank(self.dp_process_group, dst)
                async_handle = dist.reduce(grad_slice, dst=dst_rank, group=self.dp_process_group, async_op=True)
                async_handles.append(async_handle)
            for handle in async_handles:
                handle.wait()

    def copy_grads_in_partition(self, param):
        if self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()
            see_memory_usage(f'before copying {total_size} gradients into partition')
            self.grads_in_partition = torch.empty(int(total_size), dtype=torch.half)
            see_memory_usage(f'after copying {total_size} gradients into partition')
        new_grad_tensor = self.grads_in_partition.narrow(0, self.grads_in_partition_offset, param.numel())
        new_grad_tensor.copy_(param.grad.view(-1))
        param.grad.data = new_grad_tensor.data.view_as(param.grad)
        self.grads_in_partition_offset += param.numel()

    def reduce_ipg_grads(self):
        if self.overlap_comm:
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()
        if self.contiguous_gradients:
            self.average_tensor(self.ipg_buffer[self.ipg_index])
        else:
            self.buffered_reduce_fallback(None, self.grads_in_ipg_bucket, elements_per_buffer=self.elements_in_ipg_bucket)
        with torch.cuda.stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:
                self.params_already_reduced[param_id] = True
                if not self.is_param_in_current_partition[param_id]:
                    if self.overlap_comm and self.contiguous_gradients is False:
                        if self.previous_reduced_grads is None:
                            self.previous_reduced_grads = []
                        self.previous_reduced_grads.append(param)
                    else:
                        param.grad = None
                elif self.contiguous_gradients:
                    self.copy_grads_in_partition(param)
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):

        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True
        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = _flatten_dense_tensors(tensors)

        def print_func():
            None
        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):

        def get_reducable_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(total_elements - start, self.partition_size[i] - self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0, int(start), int(num_elements))
            elif num_elements == total_elements:
                return grad.clone()
            else:
                return grad.clone().contiguous().view(-1).narrow(0, int(start), int(num_elements))
        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducable_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def sequential_execution(self, function, message, group=None):
        if group is None:
            group = self.dp_process_group
        if dist.get_rank(group=group) == 0:
            None
        for id in range(dist.get_world_size(group=group)):
            if id == dist.get_rank(group=group):
                function()
            dist.barrier(group=group)

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zero_like(param)

    def allreduce_bucket(self, bucket, allreduce_always_fp32=False, rank=None, log=None):
        rank = None
        tensor = flatten(bucket)
        tensor_to_allreduce = tensor
        if pg_correctness_test:
            allreduce_always_fp32 = True
        if allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()
        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))
        if rank is None:
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = _get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)
        if allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)
        return tensor

    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        if self.overlap_comm:
            torch.cuda.synchronize()
            if self.previous_reduced_grads is not None:
                for param in self.previous_reduced_grads:
                    param.grad = None
                self.previous_reduced_grads = None
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000, rank=None, log=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    def buffered_reduce_fallback(self, rank, grads, elements_per_buffer=500000000, log=None):
        split_buckets = split_half_float_double(grads)
        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer, rank=rank, log=log)

    def get_data_parallel_partitions(self, tensor):
        partitions = []
        dp = dist.get_world_size(group=self.dp_process_group)
        dp_id = dist.get_rank(group=self.dp_process_group)
        total_num_elements = tensor.numel()
        base_size = total_num_elements // dp
        remaining = total_num_elements % dp
        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)
        current_index = 0
        first_offset = 0
        for tensor in tensor_list:
            tensor_size = tensor.numel()
            if current_index >= start_index and current_index < end_index:
                params_in_partition.append(tensor)
            elif start_index > current_index and start_index < current_index + tensor_size:
                params_in_partition.append(tensor)
                assert first_offset == 0, 'This can happen either zero or only once as this must be the first tensor in the partition'
                first_offset = start_index - current_index
            else:
                params_not_in_partition.append(tensor)
            current_index = current_index + tensor_size
        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None:
            torch.distributed.all_reduce(tensor=tensor, op=op)
        else:
            torch.distributed.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = torch.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            for g, p in zip(gradients, params):
                if is_model_parallel_parameter(p) or self.model_parallel_rank == 0:
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm_cuda = torch.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.SUM)
            total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1
        return total_norm

    def get_flat_partition(self, tensor_list, first_offset, partition_size, return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                continue
            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
            if num_elements > partition_size - current_size:
                num_elements = partition_size - current_size
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
            else:
                flat_tensor_list.append(tensor)
            current_size = current_size + num_elements
        if current_size < partition_size:
            flat_tensor_list.append(torch.zeros(int(partition_size - current_size), dtype=tensor_list[0].dtype, device=tensor_list[0].device))
        if return_tensor_list:
            return flat_tensor_list
        return _flatten_dense_tensors(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        see_memory_usage(f'In step before checking overflow')
        self.check_overflow()
        timers = self.timers
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad()
            see_memory_usage('After overflow after clearing gradients')
            None
            timers('optimizer_step').start()
            timers('optimizer_step').stop()
            timers('optimizer_allgather').start()
            timers('optimizer_allgather').stop()
            return
        norm_groups = []
        single_partition_grad_groups = []
        skip = False
        partition_id = dist.get_rank(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):
            norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))
            self.free_grad_in_param_list(self.params_not_in_partition[i])
            if partition_id == dist.get_world_size(group=self.dp_process_group) - 1:
                single_grad_partition = flatten_dense_tensors_aligned(self.averaged_gradients[i], int(self.partition_size[i]), self.dp_process_group)
            else:
                single_grad_partition = _flatten_dense_tensors(self.averaged_gradients[i])
            assert single_grad_partition.numel() == self.partition_size[i], 'averaged gradients have different number of elements that partition size {} {} {} {}'.format(single_grad_partition.numel(), self.partition_size[i], i, partition_id)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
            self.free_grad_in_param_list(self.params_in_partition[i])
            self.averaged_gradients[i] = None
            single_partition_grad_groups.append(single_grad_partition)
        self.unscale_and_clip_grads(single_partition_grad_groups, norm_groups)
        timers('optimizer_step').start()
        self.optimizer.step()
        for group in self.single_partition_of_fp32_groups:
            group.grad = None
        for i in range(len(norm_groups)):
            for fp16_partitions, fp32_partition in zip(self.parallel_partitioned_fp16_groups, self.single_partition_of_fp32_groups):
                fp16_partitions[partition_id].data.copy_(fp32_partition.data)
        timers('optimizer_step').stop()
        timers('optimizer_allgather').start()
        for group_id, partitioned_params in enumerate(self.parallel_partitioned_fp16_groups):
            dp_world_size = dist.get_world_size(group=self.dp_process_group)
            num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // self.allgather_bucket_size)
            if num_shards == 1:
                dist.all_gather(partitioned_params, partitioned_params[partition_id], group=self.dp_process_group)
            else:
                shard_size = partitioned_params[partition_id].numel() // num_shards
                num_elements = shard_size
                for shard_id in range(num_shards):
                    if shard_id == num_shards - 1:
                        if shard_size * num_shards >= partitioned_params[partition_id].numel():
                            break
                        else:
                            num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size
                    shard_list = []
                    for dp_id in range(dp_world_size):
                        curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements)
                        shard_list.append(curr_shard)
                    dist.all_gather(shard_list, shard_list[partition_id], group=self.dp_process_group)
        timers('optimizer_allgather').stop()
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        see_memory_usage('After zero_optimizer step')
        return

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm ** 2.0
        total_norm = math.sqrt(total_norm)
        combined_scale = self.loss_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.loss_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale
        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1.0 / combined_scale)
            else:
                grad.data.mul_(1.0 / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True
        return False

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.fp16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.has_overflow_partitioned_grads_serial()
            overflow_gpu = torch.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX, group=self.dp_process_group)
        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)
            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.ByteTensor([overflow])
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=torch.distributed.ReduceOp.MAX)
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                if dist.get_rank() == 0 and j is not None:
                    _handle_overflow(cpu_sum, x, j)
                return True
            return False

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(self.reduce_bucket_size, dtype=torch.half)
            self.ipg_buffer.append(buf_0)
            if self.overlap_comm:
                buf_1 = torch.empty(self.reduce_bucket_size, dtype=torch.half)
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value
    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['single_partition_of_fp32_groups'] = self.single_partition_of_fp32_groups
        state_dict['partition_count'] = self.partition_count
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if 'partition_count' in state_dict and state_dict['partition_count'] == self.partition_count:
            for current, saved in zip(self.single_partition_of_fp32_groups, state_dict['single_partition_of_fp32_groups']):
                current.data.copy_(saved.data)
        else:
            partition_id = dist.get_rank(group=self.dp_process_group)
            for fp16_partitions, fp32_partition in zip(self.parallel_partitioned_fp16_groups, self.single_partition_of_fp32_groups):
                fp32_partition.data.copy_(fp16_partitions[partition_id].data)


class CheckOverflow(object):
    """Checks for overflow in gradient across parallel process"""

    def __init__(self, param_groups=None, mpu=None, zero_reduce_scatter=False):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)

    def check_using_norm(self, norm_group):
        overflow = -1 in norm_group
        if self.mpu is not None:
            overflow_gpu = torch.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
            overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        if param_groups is None:
            params = self.params
        else:
            assert param_groups is not None, 'self.params and param_groups both cannot be none'
            for group in param_groups:
                for param in group:
                    params.append(param)
        return self.has_overflow(params)

    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params):
        overflow = self.has_overflow_serial(params)
        overflow_gpu = torch.ByteTensor([overflow])
        if self.zero_reduce_scatter:
            torch.distributed.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX, group=torch.distributed.group.WORLD)
        elif self.mpu is not None:
            torch.distributed.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    @staticmethod
    def _has_inf_or_nan(x, i):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                _handle_overflow(cpu_sum, x, i)
                return True
            return False


def _initialize_parameter_parallel_groups(parameter_parallel_size=None):
    data_parallel_size = int(dist.get_world_size())
    if parameter_parallel_size is None:
        parameter_parallel_size = int(data_parallel_size)
    None
    assert data_parallel_size % parameter_parallel_size == 0, 'world size should be divisible by parameter parallel size'
    rank = dist.get_rank()
    my_group = None
    for i in range(dist.get_world_size() // parameter_parallel_size):
        ranks = range(i * parameter_parallel_size, (i + 1) * parameter_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            my_group = group
    return my_group


def _single_range_check(current_index, start_index, end_index, tensor_size):
    offset = 0
    if current_index >= start_index and current_index < end_index:
        return True, offset
    elif start_index > current_index and start_index < current_index + tensor_size:
        offset = start_index - current_index
        return True, offset
    else:
        return False, offset


def _range_check(current_index, element_intervals, tensor_size):
    results = []
    for comm_idx, interval in enumerate(element_intervals):
        start_index, end_index = interval
        contained, offset = _single_range_check(current_index, start_index, end_index, tensor_size)
        if contained:
            results.append((contained, offset, comm_idx))
    if len(results) == 0:
        return [(False, 0, -1)]
    return results


def pprint(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        None


def flatten_dense_tensors_sub_partition_aligned(tensor_list, dp, max_elements_per_comm, pg):
    num_elements = 0
    for tensor in tensor_list:
        num_elements = num_elements + tensor.numel()
    pprint('Total number of elements in model: {}, max elements per com: {}'.format(num_elements, max_elements_per_comm))
    max_elements_per_comm = min(max_elements_per_comm, num_elements)
    sub_partition_size = int(max_elements_per_comm // dp)
    alignment = sub_partition_size
    remaining = int(num_elements % alignment)
    elements_to_add = 0
    if remaining:
        elements_to_add = alignment - remaining
        pprint('adding pad tensor for alignment, {} + {}->{}'.format(num_elements, elements_to_add, num_elements + elements_to_add))
    else:
        padded_tensor_list = tensor_list
    num_partitions = int((num_elements + elements_to_add) // sub_partition_size)
    assert (num_elements + elements_to_add) % sub_partition_size == 0, 'num elements should be aligned by sub partition size'
    num_comm_intervals = int(num_partitions // dp)
    partition_remaining = int(num_partitions % dp)
    pprint('num_comm_intervals={}, partition_remaining={}'.format(num_comm_intervals, partition_remaining))
    if partition_remaining != 0:
        pprint('adding pad tensor and/or extra sub partition')
        num_comm_intervals += 1
        aligned_comm_elements = num_comm_intervals * sub_partition_size * dp
        elements_to_add = aligned_comm_elements - num_elements
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
        pprint('adding pad tensor and/or extra sub partition, {} + {}->{}'.format(num_elements, elements_to_add, num_elements + elements_to_add))
        num_elements += elements_to_add
    elif elements_to_add > 0:
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
        num_elements += elements_to_add
    if pg is None or dist.get_rank(group=pg) == 0:
        None
    padded_num_elems = 0
    for p in padded_tensor_list:
        padded_num_elems += p.numel()
    assert num_elements == padded_num_elems, '{} != {}, rank={}'.format(num_elements, padded_num_elems, dist.get_rank())
    return _flatten_dense_tensors(padded_tensor_list)


def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        for p in parameters:
            if mpu is not None:
                if mpu.get_model_parallel_rank() == 0 or hasattr(p, 'model_parallel') and p.model_parallel:
                    param_norm = p.grad.data.float().norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1
    return total_norm


class FP16_DeepSpeedZeroOptimizer_Stage1(object):
    """
    FP16_DeepSpeedZeroOptimizer_Stage1 designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    This version aligns with stage-1 in the paper above.
    """

    def __init__(self, init_optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, dp_process_group=None, partition_size=None, mpu=None, all_gather_partitions=True, allgather_size=500000000, clip_grad=0.0, max_elements_per_comm=500000000.0):
        if dp_process_group is not None and partition_size is not None:
            raise ValueError('Cannot specify both dp_process_group and partition size')
        if dp_process_group is None:
            dp_process_group = _initialize_parameter_parallel_groups(partition_size)
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.verbose = verbose
        self.dp_process_group = dp_process_group
        self.all_gather_partitions = all_gather_partitions
        self.allgather_size = allgather_size
        self.max_elements_per_comm = max_elements_per_comm
        None
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.parallel_sub_partitioned_fp16_groups = []
        self.parallel_comm_sub_partitioned_fp16_groups = []
        self.local_sub_partitions_of_fp32_groups = []
        self.params_not_local = []
        self.params_in_rank_sub_partitions = []
        self.params_in_rank_sub_partitions_offsets = []
        self.sub_partition_sizes = []
        self.num_comm_intervals_per_group = []
        local_rank = dist.get_rank(group=self.dp_process_group)
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            self.fp16_groups_flat.append(flatten_dense_tensors_sub_partition_aligned(tensor_list=self.fp16_groups[i], dp=dist.get_world_size(group=self.dp_process_group), max_elements_per_comm=self.max_elements_per_comm, pg=self.dp_process_group))
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            comm_partitions, dp_sub_partitions, element_intervals, sub_partition_size, num_comm_intervals = self.get_data_parallel_sub_partitions(tensor=self.fp16_groups_flat[i], max_elements_per_comm=self.max_elements_per_comm, world_size=dist.get_world_size(group=self.dp_process_group), dp_process_group=self.dp_process_group)
            self.parallel_comm_sub_partitioned_fp16_groups.append(comm_partitions)
            self.parallel_sub_partitioned_fp16_groups.append(dp_sub_partitions)
            self.sub_partition_sizes.append(sub_partition_size)
            self.num_comm_intervals_per_group.append(num_comm_intervals)
            local_sub_partitions = []
            for sub_partition in self.parallel_sub_partitioned_fp16_groups[i][local_rank]:
                fp32_sub_partition = sub_partition.clone().float().detach()
                fp32_sub_partition.requires_grad = True
                local_sub_partitions.append(fp32_sub_partition)
            self.local_sub_partitions_of_fp32_groups.append(local_sub_partitions)
            param_group['params'] = self.local_sub_partitions_of_fp32_groups[i]
            params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local = self.get_all_sub_partition_info(tensor_list=self.fp16_groups[i], all_element_intervals=element_intervals, local_rank=local_rank, world_size=dist.get_world_size(group=self.dp_process_group))
            self.params_in_rank_sub_partitions.append(params_in_rank_sub_partition)
            self.params_not_local.append(params_not_local)
            self.params_in_rank_sub_partitions_offsets.append(params_in_rank_sub_partitions_offsets)
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)
            self.dynamic_loss_scale = True
        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0
        self.mpu = mpu
        self.clip_grad = clip_grad
        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu, zero_reduce_scatter=True)

    @staticmethod
    def get_data_parallel_sub_partitions(tensor, max_elements_per_comm, world_size, dp_process_group=None):
        total_num_elements = tensor.numel()
        max_elements_per_comm = min(total_num_elements, max_elements_per_comm)
        sub_partition_size = int(max_elements_per_comm // world_size)
        num_sub_partitions = int(total_num_elements // sub_partition_size)
        assert total_num_elements % sub_partition_size == 0, '{} % {} != 0'.format(total_num_elements, sub_partition_size)
        num_comm_intervals = int(num_sub_partitions // world_size)
        assert num_sub_partitions % world_size == 0, '{} % {} != 0'.format(num_sub_partitions, world_size)
        if not dist.is_initialized() or dist.get_rank(group=dp_process_group) == 0:
            None
            None
            None
            None
            None
            None
            None
            None
        comm_partitions = []
        for _ in range(num_comm_intervals):
            comm_partitions.append([])
        start = 0
        comm_id = 0
        element_intervals = defaultdict(list)
        for idx in range(num_sub_partitions):
            rank_id = idx % world_size
            sub_partition = tensor.narrow(0, start, sub_partition_size)
            element_intervals[rank_id].append((start, start + sub_partition_size))
            comm_partitions[comm_id].append(sub_partition)
            start = start + sub_partition_size
            if rank_id == world_size - 1:
                comm_id += 1
        sub_partitions = []
        for _ in range(world_size):
            sub_partitions.append([])
        for comm_id, partitions in enumerate(comm_partitions):
            for rank_id, partition in enumerate(partitions):
                sub_partitions[rank_id].append(partition)
        return comm_partitions, sub_partitions, element_intervals, sub_partition_size, num_comm_intervals

    @staticmethod
    def get_all_sub_partition_info(tensor_list, all_element_intervals, local_rank, world_size):
        params_not_local = []
        params_in_rank_sub_partition = []
        params_in_rank_sub_partitions_offsets = []
        for rank in range(world_size):
            params_in_local_sub_partition = []
            local_sub_partition_offsets = []
            comm_tensor_list = []
            comm_offset_list = []
            current_index = 0
            prev_comm_idx = 0
            for iii, tensor in enumerate(tensor_list):
                tensor_size = tensor.numel()
                results_list = _range_check(current_index, all_element_intervals[rank], tensor_size)
                for contained, offset, comm_idx in results_list:
                    if contained:
                        if prev_comm_idx != comm_idx:
                            params_in_local_sub_partition.append(comm_tensor_list)
                            comm_tensor_list = []
                            local_sub_partition_offsets.append(comm_offset_list)
                            comm_offset_list = []
                        comm_tensor_list.append(tensor)
                        comm_offset_list.append(offset)
                        prev_comm_idx = comm_idx
                    elif rank == local_rank:
                        params_not_local.append(tensor)
                current_index = current_index + tensor_size
            params_in_local_sub_partition.append(comm_tensor_list)
            local_sub_partition_offsets.append(comm_offset_list)
            params_in_rank_sub_partition.append(params_in_local_sub_partition)
            params_in_rank_sub_partitions_offsets.append(local_sub_partition_offsets)
        return params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local

    @staticmethod
    def get_flat_sub_partitions(comm_tensor_list, comm_param_offsets, sub_partition_size, dtype, num_comm_intervals=None, default_device=None, return_partition_params=False):
        partition_params = []
        final_param_offsets = []
        flat_sub_partitions = []
        for tensor_list, param_offsets in zip(comm_tensor_list, comm_param_offsets):
            flat_tensor_list = []
            current_size = 0
            my_offsets = []
            my_params = []
            if dtype is None:
                dtype = tensor_list[0].dtype
            for i, tensor in enumerate(tensor_list):
                if tensor.grad is None:
                    tensor.grad = torch.zeros(tensor.size(), dtype=tensor.dtype, device=tensor.device)
                param = tensor
                tensor = tensor.grad
                num_elements = tensor.numel()
                tensor_offset = 0
                if i == 0 and param_offsets[i] > 0:
                    tensor_offset = param_offsets[i]
                    num_elements = num_elements - tensor_offset
                if num_elements > sub_partition_size - current_size:
                    num_elements = sub_partition_size - current_size
                if tensor_offset > 0 or num_elements < tensor.numel():
                    flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
                else:
                    flat_tensor_list.append(tensor)
                my_params.append(param)
                my_offsets.append((current_size, num_elements))
                current_size = current_size + num_elements
            if current_size < sub_partition_size:
                my_offsets.append((None, None))
                my_params.append(None)
                if len(tensor_list) == 0:
                    assert default_device != None
                    flat_tensor_list.append(torch.zeros(int(sub_partition_size - current_size), dtype=dtype, device=default_device))
                else:
                    flat_tensor_list.append(torch.zeros(int(sub_partition_size - current_size), dtype=dtype, device=tensor_list[0].device))
            partition_params.append(my_params)
            final_param_offsets.append(my_offsets)
            assert len(flat_tensor_list) == len(my_offsets), '{} {}'.format(len(flat_tensor_list), len(my_offsets))
            flat_sub_partitions.append(_flatten_dense_tensors(flat_tensor_list))
        if num_comm_intervals is not None and len(flat_sub_partitions) < num_comm_intervals:
            device = flat_sub_partitions[0].device
            for _ in range(num_comm_intervals - len(flat_sub_partitions)):
                flat_sub_partitions.append(torch.zeros(int(sub_partition_size), dtype=dtype, device=device))
                partition_params.append([None])
                final_param_offsets.append([(None, None)])
        if return_partition_params:
            assert len(flat_sub_partitions) == len(partition_params)
            assert len(partition_params) == len(final_param_offsets), '{} {}'.format(len(partition_params), len(final_param_offsets))
            return flat_sub_partitions, partition_params, final_param_offsets
        return flat_sub_partitions

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            if isinstance(p, list):
                for _p in p:
                    _p.grad = None
            else:
                p.grad = None

    def reduce_scatter_gradients(self, postscale_gradients, gradient_predivide_factor, gradient_average):
        world_size = dist.get_world_size(group=self.dp_process_group)
        local_rank = dist.get_rank(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):
            partition_param_map = {}
            param_partition_map = {}
            my_params = set()
            num_comm_intervals = self.num_comm_intervals_per_group[i]
            all_sub_partitions = []
            for rank in range(world_size):
                grad_sub_partitions, partition_params, param_offsets = self.get_flat_sub_partitions(comm_tensor_list=self.params_in_rank_sub_partitions[i][rank], comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i][rank], sub_partition_size=self.sub_partition_sizes[i], dtype=torch.half, num_comm_intervals=self.num_comm_intervals_per_group[i], default_device='cuda', return_partition_params=True)
                all_sub_partitions.append(grad_sub_partitions)
                for comm_idx, part in enumerate(grad_sub_partitions):
                    partition_param_map[part] = partition_params[comm_idx], param_offsets[comm_idx]
                for comm_idx, params in enumerate(partition_params):
                    for pidx, p in enumerate(params):
                        if rank == local_rank:
                            my_params.add(p)
                        if p in param_partition_map:
                            param_partition_map[p].append(grad_sub_partitions[comm_idx])
                        else:
                            param_partition_map[p] = [grad_sub_partitions[comm_idx]]
                assert len(grad_sub_partitions) == num_comm_intervals
            if not postscale_gradients:
                raise NotImplementedError('pre-scale_gradients is not implemented')
            all_comm_partitions = []
            for comm_idx in range(num_comm_intervals):
                single_comm_all_partitions = []
                for rank in range(world_size):
                    single_comm_all_partitions.append(all_sub_partitions[rank][comm_idx])
                dist.reduce_scatter(output=single_comm_all_partitions[local_rank], input_list=single_comm_all_partitions, group=self.dp_process_group)
                if gradient_average:
                    for partition in single_comm_all_partitions:
                        partition.mul_(gradient_predivide_factor / world_size)
                all_comm_partitions.append(single_comm_all_partitions)
            for p in my_params:
                partitions = param_partition_map[p]
                parts = []
                for part in partitions:
                    params, offsets = partition_param_map[part]
                    found = False
                    for p_idx, _p in enumerate(params):
                        if p.__hash__() == _p.__hash__():
                            found = True
                            if offsets[p_idx][0] is not None:
                                my_part = part.narrow(0, offsets[p_idx][0], offsets[p_idx][1])
                                parts.append(my_part)
                    assert found
                if p is not None:
                    updated_grad = _unflatten_dense_tensors(torch.cat(parts), [p])
                    p.grad.copy_(updated_grad[0])

    def step(self, closure=None):
        self.overflow = self.overflow_checker.check()
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            self.zero_grad()
            if self.verbose:
                None
            return self.overflow
        norm_groups = []
        local_sub_partitions_grad_groups = []
        partition_id = dist.get_rank(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):
            norm_groups.append(get_grad_norm(group, mpu=self.mpu))
            self.free_grad_in_param_list(self.params_not_local[i])
            local_grad_sub_partitions = self.get_flat_sub_partitions(comm_tensor_list=self.params_in_rank_sub_partitions[i][partition_id], comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i][partition_id], sub_partition_size=self.sub_partition_sizes[i], dtype=self.local_sub_partitions_of_fp32_groups[i][0].dtype, num_comm_intervals=self.num_comm_intervals_per_group[i], default_device=self.local_sub_partitions_of_fp32_groups[i][0].device)
            for idx, sub_partition_param in enumerate(self.local_sub_partitions_of_fp32_groups[i]):
                sub_partition_param.grad = local_grad_sub_partitions[idx]
            self.free_grad_in_param_list(self.params_in_rank_sub_partitions[i][partition_id])
            local_sub_partitions_grad_groups.append(local_grad_sub_partitions)
        self.unscale_and_clip_grads(local_sub_partitions_grad_groups, norm_groups)
        self.optimizer.step()
        for group in self.local_sub_partitions_of_fp32_groups:
            for idx, sub_partition_param in enumerate(group):
                sub_partition_param.grad = None
        for fp16_all_sub_partitions, fp32_local_sub_partitions in zip(self.parallel_sub_partitioned_fp16_groups, self.local_sub_partitions_of_fp32_groups):
            for local_sub_partition_param_fp16, local_sub_partition_param_fp32 in zip(fp16_all_sub_partitions[partition_id], fp32_local_sub_partitions):
                local_sub_partition_param_fp16.data.copy_(local_sub_partition_param_fp32.data)
        for fp16_all_sub_partitions in self.parallel_comm_sub_partitioned_fp16_groups:
            for comm_id, sub_partitions in enumerate(fp16_all_sub_partitions):
                dist.all_gather(sub_partitions, sub_partitions[partition_id], group=self.dp_process_group)
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm ** 2.0
        total_norm = math.sqrt(total_norm)
        combined_scale = self.loss_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.loss_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale
        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1.0 / combined_scale)
            else:
                grad.data.mul_(1.0 / combined_scale)

    def backward(self, loss, retain_graph=False):
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value
    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['local_sub_partitions_of_fp32_groups'] = self.local_sub_partitions_of_fp32_groups
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for curr_group, saved_group in zip(self.local_sub_partitions_of_fp32_groups, state_dict['local_sub_partitions_of_fp32_groups']):
            for curr_param, saved_param in zip(curr_group, saved_group):
                curr_param.data.copy_(saved_param.data)


def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        for p in parameters:
            if mpu is not None:
                if mpu.get_model_parallel_rank() == 0 or hasattr(p, 'model_parallel') and p.model_parallel:
                    try:
                        param_norm = float(torch.norm(p, norm_type, dtype=torch.float32))
                    except TypeError as err:
                        param_norm = float(torch.norm(p.float(), norm_type))
                    total_norm += param_norm ** norm_type
            else:
                try:
                    param_norm = float(torch.norm(p, norm_type, dtype=torch.float32))
                except TypeError as err:
                    param_norm = float(torch.norm(p.float(), norm_type))
                total_norm += param_norm ** norm_type
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1
    return total_norm


class FP16_Optimizer(object):
    """
   FP16 Optimizer for training fp16 models. Handles loss scaling.

   For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self, init_optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, initial_dynamic_scale=2 ** 32, dynamic_loss_args=None, verbose=True, mpu=None, clip_grad=0.0, fused_adam_legacy=False):
        self.fused_adam_legacy = fused_adam_legacy
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            self.fp32_groups_flat[i].requires_grad = True
            param_group['params'] = [self.fp32_groups_flat[i]]
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2
            if dynamic_loss_args is None:
                self.cur_scale = initial_dynamic_scale
                self.scale_window = 1000
                self.min_loss_scale = 1
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose
        self.clip_grad = clip_grad
        self.norm_type = 2
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.mpu = None
        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu)

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step_fused_adam(self, closure=None):
        """
        Not supporting closure.
        """
        grads_groups_flat = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads_groups_flat.append(_flatten_dense_tensors([(torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad) for p in group]))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))
        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                None
            return self.overflow
        combined_scale = self.unscale_and_clip_grads(grads_groups_flat, norm_groups, apply_scale=False)
        self.optimizer.step(grads=[[g] for g in grads_groups_flat], output_params=[[p] for p in self.fp16_groups_flat], scale=combined_scale, grad_norms=norm_groups)
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        return self.overflow

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        if self.fused_adam_legacy:
            return self.step_fused_adam()
        grads_groups_flat = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype
            grads_groups_flat.append(_flatten_dense_tensors([(torch.zeros(p.size(), dtype=data_type, device=p.device) if p.grad is None else p.grad) for p in group]))
            self.fp32_groups_flat[i].grad = grads_groups_flat[i]
            norm_groups.append(get_grad_norm(self.fp32_groups_flat, mpu=self.mpu))
        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                None
            return self.overflow
        self.unscale_and_clip_grads(grads_groups_flat, norm_groups)
        self.optimizer.step()
        for group in self.fp32_groups_flat:
            group.grad = None
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data.copy_(q.data)
        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups, apply_scale=True):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm ** 2.0
        total_norm = math.sqrt(total_norm)
        combined_scale = self.cur_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.cur_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale
        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1.0 / combined_scale)
        return combined_scale

    def backward(self, loss):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = loss.float() * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    None
                    None
            else:
                stable_interval = self.cur_iter - self.last_overflow_iter - 1
                if stable_interval > 0 and stable_interval % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        None
                        None
        elif skip:
            None
            None
        self.cur_iter += 1
        return

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict['clip_grad'] = self.clip_grad
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.clip_grad = state_dict['clip_grad']
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
            current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)


class FP16_UnfusedOptimizer(object):
    """
    FP16 Optimizer without weight fusion to support LAMB optimizer

    For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self, init_optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, mpu=None, clip_grad=0.0, fused_lamb_legacy=False):
        self.fused_lamb_legacy = fused_lamb_legacy
        if torch.distributed.get_rank() == 0:
            logging.info(f'Fused Lamb Legacy : {self.fused_lamb_legacy} ')
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp32_groups = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            fp32_group = [p.clone().float().detach() for p in param_group['params']]
            for p in fp32_group:
                p.requires_grad = True
            self.fp32_groups.append(fp32_group)
            param_group['params'] = self.fp32_groups[i]
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2.0
            if dynamic_loss_args is None:
                self.cur_scale = 1.0 * 2 ** 16
                self.scale_window = 1000
                self.min_loss_scale = 0.25
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose
        self.clip_grad = clip_grad
        self.norm_type = 2
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.mpu = None
        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu)

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step_fused_lamb(self, closure=None):
        """
        Not supporting closure.
        """
        grads_groups_flat = []
        grads_groups = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads = [(torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad) for p in group]
            grads_groups.append(grads)
            grads_groups_flat.append(_flatten_dense_tensors(grads))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))
        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                None
            return self.overflow
        combined_scale = self.unscale_and_clip_grads(norm_groups, apply_scale=False)
        self.optimizer.step(grads=grads_groups, output_params=self.fp16_groups, scale=combined_scale)
        return self.overflow

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        if self.fused_lamb_legacy:
            return self.step_fused_lamb()
        self.overflow = self.overflow_checker.check()
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                None
            return self.overflow
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            norm_groups.append(get_grad_norm(group, mpu=self.mpu))
            for fp32_param, fp16_param in zip(self.fp32_groups[i], self.fp16_groups[i]):
                if fp16_param.grad is None:
                    fp32_param.grad = torch.zeros(fp16_param.size(), dtype=fp32_param.dtype, device=fp32_param.device)
                else:
                    fp32_param.grad = fp16_param.grad
        self.unscale_and_clip_grads(norm_groups)
        self.optimizer.step()
        for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
            for fp32_param, fp16_param in zip(fp32_group, fp16_group):
                fp32_param.grad = None
                fp16_param.data.copy_(fp32_param.data)
        return self.overflow

    def unscale_and_clip_grads(self, norm_groups, apply_scale=True):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm ** 2.0
        total_norm = math.sqrt(total_norm)
        combined_scale = self.cur_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.cur_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale
        if apply_scale:
            for group in self.fp32_groups:
                for param in group:
                    if param.grad is not None:
                        param.grad.data.mul_(1.0 / combined_scale)
        return combined_scale

    def backward(self, loss):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = loss.float() * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    None
                    None
            else:
                stable_interval = self.cur_iter - self.last_overflow_iter - 1
                if stable_interval > 0 and stable_interval % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        None
                        None
        elif skip:
            None
            None
        self.cur_iter += 1
        return

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups'] = self.fp32_groups
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for current_group, saved_group in zip(self.fp32_groups, state_dict['fp32_groups']):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)


class FusedLamb(torch.optim.Optimizer):
    """Implements LAMB algorithm. Currently GPU-only.  Requires DeepSpeed adapted Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    For usage example please see, TODO DeepSpeed Tutorial

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    https://arxiv.org/abs/1904.00962


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 0.999), eps=1e-08, eps_inside_sqrt=False, weight_decay=0.0, max_grad_norm=0.0, max_coeff=10.0, min_coeff=0.01, amsgrad=False):
        global fused_lamb_cuda
        fused_lamb_cuda = importlib.import_module('fused_lamb_cuda')
        if amsgrad:
            raise RuntimeError('FusedLamb does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm, max_coeff=max_coeff, min_coeff=min_coeff)
        super(FusedLamb, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.lamb_coeffs = []

    def step(self, closure=None, grads=None, output_params=None, scale=1.0, grad_norms=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()
        if grads is None:
            grads_group = [None] * len(self.param_groups)
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads
        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params
        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)
        del self.lamb_coeffs[:]
        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(self.param_groups, grads_group, output_params_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group['params'])
            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]
            bias_correction = 1 if group['bias_correction'] else 0
            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group, output_params_this_group, grad_norm_group):
                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    clip = (grad_norm / scale + 1e-06) / group['max_grad_norm']
                    if clip > 1:
                        combined_scale = clip * scale
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']
                state['step'] += 1
                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                lamb_coeff = fused_lamb_cuda.lamb(p.data, out_p, exp_avg, exp_avg_sq, grad, group['lr'], beta1, beta2, max_coeff, min_coeff, group['eps'], combined_scale, state['step'], self.eps_mode, bias_correction, group['weight_decay'])
                self.lamb_coeffs.append(lamb_coeff)
        return loss

    def get_lamb_coeffs(self):
        lamb_coeffs = [lamb_coeff.item() for lamb_coeff in self.lamb_coeffs]
        return lamb_coeffs


MEMORY_OPT_ALLREDUCE_SIZE = 500000000


ROUTE_EVAL = 'eval'


ROUTE_PREDICT = 'predict'


ROUTE_TRAIN = 'train'


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            None
    else:
        None


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""


    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += time.time() - self.start_time
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self.elapsed_
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = 'mem_allocated: {:.4f} GB'.format(torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = 'max_mem_allocated: {:.4f} GB'.format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = 'cache_allocated: {:.4f} GB'.format(torch.cuda.memory_cached() / (1024 * 1024 * 1024))
        max_cache = 'max_cache_allocated: {:.4f} GB'.format(torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return ' | {} | {} | {} | {}'.format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if memory_breakdown:
            string += self.memory_usage()
        print_rank_0(string)


TORCH_DISTRIBUTED_DEFAULT_PORT = '29500'


class ThroughputTimer:

    def __init__(self, batch_size, num_workers, start_step=2, steps_per_output=50, monitor_memory=True, logging_fn=None):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = 1
        self.num_workers = num_workers
        self.start_step = start_step
        self.epoch_count = 0
        self.local_step_count = 0
        self.total_step_count = 0
        self.total_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            self.logging = logging.info
        self.initialized = False

    def update_epoch_count(self):
        self.epoch_count += 1
        self.local_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        self._init_timer()
        self.started = True
        if self.total_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()

    def stop(self, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.total_step_count += 1
        self.local_step_count += 1
        if self.total_step_count > self.start_step:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            if self.local_step_count % self.steps_per_output == 0:
                if report_speed:
                    self.logging('{}/{}, SamplesPerSec={}'.format(self.epoch_count, self.local_step_count, self.avg_samples_per_sec()))
                if self.monitor_memory:
                    virt_mem = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    self.logging('{}/{}, vm percent: {}, swap percent: {}'.format(self.epoch_count, self.local_step_count, virt_mem.percent, swap.percent))

    def avg_samples_per_sec(self):
        if self.total_step_count > 0:
            samples_per_step = self.batch_size * self.num_workers
            total_step_offset = self.total_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            return samples_per_step / avg_time_per_step
        return float('-inf')


ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1


def print_configuration(args, name):
    None
    for arg in sorted(vars(args)):
        dots = '.' * (29 - len(arg))
        None


def split_half_float_double_csr(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor', CSRTensor.type()]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


class MultiOutputModel(torch.nn.Module):

    def __init__(self, hidden_dim, weight_value):
        super(MultiOutputModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear.weight.data.fill_(weight_value)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        losses = []
        for x, y in zip(inputs, targets):
            hidden_dim = self.linear(x)
            loss = self.cross_entropy_loss(hidden_dim, y)
            losses.append(loss)
        return tuple(losses)

