import sys
_module = sys.modules[__name__]
del sys
mesa = _module
custom_bn = _module
custom_conv = _module
custom_fc = _module
custom_gelu = _module
custom_layer_norm = _module
custom_matmul = _module
custom_quant = _module
custom_relu = _module
custom_softmax = _module
packbit = _module
policy = _module
setup = _module

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


import torch.nn.functional as F


import logging


import math


from torch.utils import cpp_extension


def SyncBatchNorm_backward(saved_input, weight, mean, invstd, count_tensor, process_group, needs_input_grad, grad_output):
    if not grad_output.is_contiguous(memory_format=torch.channels_last):
        grad_output = grad_output.contiguous()
    grad_input = grad_weight = grad_bias = None
    sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, True, needs_input_grad[0], needs_input_grad[1])
    if True:
        num_channels = sum_dy.shape[0]
        combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
        torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
        sum_dy, sum_dy_xmu = torch.split(combined, num_channels)
        grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor)
    return grad_input, grad_weight, grad_bias


def SyncBatchNorm_forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
    if not input.is_contiguous(memory_format=torch.channels_last):
        input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    size = int(input.numel() // input.size(1))
    if size == 1 and world_size < 2:
        raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
    mean, invstd = torch.batch_norm_stats(input, eps)
    count = torch.full((1,), input.numel() // input.size(1), dtype=mean.dtype, device=mean.device)
    num_channels = input.shape[1]
    combined = torch.cat([mean, invstd, count], dim=0)
    combined_list = [torch.empty_like(combined) for k in range(world_size)]
    dist.all_gather(combined_list, combined, process_group, async_op=False)
    combined = torch.stack(combined_list, dim=0)
    mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
    mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all.view(-1))
    self.process_group = process_group
    out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
    return out


class batchnorm2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps, clip_val, level, iteration, ema_decay, quant_groups, shift):
        if need_sync:
            output = SyncBatchNorm_forward(ctx, input, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, average_factor, process_group, world_size)
        else:
            output, save_mean, save_var, reverse = native.batch_norm_forward(input, weight, bias, mean, var, training, average_factor, eps)
            if training:
                ctx.bn_parameter = weight, bias, mean, var, save_mean, save_var, reverse, eps
                custom_quant.Quant.forward(ctx, input, clip_val, level, iteration, ema_decay, quant_groups, shift)
        if training:
            ctx.need_sync = need_sync
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.need_sync:
            grad_output, grad_bn_weight, grad_bn_bias = SyncBatchNorm_backward(input, bn_weight, bn_mean, bn_invstd, bn_count_all, bn_process_group, ctx.needs_input_grad[7:9], grad_output)
        else:
            weight, bias, running_mean, running_var, save_mean, save_var, reverse, eps = ctx.bn_parameter
            input = custom_quant.Quant.restore(ctx)
            grad_input, grad_weight, grad_bias = native.batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, 0, reverse)
            ctx.bn_input = None
            ctx.bn_parameter = None
        ctx.need_sync = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def bn_pre_forward(self, input):
    self._check_input_dim(input)
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum
    if self.training and self.track_running_stats:
        if self.num_batches_tracked is not None:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum
    if self.training:
        bn_training = True
    else:
        bn_training = self.running_mean is None and self.running_var is None
    assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
    assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
    running_mean = self.running_mean if not self.training or self.track_running_stats else None
    running_var = self.running_var if not self.training or self.track_running_stats else None
    need_sync = bn_training and input.is_cuda and hasattr(self, 'process_group')
    process_group = None
    world_size = 1
    if need_sync:
        process_group = torch.distributed.group.WORLD
        if self.process_group:
            process_group = self.process_group
        try:
            world_size = torch.distributed.get_world_size(process_group)
        except AssertionError:
            world_size = 1
        need_sync = world_size > 1
    if need_sync:
        if not self.ddp_gpu_size:
            raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
    return exponential_average_factor, bn_training, running_mean, running_var, need_sync, process_group, world_size


class conv2d_uniform(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, clip_val, level, iteration, ema_decay, quant_groups, shift):
        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        ctx.conv_weight = weight, bias
        ctx.hyperparameters_conv = stride, padding, dilation, groups
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = None, None, None
        weight, bias = ctx.conv_weight
        stride, padding, dilation, groups = ctx.hyperparameters_conv
        x = custom_quant.Quant.restore(ctx)
        benchmark = True
        deterministic = True
        allow_tf32 = True
        output_mask = [True, True]
        grad_output = grad_output
        x = x
        if torch.__version__ >= '1.7':
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask)
        else:
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask)
        x = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)
        ctx.conv_weight = None
        ctx.hyperparameters_conv = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None


class linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        ctx.save_for_backward(weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None
        weight, bias = ctx.saved_tensors
        input = custom_quant.Quant.restore(ctx)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class gelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        y = F.gelu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = custom_quant.Quant.restore(ctx)
        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)
        return grad_input, None, None, None, None, None, None


class layer_norm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        if x.dtype != weight.data.dtype:
            x = x
        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        if torch.__version__ >= '1.8':
            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, normalized_shape, weight, bias, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, normalized_shape, weight, bias, eps)
            ctx.layer_norm_parameters = mean, rstd, weight, bias, normalized_shape
        else:
            N = 1
            if isinstance(normalized_shape, int):
                N = normalized_shape
            elif isinstance(normalized_shape, (list, tuple)):
                for i in normalized_shape:
                    N *= i
            else:
                raise RuntimeError('type of normalized_shape'.format(type(normalized_shape)))
            M = x.nelement() // N
            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, weight, bias, M, N, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, weight, bias, M, N, eps)
            ctx.layer_norm_parameters = mean, rstd, weight, M, N
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None
        grad_output = grad_output.contiguous()
        x = custom_quant.Quant.restore(ctx)
        output_mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]
        if torch.__version__ >= '1.8':
            mean, rstd, weight, bias, normalized_shape = ctx.layer_norm_parameters
            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cuda(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
            else:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cpu(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
        else:
            mean, rstd, weight, M, N = ctx.layer_norm_parameters
            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cuda(grad_output, x, mean, rstd, weight, M, N, output_mask)
            else:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cpu(grad_output, x, mean, rstd, weight, M, N, output_mask)
        ctx.layer_norm_parameters = None
        return grad_input, None, grad_weight, grad_bias, None, None, None, None, None, None, None


class matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1, input2, clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None, clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None):
        custom_quant.Quant.forward(ctx, input1, clip_val1, level1, iteration1, ema_decay1, quant_groups1, shift1, '_1')
        custom_quant.Quant.forward(ctx, input2, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2, '_2')
        output = input1.matmul(input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None
        input1 = custom_quant.Quant.restore(ctx, '_1')
        input2 = custom_quant.Quant.restore(ctx, '_2')
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1.transpose(-2, -1).matmul(grad_output)
        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None


class MatMul(nn.Module):

    def __init__(self, args=None, logger=None, quant_groups=1):
        super(MatMul, self).__init__()
        self.quant1 = custom_quant.quantization(tag='matmul-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='matmul-2', quant_groups=quant_groups)
        self.tag = 'matmul'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x1, x2):
        if self.quant1.enable and self.quant2.enable and self.training:
            y = matmul.apply(x1, x2, self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift, self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift)
        else:
            y = torch.matmul(x1, x2)
        return y


def depack_group(x, groups, input_shape):
    """
    Reshaping activations to their original shape
    """
    if len(input_shape) == 3:
        B, N, C = input_shape
        x = x.reshape(groups, B, N, C // groups).permute(1, 2, 0, 3).reshape(B, N, C)
    elif len(input_shape) == 2:
        B, C = input_shape
        x = x.reshape(groups, B, C // groups).permute(1, 0, 2).reshape(B, C)
    else:
        B, H, N, D = input_shape
        if groups < H:
            x = x.reshape(groups, -1, B, N, D).reshape(H, B, N, D).permute(1, 0, 2, 3)
        else:
            x = x.reshape(groups, B, N, D).permute(1, 0, 2, 3)
    return x.contiguous()


def pack_group(x, groups):
    """
    Reshaping activations for quantization
    """
    input_shape = x.shape
    if len(input_shape) == 3:
        B, N, C = input_shape
        x = x.reshape(B, N, groups, C // groups).permute(2, 0, 1, 3).reshape(groups, -1)
    elif len(input_shape) == 2:
        B, C = input_shape
        x = x.reshape(B, groups, C // groups).permute(1, 0, 2).reshape(groups, -1)
    else:
        assert len(input_shape) == 4
        B, H, N, D = input_shape
        if groups < H:
            x = x.permute(1, 0, 2, 3).reshape(groups, -1, B, N, D).reshape(groups, -1)
        else:
            x = x.permute(1, 0, 2, 3).reshape(groups, -1)
    return x.contiguous()


def update_clip_val_shift(input, clip_val, shift, iteration, ema_decay, level):
    """
    Update quantization parameters: clip_val and shift. 
    """
    max_value = torch.amax(input, 1)
    min_value = torch.amin(input, 1)
    clip_range = max_value - min_value
    if iteration == 0:
        clip_val.data = clip_range
        shift.data = min_value
    else:
        clip_val.sub_((1 - ema_decay) * (clip_val - clip_range))
        shift.sub_((1 - ema_decay) * (shift - min_value))
    iteration.add_(1)


class Quant(object):

    def __init__(self, args=None, logger=None, enable=False, tag='fm', quant_groups=1):
        assert isinstance(self, nn.Module)
        if type(quant_groups) is tuple:
            quant_groups = quant_groups[0]
        self.enable = enable
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.quant_groups = quant_groups
        self.level = 256
        self.tag = tag
        self.index = -1
        self.ema_decay = 0.9
        self.clip_val = nn.Parameter(torch.Tensor([1.0] * quant_groups))
        self.shift = nn.Parameter(torch.Tensor([0.0] * quant_groups))
        self.args = args
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()
        self.logger = logger
        self.requires_grad = False


        class logger_wrapper(object):

            def info(self, string):
                None
        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            elif args is None:
                self.logger = logger_wrapper()
            else:
                logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
                self.logger = logging.getLogger(logger_root + __name__)
        self.verbose = self.logger.info
        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None:
                self.level = int(2 ** args.fm_bit)
            if hasattr(args, 'fm_level') and args.fm_level is not None:
                self.level = args.fm_level
            if hasattr(args, 'fm_stable'):
                self.stable = args.fm_stable
            if hasattr(args, 'fm_correlate'):
                self.correlate = args.fm_correlate
            if hasattr(args, 'fm_enable'):
                self.enable = self.enable or args.fm_enable
            if hasattr(args, 'fm_nno'):
                self.non_negative_only = self.non_negative_only and args.nno
            if hasattr(args, 'fm_half_range'):
                self.non_negative_only = self.non_negative_only and args.fm_half_range
            self.verbose('index({})-level({})-quant_groups({})'.format(self.index, self.level, self.quant_groups))
        self.items = ['clip_val', 'level', 'stable', 'ema_decay', 'quant_groups']
        self.clip_val.requires_grad = False
        self.shift.requires_grad = False

    def __str__(self):
        """
        For logging
        """
        if hasattr(self, 'repr'):
            string = self.repr
        if hasattr(self, 'string'):
            string = self.string + string
        string = string + '-index({})-tag({})'.format(self.index, self.tag)
        if self.enable:
            for item in self.items:
                if hasattr(self, item):
                    value = getattr(self, item)
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            value = value.item()
                        else:
                            continue
                    string = string + '-{}({})'.format(item, value)
        if hasattr(self, 'norm'):
            string += '\n\t-' + str(self.norm)
        return string

    def update_quantization_parameter(self, **parameters):
        """
        Setup compressed layers before training according to the custom policy.
        """
        feedback = dict()
        index = self.index
        if 'index' in parameters:
            if isinstance(parameters['index'], int):
                index = parameters['index']
            elif isinstance(parameters['index'], dict) and self.tag in parameters['index']:
                index = parameters['index'][self.tag]
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))
        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or isinstance(by_index, str) and by_index != 'all':
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.logger.warning('unexpect string in by_index: {}'.format(by_index))
            if by_index == 'all' or self.index in by_index:
                if 'by_tag' in parameters and self.tag in parameters['by_tag'] or 'by_tag' not in parameters:
                    for k, v in list(parameters.items()):
                        if hasattr(self, '{}'.format(k)):
                            if isinstance(getattr(self, k), bool):
                                v = False if v in ['False', 'false', False] else True
                            elif isinstance(getattr(self, k), int):
                                v = int(v)
                            elif isinstance(getattr(self, k), float):
                                v = float(v)
                            elif isinstance(getattr(self, k), str):
                                v = v.replace("'", '').replace('"', '')
                                if 'same' in v:
                                    v = v.replace('same', str(self.index))
                                elif 'last' in v:
                                    v = v.replace('last', str(self.index - 1))
                            if isinstance(getattr(self, k), torch.Tensor):
                                with torch.no_grad():
                                    if getattr(self, 'progressive', False):
                                        if 'lsq' in self.args.keyword or '{}_lsq'.format(self.tag) in self.args.keyword:
                                            if k in ['level_num']:
                                                v = float(v)
                                                v = v if v > 0 else self.level_num.item() + v
                                                assert v > 1.9, 'level_num should be at least 2'
                                                scale = (v - 1) / (self.level_num.item() - 1)
                                                self.clip_val.mul_(scale)
                                                self.verbose('update {}_clip_val to {} for index {}'.format(self.tag, self.clip_val, self.index))
                                                if 'reset_momentum_list' in feedback:
                                                    feedback['reset_momentum_list'].append(self.clip_val)
                                                else:
                                                    feedback['reset_momentum_list'] = [self.clip_val]
                                    getattr(self, k).fill_(float(v))
                                self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                            else:
                                setattr(self, '{}'.format(k), v)
                                self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                            if self.enable:
                                assert hasattr(self, 'iteration'), 'cannot enable quantization for current layer. Likely an error in policy file'
                        if k in ['global_buffer'] and hasattr(self.args, 'global_buffer'):
                            v = str(v)
                            if isinstance(getattr(self.args, k, None), dict) and hasattr(self, v) and self.enable:
                                key = '{}-{}-{}'.format(v, self.index, self.tag)
                                self.args.global_buffer[key] = getattr(self, v)
                                self.verbose('update global_buffer (current length: {}), key: {}'.format(len(self.args.global_buffer), key))
        self.clip_val.requires_grad = False
        self.shift.requires_grad = False
        if not self.enable:
            return None
        else:
            if hasattr(self, 'quant_loss_function') and isinstance(self.quant_loss_function, str):
                qlf = self.quant_loss_function.split()
                quant_loss_function = []
                for loss_method in qlf:
                    if loss_method == 'L2':
                        quant_loss_function.append(nn.MSELoss())
                    elif loss_method == 'L1':
                        quant_loss_function.append(nn.L1Loss())
                if len(quant_loss_function) != 0:
                    self.quant_loss_function = quant_loss_function
                    self.verbose('update quant_loss_function: {} for layer(index:{}, tag:{})'.format(self.quant_loss_function, self.index, self.tag))
                else:
                    self.quant_loss_function = 'none'
            if hasattr(self, 'method'):
                assert self.method != 'none', 'quantization enable but without specific method in layer(index:{}, tag:{})'.format(self.index, self.tag)
            return feedback

    @staticmethod
    def forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift, identifier='_'):
        """
        Quantizing activations at the forward pass during training.
        """
        input_shape = x.shape
        x = pack_group(x, quant_groups)
        quant_shape = x.shape
        update_clip_val_shift(x.detach(), clip_val, shift, iteration, ema_decay, level)
        setattr(ctx, 'clip_val{}'.format(identifier), clip_val)
        setattr(ctx, 'shift{}'.format(identifier), shift)
        setattr(ctx, 'input_type{}'.format(identifier), x.dtype)
        setattr(ctx, 'input_shape{}'.format(identifier), input_shape)
        scale = (level - 1) / clip_val.abs()
        shift = shift
        x = ext_quant.pack_single_precision(x, scale, shift, 8, True)
        setattr(ctx, 'quant_shape{}'.format(identifier), quant_shape)
        setattr(ctx, 'input{}'.format(identifier), x)
        setattr(ctx, 'level{}'.format(identifier), level)

    @staticmethod
    def restore(ctx, identifier='_'):
        """
        Dequantizing activations at the backward pass during training.
        """
        input = getattr(ctx, 'input{}'.format(identifier))
        level = getattr(ctx, 'level{}'.format(identifier))
        input_shape = getattr(ctx, 'input_shape{}'.format(identifier))
        clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
        shift = getattr(ctx, 'shift{}'.format(identifier))
        quant_shape = getattr(ctx, 'quant_shape{}'.format(identifier))
        input_type = getattr(ctx, 'input_type{}'.format(identifier))
        scale = (level - 1) / clip_val.abs()
        shift = shift
        y = ext_quant.unpack_single_precision(input, 8, scale, shift, quant_shape[0], quant_shape[1])
        y = depack_group(y, quant_shape[0], input_shape)
        setattr(ctx, 'quant_shape{}'.format(identifier), None)
        setattr(ctx, 'input_type{}'.format(identifier), None)
        setattr(ctx, 'input{}'.format(identifier), None)
        setattr(ctx, 'clip_val{}'.format(identifier), None)
        setattr(ctx, 'shift{}'.format(identifier), None)
        setattr(ctx, 'input_shape{}'.format(identifier), None)
        setattr(ctx, 'level{}'.format(identifier), None)
        return y

    @staticmethod
    def backward(ctx, grad_input, identifier='_'):
        assert 1 == 0, 'Not Support'


class quantization(nn.Module, Quant):

    def __init__(self, args=None, logger=None, tag='fm', quant_groups=None):
        super(quantization, self).__init__()
        Quant.__init__(self, args=args, logger=logger, tag=tag, quant_groups=quant_groups)

    def __repr__(self):
        return self.__str__()


class relu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, inplace=False, dim=1, keep_tensor=True):
        if inplace:
            output = x.clamp_(min=0)
        else:
            output = x.clamp(min=0)
        if keep_tensor:
            y = output
        else:
            y = x <= 0
            y = packbit.packbits_padded(y, dim=dim)
            ctx.relu_dim = dim
        ctx.relu_output = y
        ctx.relu_keep_tensor = keep_tensor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.relu_output
        if ctx.relu_keep_tensor:
            y = y <= 0
        else:
            y = packbit.unpackbits_padded(y, dim=ctx.relu_dim)
            ctx.relu_dim = None
        grad_input = grad_output.masked_fill(y, 0)
        ctx.relu_output = None
        ctx.relu_keep_tensor = None
        return grad_input, None, None, None, None


class ReLU(nn.ReLU, custom_quant.Quant):

    def __init__(self, inplace=False, dim=1, args=None, logger=None):
        super(ReLU, self).__init__(inplace)
        self.repr = super(ReLU, self).__repr__()
        custom_quant.Quant.__init__(self, args=args, logger=logger)
        self.dim = dim
        self.keep_tensor = False
        self.tag = 'relu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            y = relu.apply(x, self.inplace, self.dim, self.keep_tensor)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y


class softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim, clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None, clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None):
        custom_quant.Quant.forward(ctx, x, clip_val1, level1, iteration1, ema_decay1, quant_groups1, shift1, '_1')
        y = F.softmax(x, dim)
        custom_quant.Quant.forward(ctx, y, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2, '_2')
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = custom_quant.Quant.restore(ctx, '_1')
        y = custom_quant.Quant.restore(ctx, '_2')
        if x.is_cuda:
            grad_input = native.softmax_backward_cuda(grad_output, y, ctx.dim, x)
        else:
            grad_input = native.softmax_backward_cpu(grad_output, y, ctx.dim, x)
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None


class Softmax(nn.Softmax):

    def __init__(self, dim=None, args=None, logger=None, quant_groups=1):
        super(Softmax, self).__init__(dim=dim)
        self.quant1 = custom_quant.quantization(tag='softmax-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='softmax-2', quant_groups=quant_groups)
        self.tag = 'softmax'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x):
        if self.quant1.enable and self.quant2.enable and self.training:
            y = softmax.apply(x, self.dim, self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift, self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift)
        else:
            y = F.softmax(x, self.dim)
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MatMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Softmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ziplab_Mesa(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

