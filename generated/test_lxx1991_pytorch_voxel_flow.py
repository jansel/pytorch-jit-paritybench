import sys
_module = sys.modules[__name__]
del sys
datasets = _module
ucf_101 = _module
ucf_101_test = _module
models = _module
voxel_flow = _module
ops = _module
sync_bn = _module
_ext = _module
sync_bn_lib = _module
build = _module
functions = _module
sync_bn = _module
modules = _module
sync_bn = _module
utils = _module
config = _module
eval = _module
optim = _module
transforms = _module
train = _module

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


import random


import numpy as np


import torch


from torch.utils.data import Dataset


from torch import nn


from torch import cuda


from torch.autograd import Function


import queue


from torch.autograd import Variable


import torch.optim


import time


import torch.backends.cudnn as cudnn


class _sync_batch_norm(Function):

    def __init__(self, momentum, eps, queue):
        super(_sync_batch_norm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.queue = queue
        self.allreduce_num = len(self.queue) + 1

    def all_reduce_thread(self, input):
        input_device = input.get_device()
        if input_device == 0:
            data_list = [input]
            for i in range(self.allreduce_num - 1):
                data_list.append(self.queue[i].get())
            cuda.synchronize()
            cuda.nccl.all_reduce(data_list)
            cuda.synchronize()
            for i in range(self.allreduce_num - 1):
                self.queue[i].task_done()
        else:
            self.queue[input_device - 1].put(input)
            self.queue[input_device - 1].join()
        return input

    def forward(self, input, running_mean, running_var, weight, bias):
        with torch.cuda.device_of(input):
            mean = input.new().resize_(input.size(1)).zero_()
            var = input.new().resize_(input.size(1)).zero_()
            x_std = input.new().resize_(input.size(1)).zero_()
            x_norm = input.new().resize_as_(input)
            output = input.new().resize_as_(input)
        sync_bn_lib.bn_forward_mean_before_allreduce(input, mean, self.allreduce_num)
        mean = self.all_reduce_thread(mean)
        sync_bn_lib.bn_forward_var_before_allreduce(input, mean, var, output, self.allreduce_num)
        var = self.all_reduce_thread(var)
        sync_bn_lib.bn_forward_after_allreduce(mean, running_mean, var, running_var, x_norm, x_std, weight, bias, output, self.eps, 1.0 - self.momentum)
        self.save_for_backward(weight, bias)
        self.mean = mean
        self.x_norm = x_norm
        self.x_std = x_std
        return output

    def backward(self, grad_output):
        weight, bias = self.saved_tensors
        with torch.cuda.device_of(grad_output):
            grad_input = grad_output.new().resize_as_(grad_output).zero_()
            grad_weight = grad_output.new().resize_as_(weight).zero_()
            grad_bias = grad_output.new().resize_as_(bias).zero_()
            grad_local_weight = grad_output.new().resize_as_(weight).zero_()
            grad_local_bias = grad_output.new().resize_as_(bias).zero_()
        sync_bn_lib.bn_backward_before_allreduce(grad_output, self.x_norm, self.mean, self.x_std, grad_input, grad_local_weight, grad_local_bias, grad_weight, grad_bias)
        grad_local_weight = self.all_reduce_thread(grad_local_weight)
        grad_local_bias = self.all_reduce_thread(grad_local_bias)
        sync_bn_lib.bn_backward_after_allreduce(grad_output, self.x_norm, grad_local_weight, grad_local_bias, weight, self.x_std, grad_input, self.allreduce_num)
        return grad_input, None, None, grad_weight, grad_bias


def sync_batch_norm(input, running_mean, running_var, weight=None, bias=None, momentum=0.1, eps=1e-05, queue=None):
    """Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _torch_ext.batchnormtrain:

    .. math::

        y = \\frac{x - \\mu[x]}{ \\sqrt{var[x] + \\epsilon}} * \\gamma + \\beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _sync_batch_norm(momentum, eps, queue)(input, running_mean, running_var, weight, bias)


class SyncBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, *args, parallel=False, **kwargs):
        self.parallel = parallel
        self.queue = [queue.Queue(1) for _ in range(torch.cuda.device_count() - 1)]
        super(SyncBatchNorm2d, self).__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training and self.parallel:
                B, C, H, W = input.size()
                rm = Variable(self.running_mean, requires_grad=False)
                rv = Variable(self.running_var, requires_grad=False)
                output = sync_batch_norm(input.view(B, C, -1).contiguous(), rm, rv, self.weight, self.bias, self.momentum, self.eps, self.queue)
                self.running_mean = rm.data
                self.running_var = rv.data
                return output.view(B, C, H, W)
            else:
                return super(SyncBatchNorm2d, self).forward(input)
        else:
            raise RuntimeError('unknown input type')


def convert_bn(model, training, parallel=False):
    if isinstance(model, torch.nn.Module):
        if parallel:
            if not training:
                raise RuntimeError('unsupported parallel during testing')
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train(training)
            if isinstance(m, SyncBatchNorm2d):
                m.parallel = parallel
    else:
        raise RuntimeError('unknown input type')


def meshgrid(height, width):
    x_t = torch.matmul(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))
    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y


class VoxelFlow(nn.Module):

    def __init__(self, config):
        super(VoxelFlow, self).__init__()
        self.config = config
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.syn_type = config.syn_type
        bn_param = config.bn_param
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = BatchNorm2d(64, **bn_param)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = BatchNorm2d(128, **bn_param)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_bn = BatchNorm2d(256, **bn_param)
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = BatchNorm2d(256, **bn_param)
        self.deconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv1_bn = BatchNorm2d(256, **bn_param)
        self.deconv2 = nn.Conv2d(384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = BatchNorm2d(128, **bn_param)
        self.deconv3 = nn.Conv2d(192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv3_bn = BatchNorm2d(64, **bn_param)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(VoxelFlow, self).train(mode)
        if mode:
            convert_bn(self, self.config.bn_training, self.config.bn_parallel)
        else:
            convert_bn(self, False, False)

    def get_optim_policies(self):
        outs = []
        outs.extend(self.get_module_optim_policies(self, self.config, 'model'))
        return outs

    def get_module_optim_policies(self, module, config, prefix):
        weight = []
        bias = []
        bn = []
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                bn.extend(list(m.parameters()))
        return [{'params': weight, 'lr_mult': config.mult_conv_w[0], 'decay_mult': config.mult_conv_w[1], 'name': prefix + ' weight'}, {'params': bias, 'lr_mult': config.mult_conv_b[0], 'decay_mult': config.mult_conv_b[1], 'name': prefix + ' bias'}, {'params': bn, 'lr_mult': config.mult_bn[0], 'decay_mult': config.mult_bn[1], 'name': prefix + ' bn scale/shift'}]

    def forward(self, x, syn_type='inter'):
        input = x
        input_size = tuple(x.size()[2:4])
        x = self.conv1(x)
        x = self.conv1_bn(x)
        conv1 = self.relu(x)
        x = self.pool(conv1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)
        x = self.pool(conv2)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        conv3 = self.relu(x)
        x = self.pool(conv3)
        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = nn.functional.tanh(x)
        flow = x[:, 0:2, :, :]
        mask = x[:, 2:3, :, :]
        grid_x, grid_y = meshgrid(input_size[0], input_size[1])
        with torch.device(input.get_device()):
            grid_x = torch.autograd.Variable(grid_x.repeat([input.size()[0], 1, 1]))
            grid_y = torch.autograd.Variable(grid_y.repeat([input.size()[0], 1, 1]))
        flow = 0.5 * flow
        if self.syn_type == 'inter':
            coor_x_1 = grid_x - flow[:, 0, :, :]
            coor_y_1 = grid_y - flow[:, 1, :, :]
            coor_x_2 = grid_x + flow[:, 0, :, :]
            coor_y_2 = grid_y + flow[:, 1, :, :]
        elif self.syn_type == 'extra':
            coor_x_1 = grid_x - flow[:, 0, :, :] * 2
            coor_y_1 = grid_y - flow[:, 1, :, :] * 2
            coor_x_2 = grid_x - flow[:, 0, :, :]
            coor_y_2 = grid_y - flow[:, 1, :, :]
        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)
        output_1 = torch.nn.functional.grid_sample(input[:, 0:3, :, :], torch.stack([coor_x_1, coor_y_1], dim=3), padding_mode='border')
        output_2 = torch.nn.functional.grid_sample(input[:, 3:6, :, :], torch.stack([coor_x_2, coor_y_2], dim=3), padding_mode='border')
        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])
        x = mask * output_1 + (1.0 - mask) * output_2
        return x


class DataParallelwithSyncBN(torch.nn.DataParallel):

    def replicate(self, module, device_ids):
        replicas = super(DataParallelwithSyncBN, self).replicate(module, device_ids)
        sync_bn_dict = {}
        for n, m in replicas[0].named_modules():
            if isinstance(m, SyncBatchNorm2d):
                sync_bn_dict[n] = m
        for i in range(1, len(replicas)):
            for n, m in replicas[i].named_modules():
                if isinstance(m, SyncBatchNorm2d):
                    m.queue = sync_bn_dict[n].queue
        return replicas


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SyncBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lxx1991_pytorch_voxel_flow(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

