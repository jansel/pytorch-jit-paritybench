import sys
_module = sys.modules[__name__]
del sys
config = _module
defaults = _module
dataset = _module
eval = _module
eval_multipro = _module
nn = _module
modules = _module
batchnorm = _module
comm = _module
replicate = _module
test_numeric_batchnorm = _module
test_sync_batchnorm = _module
unittest = _module
parallel = _module
data_parallel = _module
utils = _module
data = _module
dataloader = _module
distributed = _module
sampler = _module
th = _module
models = _module
hrnet = _module
mobilenet = _module
models = _module
resnet = _module
resnext = _module
test = _module
train = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn as nn


from scipy.io import loadmat


import math


import collections


import torch.nn.functional as F


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import functools


from torch.nn.parallel.data_parallel import DataParallel


from torch.autograd import Variable


import torch.cuda as cuda


from torch.nn.parallel._functions import Gather


from torch.distributed import get_world_size


from torch.distributed import get_rank


import logging


import random


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.001, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None
        self._moving_average_fraction = 1.0 - momentum
        self.register_buffer('_tmp_running_mean', torch.zeros(self.
            num_features))
        self.register_buffer('_tmp_running_var', torch.ones(self.num_features))
        self.register_buffer('_running_iter', torch.ones(1))
        self._tmp_running_mean = self.running_mean.clone() * self._running_iter
        self._tmp_running_var = self.running_var.clone() * self._running_iter

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _add_weighted(self, dest, delta, alpha=1, beta=1, bias=0):
        """return *dest* by `dest := dest*alpha + delta*beta + bias`"""
        return dest * alpha + delta * beta + bias

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self._tmp_running_mean = self._add_weighted(self._tmp_running_mean,
            mean.data, alpha=self._moving_average_fraction)
        self._tmp_running_var = self._add_weighted(self._tmp_running_var,
            unbias_var.data, alpha=self._moving_average_fraction)
        self._running_iter = self._add_weighted(self._running_iter, 1,
            alpha=self._moving_average_fraction)
        self.running_mean = self._tmp_running_mean / self._running_iter
        self.running_var = self._tmp_running_var / self._running_iter
        return mean, bias_var.clamp(self.eps) ** -0.5


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module,
            device_ids)
        execute_replication_callbacks(modules)
        return modules


def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """

    def gather_map(outputs):
        out = outputs[0]
        if torch.is_tensor(out):
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)


class DictGatherDataParallel(nn.DataParallel):

    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


BatchNorm2d = SynchronizedBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks,
            num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), BatchNorm2d(
                num_channels[branch_index] * block.expansion, momentum=
                BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels
            [branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, 1, 0, bias
                        =False), BatchNorm2d(num_inchannels[i], momentum=
                        BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), BatchNorm2d(
                                num_outchannels_conv3x3, momentum=BN_MOMENTUM))
                                )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), BatchNorm2d(
                                num_outchannels_conv3x3, momentum=
                                BN_MOMENTUM), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                        size=(height_output, width_output), mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HRNetV2(nn.Module):

    def __init__(self, n_class, **kwargs):
        super(HRNetV2, self).__init__()
        extra = {'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK':
            'BASIC', 'NUM_BLOCKS': (4, 4), 'NUM_CHANNELS': (48, 96),
            'FUSE_METHOD': 'SUM'}, 'STAGE3': {'NUM_MODULES': 4,
            'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4),
            'NUM_CHANNELS': (48, 96, 192), 'FUSE_METHOD': 'SUM'}, 'STAGE4':
            {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
            'NUM_BLOCKS': (4, 4, 4, 4), 'NUM_CHANNELS': (48, 96, 192, 384),
            'FUSE_METHOD': 'SUM'}, 'FINAL_CONV_KERNEL': 1}
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels, multi_scale_output=True)

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, 1, 1, bias=False), BatchNorm2d(
                        num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i
                        ] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels,
                        outchannels, 3, 2, 1, bias=False), BatchNorm2d(
                        outchannels, momentum=BN_MOMENTUM), nn.ReLU(inplace
                        =True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.
                expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True
        ):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block,
                num_blocks, num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, return_feature_maps=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return [x]


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True
                ), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=
                hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(
                inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=
                False), BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.
            last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SegmentationModuleBase(nn.Module):

    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class Resnet(nn.Module):

    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution + BN + relu"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False), BatchNorm2d(out_planes), nn.
        ReLU(inplace=True))


class C1DeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class C1(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPM(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False), BatchNorm2d
                (512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False), BatchNorm2d(
            512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512,
            num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear',
                align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False), BatchNorm2d
                (512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False), BatchNorm2d(
            512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512,
            num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear',
                align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class UPerNet(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512,
                kernel_size=1, bias=False), BatchNorm2d(512), nn.ReLU(
                inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 
            512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim,
                kernel_size=1, bias=False), BatchNorm2d(fpn_dim), nn.ReLU(
                inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim,
                fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) *
            fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(pool_scale(
                conv5), (input_size[2], input_size[3]), mode='bilinear',
                align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode=
                'bilinear', align_corners=False)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i
                ], output_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x
        x = nn.functional.log_softmax(x, dim=1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(GroupBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, groups=32, num_classes=1000):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
            groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
            groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
            groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1
                    ] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_CSAILVision_semantic_segmentation_pytorch(_paritybench_base):
    pass
