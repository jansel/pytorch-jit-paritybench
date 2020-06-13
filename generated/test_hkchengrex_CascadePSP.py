import sys
_module = sys.modules[__name__]
del sys
dataset = _module
make_bb_trans = _module
offline_dataset = _module
online_dataset = _module
split_dataset = _module
eval = _module
eval_helper = _module
eval_memory_usage = _module
eval_post = _module
eval_post_ade = _module
models = _module
psp = _module
extractors = _module
pspnet = _module
sobel_op = _module
sync_batchnorm = _module
batchnorm = _module
comm = _module
replicate = _module
unittest = _module
binary_mask_negate = _module
convert_binary = _module
convert_deeplab_outputs = _module
convert_refinenet_output = _module
convert_psp_outputs = _module
scripts = _module
ade_expand_inst = _module
all_plus_one = _module
download_training_dataset = _module
segmentation_refinement = _module
download = _module
eval_helper = _module
main = _module
extractors = _module
pspnet = _module
setup = _module
test = _module
train = _module
util = _module
boundary_modification = _module
compute_boundary_acc = _module
de_transform = _module
file_buffer = _module
hyper_para = _module
image_saver = _module
log_integrator = _module
logger = _module
metrics_compute = _module
model_saver = _module
util = _module

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


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


from collections import OrderedDict


import math


from torch.utils import model_zoo


from torch import nn


from torch.nn import functional as F


import numpy as np


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import functools


from torch.nn.parallel.data_parallel import DataParallel


from torch import optim


from torch.utils.data import ConcatDataset


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
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

    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)
        x_2 = self.layer1(x)
        x = self.layer2(x_2)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, x_1, x_2


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for
            size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1),
            out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode=
            'bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(SynchronizedBatchNorm2d(in_channels), nn.
            ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 3,
            padding=1), SynchronizedBatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.conv2 = nn.Sequential(SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3,
            padding=1), SynchronizedBatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear',
            align_corners=False)
        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)
        p = p + sc
        p2 = self.conv2(p)
        return p + p2


class PSPNet(nn.Module):

    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048,
        deep_features_size=1024, backend='resnet34', pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.up_1 = PSPUpsample(1024, 1024 + 256, 512)
        self.up_2 = PSPUpsample(512, 512 + 64, 256)
        self.up_3 = PSPUpsample(256, 256 + 3, 32)
        self.final_28 = nn.Sequential(nn.Conv2d(1024, 32, kernel_size=1),
            nn.ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1))
        self.final_56 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1), nn
            .ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1))
        self.final_11 = nn.Conv2d(32 + 3, 32, kernel_size=1)
        self.final_21 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, seg, inter_s8=None, inter_s4=None):
        images = {}
        """
        First iteration, s8 output
        """
        if inter_s8 is None:
            p = torch.cat((x, seg, seg, seg), 1)
            f, f_1, f_2 = self.feats(p)
            p = self.psp(f)
            inter_s8 = self.final_28(p)
            r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s8 = torch.tanh(r_inter_s8)
            images['pred_28'] = torch.sigmoid(r_inter_s8)
            images['out_28'] = r_inter_s8
        else:
            r_inter_tanh_s8 = inter_s8
        """
        Second iteration, s8 output
        """
        if inter_s4 is None:
            p = torch.cat((x, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)
            f, f_1, f_2 = self.feats(p)
            p = self.psp(f)
            inter_s8_2 = self.final_28(p)
            r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)
            p = self.up_1(p, f_2)
            inter_s4 = self.final_56(p)
            r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s4 = torch.tanh(r_inter_s4)
            images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
            images['out_28_2'] = r_inter_s8_2
            images['pred_56'] = torch.sigmoid(r_inter_s4)
            images['out_56'] = r_inter_s4
        else:
            r_inter_tanh_s8_2 = inter_s8
            r_inter_tanh_s4 = inter_s4
        """
        Third iteration, s1 output
        """
        p = torch.cat((x, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)
        f, f_1, f_2 = self.feats(p)
        p = self.psp(f)
        inter_s8_3 = self.final_28(p)
        r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode=
            'bilinear', align_corners=False)
        p = self.up_1(p, f_2)
        inter_s4_2 = self.final_56(p)
        r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode=
            'bilinear', align_corners=False)
        p = self.up_2(p, f_1)
        p = self.up_3(p, x)
        """
        Final output
        """
        p = F.relu(self.final_11(torch.cat([p, x], 1)), inplace=True)
        p = self.final_21(p)
        pred_224 = torch.sigmoid(p)
        images['pred_224'] = pred_224
        images['out_224'] = p
        images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)
        images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)
        images['out_28_3'] = r_inter_s8_3
        images['out_56_2'] = r_inter_s4_2
        return images


class SobelOperator(nn.Module):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel).unsqueeze(0
            ).unsqueeze(0).float()
        self.conv_x.weight.requires_grad = False
        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel).unsqueeze(0
            ).unsqueeze(0).float()
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        x = x.view(b, c, h, w)
        return x


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


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


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


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

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

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


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

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

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.data
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)
        x_2 = self.layer1(x)
        x = self.layer2(x_2)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, x_1, x_2


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for
            size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1),
            out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode=
            'bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels, out_channels, 3, padding=
            1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.
            Conv2d(out_channels, out_channels, 3, padding=1))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding
            =1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.
            Conv2d(out_channels, out_channels, 3, padding=1))
        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear',
            align_corners=False)
        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)
        p = p + sc
        p2 = self.conv2(p)
        return p + p2


class RefinementModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.feats = extractors.resnet50()
        self.psp = PSPModule(2048, 1024, (1, 2, 3, 6))
        self.up_1 = PSPUpsample(1024, 1024 + 256, 512)
        self.up_2 = PSPUpsample(512, 512 + 64, 256)
        self.up_3 = PSPUpsample(256, 256 + 3, 32)
        self.final_28 = nn.Sequential(nn.Conv2d(1024, 32, kernel_size=1),
            nn.ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1))
        self.final_56 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1), nn
            .ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1))
        self.final_11 = nn.Conv2d(32 + 3, 32, kernel_size=1)
        self.final_21 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, seg, inter_s8=None, inter_s4=None):
        images = {}
        """
        First iteration, s8 output
        """
        if inter_s8 is None:
            p = torch.cat((x, seg, seg, seg), 1)
            f, f_1, f_2 = self.feats(p)
            p = self.psp(f)
            inter_s8 = self.final_28(p)
            r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s8 = torch.tanh(r_inter_s8)
            images['pred_28'] = torch.sigmoid(r_inter_s8)
            images['out_28'] = r_inter_s8
        else:
            r_inter_tanh_s8 = inter_s8
        """
        Second iteration, s8 output
        """
        if inter_s4 is None:
            p = torch.cat((x, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)
            f, f_1, f_2 = self.feats(p)
            p = self.psp(f)
            inter_s8_2 = self.final_28(p)
            r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)
            p = self.up_1(p, f_2)
            inter_s4 = self.final_56(p)
            r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode=
                'bilinear', align_corners=False)
            r_inter_tanh_s4 = torch.tanh(r_inter_s4)
            images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
            images['out_28_2'] = r_inter_s8_2
            images['pred_56'] = torch.sigmoid(r_inter_s4)
            images['out_56'] = r_inter_s4
        else:
            r_inter_tanh_s8_2 = inter_s8
            r_inter_tanh_s4 = inter_s4
        """
        Third iteration, s1 output
        """
        p = torch.cat((x, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)
        f, f_1, f_2 = self.feats(p)
        p = self.psp(f)
        inter_s8_3 = self.final_28(p)
        r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode=
            'bilinear', align_corners=False)
        p = self.up_1(p, f_2)
        inter_s4_2 = self.final_56(p)
        r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode=
            'bilinear', align_corners=False)
        p = self.up_2(p, f_1)
        p = self.up_3(p, x)
        """
        Final output
        """
        p = F.relu(self.final_11(torch.cat([p, x], 1)), inplace=True)
        p = self.final_21(p)
        pred_224 = torch.sigmoid(p)
        images['pred_224'] = pred_224
        images['out_224'] = p
        images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)
        images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)
        images['out_28_3'] = r_inter_s8_3
        images['out_56_2'] = r_inter_s4_2
        return images


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hkchengrex_CascadePSP(_paritybench_base):
    pass
    def test_000(self):
        self._check(PSPModule(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SobelOperator(*[], **{'epsilon': 4}), [torch.rand([4, 4, 4, 4])], {})

