import sys
_module = sys.modules[__name__]
del sys
data = _module
input_dataset = _module
eval = _module
ASPP = _module
AlphaGAN = _module
AlphaLoss = _module
AtrousResNet = _module
Decoder = _module
Encoder = _module
NLayerDiscriminator = _module
model = _module
sync_batchnorm = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
train = _module
Tester = _module
utils = _module
visualize = _module

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


import torchvision.transforms as transforms


import numpy as np


import torch as t


import torch.nn as nn


from torch import nn


import torch.nn.functional as F


import torch.optim.lr_scheduler as lr_scheduler


import torchvision as tv


import functools


import collections


import torch


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


from torch.nn.parallel.data_parallel import DataParallel


class _AsppBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size,
        dilation_rate, BatchNorm):
        super(_AsppBlock, self).__init__()
        if dilation_rate == 1:
            self.atrous_conv = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, bias=False)
        else:
            self.atrous_conv = nn.Conv2d(in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation_rate, padding=
                dilation_rate, bias=False)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, _input):
        x = self.atrous_conv(_input)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ASPP, self).__init__()
        self.aspp_1 = _AsppBlock(in_channels, 256, kernel_size=1,
            dilation_rate=1, BatchNorm=BatchNorm)
        self.aspp_6 = _AsppBlock(in_channels, 256, kernel_size=3,
            dilation_rate=6, BatchNorm=BatchNorm)
        self.aspp_12 = _AsppBlock(in_channels, 256, kernel_size=3,
            dilation_rate=12, BatchNorm=BatchNorm)
        self.aspp_18 = _AsppBlock(in_channels, 256, kernel_size=3,
            dilation_rate=18, BatchNorm=BatchNorm)
        self.image_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn
            .Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, bias=False), BatchNorm(out_channels), nn.ReLU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=5 * out_channels,
            out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels), nn.ReLU(True))
        self._init_weight()

    def forward(self, x):
        aspp1 = self.aspp_1(x)
        aspp6 = self.aspp_6(x)
        aspp12 = self.aspp_12(x)
        aspp18 = self.aspp_18(x)
        im_p = self.image_pooling(x)
        im_p = F.interpolate(im_p, size=aspp18.size()[2:], mode='bilinear',
            align_corners=True)
        aspp = [aspp1, aspp6, aspp12, aspp18, im_p]
        aspp = t.cat(aspp, dim=1)
        return self.conv1(aspp)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def BatchNormGroup(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)


class NetG(nn.Module):

    def __init__(self, sync_bn=True):
        super(NetG, self).__init__()
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = BatchNormGroup
        self.encoder = Encoder(BatchNorm)
        self.decoder = Decoder(BatchNorm)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class AlphaLoss(nn.Module):

    def __init__(self, eps=1e-06):
        super(AlphaLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, truth, unknown_region_size):
        diff = predict - truth
        losses = t.sqrt(diff.pow(2) + self.eps * self.eps)
        loss = losses.sum() / (unknown_region_size + 1.0)
        return loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, BatchNorm, stride=1, dilation=1,
        downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        if dilation != 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=
                stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
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

    def __init__(self, block, layers, BatchNorm, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], BatchNorm,
            stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], BatchNorm,
            stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], BatchNorm,
            stride=2, dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, BatchNorm, stride=1,
        dilation=1):
        downsample = None
        downsample_stride = stride if dilation == 1 else 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=downsample_stride,
                bias=False), BatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, BatchNorm, stride,
            dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connection1 = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connection2 = x
        x, max_index = self.maxpool(x)
        x = self.layer1(x)
        skip_connection3 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return (x, skip_connection1, skip_connection2, skip_connection3,
            max_index)


class BilinearUpSample(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BilinearUpSample, self).__init__()
        """
        self.conv = nn.Sequential(
            conv3x3(in_planes=in_planes, out_planes=out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        """

    def forward(self, x):
        in_size = x.size()
        n, c, h, w = in_size
        x = F.interpolate(x, size=(h * 2, w * 2), mode='bilinear',
            align_corners=True)
        return x


class Decoder(nn.Module):

    def __init__(self, BatchNorm):
        super(Decoder, self).__init__()
        self.bilinear_1 = BilinearUpSample(in_planes=256, out_planes=256)
        self.skip_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels
            =64, kernel_size=1, bias=False), BatchNorm(64), nn.ReLU(inplace
            =True))
        self.deconv1_x = nn.Sequential(nn.Conv2d(in_channels=256 + 64,
            out_channels=256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256), nn.ReLU(inplace=True), nn.Conv2d(in_channels=
            256, out_channels=128, kernel_size=3, padding=1, bias=False),
            BatchNorm(128), nn.ReLU(inplace=True), nn.Conv2d(in_channels=
            128, out_channels=64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64), nn.ReLU(inplace=True))
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.skip_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
            32, kernel_size=1, bias=False), BatchNorm(32), nn.ReLU(inplace=
            True))
        self.deconv2_x = nn.Sequential(nn.Conv2d(in_channels=64 + 32,
            out_channels=64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64), nn.ReLU(inplace=True), nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=1, output_padding=1, bias=False), BatchNorm(64), nn.
            ReLU(inplace=True), nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=3, padding=1, bias=False), BatchNorm(32), nn.ReLU(
            inplace=True))
        self.deconv3_x = nn.Sequential(nn.Conv2d(in_channels=32 + 3,
            out_channels=32, kernel_size=3, padding=1, bias=False),
            BatchNorm(32), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32,
            out_channels=32, kernel_size=3, padding=1, bias=False),
            BatchNorm(32), nn.ReLU(inplace=True))
        self.deconv4_x = nn.Sequential(nn.Conv2d(in_channels=32,
            out_channels=1, kernel_size=3, padding=1, bias=False), nn.Sigmoid()
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, skip_connection1, skip_connection2, skip_connection3, max_index = x
        x = self.bilinear_1(x)
        skip_connection3 = self.skip_3(skip_connection3)
        x = t.cat([x, skip_connection3], dim=1)
        x = self.deconv1_x(x)
        x = self.unpooling(x, max_index)
        skip_connection2 = self.skip_2(skip_connection2)
        x = t.cat([x, skip_connection2], dim=1)
        x = self.deconv2_x(x)
        skip_connection1 = skip_connection1[:, 0:3, :, :]
        x = t.cat([x, skip_connection1], dim=1)
        x = self.deconv3_x(x)
        x = self.deconv4_x(x)
        return x


def resnet50(BatchNorm):
    model = ResNet(Bottleneck, [3, 4, 6, 3], BatchNorm)
    return model


class Encoder(nn.Module):

    def __init__(self, BatchNorm):
        super(Encoder, self).__init__()
        self.resnet50 = resnet50(BatchNorm)
        self.aspp = ASPP(2048, 256, BatchNorm)

    def _initialize_weights(self):
        pretrained_resnet50 = tv.models.resnet50(pretrained=True)
        pretrained_dict = pretrained_resnet50.state_dict()
        atrous_resnet_dict = self.resnet50.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
            atrous_resnet_dict}
        atrous_resnet_dict.update(pretrained_dict)
        self.resnet50.load_state_dict(atrous_resnet_dict)

    def forward(self, x):
        (x, skip_connection1, skip_connection2, skip_connection3, max_index
            ) = self.resnet50(x)
        x = self.aspp(x)
        return (x, skip_connection1, skip_connection2, skip_connection3,
            max_index)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.model(input)


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


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
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
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum
                    ) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum
                    ) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum
                ) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1
            ) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0,
            2, 3).contiguous()


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_CDOTAD_AlphaGAN_Matting(_paritybench_base):
    pass
    def test_000(self):
        self._check(AlphaLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BatchNorm2dReimpl(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BilinearUpSample(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DataParallelWithCallback(*[], **{'module': _mock_layer()}), [], {'input': torch.rand([4, 4])})

