import sys
_module = sys.modules[__name__]
del sys
config = _module
config2 = _module
config_dense = _module
config_final = _module
coco_masks_hdf5 = _module
see_coco_data = _module
mydataset = _module
demo_image = _module
evaluate = _module
models = _module
ae_layer = _module
ae_pose = _module
focal_loss = _module
layers = _module
layers_transposed = _module
layers_transposed_final = _module
loss_model = _module
loss_model_parallel = _module
posenet = _module
posenet2 = _module
posenet3 = _module
posenet_final = _module
posenet_independent = _module
parallel_encoding = _module
paralle = _module
py_cocodata_server = _module
py_data_heatmapper = _module
py_data_iterator = _module
test_inference_speed = _module
train = _module
train_distributed = _module
train_distributed_SWA = _module
train_parallel = _module
utils = _module
config_reader = _module
util = _module
draw_net = _module

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


from torch.utils.data import Dataset


import numpy as np


import torch.nn.functional as F


import torch


import math


from scipy.ndimage.filters import gaussian_filter


import torch.optim as optim


import warnings


from itertools import product


from torch import nn


from torch.autograd import Function


from torch.autograd import Variable


import functools


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.distributed import DistributedDataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from torch.utils.data import DataLoader


import torch.cuda


import torch.nn as nn


import torch.distributed as dist


from torch.utils.data.dataloader import DataLoader


from torch.nn import functional as F


class Full(nn.Module):

    def __init__(self, inp_dim, out_dim, bn=False, relu=False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv(nn.Module):

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False,
        relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
            padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, '{} {}'.format(x.size()[1],
            self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


Pool = nn.MaxPool2d


class Hourglass(nn.Module):

    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        if n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Merge(nn.Module):

    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(Conv(3, 64, 7, 2, bn=bn), Conv(64, 128, bn
            =bn), nn.MaxPool2d(2, 2), Conv(128, 128, bn=bn), Conv(128,
            inp_dim, bn=bn))
        self.features = nn.ModuleList([nn.Sequential(Hourglass(4, inp_dim,
            bn, increase), Conv(inp_dim, inp_dim, 3, bn=False), Conv(
            inp_dim, inp_dim, 3, bn=False)) for i in range(nstack)])
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn
            =False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in
            range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            preds_instack = []
            feature = self.features[i](x)
            preds_instack.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds_instack[-1]
                    ) + self.merge_features[i](feature)
            preds.append(preds_instack)
        return preds

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)

    def forward(self, inp_imgs, target_tuple):
        output_tuple = self.posenet(inp_imgs)
        loss = self.criterion(output_tuple, target_tuple)
        if not self.training:
            return output_tuple, loss
        else:
            return loss


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class Residual(nn.Module):
    """Residual Block for original Hourglass Network"""

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(nn.BatchNorm2d(ins), nn.LeakyReLU(
            negative_slope=0.01, inplace=True), nn.Conv2d(ins, outs // 2, 1
            ), nn.BatchNorm2d(outs // 2), nn.LeakyReLU(negative_slope=0.01,
            inplace=True), nn.Conv2d(outs // 2, outs // 2, 3, 1, 1), nn.
            BatchNorm2d(outs // 2), nn.LeakyReLU(negative_slope=0.01,
            inplace=True), nn.Conv2d(outs // 2, outs, 1))
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class Conv(nn.Module):

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False,
        relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=False)
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=True)

    def forward(self, x):
        assert x.size()[1
            ] == self.inp_dim, 'input channel {} dese not fit kernel channel {}'.format(
            x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Backbone(nn.Module):

    def __init__(self, nFeat=256, inplanes=3, resBlock=Residual):
        super(Backbone, self).__init__()
        self.nFeat = nFeat
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2,
            padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""

    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.increase = increase
        self.bn = bn
        self.resBlock = resBlock
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_single_residual(self, depth_id):
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), 
            self.nFeat + self.increase * (depth_id + 1), bn=self.bn)

    def _make_lower_residual(self, depth_id):
        return [self.resBlock(self.nFeat + self.increase * depth_id, self.
            nFeat + self.increase * depth_id, bn=self.bn), self.resBlock(
            self.nFeat + self.increase * depth_id, self.nFeat + self.
            increase * (depth_id + 1), bn=self.bn), self.resBlock(self.
            nFeat + self.increase * (depth_id + 1), self.nFeat + self.
            increase * depth_id, bn=self.bn)]

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            res = self._make_lower_residual(i)
            if i == self.depth - 1:
                res.append(self._make_single_residual(i))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == self.depth - 1:
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        up2 = self.upsample(low3)
        return up1 + up2

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]


class SELayer(nn.Module):

    def __init__(self, inp_dim, reduction=16):
        """
        Squeeze and Excitation
        :param inp_dim: the channel of input tensor
        :param reduction: channel compression ratio
        :return output the tensor with the same shape of input
        """
        assert inp_dim > reduction, 'Make sure your input channel bigger than reduction which equals to {}'.format(
            reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(inp_dim, inp_dim // reduction),
            nn.ReLU(inplace=True), nn.Linear(inp_dim // reduction, inp_dim),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Residual(nn.Module):
    """Residual Block modified by us"""

    def __init__(self, ins, outs, bn=True, relu=True):
        super(Residual, self).__init__()
        self.relu_flag = relu
        self.convBlock = nn.Sequential(nn.Conv2d(ins, outs // 2, 1, bias=
            False), nn.BatchNorm2d(outs // 2), nn.LeakyReLU(negative_slope=
            0.01, inplace=True), nn.Conv2d(outs // 2, outs // 2, 3, 1, 1,
            bias=False), nn.BatchNorm2d(outs // 2), nn.LeakyReLU(
            negative_slope=0.01, inplace=True), nn.Conv2d(outs // 2, outs, 
            1, bias=False), nn.BatchNorm2d(outs))
        if ins != outs:
            self.skipConv = nn.Sequential(nn.Conv2d(ins, outs, 1, bias=
                False), nn.BatchNorm2d(outs))
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        if self.relu_flag:
            x = self.relu(x)
            return x
        else:
            return x


class BasicResidual(nn.Module):
    """
    Basic block used in ResNet, CornerNet, CenterNet, etc.
    Used as the basic block to replace 3*3 convolution, increasing the shortcuts in network
    """

    def __init__(self, inp_dim, out_dim, stride=1, bn=True, relu=True):
        super(BasicResidual, self).__init__()
        self.relu_flag = relu
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1),
            stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1),
            stride=(stride, stride), bias=False), nn.BatchNorm2d(out_dim)
            ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        skip = self.skip(x)
        if self.relu_flag:
            out = self.relu(bn2 + skip)
        else:
            out = bn2 + skip
        return out


class Conv(nn.Module):

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True,
        relu=True, dropout=False, dialated=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=False, dilation=1)
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=True, dilation=1)

    def forward(self, x):
        assert x.size()[1
            ] == self.inp_dim, 'input channel {} dese not fit kernel channel {}'.format(
            x.size()[1], self.inp_dim)
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DilatedConv(nn.Module):
    """
    Dilated convolutional layer of stride=1 only!
    """

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True,
        relu=True, dropout=False, dialation=3):
        super(DilatedConv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=dialation, bias=False, dilation=dialation)
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=dialation, bias=True, dilation=dialation)

    def forward(self, x):
        assert x.size()[1
            ] == self.inp_dim, 'input channel {} dese not fit kernel channel {}'.format(
            x.size()[1], self.inp_dim)
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Backbone(nn.Module):
    """
    Input Tensor: a batch of images with shape (N, C, H, W)
    """

    def __init__(self, nFeat=256, inplanes=3, resBlock=Residual,
        dilatedBlock=DilatedConv):
        super(Backbone, self).__init__()
        self.nFeat = nFeat
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.dilation = nn.Sequential(dilatedBlock(128, 128, dialation=3),
            dilatedBlock(128, 128, dialation=3), dilatedBlock(128, 128,
            dialation=4), dilatedBlock(128, 128, dialation=4), dilatedBlock
            (128, 128, dialation=5), dilatedBlock(128, 128, dialation=5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x1 = self.dilation(x)
        concat_merge = torch.cat([x, x1], dim=1)
        return concat_merge


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""

    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=
        Residual, convBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.increase = increase
        self.bn = bn
        self.resBlock = resBlock
        self.convBlock = convBlock
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_single_residual(self, depth_id):
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), 
            self.nFeat + self.increase * (depth_id + 1), bn=self.bn)

    def _make_lower_residual(self, depth_id):
        pack_layers = [self.resBlock(self.nFeat + self.increase * depth_id,
            self.nFeat + self.increase * depth_id, bn=self.bn), self.
            resBlock(self.nFeat + self.increase * depth_id, self.nFeat + 
            self.increase * (depth_id + 1), bn=self.bn), self.resBlock(self
            .nFeat + self.increase * (depth_id + 1), self.nFeat + self.
            increase * depth_id, bn=self.bn), self.convBlock(self.nFeat + 
            self.increase * depth_id, self.nFeat + self.increase * depth_id,
            bn=self.bn)]
        return pack_layers

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            res = self._make_lower_residual(i)
            if i == self.depth - 1:
                res.append(self._make_single_residual(i))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == self.depth - 1:
            low2 = self.hg[depth_id][4](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        up2 = self.upsample(low3)
        deconv1 = self.hg[depth_id][3](up2)
        return up1 + deconv1

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]


class SELayer(nn.Module):

    def __init__(self, inp_dim, reduction=16):
        """
        Squeeze and Excitation
        :param inp_dim: the channel of input tensor
        :param reduction: channel compression ratio
        :return output the tensor with the same shape of input
        """
        assert inp_dim > reduction, f'Make sure your input channel bigger than reduction which equals to {reduction}'
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(inp_dim, inp_dim // reduction),
            nn.LeakyReLU(inplace=True), nn.Linear(inp_dim // reduction,
            inp_dim), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Residual(nn.Module):
    """Residual Block modified by us"""

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(nn.Conv2d(ins, outs // 2, 1, bias=
            False), nn.BatchNorm2d(outs // 2), nn.LeakyReLU(negative_slope=
            0.01, inplace=True), nn.Conv2d(outs // 2, outs // 2, 3, 1, 1,
            bias=False), nn.BatchNorm2d(outs // 2), nn.LeakyReLU(
            negative_slope=0.01, inplace=True), nn.Conv2d(outs // 2, outs, 
            1, bias=False), nn.BatchNorm2d(outs))
        if ins != outs:
            self.skipConv = nn.Sequential(nn.Conv2d(ins, outs, 1, bias=
                False), nn.BatchNorm2d(outs))
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        x = self.relu(x)
        return x


class Conv(nn.Module):

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False,
        relu=True, dropout=False):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=False)
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                padding=(kernel_size - 1) // 2, bias=True)

    def forward(self, x):
        assert x.size()[1
            ] == self.inp_dim, 'input channel {} dese not fit kernel channel {}'.format(
            x.size()[1], self.inp_dim)
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Backbone(nn.Module):

    def __init__(self, nFeat=256, inplanes=3, resBlock=Residual):
        super(Backbone, self).__init__()
        self.nFeat = nFeat
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""

    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.increase = increase
        self.bn = bn
        self.resBlock = resBlock
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_single_residual(self, depth_id):
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), 
            self.nFeat + self.increase * (depth_id + 1), bn=self.bn)

    def _make_lower_residual(self, depth_id):
        pack_layers = [self.resBlock(self.nFeat + self.increase * depth_id,
            self.nFeat + self.increase * depth_id, bn=self.bn, relu=False),
            self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat +
            self.increase * (depth_id + 1), bn=self.bn), self.resBlock(self
            .nFeat + self.increase * (depth_id + 1), self.nFeat + self.
            increase * depth_id, bn=self.bn), self.resBlock(self.nFeat + 
            self.increase * depth_id, self.nFeat + self.increase * depth_id,
            bn=self.bn), self.resBlock(self.nFeat + self.increase *
            depth_id, self.nFeat + self.increase * depth_id, bn=self.bn,
            relu=False), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
        return pack_layers

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            res = self._make_lower_residual(i)
            if i == self.depth - 1:
                res.append(self._make_single_residual(i))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == self.depth - 1:
            low2 = self.hg[depth_id][6](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        up2 = self.upsample(low3)
        deconv1 = self.hg[depth_id][3](up2)
        deconv2 = self.hg[depth_id][4](deconv1)
        up1 += deconv2
        out = self.hg[depth_id][5](up1)
        return out

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]


class SELayer(nn.Module):

    def __init__(self, inp_dim, reduction=16):
        """
        Squeeze and Excitation
        :param inp_dim: the channel of input tensor
        :param reduction: channel compression ratio
        :return output the tensor with the same shape of input
        """
        assert inp_dim > reduction, 'Make sure your input channel bigger than reduction which equals to {}'.format(
            reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(inp_dim, inp_dim // reduction),
            nn.LeakyReLU(inplace=True), nn.Linear(inp_dim // reduction,
            inp_dim), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiTaskLoss(nn.Module):

    def __init__(self, opt, config, heatmap_weight=1, offset_weight=1, **kwargs
        ):
        super(MultiTaskLoss, self).__init__()
        self.nstack = opt.nstack
        self.batch_size = opt.batch_size
        self.offset_start = config.offset_start
        self.heat_start = config.heat_start
        self.bkg_start = config.bkg_start
        self.multi_task_weight = opt.multi_task_weight
        self.keypoint_task_weight = opt.keypoint_task_weight
        self.scale_weight = opt.scale_weight
        self.nstack_weight = opt.nstack_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight

    def forward(self, pred_tuple, target_tuple):
        """
        Compute the multi-task total loss
        :param pred_tuple: [nstack * [(bacth,C,128,128), (bacth,C,64,64), (bacth,C,32,32),  (bacth,C,16,16)], (bacth,C,8,8)]
        :param target_tuple: target tensors, i.e.,
         mask_misses,   heatmaps,       offsets,       mask_offsets,
        [batch,1,128,128], [batch,43,128,128], [batch,36,128,128], [batch,36,128,128]
        :return: scalar tensor
        """
        pred_scale_tensors = [torch.cat([pred_tuple[j][i][None, ...] for j in
            range(self.nstack)], dim=0) for i in range(5)]
        loss_scales = [(self._loss_per_scale(pred_scale_tensors[i],
            target_tuple) * self.scale_weight[i]) for i in range(5)]
        loss_per_batch = sum(loss_scales) / sum(self.scale_weight
            ) / self.batch_size
        return loss_per_batch

    def _loss_per_scale(self, pred, target):
        """
        Compute the loss on a particular scale.
        :param pred: tensor (nstack, bacth, C, H, W)
        :param target: mask_misses, heatmaps, offsets, mask_offsets of shape (N, C, H, W)
        :return:
        """
        pred_heatmap = pred
        gt_heatmaps = F.adaptive_avg_pool2d(target[1], output_size=pred.
            shape[-2:])
        gt_mask_misses = F.interpolate(target[0], size=pred.shape[-2:],
            mode='bilinear')
        gt_mask_misses[gt_mask_misses < 0.5] = 0
        heatmap_loss = self.focal_l2_loss(pred_heatmap, gt_heatmaps[None,
            ...], gt_mask_misses[None, ...], self.heat_start, self.
            bkg_start, nstack_weight=self.nstack_weight, multi_task_weight=
            self.multi_task_weight, keypoint_task_weight=self.
            keypoint_task_weight)
        return heatmap_loss

    @staticmethod
    def l1_loss(pred, target, mask_offset, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the smooth L1 loss of offset feature maps
        :param pred: predicted tensor (nstack, batch, channel, height, width), predicted feature maps
        :param target: target tensor (nstack, batch, channel, height, width)
        :param mask_offset: tensor (nstack, batch, channel, height, width)
        :param nstack_weight:
        :return:
        """
        out = torch.abs(pred - target) * mask_offset
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss

    @staticmethod
    def l2_loss(s, sxing, mask_miss, heat_start, bkg_start,
        multi_task_weight=0.1, keypoint_task_weight=1, nstack_weight=[1, 1,
        1, 1]):
        """
        Compute the L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (1, batch, channel, height, width)
        :param mask_miss: tensor (1, batch, 1, height, width)
        :return: a scalar tensor
        """
        mask = mask_miss.expand_as(sxing).clone()
        del mask_miss
        mask[:, :, (-2), :, :] *= multi_task_weight
        mask[:, :, heat_start:bkg_start, :, :] *= keypoint_task_weight
        out = (s - sxing) ** 2 * mask
        loss_nstack = out.sum(dim=4).sum(dim=3).sum(dim=2).sum(dim=1)
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss

    @staticmethod
    def focal_l2_loss(s, sxing, mask_miss, heat_start, bkg_start, gamma=1,
        multi_task_weight=0.1, keypoint_task_weight=1, nstack_weight=[1, 1,
        1, 1], alpha=0.0, beta=0.0):
        """
        Compute the focal L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (nstack, batch, 1, height, width)
        :param gamma: focusing parameter
        :return: a scalar tensor
        """
        mask = mask_miss.expand_as(sxing).clone()
        del mask_miss
        mask[:, :, (-2), :, :] *= multi_task_weight
        mask[:, :, heat_start:bkg_start, :, :] *= keypoint_task_weight
        st = torch.where(torch.ge(sxing, 0.01), s - alpha, 1 - s - beta)
        factor = torch.abs(1.0 - st)
        out = (s - sxing) ** 2 * factor * mask
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss


class MultiTaskLossParallel(nn.Module):

    def __init__(self, opt, config, heatmap_weight=1, offset_weight=1, **kwargs
        ):
        super(MultiTaskLossParallel, self).__init__()
        self.nstack = opt.nstack
        self.batch_size = opt.batch_size
        self.offset_start = config.offset_start
        self.multi_task_weight = opt.multi_task_weight
        self.scale_weight = opt.scale_weight
        self.nstack_weight = opt.nstack_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight

    def forward(self, pred_tuple, target_tuple):
        """
        Compute the multi-task total loss
        :param pred_tuple: [nstack * [(bacth,C,128,128), (bacth,C,64,64), (bacth,C,32,32),
        (bacth,C,16,16)], (bacth,C,8,8)]
        :param target_tuple: target tensors, i.e.,
         mask_misses,   heatmaps,       offsets,       mask_offsets,
        [batch,1,128,128], [batch,44,128,128], [batch,36,128,128], [batch,36,128,128]
        :return: scalar tensor
        """
        pred_scale_tensors = [torch.cat([pred_tuple[j][i][None, ...] for j in
            range(self.nstack)], dim=0) for i in range(5)]
        loss_scales = [(self._loss_per_scale(pred_scale_tensors[i],
            target_tuple) * self.scale_weight[i]) for i in range(5)]
        loss_per_batch = sum(loss_scales) / sum(self.scale_weight)
        return loss_per_batch

    def _loss_per_scale(self, pred, target):
        """
        Compute the loss on a particular scale.
        :param pred: tensor (nstack, bacth, C, H, W)
        :param target: mask_misses, heatmaps, offsets, mask_offsets of shape (N, C, H, W)
        :return:
        """
        pred_heatmap = pred[:, :, :self.offset_start]
        gt_mask_misses = F.interpolate(target[0], size=pred.shape[-2:],
            mode='bilinear')
        gt_heatmaps = F.adaptive_avg_pool2d(target[1], output_size=pred.
            shape[-2:])
        heatmap_loss = self.l2_loss(pred_heatmap, gt_heatmaps[None, ...],
            gt_mask_misses[None, ...], nstack_weight=self.nstack_weight)
        return heatmap_loss

    @staticmethod
    def focal_l2_loss(s, sxing, mask_miss, gamma=2, nstack_weight=[1, 1, 1, 1]
        ):
        """
        Compute the focal L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (1, batch, 1, height, width)
        :param gamma: focusing parameter
        :return: a scalar tensor
        """
        st = torch.where(torch.ge(sxing, 0.01), s, 1 - s)
        factor = (1.0 - st) ** gamma
        out = (s - sxing) ** 2 * factor * mask_miss
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss

    @staticmethod
    def l1_loss(pred, target, mask_offset, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the  L1 loss of offset feature maps
        :param pred: predicted tensor (nstack, batch, channel, height, width), predicted feature maps
        :param target: target tensor (nstack, batch, channel, height, width)
        :param mask_offset: tensor (nstack, batch, channel, height, width)
        :param nstack_weight:
        :return:
        """
        out = torch.abs(pred - target) * mask_offset
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss

    @staticmethod
    def l2_loss(s, sxing, mask_miss, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (nstack, batch, 1, height, width)
        :return: a scalar tensor
        """
        out = (s - sxing) ** 2 * mask_miss
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        None
        weight_loss = [(loss_nstack[i] * nstack_weight[i]) for i in range(
            len(nstack_weight))]
        loss = sum(weight_loss) / sum(nstack_weight)
        return loss


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        self.before_regress = nn.ModuleList([nn.Sequential(Conv(inp_dim + i *
            increase, inp_dim, 3, bn=bn, dropout=False), Conv(inp_dim,
            inp_dim, 3, bn=bn, dropout=False), SELayer(inp_dim)) for i in
            range(5)])

    def forward(self, fms):
        assert len(fms
            ) == 5, 'hourglass output {} tensors,but 5 scale heatmaps are supervised'.format(
            len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = Backbone(nFeat=inp_dim)
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=
            bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase,
            bn=bn) for _ in range(nstack)])
        self.outs = nn.ModuleList([nn.ModuleList([Conv(inp_dim, oup_dim, 1,
            relu=False, bn=False) for j in range(5)]) for i in range(nstack)])
        self.merge_features = nn.ModuleList([nn.ModuleList([Merge(inp_dim, 
            inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([nn.ModuleList([Merge(oup_dim, 
            inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
            range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)
            if i == 0:
                features_cache = [torch.zeros_like(hourglass_feature[scale]
                    ) for scale in range(5)]
            else:
                hourglass_feature = [(hourglass_feature[scale] +
                    features_cache[scale]) for scale in range(5)]
            features_instack = self.features[i](hourglass_feature)
            for j in range(5):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]
                            ) + self.merge_features[i][j](features_instack[j])
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
                    else:
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, increase=opt.increase)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)
        self.swa = swa

    def forward(self, input_all):
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            loss = self.criterion(output_tuple, target_tuple)
            return output_tuple, loss
        elif not self.swa:
            loss = self.criterion(output_tuple, target_tuple)
            return loss
        else:
            return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, init_weights=False, increase=opt.increase)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        self.before_regress = nn.ModuleList([nn.Sequential(Conv(inp_dim + i *
            increase, inp_dim + i * increase, 3, bn=bn, dropout=False),
            Conv(inp_dim + i * increase, inp_dim + i * increase, 3, bn=bn,
            dropout=False)) for i in range(5)])

    def forward(self, fms):
        assert len(fms
            ) == 5, 'hourglass output {} tensors,but 5 scale heatmaps are supervised'.format(
            len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = Backbone(nFeat=inp_dim)
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=
            bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase,
            bn=bn) for _ in range(nstack)])
        self.outs = nn.ModuleList([nn.ModuleList([Conv(inp_dim + j *
            increase, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for
            i in range(nstack)])
        self.channel_attention = nn.ModuleList([nn.ModuleList([SELayer(
            inp_dim + j * increase) for j in range(5)]) for i in range(nstack)]
            )
        self.merge_features = nn.ModuleList([nn.ModuleList([Merge(inp_dim +
            j * increase, inp_dim + j * increase) for j in range(5)]) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([nn.ModuleList([Merge(oup_dim, 
            inp_dim + j * increase) for j in range(5)]) for i in range(
            nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)
            if i == 0:
                features_cache = [torch.zeros_like(hourglass_feature[scale]
                    ) for scale in range(5)]
                for s in range(5):
                    hourglass_feature[s] = self.channel_attention[i][s](
                        hourglass_feature[s])
            else:
                for k in range(5):
                    hourglass_feature_attention = self.channel_attention[i][k](
                        hourglass_feature[k])
                    hourglass_feature[k
                        ] = hourglass_feature_attention + features_cache[k]
            features_instack = self.features[i](hourglass_feature)
            for j in range(5):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]
                            ) + self.merge_features[i][j](features_instack[j])
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
                    else:
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)

    def forward(self, inp_imgs, target_tuple):
        output_tuple = self.posenet(inp_imgs)
        loss = self.criterion(output_tuple, target_tuple)
        if not self.training:
            return output_tuple, loss
        else:
            return loss


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, init_weights=False)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        self.before_regress = nn.ModuleList([nn.Sequential(Conv(inp_dim + i *
            increase, inp_dim + i * increase, 3, bn=bn, dropout=False)) for
            i in range(5)])

    def forward(self, fms):
        assert len(fms
            ) == 5, 'hourglass output {} tensors,but 5 scale heatmaps are supervised'.format(
            len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(Conv(3, 64, 7, 2, bn=bn), Conv(64, 128, bn
            =bn), nn.MaxPool2d(2, 2), Conv(128, 128, bn=bn), Conv(128,
            inp_dim, bn=bn))
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=
            bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase,
            bn=bn) for _ in range(nstack)])
        self.outs = nn.ModuleList([nn.ModuleList([Conv(inp_dim + j *
            increase, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for
            i in range(nstack)])
        self.channel_attention = nn.ModuleList([nn.ModuleList([SELayer(
            inp_dim + j * increase) for j in range(5)]) for i in range(nstack)]
            )
        self.merge_features = nn.ModuleList([nn.ModuleList([Merge(inp_dim +
            j * increase, inp_dim + j * increase, bn=bn) for j in range(5)]
            ) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([nn.ModuleList([Merge(oup_dim, 
            inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
            range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)
            if i == 0:
                features_cache = [torch.zeros_like(hourglass_feature[scale]
                    ) for scale in range(5)]
                for s in range(5):
                    hourglass_feature[s] = self.channel_attention[i][s](
                        hourglass_feature[s])
            else:
                for k in range(5):
                    hourglass_feature_attention = self.channel_attention[i][k](
                        hourglass_feature[k])
                    hourglass_feature[k
                        ] = hourglass_feature_attention + features_cache[k]
            features_instack = self.features[i](hourglass_feature)
            for j in range(5):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]
                            ) + self.merge_features[i][j](features_instack[j])
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
                    else:
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, increase=opt.increase)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)
        self.swa = swa

    def forward(self, input_all):
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            loss = self.criterion(output_tuple, target_tuple)
            return output_tuple, loss
        elif not self.swa:
            loss = self.criterion(output_tuple, target_tuple)
            return loss
        else:
            return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, init_weights=False, increase=opt.increase)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        self.before_regress = nn.ModuleList([nn.Sequential(Conv(inp_dim + i *
            increase, inp_dim, 1, bn=bn, dropout=False), Conv(inp_dim,
            inp_dim, 3, bn=bn, dropout=False), Conv(inp_dim, inp_dim, 3, bn
            =bn, dropout=False)) for i in range(5)])

    def forward(self, fms):
        assert len(fms
            ) == 5, 'hourglass output {} tensors,but 5 scale heatmaps are supervised'.format(
            len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = Backbone(nFeat=inp_dim)
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=
            bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase,
            bn=bn) for _ in range(nstack)])
        self.outs = nn.ModuleList([nn.ModuleList([Conv(inp_dim, oup_dim, 1,
            relu=False, bn=False) for j in range(5)]) for i in range(nstack)])
        self.channel_attention = nn.ModuleList([nn.ModuleList([SELayer(
            inp_dim + j * increase) for j in range(5)]) for i in range(nstack)]
            )
        self.merge_features = nn.ModuleList([nn.ModuleList([Merge(inp_dim, 
            inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([nn.ModuleList([Merge(oup_dim, 
            inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
            range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)
            if i == 0:
                features_cache = [torch.zeros_like(hourglass_feature[scale]
                    ) for scale in range(5)]
                for s in range(5):
                    hourglass_feature[s] = self.channel_attention[i][s](
                        hourglass_feature[s])
            else:
                for k in range(5):
                    hourglass_feature_attention = self.channel_attention[i][k](
                        hourglass_feature[k])
                    hourglass_feature[k
                        ] = hourglass_feature_attention + features_cache[k]
            features_instack = self.features[i](hourglass_feature)
            for j in range(5):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]
                            ) + self.merge_features[i][j](features_instack[j])
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
                    else:
                        features_cache[j] = self.merge_preds[i][j](
                            preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, increase=opt.increase)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)
        self.swa = swa

    def forward(self, input_all):
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            loss = self.criterion(output_tuple, target_tuple)
            return output_tuple, loss
        elif not self.swa:
            loss = self.criterion(output_tuple, target_tuple)
            return loss
        else:
            return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn, init_weights=False, increase=opt.increase)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        self.before_regress = nn.ModuleList([nn.Sequential(Conv(inp_dim + i *
            increase, inp_dim + i * increase, 3, bn=bn), Conv(inp_dim + i *
            increase, inp_dim + i * increase, 3, bn=bn)) for i in range(5)])

    def forward(self, fms):
        assert len(fms
            ) == 5, 'hourglass output {} tensors,but 5 scale heatmaps are supervised'.format(
            len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn:
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(Conv(3, 64, 7, 2, bn=bn), Conv(64, 128, bn
            =bn), nn.MaxPool2d(2, 2), Conv(128, 128, bn=bn), Conv(128,
            inp_dim, bn=bn))
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=
            bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase,
            bn=bn) for _ in range(nstack)])
        self.outs = nn.ModuleList([nn.ModuleList([Conv(inp_dim + j *
            increase, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for
            i in range(nstack)])
        self.merge_features = nn.ModuleList([nn.ModuleList([Merge(inp_dim +
            j * increase, inp_dim + j * increase) for j in range(5)]) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([nn.ModuleList([Merge(oup_dim, 
            inp_dim + j * increase) for j in range(5)]) for i in range(
            nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)
            features_instack = self.features[i](hourglass_feature)
            for j in range(5):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]
                            ) + self.merge_features[i][j](features_instack[j])
            pred.append(preds_instack)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False, dist=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn)
        self.criterion = MultiTaskLoss(opt, config
            ) if dist else MultiTaskLossParallel(opt, config)

    def forward(self, inp_imgs, target_tuple):
        output_tuple = self.posenet(inp_imgs)
        loss = self.criterion(output_tuple, target_tuple)
        if not self.training:
            return output_tuple, loss
        else:
            return loss


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """

    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.
            num_layers, bn=bn)

    def forward(self, inp_imgs):
        output_tuple = self.posenet(inp_imgs)
        if not self.training:
            return output_tuple
        else:
            raise ValueError('\nOnly eval mode is available!!')


class DistributedDataParallelModel(DistributedDataParallel):
    """Implements data parallelism at the module level for the DistributedDataParallel module.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = encoding.nn.DistributedDataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.
    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                if not isinstance(input, (list, tuple)):
                    input = input,
                if not isinstance(target, (list, tuple)):
                    target = target,
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage.
    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Examples::
        smoothing = GaussianSmoothing(3, 5, 1)
        input = torch.rand(1, 3, 100, 100)
        # 使用的是镜面对称的padding策略，这样更合理，避免硬性添加constant边缘导致边缘像素滤波异常
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for
            size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((
                mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.
                format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hellojialee_Improved_Body_Parts(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Backbone(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(BasicResidual(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Conv(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(DilatedConv(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Full(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Hourglass(*[], **{'depth': 1, 'nFeat': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Merge(*[], **{'x_dim': 4, 'y_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(Residual(*[], **{'ins': 4, 'outs': 4}), [torch.rand([4, 4, 4, 4])], {})

