import sys
_module = sys.modules[__name__]
del sys
Kitti_loader = _module
Datasets = _module
dataloader = _module
benchmark_metrics = _module
loss = _module
ERFNet = _module
Models = _module
model = _module
test = _module
utils = _module
main = _module

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
xrange = range
wraps = functools.wraps


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import torch


import random


import torchvision.transforms.functional as F


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data


import torchvision.transforms as transforms


import time


import torch.optim


from torch.optim import lr_scheduler


import torch.nn.init as init


import torch.distributed as dist


import warnings


class MAE_loss(nn.Module):

    def __init__(self):
        super(MAE_loss, self).__init__()

    def forward(self, prediction, gt, input, epoch=0):
        prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt)
        mask = (gt > 0).detach()
        mae_loss = torch.mean(abs_err[mask])
        return mae_loss


class MAE_log_loss(nn.Module):

    def __init__(self):
        super(MAE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=0)
        abs_err = torch.abs(torch.log(prediction + 1e-06) - torch.log(gt + 1e-06))
        mask = (gt > 0).detach()
        mae_log_loss = torch.mean(abs_err[mask])
        return mae_log_loss


class MSE_loss(nn.Module):

    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        err = prediction[:, 0:1] - gt
        mask = (gt > 0).detach()
        mse_loss = torch.mean(err[mask] ** 2)
        return mse_loss


class MSE_loss_uncertainty(nn.Module):

    def __init__(self):
        super(MSE_loss_uncertainty, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        mask = (gt > 0).detach()
        depth = prediction[:, 0:1, :, :]
        conf = torch.abs(prediction[:, 1:, :, :])
        err = depth - gt
        conf_loss = torch.mean(0.5 * err[mask] ** 2 * torch.exp(-conf[mask]) + 0.5 * conf[mask])
        return conf_loss


class MSE_log_loss(nn.Module):

    def __init__(self):
        super(MSE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=0)
        err = torch.log(prediction + 1e-06) - torch.log(gt + 1e-06)
        mask = (gt > 0).detach()
        mae_log_loss = torch.mean(err[mask] ** 2)
        return mae_log_loss


class Huber_loss(nn.Module):

    def __init__(self, delta=10):
        super(Huber_loss, self).__init__()
        self.delta = delta

    def forward(self, outputs, gt, input, epoch=0):
        outputs = outputs[:, 0:1, :, :]
        err = torch.abs(outputs - gt)
        mask = (gt > 0).detach()
        err = err[mask]
        squared_err = 0.5 * err ** 2
        linear_err = err - 0.5 * self.delta
        return torch.mean(torch.where(err < self.delta, squared_err, linear_err))


class Berhu_loss(nn.Module):

    def __init__(self, delta=0.05):
        super(Berhu_loss, self).__init__()
        self.delta = delta

    def forward(self, prediction, gt, epoch=0):
        prediction = prediction[:, 0:1]
        err = torch.abs(prediction - gt)
        mask = (gt > 0).detach()
        err = torch.abs(err[mask])
        c = self.delta * err.max().item()
        squared_err = (err ** 2 + c ** 2) / (2 * c)
        linear_err = err
        return torch.mean(torch.where(err > c, squared_err, linear_err))


class Huber_delta1_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, gt, input):
        mask = (gt > 0).detach().float()
        loss = F.smooth_l1_loss(prediction * mask, gt * mask, reduction='none')
        return torch.mean(loss)


class Disparity_Loss(nn.Module):

    def __init__(self, order=2):
        super(Disparity_Loss, self).__init__()
        self.order = order

    def forward(self, prediction, gt):
        mask = (gt > 0).detach()
        gt = gt[mask]
        gt = 1.0 / gt
        prediction = prediction[mask]
        err = torch.abs(prediction - gt)
        err = torch.mean(err ** self.order)
        return err


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        chans = 32 if in_channels > 16 else 16
        self.initial_block = DownsamplerBlock(in_channels, chans)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(chans, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = UpsamplerBlock(128, 64)
        self.layer2 = non_bottleneck_1d(64, 0, 1)
        self.layer3 = non_bottleneck_1d(64, 0, 1)
        self.layer4 = UpsamplerBlock(64, 32)
        self.layer5 = non_bottleneck_1d(32, 0, 1)
        self.layer6 = non_bottleneck_1d(32, 0, 1)
        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        em2 = output
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        em1 = output
        output = self.output_conv(output)
        return output, em1, em2


class Net(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))


class hourglass_1(nn.Module):

    def __init__(self, channels_in):
        super(hourglass_1, self).__init__()
        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1), nn.ReLU(inplace=True))
        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)
        self.conv3 = nn.Sequential(convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=2, pad=1, dilation=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=1, pad=1, dilation=1))
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in * 4, channels_in * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(channels_in * 2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in * 2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, em1), 1)
        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, em2), 1)
        out = self.conv5(x_prime)
        out = self.conv6(out)
        return out, x, x_prime


class hourglass_2(nn.Module):

    def __init__(self, channels_in):
        super(hourglass_2, self).__init__()
        self.conv1 = nn.Sequential(convbn(channels_in, channels_in * 2, kernel_size=3, stride=2, pad=1, dilation=1), nn.BatchNorm2d(channels_in * 2), nn.ReLU(inplace=True))
        self.conv2 = convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=1, pad=1, dilation=1)
        self.conv3 = nn.Sequential(convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=2, pad=1, dilation=1), nn.BatchNorm2d(channels_in * 2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn(channels_in * 2, channels_in * 4, kernel_size=3, stride=1, pad=1, dilation=1))
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in * 4, channels_in * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(channels_in * 2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in * 2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)
        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)
        out = self.conv5(x_prime)
        out = self.conv6(out)
        return out


class uncertainty_net(nn.Module):

    def __init__(self, in_channels, out_channels=1, thres=15):
        super(uncertainty_net, self).__init__()
        out_chan = 2
        combine = 'concat'
        self.combine = combine
        self.in_channels = in_channels
        out_channels = 3
        self.depthnet = Net(in_channels=in_channels, out_channels=out_channels)
        local_channels_in = 2 if self.combine == 'concat' else 1
        self.convbnrelu = nn.Sequential(convbn(local_channels_in, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.hourglass1 = hourglass_1(32)
        self.hourglass2 = hourglass_2(32)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))
        self.activation = nn.ReLU(inplace=True)
        self.thres = thres
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, epoch=50):
        if self.in_channels > 1:
            rgb_in = input[:, 1:, :, :]
            lidar_in = input[:, 0:1, :, :]
        else:
            lidar_in = input
        embedding0, embedding1, embedding2 = self.depthnet(input)
        global_features = embedding0[:, 0:1, :, :]
        precise_depth = embedding0[:, 1:2, :, :]
        conf = embedding0[:, 2:, :, :]
        if self.combine == 'concat':
            input = torch.cat((lidar_in, global_features), 1)
        elif self.combine == 'add':
            input = lidar_in + global_features
        elif self.combine == 'mul':
            input = lidar_in * global_features
        elif self.combine == 'sigmoid':
            input = lidar_in * nn.Sigmoid()(global_features)
        else:
            input = lidar_in
        out = self.convbnrelu(input)
        out1, embedding3, embedding4 = self.hourglass1(out, embedding1, embedding2)
        out1 = out1 + out
        out2 = self.hourglass2(out1, embedding3, embedding4)
        out2 = out2 + out
        out = self.fuse(out2)
        lidar_out = out
        lidar_to_depth, lidar_to_conf = torch.chunk(out, 2, dim=1)
        lidar_to_conf, conf = torch.chunk(self.softmax(torch.cat((lidar_to_conf, conf), 1)), 2, dim=1)
        out = conf * precise_depth + lidar_to_conf * lidar_to_depth
        return out, lidar_out, precise_depth, global_features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Berhu_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (Disparity_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Huber_delta1_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Huber_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MAE_log_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MAE_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSE_log_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSE_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (UpsamplerBlock,
     lambda: ([], {'ninput': 4, 'noutput': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hourglass_2,
     lambda: ([], {'channels_in': 4}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 8, 32, 32]), torch.rand([4, 16, 16, 16])], {}),
     True),
    (non_bottleneck_1d,
     lambda: ([], {'chann': 4, 'dropprob': 0.5, 'dilated': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (uncertainty_net,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_wvangansbeke_Sparse_Depth_Completion(_paritybench_base):
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

    def test_014(self):
        self._check(*TESTCASES[14])

