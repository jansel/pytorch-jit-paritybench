import sys
_module = sys.modules[__name__]
del sys
co_transforms = _module
datasets = _module
listdataset = _module
scenelistdataset = _module
stillbox = _module
loss = _module
DepthNet = _module
models = _module
utils = _module
run_inference = _module
terminal_logger = _module
train = _module
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


import torch.nn as nn


import torch.nn.functional as F


import torch


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


import numpy as np


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


def predict_depth(in_planes, with_confidence):
    return nn.Conv2d(in_planes, 2 if with_confidence else 1, kernel_size=3,
        stride=1, padding=1, bias=True)


def adaptative_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)


def init_modules(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            xavier_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)


def post_process_depth(depth, activation_function=None, clamp=False):
    if activation_function is not None:
        depth = activation_function(depth)
    if clamp:
        depth = depth.clamp(10, 60)
    return depth[:, (0)]


def conv(in_planes, out_planes, stride=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, bias=False), nn.BatchNorm2d(
            out_planes, eps=0.001), nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, bias=True), nn.ReLU(inplace=True))


def deconv(in_planes, out_planes, batch_norm=False):
    if batch_norm:
        return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes,
            kernel_size=4, stride=2, padding=1, bias=True), nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(out_planes, eps=0.001), nn.ReLU(
            inplace=True))
    else:
        return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes,
            kernel_size=4, stride=2, padding=1, bias=True), nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1,
            bias=True), nn.ReLU(inplace=True))


class DepthNet(nn.Module):

    def __init__(self, batch_norm=False, with_confidence=False, clamp=False,
        depth_activation=None):
        super(DepthNet, self).__init__()
        self.clamp = clamp
        if depth_activation == 'elu':
            self.depth_activation = lambda x: nn.functional.elu(x) + 1
        else:
            self.depth_activation = depth_activation
        self.conv1 = conv(6, 32, stride=2, batch_norm=batch_norm)
        self.conv2 = conv(32, 64, stride=2, batch_norm=batch_norm)
        self.conv3 = conv(64, 128, stride=2, batch_norm=batch_norm)
        self.conv3_1 = conv(128, 128, batch_norm=batch_norm)
        self.conv4 = conv(128, 256, stride=2, batch_norm=batch_norm)
        self.conv4_1 = conv(256, 256, batch_norm=batch_norm)
        self.conv5 = conv(256, 256, stride=2, batch_norm=batch_norm)
        self.conv5_1 = conv(256, 256, batch_norm=batch_norm)
        self.conv6 = conv(256, 512, stride=2, batch_norm=batch_norm)
        self.conv6_1 = conv(512, 512, batch_norm=batch_norm)
        self.deconv5 = deconv(512, 256, batch_norm=batch_norm)
        self.deconv4 = deconv(513, 128, batch_norm=batch_norm)
        self.deconv3 = deconv(385, 64, batch_norm=batch_norm)
        self.deconv2 = deconv(193, 32, batch_norm=batch_norm)
        self.predict_depth6 = predict_depth(512, with_confidence)
        self.predict_depth5 = predict_depth(513, with_confidence)
        self.predict_depth4 = predict_depth(385, with_confidence)
        self.predict_depth3 = predict_depth(193, with_confidence)
        self.predict_depth2 = predict_depth(97, with_confidence)
        self.upsampled_depth6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias
            =False)
        self.upsampled_depth5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias
            =False)
        self.upsampled_depth4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias
            =False)
        self.upsampled_depth3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias
            =False)
        init_modules(self)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        out6 = self.predict_depth6(out_conv6)
        depth6 = post_process_depth(out6, clamp=self.clamp,
            activation_function=self.depth_activation)
        depth6_up = self.upsampled_depth6_to_5(out6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = adaptative_cat(out_conv5, out_deconv5, depth6_up)
        out5 = self.predict_depth5(concat5)
        depth5 = post_process_depth(out5, clamp=self.clamp,
            activation_function=self.depth_activation)
        depth5_up = self.upsampled_depth5_to_4(out5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = adaptative_cat(out_conv4, out_deconv4, depth5_up)
        out4 = self.predict_depth4(concat4)
        depth4 = post_process_depth(out4, clamp=self.clamp,
            activation_function=self.depth_activation)
        depth4_up = self.upsampled_depth4_to_3(out4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = adaptative_cat(out_conv3, out_deconv3, depth4_up)
        out3 = self.predict_depth3(concat3)
        depth3 = post_process_depth(out3, clamp=self.clamp,
            activation_function=self.depth_activation)
        depth3_up = self.upsampled_depth3_to_2(out3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = adaptative_cat(out_conv2, out_deconv2, depth3_up)
        out2 = self.predict_depth2(concat2)
        depth2 = post_process_depth(out2, clamp=self.clamp,
            activation_function=self.depth_activation)
        if self.training:
            return [depth2, depth3, depth4, depth5, depth6]
        else:
            return depth2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ClementPinard_DepthNet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DepthNet(*[], **{}), [torch.rand([4, 6, 64, 64])], {})

