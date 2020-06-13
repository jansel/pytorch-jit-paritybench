import sys
_module = sys.modules[__name__]
del sys
infer_aff = _module
infer_cls = _module
resnet38_aff = _module
resnet38_cls = _module
resnet38d = _module
vgg16_aff = _module
vgg16_cls = _module
vgg16d = _module
imutils = _module
pyutils = _module
torchutils = _module
train_aff = _module
train_cls = _module
data = _module
make_cls_labels = _module

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


import numpy as np


from torch.utils.data import DataLoader


import scipy.misc


import torch.nn.functional as F


from torch.backends import cudnn


import torch.nn as nn


import torch.sparse as sparse


from torch import nn


from torch.utils.data import Dataset


import random


class ResBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
        first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()
        self.same_shape = in_channels == out_channels and stride == 1
        if first_dilation == None:
            first_dilation = dilation
        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
            padding=first_dilation, dilation=first_dilation, bias=False)
        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)
        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3,
            padding=dilation, dilation=dilation, bias=False)
        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1,
                stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):
        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2
        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x
        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)
        x = branch1 + branch2
        if get_x_bn_relu:
            return x, x_bn_relu
        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResBlock_bot(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
        dropout=0.0):
        super(ResBlock_bot, self).__init__()
        self.same_shape = in_channels == out_channels and stride == 1
        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels // 4, 1,
            stride, bias=False)
        self.bn_branch2b1 = nn.BatchNorm2d(out_channels // 4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels // 4, out_channels // 
            2, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn_branch2b2 = nn.BatchNorm2d(out_channels // 2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels // 2, out_channels, 1,
            bias=False)
        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1,
                stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):
        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2
        branch1 = self.conv_branch1(branch2)
        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)
        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)
        x = branch1 + branch2
        if get_x_bn_relu:
            return x, x_bn_relu
        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class Normalize:

    def __init__(self, mean=(122.675, 116.669, 104.008)):
        self.mean = mean

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = imgarr[..., 2] - self.mean[2]
        proc_img[..., 1] = imgarr[..., 1] - self.mean[1]
        proc_img[..., 2] = imgarr[..., 0] - self.mean[0]
        return proc_img


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)
        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)
        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)
        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1,
            dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)
        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)
        self.not_training = [self.conv1a]
        self.normalize = Normalize()
        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):
        x = self.conv1a(x)
        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)
        x, conv5 = self.b6(x, get_x_bn_relu=True)
        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))
        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})

    def train(self, mode=True):
        super().train(mode)
        for layer in self.not_training:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False
            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False
        return


class Net(nn.Module):

    def __init__(self, fc6_dilation=1):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=
            fc6_dilation)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.normalize = Normalize()
        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv5fc']

    def forward_as_dict(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x
        return dict({'conv4': conv4, 'conv5': conv5, 'conv5fc': conv5fc})

    def train(self, mode=True):
        super().train(mode)
        for layer in self.not_training:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False


class BatchNorm2dFixed(torch.nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(BatchNorm2dFixed, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(num_features))
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var,
            self.weight, self.bias, False, eps=self.eps)

    def __call__(self, x):
        return self.forward(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jiwoon_ahn_psa(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ResBlock(*[], **{'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Net(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(BatchNorm2dFixed(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

