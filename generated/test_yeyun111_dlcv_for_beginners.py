import sys
_module = sys.modules[__name__]
del sys
downscale = _module
food_augmentation = _module
gen_label_list = _module
remove_dups_from_list = _module
remove_invalid_images = _module
sample_val = _module
kaoya_shuizhurou_roc_auc = _module
make_confusion_matrix = _module
recognize_food = _module
sort_kaoya_by_pred_prob = _module
visualize_activation = _module
gen_pairwise_imglist = _module
visualize_result = _module
bar_n_pie_chart = _module
fit_data = _module
scatter_3d = _module
surface_3d = _module
three_doors = _module
bbox_labeling = _module
detection_anno_bbox2voc = _module
image_augmentation = _module
run_augmentation = _module
run_augmentation_pool_map = _module
gen_hdf5 = _module
simple_mlp_test = _module
simple_mlp_train = _module
gen_data = _module
simple_mlp = _module
recognize_digit = _module
convert_mnist = _module
gen_caffe_imglist = _module
gen_mxnet_imglist = _module
benchmark_model = _module
score_model = _module
train_lenet5 = _module
gen_label = _module
make_noises = _module
predict = _module
visualize_conv1_kernels = _module
adversarial_example_demo = _module
argparser = _module
cgan_demo = _module
gan_demo = _module
networks = _module
sampler = _module
visualizer = _module
crop_n_resize = _module
gen_mosaic = _module
networks = _module
loss_visualizer = _module
main = _module
networks = _module
utils = _module
fuse_model = _module
gen_img_list = _module
rename_n_freeze_layers = _module

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


import torch.optim as optim


import numpy


import torch


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


from scipy.stats import chi


import torch.utils.data


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import logging


import torchvision


import torch.functional as F


from torchvision.models import resnet


from torchvision.models.resnet import conv3x3


import random


from torchvision.datasets.folder import *


from torch.optim import SGD


from torch.optim import Adadelta


from torch.optim import Adam


from torch.optim import Adagrad


from torch.optim import RMSprop


from torch.optim import ASGD


class SimpleMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        return F.sigmoid(self.map2(x))


class DeepMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        return F.sigmoid(self.map3(x))


class NetG(nn.Module):

    def __init__(self, ngf, nz, nc, ngpu):
        super(NetG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class NetD(nn.Module):

    def __init__(self, ndf, nc, ngpu):
        super(NetD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class UNetConvBlock(nn.Module):

    def __init__(self, input_nch, output_nch, kernel_size=3, activation=F.leaky_relu, use_bn=True, same_conv=True):
        super(UNetConvBlock, self).__init__()
        padding = kernel_size // 2 if same_conv else 0
        self.conv0 = nn.Conv2d(input_nch, output_nch, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(output_nch, output_nch, kernel_size, padding=padding)
        self.act = activation
        self.batch_norm = nn.BatchNorm2d(output_nch) if use_bn else None

    def forward(self, x):
        x = self.conv0(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.act(x)


class UNet(nn.Module):

    def __init__(self, conv_channels, input_nch=3, output_nch=2, use_bn=True):
        super(UNet, self).__init__()
        self.n_stages = len(conv_channels)
        down_convs = []
        up_convs = []
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_nch = input_nch
        for i, out_nch in enumerate(conv_channels):
            down_convs.append(UNetConvBlock(in_nch, out_nch, use_bn=use_bn))
            up_conv_in_ch = 2 * out_nch if i < self.n_stages - 1 else out_nch
            up_conv_out_ch = out_nch if i == 0 else in_nch
            up_convs.insert(0, UNetConvBlock(up_conv_in_ch, up_conv_out_ch, use_bn=use_bn))
            in_nch = out_nch
        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)
        self.out_conv = nn.Conv2d(conv_channels[0], output_nch, 1)

    def forward(self, x):
        down_sampled_fmaps = []
        for i in range(self.n_stages - 1):
            x = self.down_convs[i](x)
            x = self.max_pooling(x)
            down_sampled_fmaps.insert(0, x)
        x = self.down_convs[self.n_stages - 1](x)
        x = self.up_convs[0](x)
        for i, down_sampled_fmap in enumerate(down_sampled_fmaps):
            x = torch.cat([x, down_sampled_fmap], 1)
            x = self.up_convs[i + 1](x)
            x = F.upsample(x, scale_factor=2, mode='bilinear')
        return self.out_conv(x)


class BasicResBlock(nn.Module):

    def __init__(self, input_nch, output_nch, groups=1):
        super(BasicResBlock, self).__init__()
        self.transform_conv = nn.Conv2d(input_nch, output_nch, 1)
        self.bn1 = nn.BatchNorm2d(output_nch)
        self.conv1 = nn.Conv2d(output_nch, output_nch, 3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(output_nch)
        self.conv2 = nn.Conv2d(output_nch, output_nch, 3, padding=1, groups=groups, bias=False)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transform_conv(x)
        residual = x
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        out += residual
        return out


class TriangleNet(nn.Module):

    def __init__(self, conv_channels, input_nch, output_nch, groups=1):
        super(TriangleNet, self).__init__()
        self.input_nch = input_nch
        self.output_nch = output_nch
        self.pyramid_height = len(conv_channels)
        blocks = [list() for _ in range(self.pyramid_height)]
        for i in range(self.pyramid_height):
            for j in range(i, self.pyramid_height):
                if i == 0 and j == 0:
                    blocks[i].append(BasicResBlock(input_nch, conv_channels[j], groups=groups))
                else:
                    blocks[i].append(BasicResBlock(conv_channels[j - 1], conv_channels[j], groups=groups))
        for i in range(self.pyramid_height):
            blocks[i] = nn.ModuleList(blocks[i])
        self.blocks = nn.ModuleList(blocks)
        self.down_sample = nn.MaxPool2d(3, 2, 1)
        self.up_samples = nn.ModuleList([nn.Upsample(scale_factor=2 ** i, mode='bilinear') for i in range(1, self.pyramid_height)])
        self.channel_out_convs = nn.ModuleList([nn.Conv2d(conv_channels[-1], output_nch, 1) for _ in range(self.pyramid_height)])
        self.out_conv = nn.Conv2d(self.pyramid_height * conv_channels[-1], output_nch, 1)

    def forward(self, x):
        x = [self.blocks[0][0](x)]
        for i in range(1, self.pyramid_height):
            x.append(self.down_sample(x[-1]))
            for j in range(i + 1):
                x[j] = self.blocks[j][i - j](x[j])
        if self.training:
            ms_out = [self.channel_out_convs[i](x[i]) for i in range(self.pyramid_height)]
        x = [x[0]] + [self.up_samples[i - 1](x[i]) for i in range(1, self.pyramid_height)]
        out = self.out_conv(torch.cat(x, 1))
        return [out] + ms_out if self.training else out


class PSPTriangleNet(nn.Module):

    def __init__(self, conv_channels, input_nch, output_nch, groups):
        super(PSPTriangleNet, self).__init__()
        self.input_nch = input_nch
        self.output_nch = output_nch
        self.pyramid_height = len(conv_channels)
        blocks = []
        for i in range(self.pyramid_height - 1):
            if i == 0:
                blocks.append(BasicResBlock(input_nch, conv_channels[i], groups=groups))
            else:
                blocks.append(BasicResBlock(conv_channels[i - 1], conv_channels[i], groups=groups))
        ms_blocks = []
        for i in range(self.pyramid_height):
            ms_blocks.append(BasicResBlock(conv_channels[-2], conv_channels[-1] // self.pyramid_height))
        self.blocks = nn.ModuleList(blocks)
        self.ms_blocks = nn.ModuleList(ms_blocks)
        self.down_samples = nn.ModuleList([nn.MaxPool2d(2 ** i + 1, 2 ** i, 2 ** (i - 1)) for i in range(1, self.pyramid_height)])
        self.up_samples = nn.ModuleList([nn.Upsample(scale_factor=2 ** i, mode='bilinear') for i in range(1, self.pyramid_height)])
        self.channel_out_convs = nn.ModuleList([nn.Conv2d(conv_channels[-1] // self.pyramid_height, output_nch, 1) for _ in range(self.pyramid_height)])
        self.out_conv = nn.Conv2d(conv_channels[-1], output_nch, 1)

    def forward(self, x):
        for i in range(self.pyramid_height - 1):
            x = self.blocks[i](x)
        x = [self.ms_blocks[0](x)] + [self.down_samples[i](self.ms_blocks[i](x)) for i in range(self.pyramid_height - 1)]
        if self.training:
            ms_out = [self.channel_out_convs[i](x[i]) for i in range(self.pyramid_height)]
        x = [x[0]] + [self.up_samples[i - 1](x[i]) for i in range(1, self.pyramid_height)]
        out = self.out_conv(torch.cat(x, 1))
        return [out] + ms_out if self.training else out


class CrossEntropyLoss2D(nn.Module):

    def __init__(self, size_average=True):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss_2d = nn.NLLLoss2d(size_average=size_average)

    def forward(self, outputs, targets):
        return self.nll_loss_2d(F.log_softmax(outputs), targets)


class MSCrossEntropyLoss2D(nn.Module):

    def __init__(self, weights, size_average=True):
        super(MSCrossEntropyLoss2D, self).__init__()
        self.nll_loss_2d = nn.NLLLoss2d(size_average=size_average)
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.weights[0] * self.nll_loss_2d(F.log_softmax(outputs[0]), targets[0])
        for i in range(len(self.weights) - 1):
            loss += self.weights[i + 1] * self.nll_loss_2d(F.log_softmax(outputs[i + 1]), targets[i])
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicResBlock,
     lambda: ([], {'input_nch': 4, 'output_nch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepMLP,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetD,
     lambda: ([], {'ndf': 4, 'nc': 4, 'ngpu': False}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (NetG,
     lambda: ([], {'ngf': 4, 'nz': 4, 'nc': 4, 'ngpu': False}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PSPTriangleNet,
     lambda: ([], {'conv_channels': [4, 4], 'input_nch': 4, 'output_nch': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleMLP,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TriangleNet,
     lambda: ([], {'conv_channels': [4, 4], 'input_nch': 4, 'output_nch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {'conv_channels': [4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UNetConvBlock,
     lambda: ([], {'input_nch': 4, 'output_nch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yeyun111_dlcv_for_beginners(_paritybench_base):
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

