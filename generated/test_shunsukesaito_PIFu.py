import sys
_module = sys.modules[__name__]
del sys
apps = _module
crop_img = _module
eval = _module
prt_util = _module
render_data = _module
train_color = _module
train_shape = _module
lib = _module
colab_util = _module
BaseDataset = _module
EvalDataset = _module
TrainDataset = _module
data = _module
ext_transform = _module
geometry = _module
mesh_util = _module
BasePIFuNet = _module
ConvFilters = _module
ConvPIFuNet = _module
DepthNormalizer = _module
HGFilters = _module
HGPIFuNet = _module
ResBlkPIFuNet = _module
SurfaceClassifier = _module
VhullPIFuNet = _module
model = _module
net_util = _module
options = _module
renderer = _module
camera = _module
gl = _module
cam_render = _module
framework = _module
glcontext = _module
init_gl = _module
prt_render = _module
render = _module
glm = _module
mesh = _module
sample_util = _module
sdf = _module
train_util = _module

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


import time


import numpy as np


import random


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torchvision.models.resnet as resnet


import torchvision.models.vgg as vgg


import functools


from torch.nn import init


def index(feat, uv):
    """

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples[:, :, :, (0)]


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx3x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


class BasePIFuNet(nn.Module):

    def __init__(self, projection_mode='orthogonal', error_term=nn.MSELoss()):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        :param error_term:
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'
        self.error_term = error_term
        self.index = index
        self.projection = (orthogonal if projection_mode == 'orthogonal' else
            perspective)
        self.preds = None
        self.labels = None

    def forward(self, points, images, calibs, transforms=None):
        """
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        """
        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """
        None

    def query(self, points, calibs, transforms=None, labels=None):
        """
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        """
        None

    def get_preds(self):
        """
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        """
        return self.preds

    def get_error(self):
        """
        Get the network loss from the last query
        :return: loss term
        """
        return self.error_term(self.preds, self.labels)


class MultiConv(nn.Module):

    def __init__(self, filter_channels):
        super(MultiConv, self).__init__()
        self.filters = []
        for l in range(0, len(filter_channels) - 1):
            self.filters.append(nn.Conv2d(filter_channels[l],
                filter_channels[l + 1], kernel_size=4, stride=2))
            self.add_module('conv%d' % l, self.filters[l])

    def forward(self, image):
        """
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        """
        y = image
        feat_pyramid = [y]
        for i, f in enumerate(self.filters):
            y = f(y)
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            feat_pyramid.append(y)
        return feat_pyramid


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class ResNet(nn.Module):

    def __init__(self, model='resnet18'):
        super(ResNet, self).__init__()
        if model == 'resnet18':
            net = resnet.resnet18(pretrained=True)
        elif model == 'resnet34':
            net = resnet.resnet34(pretrained=True)
        elif model == 'resnet50':
            net = resnet.resnet50(pretrained=True)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv1 = net.conv1
        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, image):
        """
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        """
        y = image
        feat_pyramid = []
        y = self.layer0(y)
        feat_pyramid.append(y)
        y = self.layer1(self.pool(y))
        feat_pyramid.append(y)
        y = self.layer2(y)
        feat_pyramid.append(y)
        y = self.layer3(y)
        feat_pyramid.append(y)
        y = self.layer4(y)
        feat_pyramid.append(y)
        return feat_pyramid


class DepthNormalizer(nn.Module):

    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, z, calibs=None, index_feat=None):
        """
        Normalize z_feature
        :param z_feat: [B, 1, N] depth value for z in the image coordinate system
        :return:
        """
        z_feat = z * (self.opt.loadSize // 2) / self.opt.z_size
        return z_feat


class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.
            features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.
            features, norm=self.norm))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.
                features, self.features, norm=self.norm))
        self.add_module('b3_' + str(level), ConvBlock(self.features, self.
            features, norm=self.norm))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic',
            align_corners=True)
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):

    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack
        self.opt = opt
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2,
                padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2,
                padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv3 = ConvBlock(128, 128, self.opt.norm)
        self.conv4 = ConvBlock(128, 256, self.opt.norm)
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.
                num_hourglass, 256, self.opt.norm))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256,
                self.opt.norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 
                256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32,
                    256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, opt.
                hourglass_dim, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256,
                    kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.
                    hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        normx = x
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules[
                'conv_last' + str(i)](ll)), True)
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, tmpx.detach(), normx


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,
        last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=
        nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
            kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.
            ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                    norm_layer=norm_layer, use_dropout=use_dropout,
                    use_bias=use_bias, last=True)]
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                    norm_layer=norm_layer, use_dropout=use_dropout,
                    use_bias=use_bias)]
        if opt.use_tanh:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class SurfaceClassifier(nn.Module):

    def __init__(self, filter_channels, num_views=1, no_residual=True,
        last_op=None):
        super(SurfaceClassifier, self).__init__()
        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op
        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(filter_channels[l],
                    filter_channels[l + 1], 1))
                self.add_module('conv%d' % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(nn.Conv1d(filter_channels[l] +
                        filter_channels[0], filter_channels[l + 1], 1))
                else:
                    self.filters.append(nn.Conv1d(filter_channels[l],
                        filter_channels[l + 1], 1))
                self.add_module('conv%d' % l, self.filters[l])

    def forward(self, feature):
        """

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        """
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = f(y)
            else:
                y = f(y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2]).mean(dim
                    =1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1],
                    feature.shape[2]).mean(dim=1)
        if self.last_op:
            y = self.last_op(y)
        return y


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd,
        padding=padding, bias=bias)


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(self.bn4, nn.ReLU(True), nn.
                Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias
                =False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_shunsukesaito_PIFu(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasePIFuNet(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConvBlock(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(DepthNormalizer(*[], **{'opt': _mock_config(loadSize=4, z_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(HourGlass(*[], **{'num_modules': _mock_layer(), 'depth': 1, 'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(MultiConv(*[], **{'filter_channels': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ResNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_007(self):
        self._check(ResnetFilter(*[], **{'opt': _mock_config(use_tanh=4)}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SurfaceClassifier(*[], **{'filter_channels': [4, 4]}), [torch.rand([4, 4, 64])], {})

    def test_009(self):
        self._check(Vgg16(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

