import sys
_module = sys.modules[__name__]
del sys
datasets = _module
shapenet = _module
test = _module
Progbar = _module
loggers = _module
models = _module
depth_pred_with_sph_inpaint = _module
genre_full_model = _module
marrnet = _module
marrnet1 = _module
marrnet2 = _module
marrnetbase = _module
netinterface = _module
shapehd = _module
wgangp = _module
networks = _module
networks = _module
revresnet = _module
uresnet = _module
options = _module
options_test = _module
options_train = _module
test = _module
toolbox = _module
build = _module
calc_prob = _module
functions = _module
calc_prob = _module
setup = _module
build = _module
cam_bp = _module
cam_back_projection = _module
get_surface_mask = _module
sperical_to_tdf = _module
Spherical_backproj = _module
modules = _module
camera_backprojection_module = _module
build = _module
nnd = _module
nnd = _module
test = _module
spherical_proj = _module
train = _module
util = _module
util_cam_para = _module
util_camera = _module
util_img = _module
util_io = _module
util_loadlib = _module
util_print = _module
util_reproj = _module
util_sph = _module
util_voxel = _module
visualizer = _module

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


from scipy.io import loadmat


import torch.utils.data as data


import torch


import torch.nn as nn


import torch.nn.functional as F


from scipy.ndimage.morphology import binary_erosion


from torch import nn


import time


import torch.optim as optim


from torch.nn import init


from torch import FloatTensor


from torch import tensor


from time import time


from torch import cat


from torchvision.models import resnet18


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.autograd import Variable


from torch.nn import Module


import pandas as pd


from copy import deepcopy


import collections


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, output_padding=output_padding)


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBasicBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes, stride=stride, output_padding=1 if stride > 1 else 0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class RevResNet(nn.Module):

    def __init__(self, block, layers, planes, inplanes=None, out_planes=5):
        """
        planes: # output channels for each block
        inplanes: # input channels for the input at each layer
            If missing, it will be inferred.
        """
        if inplanes is None:
            inplanes = [512]
        self.inplanes = inplanes[0]
        super(RevResNet, self).__init__()
        inplanes_after_blocks = inplanes[4] if len(inplanes) > 4 else planes[3]
        self.deconv1 = nn.ConvTranspose2d(inplanes_after_blocks, planes[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(planes[3], out_planes, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=2)
        if len(inplanes) > 1:
            self.inplanes = inplanes[1]
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        if len(inplanes) > 2:
            self.inplanes = inplanes[2]
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        if len(inplanes) > 3:
            self.inplanes = inplanes[3]
        self.layer4 = self._make_layer(block, planes[3], layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(nn.ConvTranspose2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False, output_padding=1 if stride > 1 else 0), nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x


def revuresnet18(**kwargs):
    """
    Reverse ResNet-18 compatible with the U-Net setting
    """
    model = RevResNet(RevBasicBlock, [2, 2, 2, 2], [256, 128, 64, 64], inplanes=[512, 512, 256, 128, 128], **kwargs)
    return model


class Net(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        module_list.append(nn.Sequential(resnet.conv1 if input_planes == 3 else in_conv, resnet.bn1, resnet.relu, resnet.maxpool))
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(nn.Sequential(revresnet.deconv1, revresnet.bn1, revresnet.relu, revresnet.deconv2))
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


class CameraBackProjection(Function):

    @staticmethod
    def forward(ctx, depth_t, fl, cam_dist, res=128):
        assert depth_t.dim() == 4
        assert fl.dim() == 2 and fl.size(1) == depth_t.size(1)
        assert cam_dist.dim() == 2 and cam_dist.size(1) == depth_t.size(1)
        assert cam_dist.size(0) == depth_t.size(0)
        assert fl.size(0) == depth_t.size(0)
        in_shape = depth_t.shape
        cnt = depth_t.new(in_shape[0], in_shape[1], res, res, res).zero_()
        tdf = depth_t.new(in_shape[0], in_shape[1], res, res, res).zero_() + 1 / res
        cam_bp_lib.back_projection_forward(depth_t, cam_dist, fl, tdf, cnt)
        ctx.save_for_backward(depth_t, fl, cam_dist)
        ctx.cnt_forward = cnt
        ctx.depth_shape = in_shape
        return tdf

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        depth_t, fl, cam_dist = ctx.saved_tensors
        cnt = ctx.cnt_forward
        grad_depth = grad_output.new(ctx.depth_shape).zero_()
        grad_fl = grad_output.new(ctx.depth_shape[0], ctx.depth_shape[1]).zero_()
        grad_camdist = grad_output.new(ctx.depth_shape[0], ctx.depth_shape[1]).zero_()
        cam_bp_lib.back_projection_backward(depth_t, fl, cam_dist, cnt, grad_output, grad_depth, grad_camdist, grad_fl)
        return grad_depth, grad_fl, grad_camdist, None


class Camera_back_projection_layer(nn.Module):

    def __init__(self, res=128):
        super(Camera_back_projection_layer, self).__init__()
        assert res == 128
        self.res = 128

    def forward(self, depth_t, fl=418.3, cam_dist=2.2, shift=True):
        n = depth_t.size(0)
        if type(fl) == float:
            fl_v = fl
            fl = torch.FloatTensor(n, 1)
            fl.fill_(fl_v)
        if type(cam_dist) == float:
            cmd_v = cam_dist
            cam_dist = torch.FloatTensor(n, 1)
            cam_dist.fill_(cmd_v)
        df = CameraBackProjection.apply(depth_t, fl, cam_dist, self.res)
        return self.shift_tdf(df) if shift else df

    @staticmethod
    def shift_tdf(input_tdf, res=128):
        out_tdf = 1 - res * input_tdf
        return out_tdf


class ImageEncoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, input_nc, encode_dims=200):
        super().__init__()
        resnet_m = resnet18(pretrained=True)
        resnet_m.conv1 = nn.Conv2d(input_nc, 64, 7, stride=2, padding=3, bias=False)
        resnet_m.avgpool = nn.AdaptiveAvgPool2d(1)
        resnet_m.fc = nn.Linear(512, encode_dims)
        self.main = nn.Sequential(resnet_m)

    def forward(self, x):
        return self.main(x)


def batchnorm3d(n_feat):
    return nn.BatchNorm3d(n_feat, eps=1e-05, momentum=0.1, affine=True)


def deconv3d_2x(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1, groups=1, bias=bias)


def deconv3d_add3(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1, groups=1, bias=bias)


def relu():
    return nn.ReLU(inplace=True)


class VoxelDecoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, n_dims=200, nf=512):
        super().__init__()
        self.main = nn.Sequential(deconv3d_add3(n_dims, nf, True), batchnorm3d(nf), relu(), deconv3d_2x(nf, nf // 2, True), batchnorm3d(nf // 2), relu(), nn.Sequential(), nn.Sequential(), deconv3d_2x(nf // 2, nf // 4, True), batchnorm3d(nf // 4), relu(), deconv3d_2x(nf // 4, nf // 8, True), batchnorm3d(nf // 8), relu(), deconv3d_2x(nf // 8, nf // 16, True), batchnorm3d(nf // 16), relu(), deconv3d_2x(nf // 16, 1, True))

    def forward(self, x):
        x_vox = x.view(x.size(0), -1, 1, 1, 1)
        return self.main(x_vox)


class VoxelGenerator(nn.Module):

    def __init__(self, nz=200, nf=64, bias=False, res=128):
        super().__init__()
        layers = [deconv3d_add3(nz, nf * 8, bias), batchnorm3d(nf * 8), relu(), deconv3d_2x(nf * 8, nf * 4, bias), batchnorm3d(nf * 4), relu(), deconv3d_2x(nf * 4, nf * 2, bias), batchnorm3d(nf * 2), relu(), deconv3d_2x(nf * 2, nf, bias), batchnorm3d(nf), relu()]
        if res == 64:
            layers.append(deconv3d_2x(nf, 1, bias))
        elif res == 128:
            layers += [deconv3d_2x(nf, nf, bias), batchnorm3d(nf), relu(), deconv3d_2x(nf, 1, bias)]
        else:
            raise NotImplementedError(res)
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def conv3d_half(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1, groups=1, bias=bias)


def conv3d_minus3(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1, groups=1, bias=bias)


def relu_leaky():
    return nn.LeakyReLU(0.2, inplace=True)


class VoxelDiscriminator(nn.Module):

    def __init__(self, nf=64, bias=False, res=128):
        super().__init__()
        layers = [conv3d_half(1, nf, bias), relu_leaky(), conv3d_half(nf, nf * 2, bias), relu_leaky(), conv3d_half(nf * 2, nf * 4, bias), relu_leaky(), conv3d_half(nf * 4, nf * 8, bias), relu_leaky(), conv3d_minus3(nf * 8, 1, bias)]
        if res == 64:
            pass
        elif res == 128:
            extra_layers = [conv3d_half(nf, nf, bias), relu_leaky()]
            layers = layers[:2] + extra_layers + layers[2:]
        else:
            raise NotImplementedError(res)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        y = self.main(x)
        return y.view(-1, 1).squeeze(1)


class Conv3d_block(nn.Module):

    def __init__(self, ncin, ncout, kernel_size, stride, pad, dropout=False):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(ncin, ncout, kernel_size, stride, pad), nn.BatchNorm3d(ncout), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Deconv3d_skip(nn.Module):

    def __init__(self, ncin, ncout, kernel_size, stride, pad, extra=0, is_activate=True):
        super(Deconv3d_skip, self).__init__()
        if is_activate:
            self.net = nn.Sequential(nn.ConvTranspose3d(ncin, ncout, kernel_size, stride, pad, extra), nn.BatchNorm3d(ncout), nn.LeakyReLU())
        else:
            self.net = nn.ConvTranspose3d(ncin, ncout, kernel_size, stride, pad, extra)

    def forward(self, x, skip_in):
        y = cat((x, skip_in), dim=1)
        return self.net(y)


class Unet_3D(nn.Module):

    def __init__(self, nf=20, in_channel=2, no_linear=False):
        super(Unet_3D, self).__init__()
        self.nf = nf
        self.enc1 = Conv3d_block(in_channel, nf, 8, 2, 3)
        self.enc2 = Conv3d_block(nf, 2 * nf, 4, 2, 1)
        self.enc3 = Conv3d_block(2 * nf, 4 * nf, 4, 2, 1)
        self.enc4 = Conv3d_block(4 * nf, 8 * nf, 4, 2, 1)
        self.enc5 = Conv3d_block(8 * nf, 16 * nf, 4, 2, 1)
        self.enc6 = Conv3d_block(16 * nf, 32 * nf, 4, 1, 0)
        self.full_conv_block = nn.Sequential(nn.Linear(32 * nf, 32 * nf), nn.LeakyReLU())
        self.dec1 = Deconv3d_skip(32 * 2 * nf, 16 * nf, 4, 1, 0, 0)
        self.dec2 = Deconv3d_skip(16 * 2 * nf, 8 * nf, 4, 2, 1, 0)
        self.dec3 = Deconv3d_skip(8 * 2 * nf, 4 * nf, 4, 2, 1, 0)
        self.dec4 = Deconv3d_skip(4 * 2 * nf, 2 * nf, 4, 2, 1, 0)
        self.dec5 = Deconv3d_skip(4 * nf, nf, 8, 2, 3, 0)
        self.dec6 = Deconv3d_skip(2 * nf, 1, 4, 2, 1, 0, is_activate=False)
        self.no_linear = no_linear

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        if not self.no_linear:
            flatten = enc6.view(enc6.size()[0], self.nf * 32)
            bottleneck = self.full_conv_block(flatten)
            bottleneck = bottleneck.view(enc6.size()[0], self.nf * 32, 1, 1, 1)
            dec1 = self.dec1(bottleneck, enc6)
        else:
            dec1 = self.dec1(enc6, enc6)
        dec2 = self.dec2(dec1, enc5)
        dec3 = self.dec3(dec2, enc4)
        dec4 = self.dec4(dec3, enc3)
        dec5 = self.dec5(dec4, enc2)
        out = self.dec6(dec5, enc1)
        return out


class ViewAsLinear(nn.Module):

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBottleneck, self).__init__()
        bottleneck_planes = int(inplanes / 4)
        self.deconv1 = nn.ConvTranspose2d(inplanes, bottleneck_planes, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv2 = nn.ConvTranspose2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv3 = nn.ConvTranspose2d(bottleneck_planes, planes, kernel_size=1, bias=False, stride=stride, output_padding=1 if stride > 0 else 0)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class Net_inpaint(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        module_list.append(nn.Sequential(resnet.conv1 if input_planes == 3 else in_conv, resnet.bn1, resnet.relu, resnet.maxpool))
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=2, padding=3, bias=False, output_padding=0)
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(nn.Sequential(revresnet.deconv1, revresnet.bn1, revresnet.relu, self.deconv2))
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


class SphericalBackProjection(Function):

    @staticmethod
    def forward(ctx, spherical, grid, res=128):
        assert spherical.dim() == 4
        assert grid.dim() == 5
        assert spherical.size(0) == grid.size(0)
        assert spherical.size(1) == grid.size(1)
        assert spherical.size(2) == grid.size(2)
        assert spherical.size(3) == grid.size(3)
        assert grid.size(4) == 3
        in_shape = spherical.shape
        cnt = spherical.new(in_shape[0], in_shape[1], res, res, res).zero_()
        tdf = spherical.new(in_shape[0], in_shape[1], res, res, res).zero_()
        cam_bp_lib.spherical_back_proj_forward(spherical, grid, tdf, cnt)
        ctx.save_for_backward(spherical.detach(), grid, cnt)
        ctx.depth_shape = in_shape
        return tdf, cnt

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_phony):
        assert not np.isnan(torch.sum(grad_output.detach()))
        spherical, grid, cnt = ctx.saved_tensors
        grad_depth = grad_output.new(ctx.depth_shape).zero_()
        cam_bp_lib.spherical_back_proj_backward(spherical, grid, cnt, grad_output, grad_depth)
        try:
            assert not np.isnan(torch.sum(grad_depth))
        except:
            pdb.set_trace()
        return grad_depth, None, None


class camera_backprojection(nn.Module):

    def __init__(self, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = CameraBackProjection()

    def forward(self, depth, fl, camdist):
        return self.backprojection_layer(depth, fl, camdist, self.voxel_res)


class spherical_backprojection(nn.Module):

    def __init__(self, grid, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = SphericalBackProjection()
        assert type(grid) == torch.FloatTensor
        self.grid = Variable(grid)

    def forward(self, spherical):
        return self.backprojection_layer(spherical, self.grid, self.vox_res)


class NNDFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert xyz1.dim() == 3 and xyz2.dim() == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
        assert xyz1.type().endswith('FloatTensor') and xyz2.type().endswith('FloatTensor'), 'only FloatTensor are supported for NNDistance'
        assert xyz1.is_contiguous() and xyz2.is_contiguous()
        ctx.is_cuda = xyz1.is_cuda
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1
            dist2 = dist2
            idx1 = idx1
            idx2 = idx2
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    @once_differentiable
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        """
        Note that this function needs gradidx placeholders
        """
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()
        assert idx1.is_contiguous()
        assert idx2.is_contiguous()
        assert graddist1.type().endswith('FloatTensor') and graddist2.type().endswith('FloatTensor'), 'only FloatTensor are supported for NNDistance'
        gradxyz1 = xyz1.new(xyz1.size())
        gradxyz2 = xyz1.new(xyz2.size())
        if not graddist1.is_cuda:
            my_lib.nnd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            my_lib.nnd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


def nndistance(xyz1, xyz2):
    if xyz1.size(2) != 3:
        xyz1 = xyz1.transpose(1, 2)
    if xyz2.size(2) != 3:
        xyz2 = xyz2.transpose(1, 2)
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    dist1, dist2, _, _ = NNDFunction.apply(xyz1, xyz2)
    return dist1, dist2


class NNDModule(Module):

    def forward(self, input1, input2):
        return nndistance(input1, input2)


class CalcStopProb(Function):

    @staticmethod
    def forward(ctx, prob_in):
        assert prob_in.dim() == 5
        assert prob_in.dtype == torch.float32
        stop_prob = prob_in.new(prob_in.shape)
        stop_prob.zero_()
        calc_prob_lib.calc_prob_forward(prob_in, stop_prob)
        ctx.save_for_backward(prob_in, stop_prob)
        return stop_prob

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_in):
        prob_in, stop_prob = ctx.saved_tensors
        grad_out = grad_in.new(grad_in.shape)
        grad_out.zero_()
        stop_prob_weighted = stop_prob * grad_in
        calc_prob_lib.calc_prob_backward(prob_in, stop_prob_weighted, grad_out)
        return grad_out


class render_spherical(torch.nn.Module):

    def __init__(self, sph_res=128, z_res=256):
        super().__init__()
        self.sph_res = sph_res
        self.z_res = z_res
        self.gen_grid()
        self.calc_stop_prob = CalcStopProb().apply

    def gen_grid(self):
        res = self.sph_res
        z_res = self.z_res
        pi = np.pi
        phi = np.linspace(0, 180, res * 2 + 1)[1::2]
        theta = np.linspace(0, 360, res + 1)[:-1]
        grid = np.zeros([res, res, 3])
        for idp, p in enumerate(phi):
            for idt, t in enumerate(theta):
                grid[idp, idt, 2] = np.cos(p * pi / 180)
                proj = np.sin(p * pi / 180)
                grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
                grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
        grid = np.reshape(grid * 2, (res, res, 3))
        alpha = np.zeros([1, 1, z_res, 1])
        alpha[(0), (0), :, (0)] = np.linspace(0, 1, z_res)
        grid = grid[:, :, (np.newaxis), :]
        grid = grid * (1 - alpha)
        grid = torch.from_numpy(grid).float()
        depth_weight = torch.linspace(0, 1, self.z_res)
        self.register_buffer('depth_weight', depth_weight)
        self.register_buffer('grid', grid)

    def forward(self, vox):
        grid = self.grid.expand(vox.shape[0], -1, -1, -1, -1)
        vox = vox.permute(0, 1, 4, 3, 2)
        prob_sph = torch.nn.functional.grid_sample(vox, grid)
        prob_sph = torch.clamp(prob_sph, 1e-05, 1 - 1e-05)
        sph_stop_prob = self.calc_stop_prob(prob_sph)
        exp_depth = torch.matmul(sph_stop_prob, self.depth_weight)
        back_groud_prob = torch.prod(1.0 - prob_sph, dim=4)
        back_groud_prob = back_groud_prob * 1.0
        exp_depth = exp_depth + back_groud_prob
        return exp_depth


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv3d_block,
     lambda: ([], {'ncin': 4, 'ncout': 4, 'kernel_size': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Deconv3d_skip,
     lambda: ([], {'ncin': 4, 'ncout': 4, 'kernel_size': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64]), torch.rand([4, 3, 64, 64, 64])], {}),
     True),
    (ImageEncoder,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RevBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ViewAsLinear,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VoxelDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 200])], {}),
     True),
]

class Test_xiumingzhang_GenRe_ShapeHD(_paritybench_base):
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

