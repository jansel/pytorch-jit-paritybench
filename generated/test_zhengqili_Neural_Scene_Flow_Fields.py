import sys
_module = sys.modules[__name__]
del sys
Q_Slerp = _module
evaluation = _module
load_llff = _module
models = _module
base_model = _module
dist_model = _module
networks_basic = _module
pretrained_networks = _module
poseInterpolator = _module
render_utils = _module
run_nerf = _module
run_nerf_helpers = _module
softsplat = _module
setup = _module
colmap_read_model = _module
core = _module
corr = _module
datasets = _module
extractor = _module
raft = _module
update = _module
utils = _module
augmentor = _module
flow_viz = _module
frame_utils = _module
utils = _module
flow_utils = _module
base_model = _module
blocks = _module
midas_net = _module
transforms = _module
run_flows_video = _module
run_midas = _module
save_poses_nerf = _module

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


import numpy as np


import random


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.autograd import Variable


from torch import nn


from collections import OrderedDict


import itertools


from scipy.ndimage import zoom


import functools


import torch.nn.init as init


from collections import namedtuple


from torchvision import models as tv


from torch.utils.tensorboard import SummaryWriter


import re


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.utils.data as data


from torchvision.transforms import ColorJitter


from scipy import interpolate


import torchvision


import matplotlib


import matplotlib.pyplot as plt


from torchvision.transforms import Compose


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], version='0.1'):
        super(PerceptualLoss, self).__init__()
        None
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids, version=version)
        None
        None

    def forward(self, pred, target, mask=None, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.model.forward(target, pred, mask=mask)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def spatial_average(in_tens, mask=None, keepdim=True):
    if mask is None:
        return in_tens.mean([2, 3], keepdim=keepdim)
    else:
        mask_resized = torch.nn.functional.interpolate(mask[:, 0:1, :, :], size=[in_tens.size(2), in_tens.size(3)])
        num_valid = torch.sum(mask_resized)
        return torch.sum(in_tens * mask_resized) / (num_valid + 1e-08)


def upsample(in_tens, out_HW=(64, 64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    scale_factor_H, scale_factor_W = 1.0 * out_HW[0] / in_H, 1.0 * out_HW[1] / in_W
    return nn.Upsample(scale_factor=(scale_factor_H, scale_factor_W), mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, mask=None, retPerLayer=False):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), mask, keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), mask, keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = util.dssim(1.0 * util.tensor2im(in0.data), 1.0 * util.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var
        return ret_var


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
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
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


class NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.sf_linear = nn.Linear(W, 6)
        self.prob_linear = nn.Linear(W, 2)

    def forward(self, x):
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        sf = nn.functional.tanh(self.sf_linear(h))
        prob = nn.functional.sigmoid(self.prob_linear(h))
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return torch.cat([outputs, sf, prob], dim=-1)


class Rigid_NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(Rigid_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.w_linear = nn.Linear(W, 1)

    def forward(self, x):
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        v = nn.functional.sigmoid(self.w_linear(h))
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return torch.cat([outputs, v], -1)


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]
    while True:
        objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    return strKernel


class _FunctionSoftsplat(torch.autograd.Function):

    @staticmethod
    def forward(self, input, flow):
        self.save_for_backward(input, flow)
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]
        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth
        assert input.is_contiguous() == True
        assert flow.is_contiguous() == True
        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
        if input.is_cuda == True:
            n = output.nelement()
            cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {'input': input, 'flow': flow, 'output': output}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), output.data_ptr()])
        elif input.is_cuda == False:
            raise NotImplementedError()
        return output

    @staticmethod
    def backward(self, gradOutput):
        input, flow = self.saved_tensors
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]
        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth
        assert gradOutput.is_contiguous() == True
        gradInput = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth]) if self.needs_input_grad[0] == True else None
        gradFlow = input.new_zeros([intSamples, intFlowDepth, intFlowHeight, intFlowWidth]) if self.needs_input_grad[1] == True else None
        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_Softsplat_updateGradInput', cupy_kernel('kernel_Softsplat_updateGradInput', {'input': input, 'flow': flow, 'gradOutput': gradOutput, 'gradInput': gradInput, 'gradFlow': gradFlow}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None])
            if gradFlow is not None:
                n = gradFlow.nelement()
                cupy_launch('kernel_Softsplat_updateGradFlow', cupy_kernel('kernel_Softsplat_updateGradFlow', {'input': input, 'flow': flow, 'gradOutput': gradOutput, 'gradInput': gradInput, 'gradFlow': gradFlow}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr()])
        elif input.is_cuda == False:
            raise NotImplementedError()
        return gradInput, gradFlow


def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ['summation', 'average', 'linear', 'softmax']
    if strType == 'average':
        tenInput = torch.cat([tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])], 1)
    elif strType == 'linear':
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)
    elif strType == 'softmax':
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)
    tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)
    if strType != 'summation':
        tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 1e-07)
    return tenOutput


class ModuleSoftsplat(torch.nn.Module):

    def __init__(self, strType):
        super(ModuleSoftsplat, self).__init__()
        self.strType = strType

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class SmallEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class AlternateCorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]
        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))
        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class BasicMotionEncoder(nn.Module):

    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class BasicUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):

    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = 8 * flow.shape[2], 8 * flow.shape[3]
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class RAFT(nn.Module):

    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8)
        coords1 = coords_grid(N, H // 8, W // 8)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions


class BaseModel(torch.nn.Module):

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)
        if 'optimizer' in parameters:
            parameters = parameters['model']
        self.load_state_dict(parameters)


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        return output


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    return _make_resnet_backbone(resnet)


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


def _make_encoder(features, use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)
    return pretrained, scratch


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        None
        super(MidasNet, self).__init__()
        use_pretrained = False if path else True
        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.output_conv = nn.Sequential(nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity())
        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BasicMotionEncoder,
     lambda: ([], {'args': _mock_config(corr_levels=4, corr_radius=4)}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {}),
     True),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     False),
    (FlowHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (MidasNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResidualConvUnit,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (SmallEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SmallMotionEncoder,
     lambda: ([], {'args': _mock_config(corr_levels=4, corr_radius=4)}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {}),
     True),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_zhengqili_Neural_Scene_Flow_Fields(_paritybench_base):
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

