import sys
_module = sys.modules[__name__]
del sys
PerceptualSimilarity = _module
compute_dists = _module
compute_dists_dirs = _module
data = _module
base_data_loader = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
models = _module
base_model = _module
dist_model = _module
networks_basic = _module
pretrained_networks = _module
perceptual_loss = _module
test_dataset_model = _module
test_network = _module
train = _module
util = _module
html = _module
visualizer = _module
libs = _module
jpg_demo = _module
measure = _module
niqe = _module
psnr = _module
reco = _module
ssim = _module
ssim_theano = _module
vifp = _module
src = _module
base = _module
base_model = _module
base_trainer = _module
data_loaders = _module
dataset = _module
transform = _module
evaluate = _module
model = _module
blocks = _module
completion_model = _module
free_form_inpainting_archs = _module
i3d = _module
loss = _module
metric = _module
modules = _module
networks = _module
tsm_utils = _module
vgg = _module
video_inpainting_model = _module
FVI_checkpoint_compatibility_transform = _module
check_len = _module
clean_files = _module
clean_saved = _module
compare_results = _module
dataset_json_generator = _module
extract_frames = _module
gen_output_videos = _module
leave_32frames = _module
apply_mask = _module
copy_from_base = _module
gen_foresics_masks = _module
gen_masks = _module
merge_into_base = _module
rename = _module
reorder = _module
trainer = _module
utils = _module
directory_IO = _module
edge = _module
face = _module
flow_utils = _module
logger = _module
logging_config = _module
mask_generators = _module
readers = _module
screen = _module
visualization = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


from torch.autograd import Variable


from collections import namedtuple


from torchvision import models


import scipy


import scipy.misc


import numpy as np


import logging


import math


from torch import nn


import torch.nn.functional as F


import functools


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='net-lin', net='vgg', use_gpu=True):
        None
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        None

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is on, assumes the images are between [0,1] and then scales thembetween [-1, 1]
        If normalize is false, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        dist = self.model.forward_pair(target, pred)
        return dist


class PNet(nn.Module):
    """Pre-trained network with all channels equally weighted by default"""

    def __init__(self, pnet_type='vgg', pnet_rand=False, use_gpu=True):
        super(PNet, self).__init__()
        self.use_gpu = use_gpu
        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand
        self.shift = torch.autograd.Variable(torch.Tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1))
        self.scale = torch.autograd.Variable(torch.Tensor([0.458, 0.448, 0.45]).view(1, 3, 1, 1))
        if self.pnet_type in ['vgg', 'vgg16']:
            self.net = pn.vgg16(pretrained=not self.pnet_rand, requires_grad=False)
        elif self.pnet_type == 'alex':
            self.net = pn.alexnet(pretrained=not self.pnet_rand, requires_grad=False)
        elif self.pnet_type[:-2] == 'resnet':
            self.net = pn.resnet(pretrained=not self.pnet_rand, requires_grad=False, num=int(self.pnet_type[-2:]))
        elif self.pnet_type == 'squeeze':
            self.net = pn.squeezenet(pretrained=not self.pnet_rand, requires_grad=False)
        self.L = self.net.N_slices
        if use_gpu:
            self.net
            self.shift = self.shift
            self.scale = self.scale

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)
        if retPerLayer:
            all_scores = []
        for kk, out0 in enumerate(outs0):
            cur_score = 1.0 - util.cos_sim(outs0[kk], outs1[kk])
            if kk == 0:
                val = 1.0 * cur_score
            else:
                val = val + cur_score
            if retPerLayer:
                all_scores += [cur_score]
        if retPerLayer:
            return val, all_scores
        else:
            return val


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, use_gpu=True, spatial=False, version='0.1'):
        super(PNetLin, self).__init__()
        self.use_gpu = use_gpu
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.version = version
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        if self.pnet_tune:
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=True)
        else:
            self.net = [net_type(pretrained=not self.pnet_rand, requires_grad=False)]
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
        self.shift = torch.autograd.Variable(torch.Tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1))
        self.scale = torch.autograd.Variable(torch.Tensor([0.458, 0.448, 0.45]).view(1, 3, 1, 1))
        if use_gpu:
            if self.pnet_tune:
                self.net
            else:
                self.net[0]
            self.shift = self.shift
            self.scale = self.scale
            self.lin0
            self.lin1
            self.lin2
            self.lin3
            self.lin4
            if self.pnet_type == 'squeeze':
                self.lin5
                self.lin6

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        if self.version == '0.0':
            in0_input = in0
            in1_input = in1
        else:
            in0_input = in0_sc
            in1_input = in1_sc
        if self.pnet_tune:
            outs0 = self.net.forward(in0_input)
            outs1 = self.net.forward(in1_input)
        else:
            outs0 = self.net[0].forward(in0_input)
            outs1 = self.net[0].forward(in1_input)
        feats0 = {}
        feats1 = {}
        diffs = [0] * len(outs0)
        for kk, out0 in enumerate(outs0):
            feats0[kk] = util.normalize_tensor(outs0[kk])
            feats1[kk] = util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.spatial:
            lin_models = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                lin_models.extend([self.lin5, self.lin6])
            res = [lin_models[kk].model(diffs[kk]) for kk in range(len(diffs))]
            return res
        val = torch.mean(torch.mean(self.lin0.model(diffs[0]), dim=3), dim=2)
        val = val + torch.mean(torch.mean(self.lin1.model(diffs[1]), dim=3), dim=2)
        val = val + torch.mean(torch.mean(self.lin2.model(diffs[2]), dim=3), dim=2)
        val = val + torch.mean(torch.mean(self.lin3.model(diffs[3]), dim=3), dim=2)
        val = val + torch.mean(torch.mean(self.lin4.model(diffs[4]), dim=3), dim=2)
        if self.pnet_type == 'squeeze':
            val = val + torch.mean(torch.mean(self.lin5.model(diffs[5]), dim=3), dim=2)
            val = val + torch.mean(torch.mean(self.lin6.model(diffs[6]), dim=3), dim=2)
        val = val.view(val.size()[0], val.size()[1], 1, 1)
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

    def __init__(self, use_gpu=True, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.use_gpu = use_gpu
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()
        self.model = nn.Sequential(*[self.net])
        if self.use_gpu:
            self.net

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        if self.use_gpu:
            per = per
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
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
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
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
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
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
            self.net = models.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = models.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = models.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = models.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = models.resnet152(pretrained=pretrained)
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


class PerceptModel(torch.nn.Module):

    def __init__(self):
        super(PerceptModel, self).__init__()
        self.pred = torch.nn.Parameter(pred.data)

    def forward(self):
        return self.pred


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class Conv3dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = (np.array(kernel_size) - 1) * np.array(dilation) // 2
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)
        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == 'SN':
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f'Norm type {norm} not implemented')
        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = (np.array(kernel_size) - 1) * np.array(dilation) // 2
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        elif norm == 'SN':
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f'Norm type {norm} not implemented')
        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class NN3Dby2D(object):
    """
    Use these inner classes to mimic 3D operation by using 2D operation frame by frame.
    """


    class Base(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, xs):
            dim_length = len(xs.shape)
            if dim_length == 5:
                xs = torch.unbind(xs, dim=2)
                xs = torch.stack([self.layer(x) for x in xs], dim=2)
            elif dim_length == 4:
                xs = self.layer(xs)
            return xs


    class Conv3d(Base):

        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
            super().__init__()
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[1:]
            if isinstance(stride, tuple):
                stride = stride[1:]
            if isinstance(padding, tuple):
                padding = padding[1:]
            if isinstance(dilation, tuple):
                dilation = dilation[1:]
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.weight = self.layer.weight
            self.bias = self.layer.bias
            self.__class__.__name__ = 'Conv3dBy2D'


    class BatchNorm3d(Base):

        def __init__(self, out_channels):
            super().__init__()
            self.layer = nn.BatchNorm2d(out_channels)


    class InstanceNorm3d(Base):

        def __init__(self, out_channels, track_running_stats=True):
            super().__init__()
            self.layer = nn.InstanceNorm2d(out_channels, track_running_stats=track_running_stats)


class NN3Dby2DTSM(NN3Dby2D):


    class Conv3d(NN3Dby2D.Conv3d):

        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.__class__.__name__ = 'Conv3dBy2DTSM'

        def forward(self, xs):
            B, C, L, H, W = xs.shape
            from model.tsm_utils import tsm
            xs_tsm = tsm(xs.transpose(1, 2), L, 'zero').contiguous()
            out = self.layer(xs_tsm.view(B * L, C, H, W))
            _, C_, H_, W_ = out.shape
            return out.view(B, L, C_, H_, W_).transpose(1, 2)
            """

            # Process them frame by frame using 2d layer
            xs_tsm_unbind = torch.unbind(xs_tsm.transpose(1, 2), dim=2)
            xs_rebind = torch.stack([self.layer(x) for x in xs_tsm_unbind], dim=2)
            out = xs_rebind
            """
            """
            # This residual will cause errors due to the inconsistent channel numbers
            if xs_rebind.shape != identity.shape:
                scale_factor = [x / y for (x, y) in zip(xs_rebind.shape[2:5], identity.shape[2:5])]
                out = xs_rebind + F.interpolate(identity, scale_factor=scale_factor)
            else:
                out = xs_rebind + identity

            """
            return out


class VanillaConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'):
        super().__init__()
        if conv_by == '2d':
            self.module = NN3Dby2D
        elif conv_by == '2dtsm':
            self.module = NN3Dby2DTSM
        elif conv_by == '3d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')
        self.padding = tuple((np.array(kernel_size) - 1) * np.array(dilation) // 2) if padding == -1 else padding
        self.featureConv = self.module.Conv3d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, groups, bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = self.module.BatchNorm3d(out_channels)
        elif norm == 'IN':
            self.norm_layer = self.module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == 'SN':
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f'Norm type {norm} not implemented')
        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), scale_factor=2, conv_by='3d'):
        super().__init__()
        self.conv = VanillaConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class PartialConv(VanillaConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by)
        self.mask_sum_conv = self.module.Conv3d(1, 1, kernel_size, stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_sum_conv.weight, 1.0)
        for param in self.mask_sum_conv.parameters():
            param.requires_grad = False
        if norm == 'SN':
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
            raise NotImplementedError(f'Norm type {norm} not implemented')

    def forward(self, input_tuple):
        inp, mask = input_tuple
        output = self.featureConv(mask * inp)
        if self.featureConv.bias is not None:
            output_bias = self.featureConv.bias.view(1, -1, 1, 1, 1)
        else:
            output_bias = torch.zeros([1, 1, 1, 1, 1]).to(inp.device)
        with torch.no_grad():
            mask_sum = self.mask_sum_conv(mask)
        no_update_holes = mask_sum == 0
        mask_sum_no_zero = mask_sum.masked_fill_(no_update_holes, 1.0)
        output = (output - output_bias) / mask_sum_no_zero + output_bias
        output = output.masked_fill_(no_update_holes, 0.0)
        new_mask = torch.ones_like(mask_sum)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        if self.activation is not None:
            output = self.activation(output)
        if self.norm is not None:
            output = self.norm_layer(output)
        return output, new_mask


class PartialDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), scale_factor=2, conv_by='3d'):
        super().__init__()
        self.conv = PartialConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, input_tuple):
        inp, mask = input_tuple
        inp_resized = F.interpolate(inp, scale_factor=(1, self.scale_factor, self.scale_factor))
        with torch.no_grad():
            mask_resized = F.interpolate(mask, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv((inp_resized, mask_resized))


class VideoCN(nn.Module):

    def __init__(self, df=16):
        super().__init__()
        conv_kargs = {'norm': 'BN', 'activation': nn.ReLU(inplace=True)}
        conv_kargs_with_pad = {'norm': 'BN', 'activation': nn.ReLU(inplace=True), 'padding': -1}
        self.conv1 = Conv3dBlock(4, df, 5, **conv_kargs_with_pad)
        self.conv2 = Conv3dBlock(df, df * 2, 3, stride=(1, 2, 2), **conv_kargs_with_pad)
        self.conv3 = Conv3dBlock(df * 2, df * 4, 3, **conv_kargs_with_pad)
        self.conv4 = Conv3dBlock(df * 4, df * 8, 3, stride=(1, 2, 2), **conv_kargs_with_pad)
        self.conv5 = Conv3dBlock(df * 8, df * 16, 3, dilation=(1, 2, 2), **conv_kargs_with_pad)
        self.conv6 = Conv3dBlock(df * 16, df * 16, 3, dilation=(1, 4, 4), **conv_kargs_with_pad)
        self.conv7 = Conv3dBlock(df * 16, df * 16, 3, dilation=(1, 8, 8), **conv_kargs_with_pad)
        self.conv8 = Conv3dBlock(df * 16, df * 8, 3, **conv_kargs_with_pad)
        self.conv9 = Conv3dBlock(df * 8 * 2, df * 4, 4, stride=(1, 2, 2), padding=(1, 1, 1), transpose=True, **conv_kargs)
        self.conv10 = Conv3dBlock(df * 4 * 2, df * 2, 3, **conv_kargs_with_pad)
        self.conv11 = Conv3dBlock(df * 2 * 2, df, 4, stride=(1, 2, 2), padding=(1, 1, 1), transpose=True, **conv_kargs)
        self.conv12 = Conv3dBlock(df, 3, 3, norm=None, activation=None, padding=-1)

    def _cut_to_align(self, x, y):
        """
            Cut the boarder by 1 frame(temporal)/row(height)/col(width)
            such that dimensions of x and y could be the same.
        """
        output_x = x
        for i in range(2, 5):
            dim_x = x.shape[i]
            dim_y = y.shape[i]
            if dim_x != dim_y:
                assert dim_x == dim_y + 1, ('Only deal with the odd width inverse case of deconv. ', f'Got dim_x: {dim_x}, dim_y: {dim_y}')
                output_x = output_x.narrow(i, 0, dim_y)
        return output_x

    def forward(self, x):
        intermediate = {}
        for i in range(1, 9):
            x = getattr(self, f'conv{i}')(x)
            intermediate[i] = x
        x = torch.cat([x, intermediate[4]], dim=1)
        x = self.conv9(x)
        x = self._cut_to_align(x, intermediate[3])
        x = torch.cat([x, intermediate[3]], dim=1)
        x = self.conv10(x)
        x = torch.cat([x, intermediate[2]], dim=1)
        x = self.conv11(x)
        x = self._cut_to_align(x, intermediate[1])
        imcomplete_video = self.conv12(x)
        return imcomplete_video


class CombCN(nn.Module):

    def __init__(self, df=32):
        super().__init__()
        conv_kargs_with_pad = {'norm': 'BN', 'activation': nn.ReLU(inplace=True), 'padding': -1}
        self.conv1 = Conv2dBlock(4, df * 2, 5, **conv_kargs_with_pad)
        self.conv2 = Conv2dBlock(df * 2, df * 4, 3, stride=2, **conv_kargs_with_pad)
        self.conv3 = Conv2dBlock(df * 4 + 3, df * 4, 3, **conv_kargs_with_pad)
        self.conv4 = Conv2dBlock(df * 4, df * 8, 3, stride=2, **conv_kargs_with_pad)
        self.conv5 = Conv2dBlock(df * 8, df * 8, 3, **conv_kargs_with_pad)
        self.conv6 = Conv2dBlock(df * 8, df * 8, 3, **conv_kargs_with_pad)
        self.conv7 = Conv2dBlock(df * 8, df * 8, 3, dilation=2, **conv_kargs_with_pad)
        self.conv8 = Conv2dBlock(df * 8, df * 8, 3, dilation=4, **conv_kargs_with_pad)
        self.conv9 = Conv2dBlock(df * 8, df * 8, 3, dilation=8, **conv_kargs_with_pad)
        self.conv10 = Conv2dBlock(df * 8 * 2, df * 8, 3, dilation=16, **conv_kargs_with_pad)
        self.conv11 = Conv2dBlock(df * 8 * 2, df * 8, 3, **conv_kargs_with_pad)
        self.conv12 = Conv2dBlock(df * 8 * 2, df * 8, 3, **conv_kargs_with_pad)
        self.conv13 = Conv2dBlock(df * 8 * 2, df * 4, 4, stride=2, transpose=True, **conv_kargs_with_pad)
        self.conv14 = Conv2dBlock(df * 4 * 2, df * 4, 3, **conv_kargs_with_pad)
        self.conv15 = Conv2dBlock(df * 4 * 2 + 3, df * 2, 4, stride=2, transpose=True, **conv_kargs_with_pad)
        self.conv16 = Conv2dBlock(df * 2 * 2, df, 3, **conv_kargs_with_pad)
        self.conv17 = Conv2dBlock(df, 3, 3, norm=None, activation=None, padding=-1)

    def forward(self, x, imcomplete_frame):
        intermediate = {}
        for i in range(1, 10):
            if i == 3:
                x = torch.cat([x, imcomplete_frame], dim=1)
            x = getattr(self, f'conv{i}')(x)
            intermediate[i] = x
        for i in range(10, 17):
            if i == 15:
                x = torch.cat([x, imcomplete_frame], dim=1)
            j = 17 - i
            x = torch.cat([x, intermediate[j]], dim=1)
            x = getattr(self, f'conv{i}')(x)
        completed_frame = self.conv17(x)
        return completed_frame


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels, kernel_size=self._kernel_shape, stride=self._stride, padding=0, bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """
    VALID_ENDPOINTS = 'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions'

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        return logits

    def extract_features(self, x, target_endpoint='Logits'):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                if end_point == target_endpoint:
                    break
        if target_endpoint == 'Logits':
            return x.mean(4).mean(3).mean(2)
        else:
            return x


class ReconLoss(nn.Module):

    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


device = torch.device('cuda')


class VGGLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = vgg(output)
        target_feature = vgg(target)
        loss = self.l1_loss(output_feature.relu2_2, target_feature.relu2_2) + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3) + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(self.vgg_loss(outputs[:, (frame_idx)], targets[:, (frame_idx)]))
        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):

    def __init__(self, original_channel_norm=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def style_loss(self, output, target):
        output_features = vgg(output)
        target_features = vgg(target)
        layers = ['relu2_2', 'relu3_3', 'relu4_3']
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(self.style_loss(outputs[:, (frame_idx)], targets[:, (frame_idx)]))
        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def edge_loss(self, output, target):
        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss, output_edge, gt_edge

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']
        mean_image_loss = []
        output_edges = []
        target_edges = []
        for batch_idx in range(targets.size(0)):
            edges_o = []
            edges_t = []
            for frame_idx in range(targets.size(1)):
                loss, output_edge, target_edge = self.edge_loss(outputs[(batch_idx), frame_idx:frame_idx + 1], targets[(batch_idx), frame_idx:frame_idx + 1])
                mean_image_loss.append(loss)
                edges_o.append(output_edge)
                edges_t.append(target_edge)
            output_edges.append(torch.cat(edges_o, dim=0))
            target_edges.append(torch.cat(edges_t, dim=0))
        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        self.current_output_edges = output_edges
        self.current_target_edges = target_edges
        return mean_image_loss


class L1LossMaskedMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, mask):
        masked = 1 - mask
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):

    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        masked = 1 - mask
        l2_sum = self.l2(x * masked, y * masked)
        return l2_sum / torch.sum(masked)


class ImcompleteVideoReconLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        imcomplete_video = model_output['imcomplete_video']
        targets = data_input['targets']
        down_sampled_targets = nn.functional.interpolate(targets.transpose(1, 2), scale_factor=[1, 0.5, 0.5])
        masks = data_input['masks']
        down_sampled_masks = nn.functional.interpolate(masks.transpose(1, 2), scale_factor=[1, 0.5, 0.5])
        return self.loss_fn(imcomplete_video, down_sampled_targets, down_sampled_masks)


class CompleteFramesReconLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        masks = data_input['masks']
        return self.loss_fn(outputs, targets, masks)


class TemporalWarpingLoss(nn.Module):

    def __init__(self, flownet_checkpoint_path=None, alpha=50):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()
        self.alpha = alpha
        self.flownet_checkpoint_path = flownet_checkpoint_path
        self.flownet = None

    def get_flownet_checkpoint_path(self):
        return self.flownet_checkpoint_path

    def _setup(self):
        self.flownet = FlowNetWrapper(checkpoint_path=self.flownet_checkpoint_path)

    def _get_non_occlusion_mask(self, targets, warped_targets):
        non_occlusion_masks = torch.exp(-self.alpha * torch.sum(targets[:, 1:] - warped_targets, dim=2).pow(2)).unsqueeze(2)
        return non_occlusion_masks

    def _get_loss(self, outputs, warped_outputs, non_occlusion_masks, masks):
        return self.loss_fn(outputs[:, 1:] * non_occlusion_masks, warped_outputs * non_occlusion_masks, masks[:, 1:])

    def forward(self, data_input, model_output):
        if self.flownet is None:
            self._setup()
        targets = data_input['targets']
        outputs = model_output['outputs']
        flows = self.flownet.infer_video(targets)
        warped_targets = warp_optical_flow(targets[:, :-1], -flows).detach()
        warped_outputs = warp_optical_flow(outputs[:, :-1], -flows).detach()
        non_occlusion_masks = self._get_non_occlusion_mask(targets, warped_targets)
        model_output['warped_outputs'] = warped_outputs[0]
        model_output['warped_targets'] = warped_targets[0]
        model_output['non_occlusion_masks'] = non_occlusion_masks[0]
        flow_imgs = []
        for flow in flows[0]:
            flow_img = flow_to_image(flow.cpu().permute(1, 2, 0).detach().numpy()).transpose(2, 0, 1)
            flow_imgs.append(torch.Tensor(flow_img))
        model_output['flow_imgs'] = flow_imgs
        masks = data_input['masks']
        return self._get_loss(outputs, warped_outputs, non_occlusion_masks, masks)


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        B, L, C, H, W = outputs.shape
        x = outputs.view([B * L, C, H, W])
        masks = data_input['masks']
        masks = masks.view([B * L, -1])
        mask_areas = masks.sum(dim=1)
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum(1).sum(1).sum(1)
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum(1).sum(1).sum(1)
        return ((h_tv + w_tv) / mask_areas).mean()


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge | l1
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()
        elif type == 'l1':
            self.criterion = nn.L1Loss()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class GatedConv(VanillaConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by)
        if conv_by == '2dtsm':
            self.module = NN3Dby2D
        self.gatingConv = self.module.Conv3d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
        self.sigmoid = nn.Sigmoid()
        self.store_gated_values = False

    def gated(self, mask):
        out = self.sigmoid(mask)
        if self.store_gated_values:
            self.gated_values = out.detach().cpu()
        return out

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(VanillaDeconv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), scale_factor=2, conv_by='3d'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, scale_factor, conv_by)
        self.conv = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by=conv_by)


class BaseModule(nn.Module):

    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv
            self.DeconvBlock = PartialDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv


class DownSampleModule(BaseModule):

    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(nc_in, nf * 1, kernel_size=(3, 5, 5), stride=1, padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv2 = self.ConvBlock(nf * 1, nf * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv4 = self.ConvBlock(nf * 2, nf * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.dilated_conv1 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 2, 2))
        self.dilated_conv2 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 4, 4))
        self.dilated_conv3 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 8, 8))
        self.dilated_conv4 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 16, 16))
        self.conv7 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        a3 = self.dilated_conv3(a2)
        a4 = self.dilated_conv4(a3)
        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c4, c2


class UpSampleModule(BaseModule):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection=False):
        super().__init__(conv_type)
        self.deconv1 = self.DeconvBlock(nc_in * 2 if use_skip_connection else nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.deconv2 = self.DeconvBlock(nf * 4 if use_skip_connection else nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(nf * 1, nf // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(nf // 2, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)
        self.use_skip_connection = use_skip_connection

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c4, c2 = inp
        if self.use_skip_connection:
            d1 = self.deconv1(self.concat_feature(c8, c4))
            c9 = self.conv9(d1)
            d2 = self.deconv2(self.concat_feature(c9, c2))
        else:
            d1 = self.deconv1(c8)
            c9 = self.conv9(d1)
            d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11


class CoarseNet(nn.Module):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool=False):
        super().__init__()
        self.conv_type = conv_type
        self.downsample_module = DownSampleModule(nc_in, nf, use_bias, norm, conv_by, conv_type)
        self.upsample_module = UpSampleModule(nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)

    def preprocess(self, masked_imgs, masks, guidances):
        masked_imgs = masked_imgs.transpose(1, 2)
        masks = masks.transpose(1, 2)
        if self.conv_type == 'partial':
            if guidances is not None:
                raise NotImplementedError('Partial convolution does not support guidance')
            inp = masked_imgs, masks
        elif self.conv_type == 'gated' or self.conv_type == 'vanilla':
            guidances = torch.full_like(masks, 0.0) if guidances is None else guidances.transpose(1, 2)
            inp = torch.cat([masked_imgs, masks, guidances], dim=1)
        else:
            raise NotImplementedError(f'{self.conv_type} not implemented')
        return inp

    def postprocess(self, masked_imgs, masks, c11):
        if self.conv_type == 'partial':
            inpainted = c11[0].transpose(1, 2) * (1 - masks)
        else:
            inpainted = c11.transpose(1, 2) * (1 - masks)
        out = inpainted + masked_imgs
        return out

    def forward(self, masked_imgs, masks, guidances=None):
        inp = self.preprocess(masked_imgs, masks, guidances)
        encoded_features = self.downsample_module(inp)
        c11 = self.upsample_module(encoded_features)
        out = self.postprocess(masked_imgs, masks, c11)
        return out


class AttentionDownSampleModule(DownSampleModule):

    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(nc_in, nf, use_bias, norm, conv_by, conv_type)


class RefineNet(CoarseNet):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool=False):
        super().__init__(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.upsample_module = UpSampleModule(nf * 16, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.attention_downsample_module = AttentionDownSampleModule(nc_in, nf, use_bias, norm, conv_by, conv_type)

    def forward(self, coarse_output, masks, guidances=None):
        inp = self.preprocess(coarse_output, masks, guidances)
        encoded_features = self.downsample_module(inp)
        attention_features, offset_flow = self.attention_downsample_module(inp)
        deconv_inp = torch.cat([encoded_features, attention_features], dim=2)
        c11 = self.upsample_module(deconv_inp)
        out = self.postprocess(coarse_output, masks, c11)
        return out, offset_flow


class Generator(nn.Module):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_refine=False, use_skip_connection: bool=False):
        super().__init__()
        self.coarse_net = CoarseNet(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.use_refine = use_refine
        if self.use_refine:
            self.refine_net = RefineNet(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)

    def forward(self, masked_imgs, masks, guidances=None):
        coarse_outputs = self.coarse_net(masked_imgs, masks, guidances)
        if self.use_refine:
            refined_outputs, offset_flows = self.refine_net(coarse_outputs, masks, guidances)
            return {'outputs': refined_outputs, 'offset_flows': offset_flows, 'coarse_outputs': coarse_outputs}
        else:
            return {'outputs': coarse_outputs}


class TemporalPatchGANDiscriminator(nn.Module):

    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        kw = 3, 4, 4
        padw = 1, 1, 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1, 2, 2), padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1, 2, 2), padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdversarialLoss,
     lambda: ([], {}),
     lambda: ([], {'outputs': torch.rand([4, 4]), 'is_real': 4}),
     True),
    (Conv2dBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3dBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     False),
    (GatedConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (GatedDeconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (L1LossMaskedMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2LossMaskedMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unit3D,
     lambda: ([], {'in_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (VanillaConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (VanillaDeconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
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

class Test_amjltc295_Free_Form_Video_Inpainting(_paritybench_base):
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

    def test_015(self):
        self._check(*TESTCASES[15])

