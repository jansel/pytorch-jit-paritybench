import sys
_module = sys.modules[__name__]
del sys
ProREG = _module
base_cluster = _module
aflw_from_mat = _module
generate_300W = _module
init_path = _module
vis = _module
cluster = _module
crop_pic = _module
gen_mean_face = _module
CycleDataset = _module
GeneralDataset = _module
datasets = _module
dataset_utils = _module
point_meta = _module
debug = _module
check = _module
debug_main = _module
models = _module
basic_batch = _module
cycle_util = _module
discriminator_model = _module
gan_loss = _module
generator_model = _module
initialization = _module
itn = _module
itn_cpm = _module
model_utils = _module
resnet = _module
vgg16_base = _module
options = _module
options = _module
procedure = _module
evaluate_detector = _module
san_util = _module
train_cycle = _module
train_detector = _module
san_vision = _module
common_eval = _module
cpm_loss = _module
evaluation_util = _module
transforms = _module
utils = _module
box_utils = _module
config = _module
convert_utils = _module
file_utils = _module
flop_benchmark = _module
image_pool = _module
image_utils = _module
pts_utils = _module
stn_utils = _module
time_utils = _module
visualization = _module
draw_image_by_points = _module
save_error_image = _module
visualize = _module
san_api = _module
san_eval = _module
san_main = _module
demo_list = _module
extrct_300VW = _module
generate_300VW = _module
basic_main = _module
eval = _module
lk_main = _module
config_utils = _module
basic_args = _module
configure_utils = _module
lk_args = _module
GeneralDataset = _module
VideoDataset = _module
parse_utils = _module
lk = _module
basic_lk = _module
basic_lk_batch = _module
basic_utils = _module
basic_utils_batch = _module
log_utils = _module
logger = _module
meter = _module
LK = _module
basic = _module
basic_batch = _module
cpm_vgg16 = _module
initialization = _module
model_utils = _module
optimizer = _module
opt_utils = _module
basic_eval = _module
basic_train = _module
lk_loss = _module
lk_train = _module
losses = _module
saver = _module
starts = _module
generation = _module
flop_benchmark = _module
xvision = _module
evaluation_util = _module
transforms = _module
CREATE_DATA = _module
GEN_300VW = _module
GEN_300W = _module
GEN_MPII = _module
GEN_Panoptic_XX = _module
GEN_VoxCeleb2 = _module
GEN_WFLW = _module
GEN_aflw = _module
GEN_demo = _module
GEN_list = _module
analyze_aflw = _module
resize_mv = _module
download_MPII = _module
download_Panoptic = _module
basic_args_v2 = _module
robust_args = _module
sbr_args_v2 = _module
share_args = _module
stm_args_v2 = _module
EvalDataset = _module
GeneralDatasetV2 = _module
RobustDataset = _module
STMDataset = _module
VideoDatasetV2 = _module
WrapParallel = _module
augmentation_utils = _module
optflow_utils = _module
point_meta_v1 = _module
point_meta_v2 = _module
basic_lk = _module
basic_lk_batch = _module
basic_utils = _module
basic_utils_batch = _module
ProCPM = _module
ProCU = _module
ProHG = _module
ProHRNet = _module
ProMSPN = _module
ProREG = _module
ProTemporal_HEAT = _module
ProTemporal_REG = _module
STM_HEAT = _module
STM_REG = _module
basic_batch = _module
initialization = _module
model_utils = _module
modules = _module
multiview = _module
triangulate = _module
loss_module = _module
opt_utils = _module
temporal_loss_heatmap = _module
temporal_loss_regression = _module
basic_eval_robust = _module
basic_main_heatmap = _module
basic_main_regression = _module
debug_utils = _module
losses = _module
multiview_loss_heatmap = _module
multiview_loss_regression = _module
saver = _module
sbr_main_heatmap = _module
sbr_main_regression = _module
starts = _module
stm_main_heatmap = _module
stm_main_regression = _module
temporal_loss_heatmap = _module
temporal_loss_regression = _module
x_sbr_main_heatmap = _module
x_sbr_main_regression = _module
sbr = _module
interpolate = _module
video_utils = _module
affine_utils = _module
evaluation_util = _module
functional = _module
style_trans = _module
transforms1v = _module
transforms2v = _module
transforms3v = _module
transformsImage = _module
visualization = _module
visualization_v2 = _module
visualization_v3 = _module
basic_batch = _module
flop_benchmark = _module
initialization = _module
layer_utils = _module
model_utils = _module
student_cpm = _module
student_hg = _module
teacher = _module
test = _module

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


import torch


import torch.nn as nn


import time


import random


from copy import deepcopy


import numbers


import numpy as np


import torch.nn.functional as F


from sklearn.cluster import KMeans


import torchvision.datasets as visiondatasets


import torchvision.transforms as visiontransforms


import warnings


import math


import torch.utils.data as data


from torch.optim import lr_scheduler


from torch.nn import init


from torch.autograd import Variable


import functools


import itertools


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


from scipy.ndimage.interpolation import zoom


import copy


from scipy import interpolate


import types


import collections


from collections import defaultdict


from scipy.io import loadmat


import re


from torch import multiprocessing as mp


from torch.nn import functional as F


from copy import deepcopy as copy


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
    Args:
      input_tensor: shape(batch, channel, x_dim, y_dim)
    """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        xx_channel, yy_channel = xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, kernels, stride, pad):
    return nn.Sequential(nn.Conv2d(inp, oup, kernels, stride, pad, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def weights_init_reg(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class MobileNetV2REG(nn.Module):

    def __init__(self, input_dim, input_channel, width_mult, pts_num):
        super(MobileNetV2REG, self).__init__()
        self.pts_num = pts_num
        block = InvertedResidual
        interverted_residual_setting = [[1, 48, 1, 1], [2, 48, 5, 2], [2, 96, 1, 2], [4, 96, 6, 1], [2, 16, 1, 1]]
        input_channel = int(input_channel * width_mult)
        features = [conv_bn(input_dim, input_channel, (3, 3), 2, 1)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(nn.AdaptiveAvgPool2d((14, 14)))
        self.features = nn.Sequential(*features)
        self.S1 = nn.Sequential(CoordConv(input_channel, input_channel * 2, True, kernel_size=3, padding=1), conv_bn(input_channel * 2, input_channel * 2, (3, 3), 2, 1))
        self.S2 = nn.Sequential(CoordConv(input_channel * 2, input_channel * 4, True, kernel_size=3, padding=1), conv_bn(input_channel * 4, input_channel * 8, (7, 7), 1, 0))
        output_neurons = 14 * 14 * input_channel + 7 * 7 * input_channel * 2 + input_channel * 8
        self.locator = nn.Sequential(nn.Linear(output_neurons, pts_num * 2))
        self.apply(weights_init_reg)

    def forward(self, xinputs):
        if xinputs.dim() == 5:
            batch, seq, C, H, W = xinputs.shape
            batch_locs = self.forward_x(xinputs.view(batch * seq, C, H, W))
            _, N, _ = batch_locs.shape
            return batch_locs.view(batch, seq, N, 2)
        else:
            return self.forward_x(xinputs)

    def forward_x(self, x):
        batch, C, H, W = x.size()
        features = self.features(x)
        S1 = self.S1(features)
        S2 = self.S2(S1)
        tensors = torch.cat((features.view(batch, -1), S1.view(batch, -1), S2.view(batch, -1)), dim=1)
        batch_locs = self.locator(tensors).view(batch, self.pts_num, 2)
        return batch_locs


class NLayerDiscriminator(nn.Module):

    def __init__(self, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(3, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True), nn.InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True), nn.InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.InstanceNorm2d(dim, affine=False), nn.ReLU(True)]
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
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.InstanceNorm2d(dim, affine=False)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):

    def __init__(self, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True), nn.InstanceNorm2d(64, affine=False), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3, stride=2, padding=1, bias=True), nn.InstanceNorm2d(64 * mult * 2, affine=False), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(64 * mult, padding_type=padding_type, use_dropout=False, use_bias=True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True), nn.InstanceNorm2d(int(64 * mult / 2), affine=False), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def define_G(gpu_ids=[]):
    netG = ResnetGenerator(gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netG
    netG.apply(weights_init_xavier)
    return netG


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=1e-06):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1.0 + 2.0 * x.data / (L - 1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2
    theta = MU.np2variable(affine_parameter, heatmap.is_cuda, False)
    grid_size = torch.Size([num_pts, 1, radius * 2 + 1, radius * 2 + 1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)
    X = MU.np2variable(torch.arange(-radius, radius + 1), heatmap.is_cuda, False).view(1, 1, radius * 2 + 1)
    Y = MU.np2variable(torch.arange(-radius, radius + 1), heatmap.is_cuda, False).view(1, radius * 2 + 1, 1)
    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h
    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1), score


def get_parameters(model, bias):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.BatchNorm2d):
            if bias:
                yield m.bias
            else:
                yield m.weight


class ITN_CPM(nn.Module):

    def __init__(self, model_config):
        super(ITN_CPM, self).__init__()
        self.config = model_config.copy()
        self.downsample = 1
        self.netG_A = define_G()
        self.netG_B = define_G()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
        self.downsample = 8
        pts_num = self.config.pts_num
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        self.stage2 = nn.Sequential(nn.Conv2d(128 * 2 + pts_num * 2, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        self.stage3 = nn.Sequential(nn.Conv2d(128 * 2 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        assert self.config.num_stages >= 1, 'stages of cpm must >= 1'

    def set_mode(self, mode):
        if mode.lower() == 'train':
            self.train()
            for m in self.netG_A.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                    m.eval()
            for m in self.netG_B.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                    m.eval()
        elif mode.lower() == 'eval':
            self.eval()
        else:
            raise NameError('The wrong mode : {}'.format(mode))

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': self.netG_A.parameters(), 'lr': 0, 'weight_decay': 0}, {'params': self.netG_B.parameters(), 'lr': 0, 'weight_decay': 0}, {'params': get_parameters(self.features, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self.stage1, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.stage1, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self.stage2, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.stage2, bias=True), 'lr': base_lr * 8, 'weight_decay': 0}, {'params': get_parameters(self.stage3, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.stage3, bias=True), 'lr': base_lr * 8, 'weight_decay': 0}]
        return params_dict

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size = inputs.size(0)
        num_stages, num_pts = self.config.num_stages, self.config.pts_num - 1
        batch_cpms, batch_locs, batch_scos = [], [], []
        features, stage1s = [], []
        inputs = [inputs, (self.netG_A(inputs) + self.netG_B(inputs)) / 2]
        for input in inputs:
            feature = self.features(input)
            feature = self.CPM_feature(feature)
            features.append(feature)
            stage1s.append(self.stage1(feature))
        xfeature = torch.cat(features, 1)
        cpm_stage2 = self.stage2(torch.cat([xfeature, stage1s[0], stage1s[1]], 1))
        cpm_stage3 = self.stage3(torch.cat([xfeature, cpm_stage2], 1))
        batch_cpms = [stage1s[0], stage1s[1]] + [cpm_stage2, cpm_stage3]
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(cpm_stage3[ibatch], self.config.argmax, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        return batch_cpms, batch_locs, batch_scos, inputs[1:]


BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        cls = self.cls(feature)
        return feature, cls


class VGG16_base(nn.Module):

    def __init__(self, model_config, pts_num):
        super(VGG16_base, self).__init__()
        self.config = deepcopy(model_config)
        self.downsample = 8
        self.pts_num = pts_num
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        assert self.config['stages'] >= 1, 'stages of cpm must >= 1'
        stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config['stages']):
            stagex = nn.Sequential(nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': get_parameters(self.features, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}]
        for stage in self.stages:
            params_dict.append({'params': get_parameters(stage, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True), 'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_cpms = []
        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config['stages']):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]], 1))
            batch_cpms.append(cpm)
        return batch_cpms


class SobelConv(nn.Module):

    def __init__(self, tag, dtype):
        super(SobelConv, self).__init__()
        if tag == 'x':
            Sobel = np.array([[-1.0 / 8, 0, 1.0 / 8], [-2.0 / 8, 0, 2.0 / 8], [-1.0 / 8, 0, 1.0 / 8]])
        elif tag == 'y':
            Sobel = np.array([[-1.0 / 8, -2.0 / 8, -1.0 / 8], [0, 0, 0], [1.0 / 8, 2.0 / 8, 1.0 / 8]])
        else:
            raise NameError('Do not know this tag for Sobel Kernel : {}'.format(tag))
        Sobel = torch.from_numpy(Sobel).type(dtype)
        Sobel = Sobel.view(1, 1, 3, 3)
        self.register_buffer('weight', Sobel)
        self.tag = tag

    def forward(self, input):
        weight = self.weight.expand(input.size(1), 1, 3, 3).contiguous()
        return F.conv2d(input, weight, groups=input.size(1), padding=1)

    def __repr__(self):
        return '{name}(tag={tag})'.format(name=self.__class__.__name__, **self.__dict__)


class LK(nn.Module):

    def __init__(self, model, lkconfig, points):
        super(LK, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(lkconfig)
        self.points = points

    def forward(self, inputs):
        assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {}'.format(inputs.size())
        batch_size, sequence, C, H, W = list(inputs.size())
        gathered_inputs = inputs.view(batch_size * sequence, C, H, W)
        heatmaps, batch_locs, batch_scos = self.detector(gathered_inputs)
        heatmaps = [x.view(batch_size, sequence, self.points, H // self.downsample, W // self.downsample) for x in heatmaps]
        batch_locs, batch_scos = batch_locs.view(batch_size, sequence, self.points, 2), batch_scos.view(batch_size, sequence, self.points)
        batch_next, batch_fback, batch_back = [], [], []
        for ibatch in range(batch_size):
            feature_old = inputs[ibatch]
            nextPts, fbackPts, backPts = lk.lk_forward_backward_batch(inputs[ibatch], batch_locs[ibatch], self.config.window, self.config.steps)
            batch_next.append(nextPts)
            batch_fback.append(fbackPts)
            batch_back.append(backPts)
        batch_next, batch_fback, batch_back = torch.stack(batch_next), torch.stack(batch_fback), torch.stack(batch_back)
        return heatmaps, batch_locs, batch_scos, batch_next, batch_fback, batch_back


class Residual(nn.Module):

    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        middle = self.numOut // 2
        self.conv_A = nn.Sequential(nn.BatchNorm2d(numIn), nn.ReLU(inplace=True), nn.Conv2d(numIn, middle, kernel_size=1, dilation=1, padding=0, bias=True))
        self.conv_B = nn.Sequential(nn.BatchNorm2d(middle), nn.ReLU(inplace=True), nn.Conv2d(middle, middle, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_C = nn.Sequential(nn.BatchNorm2d(middle), nn.ReLU(inplace=True), nn.Conv2d(middle, numOut, kernel_size=1, dilation=1, padding=0, bias=True))
        if self.numIn != self.numOut:
            self.branch = nn.Sequential(nn.BatchNorm2d(self.numIn), nn.ReLU(inplace=True), nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    def forward(self, x):
        residual = x
        main = self.conv_A(x)
        main = self.conv_B(main)
        main = self.conv_C(main)
        if hasattr(self, 'branch'):
            residual = self.branch(residual)
        return main + residual


class VGG16V2(nn.Module):

    def __init__(self, config, pts_num, sigma, idim):
        super(VGG16V2, self).__init__()
        self.config = deepcopy(config)
        self.sigma = sigma
        self.downsample = 2 ** sum(self.config.pooling)
        self.pts_num = pts_num
        features = [nn.Conv2d(idim, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)]
        if self.config.pooling[0]:
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features += [nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)]
        if self.config.pooling[1]:
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features += [nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)]
        if self.config.pooling[2]:
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features += [nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*features)
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config.stages)
        stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config.stages):
            stagex = nn.Sequential(nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)
        if self.config.sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': get_parameters(self.features, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay}, {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0}]
        for stage in self.stages:
            params_dict.append({'params': get_parameters(stage, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True), 'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    def extra_repr(self):
        return '{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms, batch_locs, batch_scos = [], [], []
        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config.stages):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]], 1))
            if self.sigmoid is not None:
                cpm = self.sigmoid(cpm)
            batch_cpms.append(cpm)
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.sigma, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        return xfeature, batch_cpms, batch_locs, batch_scos


class BN_ReLU_C1x1(nn.Module):

    def __init__(self, in_num, out_num):
        super(BN_ReLU_C1x1, self).__init__()
        layers = [nn.BatchNorm2d(in_num), nn.ReLU(inplace=True), nn.Conv2d(in_num, out_num, kernel_size=1, stride=1, bias=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, ifeatures):
        if isinstance(ifeatures, list):
            ifeatures = torch.cat(ifeatures, dim=1)
        features = self.layers(ifeatures)
        return features


class DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        layers = [nn.BatchNorm2d(num_input_features), nn.ReLU(inplace=True), nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace=True), nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)]
        if drop_rate > 0:
            layers += [nn.Dropout(drop_rate)]
        self.layers = nn.Sequential(*layers)

    def forward(self, ifeatures):
        if isinstance(ifeatures, list):
            ifeatures = torch.cat(ifeatures, dim=1)
        features = self.layers(ifeatures)
        return features


class CU_Block(nn.Module):

    def __init__(self, channels, adapC, neck_size, growth_rate, num_blocks):
        super(CU_Block, self).__init__()
        self.num_blocks = num_blocks
        down_blocks, up_blocks, adapters_skip = [], [], []
        adapters_down, adapters_up = [], []
        self.cache_features = []
        for i in range(self.num_blocks):
            down_blocks.append(DenseLayer(channels, growth_rate, neck_size, 0))
            up_blocks.append(DenseLayer(channels * 2, growth_rate, neck_size, 0))
            adapters_skip.append(BN_ReLU_C1x1(channels + growth_rate, adapC))
            adapters_down.append(BN_ReLU_C1x1(channels + growth_rate, adapC))
            adapters_up.append(BN_ReLU_C1x1(channels * 2 + growth_rate, adapC))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.adapters_skip = nn.ModuleList(adapters_skip)
        self.adapters_down = nn.ModuleList(adapters_down)
        self.neck_block = DenseLayer(channels, growth_rate, neck_size, 0)
        self.adapter_neck = BN_ReLU_C1x1(channels + growth_rate, channels)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.adapters_up = nn.ModuleList(adapters_up)

    def forward(self, x, previous_features):
        skip_list = [None] * self.num_blocks
        down_outputs, up_outputs = [None] * self.num_blocks, [None] * self.num_blocks
        cache_features, cache_shapes = [], []
        for i in range(self.num_blocks):
            cache_shapes.append((x.size(2), x.size(3)))
            tinputs = [x] + previous_features[i]
            temp = self.down_blocks[i](tinputs)
            cache_features.append(temp)
            down_outputs[i] = self.adapters_down[i](tinputs + [temp])
            skip_list[i] = self.adapters_skip[i](tinputs + [temp])
            x = F.interpolate(down_outputs[i], [x.size(2) // 2, x.size(3) // 2], mode='bilinear', align_corners=True)
        temp = self.neck_block([x] + previous_features[self.num_blocks])
        cache_features.append(temp)
        x = self.adapter_neck([x] + previous_features[self.num_blocks] + [temp])
        for i in range(self.num_blocks - 1, -1, -1):
            x = F.interpolate(x, cache_shapes[i], mode='bilinear', align_corners=True)
            tinputs = [x] + previous_features[2 * self.num_blocks - i] + [skip_list[i]]
            temp = self.up_blocks[i](tinputs)
            cache_features.append(temp)
            up_outputs[i] = self.adapters_up[i](tinputs + [temp])
        outputs = up_outputs[0]
        return outputs, cache_features


class CUNet(nn.Module):

    def __init__(self, config, points, sigma, input_dim):
        super(CUNet, self).__init__()
        self.downsample = 4
        self.sigma = sigma
        self.config = copy.deepcopy(config)
        self.pts_num = points
        self.init_channel = self.config.init_channel
        self.neck_size = self.config.neck_size
        self.growth_rate = self.config.growth_rate
        self.layer_num = self.config.layer_num
        self.num_blocks = self.config.num_blocks
        self.conv = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(input_dim, self.init_channel, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(self.init_channel)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=2, stride=2))]))
        num_chans = self.init_channel
        blocks = []
        for i in range(self.layer_num):
            blocks.append(CU_Block(num_chans + i * self.growth_rate, num_chans, self.neck_size, self.growth_rate, self.num_blocks))
        self.blocks = nn.ModuleList(blocks)
        self.intermedia = nn.ModuleList()
        for i in range(1, self.layer_num):
            self.intermedia.append(BN_ReLU_C1x1(num_chans * (i + 1), num_chans))
        self.linears = nn.ModuleList()
        for i in range(0, self.layer_num):
            self.linears.append(BN_ReLU_C1x1(num_chans, self.pts_num))

    def extra_repr(self):
        return '{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        x = self.conv(inputs)
        intermediates = [[] for i in range(2 * self.num_blocks + 1)]
        features, heatmaps = [], []
        for i in range(self.layer_num):
            if i == 0:
                xinputs = x
            else:
                xinputs = self.intermedia[i - 1](features + [x])
            feature, cache_intermediate = self.blocks[i](x, intermediates)
            heatmap = self.linears[i](feature)
            features.append(feature)
            for idx, tempF in enumerate(cache_intermediate):
                intermediates[idx].append(tempF)
            heatmaps.append(heatmap)
        final_heatmap = heatmaps[-1]
        B, NP, H, W = final_heatmap.size()
        batch_locs, batch_scos = find_tensor_peak_batch(final_heatmap.view(B * NP, H, W), self.sigma, self.downsample)
        batch_locs, batch_scos = batch_locs.view(B, NP, 2), batch_scos.view(B, NP)
        return features, heatmaps, batch_locs, batch_scos


class HierarchicalPMS(nn.Module):

    def __init__(self, numIn, numOut):
        super(HierarchicalPMS, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        cA, cB, cC = self.numOut // 2, self.numOut // 4, self.numOut - self.numOut // 2 - self.numOut // 4
        assert cA + cB + cC == numOut, '({:}, {:}, {:}) = {:}'.format(cA, cB, cC, numOut)
        self.conv_A = nn.Sequential(nn.BatchNorm2d(numIn), nn.ReLU(inplace=True), nn.Conv2d(numIn, cA, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_B = nn.Sequential(nn.BatchNorm2d(cA), nn.ReLU(inplace=True), nn.Conv2d(cA, cB, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_C = nn.Sequential(nn.BatchNorm2d(cB), nn.ReLU(inplace=True), nn.Conv2d(cB, cC, kernel_size=3, dilation=1, padding=1, bias=True))
        if self.numIn != self.numOut:
            self.branch = nn.Sequential(nn.BatchNorm2d(self.numIn), nn.ReLU(inplace=True), nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    def forward(self, x):
        residual = x
        A = self.conv_A(x)
        B = self.conv_B(A)
        C = self.conv_C(B)
        main = torch.cat((A, B, C), dim=1)
        if hasattr(self, 'branch'):
            residual = self.branch(residual)
        return main + residual


class Hourglass(nn.Module):

    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats
        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))
        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)
        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))
        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)
        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)
        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)
        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)
        return up1 + up2


class HourGlassNet(nn.Module):

    def __init__(self, config, points, sigma, input_dim):
        super(HourGlassNet, self).__init__()
        self.downsample = 4
        self.sigma = sigma
        self.config = copy.deepcopy(config)
        if self.config.module == 'Residual':
            module = Residual
        elif self.config.module == 'HierarchicalPMS':
            module = HierarchicalPMS
        else:
            raise ValueError('Invaliad module for HG : {:}'.format(self.config.module))
        self.pts_num = points
        self.nStack = self.config.stages
        self.nModules = self.config.nModules
        self.nFeats = self.config.nFeats
        self.recursive = self.config.recursive
        self.conv = nn.Sequential(nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.ress = nn.Sequential(module(64, 128), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), module(128, 128), module(128, self.nFeats))
        _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []
        for i in range(self.nStack):
            feature = Hourglass(self.recursive, self.nModules, self.nFeats, module)
            feature = [feature] + [module(self.nFeats, self.nFeats) for _ in range(self.nModules)]
            feature += [nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, bias=True), nn.BatchNorm2d(self.nFeats), nn.ReLU(inplace=True)]
            feature = nn.Sequential(*feature)
            _features.append(feature)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.pts_num, kernel_size=1, stride=1, bias=True))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, bias=True))
                _tmpOut_.append(nn.Conv2d(self.pts_num, self.nFeats, kernel_size=1, stride=1, bias=True))
        self.features = nn.ModuleList(_features)
        self.tmpOuts = nn.ModuleList(_tmpOut)
        self.trsfeas = nn.ModuleList(_ll_)
        self.trstmps = nn.ModuleList(_tmpOut_)
        if self.config.sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def extra_repr(self):
        return '{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        x = self.conv(inputs)
        x = self.ress(x)
        features, heatmaps, batch_locs, batch_scos = [], [], [], []
        for i in range(self.nStack):
            feature = self.features[i](x)
            features.append(feature)
            tmpOut = self.tmpOuts[i](feature)
            if self.sigmoid is not None:
                tmpOut = self.sigmoid(tmpOut)
            heatmaps.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.trsfeas[i](feature)
                tmpOut_ = self.trstmps[i](tmpOut)
                x = x + ll_ + tmpOut_
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(heatmaps[-1][ibatch], self.sigma, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        return features, heatmaps, batch_locs, batch_scos


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], 1, None))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels, 1))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[x[i].shape[2], x[i].shape[3]], mode='bilinear', align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolutionNet(nn.Module):

    def __init__(self, config, points, sigma, input_dim):
        super(HighResolutionNet, self).__init__()
        self.inplanes = 64
        self.config = copy.deepcopy(config)
        self.sigma = sigma
        self.downsample = 4
        self.stem = nn.Sequential(nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        num_channels = self.config.S2_NUM_CHANNELS
        assert self.config.S2_BLOCK in ['BASIC', 'BOTTLENECK']
        block = BasicBlock if self.config.S2_BLOCK == 'BASIC' else Bottleneck
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage('S2', num_channels)
        num_channels = self.config.S3_NUM_CHANNELS
        block = BasicBlock if self.config.S3_BLOCK == 'BASIC' else Bottleneck
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage('S3', num_channels)
        num_channels = self.config.S4_NUM_CHANNELS
        block = BasicBlock if self.config.S4_BLOCK == 'BASIC' else Bottleneck
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage('S4', num_channels, multi_scale_output=True)
        final_inp_channels = sum(pre_stage_channels)
        self.head = nn.Sequential(nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=self.config.FINAL_CONV_KERNEL, stride=1, padding=1 if self.config.FINAL_CONV_KERNEL == 3 else 0), nn.BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(final_inp_channels, points, kernel_size=self.config.FINAL_CONV_KERNEL, stride=1, padding=1 if self.config.FINAL_CONV_KERNEL == 3 else 0))
        if self.config.sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        else:
            downsample = None
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, prefix, num_inchannels, multi_scale_output=True):
        num_modules = getattr(self.config, '{:}_NUM_MODULES'.format(prefix))
        num_branches = getattr(self.config, '{:}_NUM_BRANCHES'.format(prefix))
        num_blocks = getattr(self.config, '{:}_NUM_BLOCKS'.format(prefix))
        num_channels = getattr(self.config, '{:}_NUM_CHANNELS'.format(prefix))
        block = BasicBlock if getattr(self.config, '{:}_BLOCK'.format(prefix)) == 'BASIC' else Bottleneck
        fuse_method = getattr(self.config, '{:}_FUSE_METHOD'.format(prefix))
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def extra_repr(self):
        return '{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.config.S2_NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.config.S3_NUM_BRANCHES):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.config.S4_NUM_BRANCHES):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        feats = self.stage4(x_list)
        batch_size, _, height, width = feats[0].shape
        feat1 = F.interpolate(feats[1], size=(height, width), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(feats[2], size=(height, width), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(feats[3], size=(height, width), mode='bilinear', align_corners=False)
        cat_f = torch.cat([feats[0], feat1, feat2, feat3], 1)
        heatmaps = self.head(cat_f)
        batch_locs, batch_scos = [], []
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(heatmaps[ibatch], self.sigma, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        return feats, [heatmaps], batch_locs, batch_scos

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MSPNet(nn.Module):

    def __init__(self, config, points, sigma, input_dim):
        super(MSPNet, self).__init__()
        self.downsample = 4
        self.sigma = sigma
        self.config = copy.deepcopy(config)
        self.pts_num = points
        self.nStack = self.config.stages
        self.nModules = self.config.nModules
        self.nFeats = self.config.nFeats
        self.recursive = self.config.recursive
        self.basic = nn.Sequential(nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3), Residual(64, 128), BN_ReLU_C1x1(128, 128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.res_block = nn.Sequential(nn.Conv2d(128, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 1, bias=False))
        self.conv = Residual(128, self.nFeats)
        self.HGs = nn.ModuleList()
        _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []
        for i in range(self.nStack):
            self.HGs.append(Hourglass(self.recursive, self.nModules, self.nFeats))
            feature = [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]
            feature += [nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, bias=True), nn.BatchNorm2d(self.nFeats), nn.ReLU(inplace=True)]
            feature = nn.Sequential(*feature)
            _features.append(feature)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.pts_num, kernel_size=1, stride=1, bias=True))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size=1, stride=1, bias=True))
                _tmpOut_.append(nn.Conv2d(self.pts_num, self.nFeats, kernel_size=1, stride=1, bias=True))
        self.features = nn.ModuleList(_features)
        self.tmpOuts = nn.ModuleList(_tmpOut)
        self.trsfeas = nn.ModuleList(_ll_)
        self.trstmps = nn.ModuleList(_tmpOut_)
        self.cross_stages = nn.ModuleList()
        for i in range(self.nStack - 1):
            cross_layer = nn.ModuleList()
            for j in range(self.recursive):
                iC, oC = self.nFeats * 2 ** j, min(self.nFeats * 2 ** (j + 1), self.nFeats * 2 ** (self.recursive - 1))
                cross_layer.append(nn.ModuleList([BN_ReLU_C1x1(oC, oC), BN_ReLU_C1x1(self.nFeats, oC)]))
            self.cross_stages.append(cross_layer)

    def extra_repr(self):
        return '{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        pool = self.basic(inputs)
        ress = self.res_block(pool) + pool
        x = self.conv(ress)
        features, heatmaps, batch_locs, batch_scos = [], [], [], []
        previous_features = None
        for i in range(self.nStack):
            hg_fea, temp_downs, temp_ups = self.HGs[i](x, previous_features)
            if i + 1 < self.nStack:
                previous_features = [None] * self.recursive
                for istep in range(self.recursive):
                    previous_features[istep] = self.cross_stages[i][istep][0](temp_downs[istep]) + self.cross_stages[i][istep][1](temp_ups[istep])
            feature = self.features[i](hg_fea)
            features.append(feature)
            tmpOut = self.tmpOuts[i](feature)
            heatmaps.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.trsfeas[i](feature)
                tmpOut_ = self.trstmps[i](tmpOut)
                x = x + ll_ + tmpOut_
        final_heatmap = heatmaps[-1]
        B, NP, H, W = final_heatmap.size()
        batch_locs, batch_scos = find_tensor_peak_batch(final_heatmap.view(B * NP, H, W), self.sigma, self.downsample)
        batch_locs, batch_scos = batch_locs.view(B, NP, 2), batch_scos.view(B, NP)
        return features, heatmaps, batch_locs, batch_scos


def batch_interpolate_flow(flows, locs, return_absolute):
    assert flows.dim() == 4 and flows.size(-1) == 2, 'invalid size of flows : {:}'.format(flows.size())
    assert locs.dim() == 3 and locs.size(-1) == 2, 'invalid size of locs : {:}'.format(locs.size())
    batch, H, W, _ = flows.size()
    x0, y0 = torch.floor(locs[:, :, 0]).long(), torch.floor(locs[:, :, 1]).long()
    x1, y1 = x0 + 1, y0 + 1
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    """
  def get_matrix_by_index_trivial(tensor, _x, _y):
    assert tensor.dim() == 4 and tensor.size(0) == _x.size(0) == _y.size(0)
    (_B, _H, _W, _), _P = tensor.size(), _x.size(1)
    # get the index matrix with shape [batch, points, 2]
    indexes = []
    for i in range(_B):
      Bidx = []
      for j in range(_P):
        a = i*_H*_W*2 + _y[i,j].item()*_W*2 + _x[i,j].item()*2
        Bidx.append([a, a+1])
      indexes.append( Bidx )
    indexes = torch.tensor(indexes, device=tensor.device)
    offsets = torch.take(tensor, indexes)
    return offsets
  """

    def get_matrix_by_index(tensor, _x, _y):
        assert tensor.dim() == 4 and tensor.size(0) == _x.size(0) == _y.size(0)
        (_B, _H, _W, _), _P = tensor.size(), _x.size(1)
        indexes = torch.arange(0, _B, device=_x.device).unsqueeze(-1) * _H * _W * 2 + _y * _W * 2 + _x * 2
        indexes = torch.stack([indexes, indexes + 1], dim=-1)
        return torch.take(tensor, indexes)
    Ia = get_matrix_by_index(flows, x0, y0)
    Ib = get_matrix_by_index(flows, x0, y1)
    Ic = get_matrix_by_index(flows, x1, y0)
    Id = get_matrix_by_index(flows, x1, y1)
    x0, x1, y0, y1 = x0.type(locs.dtype), x1.type(locs.dtype), y0.type(locs.dtype), y1.type(locs.dtype)
    wa = (x1 - locs[:, :, 0]) * (y1 - locs[:, :, 1])
    wb = (x1 - locs[:, :, 0]) * (locs[:, :, 1] - y0)
    wc = (locs[:, :, 0] - x0) * (y1 - locs[:, :, 1])
    wd = (locs[:, :, 0] - x0) * (locs[:, :, 1] - y0)
    outs = Ia * wa.unsqueeze(-1) + Ib * wb.unsqueeze(-1) + Ic * wc.unsqueeze(-1) + Id * wd.unsqueeze(-1)
    if return_absolute:
        return outs, outs + locs
    else:
        return outs


def normalize_L(x, L):
    return -1.0 + 2.0 * x / (L - 1)


def normalize_points_batch(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(shape) == 2, 'invalid shape : {:}'.format(shape)
    assert isinstance(points, torch.Tensor) and points.size(-1) == 2, 'points are wrong : {:}'.format(points.shape)
    (H, W), points = shape, points.clone()
    x = normalize_L(points[..., 0], W)
    y = normalize_L(points[..., 1], H)
    return torch.stack((x, y), dim=-1)


class TemporalHEAT(nn.Module):

    def __init__(self, model, config, points):
        super(TemporalHEAT, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(config)
        self.points = points
        self.center_idx = self.config.video_L

    def forward(self, inputs, Fflows, Bflows, is_images):
        assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {}'.format(inputs.size())
        batch_size, sequence, C, H, W = list(inputs.size())
        gathered_inputs = inputs.view(batch_size * sequence, C, H, W)
        features, heatmaps, batch_locs, batch_scos = self.detector(gathered_inputs)
        heatmaps = [x.view(batch_size, sequence, self.points, H // self.downsample, W // self.downsample) for x in heatmaps]
        batch_locs, batch_scos = batch_locs.view(batch_size, sequence, self.points, 2), batch_scos.view(batch_size, sequence, self.points)
        past2now, future2now, fb_check = self.SBR_forward_backward(batch_locs, Fflows, Bflows)
        batch_locs = normalize_points_batch((H, W), batch_locs)
        past2now = normalize_points_batch((H, W), past2now)
        future2now = normalize_points_batch((H, W), future2now)
        return heatmaps, batch_locs, batch_scos, past2now, future2now, fb_check

    def SBR_forward_backward(self, landmarks, Fflows, Bflows):
        batch, frames, points, _ = landmarks.size()
        batch, _, H, W, _ = Fflows.size()
        F_gather_flows = Fflows.view(batch * (frames - 1), H, W, 2)
        F_landmarks = landmarks[:, :-1].contiguous().view(batch * (frames - 1), points, 2)
        _, past2now_locs = batch_interpolate_flow(F_gather_flows, F_landmarks, True)
        past2now_locs = past2now_locs.view(batch, frames - 1, points, 2)
        B_gather_flows = Bflows.view(batch * (frames - 1), H, W, 2)
        B_landmarks = landmarks[:, 1:].contiguous().view(batch * (frames - 1), points, 2)
        _, future2now_locs = batch_interpolate_flow(B_gather_flows, B_landmarks, True)
        future2now_locs = future2now_locs.view(batch, frames - 1, points, 2)
        current_locs = start_locs = landmarks[:, 0]
        for i in range(1, frames):
            _, current_locs = batch_interpolate_flow(Fflows[:, i - 1], current_locs, True)
        finish_locs = current_locs
        for i in range(frames - 1, 0, -1):
            _, finish_locs = batch_interpolate_flow(Bflows[:, i - 1], finish_locs, True)
        return past2now_locs, future2now_locs, start_locs - finish_locs


def denormalize_L(x, L):
    return (x + 1.0) / 2.0 * (L - 1)


def denormalize_points_batch(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(shape) == 2, 'invalid shape : {:}'.format(shape)
    assert isinstance(points, torch.Tensor) and points.shape[-1] == 2, 'points are wrong : {:}'.format(points.shape)
    (H, W), points = shape, points.clone()
    x = denormalize_L(points[..., 0], W)
    y = denormalize_L(points[..., 1], H)
    return torch.stack((x, y), dim=-1)


class TemporalREG(nn.Module):

    def __init__(self, model, config, points):
        super(TemporalREG, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(config)
        self.points = points
        self.center_idx = self.config.video_L

    def forward(self, inputs, Fflows, Bflows, is_images):
        assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {}'.format(inputs.size())
        batch_size, sequence, C, H, W = list(inputs.size())
        gathered_inputs = inputs.view(batch_size * sequence, C, H, W)
        batch_locs = self.detector(gathered_inputs)
        real_locs = denormalize_points_batch((H, W), batch_locs)
        real_locs = real_locs.view(batch_size, sequence, self.points, 2)
        batch_locs = batch_locs.view(batch_size, sequence, self.points, 2)
        batch_next, batch_fback, batch_back = [], [], []
        past2now, future2now, fb_check = self.SBR_forward_backward(real_locs, Fflows, Bflows)
        past2now, future2now = normalize_points_batch((H, W), past2now), normalize_points_batch((H, W), future2now)
        return batch_locs, past2now, future2now, fb_check

    def SBR_forward_backward(self, landmarks, Fflows, Bflows):
        batch, frames, points, _ = landmarks.size()
        batch, _, H, W, _ = Fflows.size()
        F_gather_flows = Fflows.view(batch * (frames - 1), H, W, 2)
        F_landmarks = landmarks[:, :-1].contiguous().view(batch * (frames - 1), points, 2)
        _, past2now_locs = batch_interpolate_flow(F_gather_flows, F_landmarks, True)
        past2now_locs = past2now_locs.view(batch, frames - 1, points, 2)
        B_gather_flows = Bflows.view(batch * (frames - 1), H, W, 2)
        B_landmarks = landmarks[:, 1:].contiguous().view(batch * (frames - 1), points, 2)
        _, future2now_locs = batch_interpolate_flow(B_gather_flows, B_landmarks, True)
        future2now_locs = future2now_locs.view(batch, frames - 1, points, 2)
        current_locs = start_locs = landmarks[:, 0]
        for i in range(1, frames):
            _, current_locs = batch_interpolate_flow(Fflows[:, i - 1], current_locs, True)
        finish_locs = current_locs
        for i in range(frames - 1, 0, -1):
            _, finish_locs = batch_interpolate_flow(Bflows[:, i - 1], finish_locs, True)
        return past2now_locs, future2now_locs, start_locs - finish_locs


class SpatialTemporalMultiviewHEAT(nn.Module):

    def __init__(self, model, config, points):
        super(SpatialTemporalMultiviewHEAT, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(config)
        self.points = points
        self.center_idx = self.config.video_L

    def forward(self, inputs, Fflows, Bflows, stm_inputs, is_images):
        assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {:}'.format(inputs.size())
        assert stm_inputs.dim() == 5, 'This model accepts 5 dimension stm-input tensor: {:}'.format(stm_inputs.size())
        vbatch_size, sequence, C, H, W = inputs.size()
        mbatch_size, views, C, H, W = stm_inputs.size()
        gathered_N_inputs = inputs.view(vbatch_size * sequence, C, H, W)
        gathered_M_inputs = stm_inputs.view(mbatch_size * views, C, H, W)
        gathered_inputs = torch.cat((gathered_N_inputs, gathered_M_inputs))
        gathered_F, gathered_heatmaps, gathered_batch_locs, gathered_batch_scos = self.detector(gathered_inputs)
        batch_heatmaps = [x[:vbatch_size * sequence].view(vbatch_size, sequence, self.points, H // self.downsample, W // self.downsample) for x in gathered_heatmaps]
        multiview_heatmaps = [x[vbatch_size * sequence:].view(mbatch_size, views, self.points, H // self.downsample, W // self.downsample) for x in gathered_heatmaps]
        batch_locs = gathered_batch_locs[:vbatch_size * sequence].view(vbatch_size, sequence, self.points, 2)
        batch_scos = gathered_batch_scos[:vbatch_size * sequence].view(vbatch_size, sequence, self.points)
        multiview_locs = gathered_batch_locs[vbatch_size * sequence:].view(mbatch_size, views, self.points, 2)
        multiview_scos = gathered_batch_scos[vbatch_size * sequence:].view(mbatch_size, views, self.points)
        batch_locs = batch_locs.view(vbatch_size, sequence, self.points, 2)
        batch_next, batch_fback, batch_back = [], [], []
        past2now, future2now, fb_check = self.SBR_forward_backward(batch_locs, Fflows, Bflows)
        batch_locs = normalize_points_batch((H, W), batch_locs)
        multiview_locs = normalize_points_batch((H, W), multiview_locs)
        past2now, future2now = normalize_points_batch((H, W), past2now), normalize_points_batch((H, W), future2now)
        return batch_heatmaps, batch_locs, batch_scos, past2now, future2now, fb_check, multiview_heatmaps, multiview_locs

    def SBR_forward_backward(self, landmarks, Fflows, Bflows):
        batch, frames, points, _ = landmarks.size()
        batch, _, H, W, _ = Fflows.size()
        F_gather_flows = Fflows.view(batch * (frames - 1), H, W, 2)
        F_landmarks = landmarks[:, :-1].contiguous().view(batch * (frames - 1), points, 2)
        _, past2now_locs = batch_interpolate_flow(F_gather_flows, F_landmarks, True)
        past2now_locs = past2now_locs.view(batch, frames - 1, points, 2)
        B_gather_flows = Bflows.view(batch * (frames - 1), H, W, 2)
        B_landmarks = landmarks[:, 1:].contiguous().view(batch * (frames - 1), points, 2)
        _, future2now_locs = batch_interpolate_flow(B_gather_flows, B_landmarks, True)
        future2now_locs = future2now_locs.view(batch, frames - 1, points, 2)
        current_locs = start_locs = landmarks[:, 0]
        for i in range(1, frames):
            _, current_locs = batch_interpolate_flow(Fflows[:, i - 1], current_locs, True)
        finish_locs = current_locs
        for i in range(frames - 1, 0, -1):
            _, finish_locs = batch_interpolate_flow(Bflows[:, i - 1], finish_locs, True)
        return past2now_locs, future2now_locs, start_locs - finish_locs


class SpatialTemporalMultiviewREG(nn.Module):

    def __init__(self, model, config, points):
        super(SpatialTemporalMultiviewREG, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(config)
        self.points = points
        self.center_idx = self.config.video_L

    def forward(self, inputs, Fflows, Bflows, stm_inputs, is_images, use_sbr, use_sbt):
        assert inputs.dim() == 5, 'This model accepts 5 dimension input tensor: {:}'.format(inputs.size())
        assert stm_inputs.dim() == 5, 'This model accepts 5 dimension stm-input tensor: {:}'.format(stm_inputs.size())
        vbatch_size, sequence, C, H, W = inputs.size()
        mbatch_size, views, C, H, W = stm_inputs.size()
        gathered_N_inputs = inputs.view(vbatch_size * sequence, C, H, W)
        gathered_M_inputs = stm_inputs.view(mbatch_size * views, C, H, W)
        gathered_inputs = torch.cat((gathered_N_inputs, gathered_M_inputs))
        gathered_batch_locs = self.detector(gathered_inputs)
        batch_locs = gathered_batch_locs[:vbatch_size * sequence]
        multiview_locs = gathered_batch_locs[vbatch_size * sequence:].view(mbatch_size, views, self.points, 2)
        real_locs = denormalize_points_batch((H, W), batch_locs)
        real_locs = real_locs.view(vbatch_size, sequence, self.points, 2)
        batch_locs = batch_locs.view(vbatch_size, sequence, self.points, 2)
        batch_next, batch_fback, batch_back = [], [], []
        past2now, future2now, fb_check = self.SBR_forward_backward(real_locs, Fflows, Bflows)
        past2now, future2now = normalize_points_batch((H, W), past2now), normalize_points_batch((H, W), future2now)
        return batch_locs, past2now, future2now, fb_check, multiview_locs

    def SBR_forward_backward(self, landmarks, Fflows, Bflows):
        batch, frames, points, _ = landmarks.size()
        batch, _, H, W, _ = Fflows.size()
        F_gather_flows = Fflows.view(batch * (frames - 1), H, W, 2)
        F_landmarks = landmarks[:, :-1].contiguous().view(batch * (frames - 1), points, 2)
        _, past2now_locs = batch_interpolate_flow(F_gather_flows, F_landmarks, True)
        past2now_locs = past2now_locs.view(batch, frames - 1, points, 2)
        B_gather_flows = Bflows.view(batch * (frames - 1), H, W, 2)
        B_landmarks = landmarks[:, 1:].contiguous().view(batch * (frames - 1), points, 2)
        _, future2now_locs = batch_interpolate_flow(B_gather_flows, B_landmarks, True)
        future2now_locs = future2now_locs.view(batch, frames - 1, points, 2)
        current_locs = start_locs = landmarks[:, 0]
        for i in range(1, frames):
            _, current_locs = batch_interpolate_flow(Fflows[:, i - 1], current_locs, True)
        finish_locs = current_locs
        for i in range(frames - 1, 0, -1):
            _, finish_locs = batch_interpolate_flow(Bflows[:, i - 1], finish_locs, True)
        return past2now_locs, future2now_locs, start_locs - finish_locs


class MaskedLoss(nn.Module):

    def __init__(self, xtype, reduction):
        super(MaskedLoss, self).__init__()
        reductions = ['none', 'mean', 'sum', 'batch']
        assert reduction in reductions, 'Invalid {:}, not in {:}'.format(reduction, reductions)
        self.reduction = reduction
        if self.reduction == 'batch':
            self.reduction = 'sum'
            self.divide_by_batch = True
        else:
            self.divide_by_batch = False
        if xtype == 'mse':
            self.func = F.mse_loss
        elif xtype == 'smoothL1':
            self.func = F.smooth_l1_loss
        elif xtype == 'L1':
            self.func = F.l1_loss
        else:
            raise ValueError('Invalid Type : {:}'.format(xtype))
        self.loss_l1_func = F.l1_loss
        self.loss_mse_func = F.mse_loss
        self.xtype = xtype

    def extra_repr(self):
        return 'type={xtype}, reduction={reduction}, divide_by_batch={divide_by_batch}'.format(**self.__dict__)

    def forward(self, inputs, targets, masks):
        batch_size = inputs.size(0)
        if masks is not None:
            inputs = torch.masked_select(inputs, masks.bool())
            targets = torch.masked_select(targets, masks.bool())
        x_loss = self.func(inputs, targets, reduction=self.reduction)
        if self.divide_by_batch:
            x_loss = x_loss / batch_size
        return x_loss


class HourglassNet(nn.Module):

    def __init__(self, nStack, nModules, nFeats, nJoints):
        super(HourglassNet, self).__init__()
        self.downsample = 4
        self.pts_num = nJoints
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.conv = nn.Sequential(nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.ress = nn.Sequential(Residual(64, 128), nn.MaxPool2d(kernel_size=2, stride=2), Residual(128, 128), Residual(128, self.nFeats))
        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1), nn.BatchNorm2d(self.nFeats), nn.ReLU(inplace=True))
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, nJoints, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(nJoints, self.nFeats, bias=True, kernel_size=1, stride=1))
        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x = self.conv(x)
        x = self.ress(x)
        out = []
        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
        return out


class TeacherNet(nn.Module):

    def __init__(self, input_dim, n_layers=3):
        super(TeacherNet, self).__init__()
        sequence = [nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=1, padding=1, bias=True), nn.InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.model(inputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BN_ReLU_C1x1,
     lambda: ([], {'in_num': 4, 'out_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'stride': 1, 'downsample': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (HierarchicalPMS,
     lambda: ([], {'numIn': 4, 'numOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hourglass,
     lambda: ([], {'n': 4, 'nModules': 4, 'nFeats': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (HourglassNet,
     lambda: ([], {'nStack': 4, 'nModules': 4, 'nFeats': 4, 'nJoints': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2REG,
     lambda: ([], {'input_dim': 4, 'input_channel': 4, 'width_mult': 4, 'pts_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Residual,
     lambda: ([], {'numIn': 4, 'numOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_D_X_Y_landmark_detection(_paritybench_base):
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

