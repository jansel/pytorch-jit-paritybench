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
basic_batch = _module
flop_benchmark = _module
initialization = _module
layer_utils = _module
model_utils = _module
student_cpm = _module
student_hg = _module
teacher = _module
test = _module

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


import torch.nn as nn


import random


import numpy as np


import torch.nn.functional as F


import math


from torch.nn import init


from torch.autograd import Variable


import functools


import itertools


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


from scipy.ndimage.interpolation import zoom


import copy


from copy import deepcopy


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
        xx_channel, yy_channel = xx_channel.type_as(input_tensor
            ), yy_channel.type_as(input_tensor)
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(
                yy_channel - 0.5, 2))
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
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, kernels, stride, pad):
    return nn.Sequential(nn.Conv2d(inp, oup, kernels, stride, pad, bias=
        False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


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
        interverted_residual_setting = [[1, 48, 1, 1], [2, 48, 5, 2], [2, 
            96, 1, 2], [4, 96, 6, 1], [2, 16, 1, 1]]
        input_channel = int(input_channel * width_mult)
        features = [conv_bn(input_dim, input_channel, (3, 3), 2, 1)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                features.append(block(input_channel, output_channel, stride,
                    expand_ratio=t))
                input_channel = output_channel
        features.append(nn.AdaptiveAvgPool2d((14, 14)))
        self.features = nn.Sequential(*features)
        self.S1 = nn.Sequential(CoordConv(input_channel, input_channel * 2,
            True, kernel_size=3, padding=1), conv_bn(input_channel * 2, 
            input_channel * 2, (3, 3), 2, 1))
        self.S2 = nn.Sequential(CoordConv(input_channel * 2, input_channel *
            4, True, kernel_size=3, padding=1), conv_bn(input_channel * 4, 
            input_channel * 8, (7, 7), 1, 0))
        output_neurons = (14 * 14 * input_channel + 7 * 7 * input_channel *
            2 + input_channel * 8)
        self.locator = nn.Sequential(nn.Linear(output_neurons, pts_num * 2))
        self.apply(weights_init_reg)

    def forward(self, x):
        batch, C, H, W = x.size()
        features = self.features(x)
        S1 = self.S1(features)
        S2 = self.S2(S1)
        tensors = torch.cat((features.view(batch, -1), S1.view(batch, -1),
            S2.view(batch, -1)), dim=1)
        batch_locs = self.locator(tensors).view(batch, self.pts_num, 2)
        return batch_locs


class NLayerDiscriminator(nn.Module):

    def __init__(self, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(3, 64, kernel_size=kw, stride=2, padding=padw
            ), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=True), nn.
                InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(
                0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size
            =kw, stride=1, padding=padw, bias=True), nn.InstanceNorm2d(64 *
            nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0, tensor=torch.FloatTensor):
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
            create_label = (self.real_label_var is None or self.
                real_label_var.numel() != input.numel())
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False
                    )
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None or self.
                fake_label_var.numel() != input.numel())
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False
                    )
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            use_dropout, use_bias)

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
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), nn.InstanceNorm2d(dim, affine=False), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), nn.InstanceNorm2d(dim, affine=False)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):

    def __init__(self, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7,
            padding=0, bias=True), nn.InstanceNorm2d(64, affine=False), nn.
            ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=True), nn.InstanceNorm2d(64 *
                mult * 2, affine=False), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(64 * mult, padding_type=padding_type,
                use_dropout=False, use_bias=True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                True), nn.InstanceNorm2d(int(64 * mult / 2), affine=False),
                nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
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
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init_xavier)
    return netG


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=1e-06):
    assert heatmap.dim(
        ) == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap
        .size())
    assert radius > 0 and isinstance(radius, numbers.Number
        ), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1.0 + 2.0 * x.data / (L - 1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h +
        radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, (0), (0)] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, (0), (2)] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, (1), (1)] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, (1), (2)] = (boxes[3] + boxes[1]) / 2
    theta = MU.np2variable(affine_parameter, heatmap.is_cuda, False)
    grid_size = torch.Size([num_pts, 1, radius * 2 + 1, radius * 2 + 1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)
    X = MU.np2variable(torch.arange(-radius, radius + 1), heatmap.is_cuda, 
        False).view(1, 1, radius * 2 + 1)
    Y = MU.np2variable(torch.arange(-radius, radius + 1), heatmap.is_cuda, 
        False).view(1, radius * 2 + 1, 1)
    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1
        ) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1
        ) / sum_region + index_h
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
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.
            Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU
            (inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True))
        self.downsample = 8
        pts_num = self.config.pts_num
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128,
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(
            inplace=True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        self.stage2 = nn.Sequential(nn.Conv2d(128 * 2 + pts_num * 2, 128,
            kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation
            =1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        self.stage3 = nn.Sequential(nn.Conv2d(128 * 2 + pts_num, 128,
            kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation
            =1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        assert self.config.num_stages >= 1, 'stages of cpm must >= 1'

    def set_mode(self, mode):
        if mode.lower() == 'train':
            self.train()
            for m in self.netG_A.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.
                    InstanceNorm2d):
                    m.eval()
            for m in self.netG_B.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.
                    InstanceNorm2d):
                    m.eval()
        elif mode.lower() == 'eval':
            self.eval()
        else:
            raise NameError('The wrong mode : {}'.format(mode))

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': self.netG_A.parameters(), 'lr': 0,
            'weight_decay': 0}, {'params': self.netG_B.parameters(), 'lr': 
            0, 'weight_decay': 0}, {'params': get_parameters(self.features,
            bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.features, bias=True), 'lr': 
            base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self
            .CPM_feature, bias=False), 'lr': base_lr, 'weight_decay':
            base_weight_decay}, {'params': get_parameters(self.CPM_feature,
            bias=True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params':
            get_parameters(self.stage1, bias=False), 'lr': base_lr,
            'weight_decay': base_weight_decay}, {'params': get_parameters(
            self.stage1, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters(self.stage2, bias=False), 'lr': 
            base_lr * 4, 'weight_decay': base_weight_decay}, {'params':
            get_parameters(self.stage2, bias=True), 'lr': base_lr * 8,
            'weight_decay': 0}, {'params': get_parameters(self.stage3, bias
            =False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.stage3, bias=True), 'lr': 
            base_lr * 8, 'weight_decay': 0}]
        return params_dict

    def forward(self, inputs):
        assert inputs.dim(
            ) == 4, 'This model accepts 4 dimension input tensor: {}'.format(
            inputs.size())
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
        cpm_stage2 = self.stage2(torch.cat([xfeature, stage1s[0], stage1s[1
            ]], 1))
        cpm_stage3 = self.stage3(torch.cat([xfeature, cpm_stage2], 1))
        batch_cpms = [stage1s[0], stage1s[1]] + [cpm_stage2, cpm_stage3]
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(cpm_stage3
                [ibatch], self.config.argmax, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(
            batch_scos)
        return batch_cpms, batch_locs, batch_scos, inputs[1:]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
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

    def __init__(self, model_config):
        super(VGG16_base, self).__init__()
        self.config = model_config.copy()
        self.downsample = 1
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.
            Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU
            (inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True))
        self.downsample = 8
        pts_num = self.config.pts_num
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128,
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(
            inplace=True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        self.stage2 = nn.Sequential(nn.Conv2d(128 + pts_num, 128,
            kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation
            =1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        self.stage3 = nn.Sequential(nn.Conv2d(128 + pts_num, 128,
            kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7, dilation
            =1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        assert self.config.num_stages >= 1, 'stages of cpm must >= 1'

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': get_parameters(self.features, bias=False),
            'lr': base_lr, 'weight_decay': base_weight_decay}, {'params':
            get_parameters(self.features, bias=True), 'lr': base_lr * 2,
            'weight_decay': 0}, {'params': get_parameters(self.CPM_feature,
            bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_feature, bias=True), 'lr': 
            base_lr * 2, 'weight_decay': 0}, {'params': get_parameters(self
            .stage1, bias=False), 'lr': base_lr, 'weight_decay':
            base_weight_decay}, {'params': get_parameters(self.stage1, bias
            =True), 'lr': base_lr * 2, 'weight_decay': 0}, {'params':
            get_parameters(self.stage2, bias=False), 'lr': base_lr * 4,
            'weight_decay': base_weight_decay}, {'params': get_parameters(
            self.stage2, bias=True), 'lr': base_lr * 8, 'weight_decay': 0},
            {'params': get_parameters(self.stage3, bias=False), 'lr': 
            base_lr * 4, 'weight_decay': base_weight_decay}, {'params':
            get_parameters(self.stage3, bias=True), 'lr': base_lr * 8,
            'weight_decay': 0}]
        return params_dict

    def forward(self, inputs):
        assert inputs.dim(
            ) == 4, 'This model accepts 4 dimension input tensor: {}'.format(
            inputs.size())
        batch_size = inputs.size(0)
        num_stages, num_pts = self.config.num_stages, self.config.pts_num - 1
        batch_cpms = []
        batch_locs, batch_scos = [], []
        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        stage1 = self.stage1(xfeature)
        stage2 = self.stage2(torch.cat([xfeature, stage1], 1))
        stage3 = self.stage3(torch.cat([xfeature, stage2], 1))
        batch_cpms = [stage1, stage2, stage3]
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(stage3[
                ibatch], self.config.argmax, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(
            batch_scos)
        return batch_cpms, batch_locs, batch_scos


class SobelConv(nn.Module):

    def __init__(self, tag, dtype):
        super(SobelConv, self).__init__()
        if tag == 'x':
            Sobel = np.array([[-1.0 / 8, 0, 1.0 / 8], [-2.0 / 8, 0, 2.0 / 8
                ], [-1.0 / 8, 0, 1.0 / 8]])
        elif tag == 'y':
            Sobel = np.array([[-1.0 / 8, -2.0 / 8, -1.0 / 8], [0, 0, 0], [
                1.0 / 8, 2.0 / 8, 1.0 / 8]])
        else:
            raise NameError('Do not know this tag for Sobel Kernel : {}'.
                format(tag))
        Sobel = torch.from_numpy(Sobel).type(dtype)
        Sobel = Sobel.view(1, 1, 3, 3)
        self.register_buffer('weight', Sobel)
        self.tag = tag

    def forward(self, input):
        weight = self.weight.expand(input.size(1), 1, 3, 3).contiguous()
        return F.conv2d(input, weight, groups=input.size(1), padding=1)

    def __repr__(self):
        return '{name}(tag={tag})'.format(name=self.__class__.__name__, **
            self.__dict__)


class LK(nn.Module):

    def __init__(self, model, lkconfig, points):
        super(LK, self).__init__()
        self.detector = model
        self.downsample = self.detector.downsample
        self.config = copy.deepcopy(lkconfig)
        self.points = points

    def forward(self, inputs):
        assert inputs.dim(
            ) == 5, 'This model accepts 5 dimension input tensor: {}'.format(
            inputs.size())
        batch_size, sequence, C, H, W = list(inputs.size())
        gathered_inputs = inputs.view(batch_size * sequence, C, H, W)
        heatmaps, batch_locs, batch_scos = self.detector(gathered_inputs)
        heatmaps = [x.view(batch_size, sequence, self.points, H // self.
            downsample, W // self.downsample) for x in heatmaps]
        batch_locs, batch_scos = batch_locs.view(batch_size, sequence, self
            .points, 2), batch_scos.view(batch_size, sequence, self.points)
        batch_next, batch_fback, batch_back = [], [], []
        for ibatch in range(batch_size):
            feature_old = inputs[ibatch]
            nextPts, fbackPts, backPts = lk.lk_forward_backward_batch(inputs
                [ibatch], batch_locs[ibatch], self.config.window, self.
                config.steps)
            batch_next.append(nextPts)
            batch_fback.append(fbackPts)
            batch_back.append(backPts)
        batch_next, batch_fback, batch_back = torch.stack(batch_next
            ), torch.stack(batch_fback), torch.stack(batch_back)
        return (heatmaps, batch_locs, batch_scos, batch_next, batch_fback,
            batch_back)


class VGG16_base(nn.Module):

    def __init__(self, config, pts_num):
        super(VGG16_base, self).__init__()
        self.config = deepcopy(config)
        self.downsample = 8
        self.pts_num = pts_num
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.
            Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU
            (inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True))
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True))
        assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(
            self.config.stages)
        stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1
            ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128,
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config.stages):
            stagex = nn.Sequential(nn.Conv2d(128 + pts_num, 128,
                kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True
                ), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3
                ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7,
                dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(
                128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(
                inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=
                1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
                kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True
                ), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1
                ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1,
                padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num,
                kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': get_parameters(self.features, bias=False),
            'lr': base_lr, 'weight_decay': base_weight_decay}, {'params':
            get_parameters(self.features, bias=True), 'lr': base_lr * 2,
            'weight_decay': 0}, {'params': get_parameters(self.CPM_feature,
            bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_feature, bias=True), 'lr': 
            base_lr * 2, 'weight_decay': 0}]
        for stage in self.stages:
            params_dict.append({'params': get_parameters(stage, bias=False),
                'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True),
                'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    def forward(self, inputs):
        assert inputs.dim(
            ) == 4, 'This model accepts 4 dimension input tensor: {}'.format(
            inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms, batch_locs, batch_scos = [], [], []
        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config.stages):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]
                    ], 1))
            batch_cpms.append(cpm)
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(batch_cpms
                [-1][ibatch], self.config.argmax, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(
            batch_scos)
        return batch_cpms, batch_locs, batch_scos


class Residual(nn.Module):

    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        middle = self.numOut // 2
        self.conv_A = nn.Sequential(nn.BatchNorm2d(numIn), nn.ReLU(inplace=
            True), nn.Conv2d(numIn, middle, kernel_size=1, dilation=1,
            padding=0, bias=True))
        self.conv_B = nn.Sequential(nn.BatchNorm2d(middle), nn.ReLU(inplace
            =True), nn.Conv2d(middle, middle, kernel_size=3, dilation=1,
            padding=1, bias=True))
        self.conv_C = nn.Sequential(nn.BatchNorm2d(middle), nn.ReLU(inplace
            =True), nn.Conv2d(middle, numOut, kernel_size=1, dilation=1,
            padding=0, bias=True))
        if self.numIn != self.numOut:
            self.branch = nn.Sequential(nn.BatchNorm2d(self.numIn), nn.ReLU
                (inplace=True), nn.Conv2d(self.numIn, self.numOut,
                kernel_size=1, dilation=1, padding=0, bias=True))

    def forward(self, x):
        residual = x
        main = self.conv_A(x)
        main = self.conv_B(main)
        main = self.conv_C(main)
        if hasattr(self, 'branch'):
            residual = self.branch(residual)
        return main + residual


class VGG16_base(nn.Module):

    def __init__(self, model_config, pts_num):
        super(VGG16_base, self).__init__()
        self.config = deepcopy(model_config)
        self.downsample = 8
        self.pts_num = pts_num
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            dilation=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.
            Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU
            (inplace=True), nn.Conv2d(256, 256, kernel_size=3, dilation=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512,
            kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, dilation
            =1, padding=1), nn.ReLU(inplace=True))
        self.CPM_feature = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True))
        assert self.config['stages'] >= 1, 'stages of cpm must >= 1'
        stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1
            ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128,
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=
            True), nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config['stages']):
            stagex = nn.Sequential(nn.Conv2d(128 + pts_num, 128,
                kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True
                ), nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3
                ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=7,
                dilation=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(
                128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(
                inplace=True), nn.Conv2d(128, 128, kernel_size=3, dilation=
                1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128,
                kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True
                ), nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1
                ), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=1,
                padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, pts_num,
                kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [{'params': get_parameters(self.features, bias=False),
            'lr': base_lr, 'weight_decay': base_weight_decay}, {'params':
            get_parameters(self.features, bias=True), 'lr': base_lr * 2,
            'weight_decay': 0}, {'params': get_parameters(self.CPM_feature,
            bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_feature, bias=True), 'lr': 
            base_lr * 2, 'weight_decay': 0}]
        for stage in self.stages:
            params_dict.append({'params': get_parameters(stage, bias=False),
                'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True),
                'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    def forward(self, inputs):
        assert inputs.dim(
            ) == 4, 'This model accepts 4 dimension input tensor: {}'.format(
            inputs.size())
        batch_cpms = []
        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config['stages']):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]
                    ], 1))
            batch_cpms.append(cpm)
        return batch_cpms


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


class HourglassNet(nn.Module):

    def __init__(self, nStack, nModules, nFeats, nJoints):
        super(HourglassNet, self).__init__()
        self.downsample = 4
        self.pts_num = nJoints
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.conv = nn.Sequential(nn.Conv2d(3, 64, bias=True, kernel_size=7,
            stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.ress = nn.Sequential(Residual(64, 128), nn.MaxPool2d(
            kernel_size=2, stride=2), Residual(128, 128), Residual(128,
            self.nFeats))
        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [
            ], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=
                True, kernel_size=1, stride=1), nn.BatchNorm2d(self.nFeats),
                nn.ReLU(inplace=True))
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, nJoints, bias=True,
                kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True,
                    kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(nJoints, self.nFeats, bias=True,
                    kernel_size=1, stride=1))
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
        sequence = [nn.Conv2d(input_dim, 64, kernel_size=4, stride=2,
            padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult,
                kernel_size=4, stride=2, padding=1, bias=True), nn.
                InstanceNorm2d(64 * nf_mult, affine=False), nn.LeakyReLU(
                0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size
            =4, stride=1, padding=1, bias=True), nn.InstanceNorm2d(64 *
            nf_mult, affine=False), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=4, stride=1,
            padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.model(inputs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_D_X_Y_landmark_detection(_paritybench_base):
    pass
    def test_000(self):
        self._check(AddCoords(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CoordConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(MobileNetV2REG(*[], **{'input_dim': 4, 'input_channel': 4, 'width_mult': 4, 'pts_num': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(NLayerDiscriminator(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_005(self):
        self._check(GANLoss(*[], **{}), [], {'input': torch.rand([4, 4]), 'target_is_real': 4})

    def test_006(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Residual(*[], **{'numIn': 4, 'numOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(HourglassNet(*[], **{'nStack': 4, 'nModules': 4, 'nFeats': 4, 'nJoints': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_009(self):
        self._check(TeacherNet(*[], **{'input_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

