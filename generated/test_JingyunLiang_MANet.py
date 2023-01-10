import sys
_module = sys.modules[__name__]
del sys
GTLQ_dataset = _module
GT_dataset = _module
LQ_dataset = _module
data = _module
data_sampler = _module
util = _module
interactive_explore = _module
B_model = _module
SRGAN_model = _module
SR_model = _module
models = _module
base_model = _module
lr_scheduler = _module
lr_scheduler_test = _module
MANet_arch = _module
discriminator_vgg_arch = _module
loss = _module
module_util = _module
networks = _module
options = _module
prepare_testset = _module
test = _module
train = _module
util = _module

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


import torch


import torch.utils.data as data


import torch.nn.functional as F


import logging


import torch.utils.data


import math


from torch.utils.data.sampler import Sampler


import torch.distributed as dist


from collections import OrderedDict


import torch.nn as nn


import torch.nn.init as init


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.cuda.amp import autocast as autocast


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


import functools


import torchvision


import time


import torch.multiprocessing as mp


from torchvision.utils import make_grid


from torch.autograd import Variable


import collections


from scipy import signal


import scipy


import matplotlib


import matplotlib.pyplot as plt


from scipy.interpolate import interp2d


class DegradationModel(nn.Module):

    def __init__(self, kernel_size=15, scale=4, sv_degradation=True):
        super(DegradationModel, self).__init__()
        if sv_degradation:
            self.blur_layer = util.BatchBlur_SV(l=kernel_size, padmode='replication')
            self.sample_layer = util.BatchSubsample(scale=scale)
        else:
            self.blur_layer = util.BatchBlur(l=kernel_size, padmode='replication')
            self.sample_layer = util.BatchSubsample(scale=scale)

    def forward(self, image, kernel):
        return self.sample_layer(self.blur_layer(image, kernel))


class MAConv(nn.Module):
    """ Mutual Affine Convolution (MAConv) layer """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'
        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []
        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)
            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)
            setattr(self, 'fc{}'.format(i), nn.Sequential(*[nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, kernel_size=1, stride=1, padding=0, bias=True)]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []
        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)), (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))
        return torch.cat(output, 1)


class MABlock(nn.Module):
    """ Residual block based on MAConv """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, split=2, reduction=2):
        super(MABlock, self).__init__()
        self.res = nn.Sequential(*[MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction), nn.ReLU(inplace=True), MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction)])

    def forward(self, x):
        return x + self.res(x)


class SFT_Layer(nn.Module):
    """ SFT layer """

    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class ResidualDenseBlock_5C(nn.Module):
    """  Residual Dense Block """

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_SFT(nn.Module):
    """ Residual in Residual Dense Block with SFT layer """

    def __init__(self, nf, gc=32, para=15):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, input):
        out = self.SFT(input[0], input[1])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return [out * 0.2 + input[0], input[1]]


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class MANet(nn.Module):
    """ Network of MANet"""

    def __init__(self, in_nc=3, kernel_size=21, nc=[128, 256], nb=1, split=2):
        super(MANet, self).__init__()
        self.kernel_size = kernel_size
        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)], nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0, bias=True))
        self.m_body = sequential(*[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])
        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0], kernel_size=2, stride=2, padding=0, bias=True), *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x = self.m_body(x2)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        x = self.softmax(x)
        return x


class MANet_s1(nn.Module):
    """ stage1, train MANet"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth', code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s1, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.kernel_estimation = MANet(in_nc=in_nc, kernel_size=kernel_size, nc=[manet_nf, manet_nf * 2], nb=manet_nb, split=split)

    def forward(self, x, gt_K):
        kernel = self.kernel_estimation(x)
        kernel = F.interpolate(kernel, scale_factor=self.scale, mode='nearest').flatten(2).permute(0, 2, 1)
        kernel = kernel.view(-1, kernel.size(1), self.kernel_size, self.kernel_size)
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return out, kernel


class MANet_s2(nn.Module):
    """ stage2, train nonblind RRDB-SFT"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth', code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s2, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.register_buffer('pca_matrix', torch.load(pca_path).unsqueeze(0).unsqueeze(3).unsqueeze(4))
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=code_length)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * scale ** 2, kernel_size=3, stride=1, padding=1, bias=True), nn.PixelShuffle(scale))

    def forward(self, x, gt_K):
        with torch.no_grad():
            kernel_pca_code = torch.mm(gt_K.flatten(1), self.pca_matrix.squeeze()).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            kernel = gt_K
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, kernel_pca_code])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out, kernel


class MANet_s3(nn.Module):
    """ stage3, fine-tune nonblind SR model based on MANet predictions"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth', code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s3, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.kernel_estimation = MANet(in_nc=in_nc, kernel_size=kernel_size, nc=[manet_nf, manet_nf * 2], nb=manet_nb, split=split)
        self.register_buffer('pca_matrix', torch.load(pca_path).unsqueeze(0).unsqueeze(3).unsqueeze(4))
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=code_length)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * scale ** 2, kernel_size=3, stride=1, padding=1, bias=True), nn.PixelShuffle(scale))

    def forward(self, x, gt_K):
        with torch.no_grad():
            kernel = self.kernel_estimation(x)
            kernel_pca_code = (kernel.unsqueeze(2) * self.pca_matrix).sum(1, keepdim=False)
            kernel = F.interpolate(kernel, scale_factor=self.scale, mode='nearest').flatten(2).permute(0, 2, 1)
            kernel = kernel.view(-1, kernel.size(1), self.kernel_size, self.kernel_size)
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, kernel_pca_code])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out, kernel


class Discriminator_VGG_128(nn.Module):

    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer + 1])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class GANLoss(nn.Module):

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


class ReconstructionLoss(nn.Module):

    def __init__(self, losstype='l2', eps=1e-06):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            None
            return 0


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and img.ndim in {2, 3}


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))
    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' + 'not {}'.format(type(npimg)))
    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError('Incorrect mode ({}) supplied for input type {}. Should be {}'.format(mode, np.dtype, expected_mode))
        mode = expected_mode
    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError('Only modes {} are supported for 4D inputs'.format(permitted_4_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError('Only modes {} are supported for 3D inputs'.format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))
    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float().div(255)
    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class BatchBicubic(nn.Module):

    def __init__(self, scale=4):
        super(BatchBicubic, self).__init__()
        self.scale = scale

    def forward(self, input):
        tensor = input.cpu().data
        B, C, H, W = tensor.size()
        H_new = int(H / self.scale)
        W_new = int(W / self.scale)
        tensor_view = tensor.view((B * C, 1, H, W))
        re_tensor = torch.zeros((B * C, 1, H_new, W_new))
        for i in range(B * C):
            img = to_pil_image(tensor_view[i])
            re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
        re_tensor_view = re_tensor.view((B, C, H_new, W_new))
        return re_tensor_view


class BatchSubsample(nn.Module):

    def __init__(self, scale=4):
        super(BatchSubsample, self).__init__()
        self.scale = scale

    def forward(self, input):
        return input[:, :, 0::self.scale, 0::self.scale]


class CircularPad2d(nn.Module):

    def __init__(self, pad):
        super(CircularPad2d, self).__init__()
        self.pad = pad

    def forward(self, input):
        return F.pad(input, pad=self.pad, mode='circular')


class BatchBlur(nn.Module):

    def __init__(self, l=15, padmode='reflection'):
        super(BatchBlur, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'circular':
            if l % 2 == 1:
                self.pad = CircularPad2d((l // 2, l // 2, l // 2, l // 2))
            else:
                self.pad = CircularPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        else:
            raise NotImplementedError

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))


class BatchBlur_SV(nn.Module):

    def __init__(self, l=15, padmode='reflection'):
        super(BatchBlur_SV, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'circular':
            if l % 2 == 1:
                self.pad = CircularPad2d((l // 2, l // 2, l // 2, l // 2))
            else:
                self.pad = CircularPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2)
            kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
            out_unf = (pad * kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchBlur,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 1, 15, 15])], {}),
     True),
    (BatchSubsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MAConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MANet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (MANet_s1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReconstructionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock_noBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (VGGFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_JingyunLiang_MANet(_paritybench_base):
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

