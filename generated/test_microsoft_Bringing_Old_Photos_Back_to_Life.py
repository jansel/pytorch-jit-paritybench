import sys
_module = sys.modules[__name__]
del sys
align_warp_back_multiple_dlib = _module
align_warp_back_multiple_dlib_HR = _module
detect_all_dlib = _module
detect_all_dlib_HR = _module
data = _module
base_dataset = _module
custom_dataset = _module
face_dataset = _module
image_folder = _module
pix2pix_dataset = _module
models = _module
networks = _module
architecture = _module
base_network = _module
encoder = _module
generator = _module
normalization = _module
pix2pix_model = _module
options = _module
base_options = _module
test_options = _module
test_face = _module
util = _module
iter_counter = _module
util = _module
visualizer = _module
GUI = _module
Create_Bigfile = _module
Load_Bigfile = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
online_dataset_for_old_photos = _module
detection = _module
detection_models = _module
antialiasing = _module
networks = _module
util = _module
NonLocal_feature_mapping_model = _module
base_model = _module
mapping_model = _module
models = _module
networks = _module
pix2pixHD_model = _module
pix2pixHD_model_DA = _module
base_options = _module
train_options = _module
test = _module
train_domain_A = _module
train_domain_B = _module
train_mapping = _module
image_pool = _module
util = _module
predict = _module
run = _module

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


import numpy as np


import matplotlib.pyplot as plt


from matplotlib.patches import Rectangle


import torch.nn.functional as F


import torchvision as tv


import torchvision.utils as vutils


import time


import torch.utils.data


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import torch.nn as nn


import torchvision


import torch.nn.utils.spectral_norm as spectral_norm


from torch.nn import init


import re


import scipy.misc


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import functools


from torch.autograd import Variable


import math


from torch.nn.utils import spectral_norm


from torchvision import models


from collections import OrderedDict


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.opt = opt
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)
        nhidden = 128
        pw = ks // 2
        if self.opt.no_parsing_map:
            self.mlp_shared = nn.Sequential(nn.Conv2d(3, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        else:
            self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc + 3, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, degraded_image):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        degraded_face = F.interpolate(degraded_image, size=x.size()[2:], mode='bilinear')
        if self.opt.no_parsing_map:
            actv = self.mlp_shared(degraded_face)
        else:
            actv = self.mlp_shared(torch.cat((segmap, degraded_face), dim=1))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.opt = opt
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, opt)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, opt)

    def forward(self, x, seg, degraded_image):
        x_s = self.shortcut(x, seg, degraded_image)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, degraded_image)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, degraded_image)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg, degraded_image):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, degraded_image))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, opt, activation=nn.ReLU(True), use_dropout=False, dilation=1):
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == 'zero':
            p = self.dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation), norm_layer(dim), activation]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class VGG19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class SPADEResnetBlock_non_spade(nn.Module):

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.opt = opt
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, opt)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, opt)

    def forward(self, x, seg, degraded_image):
        x_s = self.shortcut(x, seg, degraded_image)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + dx
        return out

    def shortcut(self, x, seg, degraded_image):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class SPADEGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        None
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        elif self.opt.no_parsing_map:
            self.fc = nn.Conv2d(3, 16 * nf, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '1':
            self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        else:
            self.head_0 = SPADEResnetBlock_non_spade(16 * nf, 16 * nf, opt)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '2':
            self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
            self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        else:
            self.G_middle_0 = SPADEResnetBlock_non_spade(16 * nf, 16 * nf, opt)
            self.G_middle_1 = SPADEResnetBlock_non_spade(16 * nf, 16 * nf, opt)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '3':
            self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        else:
            self.up_0 = SPADEResnetBlock_non_spade(16 * nf, 8 * nf, opt)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '4':
            self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        else:
            self.up_1 = SPADEResnetBlock_non_spade(8 * nf, 4 * nf, opt)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '5':
            self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        else:
            self.up_2 = SPADEResnetBlock_non_spade(4 * nf, 2 * nf, opt)
        if self.opt.injection_layer == 'all' or self.opt.injection_layer == '6':
            self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        else:
            self.up_3 = SPADEResnetBlock_non_spade(2 * nf, 1 * nf, opt)
        final_nc = nf
        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        sw = opt.load_size // 2 ** num_up_layers
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, degraded_image, z=None):
        seg = input
        if self.opt.use_vae:
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            if self.opt.no_parsing_map:
                x = F.interpolate(degraded_image, size=(self.sh, self.sw), mode='bilinear')
            else:
                x = F.interpolate(seg, size=(self.sh, self.sw), mode='nearest')
            x = self.fc(x)
        x = self.head_0(x, seg, degraded_image)
        x = self.up(x)
        x = self.G_middle_0(x, seg, degraded_image)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        x = self.G_middle_1(x, seg, degraded_image)
        x = self.up(x)
        x = self.up_0(x, seg, degraded_image)
        x = self.up(x)
        x = self.up_1(x, seg, degraded_image)
        x = self.up(x)
        x = self.up_2(x, seg, degraded_image)
        x = self.up(x)
        x = self.up_3(x, seg, degraded_image)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, degraded_image)
        x = self.conv_img(F.leaky_relu(x, 0.2))
        x = F.tanh(x)
        return x


class Pix2PixHDGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3, help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7, help='kernel size of the first convolution')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = 3
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)
        model = []
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2), norm_layer(nn.Conv2d(input_nc, opt.ngf, kernel_size=opt.resnet_initial_kernel_size, padding=0)), activation]
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2, kernel_size=3, stride=2, padding=1)), activation]
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)]
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int(opt.ngf * mult / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out, kernel_size=3, stride=2, padding=1, output_padding=1)), activation]
            mult = mult // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, degraded_image, z=None):
        return self.model(degraded_image)


class Pix2PixModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.ByteTensor if self.use_gpu() else torch.ByteTensor
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        input_semantics, real_image, degraded_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics, degraded_image, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, degraded_image, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, degraded_image, real_image)
            return fake_image
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return netG, netD, netE

    def preprocess_input(self, data):
        if not self.opt.isTrain:
            if self.use_gpu():
                data['label'] = data['label']
                data['image'] = data['image']
            return data['label'], data['image'], data['image']
        if self.use_gpu():
            data['label'] = data['label']
            data['degraded_image'] = data['degraded_image']
            data['image'] = data['image']
        return data['label'], data['image'], data['degraded_image']

    def compute_generator_loss(self, input_semantics, degraded_image, real_image):
        G_losses = {}
        fake_image, KLD_loss = self.generate_fake(input_semantics, degraded_image, real_image, compute_kld_loss=self.opt.use_vae)
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, degraded_image, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, degraded_image, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, degraded_image, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        fake_image = self.netG(input_semantics, degraded_image, z=z)
        assert not compute_kld_loss or self.opt.use_vae, 'You cannot compute KLD loss if opt.use_vae == False'
        return fake_image, KLD_loss

    def discriminate(self, input_semantics, fake_image, real_image):
        if self.opt.no_parsing_map:
            fake_concat = fake_image
            real_concat = real_image
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        None
    return PadLayer


class Downsample(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)), int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class UNetConvBlock(nn.Module):

    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, conv_num, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False), nn.ReflectionPad2d(1), nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, depth=5, conv_num=2, wf=6, padding=True, batch_norm=True, up_mode='upsample', with_tanh=False, sync_bn=True, antialiasing=True):
        """
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			out_channels (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
		"""
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth - 1
        prev_channels = in_channels
        self.first = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 2 ** wf, kernel_size=7), nn.LeakyReLU(0.2, True)])
        prev_channels = 2 ** wf
        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(prev_channels), nn.LeakyReLU(0.2, True), Downsample(channels=prev_channels, stride=2)]))
            else:
                self.down_sample.append(nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0), nn.BatchNorm2d(prev_channels), nn.LeakyReLU(0.2, True)]))
            self.down_path.append(UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm))
            prev_channels = 2 ** (wf + i + 1)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)
        if with_tanh:
            self.last = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3), nn.Tanh()])
        else:
            self.last = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3)])
        if sync_bn:
            self = DataParallelWithCallback(self)

    def forward(self, x):
        x = self.first(x)
        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.

		-------------------identity----------------------
		|-- downsampling -- |submodule| -- upsampling --|
	"""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
		Parameters:
			outer_nc (int) -- the number of filters in the outer conv layer
			inner_nc (int) -- the number of filters in the inner conv layer
			input_nc (int) -- the number of channels in input images/features
			submodule (UnetSkipConnectionBlock) -- previously defined submodules
			outermost (bool)    -- if this module is the outermost module
			innermost (bool)    -- if this module is the innermost module
			norm_layer          -- normalization layer
			user_dropout (bool) -- if use dropout layers.
		"""
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type='BN', use_dropout=False):
        """Construct a Unet generator
		Parameters:
			input_nc (int)  -- the number of channels in input images
			output_nc (int) -- the number of channels in output images
			num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
								image of size 128x128 will become of size 1x1 # at the bottleneck
			ngf (int)       -- the number of filters in the last conv layer
			norm_layer      -- normalization layer
		We construct the U-Net from the innermost layer to the outermost layer.
		It is a recursive process.
		"""
        super().__init__()
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NameError('Unknown norm layer')
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class Mapping_Model_with_mask(nn.Module):

    def __init__(self, nc, mc=64, n_blocks=3, norm='instance', padding_type='reflect', opt=None):
        super(Mapping_Model_with_mask, self).__init__()
        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4
        for i in range(n_up):
            ic = min(tmp_nc * 2 ** i, mc)
            oc = min(tmp_nc * 2 ** (i + 1), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        self.before_NL = nn.Sequential(*model)
        if opt.NL_res:
            self.NL = networks.NonLocalBlock2D_with_mask_Res(mc, mc, opt.NL_fusion_method, opt.correlation_renormalize, opt.softmax_temperature, opt.use_self, opt.cosin_similarity)
            None
        model = []
        for i in range(n_blocks):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        for i in range(n_up - 1):
            ic = min(64 * 2 ** (4 - i), mc)
            oc = min(64 * 2 ** (3 - i), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2d(tmp_nc, opt.feat_dim, 1, 1)]
        self.after_NL = nn.Sequential(*model)

    def forward(self, input, mask):
        x1 = self.before_NL(input)
        del input
        x2 = self.NL(x1, mask)
        del x1, mask
        x3 = self.after_NL(x2)
        del x2
        return x3


class Mapping_Model_with_mask_2(nn.Module):

    def __init__(self, nc, mc=64, n_blocks=3, norm='instance', padding_type='reflect', opt=None):
        super(Mapping_Model_with_mask_2, self).__init__()
        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4
        for i in range(n_up):
            ic = min(tmp_nc * 2 ** i, mc)
            oc = min(tmp_nc * 2 ** (i + 1), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(2):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        None
        self.before_NL = nn.Sequential(*model)
        if opt.mapping_exp == 1:
            self.NL_scale_1 = networks.Patch_Attention_4(mc, mc, 8)
        model = []
        for i in range(2):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        self.res_block_1 = nn.Sequential(*model)
        if opt.mapping_exp == 1:
            self.NL_scale_2 = networks.Patch_Attention_4(mc, mc, 4)
        model = []
        for i in range(2):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        self.res_block_2 = nn.Sequential(*model)
        if opt.mapping_exp == 1:
            self.NL_scale_3 = networks.Patch_Attention_4(mc, mc, 2)
        model = []
        for i in range(2):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        for i in range(n_up - 1):
            ic = min(64 * 2 ** (4 - i), mc)
            oc = min(64 * 2 ** (3 - i), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2d(tmp_nc, opt.feat_dim, 1, 1)]
        self.after_NL = nn.Sequential(*model)

    def forward(self, input, mask):
        x1 = self.before_NL(input)
        x2 = self.NL_scale_1(x1, mask)
        x3 = self.res_block_1(x2)
        x4 = self.NL_scale_2(x3, mask)
        x5 = self.res_block_2(x4)
        x6 = self.NL_scale_3(x5, mask)
        x7 = self.after_NL(x6)
        return x7

    def inference_forward(self, input, mask):
        x1 = self.before_NL(input)
        del input
        x2 = self.NL_scale_1.inference_forward(x1, mask)
        del x1
        x3 = self.res_block_1(x2)
        del x2
        x4 = self.NL_scale_2.inference_forward(x3, mask)
        del x3
        x5 = self.res_block_2(x4)
        del x4
        x6 = self.NL_scale_3.inference_forward(x5, mask)
        del x5
        x7 = self.after_NL(x6)
        del x6
        return x7


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label, save_dir=''):
        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
        else:
            optimizer.load_state_dict(torch.load(save_path))

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    None
                except:
                    None
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        not_initialized = Set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    None
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass


class Mapping_Model(nn.Module):

    def __init__(self, nc, mc=64, n_blocks=3, norm='instance', padding_type='reflect', opt=None):
        super(Mapping_Model, self).__init__()
        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4
        None
        for i in range(n_up):
            ic = min(tmp_nc * 2 ** i, mc)
            oc = min(tmp_nc * 2 ** (i + 1), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(n_blocks):
            model += [networks.ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt, dilation=opt.mapping_net_dilation)]
        for i in range(n_up - 1):
            ic = min(64 * 2 ** (4 - i), mc)
            oc = min(64 * 2 ** (3 - i), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2d(tmp_nc, opt.feat_dim, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class Pix2PixHDModel_Mapping(BaseModel):

    def name(self):
        return 'Pix2PixHDModel_Mapping'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_smooth_l1, stage_1_feat_l2):
        flags = True, True, use_gan_feat_loss, use_vgg_loss, True, True, use_smooth_l1, stage_1_feat_l2

        def loss_filter(g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2):
            return [l for l, f in zip((g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        self.netG_A = networks.GlobalGenerator_DCDCv2(netG_input_nc, opt.output_nc, opt.ngf, opt.k_size, opt.n_downsample_global, networks.get_norm_layer(norm_type=opt.norm), opt=opt)
        self.netG_B = networks.GlobalGenerator_DCDCv2(netG_input_nc, opt.output_nc, opt.ngf, opt.k_size, opt.n_downsample_global, networks.get_norm_layer(norm_type=opt.norm), opt=opt)
        if opt.non_local == 'Setting_42' or opt.NL_use_mask:
            if opt.mapping_exp == 1:
                self.mapping_net = Mapping_Model_with_mask_2(min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc), opt.map_mc, n_blocks=opt.mapping_n_block, opt=opt)
            else:
                self.mapping_net = Mapping_Model_with_mask(min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc), opt.map_mc, n_blocks=opt.mapping_n_block, opt=opt)
        else:
            self.mapping_net = Mapping_Model(min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc), opt.map_mc, n_blocks=opt.mapping_n_block, opt=opt)
        self.mapping_net.apply(networks.weights_init)
        if opt.load_pretrain != '':
            self.load_network(self.mapping_net, 'mapping_net', opt.which_epoch, opt.load_pretrain)
        if not opt.no_load_VAE:
            self.load_network(self.netG_A, 'G', opt.use_vae_which_epoch, opt.load_pretrainA)
            self.load_network(self.netG_B, 'G', opt.use_vae_which_epoch, opt.load_pretrainB)
            for param in self.netG_A.parameters():
                param.requires_grad = False
            for param in self.netG_B.parameters():
                param.requires_grad = False
            self.netG_A.eval()
            self.netG_B.eval()
        if opt.gpu_ids:
            self.netG_A
            self.netG_B
            self.mapping_net
        if not self.isTrain:
            self.load_network(self.mapping_net, 'mapping_net', opt.which_epoch)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.ngf * 2 if opt.feat_gan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.Smooth_L1, opt.use_two_stage_mapping)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionFeat_feat = torch.nn.L1Loss() if opt.use_l1_feat else torch.nn.MSELoss()
            if self.opt.image_L1:
                self.criterionImage = torch.nn.L1Loss()
            else:
                self.criterionImage = torch.nn.SmoothL1Loss()
            None
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)
            self.loss_names = self.loss_filter('G_Feat_L2', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'Smooth_L1', 'G_Feat_L2_Stage_1')
            if opt.no_TTUR:
                beta1, beta2 = opt.beta1, 0.999
                G_lr, D_lr = opt.lr, opt.lr
            else:
                beta1, beta2 = 0, 0.9
                G_lr, D_lr = opt.lr / 2, opt.lr * 2
            if not opt.no_load_VAE:
                params = list(self.mapping_net.parameters())
                self.optimizer_mapping = torch.optim.Adam(params, lr=G_lr, betas=(beta1, beta2))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=D_lr, betas=(beta1, beta2))
            None

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        if real_image is not None:
            real_image = Variable(real_image.data)
        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, pair=True, infer=False, last_label=None, last_image=None):
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        input_concat = input_label
        label_feat = self.netG_A.forward(input_concat, flow='enc')
        if self.opt.NL_use_mask:
            label_feat_map = self.mapping_net(label_feat.detach(), inst)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())
        fake_image = self.netG_B.forward(label_feat_map, flow='dec')
        image_feat = self.netG_B.forward(real_image, flow='enc')
        loss_feat_l2_stage_1 = 0
        loss_feat_l2 = self.criterionFeat_feat(label_feat_map, image_feat.data) * self.opt.l2_feat
        if self.opt.feat_gan:
            pred_fake_pool = self.discriminate(label_feat.detach(), label_feat_map, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            pred_real = self.discriminate(label_feat.detach(), image_feat)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD.forward(torch.cat((label_feat.detach(), label_feat_map), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            if pair:
                pred_real = self.discriminate(input_label, real_image)
            else:
                pred_real = self.discriminate(last_label, last_image)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss and pair:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    tmp = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    loss_G_GAN_Feat += D_weights * feat_weights * tmp
        else:
            loss_G_GAN_Feat = torch.zeros(1)
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat if pair else torch.zeros(1)
        smooth_l1_loss = 0
        if self.opt.Smooth_L1:
            smooth_l1_loss = self.criterionImage(fake_image, real_image) * self.opt.L1_weight
        return [self.loss_filter(loss_feat_l2, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, smooth_l1_loss, loss_feat_l2_stage_1), None if not infer else fake_image]

    def inference(self, label, inst):
        use_gpu = len(self.opt.gpu_ids) > 0
        if use_gpu:
            input_concat = label.data
            inst_data = inst
        else:
            input_concat = label.data
            inst_data = inst
        label_feat = self.netG_A.forward(input_concat, flow='enc')
        if self.opt.NL_use_mask:
            if self.opt.inference_optimize:
                label_feat_map = self.mapping_net.inference_forward(label_feat.detach(), inst_data)
            else:
                label_feat_map = self.mapping_net(label_feat.detach(), inst_data)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())
        fake_image = self.netG_B.forward(label_feat_map, flow='dec')
        return fake_image


class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True, True

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_kl, d_real, d_fake, g_featd, featd_real, featd_fake):
            return [l for l, f in zip((g_gan, g_gan_feat, g_vgg, g_kl, d_real, d_fake, g_featd, featd_real, featd_fake), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.k_size, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, opt=opt)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc if opt.no_cgan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.feat_D = networks.define_D(64, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid, 1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            None
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            None
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.feat_D, 'feat_D', opt.which_epoch, pretrained_path)
                None
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_KL', 'D_real', 'D_fake', 'G_featD', 'featD_real', 'featD_fake')
            params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.feat_D.parameters())
            self.optimizer_featD = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            None
            if opt.continue_train:
                self.load_optimizer(self.optimizer_D, 'D', opt.which_epoch)
                self.load_optimizer(self.optimizer_G, 'G', opt.which_epoch)
                self.load_optimizer(self.optimizer_featD, 'featD', opt.which_epoch)
                for param_groups in self.optimizer_D.param_groups:
                    self.old_lr = param_groups['lr']
                None
                None

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        if real_image is not None:
            real_image = Variable(real_image.data)
        if self.use_features:
            if self.opt.load_features:
                feat_map = Variable(feat_map.data)
            if self.opt.label_feat:
                inst_map = label_map
        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def feat_discriminate(self, input):
        return self.feat_D.forward(input.detach())

    def forward(self, label, inst, image, feat, infer=False):
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        hiddens = self.netG.forward(input_concat, 'enc')
        noise = Variable(torch.randn(hiddens.size()))
        fake_image = self.netG.forward(hiddens + noise, 'dec')
        real_old_feat = []
        syn_feat = []
        for index, x in enumerate(inst):
            if x == 1:
                real_old_feat.append(hiddens[index].unsqueeze(0))
            else:
                syn_feat.append(hiddens[index].unsqueeze(0))
        L = min(len(real_old_feat), len(syn_feat))
        real_old_feat = real_old_feat[:L]
        syn_feat = syn_feat[:L]
        real_old_feat = torch.cat(real_old_feat, 0)
        syn_feat = torch.cat(syn_feat, 0)
        pred_fake_feat = self.feat_discriminate(real_old_feat)
        loss_featD_fake = self.criterionGAN(pred_fake_feat, False)
        pred_real_feat = self.feat_discriminate(syn_feat)
        loss_featD_real = self.criterionGAN(pred_real_feat, True)
        pred_fake_feat_G = self.feat_D.forward(real_old_feat)
        loss_G_featD = self.criterionGAN(pred_fake_feat_G, True)
        if self.opt.no_cgan:
            pred_fake_pool = self.discriminate(None, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            pred_real = self.discriminate(None, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD.forward(fake_image)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_kl = torch.mean(torch.pow(hiddens, 2)) * self.opt.kl
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_kl, loss_D_real, loss_D_fake, loss_G_featD, loss_featD_real, loss_featD_fake), None if not infer else fake_image]

    def inference(self, label, inst, image=None, feat=None):
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)
        if self.use_features:
            if self.opt.use_encoded_image:
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.feat_D, 'featD', which_epoch, self.gpu_ids)
        self.save_optimizer(self.optimizer_G, 'G', which_epoch)
        self.save_optimizer(self.optimizer_D, 'D', which_epoch)
        self.save_optimizer(self.optimizer_featD, 'featD', which_epoch)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_featD.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            None
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):

    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


class GlobalGenerator_DCDCv2(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, k_size=3, n_downsampling=8, norm_layer=nn.BatchNorm2d, padding_type='reflect', opt=None):
        super(GlobalGenerator_DCDCv2, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, min(ngf, opt.mc), kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(opt.start_r):
            mult = 2 ** i
            model += [nn.Conv2d(min(ngf * mult, opt.mc), min(ngf * mult * 2, opt.mc), kernel_size=k_size, stride=2, padding=1), norm_layer(min(ngf * mult * 2, opt.mc)), activation]
        for i in range(opt.start_r, n_downsampling - 1):
            mult = 2 ** i
            model += [nn.Conv2d(min(ngf * mult, opt.mc), min(ngf * mult * 2, opt.mc), kernel_size=k_size, stride=2, padding=1), norm_layer(min(ngf * mult * 2, opt.mc)), activation]
            model += [ResnetBlock(min(ngf * mult * 2, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
            model += [ResnetBlock(min(ngf * mult * 2, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
        mult = 2 ** (n_downsampling - 1)
        if opt.spatio_size == 32:
            model += [nn.Conv2d(min(ngf * mult, opt.mc), min(ngf * mult * 2, opt.mc), kernel_size=k_size, stride=2, padding=1), norm_layer(min(ngf * mult * 2, opt.mc)), activation]
        if opt.spatio_size == 64:
            model += [ResnetBlock(min(ngf * mult * 2, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
        model += [ResnetBlock(min(ngf * mult * 2, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
        if opt.feat_dim > 0:
            model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), opt.feat_dim, 1, 1)]
        self.encoder = nn.Sequential(*model)
        model = []
        if opt.feat_dim > 0:
            model += [nn.Conv2d(opt.feat_dim, min(ngf * mult * 2, opt.mc), 1, 1)]
        o_pad = 0 if k_size == 4 else 1
        mult = 2 ** n_downsampling
        model += [ResnetBlock(min(ngf * mult, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
        if opt.spatio_size == 32:
            model += [nn.ConvTranspose2d(min(ngf * mult, opt.mc), min(int(ngf * mult / 2), opt.mc), kernel_size=k_size, stride=2, padding=1, output_padding=o_pad), norm_layer(min(int(ngf * mult / 2), opt.mc)), activation]
        if opt.spatio_size == 64:
            model += [ResnetBlock(min(ngf * mult, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
        for i in range(1, n_downsampling - opt.start_r):
            mult = 2 ** (n_downsampling - i)
            model += [ResnetBlock(min(ngf * mult, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
            model += [ResnetBlock(min(ngf * mult, opt.mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer, opt=opt)]
            model += [nn.ConvTranspose2d(min(ngf * mult, opt.mc), min(int(ngf * mult / 2), opt.mc), kernel_size=k_size, stride=2, padding=1, output_padding=o_pad), norm_layer(min(int(ngf * mult / 2), opt.mc)), activation]
        for i in range(n_downsampling - opt.start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, opt.mc), min(int(ngf * mult / 2), opt.mc), kernel_size=k_size, stride=2, padding=1, output_padding=o_pad), norm_layer(min(int(ngf * mult / 2), opt.mc)), activation]
        if opt.use_segmentation_model:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0)]
        else:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*model)

    def forward(self, input, flow='enc_dec'):
        if flow == 'enc':
            return self.encoder(input)
        elif flow == 'dec':
            return self.decoder(input)
        elif flow == 'enc_dec':
            x = self.encoder(input)
            x = self.decoder(x)
            return x


class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'spectral':
        norm_layer = spectral_norm()
    elif norm_type == 'SwitchNorm':
        norm_layer = SwitchNorm2d
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class NonLocalBlock2D_with_mask_Res(nn.Module):

    def __init__(self, in_channels, inter_channels, mode='add', re_norm=False, temperature=1.0, use_self=False, cosin=False):
        super(NonLocalBlock2D_with_mask_Res, self).__init__()
        self.cosin = cosin
        self.renorm = re_norm
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.mode = mode
        self.temperature = temperature
        self.use_self = use_self
        norm_layer = get_norm_layer(norm_type='instance')
        activation = nn.ReLU(True)
        model = []
        for i in range(3):
            model += [ResnetBlock(inter_channels, padding_type='reflect', activation=activation, norm_layer=norm_layer, opt=None)]
        self.res_block = nn.Sequential(*model)

    def forward(self, x, mask):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        if self.cosin:
            theta_x = F.normalize(theta_x, dim=2)
            phi_x = F.normalize(phi_x, dim=1)
        f = torch.matmul(theta_x, phi_x)
        f /= self.temperature
        f_div_C = F.softmax(f, dim=2)
        tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode='bilinear')
        mask[mask > 0] = 1.0
        mask = 1 - mask
        tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        mask *= tmp
        mask_expand = mask.view(batch_size, 1, -1)
        mask_expand = mask_expand.repeat(1, x.size(2) * x.size(3), 1)
        if self.use_self:
            mask_expand[:, range(x.size(2) * x.size(3)), range(x.size(2) * x.size(3))] = 1.0
        f_div_C = mask_expand * f_div_C
        if self.renorm:
            f_div_C = F.normalize(f_div_C, p=1, dim=2)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        W_y = self.res_block(W_y)
        if self.mode == 'combine':
            full_mask = mask.repeat(1, self.inter_channels, 1, 1)
            z = full_mask * x + (1 - full_mask) * W_y
        return z


def SN(module, mode=True):
    if mode:
        return torch.nn.utils.spectral_norm(module)
    return module


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[SN(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), opt.use_SN), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), opt.use_SN), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), opt.use_SN), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[SN(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), opt.use_SN)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, opt, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class Patch_Attention_4(nn.Module):

    def __init__(self, in_channels, inter_channels, patch_size):
        super(Patch_Attention_4, self).__init__()
        self.patch_size = patch_size
        self.F_Combine = nn.Conv2d(in_channels=1025, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        norm_layer = get_norm_layer(norm_type='instance')
        activation = nn.ReLU(True)
        model = []
        for i in range(1):
            model += [ResnetBlock(inter_channels, padding_type='reflect', activation=activation, norm_layer=norm_layer, opt=None)]
        self.res_block = nn.Sequential(*model)

    def Hard_Compose(self, input, dim, index):
        views = [input.size(0)] + [(1 if i != dim else -1) for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, z, mask):
        x = self.res_block(z)
        b, c, h, w = x.shape
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode='bilinear')
        mask[mask > 0] = 1.0
        mask_unfold = F.unfold(mask, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        non_mask_region = (torch.mean(mask_unfold, dim=1, keepdim=True) > 0.6).float()
        all_patch_num = h * w / self.patch_size / self.patch_size
        non_mask_region = non_mask_region.repeat(1, int(all_patch_num), 1)
        x_unfold = F.unfold(x, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        y_unfold = x_unfold.permute(0, 2, 1)
        x_unfold_normalized = F.normalize(x_unfold, dim=1)
        y_unfold_normalized = F.normalize(y_unfold, dim=2)
        correlation_matrix = torch.bmm(y_unfold_normalized, x_unfold_normalized)
        correlation_matrix = correlation_matrix.masked_fill(non_mask_region == 1.0, -1000000000.0)
        correlation_matrix = F.softmax(correlation_matrix, dim=2)
        R, max_arg = torch.max(correlation_matrix, dim=2)
        composed_unfold = self.Hard_Compose(x_unfold, 2, max_arg)
        composed_fold = F.fold(composed_unfold, output_size=(h, w), kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        concat_1 = torch.cat((z, composed_fold, mask), dim=1)
        concat_1 = self.F_Combine(concat_1)
        return concat_1

    def inference_forward(self, z, mask):
        x = self.res_block(z)
        b, c, h, w = x.shape
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode='bilinear')
        mask[mask > 0] = 1.0
        mask_unfold = F.unfold(mask, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        non_mask_region = (torch.mean(mask_unfold, dim=1, keepdim=True) > 0.6).float()[0, 0, :]
        all_patch_num = h * w / self.patch_size / self.patch_size
        mask_index = torch.nonzero(non_mask_region, as_tuple=True)[0]
        if len(mask_index) == 0:
            composed_fold = x
        else:
            unmask_index = torch.nonzero(non_mask_region != 1, as_tuple=True)[0]
            x_unfold = F.unfold(x, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
            Query_Patch = torch.index_select(x_unfold, 2, mask_index)
            Key_Patch = torch.index_select(x_unfold, 2, unmask_index)
            Query_Patch = Query_Patch.permute(0, 2, 1)
            Query_Patch_normalized = F.normalize(Query_Patch, dim=2)
            Key_Patch_normalized = F.normalize(Key_Patch, dim=1)
            correlation_matrix = torch.bmm(Query_Patch_normalized, Key_Patch_normalized)
            correlation_matrix = F.softmax(correlation_matrix, dim=2)
            R, max_arg = torch.max(correlation_matrix, dim=2)
            composed_unfold = self.Hard_Compose(Key_Patch, 2, max_arg)
            x_unfold[:, :, mask_index] = composed_unfold
            composed_fold = F.fold(x_unfold, output_size=(h, w), kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        concat_1 = torch.cat((z, composed_fold, mask), dim=1)
        concat_1 = self.F_Combine(concat_1)
        return concat_1


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
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGG19_torch(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19_torch, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss_torch(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss_torch, self).__init__()
        self.vgg = VGG19_torch()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (Encoder,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4, 'opt': _mock_config(use_SN=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4, 'opt': _mock_config(use_SN=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNetConvBlock,
     lambda: ([], {'conv_num': 4, 'in_size': 4, 'out_size': 4, 'padding': 4, 'batch_norm': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGG19_torch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGGLoss_torch,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_microsoft_Bringing_Old_Photos_Back_to_Life(_paritybench_base):
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

