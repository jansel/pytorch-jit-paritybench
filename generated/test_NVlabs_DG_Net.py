import sys
_module = sys.modules[__name__]
del sys
data = _module
networks = _module
random_erasing = _module
reIDfolder = _module
reIDmodel = _module
evaluate_gpu = _module
test_2label = _module
train = _module
trainer = _module
utils = _module
show1by1 = _module
show_rainbow = _module
show_smooth = _module
show_smooth_structure = _module
show_swap = _module
test_folder = _module

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


from torch import nn


from torch.autograd import Variable


import torch


import torch.nn.functional as F


from torchvision import models


import torch.nn as nn


from torch.nn import init


import torch.optim as optim


from torch.optim import lr_scheduler


import numpy as np


import torchvision


from torchvision import datasets


from torchvision import transforms


import time


import scipy.io


import torch.backends.cudnn as cudnn


import numpy.random as random


import copy


import random


from torch.utils.data import DataLoader


import math


import torchvision.utils as vutils


import torch.nn.init as init


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class MsImageDis(nn.Module):

    def __init__(self, input_dim, params, fp16):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.LAMBDA = params['LAMBDA']
        self.non_local = params['non_local']
        self.n_res = params['n_res']
        self.input_dim = input_dim
        self.fp16 = fp16
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        if not self.gan_type == 'wgan':
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                Dis = self._make_net()
                Dis.apply(weights_init('gaussian'))
                self.cnns.append(Dis)
        else:
            self.cnn = self.one_cnn()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 1, 1, 0, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            dim2 = min(dim * 2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        if self.non_local > 1:
            cnn_x += [NonlocalBlock(dim)]
        for i in range(self.n_res):
            cnn_x += [ResBlock(dim, norm=self.norm, activation=self.activ, pad_type=self.pad_type, res_type='basic')]
        if self.non_local > 0:
            cnn_x += [NonlocalBlock(dim)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def one_cnn(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(5):
            dim2 = min(dim * 2, 512)
            cnn_x += [Conv2dBlock(dim, dim2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        cnn_x += [nn.Conv2d(dim, 1, (4, 2), 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        if not self.gan_type == 'wgan':
            outputs = []
            for model in self.cnns:
                outputs.append(model(x))
                x = self.downsample(x)
        else:
            outputs = self.cnn(x)
            outputs = torch.squeeze(outputs)
        return outputs

    def calc_dis_loss(self, model, input_fake, input_real):
        input_real.requires_grad_()
        outs0 = model.forward(input_fake)
        outs1 = model.forward(input_real)
        loss = 0
        reg = 0
        Drift = 0.001
        LAMBDA = self.LAMBDA
        if self.gan_type == 'wgan':
            loss += torch.mean(outs0) - torch.mean(outs1)
            loss += Drift * (torch.sum(outs0 ** 2) + torch.sum(outs1 ** 2))
            reg += LAMBDA * self.compute_grad2(outs1, input_real).mean()
            loss = loss + reg
            return loss, reg
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
                reg += LAMBDA * self.compute_grad2(out1, input_real).mean()
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) + F.binary_cross_entropy(F.sigmoid(out1), all1))
                reg += LAMBDA * self.compute_grad2(F.sigmoid(out1), input_real).mean()
            else:
                assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)
        loss = loss + reg
        return loss, reg

    def calc_gen_loss(self, model, input_fake):
        outs0 = model.forward(input_fake)
        loss = 0
        Drift = 0.001
        if self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
            loss += Drift * torch.sum(outs0 ** 2)
            return loss
        for it, out0 in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2) * 2
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert grad_dout2.size() == x_in.size()
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg


class AdaINGen(nn.Module):

    def __init__(self, input_dim, params, fp16):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        mlp_norm = params['mlp_norm']
        id_dim = params['id_dim']
        which_dec = params['dec']
        dropout = params['dropout']
        tanh = params['tanh']
        non_local = params['non_local']
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dropout=dropout, tanh=tanh, res_type='basic')
        self.output_dim = self.enc_content.output_dim
        if which_dec == 'basic':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='basic', non_local=non_local, fp16=fp16)
        elif which_dec == 'slim':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='slim', non_local=non_local, fp16=fp16)
        elif which_dec == 'series':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='series', non_local=non_local, fp16=fp16)
        elif which_dec == 'parallel':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='parallel', non_local=non_local, fp16=fp16)
        else:
            """unkonw decoder type"""
        self.mlp_w1 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w2 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w3 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w4 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b1 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b2 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b3 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b4 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.apply(weights_init(params['init']))

    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content, ID):
        ID1 = ID[:, :2048]
        ID2 = ID[:, 2048:4096]
        ID3 = ID[:, 4096:6144]
        ID4 = ID[:, 6144:]
        adain_params_w = torch.cat((self.mlp_w1(ID1), self.mlp_w2(ID2), self.mlp_w3(ID3), self.mlp_w4(ID4)), 1)
        adain_params_b = torch.cat((self.mlp_b1(ID1), self.mlp_b2(ID2), self.mlp_b3(ID3), self.mlp_b4(ID4)), 1)
        self.assign_adain_params(adain_params_w, adain_params_b, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params_b[:, :dim].contiguous()
                std = adain_params_w[:, :dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1) > dim:
                    adain_params_b = adain_params_b[:, dim:]
                    adain_params_w = adain_params_w[:, dim:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                num_adain_params += m.num_features
        return num_adain_params


class VAEGen(nn.Module):

    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dropout=0, tanh=False, res_type='basic')
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


class StyleEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, dropout, tanh=False, res_type='basic'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        for i in range(n_downsample - 1):
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type, res_type=res_type)]
        self.model += [ASPP(dim, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        if tanh:
            self.model += [nn.Tanh()]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder_ImageNet(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder_ImageNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Decoder(nn.Module):

    def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='adain', activ='relu', pad_type='zero', res_type='basic', non_local=False, fp16=False):
        super(Decoder, self).__init__()
        self.input_dim = dim
        self.model = []
        self.model += [nn.Dropout(p=dropout)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, res_type=res_type)]
        if non_local > 0:
            self.model += [NonlocalBlock(dim)]
            None
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2), Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type, fp16=fp16)]
            dim //= 2
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output


class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Deconv(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Deconv, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(input_dim, output_dim, kernel_size=(2, 2), stride=2)]
        model += [nn.InstanceNorm2d(output_dim)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(output_dim, output_dim, kernel_size=(1, 1), stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):

    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()
        model = []
        if res_type == 'basic' or res_type == 'nonlocal':
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'slim':
            dim_half = dim // 2
            model += [Conv2dBlock(dim, dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        elif res_type == 'series':
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'parallel':
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            """unkown block type"""
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type == 'nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out


class NonlocalBlock(nn.Module):

    def __init__(self, in_dim, norm='in'):
        super(NonlocalBlock, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class ASPP(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ASPP, self).__init__()
        dim_part = dim // 2
        self.conv1 = Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)
        self.conv6 = []
        self.conv6 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv6 += [Conv2dBlock(dim_part, dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3)]
        self.conv6 = nn.Sequential(*self.conv6)
        self.conv12 = []
        self.conv12 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv12 += [Conv2dBlock(dim_part, dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6)]
        self.conv12 = nn.Sequential(*self.conv12)
        self.conv18 = []
        self.conv18 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv18 += [Conv2dBlock(dim_part, dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9)]
        self.conv18 = nn.Sequential(*self.conv18)
        self.fuse = Conv2dBlock(4 * dim_part, 2 * dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1, conv6, conv12, conv18), dim=1)
        out = self.fuse(out)
        return out


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16=fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Series2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Series2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) + x
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Parallel2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Parallel2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x)) + self.norm(x)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0), out.size(2))
        if self.activation:
            out = self.activation(out)
        return out


class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h
        return relu5_3


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor':
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):

    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ft_net(nn.Module):

    def __init__(self, class_num, norm=False, pool='avg', stride=2):
        super(ft_net, self).__init__()
        if norm:
            self.norm = True
        else:
            self.norm = False
        model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool == 'max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = 1, 1
            model_ft.layer4[0].conv2.stride = 1, 1
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        f = f.view(f.size(0), f.size(1) * self.part)
        if self.norm:
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True) + 1e-08
            f = f.div(fnorm.expand_as(f))
        x = self.classifier(x)
        return f, x


class ft_netAB(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg'):
        super(ft_netAB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool == 'max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        if stride == 1:
            self.model.layer4[0].downsample[0].stride = 1, 1
            self.model.layer4[0].conv2.stride = 1, 1
        self.classifier1 = ClassBlock(2048, class_num, 0.5)
        self.classifier2 = ClassBlock(2048, class_num, 0.75)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        f = f.view(f.size(0), f.size(1) * self.part)
        f = f.detach()
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x = []
        x.append(x1)
        x.append(x2)
        return f, x


class ft_net_dense(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class ft_net_middle(nn.Module):

    def __init__(self, class_num):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.part = 4
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1
        self.softmax = nn.Softmax(dim=1)
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        f = x
        f = f.view(f.size(0), f.size(1) * self.part)
        x = self.dropout(x)
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = x[:, :, (i)].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return f, y


class PCB_test(nn.Module):

    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.model.layer3[0].downsample[0].stride = 1, 1
        self.model.layer3[0].conv2.stride = 1, 1
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        random.seed(7)

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[(0), x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[(1), x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[(2), x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[(0), x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img.detach()
        return img.detach()


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and key in f and '.pt' in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'], gamma=hyperparameters['gamma'], last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'multistep':
        step = hyperparameters['step_size']
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[step, step + step // 2, step + step // 2 + step // 4], gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def load_config(name):
    config_path = os.path.join('./models', name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    return config


def load_network(network, name):
    save_path = os.path.join('./models', name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for src, dst in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def fliplr(img):
    """flip horizontal"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


parser = argparse.ArgumentParser()


opt = parser.parse_args()


def predict_label(teacher_models, inputs, num_class, alabel, slabel, teacher_style=0):
    if teacher_style == 0:
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count == 0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha
                outputs_t += outputs_t1 + outputs_t2
            count += 2
    elif teacher_style == 1:
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count == 0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha
                outputs_t += outputs_t1 + outputs_t2
            count += 2
        _, dlabel = torch.max(outputs_t.data, 1)
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, dlabel[i]] = 1
    elif teacher_style == 2:
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, alabel[i]] = 1
    elif teacher_style == 3:
        outputs_t = torch.ones(inputs.size(0), num_class).cuda()
    elif teacher_style == 4:
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count == 0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha
                outputs_t += outputs_t1 + outputs_t2
            count += 2
        mask = torch.zeros(outputs_t.shape)
        mask = mask.cuda()
        for i in range(inputs.size(0)):
            mask[i, alabel[i]] = 1
            mask[i, slabel[i]] = 1
        outputs_t = outputs_t * mask
    else:
        print('not valid style. teacher-style is in [0-3].')
    s = torch.sum(outputs_t, dim=1, keepdim=True)
    s = s.expand_as(outputs_t)
    outputs_t = outputs_t / s
    return outputs_t


def scale2(x):
    if x.size(2) > 128:
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')
    return x


def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp


def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[(i), :, :, :])
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY)
        xx = cv2.Canny(xx, 10, 200)
        xx = xx / 255.0 - 0.5
        xx += np.random.randn(xx.shape[0], xx.shape[1]) * 0.1
        xx = torch.from_numpy(xx.astype(np.float32))
        out[(i), :, :] = xx
    out = out.unsqueeze(1)
    return out.cuda()


def to_gray(half=False):

    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward


def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    r, g, b = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)
    batch = (batch + 1) * 255 * 0.5
    mean = tensortype(batch.data.size())
    mean[:, (0), :, :] = 103.939
    mean[:, (1), :, :] = 116.779
    mean[:, (2), :, :] = 123.68
    batch = batch.sub(Variable(mean))
    return batch


class DGNet_Trainer(nn.Module):

    def __init__(self, hyperparameters, gpu_ids=[0]):
        super(DGNet_Trainer, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        ID_class = hyperparameters['ID_class']
        if not 'apex' in hyperparameters.keys():
            hyperparameters['apex'] = False
        self.fp16 = hyperparameters['apex']
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], fp16=False)
        self.gen_b = self.gen_a
        if not 'ID_stride' in hyperparameters.keys():
            hyperparameters['ID_stride'] = 2
        if hyperparameters['ID_style'] == 'PCB':
            self.id_a = PCB(ID_class)
        elif hyperparameters['ID_style'] == 'AB':
            self.id_a = ft_netAB(ID_class, stride=hyperparameters['ID_stride'], norm=hyperparameters['norm_id'], pool=hyperparameters['pool'])
        else:
            self.id_a = ft_net(ID_class, norm=hyperparameters['norm_id'], pool=hyperparameters['pool'])
        self.id_b = self.id_a
        self.dis_a = MsImageDis(3, hyperparameters['dis'], fp16=False)
        self.dis_b = self.dis_a
        if hyperparameters['teacher'] != '':
            teacher_name = hyperparameters['teacher']
            None
            teacher_names = teacher_name.split(',')
            teacher_model = nn.ModuleList()
            teacher_count = 0
            for teacher_name in teacher_names:
                config_tmp = load_config(teacher_name)
                if 'stride' in config_tmp:
                    stride = config_tmp['stride']
                else:
                    stride = 2
                model_tmp = ft_net(ID_class, stride=stride)
                teacher_model_tmp = load_network(model_tmp, teacher_name)
                teacher_model_tmp.model.fc = nn.Sequential()
                teacher_model_tmp = teacher_model_tmp
                if self.fp16:
                    teacher_model_tmp = amp.initialize(teacher_model_tmp, opt_level='O1')
                teacher_model.append(teacher_model_tmp.eval())
                teacher_count += 1
            self.teacher_model = teacher_model
            if hyperparameters['train_bn']:
                self.teacher_model = self.teacher_model.apply(train_bn)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        if hyperparameters['single'] == 'edge':
            self.single = to_edge
        else:
            self.single = to_gray(False)
        if not 'erasing_p' in hyperparameters.keys():
            self.erasing_p = 0
        else:
            self.erasing_p = hyperparameters['erasing_p']
        self.single_re = RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])
        if not 'T_w' in hyperparameters.keys():
            hyperparameters['T_w'] = 1
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters())
        gen_params = list(self.gen_a.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        if hyperparameters['ID_style'] == 'PCB':
            ignored_params = list(map(id, self.id_a.classifier0.parameters())) + list(map(id, self.id_a.classifier1.parameters())) + list(map(id, self.id_a.classifier2.parameters())) + list(map(id, self.id_a.classifier3.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([{'params': base_params, 'lr': lr2}, {'params': self.id_a.classifier0.parameters(), 'lr': lr2 * 10}, {'params': self.id_a.classifier1.parameters(), 'lr': lr2 * 10}, {'params': self.id_a.classifier2.parameters(), 'lr': lr2 * 10}, {'params': self.id_a.classifier3.parameters(), 'lr': lr2 * 10}], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        elif hyperparameters['ID_style'] == 'AB':
            ignored_params = list(map(id, self.id_a.classifier1.parameters())) + list(map(id, self.id_a.classifier2.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([{'params': base_params, 'lr': lr2}, {'params': self.id_a.classifier1.parameters(), 'lr': lr2 * 10}, {'params': self.id_a.classifier2.parameters(), 'lr': lr2 * 10}], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, self.id_a.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([{'params': base_params, 'lr': lr2}, {'params': self.id_a.classifier.parameters(), 'lr': lr2 * 10}], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']
        self.id_criterion = nn.CrossEntropyLoss()
        self.criterion_teacher = nn.KLDivLoss(size_average=False)
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        if self.fp16:
            assert torch.backends.cudnn.enabled, 'fp16 mode requires cudnn backend to be enabled.'
            self.gen_a = self.gen_a
            self.dis_a = self.dis_a
            self.id_a = self.id_a
            self.gen_b = self.gen_a
            self.dis_b = self.dis_a
            self.id_b = self.id_a
            self.gen_a, self.gen_opt = amp.initialize(self.gen_a, self.gen_opt, opt_level='O1')
            self.dis_a, self.dis_opt = amp.initialize(self.dis_a, self.dis_opt, opt_level='O1')
            self.id_a, self.id_opt = amp.initialize(self.id_a, self.id_opt, opt_level='O1')

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out
        for i in range(x.size(0)):
            out[(i), :, :, :] = self.single_re(x[(i), :, :, :])
        return out

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:]) + 1e-08))

    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:] ** 2)

    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])

    def forward(self, x_a, x_b, xp_a, xp_b):
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, p_a = self.id_a(scale2(x_a))
        f_b, p_b = self.id_b(scale2(x_b))
        x_ba = self.gen_a.decode(s_b, f_a)
        x_ab = self.gen_b.decode(s_a, f_b)
        x_a_recon = self.gen_a.decode(s_a, f_a)
        x_b_recon = self.gen_b.decode(s_b, f_b)
        fp_a, pp_a = self.id_a(scale2(xp_a))
        fp_b, pp_b = self.id_b(scale2(xp_b))
        x_a_recon_p = self.gen_a.decode(s_a, fp_a)
        x_b_recon_p = self.gen_b.decode(s_b, fp_b)
        if self.erasing_p > 0:
            x_a_re = self.to_re(scale2(x_a.clone()))
            x_b_re = self.to_re(scale2(x_b.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))
            _, p_a = self.id_a(x_a_re)
            _, p_b = self.id_b(x_b_re)
            _, pp_a = self.id_a(xp_a_re)
            _, pp_b = self.id_b(xp_b_re)
        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p

    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b, xp_a, xp_b, l_a, l_b, hyperparameters, iteration, num_gpu):
        self.gen_opt.zero_grad()
        self.id_opt.zero_grad()
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)
        rand_num = random.uniform(0, 1)
        if hyperparameters['use_encoder_again'] >= rand_num:
            s_a_recon = self.gen_b.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen_a.enc_content(self.single(x_ba_copy))
        else:
            self.enc_content_copy = copy.deepcopy(self.gen_a.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))
        self.id_a_copy = copy.deepcopy(self.id_a)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        f_a_recon, p_a_recon = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon = self.id_b_copy(scale2(x_ab))
        log_sm = nn.LogSoftmax(dim=1)
        if hyperparameters['teacher_w'] > 0 and hyperparameters['teacher'] != '':
            if hyperparameters['ID_style'] == 'normal':
                _, p_a_student = self.id_a(scale2(x_ba_copy))
                p_a_student = log_sm(p_a_student)
                p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy), num_class=hyperparameters['ID_class'], alabel=l_a, slabel=l_b, teacher_style=hyperparameters['teacher_style'])
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)
                _, p_b_student = self.id_b(scale2(x_ab_copy))
                p_b_student = log_sm(p_b_student)
                p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy), num_class=hyperparameters['ID_class'], alabel=l_b, slabel=l_a, teacher_style=hyperparameters['teacher_style'])
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)
            elif hyperparameters['ID_style'] == 'AB':
                _, p_ba_student = self.id_a(scale2(x_ba_copy))
                p_a_student = log_sm(p_ba_student[0])
                with torch.no_grad():
                    p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy), num_class=hyperparameters['ID_class'], alabel=l_a, slabel=l_b, teacher_style=hyperparameters['teacher_style'])
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)
                _, p_ab_student = self.id_b(scale2(x_ab_copy))
                p_b_student = log_sm(p_ab_student[0])
                with torch.no_grad():
                    p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy), num_class=hyperparameters['ID_class'], alabel=l_b, slabel=l_a, teacher_style=hyperparameters['teacher_style'])
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)
                loss_B = self.id_criterion(p_ba_student[1], l_b) + self.id_criterion(p_ab_student[1], l_a)
                self.loss_teacher = hyperparameters['T_w'] * self.loss_teacher + hyperparameters['B_w'] * loss_B
        else:
            self.loss_teacher = 0.0
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0
        x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        if hyperparameters['ID_style'] == 'PCB':
            self.loss_id = self.PCB_loss(p_a, l_a) + self.PCB_loss(p_b, l_b)
            self.loss_pid = self.PCB_loss(pp_a, l_a) + self.PCB_loss(pp_b, l_b)
            self.loss_gen_recon_id = self.PCB_loss(p_a_recon, l_a) + self.PCB_loss(p_b_recon, l_b)
        elif hyperparameters['ID_style'] == 'AB':
            weight_B = hyperparameters['teacher_w'] * hyperparameters['B_w']
            self.loss_id = self.id_criterion(p_a[0], l_a) + self.id_criterion(p_b[0], l_b) + weight_B * (self.id_criterion(p_a[1], l_a) + self.id_criterion(p_b[1], l_b))
            self.loss_pid = self.id_criterion(pp_a[0], l_a) + self.id_criterion(pp_b[0], l_b)
            self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) + self.id_criterion(p_b_recon[0], l_b)
        else:
            self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b)
            self.loss_pid = self.id_criterion(pp_a, l_a) + self.id_criterion(pp_b, l_b)
            self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        if num_gpu > 1:
            self.loss_gen_adv_a = self.dis_a.module.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.module.calc_gen_loss(self.dis_b, x_ab)
        else:
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(self.dis_b, x_ab)
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])
        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + hyperparameters['gan_w'] * self.loss_gen_adv_b + hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_b + hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + hyperparameters['id_w'] * self.loss_id + hyperparameters['pid_w'] * self.loss_pid + hyperparameters['recon_id_w'] * self.loss_gen_recon_id + hyperparameters['vgg_w'] * self.loss_gen_vgg_a + hyperparameters['vgg_w'] * self.loss_gen_vgg_b + hyperparameters['teacher_w'] * self.loss_teacher
        if self.fp16:
            with amp.scale_loss(self.loss_gen_total, [self.gen_opt, self.id_opt]) as scaled_loss:
                scaled_loss.backward()
            self.gen_opt.step()
            self.id_opt.step()
        else:
            self.loss_gen_total.backward()
            self.gen_opt.step()
            self.id_opt.step()
        None

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def PCB_loss(self, inputs, labels):
        loss = 0.0
        for part in inputs:
            loss += self.id_criterion(part, labels)
        return loss / len(inputs)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ab1, x_aba, x_bab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_a.encode(self.single(x_a[i].unsqueeze(0)))
            s_b = self.gen_b.encode(self.single(x_b[i].unsqueeze(0)))
            f_a, _ = self.id_a(scale2(x_a[i].unsqueeze(0)))
            f_b, _ = self.id_b(scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_a.decode(s_a, f_a))
            x_b_recon.append(self.gen_b.decode(s_b, f_b))
            x_ba = self.gen_a.decode(s_b, f_a)
            x_ab = self.gen_b.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ab1.append(x_ab)
            s_b_recon = self.gen_a.enc_content(self.single(x_ba))
            s_a_recon = self.gen_b.enc_content(self.single(x_ab))
            f_a_recon, _ = self.id_a(scale2(x_ba))
            f_b_recon, _ = self.id_b(scale2(x_ab))
            x_aba.append(self.gen_a.decode(s_a_recon, f_a_recon))
            x_bab.append(self.gen_b.decode(s_b_recon, f_b_recon))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1, x_ab1 = torch.cat(x_ba1), torch.cat(x_ab1)
        self.train()
        return x_a, x_a_recon, x_aba, x_ab1, x_b, x_b_recon, x_bab, x_ba1

    def dis_update(self, x_ab, x_ba, x_a, x_b, hyperparameters, num_gpu):
        self.dis_opt.zero_grad()
        if num_gpu > 1:
            self.loss_dis_a, reg_a = self.dis_a.module.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.module.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        else:
            self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        None
        if self.fp16:
            with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        last_model_name = get_model_list(checkpoint_dir, 'gen')
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b = self.gen_a
        iterations = int(last_model_name[-11:-3])
        last_model_name = get_model_list(checkpoint_dir, 'dis')
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b = self.dis_a
        last_model_name = get_model_list(checkpoint_dir, 'id')
        state_dict = torch.load(last_model_name)
        self.id_a.load_state_dict(state_dict['a'])
        self.id_b = self.id_a
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.id_opt.load_state_dict(state_dict['id'])
        except:
            pass
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        None
        return iterations

    def save(self, snapshot_dir, iterations, num_gpu=1):
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict()}, gen_name)
        if num_gpu > 1:
            torch.save({'a': self.dis_a.module.state_dict()}, dis_name)
        else:
            torch.save({'a': self.dis_a.state_dict()}, dis_name)
        torch.save({'a': self.id_a.state_dict()}, id_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'id': self.id_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ClassBlock,
     lambda: ([], {'input_dim': 4, 'class_num': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ContentEncoder_ImageNet,
     lambda: ([], {'n_downsample': 4, 'n_res': 4, 'input_dim': 4, 'dim': 4, 'norm': 4, 'activ': 4, 'pad_type': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Conv2dBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Deconv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'dim': 4, 'n_blk': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (NonlocalBlock,
     lambda: ([], {'in_dim': 64}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (PCB,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResBlocks,
     lambda: ([], {'num_blocks': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ft_net,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ft_netAB,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ft_net_dense,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ft_net_middle,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_NVlabs_DG_Net(_paritybench_base):
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

