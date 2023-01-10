import sys
_module = sys.modules[__name__]
del sys
DAEDataLoader = _module
DAENet = _module
DAENet_InstanceNorm = _module
train_DAE_CelebA = _module
train_InstinsicDAE_CelebA = _module

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


import torch.utils.data as data


import numpy as np


import random


from torchvision import transforms


from torchvision import utils


import string


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import gradcheck


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


import math


class BiasReduceLoss(nn.Module):

    def __init__(self, opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()

    def forward(self, x, y, weight=1):
        w = torch.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x, 0).unsqueeze(0)
        self.loss = w * self.criterion(self.avg, y)
        return self.loss


class TotalVaryLoss(nn.Module):

    def __init__(self, opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt

    def forward(self, x, weight=1):
        w = torch.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss


class SelfSmoothLoss2(nn.Module):

    def __init__(self, opt):
        super(SelfSmoothLoss2, self).__init__()
        self.opt = opt

    def forward(self, x, weight=1):
        w = torch.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.sum(torch.mul(self.x_diff, self.x_diff)) + torch.sum(torch.mul(self.y_diff, self.y_diff))
        self.loss = w * self.loss
        return self.loss


class waspMixer(nn.Module):

    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Linear(nin, nout), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class waspIntrinsicComposer(nn.Module):

    def __init__(self, opt):
        super(waspIntrinsicComposer, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc

    def forward(self, shading, albedo):
        self.shading = shading.repeat(1, self.nc, 1, 1)
        self.img = torch.mul(self.shading, albedo)
        return self.img


class waspWarper(nn.Module):

    def __init__(self, opt):
        super(waspWarper, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize = opt.imgSize

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0, 2, 3, 1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output


class waspGridSpatialIntegral(nn.Module):

    def __init__(self, opt):
        super(waspGridSpatialIntegral, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.FloatTensor(1, 1, 1, self.w).fill_(1)
        self.filtery = torch.FloatTensor(1, 1, self.w, 1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx
            self.filtery

    def forward(self, input_diffgrid):
        fullx = F.conv_transpose2d(input_diffgrid[:, 0, :, :].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:, 1, :, :].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:, :, 0:self.w, 0:self.w], fully[:, :, 0:self.w, 0:self.w]), 1)
        return output_grid


class waspEncoder(nn.Module):

    def __init__(self, opt, ngpu=1, nc=1, ndf=32, ndim=128):
        super(waspEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, False), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, False), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, False), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, False), nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        output = self.main(input).view(-1, self.ndim)
        return output


class waspDecoder(nn.Module):

    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.InstanceNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False), nn.Hardtanh(lb, ub))

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class waspDecoderTanh(nn.Module):

    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.InstanceNorm2d(ngf * 8), nn.Tanh(), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 4), nn.Tanh(), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 2), nn.Tanh(), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf), nn.Tanh(), nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf), nn.Tanh(), nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class DenseBlockEncoder(nn.Module):

    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert n_convs > 0
        self.n_channels = n_channels
        self.n_convs = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(nn.InstanceNorm2d(n_channels), activation(*args), nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False)))

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseBlockDecoder(nn.Module):

    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert n_convs > 0
        self.n_channels = n_channels
        self.n_convs = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(nn.InstanceNorm2d(n_channels), activation(*args), nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False)))

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp = mp
        self.main = nn.Sequential(nn.InstanceNorm2d(n_channels_in), activation(*args), nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False), nn.MaxPool2d(mp))

    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.main = nn.Sequential(nn.InstanceNorm2d(n_channels_in), activation(*args), nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False))

    def forward(self, inputs):
        return self.main(inputs)


class waspDenseEncoder(nn.Module):

    def __init__(self, opt, ngpu=1, nc=1, ndf=32, ndim=128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[]):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, stride=2, padding=1), DenseBlockEncoder(ndf, 6), DenseTransitionBlockEncoder(ndf, ndf * 2, 2, activation=activation, args=args), DenseBlockEncoder(ndf * 2, 12), DenseTransitionBlockEncoder(ndf * 2, ndf * 4, 2, activation=activation, args=args), DenseBlockEncoder(ndf * 4, 24), DenseTransitionBlockEncoder(ndf * 4, ndf * 8, 2, activation=activation, args=args), DenseBlockEncoder(ndf * 8, 16), DenseTransitionBlockEncoder(ndf * 8, ndim, 4, activation=activation, args=args), f_activation(*f_args))

    def forward(self, input):
        output = self.main(input).view(-1, self.ndim)
        return output


class waspDenseDecoder(nn.Module):

    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0, 1]):
        super(waspDenseDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), DenseBlockDecoder(ngf * 8, 16), DenseTransitionBlockDecoder(ngf * 8, ngf * 4), DenseBlockDecoder(ngf * 4, 24), DenseTransitionBlockDecoder(ngf * 4, ngf * 2), DenseBlockDecoder(ngf * 2, 12), DenseTransitionBlockDecoder(ngf * 2, ngf), DenseBlockDecoder(ngf, 6), DenseTransitionBlockDecoder(ngf, ngf), nn.InstanceNorm2d(ngf), activation(*args), nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False), f_activation(*f_args))

    def forward(self, inputs):
        return self.main(inputs)


class Encoders(nn.Module):

    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)

    def forward(self, input):
        self.z = self.encoder(input)
        self.zImg = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


class Encoders_Intrinsic(nn.Module):

    def __init__(self, opt):
        super(Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)

    def forward(self, input):
        self.z = self.encoder(input)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp


class Dense_Encoders(nn.Module):

    def __init__(self, opt):
        super(Dense_Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)

    def forward(self, input):
        self.z = self.encoder(input)
        self.zImg = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


class Dense_Encoders_Intrinsic(nn.Module):

    def __init__(self, opt):
        super(Dense_Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)

    def forward(self, input):
        self.z = self.encoder(input)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp


class DecodersIntegralWarper2(nn.Module):

    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)

    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.diffentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1)) * (5.0 / self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping) - 1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping - basegrid
        self.output = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


class DecodersIntegralWarper2_Intrinsic(nn.Module):

    def __init__(self, opt):
        super(DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)

    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1, self.sdim, 1, 1))
        self.texture = self.decoderT(zT.view(-1, self.tdim, 1, 1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1)) * (5.0 / self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping) - 1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping - basegrid
        self.output = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping


class Dense_DecodersIntegralWarper2(nn.Module):

    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.warper = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)

    def forward(self, zI, zW, basegrid):
        self.img = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.diffentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1)) * (5.0 / self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping) - 1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping - basegrid
        self.output = self.warper(self.img, self.warping)
        return self.img, self.resWarping, self.output, self.warping


class Dense_DecodersIntegralWarper2_Intrinsic(nn.Module):

    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.imagedimension = opt.imgSize
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper = waspWarper(opt)
        self.integrator = waspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)

    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1, self.sdim, 1, 1))
        self.texture = self.decoderT(zT.view(-1, self.tdim, 1, 1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1)) * (5.0 / self.imagedimension)
        self.warping = self.integrator(self.diffentialWarping) - 1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping - basegrid
        self.output = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiasReduceLoss,
     lambda: ([], {'opt': _mock_config(cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecodersIntegralWarper2,
     lambda: ([], {'opt': _mock_config(imgSize=4, ngpu=False, idim=4, wdim=4, nc=4, ngf=4, batchSize=4, cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([64, 2, 4, 4])], {}),
     False),
    (DecodersIntegralWarper2_Intrinsic,
     lambda: ([], {'opt': _mock_config(imgSize=4, ngpu=False, idim=4, sdim=4, tdim=4, wdim=4, ngf=4, nc=4, batchSize=4, cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([64, 2, 4, 4])], {}),
     False),
    (DenseBlockDecoder,
     lambda: ([], {'n_channels': 4, 'n_convs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseBlockEncoder,
     lambda: ([], {'n_channels': 4, 'n_convs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseTransitionBlockDecoder,
     lambda: ([], {'n_channels_in': 4, 'n_channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseTransitionBlockEncoder,
     lambda: ([], {'n_channels_in': 4, 'n_channels_out': 4, 'mp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dense_DecodersIntegralWarper2,
     lambda: ([], {'opt': _mock_config(imgSize=4, ngpu=False, idim=4, wdim=4, nc=4, ngf=4, batchSize=4, cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([64, 2, 4, 4])], {}),
     False),
    (Dense_DecodersIntegralWarper2_Intrinsic,
     lambda: ([], {'opt': _mock_config(imgSize=4, ngpu=False, idim=4, sdim=4, tdim=4, wdim=4, ngf=4, nc=4, batchSize=4, cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([64, 2, 4, 4])], {}),
     False),
    (SelfSmoothLoss2,
     lambda: ([], {'opt': _mock_config(cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TotalVaryLoss,
     lambda: ([], {'opt': _mock_config(cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (waspDecoder,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (waspDecoderTanh,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (waspDenseDecoder,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (waspDenseEncoder,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (waspEncoder,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (waspGridSpatialIntegral,
     lambda: ([], {'opt': _mock_config(imgSize=4, cuda=False)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (waspIntrinsicComposer,
     lambda: ([], {'opt': _mock_config(ngpu=False, nc=4)}),
     lambda: ([torch.rand([4, 16, 4, 4]), torch.rand([4, 64, 4, 4])], {}),
     True),
]

class Test_zhixinshu_DeformingAutoencoders_pytorch(_paritybench_base):
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

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

