import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
image_crop = _module
models = _module
base_model = _module
networks = _module
pix2pix_model = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
util = _module
get_data = _module
html = _module
image_pool = _module
visualizer = _module
networks = _module
pix2pix_model = _module

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


from torch.nn import init


import functools


from torch.autograd import Variable


from torch.optim import lr_scheduler


import torch.nn.functional as F


import numpy as np


from collections import OrderedDict


from torch import autograd


import copy


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
            self.loss = nn.BCEWithLogitsLoss()

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


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].
                detach())
        return loss


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
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
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1,
            padding=padw), nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.ReLU()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias), nn.
                Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride
                =2, padding=padw, bias=use_bias), nn.ReLU()]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias), nn.ReLU()]
        sequence += [nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw,
            stride=1, padding=padw)]
        sequence += [nn.Conv2d(ndf * nf_mult, 128, kernel_size=kw, stride=2,
            padding=padw)]
        self.model = nn.Sequential(*sequence)
        self.conv1 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1,
            bias=True)
        self.conv2 = nn.Linear(128 * 8 ** 2, 20)
        self.aux_layer = nn.Sequential(nn.Linear(128 * 8 ** 2, 939), nn.
            Softmax())

    def forward(self, input):
        out1 = self.model(input)
        out = self.conv1(out1)
        cond = F.sigmoid(self.conv2(out1.view(out1.shape[0], -1)))
        class_pred = self.aux_layer(out1.view(out1.shape[0], -1))
        return out, cond, class_pred


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1,
            padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2,
            kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(
            ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1,
            kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
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


class Estimate(nn.Module):

    def __init__(self):
        super(Estimate, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding
            =1), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(
            64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, stride=1,
            padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 3, 3, stride=1, padding=1))

    def forward(self, x):
        x = F.tanh(self.features(x))
        return x


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
            self.loss = nn.BCEWithLogitsLoss()

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


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].
                detach())
        return loss


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
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
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1,
            padding=padw), nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.ReLU()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias), nn.
                Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride
                =2, padding=padw, bias=use_bias), nn.ReLU()]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias), nn.ReLU()]
        sequence += [nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw,
            stride=1, padding=padw)]
        sequence += [nn.Conv2d(ndf * nf_mult, 128, kernel_size=kw, stride=2,
            padding=padw)]
        self.model = nn.Sequential(*sequence)
        self.conv1 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1,
            bias=True)
        self.conv2 = nn.Linear(128 * 8 ** 2, 20)
        self.aux_layer = nn.Sequential(nn.Linear(128 * 8 ** 2, 939), nn.
            Softmax())

    def forward(self, input):
        out1 = self.model(input)
        out = self.conv1(out1)
        cond = F.sigmoid(self.conv2(out1.view(out1.shape[0], -1)))
        class_pred = self.aux_layer(out1.view(out1.shape[0], -1))
        return out, cond, class_pred


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1,
            padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2,
            kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(
            ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1,
            kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
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


class Estimate(nn.Module):

    def __init__(self):
        super(Estimate, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding
            =1), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(
            64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, stride=1,
            padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 3, 3, stride=1, padding=1))

    def forward(self, x):
        x = F.tanh(self.features(x))
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Blade6570_icface(_paritybench_base):
    pass
    def test_000(self):
        self._check(GANLoss(*[], **{}), [], {'input': torch.rand([4, 4]), 'target_is_real': 4})

    @_fails_compile()
    def test_001(self):
        self._check(PixelDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Estimate(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

