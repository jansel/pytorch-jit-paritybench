import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
color_dataset = _module
image_folder = _module
single_dataset = _module
make_ilsvrc_dataset = _module
models = _module
base_model = _module
networks = _module
pix2pix_model = _module
options = _module
base_options = _module
train_options = _module
test = _module
test_sweep = _module
train = _module
util = _module
get_data = _module
html = _module
image_pool = _module
visualizer = _module

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


from collections import OrderedDict


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


import numpy as np


class HuberLoss(nn.Module):

    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0 - in1)
        eucl = 0.5 * mann ** 2
        mask[...] = mann < self.delta
        loss = eucl * mask / self.delta + (mann - 0.5 * self.delta) * (1 - mask
            )
        return torch.sum(loss, dim=1, keepdim=True)


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum(torch.abs(in0 - in1), dim=1, keepdim=True)


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum((in0 - in1) ** 2, dim=1, keepdim=True)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class SIGGRAPHGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d,
        use_tanh=True, classification=True):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True
        model1 = [nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=
            1, bias=use_bias)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model1 += [nn.ReLU(True)]
        model1 += [norm_layer(64)]
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model2 += [nn.ReLU(True)]
        model2 += [norm_layer(128)]
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model3 += [nn.ReLU(True)]
        model3 += [norm_layer(256)]
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model4 += [nn.ReLU(True)]
        model4 += [norm_layer(512)]
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model5 += [nn.ReLU(True)]
        model5 += [norm_layer(512)]
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1,
            padding=2, bias=use_bias)]
        model6 += [nn.ReLU(True)]
        model6 += [norm_layer(512)]
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model7 += [nn.ReLU(True)]
        model7 += [norm_layer(512)]
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
            padding=1, bias=use_bias)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1, bias=use_bias)]
        model8 = [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model8 += [nn.ReLU(True)]
        model8 += [norm_layer(256)]
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
            padding=1, bias=use_bias)]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1,
            padding=1, bias=use_bias)]
        model9 = [nn.ReLU(True)]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
            bias=use_bias)]
        model9 += [nn.ReLU(True)]
        model9 += [norm_layer(128)]
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2,
            padding=1, bias=use_bias)]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1,
            padding=1, bias=use_bias)]
        model10 = [nn.ReLU(True)]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1,
            padding=1, bias=use_bias)]
        model10 += [nn.LeakyReLU(negative_slope=0.2)]
        model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0,
            dilation=1, stride=1, bias=use_bias)]
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1,
            stride=1, bias=use_bias)]
        if use_tanh:
            model_out += [nn.Tanh()]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)
        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode=
            'nearest')])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1)])

    def forward(self, input_A, input_B, mask_B):
        conv1_2 = self.model1(torch.cat((input_A, input_B, mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        if self.classification:
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(
                conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2
                .detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        return out_class, out_reg


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
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


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=
        nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer)
        self.model = unet_block

    def forward(self, input_A, input_B, mask_B):
        return self.model(torch.cat((input_A, input_B, mask_B), dim=1))


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
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


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
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
        return self.net(input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_richzhang_colorization_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_001(self):
        self._check(PixelDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

