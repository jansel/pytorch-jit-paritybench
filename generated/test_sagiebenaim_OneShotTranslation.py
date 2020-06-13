import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
image_folder = _module
single_dataset = _module
unaligned_dataset = _module
combine_A_and_B = _module
make_dataset_aligned = _module
models = _module
autoencoder_model = _module
base_model = _module
networks = _module
ost = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
get_data = _module
html = _module
image_pool = _module
visualizer = _module
data_loader = _module
main_autoencoder = _module
main_mnist_to_svhn = _module
main_svhn_to_mnist = _module
model = _module
solver_autoencoder = _module
solver_mnist_to_svhn = _module
solver_svhn_to_mnist = _module

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


from torch.autograd import Variable


import itertools


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


import torch.nn.functional as F


class pixel_norm(nn.Module):

    def forward(self, x, epsilon=1e-08):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) +
            epsilon)


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


class ResnetEncoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect',
        n_downsampling=2, start=0, end=2, input_layer=True, n_blocks=6):
        assert n_blocks >= 0
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        if input_layer:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
                kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf),
                nn.ReLU(True)]
        for i in range(start, end):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetDecoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect',
        n_downsampling=2, start=0, end=2, output_layer=True, n_blocks=6):
        assert n_blocks >= 0
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(start, end):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        if output_layer:
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


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
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
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
        nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
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

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


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
        BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
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
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


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


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias
        =False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G11(nn.Module):

    def __init__(self, conv_dim=64):
        super(G11, self).__init__()
        self.conv1 = conv(1, conv_dim, 4)
        self.conv1_svhn = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        res_dim = conv_dim * 2
        self.conv3 = conv(res_dim, res_dim, 3, 1, 1)
        self.conv4 = conv(res_dim, res_dim, 3, 1, 1)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)
        self.deconv2_svhn = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x, svhn=False):
        if svhn:
            out = F.leaky_relu(self.conv1_svhn(x), 0.05)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        if svhn:
            out = F.tanh(self.deconv2_svhn(out))
        else:
            out = F.tanh(self.deconv2(out))
        return out

    def encode(self, x, svhn=False):
        if svhn:
            out = F.leaky_relu(self.conv1_svhn(x), 0.05)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        return out

    def decode(self, out, svhn=False):
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        if svhn:
            out = F.tanh(self.deconv2_svhn(out))
        else:
            out = F.tanh(self.deconv2(out))
        return out

    def encode_params(self):
        layers_basic = list(self.conv1_svhn.parameters()) + list(self.conv1
            .parameters())
        layers_basic += list(self.conv2.parameters())
        layers_basic += list(self.conv3.parameters())
        return layers_basic

    def decode_params(self):
        layers_basic = list(self.deconv2_svhn.parameters()) + list(self.
            deconv2.parameters())
        layers_basic += list(self.deconv1.parameters())
        layers_basic += list(self.conv4.parameters())
        return layers_basic

    def unshared_parameters(self):
        return list(self.deconv2_svhn.parameters()) + list(self.conv1_svhn.
            parameters()) + list(self.deconv2.parameters()) + list(self.
            conv1.parameters())


class G22(nn.Module):

    def __init__(self, conv_dim=64):
        super(G22, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)
        self.conv1_mnist = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        res_dim = conv_dim * 2
        self.conv3 = conv(res_dim, res_dim, 3, 1, 1)
        self.conv4 = conv(res_dim, res_dim, 3, 1, 1)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)
        self.deconv2_mnist = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x, mnist=False):
        if mnist:
            out = F.leaky_relu(self.conv1_mnist(x), 0.05)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        if mnist:
            out = F.tanh(self.deconv2_mnist(out))
        else:
            out = F.tanh(self.deconv2(out))
        return out

    def encode(self, x, mnist=False):
        if mnist:
            out = F.leaky_relu(self.conv1_mnist(x), 0.05)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        return out

    def decode(self, out, mnist=False):
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        if mnist:
            out = F.tanh(self.deconv2_mnist(out))
        else:
            out = F.tanh(self.deconv2(out))
        return out

    def encode_params(self):
        layers_basic = list(self.conv1_mnist.parameters()) + list(self.
            conv1.parameters())
        layers_basic += list(self.conv2.parameters())
        layers_basic += list(self.conv3.parameters())
        return layers_basic

    def decode_params(self):
        layers_basic = list(self.deconv2_mnist.parameters()) + list(self.
            deconv2.parameters())
        layers_basic += list(self.deconv1.parameters())
        layers_basic += list(self.conv4.parameters())
        return layers_basic

    def unshared_parameters(self):
        return list(self.deconv2_mnist.parameters()) + list(self.
            conv1_mnist.parameters()) + list(self.deconv2.parameters()) + list(
            self.conv1.parameters())


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x_0):
        out = F.leaky_relu(self.conv1(x_0), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out_0 = self.fc(out).squeeze()
        return out_0


class D2(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x_0):
        out = F.leaky_relu(self.conv1(x_0), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out_0 = self.fc(out).squeeze()
        return out_0


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sagiebenaim_OneShotTranslation(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(pixel_norm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(GANLoss(*[], **{}), [], {'input': torch.rand([4, 4]), 'target_is_real': 4})

    @_fails_compile()
    def test_002(self):
        self._check(ResnetDecoder(*[], **{'input_nc': 4, 'output_nc': 4}), [torch.rand([4, 256, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(UnetGenerator(*[], **{'input_nc': 4, 'output_nc': 4, 'num_downs': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(PixelDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(G11(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(G22(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_008(self):
        self._check(D1(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_009(self):
        self._check(D2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

