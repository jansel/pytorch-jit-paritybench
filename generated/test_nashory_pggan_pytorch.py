import sys
_module = sys.modules[__name__]
del sys
config = _module
custom_layers = _module
dataloader = _module
generate_interpolated = _module
network = _module
tf_recorder = _module
utils = _module

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


import torch.nn.functional as F


import numpy as np


from torch.autograd import Variable


import copy


from torch.nn.init import kaiming_normal


from torch.nn.init import calculate_gain


class ConcatTable(nn.Module):

    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        return y


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class fadein_layer(nn.Module):

    def __init__(self, config):
        super(fadein_layer, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    def forward(self, x):
        return torch.add(x[0].mul(1.0 - self.alpha), x[1].mul(self.alpha))


class minibatch_std_concat_layer(nn.Module):

    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'
                ], 'Invalid averaging mode' % self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x -
            torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-08)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2, 3], keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0, 2, 3], keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] / self.n, self.shape[2],
                self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % self.averaging


class pixelwise_norm_layer(nn.Module):

    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-08

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5


class equalized_conv2d(nn.Module):

    def __init__(self, c_in, c_out, k_size, stride, pad, initializer=
        'kaiming', bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':
            kaiming_normal(self.conv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':
            xavier_normal(self.conv.weight)
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = torch.mean(self.conv.weight.data ** 2) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)


class equalized_deconv2d(nn.Module):

    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'
        ):
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad,
            bias=False)
        if initializer == 'kaiming':
            kaiming_normal(self.deconv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':
            xavier_normal(self.deconv.weight)
        deconv_w = self.deconv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = torch.mean(self.deconv.weight.data ** 2) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data / self.scale)

    def forward(self, x):
        x = self.deconv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)


class equalized_linear(nn.Module):

    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':
            kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        elif initializer == 'xavier':
            torch.nn.init.xavier_normal(self.linear.weight)
        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = torch.mean(self.linear.weight.data ** 2) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1, -1).expand_as(x)


class generalized_drop_out(nn.Module):

    def __init__(self, mode='mul', strength=0.4, axes=(0, 1), normalize=False):
        super(generalized_drop_out, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'
            ], 'Invalid GDropLayer mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x
        rnd_shape = [(s if axis in self.axes else 1) for axis, s in
            enumerate(x.size())]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1
        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (
            self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str


def get_module_names(model):
    names = []
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=
    False, wn=False, pixel=False, only=False):
    if wn:
        layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    else:
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        if pixel:
            layers.append(pixelwise_norm_layer())
    return layers


def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)
            new_module[-1].load_state_dict(m.state_dict())
    return new_module


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_gen()

    def first_block(self):
        layers = []
        ndim = self.ngf
        if self.flag_norm_latent:
            layers.append(pixelwise_norm_layer())
        layers = deconv(layers, self.nz, ndim, 4, 1, 3, self.flag_leaky,
            self.flag_bn, self.flag_wn, self.flag_pixelwise)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.
            flag_bn, self.flag_wn, self.flag_pixelwise)
        return nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl - 1)
            ), int(pow(2, resl - 1)), int(pow(2, resl)), int(pow(2, resl)))
        ndim = self.ngf
        if resl == 3 or resl == 4 or resl == 5:
            halving = False
            ndim = self.ngf
        elif resl == 6 or resl == 7 or resl == 8 or resl == 9 or resl == 10:
            halving = True
            for i in range(int(resl) - 5):
                ndim = ndim / 2
        ndim = int(ndim)
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if halving:
            layers = deconv(layers, ndim * 2, ndim, 3, 1, 1, self.
                flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, self.flag_pixelwise)
        else:
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return nn.Sequential(*layers), ndim, layer_name

    def to_rgb_block(self, c_in):
        layers = []
        layers = deconv(layers, c_in, self.nc, 1, 1, 0, self.flag_leaky,
            self.flag_bn, self.flag_wn, self.flag_pixelwise, only=True)
        if self.flag_tanh:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model

    def grow_network(self, resl):
        new_model = nn.Sequential()
        names = get_module_names(self.model)
        for name, module in self.model.named_children():
            if not name == 'to_rgb_block':
                new_model.add_module(name, module)
                new_model[-1].load_state_dict(module.state_dict())
        if resl >= 3 and resl <= 9:
            None
            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.Upsample(
                scale_factor=2, mode='nearest'))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)
            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))
            new_model.add_module('concat_block', ConcatTable(prev_block,
                next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            None
            high_resl_block = deepcopy_module(self.model.concat_block.
                layer2, 'high_resl_block')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block.
                layer2, 'high_resl_to_rgb')
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        None
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1, 1, 1))
        return x


def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:
        layers.append(equalized_linear(c_in, c_out))
    else:
        layers.append(Linear(c_in, c_out))
    if sig:
        layers.append(nn.Sigmoid())
    return layers


def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False,
    wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:
        layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:
        layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad,
            initializer='kaiming'))
    else:
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        if pixel:
            layers.append(pixelwise_norm_layer())
    return layers


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_sigmoid = config.flag_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def last_block(self):
        ndim = self.ndf
        layers = []
        layers.append(minibatch_std_concat_layer())
        layers = conv(layers, ndim + 1, ndim, 3, 1, 1, self.flag_leaky,
            self.flag_bn, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.flag_leaky, self.
            flag_bn, self.flag_wn, pixel=False)
        layers = linear(layers, ndim, 1, sig=self.flag_sigmoid, wn=self.flag_wn
            )
        return nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl)),
            int(pow(2, resl)), int(pow(2, resl - 1)), int(pow(2, resl - 1)))
        ndim = self.ndf
        if resl == 3 or resl == 4 or resl == 5:
            halving = False
            ndim = self.ndf
        elif resl == 6 or resl == 7 or resl == 8 or resl == 9 or resl == 10:
            halving = True
            for i in range(int(resl) - 5):
                ndim = ndim / 2
        ndim = int(ndim)
        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim * 2, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, pixel=False)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky,
                self.flag_bn, self.flag_wn, pixel=False)
        layers.append(nn.AvgPool2d(kernel_size=2))
        return nn.Sequential(*layers), ndim, layer_name

    def from_rgb_block(self, ndim):
        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.flag_leaky, self
            .flag_bn, self.flag_wn, pixel=False)
        return nn.Sequential(*layers)

    def get_init_dis(self):
        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model

    def grow_network(self, resl):
        if resl >= 3 and resl <= 9:
            None
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_downsample', nn.AvgPool2d(
                kernel_size=2))
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)
            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block
                (ndim))
            next_block.add_module('high_resl_block', inter_block)
            new_model = nn.Sequential()
            new_model.add_module('concat_block', ConcatTable(prev_block,
                next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name == 'from_rgb_block':
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            None
            high_resl_block = deepcopy_module(self.model.concat_block.
                layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.
                layer2, 'high_resl_from_rgb')
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        None
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_nashory_pggan_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(fadein_layer(*[], **{'config': _mock_config()}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(minibatch_std_concat_layer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(pixelwise_norm_layer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(equalized_conv2d(*[], **{'c_in': 4, 'c_out': 4, 'k_size': 4, 'stride': 1, 'pad': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(equalized_deconv2d(*[], **{'c_in': 4, 'c_out': 4, 'k_size': 4, 'stride': 1, 'pad': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_006(self):
        self._check(equalized_linear(*[], **{'c_in': 4, 'c_out': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_007(self):
        self._check(generalized_drop_out(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
