import sys
_module = sys.modules[__name__]
del sys
iterative_saliency = _module
real_time_saliency = _module
sal = _module
datasets = _module
cifar_dataset = _module
imagenet_dataset = _module
imagenet_synset = _module
saliency_model = _module
small = _module
utils = _module
gaussian_blur = _module
mask = _module
pt_store = _module
pytorch_fixes = _module
pytorch_trainer = _module
resnet_encoder = _module
saliency_eval = _module
saliency_train = _module
small_trainer = _module
test_black_box = _module

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


import torch.optim as torch_optim


import numpy as np


from torch.autograd import Variable


from queue import Queue


import time


from torch.nn.functional import softmax


import matplotlib


import matplotlib.pyplot as plt


from torchvision.transforms import *


from torchvision.datasets import CIFAR10


from torch.utils.data import dataloader


import random


from torchvision.datasets import ImageFolder


from torch.nn import functional as F


from torch.nn import Module


from torchvision.models.resnet import resnet50


from torch.nn.functional import conv2d


from torch.nn import *


import torch.nn.modules.loss as losses


from itertools import chain


import torch.nn


import torch.utils.data as torch_data


from torch.optim.lr_scheduler import StepLR


import torch.utils.model_zoo as model_zoo


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import Bottleneck


import torch.nn.functional as F


from torchvision.models.alexnet import alexnet


from torchvision.models.squeezenet import squeezenet1_1


def BottleneckBlock(in_channels, out_channels, stride=1, layers=1, activation_fn=lambda : torch.nn.ReLU(inplace=False)):
    assert layers > 0 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Bottleneck(current_channels, out_channels, stride=stride if layer == 0 else 1, activation_fn=activation_fn))
        current_channels = out_channels
    return Sequential(*_modules) if len(_modules) > 1 else _modules[0]


class PixelShuffleBlock(Module):

    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def SimpleCNNBlock(in_channels, out_channels, kernel_size=3, layers=1, stride=1, follow_with_bn=True, activation_fn=lambda : ReLU(True), affine=True):
    assert layers > 0 and kernel_size % 2 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer == 0 else 1, padding=kernel_size / 2, bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(BatchNorm2d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return Sequential(*_modules)


def SimpleUpsamplerSubpixel(in_channels, out_channels, kernel_size=3, activation_fn=lambda : torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [SimpleCNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn), PixelShuffleBlock(), activation_fn()]
    return Sequential(*_modules)


class UNetUpsampler(Module):

    def __init__(self, in_channels, out_channels, passthrough_channels, follow_up_residual_blocks=1, upsampler_block=SimpleUpsamplerSubpixel, upsampler_kernel_size=3, activation_fn=lambda : torch.nn.ReLU(inplace=False)):
        super(UNetUpsampler, self).__init__()
        assert follow_up_residual_blocks >= 1, 'You must follow up with residuals when using unet!'
        assert passthrough_channels >= 1, 'You must use passthrough with unet'
        self.upsampler = upsampler_block(in_channels=in_channels, out_channels=out_channels, kernel_size=upsampler_kernel_size, activation_fn=activation_fn)
        self.follow_up = BottleneckBlock(out_channels + passthrough_channels, out_channels, layers=follow_up_residual_blocks, activation_fn=activation_fn)

    def forward(self, inp, passthrough):
        upsampled = self.upsampler(inp)
        upsampled = torch.cat((upsampled, passthrough), 1)
        return self.follow_up(upsampled)


class SaliencyModel(Module):

    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales, upsampler_base, fix_encoder=True, use_simple_activation=False, allow_selector=False, num_classes=1000):
        super(SaliencyModel, self).__init__()
        assert upsampler_scales <= encoder_scales
        self.encoder = encoder
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales
        self.fix_encoder = fix_encoder
        self.use_simple_activation = use_simple_activation
        down = self.encoder_scales
        modulator_sizes = []
        for up in reversed(range(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2 ** (up + 1)
            encoder_chans = encoder_base * 2 ** down
            inc = upsampler_chans if down != encoder_scales else encoder_chans
            modulator_sizes.append(inc)
            self.add_module('up%d' % up, UNetUpsampler(in_channels=inc, passthrough_channels=encoder_chans / 2, out_channels=upsampler_chans / 2, follow_up_residual_blocks=1, activation_fn=lambda : nn.ReLU()))
            down -= 1
        self.to_saliency_chans = nn.Conv2d(upsampler_base, 2, 1)
        self.allow_selector = allow_selector
        if self.allow_selector:
            s = encoder_base * 2 ** encoder_scales
            self.selector_module = nn.Embedding(num_classes, s)
            self.selector_module.weight.data.normal_(0, 1.0 / s ** 0.5)

    def minimialistic_restore(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'
        p = os.path.join(save_dir, 'model-%d.ckpt' % 1)
        if not os.path.exists(p):
            None
            return
        for name, data in list(torch.load(p, map_location=lambda storage, loc: storage).items()):
            self._modules[name].load_state_dict(data)

    def minimalistic_save(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'
        data = {}
        for name, module in list(self._modules.items()):
            if module is self.encoder:
                continue
            data[name] = module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(data, os.path.join(save_dir, 'model-%d.ckpt' % 1))

    def get_trainable_parameters(self):
        all_params = self.parameters()
        if not self.fix_encoder:
            return set(all_params)
        unwanted = self.encoder.parameters()
        return set(all_params) - set(unwanted) - (set(self.selector_module.parameters()) if self.allow_selector else set([]))

    def forward(self, _images, _selectors=None, pt_store=None, model_confidence=0.0):
        out = self.encoder(_images)
        if self.fix_encoder:
            out = [e.detach() for e in out]
        down = self.encoder_scales
        main_flow = out[down]
        if self.allow_selector:
            assert _selectors is not None
            em = torch.squeeze(self.selector_module(_selectors.view(-1, 1)), 1)
            act = torch.sum(main_flow * em.view(-1, 2048, 1, 1), 1, keepdim=True)
            th = torch.sigmoid(act - model_confidence)
            main_flow = main_flow * th
            ex = torch.mean(torch.mean(act, 3), 2)
            exists_logits = torch.cat((-ex / 2.0, ex / 2.0), 1)
        else:
            exists_logits = None
        for up in reversed(range(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d' % up](main_flow, out[down - 1])
            down -= 1
        saliency_chans = self.to_saliency_chans(main_flow)
        if self.use_simple_activation:
            return torch.unsqueeze(torch.sigmoid(saliency_chans[:, 0, :, :] / 2), dim=1), exists_logits, out[-1]
        a = torch.abs(saliency_chans[:, 0, :, :])
        b = torch.abs(saliency_chans[:, 1, :, :])
        return torch.unsqueeze(a / (a + b), dim=1), exists_logits, out[-1]


class discrete_neuron_func(torch.autograd.Function):

    def forward(self, x):
        self.save_for_backward(x)
        x = x.clone()
        x[x > 0] = 1.0
        x[x < 0] = -1.0
        return x

    def backward(self, grad_out):
        x, = self.saved_tensors
        grad_input = grad_out.clone()
        this_grad = torch.exp(-x ** 2 / 2.0) / (3.14 / 2.0) ** 0.5
        return this_grad * grad_input


class DiscreteNeuron(Module):

    def forward(self, x):
        return discrete_neuron_func()(x)


class AssertSize(Module):

    def __init__(self, expected_dim=None):
        super(AssertSize, self).__init__()
        self.expected_dim = expected_dim

    def forward(self, x):
        if self.expected_dim is not None:
            assert self.expected_dim == x.size(2), 'expected %d got %d' % (self.expected_dim, x.size(2))
        return x


class ShapeLog(Module):

    def forward(self, x):
        None
        return x


class GlobalAvgPool(Module):

    def forward(self, x):
        x = F.avg_pool2d(x, x.size(2), stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        return x.view(x.size(0), -1)


class CustomModule(Module):

    def __init__(self, py_func):
        super(CustomModule, self).__init__()
        self.py_func = py_func

    def forward(self, inp):
        return self.py_func(inp)


class MultiModulator(Module):

    def __init__(self, embedding_size, num_classes, modulator_sizes):
        super(MultiModulator, self).__init__()
        self.emb = Embedding(num_classes, embedding_size)
        self.num_modulators = len(modulator_sizes)
        for i, m in enumerate(modulator_sizes):
            self.add_module('m%d' % i, Linear(embedding_size, m))

    def forward(self, selectors):
        """ class selector must be of shape (BS,)  Returns (BS, MODULATOR_SIZE) for each modulator."""
        em = torch.squeeze(self.emb(selectors.view(-1, 1)), 1)
        res = []
        for i in range(self.num_modulators):
            res.append(self._modules['m%d' % i](em))
        return tuple(res)


WARN_TEMPLATE = '\x1b[38;5;1mWARNING: %s\x1b[0m\n'


class EasyModule(Module):

    def save(self, save_dir, step=1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model-%d.ckpt' % step))

    def restore(self, save_dir, step=1):
        p = os.path.join(save_dir, 'model-%d.ckpt' % step)
        if not os.path.exists(p):
            None
            return
        self.load_state_dict(torch.load(p))


class Bottleneck(Module):

    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4, activation_fn=lambda : torch.nn.ReLU(inplace=False)):
        super(Bottleneck, self).__init__()
        bottleneck_channels = out_channels / bottleneck_ratio
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        self.activation_fn = activation_fn()
        if stride != 1 or in_channels != out_channels:
            self.residual_transformer = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.residual_transformer = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.residual_transformer is not None:
            residual = self.residual_transformer(residual)
        out += residual
        out = self.activation_fn(out)
        return out


class B4D_4D(Module):
    """Converts 4d to 4d representation, note increases the resolution by 2!"""

    def __init__(self, in_channels, out_channels, preprocess_layers=2, keep_resolution=False):
        super(B4D_4D, self).__init__()
        assert preprocess_layers > 0
        self.keep_resolution = keep_resolution
        self.preprocess = SimpleCNNBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, layers=preprocess_layers, stride=1, follow_with_bn=True, activation_fn=lambda : nn.ReLU(inplace=False))
        self.trns = SimpleCNNBlock(in_channels=in_channels, out_channels=4 * out_channels if not self.keep_resolution else out_channels, kernel_size=3, layers=1, stride=1, follow_with_bn=False, activation_fn=None)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.trns(x)
        if not self.keep_resolution:
            x = F.pixel_shuffle(x, 2)
        return x


def SimpleLinearBlock(in_channels, out_channels, layers=1, follow_with_bn=True, activation_fn=lambda : ReLU(inplace=False), affine=True):
    assert layers > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Linear(current_channels, out_channels, bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(BatchNorm1d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return Sequential(*_modules)


class B2D_4D(Module):
    """Note: will always return 2x2"""

    def __init__(self, in_channels, out_channels, preprocess_layers=2):
        super(B2D_4D, self).__init__()
        if preprocess_layers > 0:
            self.preprocess = SimpleLinearBlock(in_channels=in_channels, out_channels=in_channels, layers=preprocess_layers, follow_with_bn=True, activation_fn=lambda : nn.ReLU(inplace=False))
        else:
            self.preprocess = None
        self.trns = SimpleLinearBlock(in_channels=in_channels, out_channels=out_channels * 4, layers=1, follow_with_bn=False, activation_fn=None)

    def forward(self, x):
        x = self.preprocess(x) if self.preprocess else x
        x = self.trns(x)
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 2)
        x = F.pixel_shuffle(x, 2)
        return x


class B2D_2D(Module):

    def __init__(self, in_channels, out_channels, preprocess_layers=2):
        super(B2D_2D, self).__init__()
        if preprocess_layers > 0:
            self.preprocess = SimpleLinearBlock(in_channels=in_channels, out_channels=in_channels, layers=preprocess_layers, follow_with_bn=True, activation_fn=lambda : nn.ReLU(inplace=False))
        else:
            self.preprocess = None
        self.trns = SimpleLinearBlock(in_channels=in_channels, out_channels=out_channels, layers=1, follow_with_bn=False, activation_fn=None)

    def forward(self, x):
        x = self.preprocess(x) if self.preprocess else x
        x = self.trns(x)
        return x


class Veri4D(Module):

    def __init__(self, current_channels, base_channels, downsampling_blocks, activation_fn=lambda : torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)):
        super(Veri4D, self).__init__()
        _modules = []
        for i, layers in enumerate(downsampling_blocks):
            is_last = i == len(downsampling_blocks) - 1
            if is_last:
                _modules.append(SimpleCNNBlock(current_channels, base_channels, layers=layers, activation_fn=activation_fn))
            else:
                if layers - 1 > 0:
                    _modules.append(SimpleCNNBlock(current_channels, base_channels, layers=layers - 1, activation_fn=activation_fn))
                    current_channels = base_channels
                base_channels *= 2
                _modules.append(SimpleCNNBlock(current_channels, base_channels, stride=2, activation_fn=activation_fn))
                current_channels = base_channels
        _modules.append(nn.Conv2d(current_channels, 1, 1))
        _modules.append(nn.Sigmoid())
        self.trns = nn.Sequential(*_modules)

    def forward(self, x):
        return self.trns(x)


class Veri2D(Module):

    def __init__(self, current_channels, fully_connected_blocks, activation_fn=lambda : torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)):
        super(Veri2D, self).__init__()
        _modules = []
        for new_chans in fully_connected_blocks:
            _modules.append(nn.Linear(current_channels, new_chans, bias=False))
            current_channels = new_chans
            _modules.append(nn.BatchNorm1d(current_channels))
            _modules.append(activation_fn())
        _modules.append(nn.Linear(current_channels, 1))
        _modules.append(nn.Sigmoid())
        self.trns = nn.Sequential(*_modules)

    def forward(self, x):
        return self.trns(x)


class GanGroup(Module):

    def __init__(self, discriminators):
        super(GanGroup, self).__init__()
        self.keys = []
        for idx, module in enumerate(discriminators):
            key = str(idx)
            self.add_module(key, module)
            self.keys.append(key)
        self.reals = None
        self.fakes = None

    def set_data(self, reals, fakes):
        self.reals = reals
        self.fakes = fakes
        assert len(self.reals) == len(self.keys) == len(self.fakes)

    def generator_loss(self):
        return sum(self.calc_adv_loss(self._modules[k](i), 1.0) for k, i in zip(self.keys, self.fakes)) / len(self.keys)

    def discriminator_loss(self):
        losses = []
        for i, k in enumerate(self.keys):
            discriminator = self._modules[k]
            assert self.reals[i].size(0) == self.fakes[i].size(0)
            inp = torch.cat((self.reals[i], self.fakes[i]), dim=0).detach()
            r, f = torch.split(discriminator(inp), self.reals[i].size(0), dim=0)
            losses.append(self.calc_adv_loss(r, 1.0) + self.calc_adv_loss(f, 0.0))
        return sum(losses) / len(self.keys)

    def get_train_event(self, disc_steps=3, optimizer=torch_optim.SGD, **optimizer_kwargs):
        opt = optimizer(self.parameters(), **optimizer_kwargs)

        @TrainStepEvent()
        def gan_group_train_evnt(s):
            for _ in range(disc_steps):
                opt.zero_grad()
                loss = self.discriminator_loss()
                PT(disc_loss_=loss)
                loss.backward()
                opt.step()
        return gan_group_train_evnt

    @staticmethod
    def calc_adv_loss(x, target, tolerance=0.01):
        if target == 1:
            return -torch.mean(torch.log(x + tolerance))
        elif target == 0:
            return -torch.mean(torch.log(1.0 + tolerance - x))
        else:
            raise ValueError('target can only be 0 or 1')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AssertSize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (B2D_2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (B2D_4D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (CustomModule,
     lambda: ([], {'py_func': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelShuffleBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShapeLog,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Veri2D,
     lambda: ([], {'current_channels': 4, 'fully_connected_blocks': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_PiotrDabkowski_pytorch_saliency(_paritybench_base):
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

