import sys
_module = sys.modules[__name__]
del sys
argv = _module
checkpoints = _module
common = _module
daemon = _module
gpu_info = _module
run = _module
argument = _module
config = _module
gpu_info = _module
loader = _module
fs = _module
http = _module
main = _module
processing = _module
folder = _module
gif = _module
image = _module
multiple = _module
utils = _module
worker = _module
_common = _module
build = _module
setup = _module
color_transfer = _module
transform = _module
gan = _module
generator = _module
mask = _module
model = _module
opencv = _module
bodypart = _module
extract = _module
inferrer = _module
resolver = _module
correct = _module
resize = _module
watermark = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch import cuda


import logging


import time


import torch


import numpy as np


from torchvision import transforms as transforms


import functools


from collections import OrderedDict


class ResnetBlock(torch.nn.Module):
    """Define a resnet block."""

    def __init__(self, dim, padding_type, norm_layer, activation=None, use_dropout=False):
        """
        Resnet Block constuctor.

        :param dim: <> dim
        :param padding_type: <> padding_type
        :param norm_layer: <> norm_layer
        :param activation: <> activation
        :param use_dropout: <> use_dropout
        """
        super(ResnetBlock, self).__init__()
        if activation is None:
            activation = torch.nn.ReLU(True)
        self.conv_block = self.__build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    @staticmethod
    def __build_conv_block(dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        conv_block, p = ResnetBlock.__increment_padding_conv_block(conv_block, p, padding_type)
        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]
        p = 0
        conv_block, p = ResnetBlock.__increment_padding_conv_block(conv_block, p, padding_type)
        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return torch.nn.Sequential(*conv_block)

    @staticmethod
    def __increment_padding_conv_block(conv_block, p, padding_type):
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return conv_block, p

    def forward(self, x):
        """
        Resnet Block forward.

        :param x: <> input
        :return: <> out
        """
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(torch.nn.Module):
    """Global Generator."""

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=torch.nn.BatchNorm2d, padding_type='reflect'):
        """
        Global Generator Constructor.

        :param input_nc:
        :param output_nc:
        :param ngf:
        :param n_downsampling:
        :param n_blocks:
        :param norm_layer:
        :param padding_type:
        """
        if n_blocks < 0:
            raise AssertionError()
        super(GlobalGenerator, self).__init__()
        activation = torch.nn.ReLU(True)
        model = [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, i):
        """
        Global Generator forward.

        :param i: <> input
        :return:
        """
        return self.model(i)


class DeepModel(torch.nn.Module):
    """Deep Model."""

    def initialize(self, opt, gpu_ids, checkpoints_dir):
        """
        Deep Model initialize.

        :param opt: <Config> configuration to use
        :param gpu_ids: <int[]|None> gpu id to use (None = cpu)
        :param checkpoints_dir: <string> path to the directoy where models are
        :return:
        """
        self.opt = opt
        self.checkpoints_dir = checkpoints_dir
        if gpu_ids is None:
            self.gpu_ids = []
        else:
            self.gpu_ids = gpu_ids
        self.net_g = self.__define_g(opt.input_nc, opt.output_nc, opt.ngf, opt.net_g, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, self.gpu_ids)
        self.__load_network(self.net_g)

    def inference(self, label, inst):
        """
        Infere an image.

        :param label: <> label
        :param inst: <> isnt
        :return: <RGB> image
        """
        input_label, _, _, _ = self.__encode_input(label, inst, infer=True)
        input_concat = input_label
        with torch.no_grad():
            fake_image = self.net_g.forward(input_concat)
        return fake_image

    def __load_network(self, network):
        save_path = os.path.join(self.checkpoints_dir)
        state_dict = torch.load(save_path)
        if len(self.gpu_ids) > 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        network.load_state_dict(new_state_dict)

    def __encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if len(self.gpu_ids) > 0:
            input_label = label_map.data
        else:
            input_label = label_map.data
        return input_label, inst_map, real_image, feat_map

    @staticmethod
    def __weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __define_g(self, input_nc, output_nc, ngf, net_g, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm='instance', gpu_ids=None):
        norm_layer = self.__get_norm_layer(norm_type=norm)
        net_g = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
        if len(gpu_ids) > 1:
            net_g = torch.nn.DataParallel(net_g, gpu_ids)
        if len(gpu_ids) > 0:
            net_g
        net_g.apply(self.__weights_init)
        return net_g

    @staticmethod
    def __get_norm_layer(norm_type='instance'):
        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False)
        return norm_layer


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_dreamnettech_dreampower(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

