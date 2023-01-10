import sys
_module = sys.modules[__name__]
del sys
pytorchyolo = _module
detect = _module
models = _module
test = _module
train = _module
utils = _module
augmentations = _module
datasets = _module
logger = _module
loss = _module
parse_config = _module
transforms = _module
utils = _module

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


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.autograd import Variable


import matplotlib.pyplot as plt


import matplotlib.patches as patches


from matplotlib.ticker import NullLocator


from itertools import chain


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import Dataset


import warnings


from torch.utils.tensorboard import SummaryWriter


import math


import time


import torchvision


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5
        self.grid = torch.zeros(1)
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny)
            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({'batch': int(hyperparams['batch']), 'subdivisions': int(hyperparams['subdivisions']), 'width': int(hyperparams['width']), 'height': int(hyperparams['height']), 'channels': int(hyperparams['channels']), 'optimizer': hyperparams.get('optimizer'), 'momentum': float(hyperparams['momentum']), 'decay': float(hyperparams['decay']), 'learning_rate': float(hyperparams['learning_rate']), 'burn_in': int(hyperparams['burn_in']), 'max_batches': int(hyperparams['max_batches']), 'policy': hyperparams['policy'], 'lr_steps': list(zip(map(int, hyperparams['steps'].split(',')), map(float, hyperparams['scales'].split(','))))})
    assert hyperparams['height'] == hyperparams['width'], 'Height and width should be equal! Non square images are padded with zeros.'
    output_filters = [hyperparams['channels']]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(f'conv_{module_i}', nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size, stride=int(module_def['stride']), padding=pad, bias=not bn))
            if bn:
                modules.add_module(f'batch_norm_{module_i}', nn.BatchNorm2d(filters, momentum=0.1, eps=1e-05))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_{module_i}', nn.LeakyReLU(0.1))
            if module_def['activation'] == 'mish':
                modules.add_module(f'mish_{module_i}', Mish())
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f'_debug_padding_{module_i}', nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f'maxpool_{module_i}', maxpool)
        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module(f'upsample_{module_i}', upsample)
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get('groups', 1))
            modules.add_module(f'route_{module_i}', nn.Sequential())
        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module(f'shortcut_{module_i}', nn.Sequential())
        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f'yolo_{module_i}', yolo_layer)
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def['layers'].split(',')], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get('groups', 1))
                group_id = int(module_def.get('group_id', 0))
                x = combined_outputs[:, group_size * group_id:group_size * (group_id + 1)]
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)
        cutoff = None
        filename = os.path.basename(weights_path)
        if '.conv.' in filename:
            try:
                cutoff = int(filename.split('.')[-1])
            except ValueError:
                pass
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'scale_factor': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_eriklindernoren_PyTorch_YOLOv3(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

