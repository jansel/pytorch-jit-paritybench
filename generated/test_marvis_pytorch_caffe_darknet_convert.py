import sys
_module = sys.modules[__name__]
del sys
caffe2darknet = _module
caffenet = _module
cfg = _module
darknet = _module
darknet2caffe = _module
dataset = _module
debug = _module
detect = _module
main = _module
mxnet2caffe = _module
prototxt = _module
pytorch2caffe = _module
pytorch2darknet = _module
voc_eval = _module
voc_label = _module
shrink_bn_caffe = _module
tf2caffe = _module
utils = _module
valid = _module
visualize = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from torch.autograd import Variable


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import math


class FCView(nn.Module):

    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'


class Eltwise(nn.Module):

    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, x1, x2):
        if self.operation == '+' or self.operation == 'SUM':
            x = x1 + x2
        if self.operation == '*' or self.operation == 'MUL':
            x = x1 * x2
        if self.operation == '/' or self.operation == 'DIV':
            x = x1 / x2
        return x


def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    print('Loading caffemodel: ', caffemodel)
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())
    return model


def parse_prototxt(protofile):

    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0:
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in block:
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1:
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block
    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0:
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if key in props:
                if type(props[key]) == list:
                    props[key].append(value)
                else:
                    props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1:
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()
    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_prototxt(net_info):

    def format_value(value):
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '"%s"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' '] * indent)
        print('%s%s {' % (blanks, prefix))
        for key, value in list(block_info.items()):
            if type(value) == OrderedDict:
                print_block(value, key, indent + 4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)))
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)))
        print('%s}' % blanks)
    props = net_info['props']
    layers = net_info['layers']
    print('name: "%s"' % props['name'])
    print('input: "%s"' % props['input'])
    print('input_dim: %s' % props['input_dim'][0])
    print('input_dim: %s' % props['input_dim'][1])
    print('input_dim: %s' % props['input_dim'][2])
    print('input_dim: %s' % props['input_dim'][3])
    print('')
    for layer in layers:
        print_block(layer, 'layer', 0)


class FCView(nn.Module):

    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class EmptyModule(nn.Module):

    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


def parse_cfg(cfgfile):

    def erase_comment(line):
        line = line.split('#')[0]
        return line
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line.lstrip('[').rstrip(']')
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            line = erase_comment(line)
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()
    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    for block in blocks:
        print('[%s]' % block['type'])
        for key, value in block.items():
            if key != 'type':
                print('%s=%s' % (key, value))
        print('')


def save_cfg(blocks, cfgfile):
    with open(cfgfile, 'w') as fp:
        for block in blocks:
            fp.write('[%s]\n' % block['type'])
            for key, value in block.items():
                if key != 'type':
                    fp.write('%s=%s\n' % (key, value))
            fp.write('\n')


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_conv_shrink_bn(fp, conv_model, bn_model, eps=1e-05):
    if bn_model.bias.is_cuda:
        bias = (bn_model.bias.data - bn_model.running_mean * bn_model.
            weight.data / torch.sqrt(bn_model.running_var + eps))
        convert2cpu(bias).numpy().tofile(fp)
        s = conv_model.weight.data.size()
        weight = conv_model.weight.data * (bn_model.weight.data / torch.
            sqrt(bn_model.running_var + eps)).view(-1, 1, 1, 1).repeat(1, s
            [1], s[2], s[3])
        convert2cpu(weight).numpy().tofile(fp)
    else:
        bias = (bn_model.bias.data - bn_model.running_mean * bn_model.
            weight.data / torch.sqrt(bn_model.running_var + eps))
        bias.numpy().tofile(fp)
        s = conv_model.weight.data.size()
        weight = conv_model.weight.data * (bn_model.weight.data / torch.
            sqrt(bn_model.running_var + eps)).view(-1, 1, 1, 1).repeat(1, s
            [1], s[2], s[3])
        weight.numpy().tofile(fp)


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=
                False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=
                inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True
                ), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d
                (oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1),
            conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1
            ), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512,
            1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 
            1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_marvis_pytorch_caffe_darknet_convert(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Eltwise(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(EmptyModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(FCView(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MaxPoolStride1(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

