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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from torch.autograd import Variable


import random


from torch.utils.data import Dataset


import torch.optim as optim


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import torchvision


import math


from torchvision import datasets


from torchvision import transforms


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


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    None
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
        None
        for key, value in list(block_info.items()):
            if type(value) == OrderedDict:
                print_block(value, key, indent + 4)
            elif type(value) == list:
                for v in value:
                    None
            else:
                None
        None
    props = net_info['props']
    layers = net_info['layers']
    None
    None
    None
    None
    None
    None
    None
    for layer in layers:
        print_block(layer, 'layer', 0)


class CaffeNet(nn.Module):

    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)
        for name, model in self.models.items():
            self.add_module(name, model)
        if self.net_info['props'].has_key('input_shape'):
            self.width = int(self.net_info['props']['input_shape']['dim'][3])
            self.height = int(self.net_info['props']['input_shape']['dim'][2])
        else:
            self.width = int(self.net_info['props']['input_dim'][3])
            self.height = int(self.net_info['props']['input_dim'][2])
        self.has_mean = False

    def forward(self, data):
        if self.has_mean:
            batch_size = data.data.size(0)
            data = data - torch.autograd.Variable(self.mean_img.repeat(batch_size, 1, 1, 1))
        blobs = OrderedDict()
        blobs['data'] = data
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        tdata = None
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            tname = layer['top']
            bname = layer['bottom']
            if ltype == 'Data' or ltype == 'Accuracy' or ltype == 'SoftmaxWithLoss' or ltype == 'Region':
                i = i + 1
                continue
            elif ltype == 'BatchNorm':
                i = i + 1
                tname = layers[i]['top']
            if ltype != 'Eltwise':
                bdata = blobs[bname]
                tdata = self._modules[lname](bdata)
                blobs[tname] = tdata
            else:
                bdata0 = blobs[bname[0]]
                bdata1 = blobs[bname[1]]
                tdata = self._modules[lname](bdata0, bdata1)
                blobs[tname] = tdata
            i = i + 1
        return tdata

    def print_network(self):
        None
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        if self.net_info['props'].has_key('mean_file'):
            self.has_mean = True
            mean_file = self.net_info['props']['mean_file']
            blob = caffe_pb2.BlobProto()
            blob.ParseFromString(open(mean_file, 'rb').read())
            mean_img = torch.from_numpy(np.array(blob.data)).float()
            if self.net_info['props'].has_key('input_shape'):
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])
            mean_img = mean_img.view(channels, height, width)
            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)
        model = parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            None
            layers = model.layers
        lmap = {}
        for l in layers:
            lmap[l.name] = l
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                if bias and len(lmap[lname].blobs) > 1:
                    None
                    self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'BatchNorm':
                scale_layer = layers[i + 1]
                self.models[lname].running_mean.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].running_var.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[0].data)))
                self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[1].data)))
                i = i + 2
            elif ltype == 'InnerProduct':
                if type(self.models[lname]) == nn.Sequential:
                    self.models[lname][1].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname][1].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                else:
                    self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'Pooling' or ltype == 'Eltwise' or ltype == 'ReLU' or ltype == 'Region':
                i = i + 1
            else:
                None
                i = i + 1

    def create_network(self, net_info):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()
        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)
        if props.has_key('input_shape'):
            blob_channels['data'] = int(props['input_shape']['dim'][1])
            blob_height['data'] = int(props['input_shape']['dim'][2])
            blob_width['data'] = int(props['input_shape']['dim'][3])
        else:
            blob_channels['data'] = int(props['input_dim'][1])
            blob_height['data'] = int(props['input_dim'][2])
            blob_width['data'] = int(props['input_dim'][3])
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                i = i + 1
                continue
            bname = layer['bottom']
            tname = layer['top']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size, stride, pad, group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2 * pad - kernel_size) / stride + 1
                blob_height[tname] = (blob_height[bname] + 2 * pad - kernel_size) / stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                assert i + 1 < layer_num
                assert layers[i + 1]['type'] == 'Scale'
                momentum = 0.9
                if layer.has_key('batch_norm_param') and layer['batch_norm_param'].has_key('moving_average_fraction'):
                    momentum = float(layer['batch_norm_param']['moving_average_fraction'])
                channels = blob_channels[bname]
                models[lname] = nn.BatchNorm2d(channels, momentum=momentum)
                tname = layers[i + 1]['top']
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 2
            elif ltype == 'ReLU':
                inplace = bname == tname
                if layer.has_key('relu_param') and layer['relu_param'].has_key('negative_slope'):
                    negative_slope = float(layer['relu_param']['negative_slope'])
                    models[lname] = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
                else:
                    models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                padding = 0
                if layer['pooling_param'].has_key('pad'):
                    padding = int(layer['pooling_param']['pad'])
                pool_type = layer['pooling_param']['pool']
                if pool_type == 'MAX' and kernel_size == 2 and stride == 1:
                    models[lname] = MaxPoolStride1()
                    blob_width[tname] = blob_width[bname]
                    blob_height[tname] = blob_height[bname]
                else:
                    if pool_type == 'MAX':
                        models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding)
                    elif pool_type == 'AVE':
                        models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding)
                    if stride > 1:
                        blob_width[tname] = (blob_width[bname] - kernel_size + 1) / stride + 1
                        blob_height[tname] = (blob_height[bname] - kernel_size + 1) / stride + 1
                    else:
                        blob_width[tname] = blob_width[bname] - kernel_size + 1
                        blob_height[tname] = blob_height[bname] - kernel_size + 1
                blob_channels[tname] = blob_channels[bname]
                i = i + 1
            elif ltype == 'Eltwise':
                operation = 'SUM'
                if layer.has_key('eltwise_param') and layer['eltwise_param'].has_key('operation'):
                    operation = layer['eltwise_param']['operation']
                bname0 = bname[0]
                bname1 = bname[1]
                models[lname] = Eltwise(operation)
                blob_channels[tname] = blob_channels[bname0]
                blob_width[tname] = blob_width[bname0]
                blob_height[tname] = blob_height[bname0]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != -1 or blob_height[bname] != -1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(channels, filters)
                blob_channels[tname] = filters
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'Softmax':
                models[lname] = nn.Softmax()
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                loss = nn.CrossEntropyLoss()
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'Region':
                anchors = layer['region_param']['anchors'].strip('"').split(',')
                self.anchors = [float(j) for j in anchors]
                self.num_anchors = int(layer['region_param']['num'])
                self.anchor_step = len(self.anchors) / self.num_anchors
                self.num_classes = int(layer['region_param']['classes'])
                i = i + 1
            else:
                None
                i = i + 1
        return models


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
        None
        for key, value in block.items():
            if key != 'type':
                None
        None


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
        bias = bn_model.bias.data - bn_model.running_mean * bn_model.weight.data / torch.sqrt(bn_model.running_var + eps)
        convert2cpu(bias).numpy().tofile(fp)
        s = conv_model.weight.data.size()
        weight = conv_model.weight.data * (bn_model.weight.data / torch.sqrt(bn_model.running_var + eps)).view(-1, 1, 1, 1).repeat(1, s[1], s[2], s[3])
        convert2cpu(weight).numpy().tofile(fp)
    else:
        bias = bn_model.bias.data - bn_model.running_mean * bn_model.weight.data / torch.sqrt(bn_model.running_var + eps)
        bias.numpy().tofile(fp)
        s = conv_model.weight.data.size()
        weight = conv_model.weight.data * (bn_model.weight.data / torch.sqrt(bn_model.running_var + eps)).view(-1, 1, 1, 1).repeat(1, s[1], s[2], s[3])
        weight.numpy().tofile(fp)


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models) - 1]
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        if self.blocks[len(self.blocks) - 1]['type'] == 'region':
            region_block = self.blocks[len(self.blocks) - 1]
            anchors = region_block['anchors'].split(',')
            self.anchors = [float(i) for i in anchors]
            self.num_anchors = int(region_block['num'])
            self.anchor_step = len(self.anchors) / self.num_anchors
            self.num_classes = int(region_block['classes'])
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.has_mean = False

    def forward(self, x):
        if self.has_mean:
            batch_size = x.data.size(0)
            x = x - torch.autograd.Variable(self.mean_img.repeat(batch_size, 1, 1, 1))
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected' or block['type'] == 'dropout':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'cost':
                continue
            elif block['type'] == 'region':
                continue
            else:
                None
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        out_width = []
        out_height = []
        conv_id = 0
        prev_width = 0
        prev_height = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                prev_width = int(block['width'])
                prev_height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) / 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                prev_width = (prev_width + 2 * pad - kernel_size) / stride + 1
                prev_height = (prev_height + 2 * pad - kernel_size) / stride + 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                padding = 0
                if block.has_key('pad') and int(block['pad']) == 1:
                    padding = int((pool_size - 1) / 2)
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride, padding=padding)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                if stride > 1:
                    prev_width = (prev_width - kernel_size + 1) / stride + 1
                    prev_height = (prev_height - kernel_size + 1) / stride + 1
                else:
                    prev_width = prev_width - kernel_size + 1
                    prev_height = prev_height - kernel_size + 1
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                prev_width = 1
                prev_height = 1
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                prev_width = 1
                prev_height = 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                prev_width = 1
                prev_height = 1
                out_filters.append(1)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                prev_width = prev_width / stride
                prev_height = prev_height / stride
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_width = out_width[layers[0]]
                    prev_height = out_height[layers[0]]
                elif len(layers) == 2:
                    assert layers[0] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_width = out_width[layers[0]]
                    prev_height = out_height[layers[0]]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                prev_width = out_width[ind - 1]
                prev_height = out_height[ind - 1]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(EmptyModule())
            elif block['type'] == 'dropout':
                ind = len(models)
                ratio = float(block['probability'])
                prev_filters = out_filters[ind - 1]
                prev_width = out_width[ind - 1]
                prev_height = out_height[ind - 1]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(nn.Dropout2d(ratio))
            elif block['type'] == 'connected':
                prev_filters = prev_filters * prev_width * prev_height
                filters = int(block['output'])
                is_first = prev_width * prev_height != 1
                if block['activation'] == 'linear':
                    if is_first:
                        model = nn.Sequential(FCView(), nn.Linear(prev_filters, filters))
                    else:
                        model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    if is_first:
                        model = nn.Sequential(FCView(), nn.Linear(prev_filters, filters), nn.LeakyReLU(0.1, inplace=True))
                    else:
                        model = nn.Sequential(nn.Linear(prev_filters, filters), nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    if is_first:
                        model = nn.Sequential(FCView(), nn.Linear(prev_filters, filters), nn.ReLU(inplace=True))
                    else:
                        model = nn.Sequential(nn.Linear(prev_filters, filters), nn.ReLU(inplace=True))
                prev_filters = filters
                prev_width = 1
                prev_height = 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'region':
                continue
            else:
                None
        return models

    def load_weights(self, weightfile):
        if self.blocks[0].has_key('mean_file'):
            mean_file = self.blocks[0]['mean_file']
            mean_file = mean_file.strip('"')
            self.has_mean = True
            blob = caffe_pb2.BlobProto()
            blob.ParseFromString(open(mean_file, 'rb').read())
            mean_img = torch.from_numpy(np.array(blob.data)).float()
            channels = int(self.blocks[0]['channels'])
            height = int(self.blocks[0]['height'])
            width = int(self.blocks[0]['width'])
            mean_img = mean_img.view(channels, height, width)
            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        start = 0
        ind = -2
        is_first = True
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    if is_first:
                        start = load_fc(buf, start, model[1])
                        is_first = False
                    else:
                        start = load_fc(buf, start, model[0])
                elif is_first:
                    start = load_fc(buf, start, model[1])
                    is_first = False
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                None

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
        ind = -1
        is_first = True
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] == 'linear':
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False
                    else:
                        save_fc(fp, model)
                elif is_first:
                    save_fc(fp, model[1])
                    is_first = False
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                None
        fp.close()

    def save_shrink_model(self, out_cfgfile, out_weightfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(out_weightfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
        ind = -1
        is_first = True
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_shrink_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] == 'linear':
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False
                    else:
                        save_fc(fp, model)
                elif is_first:
                    save_fc(fp, model[1])
                    is_first = False
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                None
        fp.close()
        blocks = self.blocks
        for block in blocks:
            if block['type'] == 'convolutional':
                block['batch_normalize'] = '0'
        save_cfg(blocks, out_cfgfile)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Eltwise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FCView,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolStride1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
]

class Test_marvis_pytorch_caffe_darknet_convert(_paritybench_base):
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

