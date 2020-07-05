import sys
_module = sys.modules[__name__]
del sys
caffenet = _module
detection = _module
prototxt = _module
test_caffe_loader1 = _module
test_caffe_loader2 = _module
train_lenet5 = _module
train_ssd = _module
verify_deploy = _module
verify_time = _module
verify_train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import random


import numpy as np


import math


import torch


import torch.nn as nn


from torch.nn.parameter import Parameter


from torch.autograd import Variable


from torch.autograd import Function


import torch.nn.functional as F


from collections import OrderedDict


from itertools import product as product


import time


import torch.optim as optim


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def save_prototxt(net_info, protofile, region=True):
    fp = open(protofile, 'w')

    def format_value(value):
        if is_number(value):
            return value
        elif value in ['true', 'false', 'MAX', 'SUM', 'AVE', 'TRAIN', 'TEST', 'WARP', 'LINEAR', 'AREA', 'NEAREST', 'CUBIC', 'LANCZOS4', 'CENTER', 'LMDB']:
            return value
        else:
            return '"%s"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' '] * indent)
        print('%s%s {' % (blanks, prefix), file=fp)
        for key, value in list(block_info.items()):
            if type(value) == OrderedDict:
                print_block(value, key, indent + 4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)), file=fp)
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)), file=fp)
        print('%s}' % blanks, file=fp)
    props = net_info['props']
    print('name: "%s"' % props['name'], file=fp)
    if 'input' in props:
        print('input: "%s"' % props['input'], file=fp)
    if 'input_dim' in props:
        print('input_dim: %s' % props['input_dim'][0], file=fp)
        print('input_dim: %s' % props['input_dim'][1], file=fp)
        print('input_dim: %s' % props['input_dim'][2], file=fp)
        print('input_dim: %s' % props['input_dim'][3], file=fp)
    print('', file=fp)
    layers = net_info['layers']
    for layer in layers:
        if layer['type'] != 'Region' or region == True:
            print_block(layer, 'layer', 0)
    fp.close()


class CaffeData(nn.Module):

    def __init__(self, layer):
        super(CaffeData, self).__init__()
        net_info = OrderedDict()
        props = OrderedDict()
        props['name'] = 'temp network'
        net_info['props'] = props
        None
        if layer.has_key('include'):
            layer.pop('include')
        net_info['layers'] = [layer]
        rand_val = random.random()
        protofile = '.temp_data%f.prototxt' % rand_val
        save_prototxt(net_info, protofile)
        weightfile = '.temp_data%f.caffemodel' % rand_val
        open(weightfile, 'w').close()
        caffe.set_mode_cpu()
        if layer.has_key('include') and layer['include'] == 'TRAIN':
            self.net = caffe.Net(protofile, weightfile, caffe.TRAIN)
        else:
            self.net = caffe.Net(protofile, weightfile, caffe.TEST)
        self.register_buffer('data', torch.zeros(1))
        self.register_buffer('label', torch.zeros(1))

    def __repr__(self):
        return 'CaffeData()'

    def forward(self):
        self.net.forward()
        data = self.net.blobs['data'].data
        label = self.net.blobs['label'].data
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        self.data.resize_(data.size()).copy_(data)
        self.label.resize_(label.size()).copy_(label)
        return Variable(self.data), Variable(self.label)


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

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = torch.max(x, inputs[i])
        else:
            None
        return x


class Scale(nn.Module):

    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x


class Crop(nn.Module):

    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def __repr__(self):
        return 'Crop(axis=%d, offset=%d)' % (self.axis, self.offset)

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, Variable(indices))
        return x


class Slice(nn.Module):

    def __init__(self, axis, slice_points):
        super(Slice, self).__init__()
        self.axis = axis
        self.slice_points = slice_points

    def __repr__(self):
        return 'Slice(axis=%d, slice_points=%s)' % (self.axis, self.slice_points)

    def forward(self, x):
        prev = 0
        outputs = []
        is_cuda = x.data.is_cuda
        if is_cuda:
            device_id = x.data.get_device()
        for idx, slice_point in enumerate(self.slice_points):
            rng = range(prev, slice_point)
            rng = torch.LongTensor(rng)
            if is_cuda:
                rng = rng
            rng = Variable(rng)
            y = x.index_select(self.axis, rng)
            prev = slice_point
            outputs.append(y)
        return tuple(outputs)


class Concat(nn.Module):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)


class Permute(nn.Module):

    def __init__(self, order0, order1, order2, order3):
        super(Permute, self).__init__()
        self.order0 = order0
        self.order1 = order1
        self.order2 = order2
        self.order3 = order3

    def __repr__(self):
        return 'Permute(%d, %d, %d, %d)' % (self.order0, self.order1, self.order2, self.order3)

    def forward(self, x):
        x = x.permute(self.order0, self.order1, self.order2, self.order3).contiguous()
        return x


class Softmax(nn.Module):

    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Softmax(axis=%d)' % self.axis

    def forward(self, x):
        assert self.axis == len(x.size()) - 1
        orig_size = x.size()
        dims = x.size(self.axis)
        x = F.softmax(x.view(-1, dims))
        x = x.view(*orig_size)
        return x


class SoftmaxWithLoss(nn.CrossEntropyLoss):

    def __init__(self):
        super(SoftmaxWithLoss, self).__init__()

    def __repr__(self):
        return 'SoftmaxWithLoss()'

    def forward(self, input, targets):
        targets = targets.long()
        return nn.CrossEntropyLoss.forward(self, input, targets)


class Normalize(nn.Module):

    def __init__(self, n_channels, scale=1.0):
        super(Normalize, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale
        self.register_parameter('bias', None)

    def __repr__(self):
        return 'Normalize(channels=%d, scale=%f)' % (self.n_channels, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class Flatten(nn.Module):

    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Flatten(axis=%d)' % self.axis

    def forward(self, x):
        left_size = 1
        for i in range(self.axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1).contiguous()


class LRNFunc(Function):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        super(LRNFunc, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)


class LRN(nn.Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        super(LRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def __repr__(self):
        return 'LRN(size=%d, alpha=%f, beta=%f, k=%d)' % (self.size, self.alpha, self.beta, self.k)

    def forward(self, input):
        return LRNFunc(self.size, self.alpha, self.beta, self.k)(input)


class Reshape(nn.Module):

    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def __repr__(self):
        return 'Reshape(dims=%s)' % self.dims

    def forward(self, x):
        orig_dims = x.size()
        new_dims = [(orig_dims[i] if self.dims[i] == 0 else self.dims[i]) for i in range(len(self.dims))]
        return x.view(*new_dims).contiguous()


class Accuracy(nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def __repr__(self):
        return 'Accuracy()'

    def forward(self, output, label):
        max_vals, max_ids = output.data.max(1)
        n_correct = (max_ids.view(-1).float() == label.data).sum()
        batchsize = output.data.size(0)
        accuracy = float(n_correct) / batchsize
        None
        accuracy = output.data.new().resize_(1).fill_(accuracy)
        return Variable(accuracy)


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """

    def __init__(self, min_size, max_size, aspects, clip, flip, step, offset, variances):
        super(PriorBox, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.aspects = aspects
        self.clip = clip
        self.flip = flip
        self.step = step
        self.offset = offset
        self.variances = variances

    def __repr__(self):
        return 'PriorBox(min_size=%f, max_size=%f, clip=%d, step=%d, offset=%f, variances=%s)' % (self.min_size, self.max_size, self.clip, self.step, self.offset, self.variances)

    def forward(self, feature, image):
        mean = []
        feature_height = feature.size(2)
        feature_width = feature.size(3)
        image_height = image.size(2)
        image_width = image.size(3)
        for j in range(feature_height):
            for i in range(feature_width):
                cx = (i + self.offset) * self.step / image_width
                cy = (j + self.offset) * self.step / image_height
                mw = float(self.min_size) / image_width
                mh = float(self.min_size) / image_height
                mean += [cx - mw / 2.0, cy - mh / 2.0, cx + mw / 2.0, cy + mh / 2.0]
                if self.max_size > self.min_size:
                    ww = math.sqrt(mw * float(self.max_size) / image_width)
                    hh = math.sqrt(mh * float(self.max_size) / image_height)
                    mean += [cx - ww / 2.0, cy - hh / 2.0, cx + ww / 2.0, cy + hh / 2.0]
                    for aspect in self.aspects:
                        ww = mw * math.sqrt(aspect)
                        hh = mh / math.sqrt(aspect)
                        mean += [cx - ww / 2.0, cy - hh / 2.0, cx + ww / 2.0, cy + hh / 2.0]
                        if self.flip:
                            ww = mw / math.sqrt(aspect)
                            hh = mh * math.sqrt(aspect)
                            mean += [cx - ww / 2.0, cy - hh / 2.0, cx + ww / 2.0, cy + hh / 2.0]
        output1 = torch.Tensor(mean).view(-1, 4)
        output2 = torch.FloatTensor(self.variances).view(1, 4).expand_as(output1)
        if self.clip:
            output1.clamp_(max=1, min=0)
        output1 = output1.view(1, 1, -1)
        output2 = output2.contiguous().view(1, 1, -1)
        output = torch.cat([output1, output2], 1)
        if feature.data.is_cuda:
            device_id = feature.data.get_device()
            return Variable(output)
        else:
            return Variable(output)


SUPPORTED_LAYERS = ['Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss', 'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput']


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


def print_prototxt(net_info):

    def format_value(value):
        if is_number(value):
            return value
        elif value in ['true', 'false', 'MAX', 'SUM', 'AVE', 'TRAIN', 'TEST', 'WARP', 'LINEAR', 'AREA', 'NEAREST', 'CUBIC', 'LANCZOS4', 'CENTER', 'LMDB']:
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


class CaffeNet(nn.Module):

    def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN'):
        super(CaffeNet, self).__init__()
        self.omit_data_layer = omit_data_layer
        self.phase = phase
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info, width, height, channels)
        for name, model in self.models.items():
            self.add_module(name, model)
        self.has_mean = False
        if self.net_info['props'].has_key('mean_file'):
            self.has_mean = True
            self.mean_file = self.net_info['props']['mean_file']
        self.blobs = None
        self.verbose = True
        self.train_outputs = []
        self.eval_outputs = []
        self.forward_data_only = False
        self.forward_net_only = False

    def set_forward_data_only(self, flag):
        self.forward_data_only = flag
        if self.forward_data_only:
            self.forward_net_only = False

    def set_forward_net_only(self, flag):
        self.forward_net_only = flag
        if self.forward_net_only:
            self.forward_data_only = False

    def set_forward_both(self):
        self.forward_data_only = False
        self.forward_net_only = False

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'TRAIN':
            self.train()
        else:
            self.eval()

    def set_mean_file(self, mean_file):
        if mean_file != '':
            self.has_mean = True
            self.mean_file = mean_file
        else:
            self.has_mean = False
            self.mean_file = ''

    def set_train_outputs(self, *outputs):
        self.train_outputs = list(outputs)

    def set_eval_outputs(self, *outputs):
        self.eval_outputs = list(outputs)

    def get_outputs(self, outputs):
        blobs = []
        for name in outputs:
            blobs.append(self.blobs[name])
        return blobs

    def forward(self, *inputs):
        if self.training:
            self.set_phase('TRAIN')
        else:
            self.set_phase('TEST')
        self.blobs = OrderedDict()
        if len(inputs) >= 2:
            data = inputs[0]
            label = inputs[1]
            self.blobs['data'] = data
            self.blobs['label'] = label
            if self.has_mean:
                nB = data.data.size(0)
                nC = data.data.size(1)
                nH = data.data.size(2)
                nW = data.data.size(3)
                data = data - Variable(self.mean_img.view(1, nC, nH, nW).expand(nB, nC, nH, nW))
        elif len(inputs) == 1:
            data = inputs[0]
            self.blobs['data'] = data
            if self.has_mean:
                nB = data.data.size(0)
                nC = data.data.size(1)
                nH = data.data.size(2)
                nW = data.data.size(3)
                data = data - Variable(self.mean_img.view(1, nC, nH, nW).expand(nB, nC, nH, nW))
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        self.output_loss = None
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if layer.has_key('include') and layer['include'].has_key('phase'):
                phase = layer['include']['phase']
                lname = lname + '.' + phase
                if phase != self.phase:
                    i = i + 1
                    continue
            ltype = layer['type']
            tname = layer['top']
            tnames = tname if type(tname) == list else [tname]
            if ltype in ['Data', 'AnnotatedData']:
                if not self.omit_data_layer and not self.forward_net_only:
                    tdatas = self._modules[lname]()
                    if type(tdatas) != tuple:
                        tdatas = tdatas,
                    assert len(tdatas) == len(tnames)
                    for index, tdata in enumerate(tdatas):
                        self.blobs[tnames[index]] = tdata
                    output_size = self.blobs[tnames[0]].size()
                    if self.verbose:
                        None
                i = i + 1
                continue
            if self.forward_data_only:
                break
            bname = layer['bottom']
            bnames = bname if type(bname) == list else [bname]
            if True:
                bdatas = [self.blobs[name] for name in bnames]
                tdatas = self._modules[lname](*bdatas)
                if type(tdatas) != tuple:
                    tdatas = tdatas,
                assert len(tdatas) == len(tnames)
                for index, tdata in enumerate(tdatas):
                    self.blobs[tnames[index]] = tdata
                if ltype in ['SoftmaxWithLoss', 'MultiBoxLoss']:
                    if self.output_loss != None:
                        self.output_loss += tdatas[0]
                    else:
                        self.output_loss = tdatas[0]
                i = i + 1
            input_size = self.blobs[bnames[0]].size()
            output_size = self.blobs[tnames[0]].size()
            if self.verbose:
                None
        if self.forward_data_only:
            odatas = [blob for blob in self.blobs.values()]
            return tuple(odatas)
        elif self.training:
            if len(self.train_outputs) > 1:
                odatas = [self.blobs[name] for name in self.train_outputs]
                return tuple(odatas)
            elif len(self.train_outputs) == 1:
                return self.blobs[self.train_outputs[0]]
            else:
                return self.blobs
        elif len(self.eval_outputs) > 1:
            odatas = [self.blobs[name] for name in self.eval_outputs]
            return tuple(odatas)
        elif len(self.eval_outputs) == 1:
            return self.blobs[self.eval_outputs[0]]
        else:
            return self.blobs

    def get_loss(self):
        return self.output_loss

    def print_network(self):
        None
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        if self.has_mean:
            None
            mean_blob = caffe_pb2.BlobProto()
            mean_blob.ParseFromString(open(self.mean_file, 'rb').read())
            if self.net_info['props'].has_key('input_shape'):
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])
            mu = np.array(mean_blob.data)
            mu.resize(channels, height, width)
            mu = mu.mean(1).mean(1)
            mean_img = torch.from_numpy(mu).view(channels, 1, 1).expand(channels, height, width).float()
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
            if layer.has_key('include') and layer['include'].has_key('phase'):
                phase = layer['include']['phase']
                lname = lname + '.' + phase
            ltype = layer['type']
            if not lmap.has_key(lname):
                i = i + 1
                continue
            if ltype in ['Convolution', 'Deconvolution']:
                None
                convolution_param = layer['convolution_param']
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                caffe_weight = np.array(lmap[lname].blobs[0].data)
                caffe_weight = torch.from_numpy(caffe_weight).view_as(self.models[lname].weight)
                self.models[lname].weight.data.copy_(caffe_weight)
                if bias and len(lmap[lname].blobs) > 1:
                    self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'BatchNorm':
                None
                self.models[lname].running_mean.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].running_var.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))
                i = i + 1
            elif ltype == 'Scale':
                None
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'Normalize':
                None
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                i = i + 1
            elif ltype == 'InnerProduct':
                None
                if type(self.models[lname]) == nn.Sequential:
                    self.models[lname][1].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname][1].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                else:
                    self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            else:
                if not ltype in SUPPORTED_LAYERS:
                    None
                i = i + 1

    def create_network(self, net_info, input_width=None, input_height=None, input_channels=None):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()
        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)
        blob_channels['data'] = 3
        if input_channels != None:
            blob_channels['data'] = input_channels
        blob_height['data'] = 1
        if input_height != None:
            blob_height['data'] = input_height
        blob_width['data'] = 1
        if input_width != None:
            blob_width['data'] = input_width
        if props.has_key('input_shape'):
            blob_channels['data'] = int(props['input_shape']['dim'][1])
            blob_height['data'] = int(props['input_shape']['dim'][2])
            blob_width['data'] = int(props['input_shape']['dim'][3])
            self.width = int(props['input_shape']['dim'][3])
            self.height = int(props['input_shape']['dim'][2])
        elif props.has_key('input_dim'):
            blob_channels['data'] = int(props['input_dim'][1])
            blob_height['data'] = int(props['input_dim'][2])
            blob_width['data'] = int(props['input_dim'][3])
            self.width = int(props['input_dim'][3])
            self.height = int(props['input_dim'][2])
        if input_width != None and input_height != None:
            blob_width['data'] = input_width
            blob_height['data'] = input_height
            self.width = input_width
            self.height = input_height
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if layer.has_key('include') and layer['include'].has_key('phase'):
                phase = layer['include']['phase']
                lname = lname + '.' + phase
            ltype = layer['type']
            tname = layer['top']
            if ltype in ['Data', 'AnnotatedData']:
                if not self.omit_data_layer:
                    models[lname] = CaffeData(layer.copy())
                    data, label = models[lname].forward()
                    data_name = tname[0] if type(tname) == list else tname
                    blob_channels[data_name] = data.size(1)
                    blob_height[data_name] = data.size(2)
                    blob_width[data_name] = data.size(3)
                    self.height = blob_height[data_name]
                    self.width = blob_width[data_name]
                i = i + 1
                continue
            bname = layer['bottom']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                dilation = 1
                if convolution_param.has_key('dilation'):
                    dilation = int(convolution_param['dilation'])
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, groups=group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2 * pad - kernel_size) / stride + 1
                blob_height[tname] = (blob_height[bname] + 2 * pad - kernel_size) / stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                momentum = 0.9
                if layer.has_key('batch_norm_param') and layer['batch_norm_param'].has_key('moving_average_fraction'):
                    momentum = float(layer['batch_norm_param']['moving_average_fraction'])
                channels = blob_channels[bname]
                models[lname] = nn.BatchNorm2d(channels, momentum=momentum, affine=False)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Scale':
                channels = blob_channels[bname]
                models[lname] = Scale(channels)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
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
                if pool_type == 'MAX':
                    models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=True)
                elif pool_type == 'AVE':
                    models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding, ceil_mode=True)
                blob_width[tname] = int(math.ceil((blob_width[bname] + 2 * padding - kernel_size) / float(stride))) + 1
                blob_height[tname] = int(math.ceil((blob_height[bname] + 2 * padding - kernel_size) / float(stride))) + 1
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
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Dropout':
                channels = blob_channels[bname]
                dropout_ratio = float(layer['dropout_param']['dropout_ratio'])
                models[lname] = nn.Dropout(dropout_ratio, inplace=True)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Normalize':
                channels = blob_channels[bname]
                scale = float(layer['norm_param']['scale_filler']['value'])
                models[lname] = Normalize(channels, scale)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'LRN':
                local_size = int(layer['lrn_param']['local_size'])
                alpha = float(layer['lrn_param']['alpha'])
                beta = float(layer['lrn_param']['beta'])
                models[lname] = LRN(local_size, alpha, beta)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Permute':
                orders = layer['permute_param']['order']
                order0 = int(orders[0])
                order1 = int(orders[1])
                order2 = int(orders[2])
                order3 = int(orders[3])
                models[lname] = Permute(order0, order1, order2, order3)
                shape = [1, blob_channels[bname], blob_height[bname], blob_width[bname]]
                blob_channels[tname] = shape[order1]
                blob_height[tname] = shape[order2]
                blob_width[tname] = shape[order3]
                i = i + 1
            elif ltype == 'Flatten':
                axis = int(layer['flatten_param']['axis'])
                models[lname] = Flatten(axis)
                blob_channels[tname] = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Slice':
                axis = int(layer['slice_param']['axis'])
                assert axis == 1
                assert type(tname) == list
                slice_points = layer['slice_param']['slice_point']
                assert type(slice_points) == list
                assert len(slice_points) == len(tname) - 1
                slice_points = [int(s) for s in slice_points]
                shape = [1, blob_channels[bname], blob_height[bname], blob_width[bname]]
                slice_points.append(shape[axis])
                models[lname] = Slice(axis, slice_points)
                prev = 0
                for idx, tn in enumerate(tname):
                    blob_channels[tn] = slice_points[idx] - prev
                    blob_width[tn] = blob_width[bname]
                    blob_height[tn] = blob_height[bname]
                    prev = slice_points[idx]
                i = i + 1
            elif ltype == 'Concat':
                axis = 1
                if layer.has_key('concat_param') and layer['concat_param'].has_key('axis'):
                    axis = int(layer['concat_param']['axis'])
                models[lname] = Concat(axis)
                if axis == 1:
                    blob_channels[tname] = 0
                    for bn in bname:
                        blob_channels[tname] += blob_channels[bn]
                        blob_width[tname] = blob_width[bn]
                        blob_height[tname] = blob_height[bn]
                elif axis == 2:
                    blob_channels[tname] = blob_channels[bname[0]]
                    blob_width[tname] = 1
                    blob_height[tname] = 0
                    for bn in bname:
                        blob_height[tname] += blob_height[bn]
                i = i + 1
            elif ltype == 'PriorBox':
                min_size = float(layer['prior_box_param']['min_size'])
                max_size = -1
                if layer['prior_box_param'].has_key('max_size'):
                    max_size = float(layer['prior_box_param']['max_size'])
                aspects = []
                if layer['prior_box_param'].has_key('aspect_ratio'):
                    None
                    aspects = layer['prior_box_param']['aspect_ratio']
                    aspects = [float(aspect) for aspect in aspects]
                clip = layer['prior_box_param']['clip'] == 'true'
                flip = False
                if layer['prior_box_param'].has_key('flip'):
                    flip = layer['prior_box_param']['flip'] == 'true'
                step = int(layer['prior_box_param']['step'])
                offset = float(layer['prior_box_param']['offset'])
                variances = layer['prior_box_param']['variance']
                variances = [float(v) for v in variances]
                models[lname] = PriorBox(min_size, max_size, aspects, clip, flip, step, offset, variances)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'DetectionOutput':
                num_classes = int(layer['detection_output_param']['num_classes'])
                bkg_label = int(layer['detection_output_param']['background_label_id'])
                top_k = int(layer['detection_output_param']['nms_param']['top_k'])
                keep_top_k = int(layer['detection_output_param']['keep_top_k'])
                conf_thresh = float(layer['detection_output_param']['confidence_threshold'])
                nms_thresh = float(layer['detection_output_param']['nms_param']['nms_threshold'])
                models[lname] = Detection(num_classes, bkg_label, top_k, conf_thresh, nms_thresh, keep_top_k)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'MultiBoxLoss':
                num_classes = int(layer['multibox_loss_param']['num_classes'])
                overlap_threshold = float(layer['multibox_loss_param']['overlap_threshold'])
                prior_for_matching = layer['multibox_loss_param']['use_prior_for_matching'] == 'true'
                bkg_label = int(layer['multibox_loss_param']['background_label_id'])
                neg_mining = True
                neg_pos = float(layer['multibox_loss_param']['neg_pos_ratio'])
                neg_overlap = float(layer['multibox_loss_param']['neg_overlap'])
                models[lname] = MultiBoxLoss(num_classes, overlap_threshold, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, use_gpu=True)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Crop':
                axis = int(layer['crop_param']['axis'])
                offset = int(layer['crop_param']['offset'])
                models[lname] = Crop(axis, offset)
                blob_channels[tname] = blob_channels[bname[0]]
                blob_width[tname] = blob_width[bname[0]]
                blob_height[tname] = blob_height[bname[0]]
                i = i + 1
            elif ltype == 'Deconvolution':
                in_channels = blob_channels[bname]
                out_channels = int(layer['convolution_param']['num_output'])
                group = int(layer['convolution_param']['group'])
                kernel_w = int(layer['convolution_param']['kernel_w'])
                kernel_h = int(layer['convolution_param']['kernel_h'])
                stride_w = int(layer['convolution_param']['stride_w'])
                stride_h = int(layer['convolution_param']['stride_h'])
                pad_w = int(layer['convolution_param']['pad_w'])
                pad_h = int(layer['convolution_param']['pad_h'])
                kernel_size = kernel_h, kernel_w
                stride = stride_h, stride_w
                padding = pad_h, pad_w
                bias_term = layer['convolution_param']['bias_term'] != 'false'
                models[lname] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=group, bias=bias_term)
                blob_channels[tname] = out_channels
                blob_width[tname] = 2 * blob_width[bname]
                blob_height[tname] = 2 * blob_height[bname]
                i = i + 1
            elif ltype == 'Reshape':
                reshape_dims = layer['reshape_param']['shape']['dim']
                reshape_dims = [int(item) for item in reshape_dims]
                models[lname] = Reshape(reshape_dims)
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Softmax':
                axis = 1
                if layer.has_key('softmax_param') and layer['softmax_param'].has_key('axis'):
                    axis = int(layer['softmax_param']['axis'])
                models[lname] = Softmax(axis)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Accuracy':
                models[lname] = Accuracy()
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                models[lname] = SoftmaxWithLoss()
                blob_channels[tname] = 1
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            else:
                None
                i = i + 1
            input_width = blob_width[bname] if type(bname) != list else blob_width[bname[0]]
            input_height = blob_height[bname] if type(bname) != list else blob_height[bname[0]]
            input_channels = blob_channels[bname] if type(bname) != list else blob_channels[bname[0]]
            output_width = blob_width[tname] if type(tname) != list else blob_width[tname[0]]
            output_height = blob_height[tname] if type(tname) != list else blob_height[tname[0]]
            output_channels = blob_channels[tname] if type(tname) != list else blob_channels[tname[0]]
            None
        return models


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]], 1)


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    x2 = boxes[:, (2)]
    y2 = boxes[:, (3)]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detection(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, keep_top_k):
        super(Detection, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.keep_top_k = keep_top_k
        self.variance = [0.1, 0.2]

    def forward(self, loc, conf, prior):
        """
        Args:
            loc: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors*num_classes]
            prior: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,2, num_priors*4]
        """
        num = loc.size(0)
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num_classes = self.num_classes
        num_priors = prior_data.size(2) / 4
        if num == 1:
            conf_preds = conf_data.view(num_priors, self.num_classes).t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        assert num == 1
        if num_classes == 2:
            loc_data = loc_data[0].view(-1, 4).clone()
            prior_data = center_size(prior_data[0][0].view(-1, 4).clone())
            decoded_boxes = decode(loc_data, prior_data, self.variance)
            conf_scores = conf_preds[0].clone()
            num_det = 0
            cl = 1
            c_mask = conf_scores[cl].gt(self.conf_thresh)
            if c_mask.sum() == 0:
                output = torch.Tensor([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).view(1, 1, 1, 7)
                return Variable(conf.data.new().resize_(output.size()).copy_(output))
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            count = min(count, self.keep_top_k)
            extra_info = torch.FloatTensor([0.0, 1.0]).view(1, 2).expand(num_priors, 2)
            extra_info = conf.data.new().resize_(extra_info.size()).copy_(extra_info)
            output = torch.cat((extra_info[ids[:count]], scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
            return Variable(output.unsqueeze(0).unsqueeze(0))
        else:
            loc_data = loc_data[0].view(-1, 4).clone()
            prior_data = center_size(prior_data[0][0].view(-1, 4).clone())
            decoded_boxes = decode(loc_data, prior_data, self.variance)
            conf_scores = conf_preds[0].clone()
            num_det = 0
            cl = 1
            outputs = []
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                if c_mask.sum() == 0:
                    continue
                scores = conf_scores[cl][c_mask]
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                count = min(count, self.keep_top_k)
                extra_info = torch.FloatTensor([0.0, cl]).view(1, 2).expand(count, 2)
                extra_info = conf.data.new().resize_(extra_info.size()).copy_(extra_info)
                output = torch.cat((extra_info, scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                outputs.append(output)
            outputs = torch.cat(outputs, 0)
            return Variable(outputs.unsqueeze(0).unsqueeze(0))


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1)) + x_max


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, (0)] * inter[:, :, (1)]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
        best_truth_overlap[best_prior_idx[j]] = best_prior_overlap[j]
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + ¦Áloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by ¦Áwhich is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        num = loc_data.size(0)
        num_priors = loc_data.size(1) / 4
        num_classes = self.num_classes
        loc_data = loc_data.view(num, num_priors, 4)
        conf_data = conf_data.view(num, num_priors, num_classes)
        priors = priors[0][0].view(num_priors, 4)
        targets = targets.view(-1, 8)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.zeros(num, num_priors).long()
        for idx in range(num):
            sub_mask = targets[:, (0)] == idx
            if sub_mask.data.float().sum() == 0:
                continue
            sub_targets = targets[sub_mask.view(-1, 1).expand_as(targets)].view(-1, 8)
            truths = sub_targets[:, 3:7].data
            labels = sub_targets[:, (1)].data
            defaults = priors.data
            defaults = center_size(defaults)
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf).view(-1, 1) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = Variable(torch.clamp(self.negpos_ratio * num_pos.data.float(), max=pos.size(1) - 1).long())
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l + loss_c


class ParallelCaffeNet(nn.Module):

    def __init__(self, caffe_module, device_ids):
        super(ParallelCaffeNet, self).__init__()
        self.device_ids = device_ids
        self.module = nn.DataParallel(caffe_module, device_ids)

    def convert2batch(self, label, batch_size, ngpus):
        if ngpus > 1:
            num = label.size(2)
            label = label.expand(ngpus, 1, num, 8).contiguous()
            sub_sz = batch_size / ngpus
            for i in range(ngpus):
                sub_label = label[(i), (0), :, (0)]
                sub_label[sub_label > (i + 1) * sub_sz] = -1
                sub_label[sub_label < i * sub_sz] = -1
                sub_label = sub_label - sub_sz * i
                label[(i), (0), :, (0)] = sub_label
        return label

    def forward(self):
        self.module.module.set_forward_data_only(True)
        data, label = self.module.module()
        label_data = self.convert2batch(label.data, data.size(0), len(self.device_ids))
        label = Variable(label_data)
        self.module.module.set_forward_net_only(True)
        return self.module(data, label)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Accuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 64])], {}),
     False),
    (Crop,
     lambda: ([], {'axis': 4, 'offset': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCView,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {'axis': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PriorBox,
     lambda: ([], {'min_size': 4, 'max_size': 4, 'aspects': 4, 'clip': 4, 'flip': 4, 'step': 4, 'offset': 4, 'variances': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scale,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_marvis_pytorch_caffe(_paritybench_base):
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

