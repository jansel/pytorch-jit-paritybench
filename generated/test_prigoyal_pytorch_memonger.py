import sys
_module = sys.modules[__name__]
del sys
densenet = _module
resnet = _module
vnet = _module
word_language_model = _module
densenet_new = _module
resnet_new = _module
vnet_new = _module
word_language_model_new = _module
test_memory_baseline = _module
test_memory_optimized = _module

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


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import math


from torch.autograd import Variable


from torch.autograd import Function


from torch.nn import functional as F


from torch.utils.checkpoint import checkpoint_sequential


import time


from torch.utils.checkpoint import checkpoint


import torch.utils.checkpoint as checkpoint


import torch.optim


import torch.backends.cudnn as cudnn


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm_2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features
            for j in range(num_layers):
                layer = _DenseLayer(num_input_features + j * growth_rate, growth_rate, bn_size, drop_rate)
                self.features.add_module('denseblock{}_layer{}'.format(i + 1, j + 1), layer)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, chunks=None):
        modules = [module for k, module in self._modules.items()][0]
        input_var = x.detach()
        input_var.requires_grad = True
        input_var = checkpoint_sequential(modules, chunks, input_var)
        input_var = F.relu(input_var, inplace=True)
        input_var = F.avg_pool2d(input_var, kernel_size=7, stride=1).view(input_var.size(0), -1)
        input_var = self.classifier(input_var)
        return input_var


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreActResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(PreActResNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(16)), ('relu1', nn.ReLU(inplace=True))]))
        self._make_layer(block, 64, layers[0], stage=1)
        self._make_layer(block, 128, layers[1], stride=2, stage=2)
        self._make_layer(block, 256, layers[2], stride=2, stage=3)
        self.features.add_module('bn2', nn.BatchNorm2d(256 * block.expansion))
        self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AvgPool2d(8, stride=1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, stage=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.BatchNorm2d(self.inplanes), nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        self.features.add_module('PreActBottleneck_%d_%d' % (stage, 0), block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.features.add_module('PreActBottleneck_%d_%d' % (stage, i), block(self.inplanes, planes))

    def forward(self, input_var, chunks=3):
        modules = [module for k, module in self._modules.items()][0]
        input_var = checkpoint_sequential(modules, chunks, input_var)
        input_var = input_var.view(input_var.size(0), -1)
        input_var = self.fc(input_var)
        return input_var


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(ContBatchNorm3d, self)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class InputTransition(nn.Module):

    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


def passthrough(x, **kwargs):
    return x


class DownTransition(nn.Module):

    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        return out


class VNet(nn.Module):

    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def custom(self, module):

        def custom_forward(*inputs):
            if isinstance(module, UpTransition):
                inputs = module(inputs[0], inputs[1])
            else:
                inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, x):
        out16 = checkpoint.checkpoint(self.custom(self.in_tr), x)
        out32 = checkpoint.checkpoint(self.custom(self.down_tr32), out16)
        out64 = checkpoint.checkpoint(self.custom(self.down_tr64), out32)
        out128 = checkpoint.checkpoint(self.custom(self.down_tr128), out64)
        out256 = checkpoint.checkpoint(self.custom(self.down_tr256), out128)
        out = checkpoint.checkpoint(self.custom(self.up_tr256), out256, out128)
        out = checkpoint.checkpoint(self.custom(self.up_tr128), out, out64)
        out = checkpoint.checkpoint(self.custom(self.up_tr64), out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.criterion = nn.CrossEntropyLoss()
        self.nvol_grad = None
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return weight.new(self.nlayers, bsz, self.nhid).zero_(), weight.new(self.nlayers, bsz, self.nhid).zero_()
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()

    def save_grad(self):

        def hook(grad):
            self.nvol_grad = grad
        return hook

    def custom(self, start, end):

        def custom_forward(*inputs):
            output, hidden = self.rnn(inputs[0][start:end + 1], (inputs[1], inputs[2]))
            return output, hidden[0], hidden[1]
        return custom_forward

    def forward(self, input, hidden, targets, chunks=4):
        total_modules = input.shape[0]
        chunk_size = int(math.floor(float(total_modules) / chunks))
        start, end = 0, -1
        emb = self.drop(self.encoder(input))
        output = []
        for j in range(chunks):
            start = end + 1
            end = start + chunk_size - 1
            if j == chunks - 1:
                end = total_modules - 1
            out = checkpoint.checkpoint(self.custom(start, end), emb, hidden[0], hidden[1])
            output.append(out[0])
            hidden = out[1], out[2]
        output = torch.cat(output, 0)
        hidden = out[1], out[2]
        output = self.drop(output).view(output.size(0) * output.size(1), output.size(2))
        out = output.detach()
        out.requires_grad = True
        out.register_hook(self.save_grad())
        total_modules = out.shape[0]
        chunks = 10
        chunk_size = int(math.floor(float(total_modules) / chunks))
        start, end = 0, -1
        for i in range(chunks):
            start = end + 1
            end = start + chunk_size - 1
            if i == chunks - 1:
                end = total_modules - 1
            decoded = self.decoder(out[start:end + 1])
            loss = self.criterion(decoded, targets[start:end + 1])
            loss.backward()
        return output, hidden

    def backward(self, output):
        output.backward(self.nvol_grad.data)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DownTransition,
     lambda: ([], {'inChans': 4, 'nConvs': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (InputTransition,
     lambda: ([], {'outChans': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (LUConv,
     lambda: ([], {'nchan': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (OutputTransition,
     lambda: ([], {'inChans': 4, 'elu': 4, 'nll': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_prigoyal_pytorch_memonger(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

