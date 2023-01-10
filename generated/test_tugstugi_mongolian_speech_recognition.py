import sys
_module = sys.modules[__name__]
del sys
datasets = _module
background_sounds = _module
bolor_speech = _module
collate = _module
colored_noise = _module
dl_mbspeech = _module
german_speech = _module
kazakh335h_speech = _module
kazakh78h_speech = _module
libri_speech = _module
mb_speech = _module
transforms = _module
decoder = _module
eval = _module
misc = _module
lr_policies = _module
optimizers = _module
models = _module
crnn = _module
layers = _module
quartznet = _module
jasper_block = _module
jasper_encoder_decoder = _module
tiny_jasper = _module
tiny_wav2letter = _module
preprop_dataset = _module
record_and_transcribe = _module
swa = _module
train = _module
transcribe = _module
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


import numpy as np


import random


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import torch


import time


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


from torch.optim import Optimizer


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Subset


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class Speech2TextCRNN(nn.Module):

    def __init__(self, vocab):
        super(Speech2TextCRNN, self).__init__()
        imgH = 32
        nc = 1
        nclass = len(vocab)
        nh = 256
        leakyRelu = False
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)
        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        input = input.unsqueeze(1)
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class C(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, activation='relu', dropout_rate=0.0):
        """1D Convolution with the batch normalization and RELU."""
        super(C, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        assert 1 <= stride <= 2
        if dilation > 1:
            assert stride == 1
            padding = (kernel_size - 1) * dilation // 2
        else:
            padding = (kernel_size - stride + 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain('relu'))
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        if self.activation == 'relu':
            y = F.relu(y, inplace=True)
        if self.dropout_rate > 0:
            y = F.dropout(y, p=self.dropout_rate, training=self.training, inplace=False)
        return y


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, heads=-1, bias=False, use_mask=True):
        if not (heads == -1 or groups == in_channels):
            raise ValueError('Only use heads for depthwise convolutions')
        if heads != -1:
            self.real_out_channels = out_channels
            in_channels = heads
            out_channels = heads
            groups = heads
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (lens + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens
            max_len = x.size(2)
            mask = torch.arange(max_len).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).type(torch.bool), 0)
            del mask
            lens = self.get_seq_len(lens)
        if self.heads != -1:
            sh = x.shape
            x = x.view(-1, self.heads, sh[-1])
        out, lens = super(MaskedConv1d, self).forward(x), lens
        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)
        return out, lens


class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        return x


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError('Only stride OR dilation may be greater than 1')
    if dilation > 1:
        return dilation * kernel_size // 2 - 1
    return kernel_size // 2


class JasperBlock(nn.Module):

    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1, dilation=1, padding='same', dropout=0.2, activation=None, residual=True, groups=1, separable=False, heads=-1, tied=False, normalization='batch', norm_groups=1, residual_mode='add', residual_panes=[], conv_mask=False):
        super(JasperBlock, self).__init__()
        if padding != 'same':
            raise ValueError("currently only 'same' padding is supported")
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        if tied:
            rep_layer = self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val, groups=groups, heads=heads, separable=separable, normalization=normalization, norm_groups=norm_groups)
        for _ in range(repeat - 1):
            if tied:
                self.conv.extend(rep_layer)
            else:
                self.conv.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val, groups=groups, heads=heads, separable=separable, normalization=normalization, norm_groups=norm_groups))
            self.conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        if tied:
            self.conv.extend(rep_layer)
        else:
            self.conv.extend(self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val, groups=groups, heads=heads, separable=separable, normalization=normalization, norm_groups=norm_groups))
        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(modules=self._get_conv_bn_layer(ip, planes, kernel_size=1, normalization=normalization, norm_groups=norm_groups)))
        self.out = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11, stride=1, dilation=1, padding=0, bias=False, groups=1, heads=-1, separable=False, normalization='batch', norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels
        if separable:
            layers = [MaskedConv1d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, groups=in_channels, heads=heads, use_mask=self.conv_mask), MaskedConv1d(in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=bias, groups=groups, use_mask=self.conv_mask)]
        else:
            layers = [MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, groups=groups, use_mask=self.conv_mask)]
        if normalization == 'group':
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == 'instance':
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == 'layer':
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == 'batch':
            layers.append(nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1))
        else:
            raise ValueError(f'Normalization method ({normalization}) does not match one of [batch, layer, group, instance].')
        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_):
        xs, lens_orig = input_
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)
                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens
        return [out], lens


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, nn.Conv1d) or isinstance(m, MaskedConv1d):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            raise ValueError('Unknown Initialization mode: {0}'.format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


jasper_activations = {'hardtanh': nn.Hardtanh, 'relu': nn.ReLU, 'selu': nn.SELU}


class JasperEncoderDecoder(nn.Module):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu"].
        feat_in (int): Number of channels being input to this module
        num_classes (int): number of vocab including the blank character
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add" or "max".
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(self, jasper, activation, feat_in, num_classes, normalization_mode='batch', residual_mode='add', norm_groups=-1, conv_mask=True, frame_splicing=1, init_mode='xavier_uniform', **kwargs):
        super(JasperEncoderDecoder, self).__init__()
        activation = jasper_activations[activation]()
        feat_in = feat_in * frame_splicing
        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            tied = lcfg.get('tied', False)
            heads = lcfg.get('heads', -1)
            encoder_layers.append(JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'], kernel_size=lcfg['kernel'], stride=lcfg['stride'], dilation=lcfg['dilation'], dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0, residual=lcfg['residual'], groups=groups, separable=separable, heads=heads, residual_mode=residual_mode, normalization=normalization_mode, norm_groups=norm_groups, tied=tied, activation=activation, residual_panes=dense_res, conv_mask=conv_mask))
            feat_in = lcfg['filters']
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(nn.Conv1d(1024, num_classes, kernel_size=1, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length):
        s_input, length = self.encoder(([audio_signal], length))
        return self.decoder_layers(s_input[-1]), length

    def load_nvidia_nemo_weights(self, encoder_weight_path, decoder_weight_path, map_location='cpu'):
        import torch
        encoder_weight = torch.load(encoder_weight_path, map_location=map_location)
        new_encoder_weight = {}
        for k, v in encoder_weight.items():
            k = k.replace('mconv', 'conv')
            if len(v.shape) == 3:
                k = k.replace('.conv.weight', '.weight')
            new_encoder_weight[k] = v
        if decoder_weight_path:
            decoder_weight = torch.load(decoder_weight_path, map_location=map_location)
            encoder_weight.update(decoder_weight)
        self.load_state_dict(new_encoder_weight, strict=False)


class TinyJasper(nn.Module):

    def __init__(self, vocab):
        super(TinyJasper, self).__init__()
        self.first_layer = C(64, 256, 11, stride=2, dropout_rate=0.2)
        self.B1 = nn.Sequential(C(256, 256, 11, dropout_rate=0.2), C(256, 256, 11, dropout_rate=0.2), C(256, 256, 11, dropout_rate=0.2))
        self.B2 = nn.Sequential(C(256, 384, 13, dropout_rate=0.2), C(384, 384, 13, dropout_rate=0.2), C(384, 384, 13, dropout_rate=0.2))
        self.r2 = nn.Conv1d(256, 384, 1)
        self.B3 = nn.Sequential(C(384, 512, 17, dropout_rate=0.2), C(512, 512, 17, dropout_rate=0.2), C(512, 512, 17, dropout_rate=0.2))
        self.r3 = nn.Conv1d(384, 512, 1)
        self.B4 = nn.Sequential(C(512, 640, 21, dropout_rate=0.3))
        self.B5 = nn.Sequential(C(640, 768, 25, dropout_rate=0.3))
        self.r4_5 = nn.Conv1d(512, 768, 1)
        self.last_layer = nn.Sequential(C(768, 896, 29, dropout_rate=0.4, dilation=2), C(896, 1024, 1, dropout_rate=0.4), C(1024, len(vocab), 1))

    def forward(self, x):
        y = self.first_layer(x)
        y = self.B1(y) + y
        y = self.B2(y) + self.r2(y)
        y = self.B3(y) + self.r3(y)
        y = self.B5(self.B4(y)) + self.r4_5(y)
        y = self.last_layer(y)
        return y


class TinyWav2Letter(nn.Module):

    def __init__(self, vocab):
        super(TinyWav2Letter, self).__init__()
        self.first_layer = C(64, 256, 11, stride=2, dropout_rate=0.2)
        self.B1 = nn.Sequential(C(256, 256, 11, dropout_rate=0.2), C(256, 256, 11, dropout_rate=0.2), C(256, 256, 11, dropout_rate=0.2))
        self.B2 = nn.Sequential(C(256, 384, 13, dropout_rate=0.2), C(384, 384, 13, dropout_rate=0.2), C(384, 384, 13, dropout_rate=0.2))
        self.B3 = nn.Sequential(C(384, 512, 17, dropout_rate=0.2), C(512, 512, 17, dropout_rate=0.2), C(512, 512, 17, dropout_rate=0.2))
        self.B4 = nn.Sequential(C(512, 640, 21, dropout_rate=0.3))
        self.B5 = nn.Sequential()
        self.last_layer = nn.Sequential(C(640, 896, 29, dropout_rate=0.3, dilation=1), C(896, 1024, 1, dropout_rate=0.4), C(1024, len(vocab), 1))

    def forward(self, x):
        y = self.first_layer(x)
        y = self.B1(y)
        y = self.B2(y)
        y = self.B3(y)
        y = self.B4(y)
        y = self.B5(y)
        y = self.last_layer(y)
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (C,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GroupShuffle,
     lambda: ([], {'groups': 1, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tugstugi_mongolian_speech_recognition(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

