import sys
_module = sys.modules[__name__]
del sys
asr = _module
datasets = _module
_common = _module
aspire = _module
swbd = _module
tedlium = _module
kaldi = _module
_path = _module
latgen = _module
_latgen = _module
prep_ctc_trans = _module
ctc_token_fst = _module
setup = _module
models = _module
capsule1 = _module
model = _module
network = _module
predict = _module
train = _module
capsule2 = _module
model = _module
network = _module
train = _module
convnet = _module
model = _module
network = _module
predict = _module
train = _module
deepspeech_ce = _module
network = _module
predict = _module
train = _module
deepspeech_ctc = _module
network = _module
train = _module
deepspeech_var = _module
network = _module
train = _module
densenet = _module
model = _module
network = _module
predict = _module
train = _module
densenet_ctc = _module
network = _module
predict = _module
train = _module
distributed = _module
las = _module
loss = _module
network = _module
train = _module
predictor = _module
resnet_ce = _module
network = _module
predict = _module
train = _module
resnet_ctc = _module
network = _module
predict = _module
train = _module
resnet_split = _module
network = _module
predict = _module
train = _module
resnet_split_ce = _module
network = _module
predict = _module
train = _module
model = _module
network = _module
predict = _module
train = _module
trainer = _module
utils = _module
adamw = _module
dataloader = _module
dataset = _module
kaldi_io = _module
logger = _module
lr_scheduler = _module
misc = _module
mnist = _module
params = _module
plot = _module
batch_train = _module
prepare = _module
test = _module

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


import random


import numpy as np


import torch


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import torchvision.utils as tvu


import math


from collections import OrderedDict


from torch.nn.parameter import Parameter


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.distributed import DistributedSampler


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import torch.distributed as dist


from torch.nn.modules import Module


from torch.nn.modules.loss import _Loss


import torchvision.utils as vutils


from torchvision.models.densenet import _DenseLayer


from torchvision.models.densenet import _DenseBlock


from torch.optim.optimizer import Optimizer


from torch.utils.data import DataLoader


import torchaudio


import scipy.io.wavfile


from scipy.signal import tukey


from torch._C import _set_worker_signal_handlers


from torch.utils.data import Dataset


from torch.utils.data import Subset


import logging


import time


from torch.optim.lr_scheduler import _LRScheduler


from functools import reduce


from torchvision.datasets import MNIST


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(21, 3), stride=(2, 1))
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=16, kernel_size=(21, 3), stride=(4, 2))
        self.digit_capsules = CapsuleLayer(num_capsules=p.NUM_LABELS, num_route_nodes=16 * 9 * 9, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(nn.Linear(16 * p.NUM_LABELS, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 5418), nn.Sigmoid())

    def forward(self, x, y=None):
        x = x.view(-1, 2, 129, 21)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(p.NUM_LABELS)).index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, (None)]).view(x.size(0), -1))
        return classes, reconstructions


logger = logging.getLogger('pytorch-asr')


class ConvCapsule(nn.Module):

    def __init__(self, in_channel, in_dim, out_channel, out_dim, kernel_size, stride, routing=0):
        super(ConvCapsule, self).__init__()
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.out_dim = out_dim
        self.routing = routing
        self.kernel_size = kernel_size
        self.stride = stride
        if self.routing:
            self.routing_capsule = nn.Conv2d(in_channels=kernel_size * kernel_size * in_dim * in_channel, out_channels=kernel_size * kernel_size * out_dim * in_channel * out_channel, kernel_size=1, stride=1, groups=kernel_size * kernel_size * in_channel)
        else:
            self.no_routing_capsule = nn.Conv2d(in_channels=in_channel * (in_dim + 1), out_channels=out_channel * (out_dim + 1), kernel_size=kernel_size, stride=stride)

    def squash(self, tensor):
        size = tensor.size()
        if len(tensor.size()) < 5:
            tensor = torch.stack(tensor.split(self.out_dim, dim=1), dim=1)
        squared_norm = (tensor ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        outputs = scale * tensor / torch.sqrt(squared_norm)
        return outputs.view(size)

    def down_h(self, h):
        return range(h * self.stride, h * self.stride + self.kernel_size)

    def down_w(self, w):
        return range(w * self.stride, w * self.stride + self.kernel_size)

    def EM_routing(self, votes, activations):
        R = 1.0 / self.out_channel * Variable(torch.ones(self.batches, self.in_channel, self.kernel_size, self.kernel_size, self.out_channel, self.out_h, self.out_w), requires_grad=False)
        votes_reshape = votes.view(self.batches, self.in_channel, self.kernel_size, self.kernel_size, self.out_channel, self.out_dim, self.out_h, self.out_w)
        activations = activations.squeeze(dim=2)
        a_reshape = [activations[:, :, :, (self.down_w(w))][:, :, (self.down_h(h)), :] for h in range(self.out_h) for w in range(self.out_w)]
        a_stack = torch.stack(a_reshape, dim=4).view(self.batches, self.in_channel, self.kernel_size, self.kernel_size, self.out_h, self.out_w)
        for _ in range(self.routing):
            r_hat = R * a_stack[:, :, :, :, (None), :, :]
            sum_r_hat = r_hat.sum(3).sum(2).sum(1)
            u_h = torch.sum(r_hat[:, :, :, :, :, (None), :, :] * votes_reshape, dim=3).sum(2).sum(1) / sum_r_hat[:, :, (None), :, :]
            sigma_h_square = torch.sum(r_hat[:, :, :, :, :, (None), :, :] * (votes_reshape - u_h[:, (None), (None), (None), :, :, :, :]) ** 2, dim=3).sum(2).sum(1) / sum_r_hat[:, :, (None), :, :]
            cost_h = (self.beta_v[(None), :, (None), :, :] + torch.log(torch.sqrt(sigma_h_square))) * sum_r_hat[:, :, (None), :, :]
            a_hat = torch.sigmoid(self.lamda * (self.beta_a[(None), :, :, :] - cost_h.sum(2)))
            sigma_product = Variable(torch.ones(self.batches, self.out_channel, self.out_h, self.out_w), requires_grad=False)
            for dm in range(self.out_dim):
                sigma_product = sigma_product * 2 * 3.1416 * sigma_h_square[:, :, (dm), :, :]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:, (None), (None), (None), :, :, :, :]) ** 2 / (2 * sigma_h_square[:, (None), (None), (None), :, :, :, :]), dim=5) / torch.sqrt(sigma_product[:, (None), (None), (None), :, :, :]))
            R = a_hat[:, (None), (None), (None), :, :, :] * p_c / torch.sum(a_hat[:, (None), (None), (None), :, :, :] * p_c, dim=6, keepdim=True).sum(dim=5, keepdim=True).sum(dim=4, keepdim=True)
        return a_hat, u_h

    def forward(self, x, lamda=0):
        if self.routing:
            size = x.size()
            self.batches = size[0]
            out_h = int((size[2] - self.kernel_size) / self.stride) + 1
            out_w = int((size[3] - self.kernel_size) / self.stride) + 1
            self.out_h = out_h
            self.out_w = out_w
            try:
                self.beta_v
            except AttributeError:
                self.beta_v = Variable(torch.randn(self.out_channel, self.out_h, self.out_w))
                self.beta_a = Variable(torch.randn(self.out_channel, self.out_h, self.out_w))
            self.lamda = lamda
            x_reshape = x.view(size[0], self.in_channel, 1 + self.in_dim, size[2], size[3])
            activations = x_reshape[:, :, (0), :, :]
            vector = x_reshape[:, :, 1:, :, :].contiguous().view(size[0], -1, size[2], size[3])
            maps = []
            for k_h in range(self.kernel_size):
                for k_w in range(self.kernel_size):
                    onemap = [vector[:, :, (k_h + i), (k_w + j)] for i in range(0, out_h * self.stride, self.stride) for j in range(0, out_w * self.stride, self.stride)]
                    onemap = torch.stack(onemap, dim=2)
                    onemap = onemap.view(size[0], onemap.size(1), out_h, out_w)
                    maps.append(onemap)
            map_ = torch.cat(maps, dim=1)
            votes = self.routing_capsule(map_)
            output_a, output_v = self.EM_routing(votes, activations)
            outputs = torch.cat([output_a[:, :, (None), :, :], output_v], dim=2)
            return outputs.view(self.batches, self.out_channel * (self.out_dim + 1), self.out_h, self.out_w)
        else:
            outputs = self.no_routing_capsule(x)
            return outputs


COORDINATE_SCALE = 10.0


class ClassCapsule(nn.Module):

    def __init__(self, in_channel, in_dim, classes, out_dim, routing):
        super(ClassCapsule, self).__init__()
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classes = classes
        self.routing = routing
        self.beta_v = Variable(torch.randn(self.classes))
        self.beta_a = Variable(torch.randn(self.classes))
        self.capsules = nn.Conv2d(in_channels=in_channel * in_dim, out_channels=in_channel * out_dim * classes, kernel_size=1, stride=1, groups=in_channel)

    def EM_routing(self, votes, activations):
        R = 1.0 / self.classes * Variable(torch.ones(self.batches, self.in_channel, self.classes, self.h, self.w), requires_grad=False)
        activations = activations.squeeze(dim=2)
        votes_reshape = votes.view(self.batches, self.in_channel, self.classes, self.out_dim, self.h, self.w)
        for _ in range(self.routing):
            r_hat = R * activations[:, :, (None), :, :]
            sum_r_hat = r_hat.sum(4).sum(3).sum(1)
            u_h = torch.sum(r_hat[:, :, :, (None), :, :] * votes_reshape, dim=5).sum(4).sum(1) / sum_r_hat[:, :, (None)]
            sigma_h_square = torch.sum(r_hat[:, :, :, (None), :, :] * (votes_reshape - u_h[:, (None), :, :, (None), (None)]) ** 2, dim=5).sum(4).sum(1) / sum_r_hat[:, :, (None)]
            cost_h = (self.beta_v[(None), :, (None)] + torch.log(sigma_h_square)) * sum_r_hat[:, :, (None)]
            a_hat = torch.sigmoid(self.lamda * (self.beta_a[(None), :] - torch.sum(cost_h, dim=2)))
            sigma_product = Variable(torch.ones(self.batches, self.classes), requires_grad=False)
            for dm in range(self.out_dim):
                sigma_product = 2 * 3.1416 * sigma_product * sigma_h_square[:, :, (dm)]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:, (None), :, :, (None), (None)]) ** 2 / (2 * sigma_h_square[:, (None), :, :, (None), (None)]), dim=3)) / torch.sqrt(sigma_product[:, (None), :, (None), (None)])
            R = a_hat[:, (None), :, (None), (None)] * p_c / torch.sum(a_hat[:, (None), :, (None), (None)] * p_c, dim=2, keepdim=True)
        return a_hat, u_h

    def CoordinateAddition(self, vector):
        output = Variable(torch.zeros(vector.size()))
        coordinate_x = Variable(torch.FloatTensor(torch.arange(0, self.h)) / COORDINATE_SCALE, requires_grad=False)
        coordinate_y = Variable(torch.FloatTensor(torch.arange(0, self.w)) / COORDINATE_SCALE, requires_grad=False)
        output[:, :, (0), :, :] = vector[:, :, (0), :, :] + coordinate_x[(None), (None), :, (None)]
        output[:, :, (1), :, :] = vector[:, :, (1), :, :] + coordinate_y[(None), (None), (None), :]
        if output.size(2) > 2:
            output[:, :, 2:, :, :] = vector[:, :, 2:, :, :]
        return output

    def forward(self, x, lamda=0):
        self.lamda = lamda
        size = x.size()
        self.batches = size[0]
        self.h = size[2]
        self.w = size[3]
        x_reshape = x.view(size[0], self.in_channel, 1 + self.in_dim, size[2], size[3])
        activations = x_reshape[:, :, (0), :, :]
        vector = x_reshape[:, :, 1:, :, :]
        vec = self.CoordinateAddition(vector)
        vec = vec.view(size[0], -1, size[2], size[3])
        votes = self.capsules(vec)
        output_a, output_v = self.EM_routing(votes, activations)
        return output_a


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class View(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


def get_model_file_path(log_dir, prefix, desc):
    path = Path(log_dir).resolve()
    return path / f'{prefix}_{desc}.{params.MODEL_SUFFIX}'


class MultiOut(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return (m.forward(*args, **kwargs) for m in self)


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x, *args, **kwargs)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        if self.training:
            return x
        else:
            return self.logsoftmax(x)


class TemporalRowConvolution(nn.Module):

    def __init__(self, input_size, kernel_size, stride=1, padding=0, feat_first=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.weight = nn.Parameter(torch.Tensor(input_size, 1, *kernal_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.register_parameter('bias', None)
        self.feat_first = feat_first
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.kernel_size * self.input_size)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_):
        return nn._functions.thnn.auto.TemporalRowConvolution.apply(input_, kernel_size, stride, padding)


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_first=True, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=batch_first, bias=True)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, seq_lens):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        ps = nn.utils.rnn.pack_padded_sequence(x, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnn(ps)
        x, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class Lookahead(nn.Module):

    def __init__(self, n_features, context):
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        padding = torch.zeros(self.context, *input.size()[1:]).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)
        x = [x[i:i + self.context + 1] for i in range(seq_len)]
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n_features=' + str(self.n_features) + ', context=' + str(self.context) + ')'


class LSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :3 * self.hidden_size].sigmoid().chunk(3, 1)
        g = gates[:, 3 * self.hidden_size:].tanh()
        cy = f * cx + i * g
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, use_layernorm=False, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([LSTMCell(input_size=input_size if layer == 0 else hidden_size * num_directions, hidden_size=hidden_size, bias=bias, use_layernorm=use_layernorm) for layer in range(num_layers)])
        if self.bidirectional:
            self.hidden1 = nn.ModuleList([LSTMCell(input_size=input_size if layer == 0 else hidden_size * num_directions, hidden_size=hidden_size, bias=bias, use_layernorm=use_layernorm) for layer in range(num_layers)])

    def forward(self, input, hidden=None):
        seq_len, batch_size, hidden_size = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        ht = [[None] * (self.num_layers * num_directions)] * seq_len
        ct = [[None] * (self.num_layers * num_directions)] * seq_len
        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        return y, (hy, cy)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('swish', Swish())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))


class DenseNet(nn.Module):
    """ Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0.5, num_classes=1000):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(2, num_init_features, kernel_size=(41, 11), stride=(2, 2), padding=(0, 5)), nn.BatchNorm2d(num_init_features), Swish(inplace=True), nn.Conv2d(num_init_features, num_init_features, kernel_size=(21, 11), stride=(1, 1), padding=(0, 5)), nn.BatchNorm2d(num_init_features), Swish(inplace=True))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.hidden.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.hidden.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        self.hidden.add_module('norm_f', nn.BatchNorm2d(num_features))
        self.hidden.add_module('relu_f', Swish())
        self.fc = nn.Sequential(nn.Linear(131 * 4, 512), nn.Dropout(p=0.5, inplace=True), nn.Linear(512, num_classes))
        self.softmax = InferenceBatchSoftmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.hidden(x)
        x = x.transpose(2, 3).transpose(1, 2)
        x = self.fc(x.view(x.size(0), x.size(1), -1))
        x = self.softmax(x)
        return x


def onehot2int(onehot, dim=-1, keepdim=False):
    _, idx = onehot.topk(k=1, dim=dim)
    if idx.dim() == 0:
        return int(idx)
    else:
        return idx if keepdim else idx.squeeze(dim=dim)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('swish.1', Swish()),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('swish.2', Swish()),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i + 1}', layer)


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.first_call = True

        def allreduce_params():
            if self.needs_reduction:
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)
        for param in list(self.module.parameters()):

            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def weight_broadcast(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)
    """
        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)
    """

    def forward(self, *inputs, **kwargs):
        if self.first_call:
            None
            self.weight_broadcast()
            self.first_call = False
            None
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)


class EditDistanceLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, input_seq_lens, target_seq_lens):
        """
        input: BxTxH, target: BxN, input_seq_lens: B, target_seq_lens: B
        """
        batch_size = input.size(0)
        eds = list()
        for b in range(batch_size):
            x = torch.argmax(input[(b), :input_seq_lens[b]], dim=-1)
            y = target[(b), :target_seq_lens[b]]
            d = self.calculate_levenshtein(x, y)
            eds.append(d)
        loss = torch.FloatTensor(eds)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()

    def calculate_levenshtein(self, seq1, seq2):
        """
        implement the extension of the Wagner–Fischer dynamic programming algorithm
        """
        size_x, size_y = len(seq1), len(seq2)
        matrix = torch.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y
        for x in range(1, size_x):
            for y in range(1, size_y):
                cost = 0 if seq1[x] == seq2[y] else 1
                comps = torch.LongTensor([matrix[x - 1, y] + 1, matrix[x, y - 1] + 1, matrix[x - 1, y - 1] + cost])
                matrix[x, y] = torch.min(comps)
                if x > 1 and y > 1 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y]:
                    comps = torch.LongTensor([matrix[x, y], matrix[x - 2, y - 2] + cost])
                    matrix[x, y] = torch.min(comps)
        return matrix[-1, -1]


class Listener(nn.Module):

    def __init__(self, listen_vec_size, input_folding=3, rnn_type=nn.LSTM, rnn_hidden_size=256, rnn_num_layers=4, bidirectional=True, last_fc=False):
        super().__init__()
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional
        W0 = 129
        C0 = 2 * input_folding
        W1 = (W0 - 3 + 2 * 1) // 2 + 1
        C1 = 64
        W2 = (W1 - 3 + 2 * 1) // 2 + 1
        C2 = C1 * 2
        W3 = (W2 - 3 + 2 * 1) // 2 + 1
        C3 = C2 * 2
        H0 = C3 * W3
        self.feature = nn.Sequential(OrderedDict([('cv1', nn.Conv2d(C0, C1, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)), ('nl1', nn.LeakyReLU()), ('mp1', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))), ('bn1', nn.BatchNorm2d(C1)), ('cv2', nn.Conv2d(C1, C2, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)), ('nl2', nn.LeakyReLU()), ('mp2', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))), ('bn2', nn.BatchNorm2d(C2)), ('cv3', nn.Conv2d(C2, C3, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)), ('nl3', nn.LeakyReLU()), ('mp3', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))), ('bn3', nn.BatchNorm2d(C3))]))
        self.batch_first = True
        self.rnns = rnn_type(input_size=H0, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bias=True, bidirectional=bidirectional, batch_first=self.batch_first)
        if last_fc:
            self.fc = SequenceWise(nn.Sequential(OrderedDict([('fc1', nn.Linear(rnn_hidden_size, listen_vec_size, bias=False)), ('nl1', nn.LeakyReLU()), ('ln1', nn.LayerNorm(listen_vec_size, elementwise_affine=False))])))
        else:
            assert listen_vec_size == rnn_hidden_size
            self.fc = None

    def forward(self, x, seq_lens):
        h = self.feature(x)
        h = h.view(-1, h.size(1) * h.size(2), h.size(3))
        y = h.transpose(1, 2).contiguous()
        ps = nn.utils.rnn.pack_padded_sequence(y, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnns(ps)
        y, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            y = y.view(y.size(0), y.size(1), 2, -1).sum(2).view(y.size(0), y.size(1), -1)
        if self.fc is not None:
            y = self.fc(y)
        return y


class MaskedSoftmax(nn.Module):

    def __init__(self, dim=-1, epsilon=1e-05):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, e, mask=None):
        if mask is None:
            return self.softmax(e)
        else:
            shift_e = e - e.max()
            exps = torch.exp(shift_e) * mask
            sums = exps.sum(dim=self.dim, keepdim=True) + self.epsilon
            return exps / sums


class Attention(nn.Module):

    def __init__(self, state_vec_size, listen_vec_size, apply_proj=True, proj_hidden_size=256, num_heads=1):
        super().__init__()
        self.apply_proj = apply_proj
        self.num_heads = num_heads
        if apply_proj:
            self.phi = nn.Linear(state_vec_size, proj_hidden_size * num_heads, bias=True)
            self.psi = nn.Linear(listen_vec_size, proj_hidden_size, bias=False)
        else:
            assert state_vec_size == listen_vec_size * num_heads
        if num_heads > 1:
            input_size = listen_vec_size * num_heads
            self.reduce = nn.Linear(input_size, listen_vec_size, bias=True)
        self.normal = SequenceWise(MaskedSoftmax(dim=-1))

    def score(self, m, n):
        """ dot product as score function """
        return torch.bmm(m, n.transpose(1, 2))

    def forward(self, s, h, len_mask=None):
        if self.apply_proj:
            m = self.phi(s)
            n = self.psi(h)
        else:
            m = s
            n = h
        if self.num_heads > 1:
            proj_hidden_size = m.size(-1) // self.num_heads
            ee = [self.score(mi, n) for mi in torch.split(m, proj_hidden_size, dim=-1)]
            aa = [self.normal(e, len_mask) for e in ee]
            c = self.reduce(torch.cat([torch.bmm(a, h) for a in aa], dim=-1))
            a = torch.stack(aa).transpose(0, 1)
        else:
            e = self.score(m, n)
            a = self.normal(e, len_mask)
            c = torch.bmm(a, h)
            a = a.unsqueeze(dim=1)
        return c, a


def merge_last(x, n_dims):
    """merge the last n_dims to a dimension"""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def split_last(x, shape):
    """split the last dimension to given shape"""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, state_vec_size, listen_vec_size, proj_hidden_size=512, num_heads=1, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(state_vec_size, proj_hidden_size)
        self.proj_k = nn.Linear(listen_vec_size, proj_hidden_size)
        self.proj_v = nn.Linear(listen_vec_size, proj_hidden_size)
        self.drop = nn.Dropout(dropout)
        self.scores = None
        self.n_heads = num_heads

    def forward(self, q, k, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(k)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, (None), (None), :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


def int2onehot(idx, num_classes, floor=0.0):
    if not torch.is_tensor(idx):
        onehot = torch.full((1, num_classes), 0.0, dtype=torch.float)
        idx = torch.LongTensor([idx])
        onehot.scatter_(1, idx.unsqueeze(0), 1.0)
    else:
        sizes = idx.size()
        onehot = idx.new_full((idx.numel(), num_classes), 0.0, dtype=torch.float)
        onehot.scatter_(1, idx.view(-1).long().unsqueeze(1), 1.0)
        onehot = onehot.view(*sizes, -1)
    onehot = (1.0 - floor) * onehot + floor / onehot.size(-1)
    return onehot


class Speller(nn.Module):

    def __init__(self, listen_vec_size, label_vec_size, max_seq_lens=256, sos=None, eos=None, rnn_type=nn.LSTM, rnn_hidden_size=512, rnn_num_layers=2, proj_hidden_size=256, num_attend_heads=1, masked_attend=True):
        super().__init__()
        assert sos is not None and 0 <= sos < label_vec_size
        assert eos is not None and 0 <= eos < label_vec_size
        assert sos is not None and eos is not None and sos != eos
        self.label_vec_size = label_vec_size
        self.sos = label_vec_size - 2 if sos is None else sos
        self.eos = label_vec_size - 1 if eos is None else eos
        self.max_seq_lens = max_seq_lens
        self.num_eos = 3
        self.tfr = 1.0
        Hs, Hc, Hy = rnn_hidden_size, listen_vec_size, label_vec_size
        self.rnn_num_layers = rnn_num_layers
        self.rnns = rnn_type(input_size=Hy + Hc, hidden_size=Hs, num_layers=rnn_num_layers, bias=True, bidirectional=False, batch_first=True)
        self.norm = nn.LayerNorm(Hs, elementwise_affine=False)
        self.attention = Attention(state_vec_size=Hs, listen_vec_size=Hc, proj_hidden_size=proj_hidden_size, num_heads=num_attend_heads)
        self.masked_attend = masked_attend
        self.chardist = nn.Sequential(OrderedDict([('fc1', nn.Linear(Hs + Hc, 128, bias=True)), ('fc2', nn.Linear(128, label_vec_size, bias=False))]))
        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, h, seq_lens):
        bs, ts, hs = h.size()
        mask = h.new_ones((bs, ts), dtype=torch.float)
        for b in range(bs):
            mask[(b), seq_lens[b]:] = 0.0
        return mask

    def _is_sample_step(self):
        return np.random.random_sample() < self.tfr

    def forward(self, h, x_seq_lens, y=None, y_seq_lens=None):
        batch_size = h.size(0)
        sos = int2onehot(h.new_full((batch_size, 1), self.sos), num_classes=self.label_vec_size).float()
        eos = int2onehot(h.new_full((batch_size, 1), self.eos), num_classes=self.label_vec_size).float()
        hidden = None
        y_hats = list()
        attentions = list()
        in_mask = self.get_mask(h, x_seq_lens) if self.masked_attend else None
        x = torch.cat([sos, h.narrow(1, 0, 1)], dim=-1)
        y_hats_seq_lens = torch.ones((batch_size,), dtype=torch.int) * self.max_seq_lens
        bi = torch.zeros((self.num_eos, batch_size)).byte()
        if x.is_cuda:
            bi = bi
        for t in range(self.max_seq_lens):
            s, hidden = self.rnns(x, hidden)
            s = self.norm(s)
            c, a = self.attention(s, h, in_mask)
            y_hat = self.chardist(torch.cat([s, c], dim=-1))
            y_hat = self.softmax(y_hat)
            y_hats.append(y_hat)
            attentions.append(a)
            bi[t % self.num_eos] = onehot2int(y_hat.squeeze()).eq(self.eos)
            ri = y_hats_seq_lens.gt(t)
            if bi.is_cuda:
                ri = ri
            y_hats_seq_lens[bi.prod(dim=0, dtype=torch.uint8) * ri] = t + 1
            if y_hats_seq_lens.le(t + 1).all():
                break
            if y is None or not self._is_sample_step():
                x = torch.cat([y_hat, c], dim=-1)
            elif t < y.size(1):
                x = torch.cat([y.narrow(1, t, 1), c], dim=-1)
            else:
                x = torch.cat([eos, c], dim=-1)
        y_hats = torch.cat(y_hats, dim=1)
        attentions = torch.cat(attentions, dim=2)
        return y_hats, y_hats_seq_lens, attentions


class LogWithLabelSmoothing(nn.Module):

    def __init__(self, floor=0.01):
        super().__init__()
        self.floor = floor

    def forward(self, x):
        y = (1.0 - self.floor) * x + self.floor / x.size(-1)
        return y.log()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu2 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = Swish(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.relu3 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out


HEIGHT = 129


WIDTH = 31


class ResNet(nn.Module):

    def __init__(self, block, layers, input_folding=1, num_classes=1000):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_folding * 2, self.inplanes, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)), nn.BatchNorm2d(self.inplanes), Swish(inplace=True), nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5)), nn.BatchNorm2d(self.inplanes), Swish(inplace=True))
        img_size = np.array([HEIGHT, WIDTH])
        img_size = (img_size - np.array([41, 11]) + 2 * np.array([20, 5])) // np.array([2, 2]) + 1
        img_size = (img_size - np.array([21, 11]) + 2 * np.array([10, 5])) // np.array([2, 2]) + 1
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, self.inplanes, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, self.inplanes, layers[3], stride=(2, 2))
        img_size = (img_size - 3 + 2) // 2 + 1
        img_size = (img_size - 3 + 2) // 2 + 1
        img_size = (img_size - 3 + 2) // 2 + 1
        self.fc1 = nn.Linear(self.inplanes * np.prod(img_size), 1024)
        self.do1 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)
        self.softmax = InferenceBatchSoftmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.do1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = x.size()
        return x.view(x.size(0), -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'state_vec_size': 4, 'listen_vec_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (CapsuleLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (ConvCapsule,
     lambda: ([], {'in_channel': 4, 'in_dim': 4, 'out_channel': 4, 'out_dim': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 20, 64, 64])], {}),
     False),
    (DistributedDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InferenceBatchSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (LogWithLabelSmoothing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lookahead,
     lambda: ([], {'n_features': 4, 'context': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MaskedSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceWise,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jinserk_pytorch_asr(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

