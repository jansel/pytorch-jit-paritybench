import sys
_module = sys.modules[__name__]
del sys
display_samples = _module
download_mnist = _module
convert = _module
main = _module
datautils = _module
multigpu = _module
multigpu_torchrun = _module
multinode = _module
single_gpu = _module
example = _module
main = _module
char_dataset = _module
main = _module
model = _module
trainer = _module
parameter_server = _module
reinforce = _module
main = _module
rpc_parameter_server = _module
main = _module
main = _module
main = _module
rnn = _module
example = _module
conf = _module
download_saved_models = _module
neural_style = _module
neural_style = _module
transformer_net = _module
utils = _module
vgg = _module
custom_tracer = _module
inline_function = _module
invert = _module
module_tracer = _module
use_interpreter = _module
primitive_library = _module
profiling_tracer = _module
proxy_based_graph_creation = _module
replace_op = _module
subgraph_rewriter_basic_use = _module
wrap_output_dynamically = _module
main = _module
model = _module
train = _module
util = _module
main = _module
main = _module
train = _module
main = _module
main = _module
actor_critic = _module
reinforce = _module
main = _module
data = _module
dataset = _module
main = _module
model = _module
super_resolve = _module
generate_sine_wave = _module
train = _module
main = _module
data = _module
generate = _module
main = _module
model = _module

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


import matplotlib.pyplot as plt


import torch


from torchvision import models


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


from torch.utils.data import Dataset


import torch.nn.functional as F


from torch.utils.data import DataLoader


import torch.multiprocessing as mp


from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.distributed import init_process_group


from torch.distributed import destroy_process_group


import torch.distributed as dist


from torch.utils.data import random_split


import math


from torch.nn import functional as F


from collections import OrderedDict


from typing import Optional


from typing import Any


from typing import Dict


import torch.distributed.rpc as rpc


from torch import optim


import torchvision


import time


from torch.distributed.rpc import RRef


from torch.distributed.rpc import rpc_sync


from torch.distributed.rpc import rpc_async


from torch.distributed.rpc import remote


from torch.distributions import Categorical


import torch.distributed.autograd as dist_autograd


from torch.distributed.nn import RemoteModule


from torch.distributed.optim import DistributedOptimizer


from torch.distributed.rpc import TensorPipeRpcBackendOptions


from torchvision import datasets


from torchvision import transforms


from functools import wraps


from torchvision.models.resnet import Bottleneck


import numpy as np


from itertools import count


import re


from torch.optim import Adam


import torch.onnx


from collections import namedtuple


from torch.fx import symbolic_trace


from torch.fx import Tracer


from torch.fx import Graph


from torch.fx import GraphModule


from torch.fx import Node


from typing import Callable


from typing import Tuple


from typing import Union


from torch.fx import Proxy


from torch.fx.node import map_arg


import torch.fx as fx


import torch.fx


import torchvision.models as models


from torch.fx import replace_pattern


from enum import Enum


from enum import auto


import warnings


import torch.optim


import torch.utils.data.distributed


import torchvision.datasets as datasets


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import Subset


import torch.optim as O


from torch.utils.data.sampler import Sampler


from collections import deque


import copy


from torchvision import transforms as T


import torch.utils.data as data


from math import log10


import torch.nn.init as init


from torchvision.transforms import ToTensor


import matplotlib


from torch import nn


from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')


opt = parser.parse_args()


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class ToyMpModel(nn.Module):

    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = x
        x = self.relu(self.net1(x))
        x = x
        return self.net2(x)


class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config, device='cpu', dtype=torch.float32):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.attn = torch.nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, dropout=config.attn_pdrop, batch_first=True, device=device, dtype=dtype)

    def forward(self, x):
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class HybridModel(torch.nn.Module):
    """
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup)


class Net(nn.Module):

    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


class ParameterServer(nn.Module):

    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        self.input_device = torch.device('cuda:0' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')

    def forward(self, inp):
        inp = inp
        out = self.model(inp)
        out = out
        return out

    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k, v
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


def get_parameter_server(num_gpus=0):
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class TrainerNet(nn.Module):

    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote('parameter_server', get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(ParameterServer.get_param_rrefs, self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(ParameterServer.forward, self.param_server_rref, x)
        return model_output


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBase(nn.Module):

    def __init__(self, block, inplanes, num_classes=1000, groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()
        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * self._block.expansion, stride), norm_layer(planes * self._block.expansion))
        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        """
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


num_classes = 1000


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(Bottleneck, 64, *args, num_classes=num_classes, **kwargs)
        self.device = device
        self.seq = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False), self._norm_layer(self.inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), self._make_layer(64, 3), self._make_layer(128, 4, stride=2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            out = self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(Bottleneck, 512, *args, num_classes=num_classes, **kwargs)
        self.device = device
        self.seq = nn.Sequential(self._make_layer(256, 6, stride=2), self._make_layer(512, 3, stride=2), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * self._block.expansion, num_classes)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()
        self.split_size = split_size
        self.p1_rref = rpc.remote(workers[0], ResNetShard1, args=('cuda:0',) + args, kwargs=kwargs)
        self.p2_rref = rpc.remote(workers[1], ResNetShard2, args=('cuda:1',) + args, kwargs=kwargs)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


class EmbeddingTable(nn.Module):
    """
    Encoding layers of the RNNModel
    """

    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if torch.cuda.is_available():
            self.encoder = self.encoder
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)

    def forward(self, input):
        if torch.cuda.is_available():
            input = input
        return self.drop(self.encoder(input)).cpu()


class Decoder(nn.Module):
    """
    Decoding layers of the RNNModel
    """

    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
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
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class M1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class M2(torch.nn.Module):

    def forward(self, a, b):
        return a + b


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = torch.cat([x, y])
        return y


class MyElementwiseModule(torch.nn.Module):

    def forward(self, x, y):
        return x * y + y


class Foo(torch.nn.Module):

    def forward(self, x):
        with torch.profiler.record_function('foo'):
            return torch.relu(x)


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden, num_layers=config.n_layers, dropout=dropout, bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2 * config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(Linear(*lin_config), self.relu, self.dropout, Linear(*lin_config), self.relu, self.dropout, Linear(*lin_config), self.relu, self.dropout, Linear(seq_in_size, config.d_out))

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        if self.config.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()
        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(self.fc_in_features * 2, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder
            from torch.nn import TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src))
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'ntoken': 4, 'nhid': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Foo,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (M,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (M1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (M2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttentionLayer,
     lambda: ([], {'config': _mock_config(n_embd=4, n_head=4, resid_pdrop=0.5, block_size=4, attn_pdrop=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MyElementwiseModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Policy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiameseNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])], {}),
     True),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {}),
     True),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_pytorch_examples(_paritybench_base):
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

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

