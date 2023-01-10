import sys
_module = sys.modules[__name__]
del sys
core = _module
client = _module
config = _module
dataloader = _module
dataset = _module
evaluation = _module
federated = _module
metrics = _module
model = _module
schema = _module
server = _module
strategies = _module
base = _module
dga = _module
fedavg = _module
utils = _module
trainer = _module
conf = _module
e2e_trainer = _module
experiments = _module
cifar_dataset = _module
dataloader = _module
model = _module
centralized_training = _module
download_and_convert_data = _module
data = _module
dataloader = _module
model = _module
model_vgg = _module
dataloader = _module
preprocess = _module
model = _module
dataloader = _module
preprocessing = _module
model = _module
dataloader = _module
group_normalization = _module
model = _module
dataloader = _module
model = _module
dataloader = _module
dataset = _module
preprocess_mind = _module
fednewsrec_model = _module
model = _module
dataloader = _module
model = _module
trainer_pt_utils = _module
trainer_utils = _module
dataloader = _module
model = _module
utility = _module
dataloader = _module
model = _module
RL = _module
extensions = _module
privacy = _module
analysis = _module
dp_kmeans = _module
metrics = _module
quant = _module
build_vocab = _module
create_data = _module
test_e2e_trainer = _module
data_utils = _module
dataloaders_utils = _module
adamW = _module
lamb = _module
lars = _module
from_json_to_hdf5 = _module
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


import copy


import logging


import time


import numpy as np


import torch


from collections.abc import MutableMapping


from torch.utils.data import DataLoader as PyTorchDataLoader


from abc import ABC


from torch.utils.data import Dataset as PyTorchDataset


from abc import abstractmethod


import torch.distributed as dist


import torch as T


import random


from collections import defaultdict


import math


import re


import torch.nn as nn


from torch import nn


from torch.nn import functional as F


from sklearn.metrics import f1_score


import torchvision


import torchvision.transforms as transforms


import torch.nn.functional as F


import torch.optim as optim


from torch import Tensor


from torch.utils.model_zoo import load_url as load_state_dict_from_url


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


from torch.nn.modules.batchnorm import _BatchNorm


import torch.utils.model_zoo as model_zoo


from torch.nn import CrossEntropyLoss


from sklearn.metrics import roc_auc_score


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from typing import Dict


from typing import Tuple


import warnings


from typing import Iterator


from torch.utils.data.dataset import Dataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import Sampler


from typing import NamedTuple


from collections import OrderedDict


from scipy.special import betainc


from scipy.special import betaln


from copy import deepcopy


from torch.utils.data import sampler


from torch.optim import Optimizer


import collections


import functools


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(ABC, T.nn.Module):
    """This is a wrapper class for PyTorch models."""

    @abstractmethod
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

    @abstractmethod
    def loss(self, input):
        """Performs forward step and computes the loss

        Returns:
            torch: Computed loss.
        """
        pass

    @abstractmethod
    def inference(self, input):
        """Performs forward step and computes metrics
             
        Returns:
            dict: The metrics to be computed. The following keys are
            the minimum required by FLUTE during evaluations rounds: 
                - output
                - acc
                - batch_size

            More metrics can be computed by adding the key with a
            dictionary that includes the fields ´value´ and 
            ´higher_is_better´ as follows:

            {'output':output, 
             'acc': accuracy, 
             'batch_size': n_samples, 
             'f1_score': {'value':f1,'higher_is_better': True}}
        """
        pass

    def set_eval(self):
        """Bring the model into evaluation mode"""
        self.eval()

    def set_train(self):
        """Bring the model into training mode"""
        self.train()


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""

    def __init__(self, input_size, hidden_size, kernel_size, norm_type='bachnorm'):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv_3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
            self.normalization_2 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
            self.normalization_3 = nn.GroupNorm(num_groups=8, num_channels=hidden_size)
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        x = self.pool(x)
        return x


class nlp_rnn_fedshakespeare(nn.Module):

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(nlp_rnn_fedshakespeare, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(lstm_out[:, :])
        output = torch.transpose(output, 1, 2)
        return output


class RNN(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = nlp_rnn_fedshakespeare()

    def loss(self, input: torch.Tensor) ->torch.Tensor:
        """Performs forward step and computes the loss"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, target = input['x'], input['y']
        output = self.net.forward(x)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion(output, target.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, target = input['x'], input['y']
        output = self.net.forward(x)
        n_samples = x.shape[0]
        pred = torch.argmax(output, dim=1)
        mask = target != 0
        accuracy = torch.sum((pred[mask] == target[mask]).float()).item()
        accuracy = accuracy / mask.sum()
        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}


class Net(nn.Module):

    def __init__(self, input_size=1, hid_size=64, n_classes=5, kernel_size=5):
        super().__init__()
        self.rnn_layer = RNN(input_size=46, hid_size=hid_size)
        self.conv1 = ConvNormPool(input_size=input_size, hidden_size=hid_size, kernel_size=kernel_size)
        self.conv2 = ConvNormPool(input_size=hid_size, hidden_size=hid_size, kernel_size=kernel_size)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.attn = nn.Linear(hid_size, hid_size, bias=False)
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x_out, hid_states = self.rnn_layer(x)
        x = torch.cat([hid_states[0], hid_states[1]], dim=0).transpose(0, 1)
        x_attn = torch.tanh(self.attn(x))
        x = x_attn.bmm(x_out)
        x = x.transpose(2, 1)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=-1)
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        return x


class CNN(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = CNN_DropOut(False)

    def loss(self, input: torch.Tensor) ->torch.Tensor:
        """Performs forward step and computes the loss"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        criterion = nn.CrossEntropyLoss()
        return criterion(output, labels.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def group_norm(input, group, running_mean, running_var, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05):
    """Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')
    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=None, momentum=None, eps=None):
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)
        input_reshaped = input.contiguous().view(1, int(b * c / group), group, *input.size()[2:])
        out = F.batch_norm(input_reshaped, running_mean, running_var, weight=weight, bias=bias, training=use_input_stats, momentum=momentum, eps=eps)
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c / group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c / group)).mean(0, keepdim=False))
        return out.view(b, c, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean, running_var=running_var, weight=weight, bias=bias, use_input_stats=use_input_stats, momentum=momentum, eps=eps)


class _GroupNorm(_BatchNorm):

    def __init__(self, num_features, num_groups=1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features / num_groups), eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        return group_norm(input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)


class GroupNorm2d(_GroupNorm):
    """Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


def norm2d(planes, num_channels_per_group=32):
    None
    if num_channels_per_group > 0:
        return GroupNorm2d(planes, num_channels_per_group, affine=True, track_running_stats=False)
    else:
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, group_norm=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm2d(64, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group_norm=group_norm)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm2d(planes * block.expansion, group_norm))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_norm=group_norm))
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


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, vgg, num_class, topK_results=None):
        super(VGG, self).__init__()
        self.topK_results = num_class if topK_results is None else topK_results
        self.vgg = vgg
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, num_class))
        if 0:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    m.bias.data.zero_()

    def forward(self, inputs):
        inputs = inputs['x'] if T.cuda.is_available() else inputs['x']
        x = self.vgg(inputs.view(-1, 3, 32, 32))
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x

    def loss(self, inputs):
        targets = inputs['y'] if T.cuda.is_available() else inputs['y']
        output = self(inputs)
        loss = T.nn.functional.cross_entropy(output, targets)
        return loss

    def inference(self, inputs):
        targets = inputs['y'] if T.cuda.is_available() else inputs['y']
        output = self(inputs)
        accuracy = T.mean((T.argmax(output, dim=1) == targets).float()).item()
        output = {'probabilities': output.cpu().detach().numpy(), 'predictions': np.arange(0, targets.shape[0]), 'labels': targets.cpu().numpy()}
        return {'output': output, 'val_acc': accuracy, 'batch_size': targets.shape[0]}

    def get_logit(self, inputs=None, evalis=True, logmax=False):
        data, targets = inputs
        if logmax:
            Softmax = T.nn.LogSoftmax(dim=1)
        else:
            Softmax = T.nn.Softmax(dim=1)
        data = data if T.cuda.is_available() else data
        if evalis:
            self.eval()
            with T.no_grad():
                output = self.forward(data)
                logits = Softmax(output)
        else:
            self.train()
            output = self.forward(data)
            logits = Softmax(output)
        loss = T.nn.functional.cross_entropy(output, targets)
        return logits.cpu(), targets.cpu(), loss.cpu()

    def copy_state_dict(self, state_dict):
        self.state_dict = state_dict.clone()

    def set_eval(self):
        """
        Bring the model into evaluation mode
        """
        self.eval()

    def set_train(self):
        """
        Bring the model into train mode
        """
        self.train()


class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        o = self.linear(x.view(-1, 28 * 28))
        outputs = torch.sigmoid(o)
        return outputs


class LR(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = LogisticRegression(model_config['input_dim'], model_config['output_dim'])

    def loss(self, input: torch.Tensor) ->torch.Tensor:
        """Performs forward step and computes the loss"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        criterion = nn.CrossEntropyLoss()
        return criterion(output, labels.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}


class GroupNorm3d(_GroupNorm):
    """
    Assume the data format is (B, C, D, H, W)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class RESNET(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = resnet18()

    def loss(self, input: torch.Tensor) ->torch.Tensor:
        """Performs forward step and computes the loss"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}


class SuperNet(BaseModel):
    """This is the parent of the net with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = Net()

    def loss(self, input: torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'], input['y']
        output = self.net.forward(features)
        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}


class AttentivePooling(nn.Module):

    def __init__(self, dim1: int, dim2: int):
        super(AttentivePooling, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        user_vecs = self.dropout(x)
        user_att = self.tanh(self.dense(user_vecs))
        user_att = self.dense2(user_att).squeeze(2)
        user_att = self.softmax(user_att)
        result = torch.einsum('ijk,ij->ik', user_vecs, user_att)
        return result

    def fromTensorFlow(self, tfmodel):
        keras_weights = tfmodel.layers[1].get_weights()
        self.dense.weight.data = torch.tensor(keras_weights[0]).transpose(0, 1)
        self.dense.bias.data = torch.tensor(keras_weights[1])
        keras_weights = tfmodel.layers[2].get_weights()
        self.dense2.weight.data = torch.tensor(keras_weights[0]).transpose(0, 1)
        self.dense2.bias.data = torch.tensor(keras_weights[1])


class Attention(nn.Module):

    def __init__(self, input_dim, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.WQ = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.WK = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.WV = nn.Linear(self.input_dim, self.output_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.WQ.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.WK.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.WV.weight, gain=np.sqrt(2))

    def fromTensorFlow(self, tf, criteria=lambda l: l.name.startswith('attention')):
        for l in tf.layers:
            None
            if criteria(l):
                weights = l.get_weights()
                self.WQ.weight.data = torch.tensor(weights[0].transpose())
                self.WK.weight.data = torch.tensor(weights[1].transpose())
                self.WV.weight.data = torch.tensor(weights[2].transpose())

    def forward(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        Q_seq = self.WQ(Q_seq)
        Q_seq = torch.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))
        Q_seq = torch.transpose(Q_seq, 1, 2)
        K_seq = self.WK(K_seq)
        K_seq = torch.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = torch.transpose(K_seq, 1, 2)
        V_seq = self.WV(V_seq)
        V_seq = torch.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))
        V_seq = torch.transpose(V_seq, 1, 2)
        A = torch.einsum('ijkl,ijml->ijkm', Q_seq, K_seq) / self.size_per_head ** 0.5
        A = torch.softmax(A, dim=-1)
        O_seq = torch.einsum('ijkl,ijlm->ijkm', A, V_seq)
        O_seq = torch.transpose(O_seq, 1, 2)
        O_seq = torch.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))
        return O_seq


class Permute(nn.Module):

    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SwapTrailingAxes(nn.Module):

    def __init__(self):
        super(SwapTrailingAxes, self).__init__()

    def forward(self, x):
        return x.transpose(-2, -1)


class DocEncoder(nn.Module):

    def __init__(self):
        super(DocEncoder, self).__init__()
        self.phase1 = nn.Sequential(nn.Dropout(0.2), SwapTrailingAxes(), nn.Conv1d(300, 400, 3), nn.ReLU(), nn.Dropout(0.2), SwapTrailingAxes())
        self.attention = Attention(400, 20, 20)
        self.phase2 = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), AttentivePooling(30, 400))

    def fromTensorFlow(self, tfDoc):
        None
        for l in self.phase1:
            if 'conv' in l._get_name().lower():
                None
                for lt in tfDoc.layers:
                    None
                    if 'conv' in lt.name.lower():
                        weights = lt.get_weights()
                        l.weight.data = torch.tensor(weights[0]).transpose(0, 2)
                        l.bias.data = torch.tensor(weights[1])
                        break
                break
        self.attention.fromTensorFlow(tfDoc)
        None
        for l in self.phase2:
            if 'attentive' in l._get_name().lower():
                for lt in tfDoc.layers:
                    None
                    if 'model' in lt.name.lower():
                        None
                        l.fromTensorFlow(lt)

    def forward(self, x):
        l_cnnt = self.phase1(x)
        l_cnnt = self.attention([l_cnnt] * 3)
        result = self.phase2(l_cnnt)
        return result


class VecTail(nn.Module):

    def __init__(self, n):
        super(VecTail, self).__init__()
        self.n = n

    def forward(self, x):
        return x[:, -self.n:, :]


class UserEncoder(nn.Module):

    def __init__(self):
        super(UserEncoder, self).__init__()
        self.attention2 = Attention(400, 20, 20)
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = AttentivePooling(50, 400)
        self.tail2 = VecTail(20)
        self.gru2 = nn.GRU(400, 400, bidirectional=False, batch_first=True)
        self.pool3 = AttentivePooling(2, 400)

    def forward(self, news_vecs_input):
        user_vecs2 = self.attention2([news_vecs_input] * 3)
        user_vecs2 = self.dropout2(user_vecs2)
        user_vec2 = self.pool2(user_vecs2)
        user_vecs1 = self.tail2(news_vecs_input)
        self.gru2.flatten_parameters()
        user_vec1, _u_hidden = self.gru2(user_vecs1)
        user_vec1 = user_vec1[:, -1, :]
        user_vecs = torch.stack([user_vec1, user_vec2], dim=1)
        vec = self.pool3(user_vecs)
        return vec

    def fromTensorFlow(self, tfU):
        for l in tfU.layers:
            None
            if l.name == 'model_1':
                self.pool2.fromTensorFlow(l)
            elif l.name == 'model_2':
                self.pool3.fromTensorFlow(l)
            elif l.name == 'gru_1':
                None
                weights = l.get_weights()
                for p in self.gru2.named_parameters():
                    s1 = p[1].data.shape
                    if p[0] == 'weight_ih_l0':
                        p[1].data = torch.tensor(weights[0]).transpose(0, 1).contiguous()
                    elif p[0] == 'weight_hh_l0':
                        p[1].data = torch.tensor(weights[1]).transpose(0, 1).contiguous()
                    elif p[0] == 'bias_ih_l0':
                        p[1].data = torch.tensor(weights[2])
                    elif p[0] == 'bias_hh_l0':
                        p[1].data = torch.zeros(p[1].data.shape)
                    None
        self.attention2.fromTensorFlow(tfU)


class TimeDistributed(nn.Module):

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        output = torch.tensor([])
        for i in range(x.size(1)):
            output_t = self.module(x[:, i, :, :])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class FedNewsRec(nn.Module):

    def __init__(self, title_word_embedding_matrix):
        super(FedNewsRec, self).__init__()
        self.doc_encoder = DocEncoder()
        self.user_encoder = UserEncoder()
        self.title_word_embedding_layer = nn.Embedding.from_pretrained(torch.tensor(title_word_embedding_matrix, dtype=torch.float), freeze=True)
        self.softmax = nn.Softmax(dim=1)
        self.click_td = TimeDistributed(self.doc_encoder)
        self.can_td = TimeDistributed(self.doc_encoder)

    def forward(self, click_title, can_title):
        click_word_vecs = self.title_word_embedding_layer(click_title)
        can_word_vecs = self.title_word_embedding_layer(can_title)
        click_vecs = self.click_td(click_word_vecs)
        can_vecs = self.can_td(can_word_vecs)
        user_vec = self.user_encoder(click_vecs)
        scores = torch.einsum('ijk,ik->ij', can_vecs, user_vec)
        logits = scores
        return logits, user_vec

    def news_encoder(self, news_title):
        news_word_vecs = self.title_word_embedding_layer(news_title)
        news_vec = self.doc_encoder(news_word_vecs)
        return news_vec


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


class FEDNEWS(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        root_data_path = model_config['embbeding_path']
        embedding_path = model_config['embbeding_path']
        news, news_index, category_dict, subcategory_dict, word_dict = self.read_news(root_data_path, ['train', 'val'])
        title_word_embedding_matrix, _ = self.load_matrix(embedding_path, word_dict)
        self.net = FedNewsRec(title_word_embedding_matrix)

    def loss(self, input: torch.Tensor) ->torch.Tensor:
        """Performs forward step and computes the loss"""
        if not self.net.training:
            return torch.tensor(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        (click, sample), label = input['x'], input['y']
        click = click
        sample = sample
        label = label
        criterion = CrossEntropyLoss()
        output, _ = self.net.forward(click, sample)
        return criterion(output, label)

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        (nv_hist, nv_imp), labels = input['x'], input['y']
        nv_hist = nv_hist
        nv_imp = nv_imp
        nv = self.net.news_encoder(nv_imp).detach().cpu().numpy()
        nv_hist = self.net.news_encoder(nv_hist)
        uv = self.net.user_encoder(nv_hist.unsqueeze(0)).detach().cpu().numpy()[0]
        score = np.dot(nv, uv)
        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        acc = ndcg_score(labels, score, k=1)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)
        return {'output': None, 'acc': acc, 'batch_size': 1, 'auc': {'value': auc, 'higher_is_better': True}, 'mrr': {'value': mrr, 'higher_is_better': True}, 'ndcg5': {'value': ndcg5, 'higher_is_better': True}, 'ndcg10': {'value': ndcg10, 'higher_is_better': True}}

    def read_news(self, root_data_path, modes):
        news = {}
        category = []
        subcategory = []
        news_index = {}
        index = 1
        word_dict = {}
        word_index = 1
        for mode in modes:
            with open(os.path.join(root_data_path, mode, 'news.tsv'), encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                splited = line.strip('\n').split('\t')
                doc_id, vert, subvert, title = splited[0:4]
                if doc_id in news_index:
                    continue
                news_index[doc_id] = index
                index += 1
                category.append(vert)
                subcategory.append(subvert)
                title = title.lower()
                title = word_tokenize(title)
                news[doc_id] = [vert, subvert, title]
                for word in title:
                    word = word.lower()
                    if not word in word_dict:
                        word_dict[word] = word_index
                        word_index += 1
        category = list(set(category))
        subcategory = list(set(subcategory))
        category_dict = {}
        index = 1
        for c in category:
            category_dict[c] = index
            index += 1
        subcategory_dict = {}
        index = 1
        for c in subcategory:
            subcategory_dict[c] = index
            index += 1
        return news, news_index, category_dict, subcategory_dict, word_dict

    def load_matrix(self, embedding_path, word_dict):
        embedding_matrix = np.zeros((len(word_dict) + 1, 300))
        have_word = []
        with open(os.path.join(embedding_path, 'glove.840B.300d.txt'), 'rb') as f:
            while True:
                l = f.readline()
                if len(l) == 0:
                    break
                l = l.split()
                word = l[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in l[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
        return embedding_matrix, have_word


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


def print_rank(str, loglevel=logging.INFO):
    str = '{} : {}'.format(time.ctime(), str)
    logging.log(loglevel, str)


class ComputeMetrics:

    def __init__(self, p: EvalPrediction, mask=None):
        self.EvalPrediction = EvalPrediction
        self.compute_metrics(self.EvalPrediction)

    @staticmethod
    def compute_metrics(p: EvalPrediction, mask=None):
        print_rank('Prediction Block Size: {}'.format(p.predictions.size()), loglevel=logging.DEBUG)
        if len(list(p.predictions.size())) < 3:
            if len(list(p.predictions.size())) < 2:
                print_rank('There is something REALLY wrong with prediction tensor:'.format(p.predictions.size()), loglevel=logging.INFO)
                return {'acc': torch.tensor(0.0)}
            print_rank('There is something wrong with prediction tensor:'.format(p.predictions.size()), loglevel=logging.INFO)
            preds = np.argmax(p.predictions, axis=1)
        else:
            preds = np.argmax(p.predictions, axis=2)
        if mask is None:
            return {'acc': (preds == p.label_ids).float().mean()}
        else:
            valid = mask == 1
            return {'acc': (preds.eq(p.label_ids.cpu()) * valid.cpu()).float().mean()}


def _get_first_shape(arrays):
    """Return the shape of the first array found in the nested struct `arrays`."""
    if isinstance(arrays, (list, tuple)):
        return _get_first_shape(arrays[0])
    return arrays.shape


def nested_expand_like(arrays, new_seq_length, padding_index=-100):
    """ Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_expand_like(x, new_seq_length, padding_index=padding_index) for x in arrays)
    result = np.full_like(arrays, padding_index, shape=(arrays.shape[0], new_seq_length) + arrays.shape[2:])
    result[:, :arrays.shape[1]] = arrays
    return result


def nested_new_like(arrays, num_samples, padding_index=-100):
    """ Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_new_like(x, num_samples) for x in arrays)
    return np.full_like(arrays, padding_index, shape=(num_samples, *arrays.shape[1:]))


def nested_truncate(tensors, limit):
    """Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.
    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:
        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`
    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:
        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`
    The first batch treated on each process will be
        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`
    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:
        :obj:`[0, 1, 6, 7, 12, 13]`
    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:
        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`
    For some reason, that's not going to roll their boat. This class is there to solve that problem.
    Args:
        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
        padding_index (:obj:`int`, `optional`, defaults to -100):
            The padding index to use if the arrays don't all have the same sequence length.
    """

    def __init__(self, world_size, num_samples, make_multiple_of=None, padding_index=-100):
        self.world_size = world_size
        self.num_samples = num_samples
        total_size = world_size if make_multiple_of is None else world_size * make_multiple_of
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        self.process_length = self.total_samples // world_size
        self._storage = None
        self._offsets = None
        self.padding_index = padding_index

    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        if arrays is None:
            return
        if self._storage is None:
            self._storage = nested_new_like(arrays, self.total_samples, padding_index=self.padding_index)
            self._offsets = list(range(0, self.total_samples, self.process_length))
        else:
            storage_shape = _get_first_shape(self._storage)
            arrays_shape = _get_first_shape(arrays)
            if len(storage_shape) > 1 and storage_shape[1] < arrays_shape[1]:
                self._storage = nested_expand_like(self._storage, arrays_shape[1], padding_index=self.padding_index)
        slice_len = self._nested_set_tensors(self._storage, arrays)
        for i in range(self.world_size):
            self._offsets[i] += slice_len

    def _nested_set_tensors(self, storage, arrays):
        if isinstance(arrays, (list, tuple)):
            for x, y in zip(storage, arrays):
                slice_len = self._nested_set_tensors(x, y)
            return slice_len
        assert arrays.shape[0] % self.world_size == 0, f'Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}.'
        slice_len = arrays.shape[0] // self.world_size
        for i in range(self.world_size):
            if len(arrays.shape) == 1:
                storage[self._offsets[i]:self._offsets[i] + slice_len] = arrays[i * slice_len:(i + 1) * slice_len]
            else:
                storage[self._offsets[i]:self._offsets[i] + slice_len, :arrays.shape[1]] = arrays[i * slice_len:(i + 1) * slice_len]
        return slice_len

    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        if self._storage is None:
            return
        if self._offsets[0] != self.process_length:
            logger.warn('Not all data has been set. Are you sure you passed all values?')
        return nested_truncate(self._storage, self.num_samples)


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), dim=0)
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[:array1.shape[0], :array1.shape[1]] = array1
    result[array1.shape[0]:, :array2.shape[1]] = array2
    return result


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]
    result = tensor1.new_full(new_shape, padding_index)
    result[:tensor1.shape[0], :tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, :tensor2.shape[1]] = tensor2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(new_tensors), f'Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}.'
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f'Unsupported type for concatenation: got {type(tensors)}')


def nested_detach(tensors):
    """Detach `tensors` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def nested_numpify(tensors):
    """Numpify `tensors` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x):
    return x if torch.cuda.is_available() else x


class BERT(BaseModel):

    def __init__(self, model_config, **kwargs):
        super(BERT, self).__init__()
        """
            from transformers import RobertaConfig
            config = RobertaConfig(
                        vocab_size=52_000,
                        max_position_embeddings=514,
                        num_attention_heads=12,
                        num_hidden_layers=6,
                        type_vocab_size=1,
            )

            from transformers import RobertaTokenizerFast
            tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

            from transformers import RobertaForMaskedLM
            model = RobertaForMaskedLM(config=config)
        """
        args = model_config['BERT']
        model_args, training_args = args['model'], args['training']
        set_seed(training_args['seed'])
        self.gradient_accumulation_steps = model_args.get('gradient_accumulation_steps', 1)
        self.past_index = model_args.get('past_index', -1)
        self.prediction_loss_only = model_args.get('prediction_loss_only', True)
        self.eval_accumulation_steps = model_args.get('eval_accumulation_steps', None)
        self.label_names = model_args.get('label_names', None)
        self.batch_size = training_args['batch_size']
        self.model_name = model_args['model_name']
        if 'model_name_or_path' not in model_args:
            model_args['model_name_or_path'] = self.model_name
        if training_args['label_smoothing_factor'] != 0:
            self.label_smoother = LabelSmoother(epsilon=training_args['label_smoothing_factor'])
        else:
            self.label_smoother = None
        self.label_names = ['labels'] if self.label_names is None else self.label_names
        config_kwargs = {'cache_dir': model_args['cache_dir'], 'revision': None, 'use_auth_token': None}
        if 'config_name' in model_args:
            config = AutoConfig.from_pretrained(model_args['config_name'], **config_kwargs)
        elif 'model_name_or_path' in model_args:
            config = AutoConfig.from_pretrained(model_args['model_name_or_path'], **config_kwargs)
        else:
            raise ValueError('You are instantiating a new configuration from scratch. This is not supported by this script.')
        tokenizer_kwargs = {'cache_dir': model_args['cache_dir'], 'use_fast': model_args['use_fast_tokenizer'], 'use_auth_token': None}
        if 'tokenizer_name' in model_args:
            tokenizer = AutoTokenizer.from_pretrained(model_args['tokenizer_name'], **tokenizer_kwargs)
        elif 'model_name_or_path' in model_args:
            None
            tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'], **tokenizer_kwargs)
        else:
            raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script.')
        self.output_layer_size = len(tokenizer)
        if 'model_name_or_path' in model_args:
            None
            self.model = AutoModelForMaskedLM.from_pretrained(model_args['model_name_or_path'], from_tf=False, config=config, cache_dir=model_args['cache_dir'], use_auth_token=None)
            if 'adapter' in model_args:
                if model_args['adapter']:
                    self.model.add_adapter('FLUTE')
                    self.model.train_adapter('FLUTE')
        else:
            raise ValueError('You are instantiating a new model from scratch. This is not supported by this script.')
        self.model.resize_token_embeddings(self.output_layer_size)
        total_params = 0
        trainable_params = 0
        for p in self.model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print_rank(f'Total parameters count: {total_params}', loglevel=logging.DEBUG)
        print_rank(f'Trainable parameters count: {trainable_params}', loglevel=logging.DEBUG)
        print_rank(f'Original Bert parameters count: {total_params - trainable_params}', loglevel=logging.DEBUG)

    def copy_state_dict(self, state_dict):
        self.model.state_dict = state_dict.clone()

    def get_model(self):
        return self.model

    def _prepare_inputs(self, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, T.Tensor):
                inputs[k] = to_device(v)
        if self.past_index >= 0 and self._past is not None:
            inputs['mems'] = self._past
        return inputs

    def forward(self, inputs):
        inputs = self._prepare_inputs(inputs)
        return self.model(**inputs)

    def loss(self, inputs):
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`T.Tensor`: The tensor with training loss on this batch.
        """
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(inputs)
        loss = loss / self.gradient_accumulation_steps
        return loss

    def compute_loss(self, inputs_orig, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.

        inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        """
        inputs = copy.deepcopy(inputs_orig)
        if self.label_smoother is not None and 'labels' in inputs:
            labels = inputs['labels'].detach().cpu()
        else:
            labels = None
        if 'roberta' in self.model_name:
            if 'attention_mask' in inputs:
                inputs.pop('attention_mask')
            if 'special_tokens_mask' in inputs:
                inputs.pop('special_tokens_mask')
        outputs = self.model(**inputs)
        if self.past_index >= 0:
            self._past = outputs[self.past_index]
        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def inference(self, inputs, ignore_keys: Optional[List[str]]=[], metric_key_prefix: str='eval') ->List[float]:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.
        Args:
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        .. note::
            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.
        Returns: `NamedTuple` A namedtuple with the following keys:
            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        output, batch_size = self.prediction_loop(inputs, description='Evaluation', ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        return {'output': output['eval_loss'], 'acc': output['eval_acc'], 'batch_size': batch_size[0]}

    def prediction_loop(self, inputs, description: str, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='eval') ->Union[Dict, List[int]]:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        out_label_ids = None
        if 'labels' in inputs:
            out_label_ids = inputs['labels'].detach().cpu()
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask'].detach().cpu()
        losses_host = None
        preds_host = None
        labels_host = None
        world_size = 1
        num_hosts = 1
        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_hosts, make_multiple_of=self.batch_size)
        if not self.prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_hosts)
            labels_gatherer = DistributedTensorGatherer(world_size, num_hosts)
        self.model.eval()
        if self.past_index >= 0:
            self._past = None
        loss, logits, _ = self.prediction_step(inputs, ignore_keys=ignore_keys, has_labels=True)
        if loss is not None:
            losses = loss.repeat(self.batch_size).cpu()
            losses_host = losses if losses_host is None else T.cat((losses_host, losses), dim=0)
        if logits is not None:
            preds_host = logits.detach().cpu() if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
        if out_label_ids is not None:
            labels_host = out_label_ids if labels_host is None else nested_concat(labels_host, out_label_ids, padding_index=-100)
        if self.eval_accumulation_steps is not None:
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, 'eval_losses'))
            if not self.prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, 'eval_preds'))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, 'eval_label_ids'))
            losses_host, preds_host, labels_host = None, None, None
        if self.past_index and hasattr(self, '_past'):
            delattr(self, '_past')
        if num_hosts > 1:
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, 'eval_losses'), want_masked=True)
            if not self.prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, 'eval_preds'))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, 'eval_label_ids'))
            eval_loss = eval_losses_gatherer.finalize()
            preds = preds_gatherer.finalize() if not self.prediction_loss_only else None
            label_ids = labels_gatherer.finalize() if not self.prediction_loss_only else None
        else:
            eval_loss = losses_host
            preds = preds_host
            label_ids = labels_host
        if preds is not None and label_ids is not None:
            metrics = ComputeMetrics.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids), attention_mask)
        else:
            metrics = {}
        if eval_loss is not None:
            metrics[f'{metric_key_prefix}_loss'] = eval_loss.mean().item()
        for key in list(metrics.keys()):
            if not key.startswith(f'{metric_key_prefix}_'):
                metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key).item()
        return metrics, preds.size()

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        return nested_numpify(tensors)

    def prediction_step(self, inputs, ignore_keys: Optional[List[str]]=None, has_labels: bool=None) ->Tuple[Optional[float], Optional[T.Tensor], Optional[T.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[float], Optional[T.Tensor], Optional[T.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        if has_labels:
            labels = inputs['labels'].detach().cpu()
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        with T.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = self.model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                if self.past_index >= 0:
                    self._past = outputs[self.past_index - 1]
        if self.prediction_loss_only:
            return loss, None, None
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return loss, logits, labels

    def floating_point_ops(self, inputs):
        """
        For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.
        Args:
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
        Returns:
            :obj:`int`: The number of floating-point operations.
        """
        if hasattr(self.model, 'floating_point_ops'):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def set_eval(self):
        """
        Bring the model into evaluation mode
        """
        self.model.eval()

    def set_train(self):
        """
        Bring the model into train mode
        """
        self.model.train()


class GRU2(T.nn.Module):

    def __init__(self, input_size, hidden_size, input_bias, hidden_bias):
        super(GRU2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = T.nn.Linear(input_size, 3 * hidden_size, input_bias)
        self.w_hh = T.nn.Linear(hidden_size, 3 * hidden_size, hidden_bias)

    def _forward_cell(self, input: Tensor, hidden: Tensor) ->Tensor:
        g_i = self.w_ih(input)
        g_h = self.w_hh(hidden)
        i_r, i_i, i_n = g_i.chunk(3, 1)
        h_r, h_i, h_n = g_h.chunk(3, 1)
        reset_gate = T.sigmoid(i_r + h_r)
        input_gate = T.sigmoid(i_i + h_i)
        new_gate = T.tanh(i_n + reset_gate * h_n)
        hy = new_gate + input_gate * (hidden - new_gate)
        return hy

    def forward(self, input: Tensor) ->Tuple[Tensor, Tensor]:
        hiddens: List[Tensor] = [to_device(T.zeros((input.shape[0], self.hidden_size)))]
        for step in range(input.shape[1]):
            hidden = self._forward_cell(input[:, step], hiddens[-1])
            hiddens.append(hidden)
        return T.stack(hiddens, dim=1), hiddens[-1]


class Embedding(T.nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.table = T.nn.Parameter(T.zeros((vocab_size, embedding_size)))
        self.unembedding_bias = T.nn.Parameter(T.zeros(vocab_size))
        delta = (3 / self.table.shape[1]) ** 0.5
        T.nn.init.uniform_(self.table, -delta, delta)

    def forward(self, input: Tensor, embed: bool) ->Tensor:
        if embed:
            output = T.nn.functional.embedding(input, self.table)
        else:
            output = input @ self.table.t() + self.unembedding_bias
        return output


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.

    Args:
        X (ndarray): x, probably should be floats.
        theta (float): used as a multiplier prior to exponentiation. Default = 1.0
        axis : axis to compute values along. Default is the first non-singleton axis.

    Returns:
        An array the same size as X. The result will sum to 1 along the specified axis.
    """
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p


class GRU(BaseModel):

    def __init__(self, model_config, OOV_correct=False, dropout=0.0, topK_results=1, wantLogits=False, **kwargs):
        super(GRU, self).__init__()
        self.vocab_size = model_config['vocab_size']
        self.embedding_size = model_config['embed_dim']
        self.hidden_size = model_config['hidden_dim']
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        self.rnn = GRU2(self.embedding_size, self.hidden_size, True, True)
        self.squeeze = T.nn.Linear(self.hidden_size, self.embedding_size, bias=False)
        self.OOV_correct = OOV_correct
        self.topK_results = topK_results
        self.dropout = dropout
        self.wantLogits = wantLogits
        if self.dropout > 0.0:
            self.drop_layer = T.nn.Dropout(p=self.dropout)

    def forward(self, input: T.Tensor) ->Tuple[Tensor, Tensor]:
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        embedding = self.embedding(input, True)
        hiddens, state = self.rnn(embedding)
        if self.dropout > 0.0:
            hiddens = self.drop_layer(hiddens)
        output = self.embedding(self.squeeze(hiddens), False)
        return output, state

    def loss(self, input: T.Tensor) ->T.Tensor:
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        non_pad_mask = input >= 0
        input = input * non_pad_mask.long()
        non_pad_mask = non_pad_mask.view(-1)
        output, _ = self.forward(input[:, :-1])
        targets = input.view(-1)[non_pad_mask]
        preds = output.view(-1, self.vocab_size)[non_pad_mask]
        return T.nn.functional.cross_entropy(preds, targets)

    def inference(self, input):
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        non_pad_mask = input >= 0
        input = input * non_pad_mask.long()
        non_pad_mask = non_pad_mask.view(-1)
        output, _ = self.forward(input[:, :-1])
        targets = input.view(-1)[non_pad_mask]
        preds = output.view(-1, self.vocab_size)[non_pad_mask]
        probs_topK, preds_topK = T.topk(preds, self.topK_results, sorted=True, dim=1)
        probs, preds = probs_topK[:, 0], preds_topK[:, 0]
        if self.OOV_correct:
            acc = preds.eq(targets).float().mean()
        else:
            valid = preds != 0
            acc = (preds.eq(targets) * valid).float().mean()
        if self.wantLogits:
            if 1:
                output = {'probabilities': softmax(probs_topK.cpu().detach().numpy(), axis=1), 'predictions': preds_topK.cpu().detach().numpy(), 'labels': targets.cpu().detach().numpy()}
            else:
                output = {'probabilities': probs_topK.cpu().detach().numpy(), 'predictions': preds_topK.cpu().detach().numpy(), 'labels': targets.cpu().detach().numpy()}
        return {'output': output, 'acc': acc.item(), 'batch_size': input.shape[0]}


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = x.contiguous()
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, dropout=0.0, multi=1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm_activate = batch_norm
        self.bidirectional = bidirectional
        self.multi = multi
        self.dropout = dropout
        if self.batch_norm_activate:
            self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size))
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True, batch_first=True, dropout=self.dropout)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.batch_norm_activate:
            x = x.contiguous()
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional and self.multi < 2:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class NeuralNetwork(nn.Module):

    def __init__(self, params, wantLSTM=False, batch_norm=False):
        super(NeuralNetwork, self).__init__()
        """
        The following parameters need revisiting
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        criterion = nn.MSELoss()

        """
        self.wantLSTM = wantLSTM
        self.batch_norm = batch_norm
        params = [int(x) for x in params.split(',')]
        layers = []
        self.softmax = nn.Softmax(dim=1)
        if self.wantLSTM:
            rnns = []
            for i in range(1, len(params) - 2):
                multi = 1 if i == 1 else 1
                rnn = BatchRNN(input_size=params[i - 1] * multi, hidden_size=params[i], rnn_type=nn.LSTM, bidirectional=True, batch_norm=batch_norm, multi=1, dropout=0.0)
                rnns.append(('%d' % (i - 1), rnn))
            self.rnn = nn.Sequential(OrderedDict(rnns))
            layers.append(nn.Linear(params[-3], params[-2], bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            mlp = nn.Sequential(*layers)
            self.mlp = nn.Sequential(SequenceWise(mlp))
        else:
            if self.batch_norm:
                self.batch_norm = nn.BatchNorm1d(params[0])
            for i in range(1, len(params) - 1):
                layers.append(nn.Linear(params[i - 1], params[i], bias=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.wantLSTM:
            x = self.rnn(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        out = self.mlp(x)
        out = out.squeeze()
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentivePooling,
     lambda: ([], {'dim1': 4, 'dim2': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvNormPool,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DocEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 300, 300])], {}),
     False),
    (Embedding,
     lambda: ([], {'vocab_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (GRU2,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'input_bias': 4, 'hidden_bias': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GroupNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GroupNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (SequenceWise,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwapTrailingAxes,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeDistributed,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VecTail,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_GroupNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_microsoft_msrflute(_paritybench_base):
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

