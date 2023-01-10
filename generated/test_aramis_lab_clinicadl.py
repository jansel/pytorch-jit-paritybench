import sys
_module = sys.modules[__name__]
del sys
clinicadl = _module
cmdline = _module
generate = _module
generate = _module
generate_cli = _module
generate_random_cli = _module
generate_shepplogan_cli = _module
generate_trivial_cli = _module
generate_utils = _module
interpret = _module
gradients = _module
interpret_cli = _module
predict = _module
predict_cli = _module
prepare_data = _module
prepare_data = _module
prepare_data_cli = _module
prepare_data_utils = _module
quality_check = _module
qc_cli = _module
t1_linear = _module
cli = _module
quality_check = _module
utils = _module
t1_volume = _module
random_search = _module
random_search_cli = _module
random_search_utils = _module
masks = _module
models = _module
train = _module
from_json_cli = _module
list_models_cli = _module
resume = _module
resume_cli = _module
tasks = _module
classification_cli = _module
reconstruction_cli = _module
regression_cli = _module
task_utils = _module
train_cli = _module
train_option = _module
train_utils = _module
tsvtools = _module
analysis = _module
analysis_cli = _module
get_labels = _module
get_labels_cli = _module
get_metadata = _module
get_metadata_cli = _module
get_progression = _module
get_progression_cli = _module
getlabels = _module
kfold = _module
kfold_cli = _module
prepare_experiment_cli = _module
split = _module
split_cli = _module
caps_dataset = _module
data = _module
cli_param = _module
argument = _module
option = _module
option_group = _module
cmdline_utils = _module
descriptors = _module
early_stopping = _module
exceptions = _module
logger = _module
maps_manager = _module
iotools = _module
logwriter = _module
maps_manager = _module
maps_manager_utils = _module
meta_maps = _module
getter = _module
metric_module = _module
network = _module
autoencoder = _module
cnn_transformer = _module
cnn = _module
models = _module
random = _module
resnet = _module
network = _module
network_utils = _module
sub_network = _module
vae = _module
base_vae = _module
vae_utils = _module
vanilla_vae = _module
preprocessing = _module
seed = _module
split_manager = _module
single_split = _module
task_manager = _module
classification = _module
reconstruction = _module
regression = _module
task_manager = _module
tsvtools_utils = _module
tests = _module
conftest = _module
test_cli = _module
test_generate = _module
test_interpret = _module
test_predict = _module
test_prepare_data = _module
test_qc = _module
test_random_search = _module
test_resume = _module
test_train_ae = _module
test_train_cnn = _module
test_train_from_json = _module
test_transfer_learning = _module
test_tsvtools = _module
testing_tools = _module

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


from typing import Dict


from typing import Optional


from typing import Tuple


import numpy as np


import pandas as pd


import torch


from logging import getLogger


from time import time


from typing import Any


from typing import List


from typing import Union


from torch.utils.data import DataLoader


import torch.nn as nn


from torch.utils.data import Dataset


import abc


from typing import Callable


import torchvision.transforms as transforms


import logging


from copy import deepcopy


from torch import nn


import torch.utils.model_zoo as model_zoo


from torchvision.models.resnet import BasicBlock


import math


import torch.cuda


from collections import OrderedDict


import torch.nn.functional as F


import random


from torch.nn.functional import softmax


from torch.utils.data import sampler


from abc import abstractmethod


from typing import Sequence


from torch import Tensor


from torch.nn.modules.loss import _Loss


from torch.utils.data import Sampler


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
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetQC(nn.Module):

    def __init__(self, block, layers, num_classes=2, use_ref=False, zero_init_residual=False):
        super(ResNetQC, self).__init__()
        self.inplanes = 64
        self.use_ref = use_ref
        self.feat = 3
        self.expansion = block.expansion
        self.conv1 = nn.Conv2d(2 if self.use_ref else 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.addon = nn.Sequential(nn.Conv2d(self.feat * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(512 * block.expansion), nn.ReLU(inplace=True), nn.Conv2d(512 * block.expansion, 32, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=0, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=False), nn.Dropout2d(p=0.5, inplace=True), nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 2 if self.use_ref else 1, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 512 * self.feat * self.expansion, 7, 7)
        x = self.addon(x)
        x = x.view(x.size(0), -1)
        return x


class CropMaxUnpool2d(nn.Module):

    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool2d, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[2]
            y1 = padding[0]
            output = output[:, :, x1:, y1:]
        return output


class CropMaxUnpool3d(nn.Module):

    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool3d, self).__init__()
        self.unpool = nn.MaxUnpool3d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[4]
            y1 = padding[2]
            z1 = padding[0]
            output = output[:, :, x1:, y1:, z1:]
        return output


class PadMaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad2d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [(self.stride - f_maps.size(i + 2) % self.stride) for i in range(2)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0
        self.pad.padding = coords[1], 0, coords[0], 0
        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))
            if self.return_pad:
                return output, indices, (coords[1], 0, coords[0], 0)
            else:
                return output, indices
        else:
            output = self.pool(self.pad(f_maps))
            if self.return_pad:
                return output, (coords[1], 0, coords[0], 0)
            else:
                return output


class PadMaxPool3d(nn.Module):

    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [(self.stride - f_maps.size(i + 2) % self.stride) for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0
        self.pad.padding = coords[2], 0, coords[1], 0, coords[0], 0
        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))
            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices
        else:
            output = self.pool(self.pad(f_maps))
            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class Reshape(nn.Module):

    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(*self.size)


class CNN_Transformer(nn.Module):

    def __init__(self, model=None):
        """
        Construct an autoencoder from a given CNN. The encoder part corresponds to the convolutional part of the CNN.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        """
        from copy import deepcopy
        super(CNN_Transformer, self).__init__()
        self.level = 0
        if model is not None:
            self.encoder = deepcopy(model.convolutions)
            self.decoder = self.construct_inv_layers(model)
            for i, layer in enumerate(self.encoder):
                if isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d):
                    self.encoder[i].set_new_return()
                elif isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d):
                    self.encoder[i].return_indices = True
        else:
            self.encoder = nn.Sequential()
            self.decoder = nn.Sequential()

    def __len__(self):
        return len(self.encoder)

    def construct_inv_layers(self, model):
        """
        Implements the decoder part from the CNN. The decoder part is the symmetrical list of the encoder
        in which some layers are replaced by their transpose counterpart.
        ConvTranspose and ReLU layers are inverted in the end.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        :return: (Module) decoder part of the Autoencoder
        """
        inv_layers = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv3d):
                inv_layers.append(nn.ConvTranspose3d(layer.out_channels, layer.in_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding))
                self.level += 1
            elif isinstance(layer, nn.Conv2d):
                inv_layers.append(nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding))
                self.level += 1
            elif isinstance(layer, PadMaxPool3d):
                inv_layers.append(CropMaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, PadMaxPool2d):
                inv_layers.append(CropMaxUnpool2d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.Linear):
                inv_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, nn.Flatten):
                inv_layers.append(Reshape(model.flattened_shape))
            elif isinstance(layer, nn.LeakyReLU):
                inv_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
            else:
                inv_layers.append(deepcopy(layer))
        inv_layers = self.replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)

    @staticmethod
    def replace_relu(inv_layers):
        """
        Invert convolutional and ReLU layers (give empirical better results)

        :param inv_layers: (list) list of the layers of decoder part of the Auto-Encoder
        :return: (list) the layers with the inversion
        """
        idx_relu, idx_conv = -1, -1
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.ConvTranspose3d):
                idx_conv = idx
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                idx_relu = idx
            if idx_conv != -1 and idx_relu != -1:
                inv_layers[idx_relu], inv_layers[idx_conv] = inv_layers[idx_conv], inv_layers[idx_relu]
                idx_conv, idx_relu = -1, -1
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.BatchNorm3d):
                conv = inv_layers[idx + 1]
                inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)
        return inv_layers


class ResNetDesigner(nn.Module):

    def __init__(self, input_size, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetDesigner, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        out = self.conv1(input_tensor)
        out = self.relu(self.bn1(out))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self.avgpool = nn.AvgPool2d((out.size(2), out.size(3)), stride=1)
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


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, gpu=True):
        super(Network, self).__init__()
        self.device = self._select_device(gpu)

    @staticmethod
    def _select_device(gpu):
        from numpy import argmax
        logger = getLogger('clinicadl.networks')
        if not gpu:
            return 'cpu'
        else:
            try:
                _ = os.environ['CUDA_VISIBLE_DEVICES']
                return 'cuda'
            except KeyError:
                try:
                    nvmlInit()
                    memory_list = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free for i in range(torch.cuda.device_count())]
                    free_gpu = argmax(memory_list)
                    return f'cuda:{free_gpu}'
                except NVMLError:
                    logger.warning('NVML library is not installed. GPU will be chosen arbitrarily')
                    return 'cuda'

    @staticmethod
    @abc.abstractmethod
    def get_input_size() ->str:
        """This static method is used for list_models command.
        Must return the shape of the input size expected (C@HxW or C@HxWxD) for each architecture.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_dimension() ->str:
        """This static method is used for list_models command.
        Return '2D', '3D' or '2D and 3D'
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_task() ->list:
        """This static method is used for list_models command.
        Return the list of tasks for which the model is made.
        """
        pass

    @abc.abstractproperty
    def layers(self):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        pass

    def transfer_weights(self, state_dict, transfer_class):
        self.load_state_dict(state_dict)


class ClinicaDLException(Exception):
    """Base class for ClinicaDL exceptions."""


class ClinicaDLNetworksError(ClinicaDLException):
    """Base class for Networks exceptions."""


class CNN(Network):

    def __init__(self, convolutions, fc, n_classes, gpu=False):
        super().__init__(gpu=gpu)
        self.convolutions = convolutions
        self.fc = fc
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict([(k.replace('encoder.', ''), v) for k, v in state_dict.items() if 'encoder' in k])
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(f'Cannot transfer weights from {transfer_class} to CNN.')

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        images, labels = input_dict['image'], input_dict['label']
        train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])
        return train_output, {'loss': loss}


class AutoEncoder(Network):

    def __init__(self, encoder, decoder, gpu=False):
        super().__init__(gpu=gpu)
        self.encoder = encoder
        self.decoder = decoder

    @property
    def layers(self):
        return nn.Sequential(self.encoder, self.decoder)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, AutoEncoder):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, CNN):
            encoder_dict = OrderedDict([(k.replace('convolutions.', ''), v) for k, v in state_dict.items() if 'convolutions' in k])
            self.encoder.load_state_dict(encoder_dict)
        else:
            raise ClinicaDLNetworksError(f'Cannot transfer weights from {transfer_class} to Autoencoder.')

    def predict(self, x):
        _, output = self.forward(x)
        return output

    def forward(self, x):
        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if (isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d)) and layer.return_indices and layer.return_pad:
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif (isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d)) and layer.return_indices:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        code = x.clone()
        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d) or isinstance(layer, CropMaxUnpool2d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d) or isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)
        return code, x

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        images = input_dict['image']
        train_output = self.predict(images)
        loss = criterion(train_output, images)
        return train_output, {'loss': loss}


class BaseVAE(Network):

    def __init__(self, encoder, decoder, mu_layer, var_layer, latent_size, gpu=True, is_3D=False, recons_weight=1, KL_weight=1):
        super(BaseVAE, self).__init__(gpu=gpu)
        self.lambda1 = recons_weight
        self.lambda2 = KL_weight
        self.latent_size = latent_size
        self.is_3D = is_3D
        self.encoder = encoder
        self.mu_layer = mu_layer
        self.var_layer = var_layer
        self.decoder = decoder

    @property
    def layers(self):
        return torch.nn.Sequential(self.encoder, self.mu_layer, self.var_layer, self.decoder)

    def predict(self, x):
        output, _, _ = self.forward(x)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        images = input_dict['image']
        recon_images, mu, log_var = self.forward(images)
        recon_loss = criterion(recon_images, images)
        if self.is_3D:
            kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        else:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = self.lambda1 * recon_loss + self.lambda2 * kl_loss / self.latent_size
        loss_dict = {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
        return recon_images, loss_dict

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu_layer(h), self.var_layer(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class EncoderLayer2D(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 2D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderLayer2D, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        x = F.leaky_relu(self.layer(x), negative_slope=0.2, inplace=True)
        return x


class DecoderLayer2D(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 2D transposed convolutional layer,
    a batch normalization layer with a relu activation function.
    """

    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1):
        super(DecoderLayer2D, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class EncoderLayer3D(nn.Module):
    """
    Class defining the encoder's part of the Autoencoder.
    This layer is composed of one 3D convolutional layer,
    a batch normalization layer with a leaky relu
    activation function.
    """

    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderLayer3D, self).__init__()
        self.layer = nn.Sequential(nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm3d(output_channels))

    def forward(self, x):
        x = F.leaky_relu(self.layer(x), negative_slope=0.2, inplace=True)
        return x


class DecoderLayer3D(nn.Module):
    """
    Class defining the decoder's part of the Autoencoder.
    This layer is composed of one 3D transposed convolutional layer,
    a batch normalization layer with a relu activation function.
    """

    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
        super(DecoderLayer3D, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False), nn.BatchNorm3d(output_channels))

    def forward(self, x):
        x = F.relu(self.layer(x), inplace=True)
        return x


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten2D(nn.Module):

    def __init__(self, channel, height, width):
        super(Unflatten2D, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class Unflatten3D(nn.Module):

    def __init__(self, channel, height, width, depth):
        super(Unflatten3D, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.depth = depth

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width, self.depth)


class VAE_Encoder(nn.Module):
    """"""

    def __init__(self, input_shape, n_conv=4, first_layer_channels=32, latent_dim=1, feature_size=1024):
        """
        Feature size is the size of the vector if latent_dim=1
        or is the number of feature maps (number of channels) if latent_dim=2
        """
        super(VAE_Encoder, self).__init__()
        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        decoder_padding = []
        tensor_h, tensor_w = self.input_h, self.input_w
        self.layers = []
        self.layers.append(EncoderLayer2D(self.input_c, first_layer_channels))
        padding_h, padding_w = 0, 0
        if tensor_h % 2 != 0:
            padding_h = 1
        if tensor_w % 2 != 0:
            padding_w = 1
        decoder_padding.append([padding_h, padding_w])
        tensor_h, tensor_w = tensor_h // 2, tensor_w // 2
        for i in range(n_conv - 1):
            self.layers.append(EncoderLayer2D(first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)))
            padding_h, padding_w = 0, 0
            if tensor_h % 2 != 0:
                padding_h = 1
            if tensor_w % 2 != 0:
                padding_w = 1
            decoder_padding.append([padding_h, padding_w])
            tensor_h, tensor_w = tensor_h // 2, tensor_w // 2
        self.decoder_padding = decoder_padding
        if latent_dim == 1:
            n_pix = first_layer_channels * 2 ** (n_conv - 1) * (self.input_h // 2 ** n_conv) * (self.input_w // 2 ** n_conv)
            self.layers.append(nn.Sequential(Flatten(), nn.Linear(n_pix, feature_size), nn.ReLU()))
        elif latent_dim == 2:
            self.layers.append(nn.Sequential(nn.Conv2d(first_layer_channels * 2 ** (n_conv - 1), feature_size, 3, stride=1, padding=1, bias=False), nn.ReLU()))
        else:
            raise AttributeError('Bad latent dimension specified. Latent dimension must be 1 or 2')
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.sequential(x)
        return z


class VAE_Decoder(nn.Module):
    """"""

    def __init__(self, input_shape, latent_size, n_conv=4, last_layer_channels=32, latent_dim=1, feature_size=1024, padding=None):
        """
        Feature size is the size of the vector if latent_dim=1
        or is the W/H of the output channels if laten_dim=2
        """
        super(VAE_Decoder, self).__init__()
        self.input_c = input_shape[0]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        if not padding:
            output_padding = [[0, 0] for i in range(n_conv - 1)]
        else:
            output_padding = padding
        self.layers = []
        if latent_dim == 1:
            n_pix = last_layer_channels * 2 ** (n_conv - 1) * (self.input_h // 2 ** n_conv) * (self.input_w // 2 ** n_conv)
            self.layers.append(nn.Sequential(nn.Linear(latent_size, feature_size), nn.ReLU(), nn.Linear(feature_size, n_pix), nn.ReLU(), Unflatten2D(last_layer_channels * 2 ** (n_conv - 1), self.input_h // 2 ** n_conv, self.input_w // 2 ** n_conv), nn.ReLU()))
        elif latent_dim == 2:
            self.layers.append(nn.Sequential(nn.ConvTranspose2d(latent_size, feature_size, 3, stride=1, padding=1, bias=False), nn.ReLU(), nn.ConvTranspose2d(feature_size, last_layer_channels * 2 ** (n_conv - 1), 3, stride=1, padding=1, bias=False), nn.ReLU()))
        else:
            raise AttributeError('Bad latent dimension specified. Latent dimension must be 1 or 2')
        for i in range(n_conv - 1, 0, -1):
            self.layers.append(DecoderLayer2D(last_layer_channels * 2 ** i, last_layer_channels * 2 ** (i - 1)))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(last_layer_channels, self.input_c, 4, stride=2, padding=1, output_padding=output_padding[0], bias=False), nn.Sigmoid()))
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, z):
        y = self.sequential(z)
        return y


class VanillaDenseVAE(BaseVAE):
    """
    This network is a 2D convolutional variational autoencoder with a dense latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size, latent_space_size=128, feature_size=1024, recons_weight=1, KL_weight=1, gpu=True):
        n_conv = 4
        io_layer_channel = 32
        encoder = VAE_Encoder(input_size=input_size, feature_size=feature_size, latent_dim=1, n_conv=n_conv, first_layer_channels=io_layer_channel)
        mu_layer = nn.Linear(feature_size, latent_space_size)
        var_layer = nn.Linear(feature_size, latent_space_size)
        decoder = VAE_Decoder(input_size=input_size, latent_size=latent_space_size, feature_size=feature_size, latent_dim=1, n_conv=n_conv, last_layer_channels=io_layer_channel, padding=encoder.decoder_padding)
        super(VanillaDenseVAE, self).__init__(encoder, decoder, mu_layer, var_layer, latent_space_size, gpu=gpu, recons_weight=recons_weight, KL_weight=KL_weight, is_3D=False)

    @staticmethod
    def get_input_size():
        return '1@128x128'

    @staticmethod
    def get_dimension():
        return '2D'

    @staticmethod
    def get_task():
        return ['reconstruction']


class VanillaSpatialVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a spacial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size, latent_space_size=128, feature_size=1024, recons_weight=1, KL_weight=1, gpu=True):
        feature_channels = 64
        latent_channels = 1
        n_conv = 4
        io_layer_channel = 32
        encoder = VAE_Encoder(input_size=input_size, feature_size=feature_channels, latent_dim=2, n_conv=n_conv, first_layer_channels=io_layer_channel)
        mu_layer = nn.Conv2d(feature_channels, latent_channels, 3, stride=1, padding=1, bias=False)
        var_layer = nn.Conv2d(feature_channels, latent_channels, 3, stride=1, padding=1, bias=False)
        decoder = VAE_Decoder(input_size=input_size, latent_size=latent_channels, feature_size=feature_channels, latent_dim=2, n_conv=n_conv, last_layer_channels=io_layer_channel, padding=encoder.decoder_padding)
        super(VanillaSpatialVAE, self).__init__(encoder, decoder, mu_layer, var_layer, latent_space_size, gpu=gpu, recons_weight=recons_weight, KL_weight=KL_weight, is_3D=False)

    @staticmethod
    def get_input_size():
        return '1@128x128'

    @staticmethod
    def get_dimension():
        return '2D'

    @staticmethod
    def get_task():
        return ['reconstruction']


class Vanilla3DVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a spacial latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size, latent_space_size=256, feature_size=1024, recons_weight=1, KL_weight=1, gpu=True):
        n_conv = 4
        first_layer_channels = 32
        last_layer_channels = 32
        feature_channels = 512
        latent_channels = 1
        decoder_output_padding = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        input_c = input_size[0]
        encoder_layers = []
        encoder_layers.append(EncoderLayer3D(input_c, first_layer_channels))
        for i in range(n_conv - 1):
            encoder_layers.append(EncoderLayer3D(first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)))
        encoder_layers.append(nn.Sequential(nn.Conv3d(first_layer_channels * 2 ** (n_conv - 1), feature_channels, 4, stride=2, padding=1, bias=False), nn.ReLU()))
        encoder = nn.Sequential(*encoder_layers)
        mu_layer = nn.Conv3d(feature_channels, latent_channels, 3, stride=1, padding=1, bias=False)
        var_layer = nn.Conv3d(feature_channels, latent_channels, 3, stride=1, padding=1, bias=False)
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.ConvTranspose3d(latent_channels, feature_channels, 3, stride=1, padding=1, bias=False), nn.ReLU(), nn.ConvTranspose3d(feature_channels, last_layer_channels * 2 ** (n_conv - 1), 4, stride=2, padding=1, output_padding=[0, 1, 1], bias=False), nn.ReLU()))
        for i in range(n_conv - 1, 0, -1):
            decoder_layers.append(DecoderLayer3D(last_layer_channels * 2 ** i, last_layer_channels * 2 ** (i - 1), output_padding=decoder_output_padding[-i]))
        decoder_layers.append(nn.Sequential(nn.ConvTranspose3d(last_layer_channels, input_c, 4, stride=2, padding=1, output_padding=[1, 0, 1], bias=False), nn.Sigmoid()))
        decoder = nn.Sequential(*decoder_layers)
        super(Vanilla3DVAE, self).__init__(encoder, decoder, mu_layer, var_layer, latent_space_size, gpu=gpu, recons_weight=recons_weight, KL_weight=KL_weight, is_3D=False)

    @staticmethod
    def get_input_size():
        return '1@128x128x128'

    @staticmethod
    def get_dimension():
        return '3D'

    @staticmethod
    def get_task():
        return ['reconstruction']


class Vanilla3DdenseVAE(BaseVAE):
    """
    This network is a 3D convolutional variational autoencoder with a dense latent space.

    reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, input_size, latent_space_size=256, feature_size=1024, n_conv=4, io_layer_channels=8, recons_weight=1, KL_weight=1, gpu=True):
        first_layer_channels = io_layer_channels
        last_layer_channels = io_layer_channels
        decoder_output_padding = []
        input_c = input_size[0]
        input_d = input_size[1]
        input_h = input_size[2]
        input_w = input_size[3]
        d, h, w = input_d, input_h, input_w
        encoder_layers = []
        encoder_layers.append(EncoderLayer3D(input_c, first_layer_channels))
        decoder_output_padding.append([d % 2, h % 2, w % 2])
        d, h, w = d // 2, h // 2, w // 2
        for i in range(n_conv - 1):
            encoder_layers.append(EncoderLayer3D(first_layer_channels * 2 ** i, first_layer_channels * 2 ** (i + 1)))
            decoder_output_padding.append([d % 2, h % 2, w % 2])
            d, h, w = d // 2, h // 2, w // 2
        n_pix = first_layer_channels * 2 ** (n_conv - 1) * (input_d // 2 ** n_conv) * (input_h // 2 ** n_conv) * (input_w // 2 ** n_conv)
        encoder_layers.append(Flatten())
        if feature_size == 0:
            feature_space = n_pix
        else:
            feature_space = feature_size
            encoder_layers.append(nn.Sequential(nn.Linear(n_pix, feature_space), nn.ReLU()))
        encoder = nn.Sequential(*encoder_layers)
        mu_layer = nn.Linear(feature_space, latent_space_size)
        var_layer = nn.Linear(feature_space, latent_space_size)
        decoder_layers = []
        if feature_size == 0:
            decoder_layers.append(nn.Sequential(nn.Linear(latent_space_size, n_pix), nn.ReLU()))
        else:
            decoder_layers.append(nn.Sequential(nn.Linear(latent_space_size, feature_size), nn.ReLU(), nn.Linear(feature_size, n_pix), nn.ReLU()))
        decoder_layers.append(Unflatten3D(last_layer_channels * 2 ** (n_conv - 1), input_d // 2 ** n_conv, input_h // 2 ** n_conv, input_w // 2 ** n_conv))
        for i in range(n_conv - 1, 0, -1):
            decoder_layers.append(DecoderLayer3D(last_layer_channels * 2 ** i, last_layer_channels * 2 ** (i - 1), output_padding=decoder_output_padding[i]))
        decoder_layers.append(nn.Sequential(nn.ConvTranspose3d(last_layer_channels, input_c, 4, stride=2, padding=1, output_padding=decoder_output_padding[0], bias=False), nn.Sigmoid()))
        decoder = nn.Sequential(*decoder_layers)
        super(Vanilla3DdenseVAE, self).__init__(encoder, decoder, mu_layer, var_layer, latent_space_size, gpu=gpu, is_3D=False, recons_weight=recons_weight, KL_weight=KL_weight)

    @staticmethod
    def get_input_size():
        return '1@128x128x128'

    @staticmethod
    def get_dimension():
        return '3D'

    @staticmethod
    def get_task():
        return ['reconstruction']


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNN,
     lambda: ([], {'convolutions': _mock_layer(), 'fc': _mock_layer(), 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecoderLayer2D,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderLayer3D,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (EncoderLayer2D,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderLayer3D,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PadMaxPool2d,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PadMaxPool3d,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Unflatten2D,
     lambda: ([], {'channel': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unflatten3D,
     lambda: ([], {'channel': 4, 'height': 4, 'width': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_aramis_lab_clinicadl(_paritybench_base):
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

