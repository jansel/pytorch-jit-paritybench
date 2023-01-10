import sys
_module = sys.modules[__name__]
del sys
run_convert_from_tf = _module
run_generator = _module
run_metrics = _module
run_projector = _module
run_training = _module
stylegan2 = _module
external_models = _module
inception = _module
lpips = _module
loss_fns = _module
metrics = _module
fid = _module
ppl = _module
models = _module
modules = _module
project = _module
train = _module
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


import re


import torch


import warnings


import numpy as np


from torch import multiprocessing as mp


import torch.nn as nn


import torch.nn.functional as F


from torchvision import models


from torch import nn


import torchvision


from torch.nn import functional as F


import numbers


import scipy


import copy


from collections import OrderedDict


import functools


import time


import torch.utils.tensorboard


import collections


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3FeatureExtractor(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_block=DEFAULT_BLOCK_INDEX, pixel_min=-1, pixel_max=1):
        """
        Build pretrained InceptionV3
        Arguments:
            output_block (int): Index of block to return features of.
                Possible values are:
                    - 0: corresponds to output of first max pooling
                    - 1: corresponds to output of second max pooling
                    - 2: corresponds to output which is fed to aux classifier
                    - 3: corresponds to output of final average pooling
            pixel_min (float): Min value for inputs. Default value is -1.
            pixel_max (float): Max value for inputs. Default value is 1.
        """
        super(InceptionV3FeatureExtractor, self).__init__()
        assert 0 <= output_block <= 3, '`output_block` can only be ' + '0 <= `output_block` <= 3.'
        inception = fid_inception_v3()
        blocks = []
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        blocks.append(nn.Sequential(*block0))
        if output_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            blocks.append(nn.Sequential(*block1))
        if output_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            blocks.append(nn.Sequential(*block2))
        if output_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            blocks.append(nn.Sequential(*block3))
        self.main = nn.Sequential(*blocks)
        self.pixel_nin = pixel_min
        self.pixel_max = pixel_max
        self.requires_grad_(False)
        self.eval()

    def _scale(self, x):
        if self.pixel_min != -1 or self.pixel_max != 1:
            x = (2 * x - self.pixel_min - self.pixel_max) / (self.pixel_max - self.pixel_min)
        return x

    def forward(self, input):
        """
        Get Inception feature maps.
        Arguments:
            input (torch.Tensor)
        Returns:
            feature_maps (torch.Tensor)
        """
        return self.main(input)


class LPIPS_VGG16(nn.Module):
    _FEATURE_IDX = [0, 4, 9, 16, 23, 30]
    _LINEAR_WEIGHTS_URL = 'https://github.com/richzhang/PerceptualSimilarity' + '/blob/master/lpips/weights/v0.1/vgg.pth?raw=true'

    def __init__(self, pixel_min=-1, pixel_max=1):
        super(LPIPS_VGG16, self).__init__()
        features = torchvision.models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        linear_weights = torch.utils.model_zoo.load_url(self._LINEAR_WEIGHTS_URL)
        for i in range(1, len(self._FEATURE_IDX)):
            idx_range = range(self._FEATURE_IDX[i - 1], self._FEATURE_IDX[i])
            self.slices.append(nn.Sequential(*[features[j] for j in idx_range]))
        self.linear_layers = nn.ModuleList()
        for weight in torch.utils.model_zoo.load_url(self._LINEAR_WEIGHTS_URL).values():
            weight = weight.view(1, -1)
            linear = nn.Linear(weight.size(1), 1, bias=False)
            linear.weight.data.copy_(weight)
            self.linear_layers.append(linear)
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188]).view(1, -1, 1, 1))
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45]).view(1, -1, 1, 1))
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.requires_grad_(False)
        self.eval()

    def _scale(self, x):
        if self.pixel_min != -1 or self.pixel_max != 1:
            x = (2 * x - self.pixel_min - self.pixel_max) / (self.pixel_max - self.pixel_min)
        return (x - self.shift) / self.scale

    @staticmethod
    def _normalize_tensor(feature_maps, eps=1e-08):
        rnorm = torch.rsqrt(torch.sum(feature_maps ** 2, dim=1, keepdim=True) + eps)
        return feature_maps * rnorm

    def forward(self, x0, x1, eps=1e-08):
        x0, x1 = self._scale(x0), self._scale(x1)
        dist = 0
        for slice, linear in zip(self.slices, self.linear_layers):
            x0, x1 = slice(x0), slice(x1)
            _x0, _x1 = self._normalize_tensor(x0, eps), self._normalize_tensor(x1, eps)
            dist += linear(torch.mean((_x0 - _x1) ** 2, dim=[-1, -2]))
        return dist.view(-1)


class _BaseModel(nn.Module):
    """
    Adds some base functionality to models that inherit this class.
    """

    def __init__(self):
        super(_BaseModel, self).__setattr__('kwargs', {})
        super(_BaseModel, self).__setattr__('_defaults', {})
        super(_BaseModel, self).__init__()

    def _update_kwargs(self, **kwargs):
        """
        Update the current keyword arguments. Overrides any
        default values set.
        Arguments:
            **kwargs: Keyword arguments
        """
        self.kwargs.update(**kwargs)

    def _update_default_kwargs(self, **defaults):
        """
        Update the default values for keyword arguments.
        Arguments:
            **defaults: Keyword arguments
        """
        self._defaults.update(**defaults)

    def __getattr__(self, name):
        """
        Try to get the keyword argument for this attribute.
        If no keyword argument of this name exists, try to
        get the attribute directly from this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        Returns:
            value
        """
        try:
            return self.__getattribute__('kwargs')[name]
        except KeyError:
            try:
                return self.__getattribute__('_defaults')[name]
            except KeyError:
                return super(_BaseModel, self).__getattr__(name)

    def __setattr__(self, name, value):
        """
        Try to set the keyword argument for this attribute.
        If no keyword argument of this name exists, set
        the attribute directly for this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
            value
        """
        if name != '__dict__' and (name in self.kwargs or name in self._defaults):
            self.kwargs[name] = value
        else:
            super(_BaseModel, self).__setattr__(name, value)

    def __delattr__(self, name):
        """
        Try to delete the keyword argument for this attribute.
        If no keyword argument of this name exists, delete
        the attribute of this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        """
        deleted = False
        if name in self.kwargs:
            del self.kwargs[name]
            deleted = True
        if name in self._defaults:
            del self._defaults[name]
            deleted = True
        if not deleted:
            super(_BaseModel, self).__delattr__(name)

    def clone(self):
        """
        Create a copy of this model.
        Returns:
            model_copy (nn.Module)
        """
        return copy.deepcopy(self)

    def _get_state_dict(self):
        """
        Delegate function for getting the state dict.
        Should be overridden if state dict has to be
        fetched in abnormal way.
        """
        return self.state_dict()

    def _set_state_dict(self, state_dict):
        """
        Delegate function for loading the state dict.
        Should be overridden if state dict has to be
        loaded in abnormal way.
        """
        self.load_state_dict(state_dict)

    def _serialize(self, half=False):
        """
        Turn model arguments and weights into
        a dict that can safely be pickled and unpickled.
        Arguments:
            half (bool): Save weights in half precision.
                Default value is False.
        """
        state_dict = self._get_state_dict()
        for key in state_dict.keys():
            values = state_dict[key].cpu()
            if torch.is_floating_point(values):
                if half:
                    values = values.half()
                else:
                    values = values.float()
            state_dict[key] = values
        return {'name': self.__class__.__name__, 'kwargs': self.kwargs, 'state_dict': state_dict}

    @classmethod
    def load(cls, fpath, map_location='cpu'):
        """
        Load a model of this class.
        Arguments:
            fpath (str): File path of saved model.
            map_location (str, int, torch.device): Weights and
                buffers will be loaded into this device.
                Default value is 'cpu'.
        """
        model = load(fpath, map_location=map_location)
        assert isinstance(model, cls), 'Trying to load a `{}` '.format(type(model)) + 'model from {}.load()'.format(cls.__name__)
        return model

    def save(self, fpath, half=False):
        """
        Save this model.
        Arguments:
            fpath (str): File path of save location.
            half (bool): Save weights in half precision.
                Default value is False.
        """
        torch.save(self._serialize(half=half), fpath)


class _BaseParameterizedModel(_BaseModel):
    """
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        lr_mul (float): The learning rate multiplier for this
            model. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value depends on model type and can
            be found in the original paper for StyleGAN.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8."""

    def __init__(self, **kwargs):
        super(_BaseParameterizedModel, self).__init__()
        self._update_default_kwargs(activation='lrelu:0.2', lr_mul=1, weight_scale=True, eps=1e-08)
        self._update_kwargs(**kwargs)


class GeneratorMapping(_BaseParameterizedModel):
    """
    Latent mapping model, handles the
    transformation of latents into disentangled
    latents.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. Unused by default.
        out_size (int, optional): The size of the disentangled
            latents output by this model. If not specified,
            the outputs will have the same size as the input
            latents.
        num_layers (int): Number of dense layers in this
            model. Default value is 8.
        hidden (int, optional): Number of hidden features of layers.
            If unspecified, this is the same size as the latents.
        normalize_input (bool): Normalize the input of this
            model. Default value is True."""
    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorMapping, self).__init__()
        self._update_default_kwargs(latent_size=512, label_size=0, out_size=None, num_layers=8, hidden=None, normalize_input=True, lr_mul=0.01)
        self._update_kwargs(**kwargs)
        in_features = self.latent_size
        out_features = self.hidden or self.latent_size
        self.embedding = None
        if self.label_size:
            self.embedding = nn.Embedding(self.label_size, self.latent_size)
            in_features += self.latent_size
        dense_layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                out_features = self.out_size or self.latent_size
            dense_layers.append(modules.BiasActivationWrapper(layer=modules.DenseLayer(in_features=in_features, out_features=out_features, lr_mul=self.lr_mul, weight_scale=self.weight_scale, gain=1), features=out_features, use_bias=True, activation=self.activation, bias_init=0, lr_mul=self.lr_mul, weight_scale=self.weight_scale))
            in_features = out_features
        self.main = nn.Sequential(*dense_layers)

    def forward(self, latents, labels=None):
        """
        Get the disentangled latents from the input latents
        and optional labels.
        Arguments:
            latents (torch.Tensor): Tensor of shape (batch_size, latent_size).
            labels (torch.Tensor, optional): Labels for conditioning of latents
                if there are any.
        Returns:
            dlatents (torch.Tensor): Disentangled latents of same shape as
                `latents` argument.
        """
        assert latents.dim() == 2 and latents.size(-1) == self.latent_size, 'Incorrect input shape. Should be ' + '(batch_size, {}) '.format(self.latent_size) + 'but received {}'.format(tuple(latents.size()))
        x = latents
        if labels is not None:
            assert self.embedding is not None, 'No embedding layer found, please ' + 'specify the number of possible labels ' + 'in the constructor of this class if ' + 'using labels.'
            assert len(labels) == len(latents), 'Received different number of labels ' + '({}) and latents ({}).'.format(len(labels), len(latents))
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.int64)
            assert labels.dtype == torch.int64, 'Labels should be integer values ' + 'of dtype torch.in64 (long)'
            y = self.embedding(labels)
            x = torch.cat([x, y], dim=-1)
        else:
            assert self.embedding is None, 'Missing input labels.'
        if self.normalize_input:
            x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.main(x)


class _BaseAdverserialModel(_BaseParameterizedModel):
    """
        data_channels (int): Number of channels of the data.
            Default value is 3.
        base_shape (list, tuple): This is the shape of the feature
            activations when it is most compact and still has the
            same number of dims as the data. This is one of the
            arguments that controls what shape the data will be.
            the value of each size in the shape is going to double
            in size for number of `channels` - 1.
            Example:
                `data_channels=3`
                `base_shape=(4, 2)`
                and 9 `channels` in total will give us a shape of
                (3, 4 * 2^(9 - 1), 2 * 2^(9 - 1)) which is the
                same as (3, 1024, 512).
            Default value is (4, 4).
        channels (int, list, optional): The channels of each block
            of layers. If int, this many channel values will be
            created with sensible default values optimal for image
            synthesis. If list, the number of blocks in this model
            will be the same as the number of channels in the list.
            Default value is the int value 9 which will create the
            following channels: [32, 32, 64, 128, 256, 512, 512, 512, 512].
            These are the channel values used in the stylegan2 paper for
            their FFHQ-trained face generation network.
            If channels is given as a list it should be in the order:
                Generator: last layer -> first layer
                Discriminator: first layer -> last layer
        resnet (bool): Use resnet connections.
            Defaults:
                Generator: False
                Discriminator: True
        skip (bool): Use skip connections for data.
            Defaults:
                Generator: True
                Discriminator: False
        fused_resample (bool): Fuse any up- or downsampling that
            is paired with a convolutional layer into a strided
            convolution (transposed if upsampling was used).
            Default value is True.
        conv_resample_mode (str): The resample mode of up- or
            downsampling layers. If `fused_resample=True` only
            'FIR' and 'none' can be used. Else, 'FIR' or anything
            that can be passed to torch.nn.functional.interpolate
            is a valid mode (and 'max' but only for downsampling
            operations). Default value is 'FIR'.
        conv_filter (int, list): The filter to use if
            `conv_resample_mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        skip_resample_mode (str): If `skip=True`, this
            mode is used for the resamplings of skip
            connections of different sizes. Same possible
            values as `conv_filter` (except 'none', which
            can not be used). Default value is 'FIR'.
        skip_filter (int, list): Same description as
            `conv_filter` but for skip connections.
            Only used if `skip_resample_mode='FIR'` and
            `skip=True`. Default value is a low pass
            filter of [1, 3, 3, 1].
        kernel_size (int): The size of the convolutional kernels.
            Default value is 3.
        conv_pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        conv_pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): The padding mode for FIR
            filters. Same possible values as `conv_pad_mode`.
            Default value is 'constant'.
        filter_pad_constant (float): The value to use for FIR
            padding if `filter_pad_mode='constant'`. Default
            value is 0.
        pad_once (bool): If FIR filter is used in conjunction with a
            conv layer, do all the padding for both convolution and
            FIR in the FIR layer instead of once per layer.
            Default value is True.
        conv_block_size (int): The number of conv layers in
            each conv block. Default value is 2."""
    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(_BaseAdverserialModel, self).__init__()
        self._update_default_kwargs(data_channels=3, base_shape=(4, 4), channels=9, resnet=False, skip=False, fused_resample=True, conv_resample_mode='FIR', conv_filter=[1, 3, 3, 1], skip_resample_mode='FIR', skip_filter=[1, 3, 3, 1], kernel_size=3, conv_pad_mode='constant', conv_pad_constant=0, filter_pad_mode='constant', filter_pad_constant=0, pad_once=True, conv_block_size=2)
        self._update_kwargs(**kwargs)
        self.dim = len(self.base_shape)
        assert 1 <= self.dim <= 3, '`base_shape` can only have 1, 2 or 3 dimensions.'
        if isinstance(self.channels, int):
            num_channels = self.channels
            self.channels = [min(32 * 2 ** i, 512) for i in range(min(8, num_channels))]
            if len(self.channels) < num_channels:
                self.channels = [32] * (num_channels - len(self.channels)) + self.channels


class GeneratorSynthesis(_BaseAdverserialModel):
    """
    The synthesis model that takes latents and synthesises
    some data.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        modulate_data_out (bool): Apply style to the data output
            layers. These layers are projections of the feature
            maps into the space of the data. Default value is True.
        noise (bool): Add noise after each conv style layer.
            Default value is True."""
    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorSynthesis, self).__init__()
        self._update_default_kwargs(latent_size=512, demodulate=True, modulate_data_out=True, noise=True, resnet=False, skip=True)
        self._update_kwargs(**kwargs)
        self.const = torch.nn.Parameter(torch.empty(self.channels[-1], *self.base_shape).normal_())
        conv_block_kwargs = dict(latent_size=self.latent_size, demodulate=self.demodulate, resnet=self.resnet, up=True, num_layers=self.conv_block_size, filter=self.conv_filter, activation=self.activation, mode=self.conv_resample_mode, fused=self.fused_resample, kernel_size=self.kernel_size, pad_mode=self.conv_pad_mode, pad_constant=self.conv_pad_constant, filter_pad_mode=self.filter_pad_mode, filter_pad_constant=self.filter_pad_constant, pad_once=self.pad_once, noise=self.noise, lr_mul=self.lr_mul, weight_scale=self.weight_scale, gain=1, dim=self.dim, eps=self.eps)
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(modules.GeneratorConvBlock(**{**conv_block_kwargs, 'in_channels': self.channels[-1], 'out_channels': self.channels[-1], 'resnet': False, 'up': False, 'num_layers': 1}))
        for i in range(1, len(self.channels)):
            self.conv_blocks.append(modules.GeneratorConvBlock(in_channels=self.channels[-i], out_channels=self.channels[-i - 1], **conv_block_kwargs))
        self.to_data_layers = nn.ModuleList()
        for i in range(1, len(self.channels) + 1):
            to_data = None
            if i == len(self.channels) or self.skip:
                to_data = modules.BiasActivationWrapper(layer=modules.ConvLayer(**{**conv_block_kwargs, 'in_channels': self.channels[-i], 'out_channels': self.data_channels, 'modulate': self.modulate_data_out, 'demodulate': False, 'kernel_size': 1}), **{**conv_block_kwargs, 'features': self.data_channels, 'use_bias': True, 'activation': 'linear', 'bias_init': 0})
            self.to_data_layers.append(to_data)
        self.upsample = None
        if self.skip:
            self.upsample = modules.Upsample(mode=self.skip_resample_mode, filter=self.skip_filter, filter_pad_mode=self.filter_pad_mode, filter_pad_constant=self.filter_pad_constant, gain=1, dim=self.dim)
        self._num_latents = 1 + self.conv_block_size * (len(self.channels) - 1)
        if self.modulate_data_out:
            self._num_latents += 1

    def __len__(self):
        """
        Get the number of affine (style) layers of this model.
        """
        return self._num_latents

    def random_noise(self):
        """
        Set injected noise to be random for each new input.
        """
        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                module.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        rtn_tensors = []
        if not self.noise:
            return rtn_tensors
        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                has_noise_shape = module.has_noise_shape()
                device = module.weight.device
                dtype = module.weight.dtype
                break
        if not has_noise_shape:
            with torch.no_grad():
                self(torch.zeros(1, len(self), self.latent_size, device=device, dtype=dtype))
        i = 0
        for block in self.conv_blocks:
            for layer in block.conv_block:
                for module in layer.modules():
                    if isinstance(module, modules.NoiseInjectionWrapper):
                        noise_tensor = None
                        if noise_tensors is not None:
                            if i < len(noise_tensors):
                                noise_tensor = noise_tensors[i]
                                i += 1
                            else:
                                rtn_tensors.append(None)
                                continue
                        rtn_tensors.append(module.static_noise(trainable=trainable, noise_tensor=noise_tensor))
        if noise_tensors is not None:
            assert len(rtn_tensors) == len(noise_tensors), 'Got a list of {} '.format(len(noise_tensors)) + 'noise tensors but there are ' + '{} noise layers in this model'.format(len(rtn_tensors))
        return rtn_tensors

    def forward(self, latents):
        """
        Synthesise some data from input latents.
        Arguments:
            latents (torch.Tensor): Latent vectors of shape
                (batch_size, num_affine_layers, latent_size)
                where num_affine_layers is the value returned
                by __len__() of this class.
        Returns:
            synthesised (torch.Tensor): Synthesised data.
        """
        assert latents.dim() == 3 and latents.size(1) == len(self), 'Input mismatch, expected latents of shape ' + '(batch_size, {}, latent_size) '.format(len(self)) + 'but got {}.'.format(tuple(latents.size()))
        x = self.const.unsqueeze(0)
        y = None
        layer_idx = 0
        for block, to_data in zip(self.conv_blocks, self.to_data_layers):
            block_latents = latents[:, layer_idx:layer_idx + len(block)]
            x = block(input=x, latents=block_latents)
            layer_idx += len(block)
            if self.upsample is not None and layer_idx < len(self):
                if y is not None:
                    y = self.upsample(y)
            if to_data is not None:
                t = to_data(input=x, latent=latents[:, layer_idx])
                y = t if y is None else y + t
        return y


class Generator(_BaseModel):
    """
    A wrapper class for the latent mapping model
    and synthesis (generator) model.
    Keyword Arguments:
        G_mapping (GeneratorMapping)
        G_synthesis (GeneratorSynthesis)
        dlatent_avg_beta (float): The beta value
            of the exponential moving average
            of the dlatents. This statistic
            is used for truncation of dlatents.
            Default value is 0.995
    """

    def __init__(self, *, G_mapping, G_synthesis, **kwargs):
        super(Generator, self).__init__()
        self._update_default_kwargs(dlatent_avg_beta=0.995)
        self._update_kwargs(**kwargs)
        assert isinstance(G_mapping, GeneratorMapping), '`G_mapping` has to be an instance of `model.GeneratorMapping`'
        assert isinstance(G_synthesis, GeneratorSynthesis), '`G_synthesis` has to be an instance of `model.GeneratorSynthesis`'
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.register_buffer('dlatent_avg', torch.zeros(self.G_mapping.latent_size))
        self.set_truncation()

    @property
    def latent_size(self):
        return self.G_mapping.latent_size

    @property
    def label_size(self):
        return self.G_mapping.label_size

    def _get_state_dict(self):
        state_dict = OrderedDict()
        self._save_to_state_dict(destination=state_dict, prefix='', keep_vars=False)
        return state_dict

    def _set_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

    def _serialize(self, half=False):
        state = super(Generator, self)._serialize(half=half)
        for name in ['G_mapping', 'G_synthesis']:
            state[name] = getattr(self, name)._serialize(half=half)
        return state

    def set_truncation(self, truncation_psi=None, truncation_cutoff=None):
        """
        Set the truncation of dlatents before they are passed to the
        synthesis model.
        Arguments:
            truncation_psi (float): Beta value of linear interpolation between
                the average dlatent and the current dlatent. 0 -> 100% average,
                1 -> 0% average.
            truncation_cutoff (int, optional): Truncation is only used up until
                this affine layer index.
        """
        layer_psi = None
        if truncation_psi is not None and truncation_psi != 1 and truncation_cutoff != 0:
            layer_psi = torch.ones(len(self.G_synthesis))
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi_mask = torch.arange(len(layer_psi)) < truncation_cutoff
                layer_psi[layer_psi_mask] *= truncation_psi
            layer_psi = layer_psi.view(1, -1, 1)
            layer_psi = layer_psi
        self.register_buffer('layer_psi', layer_psi)

    def random_noise(self):
        """
        Set noise of synthesis model to be random for every
        input.
        """
        self.G_synthesis.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        return self.G_synthesis.static_noise(trainable=trainable, noise_tensors=noise_tensors)

    def __len__(self):
        """
        Get the number of affine (style) layers of the synthesis model.
        """
        return len(self.G_synthesis)

    def truncate(self, dlatents):
        """
        Truncate the dlatents.
        Arguments:
            dlatents (torch.Tensor)
        Returns:
            truncated_dlatents (torch.Tensor)
        """
        if self.layer_psi is not None:
            dlatents = utils.lerp(self.dlatent_avg, dlatents, self.layer_psi)
        return dlatents

    def forward(self, latents=None, labels=None, dlatents=None, return_dlatents=False, mapping_grad=True, latent_to_layer_idx=None):
        """
        Synthesize some data from latent inputs. The latents
        can have an extra optional dimension, where latents
        from this dimension will be distributed to the different
        affine layers of the synthesis model. The distribution
        is a index to index mapping if the amount of latents
        is the same as the number of affine layers. Otherwise,
        latents are distributed consecutively for a random
        number of layers before the next latent is used for
        some random amount of following layers. If the size
        of this extra dimension is 1 or it does not exist,
        the same latent is passed to every affine layer.

        Latents are first mapped to disentangled latents (`dlatents`)
        and are then optionally truncated (if model is in eval mode
        and truncation options have been set.) Set up truncation by
        calling `set_truncation()`.
        Arguments:
            latents (torch.Tensor): The latent values of shape
                (batch_size, N, num_features) where N is an
                optional dimension. This argument is not required
                if `dlatents` is passed.
            labels (optional): A sequence of labels, one for
                each index in the batch dimension of the input.
            dlatents (torch.Tensor, optional): Skip the latent
                mapping model and feed these dlatents straight
                to the synthesis model. The same type of distribution
                to affine layers as is described in this function
                description is also used for dlatents.
                NOTE: Explicitly passing dlatents to this function
                    will stop them from being truncated. If required,
                    do this manually by calling the `truncate()` function
                    of this model.
            return_dlatents (bool): Return not only the synthesized
                data, but also the dlatents. The dlatents tensor
                will also have its `requires_grad` set to True
                before being passed to the synthesis model for
                use with pathlength regularization during training.
                This requires training to be enabled (`thismodel.train()`).
                Default value is False.
            mapping_grad (bool): Let gradients be calculated when passing
                latents through the latent mapping model. Should be
                set to False when only optimising the synthesiser parameters.
                Default value is True.
            latent_to_layer_idx (list, tuple, optional): A manual mapping
                of the latent vectors to the affine layers of this network.
                Each position in this sequence maps the affine layer of the
                same index to an index of the latents. The latents should
                have a shape of (batch_size, N, num_features) and this argument
                should be a list of the same length as number of affine layers
                in this model (can be found by calling len(thismodel)) with values
                in the range [0, N - 1]. Without this argument, latents are distributed
                according to this function description.
        """
        num_latents = 1
        truncate = False
        if dlatents is None:
            truncate = True
            assert latents is not None, 'Either the `latents` ' + 'or the `dlatents` argument is required.'
            if labels is not None:
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.int64)
            if latents.dim() == 3:
                num_latents = latents.size(1)
                latents = latents.view(-1, latents.size(-1))
                if labels is not None:
                    labels = labels.unsqueeze(1).repeat(1, num_latents).view(-1)
            with torch.set_grad_enabled(mapping_grad):
                dlatents = self.G_mapping(latents=latents, labels=labels)
        elif dlatents.dim() == 3:
            num_latents = dlatents.size(1)
        dlatents = dlatents.view(-1, num_latents, dlatents.size(-1))
        if num_latents == 1:
            dlatents = dlatents.expand(dlatents.size(0), len(self), dlatents.size(2))
        elif num_latents != len(self):
            assert dlatents.size(1) <= len(self), 'More latents ({}) than number '.format(dlatents.size(1)) + 'of generator layers ({}) received.'.format(len(self))
            if not latent_to_layer_idx:
                cutoffs = np.random.choice(np.arange(1, len(self)), dlatents.size(1) - 1, replace=False)
                cutoffs = [0] + sorted(cutoffs.tolist()) + [len(self)]
                dlatents = [dlatents[:, i].unsqueeze(1).expand(-1, cutoffs[i + 1] - cutoffs[i], dlatents.size(2)) for i in range(dlatents.size(1))]
                dlatents = torch.cat(dlatents, dim=1)
            else:
                assert len(latent_to_layer_idx) == len(self), 'The latent index to layer index mapping does ' + 'not have the same number of elements ' + '({}) as the number of '.format(len(latent_to_layer_idx)) + 'generator layers ({})'.format(len(self))
                dlatents = dlatents[:, latent_to_layer_idx]
        if self.training and self.dlatent_avg_beta != 1:
            with torch.no_grad():
                batch_dlatent_avg = dlatents[:, 0].mean(dim=0)
                self.dlatent_avg = utils.lerp(batch_dlatent_avg, self.dlatent_avg, self.dlatent_avg_beta)
        if truncate and not self.training:
            dlatents = self.truncate(dlatents)
        if return_dlatents and self.training:
            dlatents.requires_grad_(True)
        synth = self.G_synthesis(latents=dlatents)
        if return_dlatents:
            return synth, dlatents
        return synth


class Discriminator(_BaseAdverserialModel):
    """
    The discriminator scores data inputs.
    Keyword Arguments:
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. The discriminator will calculate scores
            for each possible label and only returns the score
            from the label passed with the input data. If no
            labels are used, only one score is calculated.
            Disabled by default.
        mbstd_group_size (int): Group size for minibatch std
            before the final conv layer. A value of 0 indicates
            not to use minibatch std, and a value of -1 indicates
            that the group should be over the entire batch.
            This is used for increasing variety of the outputs of
            the generator. Default value is 4.
            NOTE: Scores for the same data may vary depending
                on batch size when using a value of -1.
            NOTE: If a value > 0 is given, every input batch
                must have a size evenly divisible by this value.
        dense_hidden (int, optional): The number of hidden features
            of the first dense layer. By default, this is the same as
            the number of channels in the final conv layer."""
    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self._update_default_kwargs(label_size=0, mbstd_group_size=4, dense_hidden=None, resnet=True, skip=False)
        self._update_kwargs(**kwargs)
        conv_block_kwargs = dict(resnet=self.resnet, down=True, num_layers=self.conv_block_size, filter=self.conv_filter, activation=self.activation, mode=self.conv_resample_mode, fused=self.fused_resample, kernel_size=self.kernel_size, pad_mode=self.conv_pad_mode, pad_constant=self.conv_pad_constant, filter_pad_mode=self.filter_pad_mode, filter_pad_constant=self.filter_pad_constant, pad_once=self.pad_once, noise=False, lr_mul=self.lr_mul, weight_scale=self.weight_scale, gain=1, dim=self.dim, eps=self.eps)
        self.conv_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.conv_blocks.append(modules.DiscriminatorConvBlock(in_channels=self.channels[i], out_channels=self.channels[i + 1], **conv_block_kwargs))
        final_conv_block = []
        if self.mbstd_group_size:
            final_conv_block.append(modules.MinibatchStd(group_size=self.mbstd_group_size, eps=self.eps))
        final_conv_block.append(modules.DiscriminatorConvBlock(**{**conv_block_kwargs, 'in_channels': self.channels[-1] + (1 if self.mbstd_group_size else 0), 'out_channels': self.channels[-1], 'resnet': False, 'down': False, 'num_layers': 1}))
        self.conv_blocks.append(nn.Sequential(*final_conv_block))
        self.from_data_layers = nn.ModuleList()
        for i in range(len(self.channels)):
            from_data = None
            if i == 0 or self.skip:
                from_data = modules.BiasActivationWrapper(layer=modules.ConvLayer(**{**conv_block_kwargs, 'in_channels': self.data_channels, 'out_channels': self.channels[i], 'modulate': False, 'demodulate': False, 'kernel_size': 1}), **{**conv_block_kwargs, 'features': self.channels[i], 'use_bias': True, 'activation': self.activation, 'bias_init': 0})
            self.from_data_layers.append(from_data)
        self.downsample = None
        if self.skip:
            self.downsample = modules.Downsample(mode=self.skip_resample_mode, filter=self.skip_filter, filter_pad_mode=self.filter_pad_mode, filter_pad_constant=self.filter_pad_constant, gain=1, dim=self.dim)
        dense_layers = []
        in_features = self.channels[-1] * np.prod(self.base_shape)
        out_features = self.dense_hidden or self.channels[-1]
        activation = self.activation
        for _ in range(2):
            dense_layers.append(modules.BiasActivationWrapper(layer=modules.DenseLayer(in_features=in_features, out_features=out_features, lr_mul=self.lr_mul, weight_scale=self.weight_scale, gain=1), features=out_features, activation=activation, use_bias=True, bias_init=0, lr_mul=self.lr_mul, weight_scale=self.weight_scale))
            in_features = out_features
            out_features = max(1, self.label_size)
            activation = 'linear'
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, input, labels=None):
        """
        Takes some data and optionally its labels and
        produces one score logit per data input.
        Arguments:
            input (torch.Tensor)
            labels (torch.Tensor, list, optional)
        Returns:
            score_logits (torch.Tensor)
        """
        x = None
        y = input
        for i, (block, from_data) in enumerate(zip(self.conv_blocks, self.from_data_layers)):
            if from_data is not None:
                t = from_data(y)
                x = t if x is None else x + t
            x = block(input=x)
            if self.downsample is not None and i != len(self.conv_blocks) - 1:
                y = self.downsample(y)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        if labels is not None:
            x = x[torch.arange(x.size(0)), labels].unsqueeze(-1)
        return x


class Swish(nn.Module):
    """
    Performs the 'Swish' non-linear activation function.
    https://arxiv.org/pdf/1710.05941.pdf
    Arguments:
        affine (bool): Multiply the input to sigmoid
            with a learnable scale. Default value is False.
    """

    def __init__(self, affine=False):
        super(Swish, self).__init__()
        if affine:
            self.beta = nn.Parameter(torch.tensor([1.0]))
        self.affine = affine

    def forward(self, input, *args, **kwargs):
        """
        Apply the swish non-linear activation function
        and return the results.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        if self.affine:
            x *= self.beta
        return x * torch.sigmoid(x)


def _get_weight_and_coef(shape, lr_mul=1, weight_scale=True, gain=1, fill=None):
    """
    Get an intialized weight and its runtime coefficients as an nn.Parameter tensor.
    Arguments:
        shape (tuple, list): Shape of weight tensor.
        lr_mul (float): The learning rate multiplier for
            this weight. Default value is 1.
        weight_scale (bool): Use weight scaling for equalized
            learning rate. Default value is True.
        gain (float): The gain of the weight. Default value is 1.
        fill (float, optional): Instead of initializing the weight
            with scaled normally distributed values, fill it with
            this value. Useful for bias weights.
    Returns:
        weight (nn.Parameter)
    """
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)
    if weight_scale:
        init_std = 1 / lr_mul
        runtime_coef = he_std * lr_mul
    else:
        init_std = he_std / lr_mul
        runtime_coef = lr_mul
    weight = torch.empty(*shape)
    if fill is None:
        weight.normal_(0, init_std)
    else:
        weight.fill_(fill)
    return nn.Parameter(weight), runtime_coef


def get_activation(activation):
    """
    Get the module for a specific activation function and its gain if
    it can be calculated.
    Arguments:
        activation (str, callable, nn.Module): String representing the activation.
    Returns:
        activation_module (torch.nn.Module): The module representing
            the activation function.
        gain (float): The gain value. Defaults to 1 if it can not be calculated.
    """
    if isinstance(activation, nn.Module) or callable(activation):
        return activation, 1.0
    if isinstance(activation, str):
        activation = activation.lower()
    if activation in [None, 'linear']:
        return nn.Identity(), 1.0
    lrelu_strings = 'leaky', 'leakyrely', 'leaky_relu', 'leaky relu', 'lrelu'
    if activation.startswith(lrelu_strings):
        for l_s in lrelu_strings:
            activation = activation.replace(l_s, '')
        slope = ''.join(char for char in activation if char.isdigit() or char == '.')
        slope = float(slope) if slope else 0.01
        return nn.LeakyReLU(slope), np.sqrt(2)
    elif activation.startswith('swish'):
        return Swish(affine=activation != 'swish'), np.sqrt(2)
    elif activation in ['relu']:
        return nn.ReLU(), np.sqrt(2)
    elif activation in ['elu']:
        return nn.ELU(), 1.0
    elif activation in ['prelu']:
        return nn.PReLU(), np.sqrt(2)
    elif activation in ['rrelu', 'randomrelu']:
        return nn.RReLU(), np.sqrt(2)
    elif activation in ['selu']:
        return nn.SELU(), 1.0
    elif activation in ['softplus']:
        return nn.Softplus(), 1
    elif activation in ['softsign']:
        return nn.Softsign(), 1
    elif activation in ['sigmoid', 'logistic']:
        return nn.Sigmoid(), 1.0
    elif activation in ['tanh']:
        return nn.Tanh(), 1.0
    else:
        raise ValueError('Activation "{}" not available.'.format(activation))


class BiasActivationWrapper(nn.Module):
    """
    Wrap a module to add bias and non-linear activation
    to any outputs of that module.
    Arguments:
        layer (nn.Module): The module to wrap.
        features (int, optional): The number of features
            of the output of the `layer`. This argument
            has to be specified if `use_bias=True`.
        use_bias (bool): Add bias to the output.
            Default value is True.
        activation (str, nn.Module, callable, optional):
            non-linear activation function to use.
            Unused if notspecified.
        bias_init (float): Value to initialize bias
            weight with. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the bias weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
    """

    def __init__(self, layer, features=None, use_bias=True, activation='linear', bias_init=0, lr_mul=1, weight_scale=True, *args, **kwargs):
        super(BiasActivationWrapper, self).__init__()
        self.layer = layer
        bias = None
        bias_coef = None
        if use_bias:
            assert features, '`features` is required when using bias.'
            bias, bias_coef = _get_weight_and_coef(shape=[features], lr_mul=lr_mul, weight_scale=False, fill=bias_init)
        self.register_parameter('bias', bias)
        self.bias_coef = bias_coef
        self.act, self.gain = get_activation(activation)

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        bias (if set) and run through non-linear activation
        function (if set).
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        if self.bias is not None:
            bias = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            if self.bias_coef != 1:
                bias = self.bias_coef * bias
            x += bias
        x = self.act(x)
        if self.gain != 1:
            x *= self.gain
        return x

    def extra_repr(self):
        return 'bias={}'.format(self.bias is not None)


class NoiseInjectionWrapper(nn.Module):
    """
    Wrap a module to add noise scaled by a
    learnable parameter to any outputs of the
    wrapped module.
    Noise is randomized for each output but can
    be set to static noise by calling `static_noise()`
    of this object. This can only be done once data
    has passed through this layer at least once so that
    the shape of the static noise to create is known.
    Check if the shape is known by calling `has_noise_shape()`.
    Arguments:
        layer (nn.Module): The module to wrap.
        same_over_batch (bool): Repeat the same
            noise values over the entire batch
            instead of creating separate noise
            values for each entry in the batch.
            Default value is True.
    """

    def __init__(self, layer, same_over_batch=True):
        super(NoiseInjectionWrapper, self).__init__()
        self.layer = layer
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('noise_storage', None)
        self.same_over_batch = same_over_batch
        self.random_noise()

    def has_noise_shape(self):
        """
        If this module has had data passed through it
        the noise shape is known and this function returns
        True. Else False.
        Returns:
            noise_shape_known (bool)
        """
        return self.noise_storage is not None

    def random_noise(self):
        """
        Randomize noise for each
        new output.
        """
        self._fixed_noise = False
        if isinstance(self.noise_storage, nn.Parameter):
            noise_storage = self.noise_storage
            del self.noise_storage
            self.register_buffer('noise_storage', noise_storage.data)

    def static_noise(self, trainable=False, noise_tensor=None):
        """
        Set up static noise that can optionally be a trainable
        parameter. Static noise does not change between inputs
        unless the user has altered its values. Returns the tensor
        object that stores the static noise.
        Arguments:
            trainable (bool): Wrap the static noise tensor in
                nn.Parameter to make it trainable. The returned
                tensor will be wrapped.
            noise_tensor (torch.Tensor, optional): A predefined
                static noise tensor. If not passed, one will be
                created.
        """
        assert self.has_noise_shape(), 'Noise shape is unknown'
        if noise_tensor is None:
            noise_tensor = self.noise_storage
        else:
            noise_tensor = noise_tensor
        if trainable and not isinstance(noise_tensor, nn.Parameter):
            noise_tensor = nn.Parameter(noise_tensor)
        if isinstance(self.noise_storage, nn.Parameter) and not trainable:
            del self.noise_storage
            self.register_buffer('noise_storage', noise_tensor)
        else:
            self.noise_storage = noise_tensor
        self._fixed_noise = True
        return noise_tensor

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Saves module state to `destination` dictionary, containing a state
        submodule in :meth:`~torch.nn.Module.state_dict`.

        Overridden to ignore the noise storage buffer.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if name != 'noise_storage' and param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if name != 'noise_storage' and buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Overridden to ignore noise storage buffer.
        """
        key = prefix + 'noise_storage'
        if key in state_dict:
            del state_dict[key]
        return super(NoiseInjectionWrapper, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        noise to its outputs before returning them.
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        noise_shape = list(x.size())
        noise_shape[1] = 1
        if self.same_over_batch:
            noise_shape[0] = 1
        if self.noise_storage is None or list(self.noise_storage.size()) != noise_shape:
            if not self._fixed_noise:
                self.noise_storage = torch.empty(*noise_shape, dtype=self.weight.dtype, device=self.weight.device)
            else:
                assert list(self.noise_storage.size()[2:]) == noise_shape[2:], 'A data size {} has been encountered, '.format(x.size()[2:]) + 'the static noise previously set up does ' + 'not match this size {}'.format(self.noise_storage.size()[2:])
                assert self.noise_storage.size(0) == 1 or self.noise_storage.size(0) == x.size(0), 'Static noise batch size mismatch! ' + 'Noise batch size: {}, '.format(self.noise_storage.size(0)) + 'input batch size: {}'.format(x.size(0))
                assert self.noise_storage.size(1) == 1 or self.noise_storage.size(1) == x.size(1), 'Static noise channel size mismatch! ' + 'Noise channel size: {}, '.format(self.noise_storage.size(1)) + 'input channel size: {}'.format(x.size(1))
        if not self._fixed_noise:
            self.noise_storage.normal_()
        x += self.weight * self.noise_storage
        return x

    def extra_repr(self):
        return 'static_noise={}'.format(self._fixed_noise)


def _apply_conv(input, *args, transpose=False, **kwargs):
    """
    Perform a 1d, 2d or 3d convolution with specified
    positional and keyword arguments. Which type of
    convolution that is used depends on shape of data.
    Arguments:
        input (torch.Tensor): The input data for the
            convolution.
        *args: Positional arguments for the convolution.
    Keyword Arguments:
        transpose (bool): Transpose the convolution.
            Default value is False
        **kwargs: Keyword arguments for the convolution.
    """
    dim = input.dim() - 2
    conv_fn = getattr(F, 'conv{}{}d'.format('_transpose' if transpose else '', dim))
    return conv_fn(*args, input=input, **kwargs)


class FilterLayer(nn.Module):
    """
    Apply a filter by using convolution.
    Arguments:
        filter_kernel (torch.Tensor): The filter kernel to use.
            Should be of shape `dims * (k,)` where `k` is the
            kernel size and `dims` is the number of data dimensions
            (excluding batch and channel dimension).
        stride (int): The stride of the convolution.
        pad0 (int): Amount to pad start of each data dimension.
            Default value is 0.
        pad1 (int): Amount to pad end of each data dimension.
            Default value is 0.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
    """

    def __init__(self, filter_kernel, stride=1, pad0=0, pad1=0, pad_mode='constant', pad_constant=0, *args, **kwargs):
        super(FilterLayer, self).__init__()
        dim = filter_kernel.dim()
        filter_kernel = filter_kernel.view(1, 1, *filter_kernel.size())
        self.register_buffer('filter_kernel', filter_kernel)
        self.stride = stride
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
            self.pad_mode = pad_mode
            self.pad_constant = pad_constant

    def forward(self, input, **kwargs):
        """
        Pad the input and run the filter over it
        before returning the new values.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        conv_kwargs = dict(weight=self.filter_kernel.repeat(input.size(1), *([1] * (self.filter_kernel.dim() - 1))), stride=self.stride, groups=input.size(1))
        if self.fused_pad:
            conv_kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, transpose=False, **conv_kwargs)

    def extra_repr(self):
        return 'filter_size={}, stride={}'.format(tuple(self.filter_kernel.size()[2:]), self.stride)


def _setup_filter_kernel(filter_kernel, gain=1, up_factor=1, dim=2):
    """
    Set up a filter kernel and return it as a tensor.
    Arguments:
        filter_kernel (int, list, torch.tensor, None): The filter kernel
            values to use. If this value is an int, a binomial filter of
            this size is created. If a sequence with a single axis is used,
            it will be expanded to the number of `dims` specified. If value
            is None, a filter of values [1, 1] is used.
        gain (float): Gain of the filter kernel. Default value is 1.
        up_factor (int): Scale factor. Should only be given for upscaling filters.
            Default value is 1.
        dim (int): Number of dimensions of data. Default value is 2.
    Returns:
        filter_kernel_tensor (torch.Tensor)
    """
    filter_kernel = filter_kernel or 2
    if isinstance(filter_kernel, (int, float)):

        def binomial(n, k):
            if k in [1, n]:
                return 1
            return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
        filter_kernel = [binomial(filter_kernel, k) for k in range(1, filter_kernel + 1)]
    if not torch.is_tensor(filter_kernel):
        filter_kernel = torch.tensor(filter_kernel)
    filter_kernel = filter_kernel.float()
    if filter_kernel.dim() == 1:
        _filter_kernel = filter_kernel.unsqueeze(0)
        while filter_kernel.dim() < dim:
            filter_kernel = torch.matmul(filter_kernel.unsqueeze(-1), _filter_kernel)
    assert all(filter_kernel.size(0) == s for s in filter_kernel.size())
    filter_kernel /= filter_kernel.sum()
    filter_kernel *= gain * up_factor ** 2
    return filter_kernel.float()


class Upsample(nn.Module):
    """
    Performs upsampling without learnable parameters that doubles
    the size of data.
    Arguments:
        mode (str): 'FIR' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self, mode='FIR', filter=[1, 3, 3, 1], filter_pad_mode='constant', filter_pad_constant=0, gain=1, dim=2, *args, **kwargs):
        super(Upsample, self).__init__()
        assert mode != 'max', "mode 'max' can only be used for downsampling."
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=gain, up_factor=2, dim=dim)
            pad = filter_kernel.size(-1) - 1
            self.filter = FilterLayer(filter_kernel=filter_kernel, pad0=(pad + 1) // 2 + 1, pad1=pad // 2, pad_mode=filter_pad_mode, pad_constant=filter_pad_constant)
            self.register_buffer('weight', torch.ones(*[(1) for _ in range(dim + 2)]))
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Upsample inputs.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = _apply_conv(input=input, weight=self.weight.expand(input.size(1), *self.weight.size()[1:]), groups=input.size(1), stride=2, transpose=True)
            x = self.filter(x)
        else:
            interp_kwargs = dict(scale_factor=2, mode=self.mode)
            if 'linear' in self.mode or 'cubic' in self.mode:
                interp_kwargs.update(align_corners=False)
            x = F.interpolate(input, **interp_kwargs)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class Downsample(nn.Module):
    """
    Performs downsampling without learnable parameters that
    reduces size of data by half.
    Arguments:
        mode (str): 'FIR', 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self, mode='FIR', filter=[1, 3, 3, 1], filter_pad_mode='constant', filter_pad_constant=0, gain=1, dim=2, *args, **kwargs):
        super(Downsample, self).__init__()
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=gain, up_factor=1, dim=dim)
            pad = filter_kernel.size(-1) - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            self.filter = FilterLayer(filter_kernel=filter_kernel, stride=2, pad0=pad0, pad1=pad1, pad_mode=filter_pad_mode, pad_constant=filter_pad_constant)
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Downsample inputs to half its size.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = self.filter(input)
        elif self.mode == 'max':
            return getattr(F, 'max_pool{}d'.format(input.dim() - 2))(input)
        else:
            x = F.interpolate(input, scale_factor=0.5, mode=self.mode)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class MinibatchStd(nn.Module):
    """
    Adds the aveage std of each data point over a
    slice of the minibatch to that slice as a new
    feature map. This gives an output with one extra
    channel.
    Arguments:
        group_size (int): Number of entries in each slice
            of the batch. If <= 0, the entire batch is used.
            Default value is 4.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(self, group_size=4, eps=1e-08, *args, **kwargs):
        super(MinibatchStd, self).__init__()
        if group_size is None or group_size <= 0:
            group_size = 0
        assert group_size != 1, 'Can not use 1 as minibatch std group size.'
        self.group_size = group_size
        self.eps = eps

    def forward(self, input, **kwargs):
        """
        Add a new feature map to the input containing the average
        standard deviation for each slice.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        group_size = self.group_size or input.size(0)
        assert input.size(0) >= group_size, 'Can not use a smaller batch size ' + '({}) than the specified '.format(input.size(0)) + 'group size ({}) '.format(group_size) + 'of this minibatch std layer.'
        assert input.size(0) % group_size == 0, 'Can not use a batch of a size ' + '({}) that is not '.format(input.size(0)) + 'evenly divisible by the group size ({})'.format(group_size)
        x = input
        y = input.view(group_size, -1, *input.size()[1:])
        y = y.float()
        y -= y.mean(dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + self.eps)
        y = torch.mean(y.view(y.size(0), -1), dim=-1)
        y = y.view(-1, *([1] * (input.dim() - 1)))
        y = y
        y = y.repeat(group_size, *([1] * (y.dim() - 1)))
        y = y.expand(y.size(0), 1, *x.size()[2:])
        x = torch.cat([x, y], dim=1)
        return x

    def extra_repr(self):
        return 'group_size={}'.format(self.group_size or '-1')


class DenseLayer(nn.Module):
    """
    A fully connected layer.
    NOTE: No bias is applied in this layer.
    Arguments:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
    """

    def __init__(self, in_features, out_features, lr_mul=1, weight_scale=True, gain=1, *args, **kwargs):
        super(DenseLayer, self).__init__()
        weight, weight_coef = _get_weight_and_coef(shape=[out_features, in_features], lr_mul=lr_mul, weight_scale=weight_scale, gain=gain)
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef

    def forward(self, input, **kwargs):
        """
        Perform a matrix multiplication of the weight
        of this layer and the input.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        return input.matmul(weight.t())

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.weight.size(1), self.weight.size(0))


class ConvLayer(nn.Module):
    """
    A convolutional layer that can have its outputs
    modulated (style mod). It can also normalize outputs.
    These operations are done by modifying the convolutional
    kernel weight and employing grouped convolutions for
    efficiency.
    NOTE: No bias is applied in this layer.
    NOTE: Amount of padding used is the same as 'SAME'
        argument in tensorflow for conv padding.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int, optional): The size of the
            latents to use for modulating this convolution.
            Only required when `modulate=True`.
        modulate (bool): Applies a "style" to the outputs
            of the layer. The style is given by a latent
            vector passed with the input to this layer.
            A dense layer is added that projects the
            values of the latent into scales for the
            data channels.
            Default value is False.
        demodulate (bool): Normalize std of outputs.
            Can only be set to True when `modulate=True`.
            Default value is False.
        kernel_size (int): The size of the kernel.
            Default value is 3.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(self, in_channels, out_channels, latent_size=None, modulate=False, demodulate=False, kernel_size=3, pad_mode='constant', pad_constant=0, lr_mul=1, weight_scale=True, gain=1, dim=2, eps=1e-08, *args, **kwargs):
        super(ConvLayer, self).__init__()
        assert modulate or not demodulate, '`demodulate=True` can ' + 'only be used when `modulate=True`'
        if modulate:
            assert latent_size is not None, 'When using `modulate=True`, ' + '`latent_size` has to be specified.'
        kernel_shape = [out_channels, in_channels] + dim * [kernel_size]
        weight, weight_coef = _get_weight_and_coef(shape=kernel_shape, lr_mul=lr_mul, weight_scale=weight_scale, gain=gain)
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef
        if modulate:
            self.dense = BiasActivationWrapper(layer=DenseLayer(in_features=latent_size, out_features=in_channels, lr_mul=lr_mul, weight_scale=weight_scale, gain=1), features=in_channels, use_bias=True, activation='linear', bias_init=1, lr_mul=lr_mul, weight_scale=weight_scale)
        self.dense_reshape = [-1, 1, in_channels] + dim * [1]
        self.dmod_reshape = [-1, out_channels, 1] + dim * [1]
        pad = kernel_size - 1
        pad0 = pad - pad // 2
        pad1 = pad - pad0
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
        self.pad_mode = pad_mode
        self.pad_constant = pad_constant
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.modulate = modulate
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.lr_mul = lr_mul
        self.weight_scale = weight_scale
        self.gain = gain
        self.dim = dim
        self.eps = eps

    def forward_mod(self, input, latent, weight, **kwargs):
        """
        Run the forward operation with modulation.
        Automatically called from `forward()` if modulation
        is enabled.
        """
        assert latent is not None, 'A latent vector is ' + 'required for the forwad pass of a modulated conv layer.'
        style_mod = self.dense(input=latent)
        style_mod = style_mod.view(*self.dense_reshape)
        weight = weight.unsqueeze(0)
        weight = weight * style_mod
        if self.demodulate:
            dmod = torch.rsqrt(torch.sum(weight.view(weight.size(0), weight.size(1), -1) ** 2, dim=-1) + self.eps)
            dmod = dmod.view(*self.dmod_reshape)
            weight = weight * dmod
        x = input.view(1, -1, *input.size()[2:])
        weight = weight.view(-1, *weight.size()[2:])
        x = self._process(input=x, weight=weight, groups=input.size(0))
        x = x.view(-1, self.out_channels, *x.size()[2:])
        return x

    def forward(self, input, latent=None, **kwargs):
        """
        Convolve the input.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor, optional)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        if self.modulate:
            return self.forward_mod(input=input, latent=latent, weight=weight)
        return self._process(input=input, weight=weight)

    def _process(self, input, weight, **kwargs):
        """
        Pad input and convolve it returning the result.
        """
        x = input
        if self.fused_pad:
            kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, weight=weight, transpose=False, **kwargs)

    def extra_repr(self):
        string = 'in_channels={}, out_channels={}'.format(self.weight.size(1), self.weight.size(0))
        string += ', modulate={}, demodulate={}'.format(self.modulate, self.demodulate)
        return string


def _setup_mod_weight_for_t_conv(weight, in_channels, out_channels):
    """
    Reshape a modulated conv weight for use with a transposed convolution.
    Arguments:
        weight (torch.Tensor)
        in_channels (int)
        out_channels (int)
    Returns:
        reshaped_weight (torch.Tensor)
    """
    weight = weight.view(-1, out_channels, in_channels, *weight.size()[2:])
    weight = weight.transpose(1, 2)
    weight = weight.reshape(-1, out_channels, *weight.size()[3:])
    return weight


class ConvUpLayer(ConvLayer):
    """
    A convolutional upsampling layer that doubles the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the upsampling operation with the
            convolution, turning this layer into a strided transposed
            convolution. Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(self, *args, fused=True, mode='FIR', filter=[1, 3, 3, 1], filter_pad_mode='constant', filter_pad_constant=0, pad_once=True, **kwargs):
        super(ConvUpLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], 'Fused conv upsample can only use ' + "'FIR' or 'none' for resampling " + '(`mode` argument).'
            self.padding = np.ceil(self.kernel_size / 2 - 1)
            self.output_padding = 2 * (self.padding + 1) - self.kernel_size
            if not self.modulate:
                self.weight = nn.Parameter(self.weight.data.transpose(0, 1).contiguous())
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=self.gain, up_factor=2, dim=self.dim)
                if pad_once:
                    self.padding = 0
                    self.output_padding = 0
                    pad = filter_kernel.size(-1) - 2 - (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2 + 1,
                    pad1 = pad // 2 + 1,
                else:
                    pad = filter_kernel.size(-1) - 1
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(filter_kernel=filter_kernel, pad0=pad0, pad1=pad1, pad_mode=filter_pad_mode, pad_constant=filter_pad_constant)
        else:
            assert mode != 'none', "'none' can not be used as " + 'sampling `mode` when `fused=False` as upsampling ' + 'has to be performed separately from the conv layer.'
            self.upsample = Upsample(mode=mode, filter=filter, filter_pad_mode=filter_pad_mode, filter_pad_constant=filter_pad_constant, channels=self.in_channels, gain=self.gain, dim=self.dim)
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            if self.modulate:
                weight = _setup_mod_weight_for_t_conv(weight=weight, in_channels=self.in_channels, out_channels=self.out_channels)
            pad_out = False
            if self.pad_mode == 'constant' and self.pad_constant == 0:
                if self.filter is not None or not self.pad_once:
                    kwargs.update(padding=self.padding, output_padding=self.output_padding)
            elif self.filter is None:
                if self.padding:
                    x = F.pad(x, [self.padding] * 2 * self.dim, mode=self.pad_mode, value=self.pad_constant)
                pad_out = self.output_padding != 0
            kwargs.update(stride=2)
            x = _apply_conv(input=x, weight=weight, transpose=True, **kwargs)
            if pad_out:
                x = F.pad(x, [self.output_padding, 0] * self.dim, mode=self.pad_mode, value=self.pad_constant)
            if self.filter is not None:
                x = self.filter(x)
        else:
            x = super(ConvUpLayer, self)._process(input=self.upsample(input=x), weight=weight, **kwargs)
        return x

    def extra_repr(self):
        string = super(ConvUpLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


class ConvDownLayer(ConvLayer):
    """
    A convolutional downsampling layer that halves the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the downsampling operation with the
            convolution, turning this layer into a strided convolution.
            Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(self, *args, fused=True, mode='FIR', filter=[1, 3, 3, 1], filter_pad_mode='constant', filter_pad_constant=0, pad_once=True, **kwargs):
        super(ConvDownLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], 'Fused conv downsample can only use ' + "'FIR' or 'none' for resampling " + '(`mode` argument).'
            pad = self.kernel_size - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            if pad0 == pad1 and (pad0 == 0 or self.pad_mode == 'constant' and self.pad_constant == 0):
                self.fused_pad = True
                self.padding = pad0
            else:
                self.fused_pad = False
                self.padding = [pad0, pad1] * self.dim
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=self.gain, up_factor=1, dim=self.dim)
                if pad_once:
                    self.fused_pad = True
                    self.padding = 0
                    pad = filter_kernel.size(-1) - 2 + (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2,
                    pad1 = pad // 2,
                else:
                    pad = filter_kernel.size(-1) - 1
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(filter_kernel=filter_kernel, pad0=pad0, pad1=pad1, pad_mode=filter_pad_mode, pad_constant=filter_pad_constant)
                self.pad_once = pad_once
        else:
            assert mode != 'none', "'none' can not be used as " + 'sampling `mode` when `fused=False` as downsampling ' + 'has to be performed separately from the conv layer.'
            self.downsample = Downsample(mode=mode, filter=filter, pad_mode=filter_pad_mode, pad_constant=filter_pad_constant, channels=self.in_channels, gain=self.gain, dim=self.dim)
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            kwargs.update(stride=2)
            if self.filter is not None:
                x = self.filter(input=x)
        else:
            x = self.downsample(input=x)
        x = super(ConvDownLayer, self)._process(input=x, weight=weight, **kwargs)
        return x

    def extra_repr(self):
        string = super(ConvDownLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


def _get_layer(layer_class, kwargs, wrap=False, noise=False):
    """
    Create a layer and wrap it in optional
    noise and/or bias/activation layers.
    Arguments:
        layer_class: The class of the layer to construct.
        kwargs (dict): The keyword arguments to use for constructing
            the layer and optionally the bias/activaiton layer.
        wrap (bool): Wrap the layer in an bias/activation layer and
            optionally a noise injection layer. Default value is False.
        noise (bool): Inject noise before the bias/activation wrapper.
            This can only be done when `wrap=True`. Default value is False.
    """
    layer = layer_class(**kwargs)
    if wrap:
        if noise:
            layer = NoiseInjectionWrapper(layer)
        layer = BiasActivationWrapper(layer, **kwargs)
    return layer


class GeneratorConvBlock(nn.Module):
    """
    A convblock for the synthesiser model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int): The size of the latent vectors.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        up (bool): Upsample the data to twice its size. This is
            performed in the first layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `up=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of upsampling layers.
            Only used when `up=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `up=True`, fuse the upsample operation
            and the first convolutional layer into a transposed
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        noise (bool): Add noise to the output of each layer. Default value
            is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(self, in_channels, out_channels, latent_size, demodulate=True, resnet=False, up=False, num_layers=2, filter=[1, 3, 3, 1], activation='leaky:0.2', mode='FIR', fused=True, kernel_size=3, pad_mode='constant', pad_constant=0, filter_pad_mode='constant', filter_pad_constant=0, pad_once=True, use_bias=True, noise=True, lr_mul=1, weight_scale=True, gain=1, dim=2, eps=1e-08, *args, **kwargs):
        super(GeneratorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(features=out_channels, modulate=True)
        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if up:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, '`mode` {} '.format(mode) + 'is not one of the available sample ' + 'modes {}.'.format(available_sampling)
        self.conv_block = nn.ModuleList()
        while len(self.conv_block) < num_layers:
            use_up = up and not self.conv_block
            self.conv_block.append(_get_layer(ConvUpLayer if use_up else ConvLayer, layer_kwargs, wrap=True, noise=noise))
            layer_kwargs.update(in_channels=out_channels)
        self.projection = None
        if resnet:
            projection_kwargs = {**layer_kwargs, 'in_channels': in_channels, 'kernel_size': 1, 'modulate': False, 'demodulate': False}
            self.projection = _get_layer(ConvUpLayer if up else ConvLayer, projection_kwargs, wrap=False)
        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, latents=None, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if latents.dim() == 2:
            latents.unsqueeze(1)
        if latents.size(1) == 1:
            latents = latents.repeat(1, len(self), 1)
        assert latents.size(1) == len(self), 'Number of latent inputs ' + '({}) does not match '.format(latents.size(1)) + 'number of conv layers ' + '({}) in block.'.format(len(self))
        x = input
        for i, layer in enumerate(self.conv_block):
            x = layer(input=x, latent=latents[:, i])
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x


class DiscriminatorConvBlock(nn.Module):
    """
    A convblock for the discriminator model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        down (bool): Downsample the data to twice its size. This is
            performed in the last layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `down=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of downsampling layers.
            Only used when `down=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, 'max' or anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `down=True`, fuse the downsample operation
            and the last convolutional layer into a strided
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
    """

    def __init__(self, in_channels, out_channels, resnet=False, down=False, num_layers=2, filter=[1, 3, 3, 1], activation='leaky:0.2', mode='FIR', fused=True, kernel_size=3, pad_mode='constant', pad_constant=0, filter_pad_mode='constant', filter_pad_constant=0, pad_once=True, use_bias=True, lr_mul=1, weight_scale=True, gain=1, dim=2, *args, **kwargs):
        super(DiscriminatorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(out_channels=in_channels, features=in_channels, modulate=False, demodulate=False)
        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if down:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('max')
                available_sampling.append('area')
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, '`mode` {} '.format(mode) + 'is not one of the available sample ' + 'modes {}'.format(available_sampling)
        self.conv_block = nn.ModuleList()
        while len(self.conv_block) < num_layers:
            if len(self.conv_block) == num_layers - 1:
                layer_kwargs.update(out_channels=out_channels, features=out_channels)
            use_down = down and len(self.conv_block) == num_layers - 1
            self.conv_block.append(_get_layer(ConvDownLayer if use_down else ConvLayer, layer_kwargs, wrap=True, noise=False))
        self.projection = None
        if resnet:
            projection_kwargs = {**layer_kwargs, 'in_channels': in_channels, 'kernel_size': 1, 'modulate': False, 'demodulate': False}
            self.projection = _get_layer(ConvDownLayer if down else ConvLayer, projection_kwargs, wrap=False)
        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        for layer in self.conv_block:
            x = layer(input=x)
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x


class Projector(nn.Module):
    """
    Projects data to latent space and noise tensors.
    Arguments:
        G (Generator)
        dlatent_avg_samples (int): Number of dlatent samples
            to collect to find the mean and std.
            Default value is 10 000.
        dlatent_avg_label (int, torch.Tensor, optional): The label to
            use when gathering dlatent statistics.
        dlatent_device (int, str, torch.device, optional): Device to use
            for gathering statistics of dlatents. By default uses
            the same device as parameters of `G` reside on.
        dlatent_batch_size (int): The batch size to sample
            dlatents with. Default value is 1024.
        lpips_model (nn.Module): A model that returns feature the distance
            between two inputs. Default value is the LPIPS VGG16 model.
        lpips_size (int, optional): Resize any data fed to `lpips_model` by scaling
            the data so that its smallest side is the same size as this
            argument. Only has a default value of 256 if `lpips_model` is unspecified.
        verbose (bool): Write progress of dlatent statistics gathering to stdout.
            Default value is True.
    """

    def __init__(self, G, dlatent_avg_samples=10000, dlatent_avg_label=None, dlatent_device=None, dlatent_batch_size=1024, lpips_model=None, lpips_size=None, verbose=True):
        super(Projector, self).__init__()
        assert isinstance(G, models.Generator)
        G.eval().requires_grad_(False)
        self.G_synthesis = G.G_synthesis
        G_mapping = G.G_mapping
        dlatent_batch_size = min(dlatent_batch_size, dlatent_avg_samples)
        if dlatent_device is None:
            dlatent_device = next(G_mapping.parameters()).device()
        else:
            dlatent_device = torch.device(dlatent_device)
        G_mapping
        latents = torch.empty(dlatent_avg_samples, G_mapping.latent_size).normal_()
        dlatents = []
        labels = None
        if dlatent_avg_label is not None:
            labels = torch.tensor(dlatent_avg_label).long().view(-1).repeat(dlatent_batch_size)
        if verbose:
            progress = utils.ProgressWriter(np.ceil(dlatent_avg_samples / dlatent_batch_size))
            progress.write('Gathering dlatents...', step=False)
        for i in range(0, dlatent_avg_samples, dlatent_batch_size):
            batch_latents = latents[i:i + dlatent_batch_size]
            batch_labels = None
            if labels is not None:
                batch_labels = labels[:len(batch_latents)]
            with torch.no_grad():
                dlatents.append(G_mapping(batch_latents, labels=batch_labels).cpu())
            if verbose:
                progress.step()
        if verbose:
            progress.write('Done!', step=False)
            progress.close()
        dlatents = torch.cat(dlatents, dim=0)
        self.register_buffer('_dlatent_avg', dlatents.mean(dim=0).view(1, 1, -1))
        self.register_buffer('_dlatent_std', torch.sqrt(torch.sum((dlatents - self._dlatent_avg) ** 2) / dlatent_avg_samples + 1e-08).view(1, 1, 1))
        if lpips_model is None:
            warnings.warn('Using default LPIPS distance metric based on VGG 16. ' + 'This metric will only work on image data where values are in ' + 'the range [-1, 1], please specify an lpips module if you want ' + 'to use other kinds of data formats.')
            lpips_model = lpips.LPIPS_VGG16(pixel_min=-1, pixel_max=1)
            lpips_size = 256
        self.lpips_model = lpips_model.eval().requires_grad_(False)
        self.lpips_size = lpips_size
        self

    def _scale_for_lpips(self, data):
        if not self.lpips_size:
            return data
        scale_factor = self.lpips_size / min(data.size()[2:])
        if scale_factor == 1:
            return data
        mode = 'nearest'
        if scale_factor < 1:
            mode = 'area'
        return F.interpolate(data, scale_factor=scale_factor, mode=mode)

    def _check_job(self):
        assert self._job is not None, 'Call `start()` first to set up target.'
        if self._job.dlatent_param.device != self._dlatent_avg.device:
            self._job.dlatent_param = self._job.dlatent_param
            self._job.opt.load_state_dict(utils.move_to_device(self._job.opt.state_dict(), self._dlatent_avg.device)[0])

    def generate(self):
        """
        Generate an output with the current dlatent and noise values.
        Returns:
            output (torch.Tensor)
        """
        self._check_job()
        with torch.no_grad():
            return self.G_synthesis(self._job.dlatent_param)

    def get_dlatent(self):
        """
        Get a copy of the current dlatent values.
        Returns:
            dlatents (torch.Tensor)
        """
        self._check_job()
        return self._job.dlatent_param.data.clone()

    def get_noise(self):
        """
        Get a copy of the current noise values.
        Returns:
            noise_tensors (list)
        """
        self._check_job()
        return [noise.data.clone() for noise in self._job.noise_params]

    def start(self, target, num_steps=1000, initial_learning_rate=0.1, initial_noise_factor=0.05, lr_rampdown_length=0.25, lr_rampup_length=0.05, noise_ramp_length=0.75, regularize_noise_weight=100000.0, verbose=True, verbose_prefix=''):
        """
        Set up a target and its projection parameters.
        Arguments:
            target (torch.Tensor): The data target. This should
                already be preprocessed (scaled to correct value range).
            num_steps (int): Number of optimization steps. Default
                value is 1000.
            initial_learning_rate (float): Default value is 0.1.
            initial_noise_factor (float): Default value is 0.05.
            lr_rampdown_length (float): Default value is 0.25.
            lr_rampup_length (float): Default value is 0.05.
            noise_ramp_length (float): Default value is 0.75.
            regularize_noise_weight (float): Default value is 1e5.
            verbose (bool): Write progress to stdout every time
                `step()` is called.
            verbose_prefix (str, optional): This is written before
                any other output to stdout.
        """
        if target.dim() == self.G_synthesis.dim + 1:
            target = target.unsqueeze(0)
        assert target.dim() == self.G_synthesis.dim + 2, 'Number of dimensions of target data is incorrect.'
        target = target
        target_scaled = self._scale_for_lpips(target)
        dlatent_param = nn.Parameter(self._dlatent_avg.clone().repeat(target.size(0), len(self.G_synthesis), 1))
        noise_params = self.G_synthesis.static_noise(trainable=True)
        params = [dlatent_param] + noise_params
        opt = torch.optim.Adam(params)
        noise_tensor = torch.empty_like(dlatent_param)
        if verbose:
            progress = utils.ProgressWriter(num_steps)
            value_tracker = utils.ValueTracker()
        self._job = utils.AttributeDict(**locals())
        self._job.current_step = 0

    def step(self, steps=1):
        """
        Take a projection step.
        Arguments:
            steps (int): Number of steps to take. If this
                exceeds the remaining steps of the projection
                that amount of steps is taken instead. Default
                value is 1.
        """
        self._check_job()
        remaining_steps = self._job.num_steps - self._job.current_step
        if not remaining_steps > 0:
            warnings.warn('Trying to take a projection step after the ' + 'final projection iteration has been completed.')
        if steps < 0:
            steps = remaining_steps
        steps = min(remaining_steps, steps)
        if not steps > 0:
            return
        for _ in range(steps):
            if self._job.current_step >= self._job.num_steps:
                break
            t = self._job.current_step / self._job.num_steps
            noise_strength = self._dlatent_std * self._job.initial_noise_factor * max(0.0, 1.0 - t / self._job.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self._job.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self._job.lr_rampup_length)
            learning_rate = self._job.initial_learning_rate * lr_ramp
            for param_group in self._job.opt.param_groups:
                param_group['lr'] = learning_rate
            dlatents = self._job.dlatent_param + noise_strength * self._job.noise_tensor.normal_()
            output = self.G_synthesis(dlatents)
            assert output.size() == self._job.target.size(), 'target size {} does not fit output size {} of generator'.format(target.size(), output.size())
            output_scaled = self._scale_for_lpips(output)
            lpips_distance = torch.mean(self.lpips_model(output_scaled, self._job.target_scaled))
            reg_loss = 0
            for p in self._job.noise_params:
                size = min(p.size()[2:])
                dim = p.dim() - 2
                while True:
                    reg_loss += torch.mean((p * p.roll(shifts=[1] * dim, dims=list(range(2, 2 + dim)))) ** 2)
                    if size <= 8:
                        break
                    p = F.interpolate(p, scale_factor=0.5, mode='area')
                    size = size // 2
            loss = lpips_distance + self._job.regularize_noise_weight * reg_loss
            self._job.opt.zero_grad()
            loss.backward()
            self._job.opt.step()
            for p in self._job.noise_params:
                with torch.no_grad():
                    p_mean = p.mean(dim=list(range(1, p.dim())), keepdim=True)
                    p_rstd = torch.rsqrt(torch.mean((p - p_mean) ** 2, dim=list(range(1, p.dim())), keepdim=True) + 1e-08)
                    p.data = (p.data - p_mean) * p_rstd
            self._job.current_step += 1
            if self._job.verbose:
                self._job.value_tracker.add('loss', float(loss))
                self._job.value_tracker.add('lpips_distance', float(lpips_distance))
                self._job.value_tracker.add('noise_reg', float(reg_loss))
                self._job.value_tracker.add('lr', learning_rate, beta=0)
                self._job.progress.write(self._job.verbose_prefix, str(self._job.value_tracker))
                if self._job.current_step >= self._job.num_steps:
                    self._job.progress.close()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvDownLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvUpLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiscriminatorConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MinibatchStd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoiseInjectionWrapper,
     lambda: ([], {'layer': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_adriansahlman_stylegan2_pytorch(_paritybench_base):
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

