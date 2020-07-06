import sys
_module = sys.modules[__name__]
del sys
TkTorchWindow = _module
config = _module
decomposition = _module
estimators = _module
interactive = _module
models = _module
biggan = _module
pytorch_pretrained_biggan = _module
convert_tf_to_pytorch = _module
file_utils = _module
model = _module
utils = _module
setup = _module
stylegan = _module
model = _module
stylegan2 = _module
wrappers = _module
netdissect = _module
aceoptimize = _module
aceplotablate = _module
acesummarize = _module
actviz = _module
autoeval = _module
broden = _module
dissection = _module
easydict = _module
evalablate = _module
fullablate = _module
modelconfig = _module
nethook = _module
parallelfolder = _module
pidfile = _module
plotutil = _module
proggan = _module
progress = _module
runningstats = _module
sampler = _module
segdata = _module
segmenter = _module
segmodel = _module
models = _module
resnet = _module
resnext = _module
segviz = _module
server = _module
serverstate = _module
statedict = _module
allunitsample = _module
ganseg = _module
makesample = _module
upsegmodel = _module
models = _module
prroi_pool = _module
build = _module
functional = _module
prroi_pool = _module
test_prroi_pooling2d = _module
resnet = _module
resnext = _module
workerpool = _module
zdataset = _module
notebook_init = _module
notebook_utils = _module
layerwise_z_test = _module
partial_forward_test = _module
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


import numpy as np


import time


import torch


from torch.autograd import Variable


import re


from types import SimpleNamespace


from scipy.cluster.vq import kmeans


from torch.nn.functional import interpolate


from functools import partial


from itertools import chain


import logging


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.functional import normalize


import math


from collections import OrderedDict


import random


from abc import abstractmethod


from abc import ABC as AbstractBaseClass


import numbers


import numpy


from torchvision import transforms


from torch.utils.data import TensorDataset


from scipy.ndimage.morphology import binary_dilation


from torchvision.datasets.folder import default_loader


from scipy import ndimage


import types


import torchvision


from collections import defaultdict


import torch.utils.data as data


import itertools


from torch.utils.data.sampler import Sampler


from torchvision.transforms.functional import to_tensor


from torchvision.transforms.functional import normalize


from torch.utils.data import DataLoader


from collections.abc import MutableMapping


from collections.abc import Mapping


from scipy.io import savemat


import torch.autograd as ag


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        out = x + self.gamma * attn_g
        return out


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


class BigGANBatchNorm(nn.Module):
    """ This is a batch norm module that can handle conditional input and can be provided with pre-computed
        activation means and variances for various truncation parameters.

        We cannot just rely on torch.batch_norm since it cannot handle
        batched weights (pytorch 1.0.1). We computate batch_norm our-self without updating running means and variances.
        If you want to train this model you should add running means and variance computation logic.
    """

    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=0.0001, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional
        self.register_buffer('running_means', torch.zeros(n_stats, num_features))
        self.register_buffer('running_vars', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)
        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:
            running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]
        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)
            out = (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=False, momentum=0.0, eps=self.eps)
        return out


class GenBlock(nn.Module):

    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False, n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        middle_size = in_size // reduction_factor
        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)
        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)
        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)
        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)
        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x
        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)
        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)
        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)
        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, (...)]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')
        out = x + x0
        return out


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2
        self.gen_z = snlinear(in_features=condition_vector_dim, out_features=4 * 4 * 16 * ch, eps=config.eps)
        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch * layer[1], eps=config.eps))
            layers.append(GenBlock(ch * layer[1], ch * layer[2], condition_vector_dim, up_sample=layer[0], n_stats=config.n_stats, eps=config.eps))
        self.layers = nn.ModuleList(layers)
        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation):
        z = self.gen_z(cond_vector[0])
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()
        cond_idx = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector[cond_idx], truncation)
                cond_idx += 1
            else:
                z = layer(z)
        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, (...)]
        z = self.tanh(z)
        return z


class BaseModel(AbstractBaseClass, torch.nn.Module):

    def __init__(self, model_name, class_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.outclass = class_name

    @abstractmethod
    def partial_forward(self, x, layer_name):
        pass

    @abstractmethod
    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        pass

    def get_max_latents(self):
        return 1

    def latent_space_name(self):
        return 'Z'

    def get_latent_shape(self):
        return tuple(self.sample_latent(1).shape)

    def get_latent_dims(self):
        return np.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    def forward(self, x):
        out = self.model.forward(x)
        return 0.5 * (out + 1)

    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            z = [(torch.tensor(l) if not torch.is_tensor(l) else l) for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()

    def get_conditional_state(self, z):
        return None

    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)


class BigGAN(BaseModel):

    def __init__(self, device, resolution, class_name, truncation=1.0):
        super(BigGAN, self).__init__(f'BigGAN-{resolution}', class_name)
        self.device = device
        self.truncation = truncation
        self.load_model(f'biggan-deep-{resolution}')
        self.set_output_class(class_name or 'husky')
        self.name = f'BigGAN-{resolution}-{self.outclass}-t{self.truncation}'
        self.has_latent_residual = True

    def load_model(self, name):
        if name not in biggan.model.PRETRAINED_MODEL_ARCHIVE_MAP:
            raise RuntimeError('Unknown BigGAN model name', name)
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        model_path = Path(checkpoint_root) / name
        os.makedirs(model_path, exist_ok=True)
        model_file = model_path / biggan.model.WEIGHTS_NAME
        config_file = model_path / biggan.model.CONFIG_NAME
        model_url = biggan.model.PRETRAINED_MODEL_ARCHIVE_MAP[name]
        config_url = biggan.model.PRETRAINED_CONFIG_ARCHIVE_MAP[name]
        for filename, url in ((model_file, model_url), (config_file, config_url)):
            if not filename.is_file():
                None
                with open(filename, 'wb') as f:
                    if url.startswith('s3://'):
                        biggan.s3_get(url, f)
                    else:
                        biggan.http_get(url, f)
        self.model = biggan.BigGAN.from_pretrained(model_path)

    def sample_latent(self, n_samples=1, truncation=None, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        noise_vector = biggan.truncated_noise_sample(truncation=truncation or self.truncation, batch_size=n_samples, seed=seed)
        noise = torch.from_numpy(noise_vector)
        return noise

    def get_max_latents(self):
        return len(self.model.config.layers) + 1

    def get_conditional_state(self, z):
        return self.v_class

    def set_conditional_state(self, z, c):
        self.v_class = c

    def is_valid_class(self, class_id):
        if isinstance(class_id, int):
            return class_id < 1000
        elif isinstance(class_id, str):
            return biggan.one_hot_from_names([class_id.replace(' ', '_')]) is not None
        else:
            raise RuntimeError(f'Unknown class identifier {class_id}')

    def set_output_class(self, class_id):
        if isinstance(class_id, int):
            self.v_class = torch.from_numpy(biggan.one_hot_from_int([class_id]))
            self.outclass = f'class{class_id}'
        elif isinstance(class_id, str):
            self.outclass = class_id.replace(' ', '_')
            self.v_class = torch.from_numpy(biggan.one_hot_from_names([class_id]))
        else:
            raise RuntimeError(f'Unknown class identifier {class_id}')

    def forward(self, x):
        if isinstance(x, list):
            c = self.v_class.repeat(x[0].shape[0], 1)
            class_vector = len(x) * [c]
        else:
            class_vector = self.v_class.repeat(x.shape[0], 1)
        out = self.model.forward(x, class_vector, self.truncation)
        return 0.5 * (out + 1)

    def partial_forward(self, x, layer_name):
        if layer_name in ['embeddings', 'generator.gen_z']:
            n_layers = 0
        elif 'generator.layers' in layer_name:
            layer_base = re.match('^generator\\.layers\\.[0-9]+', layer_name)[0]
            n_layers = int(layer_base.split('.')[-1]) + 1
        else:
            n_layers = len(self.model.config.layers)
        if not isinstance(x, list):
            x = self.model.n_latents * [x]
        if isinstance(self.v_class, list):
            labels = [c.repeat(x[0].shape[0], 1) for c in class_label]
            embed = [self.model.embeddings(l) for l in labels]
        else:
            class_label = self.v_class.repeat(x[0].shape[0], 1)
            embed = len(x) * [self.model.embeddings(class_label)]
        assert len(x) == self.model.n_latents, f'Expected {self.model.n_latents} latents, got {len(x)}'
        assert len(embed) == self.model.n_latents, f'Expected {self.model.n_latents} class vectors, got {len(class_label)}'
        cond_vectors = [torch.cat((z, e), dim=1) for z, e in zip(x, embed)]
        z = self.model.generator.gen_z(cond_vectors[0])
        z = z.view(-1, 4, 4, 16 * self.model.generator.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()
        cond_idx = 1
        for i, layer in enumerate(self.model.generator.layers[:n_layers]):
            if isinstance(layer, biggan.GenBlock):
                z = layer(z, cond_vectors[cond_idx], self.truncation)
                cond_idx += 1
            else:
                z = layer(z)
        return None


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** -0.5
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True, intermediate=None, upscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** -0.5
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)
        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)
        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):

    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)
        x = x * (style[:, (0)] + 1.0) + style[:, (1)]
        return x


class PixelNormLayer(nn.Module):

    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


class BlurLayer(nn.Module):

    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, (None)] * kernel[(None), :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1))
        return x


class G_mapping(nn.Sequential):

    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)), 'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [('pixel_norm', PixelNormLayer()), ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense0_act', act), ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense1_act', act), ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense2_act', act), ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense3_act', act), ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense4_act', act), ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense5_act', act), ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense6_act', act), ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense7_act', act)]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        return super().forward(x)


class Truncation(nn.Module):

    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer('avg_latent', avg_latent)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNorm()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):

    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale)
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, (0)]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, (0)])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, (1)])
        return x


class GSynthesisBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale, intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, (0)])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, (1)])
        return x


class G_synthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_styles=True, const_input_layer=True, use_noise=True, randomize_noise=True, nonlinearity='lrelu', use_wscale=True, use_pixel_norm=False, use_instance_norm=True, dtype=torch.float32, blur_filter=[1, 2, 1]):
        super().__init__()

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        act, gain = {'relu': (torch.relu, np.sqrt(2)), 'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            if res == 2:
                blocks.append((name, InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            else:
                blocks.append((name, GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents_in):
        batch_size = dlatents_in.size(0)
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2 * i:2 * i + 2])
            else:
                x = m(x, dlatents_in[:, 2 * i:2 * i + 2])
        rgb = self.torgb(x)
        return rgb


class StyleGAN_G(nn.Sequential):

    def __init__(self, resolution, truncation=1.0):
        self.resolution = resolution
        self.layers = OrderedDict([('g_mapping', G_mapping()), ('g_synthesis', G_synthesis(resolution=resolution))])
        super().__init__(self.layers)

    def forward(self, x, latent_is_w=False):
        if isinstance(x, list):
            assert len(x) == 18, 'Must provide 1 or 18 latents'
            if not latent_is_w:
                x = [self.layers['g_mapping'].forward(l) for l in x]
            x = torch.stack(x, dim=1)
        else:
            if not latent_is_w:
                x = self.layers['g_mapping'].forward(x)
            x = x.unsqueeze(1).expand(-1, 18, -1)
        x = self.layers['g_synthesis'].forward(x)
        return x

    def load_weights(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))

    def export_from_tf(self, pickle_path):
        module_path = Path(__file__).parent / 'stylegan_tf'
        sys.path.append(str(module_path.resolve()))
        import torch
        import collections
        dnnlib.tflib.init_tf()
        weights = pickle.load(open(pickle_path, 'rb'))
        weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k, v in w.trainables.items()]) for w in weights]
        state_G, state_D, state_Gs = weights_pt

        def key_translate(k):
            k = k.lower().split('/')
            if k[0] == 'g_synthesis':
                if not k[1].startswith('torgb'):
                    k.insert(1, 'blocks')
                k = '.'.join(k)
                k = k.replace('const.const', 'const').replace('const.bias', 'bias').replace('const.stylemod', 'epi1.style_mod.lin').replace('const.noise.weight', 'epi1.top_epi.noise.weight').replace('conv.noise.weight', 'epi2.top_epi.noise.weight').replace('conv.stylemod', 'epi2.style_mod.lin').replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight').replace('conv0_up.stylemod', 'epi1.style_mod.lin').replace('conv1.noise.weight', 'epi2.top_epi.noise.weight').replace('conv1.stylemod', 'epi2.style_mod.lin').replace('torgb_lod0', 'torgb')
            else:
                k = '.'.join(k)
            return k

        def weight_translate(k, w):
            k = key_translate(k)
            if k.endswith('.weight'):
                if w.dim() == 2:
                    w = w.t()
                elif w.dim() == 1:
                    pass
                else:
                    assert w.dim() == 4
                    w = w.permute(3, 2, 0, 1)
            return w
        param_dict = {key_translate(k): weight_translate(k, v) for k, v in state_Gs.items() if 'torgb_lod' not in key_translate(k)}
        if 1:
            sd_shapes = {k: v.shape for k, v in self.state_dict().items()}
            param_shapes = {k: v.shape for k, v in param_dict.items()}
            for k in (list(sd_shapes) + list(param_shapes)):
                pds = param_shapes.get(k)
                sds = sd_shapes.get(k)
                if pds is None:
                    None
                elif sds is None:
                    None
                elif sds != pds:
                    None
        self.load_state_dict(param_dict, strict=False)
        torch.save(self.state_dict(), Path(pickle_path).with_suffix('.pt'))


def download_manual(url, output_name):
    outpath = Path(output_name).resolve()
    while not outpath.is_file():
        None
        None
        input('Press any key to continue...')


def download_generic(url, output_name):
    None
    session = requests.Session()
    r = session.get(url, allow_redirects=True)
    r.raise_for_status()
    if r.encoding is None:
        with open(output_name, 'wb') as f:
            f.write(r.content)
    else:
        download_manual(url, output_name)


def download_google_drive(url, output_name):
    None
    session = requests.Session()
    r = session.get(url, allow_redirects=True)
    r.raise_for_status()
    if r.encoding is not None:
        tokens = re.search('(confirm=.+)&amp;id', str(r.content))
        assert tokens is not None, 'Could not extract token from response'
        url = url.replace('id=', f'{tokens[1]}&id=')
        r = session.get(url, allow_redirects=True)
        r.raise_for_status()
    assert r.encoding is None, f'Failed to download weight file from {url}'
    with open(output_name, 'wb') as f:
        f.write(r.content)


def download_ckpt(url, output_name):
    if 'drive.google' in url:
        download_google_drive(url, output_name)
    elif 'mega.nz' in url:
        download_manual(url, output_name)
    else:
        download_generic(url, output_name)


class StyleGAN2(BaseModel):

    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN2, self).__init__('StyleGAN2', class_name or 'ffhq')
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w
        configs = {'ffhq': 1024, 'car': 512, 'cat': 256, 'church': 256, 'horse': 256, 'bedrooms': 256, 'kitchen': 256, 'places': 256}
        assert self.outclass in configs, f"Invalid StyleGAN2 class {self.outclass}, should be one of [{', '.join(configs.keys())}]"
        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN2-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def download_checkpoint(self, outfile):
        checkpoints = {'horse': 'https://drive.google.com/uc?export=download&id=18SkqWAkgt0fIwDEf2pqeaenNi4OoCo-0', 'ffhq': 'https://drive.google.com/uc?export=download&id=1FJRwzAkV-XWbxgTwxEmEACvuqF5DsBiV', 'church': 'https://drive.google.com/uc?export=download&id=1HFM694112b_im01JT7wop0faftw9ty5g', 'car': 'https://drive.google.com/uc?export=download&id=1iRoWclWVbDBAy5iXYZrQnKYSbZUqXI6y', 'cat': 'https://drive.google.com/uc?export=download&id=15vJP8GDr0FlRYpE8gD7CdeEz2mXrQMgN', 'places': 'https://drive.google.com/uc?export=download&id=1X8-wIH3aYKjgDZt4KMOtQzN1m4AlCVhm', 'bedrooms': 'https://drive.google.com/uc?export=download&id=1nZTW7mjazs-qPhkmbsOLLA_6qws-eNQu', 'kitchen': 'https://drive.google.com/uc?export=download&id=15dCpnZ1YLAnETAPB0FGmXwdBclbwMEkZ'}
        url = checkpoints[self.outclass]
        download_ckpt(url, outfile)

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt'
        self.model = stylegan2.Generator(self.resolution, 512, 8)
        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            self.download_checkpoint(checkpoint)
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['g_ema'], strict=False)
        self.latent_avg = ckpt['latent_avg']

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        rng = np.random.RandomState(seed)
        z = torch.from_numpy(rng.standard_normal(512 * n_samples).reshape(n_samples, 512)).float()
        if self.w_primary:
            z = self.model.style(z)
        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN2: cannot change output class without reloading')

    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        out, _ = self.model(x, noise=self.noise, truncation=self.truncation, truncation_latent=self.latent_avg, input_is_w=self.w_primary)
        return 0.5 * (out + 1)

    def partial_forward(self, x, layer_name):
        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise
        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]
        if len(styles) == 1:
            inject_index = self.model.n_latent
            latent = self.model.strided_style(styles[0].unsqueeze(1).repeat(1, inject_index, 1))
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)
            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            assert len(styles) == self.model.n_latent, f'Expected {self.model.n_latents} latents, got {len(styles)}'
            styles = torch.stack(styles, dim=1)
            latent = self.model.strided_style(styles)
        if 'style' in layer_name:
            return
        out = self.model.input(latent)
        if 'input' == layer_name:
            return
        out = self.model.conv1(out, latent[:, (0)], noise=noise[0])
        if 'conv1' in layer_name:
            return
        skip = self.model.to_rgb1(out, latent[:, (1)])
        if 'to_rgb1' in layer_name:
            return
        i = 1
        noise_i = 1
        for conv1, conv2, to_rgb in zip(self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs):
            out = conv1(out, latent[:, (i)], noise=noise[noise_i])
            if f'convs.{i - 1}' in layer_name:
                return
            out = conv2(out, latent[:, (i + 1)], noise=noise[noise_i + 1])
            if f'convs.{i}' in layer_name:
                return
            skip = to_rgb(out, latent[:, (i + 2)], skip)
            if f'to_rgbs.{i // 2}' in layer_name:
                return
            i += 2
            noise_i += 2
        image = skip
        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]
        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))


class StyleGAN(BaseModel):

    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN, self).__init__('StyleGAN', class_name or 'ffhq')
        self.device = device
        self.w_primary = use_w
        configs = {'ffhq': 1024, 'celebahq': 1024, 'bedrooms': 256, 'cars': 512, 'cats': 256, 'vases': 1024, 'wikiart': 512, 'fireworks': 512, 'abstract': 512, 'anime': 512, 'ukiyo-e': 512}
        assert self.outclass in configs, f"Invalid StyleGAN class {self.outclass}, should be one of [{', '.join(configs.keys())}]"
        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'stylegan/stylegan_{self.outclass}_{self.resolution}.pt'
        self.model = stylegan.StyleGAN_G(self.resolution)
        urls_tf = {'vases': 'https://thisvesseldoesnotexist.s3-us-west-2.amazonaws.com/public/network-snapshot-008980.pkl', 'fireworks': 'https://mega.nz/#!7uBHnACY!quIW-pjdDa7NqnZOYh1z5UemWwPOW6HkYSoJ4usCg9U', 'abstract': 'https://mega.nz/#!vCQyHQZT!zdeOg3VvT4922Z2UfxO51xgAfJD-NAK2nW7H_jMlilU', 'anime': 'https://mega.nz/#!vawjXISI!F7s13yRicxDA3QYqYDL2kjnc2K7Zk3DwCIYETREmBP4', 'ukiyo-e': 'https://drive.google.com/uc?id=1CHbJlci9NhVFifNQb3vCGu6zw4eqzvTd'}
        urls_torch = {'celebahq': 'https://drive.google.com/uc?export=download&id=1lGcRwNoXy_uwXkD6sy43aAa-rMHRR7Ad', 'bedrooms': 'https://drive.google.com/uc?export=download&id=1r0_s83-XK2dKlyY3WjNYsfZ5-fnH8QgI', 'ffhq': 'https://drive.google.com/uc?export=download&id=1GcxTcLDPYxQqcQjeHpLUutGzwOlXXcks', 'cars': 'https://drive.google.com/uc?export=download&id=1aaUXHRHjQ9ww91x4mtPZD0w50fsIkXWt', 'cats': 'https://drive.google.com/uc?export=download&id=1JzA5iiS3qPrztVofQAjbb0N4xKdjOOyV', 'wikiart': 'https://drive.google.com/uc?export=download&id=1fN3noa7Rsl9slrDXsgZVDsYFxV0O08Vx'}
        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            if self.outclass in urls_torch:
                download_ckpt(urls_torch[self.outclass], checkpoint)
            else:
                checkpoint_tf = checkpoint.with_suffix('.pkl')
                if not checkpoint_tf.is_file():
                    download_ckpt(urls_tf[self.outclass], checkpoint_tf)
                None
                self.model.export_from_tf(checkpoint_tf)
        self.model.load_weights(checkpoint)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        rng = np.random.RandomState(seed)
        noise = torch.from_numpy(rng.standard_normal(512 * n_samples).reshape(n_samples, 512)).float()
        if self.w_primary:
            noise = self.model._modules['g_mapping'].forward(noise)
        return noise

    def get_max_latents(self):
        return 18

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN: cannot change output class without reloading')

    def forward(self, x):
        out = self.model.forward(x, latent_is_w=self.w_primary)
        return 0.5 * (out + 1)

    def partial_forward(self, x, layer_name):
        mapping = self.model._modules['g_mapping']
        G = self.model._modules['g_synthesis']
        trunc = self.model._modules.get('truncation', lambda x: x)
        if not self.w_primary:
            x = mapping.forward(x)
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        else:
            x = x.unsqueeze(1).expand(-1, 18, -1)
        if 'g_mapping' in layer_name:
            return
        x = trunc(x)
        if layer_name == 'truncation':
            return

        def iterate(m, name, seen):
            children = getattr(m, '_modules', [])
            if len(children) > 0:
                for child_name, module in children.items():
                    seen += iterate(module, f'{name}.{child_name}', seen)
                return seen
            else:
                return [name]
        batch_size = x.size(0)
        for i, (n, m) in enumerate(G.blocks.items()):
            if i == 0:
                r = m(x[:, 2 * i:2 * i + 2])
            else:
                r = m(r, x[:, 2 * i:2 * i + 2])
            children = iterate(m, f'g_synthesis.blocks.{n}', [])
            for c in children:
                if layer_name in c:
                    return
        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')

    def set_noise_seed(self, seed):
        G = self.model._modules['g_synthesis']

        def for_each_child(this, name, func):
            children = getattr(this, '_modules', [])
            for child_name, module in children.items():
                for_each_child(module, f'{name}.{child_name}', func)
            func(this, name)

        def modify(m, name):
            if isinstance(m, stylegan.NoiseLayer):
                H, W = [int(s) for s in name.split('.')[2].split('x')]
                torch.random.manual_seed(seed)
                m.noise = torch.randn(1, 1, H, W, device=self.device, dtype=torch.float32)
        for_each_child(G, 'g_synthesis', modify)


class GANZooModel(BaseModel):

    def __init__(self, device, model_name):
        super(GANZooModel, self).__init__(model_name, 'default')
        self.device = device
        self.base_model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', model_name, pretrained=True, useGPU=device.type == 'cuda')
        self.model = self.base_model.netG
        self.name = model_name
        self.has_latent_residual = False

    def sample_latent(self, n_samples=1, seed=0, truncation=None):
        noise, _ = self.base_model.buildNoiseData(n_samples)
        return noise

    def partial_forward(self, x, layer_name):
        return self.forward(x)

    def get_conditional_state(self, z):
        return z[:, -20:]

    def set_conditional_state(self, z, c):
        z[:, -20:] = c
        return z

    def forward(self, x):
        out = self.base_model.test(x)
        return 0.5 * (out + 1)


class ProGAN(BaseModel):

    def __init__(self, device, lsun_class=None):
        super(ProGAN, self).__init__('ProGAN', lsun_class)
        self.device = device
        valid_classes = ['bedroom', 'churchoutdoor', 'conferenceroom', 'diningroom', 'kitchen', 'livingroom', 'restaurant']
        assert self.outclass in valid_classes, f'Invalid LSUN class {self.outclass}, should be one of {valid_classes}'
        self.load_model()
        self.name = f'ProGAN-{self.outclass}'
        self.has_latent_residual = False

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'progan/{self.outclass}_lsun.pth'
        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            url = f'http://netdissect.csail.mit.edu/data/ganmodel/karras/{self.outclass}_lsun.pth'
            download_ckpt(url, checkpoint)
        self.model = proggan.from_pth_file(str(checkpoint.resolve()))

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        noise = zdataset.z_sample_for_model(self.model, n_samples, seed=seed)[...]
        return noise

    def partial_forward(self, x, layer_name):
        assert isinstance(self.model, torch.nn.Sequential), 'Expected sequential model'
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for name, module in self.model._modules.items():
            x = module(x)
            if name == layer_name:
                return
        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')


def make_matching_tensor(valuedict, name, data):
    """
    Converts `valuedict[name]` to be a tensor with the same dtype, device,
    and dimension count as `data`, and caches the converted tensor.
    """
    v = valuedict.get(name, None)
    if v is None:
        return None
    if not isinstance(v, torch.Tensor):
        v = torch.from_numpy(numpy.array(v))
        valuedict[name] = v
    if not v.device == data.device or not v.dtype == data.dtype:
        assert not v.requires_grad, '%s wrong device or type' % name
        v = v
        valuedict[name] = v
    if len(v.shape) < len(data.shape):
        assert not v.requires_grad, '%s wrong dimensions' % name
        v = v.view((1,) + tuple(v.shape) + (1,) * (len(data.shape) - len(v.shape) - 1))
        valuedict[name] = v
    return v


class InstrumentedModel(torch.nn.Module):
    """
    A wrapper for hooking, probing and intervening in pytorch Modules.
    Example usage:

    ```
    model = load_my_model()
    with inst as InstrumentedModel(model):
        inst.retain_layer(layername)
        inst.edit_layer(layername, 0.5, target_features)
        inst.edit_layer(layername, offset=offset_tensor)
        inst(inputs)
        original_features = inst.retained_layer(layername)
    ```
    """

    def __init__(self, model):
        super(InstrumentedModel, self).__init__()
        self.model = model
        self._retained = OrderedDict()
        self._ablation = {}
        self._replacement = {}
        self._offset = {}
        self._hooked_layer = {}
        self._old_forward = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retain_layer(self, layername):
        """
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and retain its output each time the model is run.
        A pair (layername, aka) can be provided, and the aka will be used
        as the key for the retained value instead of the layername.
        """
        self.retain_layers([layername])

    def retain_layers(self, layernames):
        """
        Retains a list of a layers at once.
        """
        self.add_hooks(layernames)
        for layername in layernames:
            aka = layername
            if not isinstance(aka, str):
                layername, aka = layername
            if aka not in self._retained:
                self._retained[aka] = None

    def retained_features(self):
        """
        Returns a dict of all currently retained features.
        """
        return OrderedDict(self._retained)

    def retained_layer(self, aka=None, clear=False):
        """
        Retrieve retained data that was previously hooked by retain_layer.
        Call this after the model is run.  If clear is set, then the
        retained value will return and also cleared.
        """
        if aka is None:
            aka = next(self._retained.keys().__iter__())
        result = self._retained[aka]
        if clear:
            self._retained[aka] = None
        return result

    def edit_layer(self, layername, ablation=None, replacement=None, offset=None):
        """
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and modify its output each time the model is run.
        The output of the layer will be modified to be a convex combination
        of the replacement and x interpolated according to the ablation, i.e.:
        `output = x * (1 - a) + (r * a)`.
        Additionally or independently, an offset can be added to the output.
        """
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if ablation is None and replacement is not None:
            ablation = 1.0
        self.add_hooks([(layername, aka)])
        if ablation is not None:
            self._ablation[aka] = ablation
        if replacement is not None:
            self._replacement[aka] = replacement
        if offset is not None:
            self._offset[aka] = offset

    def remove_edits(self, layername=None, remove_offset=True, remove_replacement=True):
        """
        Removes edits at the specified layer, or removes edits at all layers
        if no layer name is specified.
        """
        if layername is None:
            if remove_replacement:
                self._ablation.clear()
                self._replacement.clear()
            if remove_offset:
                self._offset.clear()
            return
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if remove_replacement and aka in self._ablation:
            del self._ablation[aka]
        if remove_replacement and aka in self._replacement:
            del self._replacement[aka]
        if remove_offset and aka in self._offset:
            del self._offset[aka]

    def add_hooks(self, layernames):
        """
        Sets up a set of layers to be hooked.

        Usually not called directly: use edit_layer or retain_layer instead.
        """
        needed = set()
        aka_map = {}
        for name in layernames:
            aka = name
            if not isinstance(aka, str):
                name, aka = name
            if self._hooked_layer.get(aka, None) != name:
                aka_map[name] = aka
                needed.add(name)
        if not needed:
            return
        for name, layer in self.model.named_modules():
            if name in aka_map:
                needed.remove(name)
                aka = aka_map[name]
                self._hook_layer(layer, name, aka)
        for name in needed:
            raise ValueError('Layer %s not found in model' % name)

    def _hook_layer(self, layer, layername, aka):
        """
        Internal method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        if aka in self._hooked_layer:
            raise ValueError('Layer %s already hooked' % aka)
        if layername in self._old_forward:
            raise ValueError('Layer %s already hooked' % layername)
        self._hooked_layer[aka] = layername
        self._old_forward[layername] = layer, aka, layer.__dict__.get('forward', None)
        editor = self
        original_forward = layer.forward

        def new_forward(self, *inputs, **kwargs):
            original_x = original_forward(*inputs, **kwargs)
            x = editor._postprocess_forward(original_x, aka)
            return x
        layer.forward = types.MethodType(new_forward, layer)

    def _unhook_layer(self, aka):
        """
        Internal method to remove a hook, restoring the original forward method.
        """
        if aka not in self._hooked_layer:
            return
        layername = self._hooked_layer[aka]
        layer, check, old_forward = self._old_forward[layername]
        assert check == aka
        if old_forward is None:
            if 'forward' in layer.__dict__:
                del layer.__dict__['forward']
        else:
            layer.forward = old_forward
        del self._old_forward[layername]
        del self._hooked_layer[aka]
        if aka in self._ablation:
            del self._ablation[aka]
        if aka in self._replacement:
            del self._replacement[aka]
        if aka in self._offset:
            del self._offset[aka]
        if aka in self._retained:
            del self._retained[aka]

    def _postprocess_forward(self, x, aka):
        """
        The internal method called by the hooked layers after they are run.
        """
        if aka in self._retained:
            self._retained[aka] = x.detach()
        a = make_matching_tensor(self._ablation, aka, x)
        if a is not None:
            x = x * (1 - a)
            v = make_matching_tensor(self._replacement, aka, x)
            if v is not None:
                x += v * a
        b = make_matching_tensor(self._offset, aka, x)
        if b is not None:
            x = x + b
        return x

    def close(self):
        """
        Unhooks all hooked layers in the model.
        """
        for aka in list(self._old_forward.keys()):
            self._unhook_layer(aka)
        assert len(self._old_forward) == 0


class WScaleLayer(nn.Module):

    def __init__(self, size, fan_in, gain=numpy.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / numpy.sqrt(fan_in)
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(x_size[0], self.size, x_size[2], x_size[3])
        return x


class NormConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class DoubleResolutionLayer(nn.Module):

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x


class NormUpscaleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = DoubleResolutionLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class OutputConvBlock(nn.Module):

    def __init__(self, in_channels, tanh=False):
        super().__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0, bias=False)
        self.wscale = WScaleLayer(3, in_channels, gain=1)
        self.clamp = nn.Hardtanh() if tanh else lambda x: x

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.clamp(x)
        return x


class ProgressiveGenerator(nn.Sequential):

    def __init__(self, resolution=None, sizes=None, modify_sequence=None, output_tanh=False):
        """
        A pytorch progessive GAN generator that can be converted directly
        from either a tensorflow model or a theano model.  It consists of
        a sequence of convolutional layers, organized in pairs, with an
        upsampling and reduction of channels at every other layer; and
        then finally followed by an output layer that reduces it to an
        RGB [-1..1] image.

        The network can be given more layers to increase the output
        resolution.  The sizes argument indicates the fieature depth at
        each upsampling, starting with the input z: [input-dim, 4x4-depth,
        8x8-depth, 16x16-depth...].  The output dimension is 2 * 2**len(sizes)

        Some default architectures can be selected by supplying the
        resolution argument instead.

        The optional modify_sequence function can be used to transform the
        sequence of layers before the network is constructed.

        If output_tanh is set to True, the network applies a tanh to clamp
        the output to [-1,1] before output; otherwise the output is unclamped.
        """
        assert (resolution is None) != (sizes is None)
        if sizes is None:
            sizes = {(8): [512, 512, 512], (16): [512, 512, 512, 512], (32): [512, 512, 512, 512, 256], (64): [512, 512, 512, 512, 256, 128], (128): [512, 512, 512, 512, 256, 128, 64], (256): [512, 512, 512, 512, 256, 128, 64, 32], (1024): [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]}[resolution]
        sequence = []

        def add_d(layer, name=None):
            if name is None:
                name = 'layer%d' % (len(sequence) + 1)
            sequence.append((name, layer))
        add_d(NormConvBlock(sizes[0], sizes[1], kernel_size=4, padding=3))
        add_d(NormConvBlock(sizes[1], sizes[1], kernel_size=3, padding=1))
        for i, (si, so) in enumerate(zip(sizes[1:-1], sizes[2:])):
            add_d(NormUpscaleConvBlock(si, so, kernel_size=3, padding=1))
            add_d(NormConvBlock(so, so, kernel_size=3, padding=1))
        dim = 4 * 2 ** (len(sequence) // 2 - 1)
        add_d(OutputConvBlock(sizes[-1], tanh=output_tanh), name='output_%dx%d' % (dim, dim))
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class SegmentationModuleBase(nn.Module):

    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


class SegmentationModule(SegmentationModuleBase):

    def __init__(self, net_enc, net_dec, labeldata, loss_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_dict = nn.ModuleDict()
        if loss_scale is None:
            self.loss_scale = {'object': 1, 'part': 0.5, 'scene': 0.25, 'material': 1}
        else:
            self.loss_scale = loss_scale
        self.crit_dict['object'] = nn.NLLLoss(ignore_index=0)
        self.crit_dict['material'] = nn.NLLLoss(ignore_index=0)
        self.crit_dict['scene'] = nn.NLLLoss(ignore_index=-1)
        self.labeldata = labeldata
        object_to_num = {k: v for v, k in enumerate(labeldata['object'])}
        part_to_num = {k: v for v, k in enumerate(labeldata['part'])}
        self.object_part = {object_to_num[k]: [part_to_num[p] for p in v] for k, v in labeldata['object_part'].items()}
        self.object_with_part = sorted(self.object_part.keys())
        self.decoder.object_part = self.object_part
        self.decoder.object_with_part = self.object_with_part

    def forward(self, feed_dict, *, seg_size=None):
        if seg_size is None:
            if feed_dict['source_idx'] == 0:
                output_switch = {'object': True, 'part': True, 'scene': True, 'material': False}
            elif feed_dict['source_idx'] == 1:
                output_switch = {'object': False, 'part': False, 'scene': False, 'material': True}
            else:
                raise ValueError
            pred = self.decoder(self.encoder(feed_dict['img'], return_feature_maps=True), output_switch=output_switch)
            loss_dict = {}
            if pred['object'] is not None:
                loss_dict['object'] = self.crit_dict['object'](pred['object'], feed_dict['seg_object'])
            if pred['part'] is not None:
                part_loss = 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    part_loss += self.part_loss(pred['part'][idx_part], feed_dict['seg_part'], feed_dict['seg_object'], object_label, feed_dict['valid_part'][:, (idx_part)])
                loss_dict['part'] = part_loss
            if pred['scene'] is not None:
                loss_dict['scene'] = self.crit_dict['scene'](pred['scene'], feed_dict['scene_label'])
            if pred['material'] is not None:
                loss_dict['material'] = self.crit_dict['material'](pred['material'], feed_dict['seg_material'])
            loss_dict['total'] = sum([(loss_dict[k] * self.loss_scale[k]) for k in loss_dict.keys()])
            metric_dict = {}
            if pred['object'] is not None:
                metric_dict['object'] = self.pixel_acc(pred['object'], feed_dict['seg_object'], ignore_index=0)
            if pred['material'] is not None:
                metric_dict['material'] = self.pixel_acc(pred['material'], feed_dict['seg_material'], ignore_index=0)
            if pred['part'] is not None:
                acc_sum, pixel_sum = 0, 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    acc, pixel = self.part_pixel_acc(pred['part'][idx_part], feed_dict['seg_part'], feed_dict['seg_object'], object_label, feed_dict['valid_part'][:, (idx_part)])
                    acc_sum += acc
                    pixel_sum += pixel
                metric_dict['part'] = acc_sum.float() / (pixel_sum.float() + 1e-10)
            if pred['scene'] is not None:
                metric_dict['scene'] = self.pixel_acc(pred['scene'], feed_dict['scene_label'], ignore_index=-1)
            return {'metric': metric_dict, 'loss': loss_dict}
        else:
            output_switch = {'object': True, 'part': True, 'scene': True, 'material': True}
            pred = self.decoder(self.encoder(feed_dict['img'], return_feature_maps=True), output_switch=output_switch, seg_size=seg_size)
            return pred


class Resnet(nn.Module):

    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), SynchronizedBatchNorm2d(out_planes), nn.ReLU(inplace=True))


class C1BilinearDeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, inference=False, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class C1Bilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, inference=False, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinearDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class UPerNet(nn.Module):

    def __init__(self, nr_classes, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.0))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False), SynchronizedBatchNorm2d(fpn_dim), nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)
        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = nr_classes['scene'], nr_classes['object'], nr_classes['part'], nr_classes['material']
        self.scene_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.AdaptiveAvgPool2d(1), nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True))
        self.object_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True))
        self.part_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True))
        self.material_head = nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True))

    def forward(self, conv_out, output_switch=None, seg_size=None):
        output_dict = {k: None for k in output_switch.keys()}
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = []
        for i in range(input_size[0]):
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(pool_scale(conv5, roi.detach()), (input_size[2], input_size[3]), mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        if output_switch['scene']:
            output_dict['scene'] = self.scene_head(f)
        if output_switch['object'] or output_switch['part'] or output_switch['material']:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)
                f = F.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
                f = conv_x + f
                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse()
            if output_switch['material']:
                output_dict['material'] = self.material_head(fpn_feature_list[0])
            if output_switch['object'] or output_switch['part']:
                output_size = fpn_feature_list[0].size()[2:]
                fusion_list = [fpn_feature_list[0]]
                for i in range(1, len(fpn_feature_list)):
                    fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode='bilinear', align_corners=False))
                fusion_out = torch.cat(fusion_list, 1)
                x = self.conv_fusion(fusion_out)
                if output_switch['object']:
                    output_dict['object'] = self.object_head(x)
                if output_switch['part']:
                    output_dict['part'] = self.part_head(x)
        if self.use_softmax:
            x = output_dict['scene']
            x = x.squeeze(3).squeeze(2)
            x = F.softmax(x, dim=1)
            output_dict['scene'] = x
            for k in ['object', 'material']:
                x = output_dict[k]
                x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
                x = F.softmax(x, dim=1)
                output_dict[k] = x
            x = output_dict['part']
            x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            part_pred_list, head = [], 0
            for idx_part, object_label in enumerate(self.object_with_part):
                n_part = len(self.object_part[object_label])
                _x = F.interpolate(x[:, head:head + n_part], size=seg_size, mode='bilinear', align_corners=False)
                _x = F.softmax(_x, dim=1)
                part_pred_list.append(_x)
                head += n_part
            output_dict['part'] = part_pred_list
        else:
            for k in ['object', 'scene', 'material']:
                if output_dict[k] is None:
                    continue
                x = output_dict[k]
                x = F.log_softmax(x, dim=1)
                if k == 'scene':
                    x = x.squeeze(3).squeeze(2)
                output_dict[k] = x
            if output_dict['part'] is not None:
                part_pred_list, head = [], 0
                for idx_part, object_label in enumerate(self.object_with_part):
                    n_part = len(self.object_part[object_label])
                    x = output_dict['part'][:, head:head + n_part]
                    x = F.log_softmax(x, dim=1)
                    part_pred_list.append(x)
                    head += n_part
                output_dict['part'] = part_pred_list
        return output_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
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
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
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
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), SynchronizedBatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(GroupBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 2)
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


class ResNeXt(nn.Module):

    def __init__(self, block, layers, groups=32, num_classes=1000):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), SynchronizedBatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PrRoIPool2DFunction(ag.Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), 'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())
        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)
        features = features.contiguous()
        rois = rois.contiguous()
        params = pooled_height, pooled_width, spatial_scale
        if features.is_cuda:
            output = _prroi_pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None
        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = _prroi_pooling.prroi_pooling_backward_cuda(features, rois, output, grad_output, *ctx.params)
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = _prroi_pooling.prroi_pooling_coor_backward_cuda(features, rois, output, grad_output, *ctx.params)
        return grad_input, grad_coor, None, None, None


prroi_pool2d = PrRoIPool2DFunction.apply


class PrRoIPool2D(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BlurLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DoubleResolutionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (G_mapping,
     lambda: ([], {}),
     lambda: ([torch.rand([512, 512])], {}),
     False),
    (InstrumentedModel,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MyConv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MyLinear,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoiseLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormUpscaleConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OutputConvBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttn,
     lambda: ([], {'in_channels': 64}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (StyleGAN_G,
     lambda: ([], {'resolution': 4}),
     lambda: ([torch.rand([512, 512])], {}),
     False),
    (StyleMod,
     lambda: ([], {'latent_size': 4, 'channels': 4, 'use_wscale': 1.0}),
     lambda: ([torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upscale2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WScaleLayer,
     lambda: ([], {'size': 4, 'fan_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_harskish_ganspace(_paritybench_base):
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

