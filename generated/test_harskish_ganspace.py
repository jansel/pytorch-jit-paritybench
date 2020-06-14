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
__main__ = _module
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

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


from types import SimpleNamespace


from torch.nn.functional import interpolate


from functools import partial


from itertools import chain


import logging


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.functional import normalize


import math


from collections import OrderedDict


import re


import random


from abc import abstractmethod


from abc import ABC as AbstractBaseClass


import numpy


from torch.utils.data import TensorDataset


from scipy.ndimage.morphology import binary_dilation


import types


from collections import defaultdict


import itertools


from torch.utils.data import DataLoader


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels,
            out_channels=in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels
            =in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=
            in_channels // 2, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels // 2,
            out_channels=in_channels, kernel_size=1, bias=False, eps=eps)
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

    def __init__(self, num_features, condition_vector_dim=None, n_stats=51,
        eps=0.0001, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional
        self.register_buffer('running_means', torch.zeros(n_stats,
            num_features))
        self.register_buffer('running_vars', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)
        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim,
                out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim,
                out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:
            running_mean = self.running_means[start_idx
                ] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx
                ] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]
        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1
                )
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(
                -1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)
            out = (x - running_mean) / torch.sqrt(running_var + self.eps
                ) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight,
                self.bias, training=False, momentum=0.0, eps=self.eps)
        return out


class GenBlock(nn.Module):

    def __init__(self, in_size, out_size, condition_vector_dim,
        reduction_factor=4, up_sample=False, n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        middle_size = in_size // reduction_factor
        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=
            n_stats, eps=eps, conditional=True)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=
            middle_size, kernel_size=1, eps=eps)
        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=
            middle_size, kernel_size=3, padding=1, eps=eps)
        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=
            middle_size, kernel_size=3, padding=1, eps=eps)
        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=
            out_size, kernel_size=1, eps=eps)
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
        self.gen_z = snlinear(in_features=condition_vector_dim,
            out_features=4 * 4 * 16 * ch, eps=config.eps)
        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch * layer[1], eps=config.eps))
            layers.append(GenBlock(ch * layer[1], ch * layer[2],
                condition_vector_dim, up_sample=layer[0], n_stats=config.
                n_stats, eps=config.eps))
        self.layers = nn.ModuleList(layers)
        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.
            eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch,
            kernel_size=3, padding=1, eps=config.eps)
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


class BigGANConfig(object):
    """ Configuration class to store the configuration of a `BigGAN`. 
        Defaults are for the 128x128 model.
        layers tuple are (up-sample in the layer ?, input channels, output channels)
    """

    def __init__(self, output_dim=128, z_dim=128, class_embed_dim=128,
        channel_width=128, num_classes=1000, layers=[(False, 16, 16), (True,
        16, 16), (False, 16, 16), (True, 16, 8), (False, 8, 8), (True, 8, 4
        ), (False, 4, 4), (True, 4, 2), (False, 2, 2), (True, 2, 1)],
        attention_layer_position=8, eps=0.0001, n_stats=51):
        """Constructs BigGANConfig. """
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BigGANConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


CONFIG_NAME = 'config.json'


PRETRAINED_CONFIG_ARCHIVE_MAP = {'biggan-deep-128':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json'
    , 'biggan-deep-256':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json'
    , 'biggan-deep-512':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json'
    }


PRETRAINED_MODEL_ARCHIVE_MAP = {'biggan-deep-128':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin'
    , 'biggan-deep-256':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin'
    , 'biggan-deep-512':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin'
    }


WEIGHTS_NAME = 'pytorch_model.bin'


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


logger = logging.getLogger(__name__)


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise EnvironmentError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=
        False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** -0.5
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size,
            input_size) * init_std)
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


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, gain=2 **
        0.5, use_wscale=False, lrmul=1, bias=True, intermediate=None,
        upscale=False):
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
        self.weight = torch.nn.Parameter(torch.randn(output_channels,
            input_channels, kernel_size, kernel_size) * init_std)
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
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:,
                :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) //
                2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)
        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self
                .kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.
                kernel_size // 2)
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
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=
                x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):

    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size, channels * 2, gain=1.0, use_wscale
            =use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)
        x = x * (style[:, (0)] + 1.0) + style[:, (1)]
        return x


class PixelNormLayer(nn.Module):

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) +
            self.epsilon)


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
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.
            kernel.size(2) - 1) / 2), groups=x.size(1))
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1,
            -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], 
            factor * shape[3])
    return x


class Upscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):

    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)), 'lrelu': (nn.
            LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [('pixel_norm', PixelNormLayer()), ('dense0', MyLinear(512,
            512, gain=gain, lrmul=0.01, use_wscale=use_wscale)), (
            'dense0_act', act), ('dense1', MyLinear(512, 512, gain=gain,
            lrmul=0.01, use_wscale=use_wscale)), ('dense1_act', act), (
            'dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=
            use_wscale)), ('dense2_act', act), ('dense3', MyLinear(512, 512,
            gain=gain, lrmul=0.01, use_wscale=use_wscale)), ('dense3_act',
            act), ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01,
            use_wscale=use_wscale)), ('dense4_act', act), ('dense5',
            MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale
            )), ('dense5_act', act), ('dense6', MyLinear(512, 512, gain=
            gain, lrmul=0.01, use_wscale=use_wscale)), ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale
            =use_wscale)), ('dense7_act', act)]
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

    def __init__(self, channels, dlatent_size, use_wscale, use_noise,
        use_pixel_norm, use_instance_norm, use_styles, activation_layer):
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
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=
                use_wscale)
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

    def __init__(self, nf, dlatent_size, const_input_layer, gain,
        use_wscale, use_noise, use_pixel_norm, use_instance_norm,
        use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(dlatent_size, nf * 16, gain=gain / 4,
                use_wscale=use_wscale)
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
            use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
            use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, (0)]).view(batch_size, self
                .nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, (0)])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, (1)])
        return x


class GSynthesisBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size,
        gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
        use_styles, activation_layer):
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3,
            gain=gain, use_wscale=use_wscale, intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale,
            use_noise, use_pixel_norm, use_instance_norm, use_styles,
            activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3,
            gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale,
            use_noise, use_pixel_norm, use_instance_norm, use_styles,
            activation_layer)

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, (0)])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, (1)])
        return x


class G_synthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
        fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_styles=True,
        const_input_layer=True, use_noise=True, randomize_noise=True,
        nonlinearity='lrelu', use_wscale=True, use_pixel_norm=False,
        use_instance_norm=True, dtype=torch.float32, blur_filter=[1, 2, 1]):
        super().__init__()

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        act, gain = {'relu': (torch.relu, np.sqrt(2)), 'lrelu': (nn.
            LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            if res == 2:
                blocks.append((name, InputBlock(channels, dlatent_size,
                    const_input_layer, gain, use_wscale, use_noise,
                    use_pixel_norm, use_instance_norm, use_styles, act)))
            else:
                blocks.append((name, GSynthesisBlock(last_channels,
                    channels, blur_filter, dlatent_size, gain, use_wscale,
                    use_noise, use_pixel_norm, use_instance_norm,
                    use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale
            =use_wscale)
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
            z = [(torch.tensor(l).to(self.device) if not torch.is_tensor(l)
                 else l) for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z).to(self.device)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()

    def get_conditional_state(self, z):
        return None

    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)


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
        v = v.to(device=data.device, dtype=data.dtype)
        valuedict[name] = v
    if len(v.shape) < len(data.shape):
        assert not v.requires_grad, '%s wrong dimensions' % name
        v = v.view((1,) + tuple(v.shape) + (1,) * (len(data.shape) - len(v.
            shape) - 1))
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

    def edit_layer(self, layername, ablation=None, replacement=None, offset
        =None):
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

    def remove_edits(self, layername=None, remove_offset=True,
        remove_replacement=True):
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
        self._old_forward[layername] = layer, aka, layer.__dict__.get('forward'
            , None)
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


class ProgressiveGenerator(nn.Sequential):

    def __init__(self, resolution=None, sizes=None, modify_sequence=None,
        output_tanh=False):
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
            sizes = {(8): [512, 512, 512], (16): [512, 512, 512, 512], (32):
                [512, 512, 512, 512, 256], (64): [512, 512, 512, 512, 256, 
                128], (128): [512, 512, 512, 512, 256, 128, 64], (256): [
                512, 512, 512, 512, 256, 128, 64, 32], (1024): [512, 512, 
                512, 512, 512, 256, 128, 64, 32, 16]}[resolution]
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
        add_d(OutputConvBlock(sizes[-1], tanh=output_tanh), name=
            'output_%dx%d' % (dim, dim))
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)


class PixelNormLayer(nn.Module):

    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


class DoubleResolutionLayer(nn.Module):

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x


class WScaleLayer(nn.Module):

    def __init__(self, size, fan_in, gain=numpy.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / numpy.sqrt(fan_in)
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(x_size[0],
            self.size, x_size[2], x_size[3])
        return x


class NormConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1,
            padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.
            sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class NormUpscaleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = DoubleResolutionLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1,
            padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=numpy.
            sqrt(2) / kernel_size)
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
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0,
            bias=False)
        self.wscale = WScaleLayer(3, in_channels, gain=1)
        self.clamp = nn.Hardtanh() if tanh else lambda x: x

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.clamp(x)
        return x


class SegmentationModuleBase(nn.Module):

    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride),
        SynchronizedBatchNorm2d(out_planes), nn.ReLU(inplace=True))


class C1BilinearDeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, inference=False,
        use_softmax=False):
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
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
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

    def __init__(self, num_class=150, fc_dim=2048, inference=False,
        use_softmax=False):
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
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False,
        use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.
            Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear',
                align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinearDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, inference=False,
        use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.
            Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear',
                align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
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

    def __init__(self, num_class=150, fc_dim=4096, inference=False,
        use_softmax=False, pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512,
        1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        self.inference = inference
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512,
                kernel_size=1, bias=False), SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 
            512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim,
                kernel_size=1, bias=False), SynchronizedBatchNorm2d(fpn_dim
                ), nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim,
                fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) *
            fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interploate(pool_scale(
                conv5), (input_size[2], input_size[3]), mode='bilinear',
                align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode=
                'bilinear', align_corners=False)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i
                ], output_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.inference or self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear',
                align_corners=False)
            if self.use_softmax:
                x = nn.functional.softmax(x, dim=1)
            return x
        x = nn.functional.log_softmax(x, dim=1)
        return x


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion))
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
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
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
            groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
            groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
            groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1
                    ] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion))
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
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label,
        valid):
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
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(),
            reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion))
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
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
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
            groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
            groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
            groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1
                    ] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion))
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_harskish_ganspace(_paritybench_base):
    pass
    def test_000(self):
        self._check(BlurLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DoubleResolutionLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(G_mapping(*[], **{}), [torch.rand([512, 512])], {})

    def test_003(self):
        self._check(MyConv2d(*[], **{'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MyLinear(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(NoiseLayer(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(NormConvBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(NormUpscaleConvBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(OutputConvBlock(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(PixelNormLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(SelfAttn(*[], **{'in_channels': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_011(self):
        self._check(StyleMod(*[], **{'latent_size': 4, 'channels': 4, 'use_wscale': 1.0}), [torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(Upscale2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(WScaleLayer(*[], **{'size': 4, 'fan_in': 4}), [torch.rand([4, 4, 4, 4])], {})

