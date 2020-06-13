import sys
_module = sys.modules[__name__]
del sys
pytorch_pretrained_biggan = _module
config = _module
convert_tf_to_pytorch = _module
file_utils = _module
model = _module
utils = _module
setup = _module

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


from itertools import chain


import logging


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.functional import normalize


import math


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
        z = self.gen_z(cond_vector)
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)
        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, (...)]
        z = self.tanh(z)
        return z


WEIGHTS_NAME = 'pytorch_model.bin'


PRETRAINED_MODEL_ARCHIVE_MAP = {'biggan-deep-128':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin'
    , 'biggan-deep-256':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin'
    , 'biggan-deep-512':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin'
    }


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


logger = logging.getLogger(__name__)


PRETRAINED_CONFIG_ARCHIVE_MAP = {'biggan-deep-128':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json'
    , 'biggan-deep-256':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json'
    , 'biggan-deep-512':
    'https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json'
    }


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


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    return filename


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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_huggingface_pytorch_pretrained_BigGAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(SelfAttn(*[], **{'in_channels': 64}), [torch.rand([4, 64, 64, 64])], {})

