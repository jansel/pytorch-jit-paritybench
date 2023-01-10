import sys
_module = sys.modules[__name__]
del sys
main = _module
convert = _module
pytorch_pretrained_vit = _module
configs = _module
model = _module
transformer = _module
utils = _module
setup = _module

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


import random


import time


import warnings


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import numpy as np


from torchvision import transforms


from typing import Optional


from torch import nn


from torch.nn import functional as F


from torch import Tensor


from torch.utils import model_zoo


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config


def get_base_config():
    """Base ViT config ViT"""
    return dict(dim=768, ff_dim=3072, num_heads=12, num_layers=12, attention_dropout_rate=0.0, dropout_rate=0.1, representation_size=768, classifier='token')


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16), dim=1024, ff_dim=4096, num_heads=16, num_layers=24, attention_dropout_rate=0.0, dropout_rate=0.1, representation_size=1024))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config


PRETRAINED_MODELS = {'B_16': {'config': get_b16_config(), 'num_classes': 21843, 'image_size': (224, 224), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth'}, 'B_32': {'config': get_b32_config(), 'num_classes': 21843, 'image_size': (224, 224), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth'}, 'L_16': {'config': get_l16_config(), 'num_classes': 21843, 'image_size': (224, 224), 'url': None}, 'L_32': {'config': get_l32_config(), 'num_classes': 21843, 'image_size': (224, 224), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth'}, 'B_16_imagenet1k': {'config': drop_head_variant(get_b16_config()), 'num_classes': 1000, 'image_size': (384, 384), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'}, 'B_32_imagenet1k': {'config': drop_head_variant(get_b32_config()), 'num_classes': 1000, 'image_size': (384, 384), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth'}, 'L_16_imagenet1k': {'config': drop_head_variant(get_l16_config()), 'num_classes': 1000, 'image_size': (384, 384), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth'}, 'L_32_imagenet1k': {'config': drop_head_variant(get_l32_config()), 'num_classes': 1000, 'image_size': (384, 384), 'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth'}}


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
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def maybe_print(s: str, flag: bool):
    if flag:
        None


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom
    ntok_new = posemb_new.shape[1]
    if has_class_token:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom_factor = gs_new / gs_old, gs_new / gs_old, 1
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def load_pretrained_weights(model, model_name=None, weights_path=None, load_first_conv=True, load_fc=True, load_repr_layer=False, resize_positional_embedding=False, verbose=True, strict=True):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)
    if resize_positional_embedding:
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(posemb.shape, posemb_new.shape), verbose)
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(self, name: Optional[str]=None, pretrained: bool=False, patches: int=16, dim: int=768, ff_dim: int=3072, num_heads: int=12, num_layers: int=12, attention_dropout_rate: float=0.0, dropout_rate: float=0.1, representation_size: Optional[int]=None, load_repr_layer: bool=False, classifier: str='token', positional_embedding: str='1d', in_channels: int=3, image_size: Optional[int]=None, num_classes: Optional[int]=None):
        super().__init__()
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:
            assert name in PRETRAINED_MODELS.keys(), 'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size
        h, w = as_tuple(image_size)
        fh, fw = as_tuple(patches)
        gh, gw = h // fh, w // fw
        seq_len = gh * gw
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-06)
        self.fc = nn.Linear(pre_logits_size, num_classes)
        self.init_weights()
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(self, name, load_first_conv=in_channels == pretrained_num_channels, load_fc=num_classes == pretrained_num_classes, load_repr_layer=load_repr_layer, resize_positional_embedding=image_size != pretrained_image_size)

    @torch.no_grad()
    def init_weights(self):

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-06)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)
        x = self.transformer(x)
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]
            x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PositionWiseFeedForward,
     lambda: ([], {'dim': 4, 'ff_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEmbedding1D,
     lambda: ([], {'seq_len': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'num_layers': 1, 'dim': 4, 'num_heads': 4, 'ff_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lukemelas_PyTorch_Pretrained_ViT(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

