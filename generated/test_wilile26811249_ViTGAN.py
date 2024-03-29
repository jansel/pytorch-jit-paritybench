import sys
_module = sys.modules[__name__]
del sys
image_dataset = _module
main = _module
models = _module
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


import torch


import torchvision


import torchvision.transforms as T


from torch.utils.data import Dataset


from torch.utils.data import Sampler


import copy


import numpy as np


import torch.nn as nn


from torchvision.utils import make_grid


from torch.utils.data import DataLoader


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """

    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):

    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.0):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """

    def __init__(self, dim, num_heads=4, dim_head=None, discriminator=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias=False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias=True)
        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3
        if self.discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.num_heads))
        if self.discriminator:
            attn = torch.cdist(q, k, p=2)
        else:
            attn = torch.einsum('... i d, ... j d -> ... i j', q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim=-1)
        result = torch.einsum('... i j, ... j d -> ... i d', scale_attn_score, v)
        result = rearrange(result, 'b h t d -> b t (h d)')
        return self.w_out(result)


class DEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads=4, dim_head=None, dropout=0.0, mlp_ratio=4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class GEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads=4, dim_head=None, dropout=0.0, mlp_ratio=4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = SLN(dim)
        self.norm2 = SLN(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=dropout)

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attn(self.norm1(hl, x))) + hl
        hl_final = self.mlp(self.norm2(hl_temp, x)) + hl_temp
        return x, hl_final


class GTransformerEncoder(nn.Module):

    def __init__(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl, x):
        for block in self.blocks:
            x, hl = block(hl, x)
        return x, hl


class DTransformerEncoder(nn.Module):

    def __init__(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Generator(nn.Module):

    def __init__(self, initialize_size=8, dim=384, blocks=6, num_heads=6, dim_head=None, dropout=0, out_channels=3):
        super(Generator, self).__init__()
        self.initialize_size = initialize_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.out_channels = out_channels
        self.pos_emb1D = nn.Parameter(torch.randn(self.initialize_size * 8, dim))
        self.mlp = nn.Linear(1024, self.initialize_size * 8 * self.dim)
        self.Transformer_Encoder = GTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)
        self.w_out = nn.Sequential(SineLayer(dim, dim * 2, is_first=True, omega_0=30.0), SineLayer(dim * 2, self.initialize_size * 8 * self.out_channels, is_first=False, omega_0=30))
        self.sln_norm = SLN(self.dim)

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initialize_size * 8, self.dim)
        x, hl = self.Transformer_Encoder(self.pos_emb1D, x)
        x = self.sln_norm(hl, x)
        x = self.w_out(x)
        result = x.view(x.shape[0], 3, self.initialize_size * 8, self.initialize_size * 8)
        return result


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, patch_size=8, extend_size=2, dim=384, blocks=6, num_heads=6, dim_head=None, dropout=0):
        super(Discriminator, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        self.token_dim = in_channels * self.patch_size ** 2
        self.project_patches = nn.Linear(self.token_dim, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)

    def forward(self, img):
        stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        img_patches = img_patches.contiguous().view(img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5])
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim=1)
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)
        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'in_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SLN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SineLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wilile26811249_ViTGAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

