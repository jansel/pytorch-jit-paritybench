import sys
_module = sys.modules[__name__]
del sys
dalle = _module
models = _module
layers = _module
vqgan = _module
layers = _module
transformer = _module
tokenizer = _module
utils = _module
config = _module
sampling = _module
utils = _module
sampling_ex = _module
transfer_learning_ex = _module

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


import torch.nn as nn


from typing import Optional


from typing import Tuple


from torch.cuda.amp import autocast


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.nn import functional as F


from typing import List


import math


import random


import numpy as np


from torch.utils.data import DataLoader


import torchvision


import torchvision.transforms as transforms


class GELU(nn.Module):

    def __init__(self, use_approx=False):
        super().__init__()
        self.use_approx = use_approx

    def forward(self, x):
        if self.use_approx:
            return x * torch.sigmoid(1.702 * x)
        else:
            return F.gelu(x)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, ctx_len: int, embed_dim: int, n_heads: int, resid_pdrop: float, attn_pdrop: float, attn_bias: bool, use_mask: bool=True):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)
        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer('mask', torch.ones(ctx_len, ctx_len), persistent=False)
            self.mask = torch.tril(self.mask).view(1, ctx_len, ctx_len)

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape
        x = x.transpose(0, 1).contiguous()
        k = self.key(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        q = self.query(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        v = self.value(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        if use_cache:
            present = torch.stack([k, v])
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        if use_cache and layer_past is not None:
            att = torch.bmm(q, k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = torch.bmm(att, v)
        else:
            att = torch.bmm(q, k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            if self.use_mask:
                mask = self.mask if T == self.ctx_len else self.mask[:, :T, :T]
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = torch.bmm(att, v)
        y = y.transpose(0, 1).contiguous().view(T, B, C)
        y = self.resid_drop(self.proj(y))
        if use_cache:
            return y.transpose(0, 1).contiguous(), present
        else:
            return y.transpose(0, 1).contiguous()


class Block(nn.Module):

    def __init__(self, ctx_len: int, embed_dim: int, n_heads: int, mlp_bias: bool, attn_bias: bool, resid_pdrop: bool, attn_pdrop: bool, gelu_use_approx: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len, embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, attn_bias=attn_bias, use_mask=True)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias), GELU(gelu_use_approx), nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias), nn.Dropout(resid_pdrop))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def sample(self, x, layer_past=None):
        attn, present = self.attn(self.ln1(x), use_cache=True, layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        assert temb_channels == 0
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        assert temb is None
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, *, ch: int, out_ch: int, ch_mult: Tuple[int]=(1, 2, 4, 8), num_res_blocks: int, attn_resolutions: Tuple[int], pdrop: float=0.0, resamp_with_conv: bool=True, in_channels: int, resolution: int, z_channels: int, double_z: bool) ->None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=pdrop)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=pdrop)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=pdrop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Encoder(nn.Module):

    def __init__(self, *, ch: int, out_ch: int, ch_mult: Tuple[int]=(1, 2, 4, 8), num_res_blocks: int, attn_resolutions: Tuple[int], pdrop: float=0.0, resamp_with_conv: bool=True, in_channels: int, resolution: int, z_channels: int, double_z: Optional[bool]=None) ->None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=pdrop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=pdrop)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=pdrop)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.resolution, '{}, {}'.format(x.shape, self.resolution)
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    """
    Simplified VectorQuantizer in the original VQGAN repository
    by removing unncessary modules for sampling
    """

    def __init__(self, dim: int, n_embed: int, beta: float) ->None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_embed, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z: torch.FloatTensor) ->Tuple[torch.FloatTensor, torch.LongTensor]:
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Optional[List[int]]=None) ->torch.FloatTensor:
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


_MODELS = {'minDALL-E/1.3B': 'https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz'}


def build_tokenizer(path: str, context_length: int=64, *args, **kwargs):
    from_file = partial(CharBPETokenizer.from_file, vocab_filename=os.path.join(path, 'bpe-16k-vocab.json'), merges_filename=os.path.join(path, 'bpe-16k-merges.txt'), unk_token='[UNK]')
    tokenizer = from_file(*args, **kwargs)
    tokenizer.add_special_tokens(['[PAD]'])
    tokenizer.enable_padding(length=context_length, pad_id=tokenizer.token_to_id('[PAD]'))
    tokenizer.enable_truncation(max_length=context_length)
    None
    return tokenizer


def get_base_config(use_default=True):
    return OmegaConf.structured(DefaultConfig if use_default else FineTuningConfig)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'ctx_len': 4, 'embed_dim': 4, 'n_heads': 4, 'mlp_bias': 4, 'attn_bias': 4, 'resid_pdrop': 0.5, 'attn_pdrop': 0.5, 'gelu_use_approx': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadSelfAttention,
     lambda: ([], {'ctx_len': 4, 'embed_dim': 4, 'n_heads': 4, 'resid_pdrop': 0.5, 'attn_pdrop': 0.5, 'attn_bias': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kakaobrain_minDALL_E(_paritybench_base):
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

