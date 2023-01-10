import sys
_module = sys.modules[__name__]
del sys
neox = _module
checkpoint = _module
data = _module
evaluation = _module
dummy = _module
half_precision = _module
pipeline_parallel = _module
model = _module
samples = _module
fine_tune_biases = _module
generating_gpu = _module
generating_pipe = _module
generating_small_gpu = _module
tokenizer = _module
utils = _module
cache = _module
training = _module

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


from typing import Union


from typing import Tuple


import torch


from torch import nn


from typing import Optional


from typing import List


import torch.utils.data


import torch.nn.functional as F


import copy


import math


from typing import Set


from torch.cuda.amp import autocast


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


import torch.optim


from torch.cuda import amp


class DummyModel(nn.Module):
    """
    ## Dummy model
    """

    def __init__(self, n_vocab: int):
        super().__init__()
        self.n_vocab = n_vocab

    def forward(self, x: torch.Tensor):
        return torch.randn(x.shape + (self.n_vocab,), dtype=torch.float, device=x.device)


class Embedding(nn.Module):
    """
    ## Embedding layer

    This is a standard embeddings layer with code to load the checkpoint.
    """

    def __init__(self, n_vocab: int=50432, n_hidden: int=6144):
        """
        :param n_vocab: is the size of the vocabulary
        :param n_hidden: is the size of the embeddings
        """
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the token ids of shape `[batch_size, seq_len]`
        """
        return self.emb(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load embedding layer'):
            checkpoint.merge_params_dim_0(self.emb.weight, 'word_embeddings.weight', p1, p2)


class RoPE(torch.nn.Module):
    """
    ## Rotary Positional Embeddings

    GPT-NeoX uses [rotary positional embeddings (RoPE)](https://papers.labml.ai/paper/2104.09864).

    WE have annotated implementation of RoPE [here](https://nn.labml.ai/transformers/rope/index.html)
    with more notes the theory.
    """

    def __init__(self, d_rope: int, base: float=10000.0):
        """
        :param d_rope: is the number of features for RoPE embeddings
        :param base: is the base for $	heta_i = 10000^{rac{2(i-1)}{d}}$, which defaults to $10000$
        """
        super().__init__()
        self.theta = None
        self.cos_cached = None
        self.sin_cached = None
        self.base = base
        self.d_rope = d_rope

    @staticmethod
    def rotate_half(x: torch.Tensor):
        """
        ### Rotate the features

        $[-x^{(rac{d}{2} + 1)}, -x^{(rac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., -x^{(rac{d}{2})}]$
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int=0):
        """
        :param x: has shape `[batch, seq, n_heads, d_k]`
        :param offset: is the starting position of `x`. This is $\\gt 0$ when we have
        cached the keys and queries of previous positions
        """
        seq_len = x.shape[1] + offset
        if self.theta is None:
            theta = 1.0 / self.base ** (torch.arange(0, self.d_rope, 2).float() / self.d_rope)
            self.theta = theta.to(x.device)
        if self.cos_cached is None or seq_len > self.cos_cached.shape[1] or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)
            idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
            idx_theta2 = torch.cat((idx_theta, idx_theta), dim=-1)
            with autocast(enabled=False):
                idx_theta2 = idx_theta2.float()
                self.cos_cached = idx_theta2.cos()[None, :, None, :]
                self.sin_cached = idx_theta2.sin()[None, :, None, :]
            self.cos_cached = self.cos_cached
            self.sin_cached = self.sin_cached
        x_rope, x_pass = x[..., :self.d_rope], x[..., self.d_rope:]
        cos, sin = self.cos_cached[:, offset:seq_len], self.sin_cached[:, offset:seq_len]
        x_rope = x_rope * cos + self.rotate_half(x_rope) * sin
        return torch.cat((x_rope, x_pass), dim=-1)


class AttentionLayer(nn.Module):
    """
    ## Attention layer
    """

    def __init__(self, n_hidden: int=6144, n_heads: int=64, rope_percentage: float=0.25, mask_fill: float=-10000.0):
        """
        :param n_hidden: the number of features in embeddings
        :param n_heads: the number of attention heads
        :param rope_percentage: percentage of features to add RoPE embeddings
        :param mask_fill: masking fill value for attention matrix
        """
        super().__init__()
        self.n_heads = n_heads
        self.mask_fill = mask_fill
        self.qkv_lin = nn.Linear(n_hidden, n_hidden * 3)
        self.output = nn.Linear(n_hidden, n_hidden)
        d_k = n_hidden // n_heads
        self.rope = RoPE(int(d_k * rope_percentage))
        self.scale = 1 / math.sqrt(d_k)
        self.causal_mask = None
        self.softmax = nn.Softmax(dim=-2)

    def _get_mask(self, attn: torch.Tensor):
        """
        #### Calculate the causal mask

        * `attn` has shape [batch_size, query_seq_len, key_seq_len, n_heads]
        """
        nq, nk = attn.shape[1:3]
        if self.causal_mask is None or self.causal_mask.shape[0] != nq or self.causal_mask.shape[1] != nk or self.causal_mask.device != attn.device:
            self.causal_mask = torch.triu(attn.new_ones([nq, nk], dtype=torch.bool), 1 + nk - nq)
        return self.causal_mask[None, :, :, None]

    def forward(self, x: torch.Tensor):
        """
        :param x: has shape `[batch_size, seq_len, n_hidden]`
        """
        qkv = self.qkv_lin(x)
        qkv = qkv.view(*qkv.shape[:-1], self.n_heads, -1)
        q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)
        if get_cache().get('use_cache', False):
            prev_state_id, next_state_id = get_cache().get('state_ids')
            if prev_state_id is not None:
                k_past, v_past = get_cache().pop(f'attn_kv_{prev_state_id}')
                offset = k_past.shape[1]
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)
                k = torch.cat([k_past, k], dim=1)
                v = torch.cat([v_past, v], dim=1)
            else:
                q = self.rope(q)
                k = self.rope(k)
            get_cache().push(f'attn_kv_{next_state_id}', (k, v))
        else:
            q = self.rope(q)
            k = self.rope(k)
        with autocast(enabled=False):
            if q.dtype == torch.float16:
                attn = torch.einsum('bihk,bjhk->bijh', q.float(), k.float())
            else:
                attn = torch.einsum('bihk,bjhk->bijh', q, k)
            attn = attn * self.scale
            mask = self._get_mask(attn)
            attn.masked_fill_(mask, self.mask_fill)
            attn = self.softmax(attn)
        output = torch.einsum('bijh,bjhk->bihk', attn, v)
        output = output.reshape(*x.shape)
        return self.output(output)


class FFNLayer(nn.Module):
    """
    ## Feedforward Network
    """

    def __init__(self, n_hidden: int=6144):
        """
        :param n_hidden: is the embedding size
        """
        super().__init__()
        self.dense_h_h4 = nn.Linear(n_hidden, n_hidden * 4)
        self.activation = nn.GELU()
        self.dense_h4_h = nn.Linear(n_hidden * 4, n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: has shape `[batch_size, seq_len, n_hidden]`
        """
        x = self.dense_h_h4(x)
        x = self.activation(x)
        x = self.dense_h4_h(x)
        return x


class TransformerLayer(nn.Module):
    """
    ## Transformer Layer
    """

    def __init__(self, n_hidden: int=6144, n_heads: int=64):
        """
        :param n_hidden: is the embedding size
        :param n_heads: is the number of heads

        *Out implementation doesn't include dropout*.
        """
        super().__init__()
        self.pre_ln_attn = nn.LayerNorm(n_hidden)
        self.pre_ln_ffn = nn.LayerNorm(n_hidden)
        self.attention = AttentionLayer(n_hidden, n_heads)
        self.ffn = FFNLayer(n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """
        residual = x
        attn = self.attention(self.pre_ln_attn(x))
        ffn = self.ffn(self.pre_ln_ffn(x))
        return attn + ffn + residual

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load transformer layer'):
            checkpoint.merge_params_sum(self.attention.output.bias, 'attention.dense.bias', p1, p2)
            checkpoint.merge_params_dim_1(self.attention.output.weight, 'attention.dense.weight', p1, p2)
            checkpoint.merge_params_dim_0(self.attention.qkv_lin.bias, 'attention.query_key_value.bias', p1, p2)
            checkpoint.merge_params_dim_0(self.attention.qkv_lin.weight, 'attention.query_key_value.weight', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_attn.bias, 'input_layernorm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_attn.weight, 'input_layernorm.weight', p1, p2)
            checkpoint.merge_params_dim_0(self.ffn.dense_h_h4.bias, 'mlp.dense_h_to_4h.bias', p1, p2)
            checkpoint.merge_params_dim_0(self.ffn.dense_h_h4.weight, 'mlp.dense_h_to_4h.weight', p1, p2)
            checkpoint.merge_params_sum(self.ffn.dense_h4_h.bias, 'mlp.dense_4h_to_h.bias', p1, p2)
            checkpoint.merge_params_dim_1(self.ffn.dense_h4_h.weight, 'mlp.dense_4h_to_h.weight', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_ffn.bias, 'post_attention_layernorm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_ffn.weight, 'post_attention_layernorm.weight', p1, p2)


class FinalNorm(nn.Module):
    """
    ## Final normalization layer
    """

    def __init__(self, n_hidden: int=6144):
        """
        :param n_hidden: is the embedding size
        """
        super().__init__()
        self.ln = nn.LayerNorm(n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """
        return self.ln(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load final normalization layer'):
            checkpoint.merge_params_duplicate(self.ln.bias, 'norm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.ln.weight, 'norm.weight', p1, p2)


class ReadoutLayer(nn.Module):
    """
    Readout layer
    """

    def __init__(self, n_hidden: int=6144, n_vocab: int=50432):
        """
        :param n_hidden: is the embedding size
        :param n_vocab: is the size of the vocabulary
        """
        super().__init__()
        self.linear = nn.Linear(n_hidden, n_vocab)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """
        return self.linear(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load final linear layer'):
            checkpoint.merge_params_dim_0(self.linear.weight, 'final_linear.weight', p1, p2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DummyModel,
     lambda: ([], {'n_vocab': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RoPE,
     lambda: ([], {'d_rope': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_labmlai_neox(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

