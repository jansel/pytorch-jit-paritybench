import sys
_module = sys.modules[__name__]
del sys
classify = _module
generate = _module
former = _module
modules = _module
transformers = _module
util = _module
util = _module
setup = _module
compression = _module
test_modules = _module
test_util = _module

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


from torch import nn


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


from torch.utils.tensorboard import SummaryWriter


import random


import math


import torch.distributions as dist


import time


from collections.abc import Sequence


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """
    h, w = matrices.size(-2), matrices.size(-1)
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """
        super().__init__()
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'
        self.emb = emb
        self.heads = heads
        self.mask = mask
        s = emb // heads
        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        s = e // h
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries / e ** (1 / 4)
        keys = keys / e ** (1 / 4)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.unifyheads(out)


class SelfAttentionNarrow(nn.Module):
    """
    A self attention with a reduced parameter space (experimental).

    * Uses _the same_ key/query/value transformation on each head, but applied to a different slice of the embedding vector.
    * Dispenses with the linear layer after merging the heads.

    """

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """
        super().__init__()
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'
        self.emb = emb
        self.heads = heads
        self.mask = mask
        s = emb // heads
        self.tokeys = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues = nn.Linear(s, s, bias=False)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        s = e // h
        x = x.view(b, t, h, s)
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries / e ** (1 / 4)
        keys = keys / e ** (1 / 4)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)
        return out


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    from:

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.

    NB: Note the illogical argument order.
    """

    def __init__(self, nf, nx, he=True):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        b = torch.zeros(nf)
        if not he:
            nn.init.normal_(w, std=0.02)
        else:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        x = torch.addmm(self.bias, x, self.weight)
        x = x.view(*size_out)
        return x


class SelfAttentionGPT2(nn.Module):
    """
    This is the self-attention operation as implemented in the Huggingface port of GPT2. The code has been
    simplified to remove several features not used here but otherwise it should do exactly the same as GPT2 when run with
    normal parameters.

    It is very similar to the default SelfAttention below, with the exception of the way it's initialized and some
    small speed improvements in the custom implementation of the linear layer (the Conv1D defined above).

    We include this primarily for comparison with our own canonical implementation to check for performance differences.
    """

    def __init__(self, emb, heads, mask=False):
        super().__init__()
        self.nheads = heads
        self.emb = emb
        self.mask = mask
        self.c_attn = nn.Linear(emb, 3 * emb)
        self.c_proj = nn.Linear(emb, emb)

    def _attn(self, q, k, v):
        dot = torch.matmul(q, k)
        dot = dot / float(v.size(-1)) ** 0.5
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = nn.Softmax(dim=-1)(dot)
        return torch.matmul(dot, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.nheads, x.size(-1) // self.nheads)
        x = x.view(*new_x_shape)
        if is_key:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, input_sequence):
        b, t, e = input_sequence.size()
        query, key, value = self.c_attn(input_sequence).split(e, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class SelfAttentionWide(nn.Module):
    """
    A self-attention with a larger number of parameters than the standard one.

    Uses a full-size embedding vector for each head.
    """

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries / e ** (1 / 4)
        keys = keys / e ** (1 / 4)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def slice_diag(matrix, l, dv=None):
    """
    Take a batch of attention matrices for relative position encodings and slice out the relevant attentions. These
    are the length l sequences starting at the diagonal

    :param matrix:
    :return:
    """
    if dv is None:
        dv = d(matrix)
    h, w = matrix.size(-2), matrix.size(-1)
    assert w == 2 * l - 1, f'(h, w)= {h, w}, l={l}'
    rest = matrix.size()[:-2]
    matrix = matrix.view(-1, h, w)
    b, h, w = matrix.size()
    result = matrix.view(b, -1)
    result = torch.cat([result, torch.zeros(b, l, device=dv)], dim=1)
    assert result.size() == (b, 2 * l * l), f'result.size() {result.size()}'
    result = result.view(b, l, 2 * l)
    result = result[:, :, :l]
    result = result.view(*rest, h, l)
    return result


class SelfAttentionRelative(nn.Module):
    """
    Implementation of self-attention with relative position embeddings.

    Inspired by the Transformer-XL relative positions. Not guaranteed to be exactly the same. See
      https://youtu.be/oUhGZMCTHtI
    for an explanation.

    """

    def __init__(self, emb, pos_embedding, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """
        super().__init__()
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.pos = pos_embedding
        e, s, h = emb, emb // heads, heads
        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.tokeys_pos = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb)
        self.parma, self.parmb = nn.Parameter(torch.randn(1, h, 1, s)), nn.Parameter(torch.randn(1, h, 1, s))

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        s = e // h
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
        positions = self.pos(torch.arange(2 * t - 1, device=d(x)))[None, :].expand(b, 2 * t - 1, e)
        keys_pos = self.tokeys_pos(positions)
        assert keys_pos.size() == (b, 2 * t - 1, e)
        keys = keys.view(b, t, h, s)
        keys_pos = keys_pos.view(b, 2 * t - 1, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        keys_pos = keys_pos.transpose(1, 2).contiguous().view(b * h, 2 * t - 1, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)
        parma = self.parma.expand(b, h, t, s).contiguous().view(b * h, t, s)
        parmb = self.parmb.expand(b, h, t, s).contiguous().view(b * h, t, s)
        dot_tt = torch.einsum('bis, bjs -> bij', queries, keys)
        assert dot_tt.size() == (b * h, t, t), f'{dot_tt.size()}'
        dot_tp = torch.einsum('bis, bjs -> bij', queries, keys_pos)
        dot_tp = slice_diag(dot_tp, l=t)
        assert dot_tp.size() == (b * h, t, t), f'{dot_tp.size()}'
        dot_pt = torch.einsum('bis, bjs -> bij', parma, keys)
        assert dot_pt.size() == (b * h, t, t), f'{dot_pt.size()}'
        dot_pp = torch.einsum('bis, bjs -> bij', parmb, keys_pos)
        dot_pp = slice_diag(dot_pp, l=t)
        assert dot_pp.size() == (b * h, t, t), f'{dot_pp.size()}'
        dot = dot_tt + dot_tp + dot_pt + dot_pp
        assert dot.size() == (b * h, t, t)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default', pos_embedding=None):
        super().__init__()
        if attention_type == 'default':
            self.attention = SelfAttention(emb, heads=heads, mask=mask)
        elif attention_type == 'wide':
            self.attention = SelfAttentionWide(emb, heads=heads, mask=mask)
        elif attention_type == 'gpt2':
            self.attention = SelfAttentionGPT2(emb, heads=heads, mask=mask)
        elif attention_type == 'narrow':
            self.attention = SelfAttentionNarrow(emb, heads=heads, mask=mask)
        elif attention_type == 'relative':
            assert pos_embedding is not None
            self.attention = SelfAttentionRelative(emb, heads=heads, mask=mask, pos_embedding=pos_embedding)
        else:
            raise Exception(f'Self-attention type {type} not recognized.')
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(nn.Linear(emb, ff_hidden_mult * emb), nn.ReLU(), nn.Linear(ff_hidden_mult * emb, emb))
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x


class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length * 2 - 1 if attention_type == 'relative' else seq_length)
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type, pos_embedding=self.pos_embedding))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.tblocks(x)
        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)
        return F.log_softmax(x, dim=2)


class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()
        self.num_tokens, self.max_pool = num_tokens, max_pool
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)
        x = self.tblocks(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv1D,
     lambda: ([], {'nf': 4, 'nx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfAttentionGPT2,
     lambda: ([], {'emb': 4, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SelfAttentionWide,
     lambda: ([], {'emb': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerBlock,
     lambda: ([], {'emb': 4, 'heads': 4, 'mask': 4, 'seq_length': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_pbloem_former(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

