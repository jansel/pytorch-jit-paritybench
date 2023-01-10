import sys
_module = sys.modules[__name__]
del sys
data = _module
corpus = _module
dataset = _module
tokenization = _module
vocabulary = _module
evaluate_model = _module
evaluation = _module
configuration = _module
evaluation = _module
specification = _module
generate_sentences = _module
generation = _module
generation = _module
specification = _module
modeling = _module
attention = _module
embedding = _module
feedforward = _module
masking = _module
transformer = _module
train_model = _module
training = _module
recording = _module
specification = _module
training = _module
utils = _module
fusing = _module
visualize_metrics = _module
tests = _module
test_corpus = _module
test_tokenization = _module
test_vocabulary = _module
test_generation = _module
test_attention = _module
test_embedding = _module
test_feedforward = _module
test_masking = _module
test_transformer = _module
test_configuration = _module
test_recording = _module

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


from typing import Dict


from typing import Any


from typing import List


from typing import Optional


import torch.nn as nn


from typing import Tuple


import math


import torch.utils.checkpoint


from functools import partial


from typing import Union


import torch.optim as optim


from typing import Iterator


import torch.distributed as dist


import torch.multiprocessing as mp


import warnings


import matplotlib.pyplot as plt


import numpy as np


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-10000.0)
        x = self.dropout(x.softmax(-1))
        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dropout: float=0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        if mask is not None:
            mask = mask.unsqueeze(-3)
        return super().forward(q, k, v, mask).transpose(-3, -2).contiguous().view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads))


Past = Tuple[torch.Tensor, torch.Tensor]


class AttentionLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., query_len, past_len + kv_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., query_len, dims)
    output 2 (*)    float           (..., past_len + kv_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dims: int, dropout: float=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past: Optional[Past]=None, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, Past]:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        x = self.linear(self.attn(q, k, v, mask))
        return x, (k, v)


class PositionalEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
    ===========================================================================
    """

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self, state_dict: Dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight = state_dict[f'{prefix}weight']
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]
        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int=0) ->torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1), dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)
        return super().forward(position)


class TokenEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long or float  (..., seq_len)
                                    or (..., seq_len, embedding_dim)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
                                    or (..., seq_len, num_embeddings)
    ===========================================================================
    """

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, transposed: bool=False) ->torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)


class Swish(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x * self.sigmoid(x)


class PositionwiseFeedForward(nn.Sequential):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """

    def __init__(self, dims: int, rate: int=4, dropout: float=0.1):
        super().__init__(nn.Linear(dims, dims * rate), Swish(), nn.Dropout(dropout), nn.Linear(dims * rate, dims))


class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """

    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int=0) ->torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, offset), dtype=torch.bool, device=x.device)
        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])


class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """

    def forward(self, x: torch.Tensor, offset: int=0) ->torch.Tensor:
        seq_len = x.size(-1)
        future = torch.ones((seq_len, seq_len + offset), dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)
        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])


class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dims: int, rate: int, dropout: float=0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)

    def forward(self, x: torch.Tensor, past: Optional[Past]=None, mask: Optional[torch.Tensor]=None) ->Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)
        x = x + a
        x = x + self.ff(self.ln_ff(x))
        return x if self.training else (x, past)


class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(self, layers: int, pad_idx: int, words: int, seq_len: int, heads: int, dims: int, rate: int=4, dropout: float=0.1, bidirectional: bool=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()
        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)
        self.transformers = nn.ModuleList([TransformerLayer(heads, dims, rate, dropout) for _ in range(layers)])
        self.ln_head = LayerNorm(dims)

    def forward(self, x: torch.Tensor, past: Optional[List[Past]]=None, use_grad_ckpt: bool=False) ->Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)
            x = transformer(x, past[i] if past is not None else None, mask)
            if not self.training:
                present.append(x[1])
                x = x[0]
        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)
        return x if self.training else (x, present)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionLayer,
     lambda: ([], {'heads': 4, 'dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BaseAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FutureMasking,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PadMasking,
     lambda: ([], {'pad_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionwiseFeedForward,
     lambda: ([], {'dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_affjljoo3581_GPT2(_paritybench_base):
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

