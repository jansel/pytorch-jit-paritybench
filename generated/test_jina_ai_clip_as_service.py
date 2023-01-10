import sys
_module = sys.modules[__name__]
del sys
clip_client = _module
client = _module
helper = _module
setup = _module
conf = _module
benchmark = _module
onnx_helper = _module
clip_server = _module
executors = _module
clip_onnx = _module
clip_tensorrt = _module
clip_torch = _module
helper = _module
helper = _module
model = _module
clip = _module
clip_model = _module
clip_trt = _module
flash_attention = _module
mclip_model = _module
model = _module
openclip_model = _module
pretrained_models = _module
simple_tokenizer = _module
tokenization = _module
trt_utils = _module
setup = _module
tests = _module
conftest = _module
test_asyncio = _module
test_client = _module
test_helper = _module
test_model = _module
test_ranker = _module
test_search = _module
test_server = _module
test_simple = _module
test_tensorrt = _module
test_tokenization = _module

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


import inspect


import warnings


from functools import partial


from typing import Dict


from typing import Optional


import numpy as np


from typing import Union


import torch


from typing import Tuple


from typing import List


from typing import Callable


from typing import Any


import torch.nn as nn


from torch import Tensor


from torch.nn.functional import linear


from torch import nn


from copy import deepcopy


from typing import OrderedDict


class MultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) ->None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    def attention(self, q, k, v, batch_size=1, seqlen=77, softmax_scale=None, attention_dropout=0.0, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value. each of (B*S, H, D)
            key_padding_mask: a bool tensor of shape (B, S)

        """
        if cu_seqlens is None:
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
            output = flash_attn_unpadded_func(q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, attention_dropout, softmax_scale=softmax_scale, causal=causal)
        return output

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor]=None, need_weights: bool=False, attn_mask: Optional[Tensor]=None, average_attn_weights: bool=True) ->Tuple[Tensor, Optional[Tensor]]:
        seqlen, batch_size, embed_dim = query.shape
        q, k, v = linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        k = k.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        v = v.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        causal = attn_mask is not None
        attn_output = self.attention(q, k, v, batch_size, seqlen, causal=causal)
        attn_output = attn_output.contiguous().view(batch_size, seqlen, self.num_heads, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seqlen, batch_size, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output, None

