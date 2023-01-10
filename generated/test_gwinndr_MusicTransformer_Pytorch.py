import sys
_module = sys.modules[__name__]
del sys
e_piano = _module
evaluate = _module
generate = _module
graph_results = _module
loss = _module
music_transformer = _module
positional_encoding = _module
rpr = _module
preprocess_midi = _module
train = _module
argument_funcs = _module
constants = _module
device = _module
lr_scheduling = _module
run_model = _module

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


import torch


import torch.nn as nn


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim import Adam


import torch.nn.functional as F


from torch.nn.modules.loss import _Loss


from torch.nn.modules.normalization import LayerNorm


import math


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from torch.nn import Module


from torch.nn.modules.transformer import _get_clones


from torch.nn.modules.linear import Linear


from torch.nn.modules.dropout import Dropout


from torch.nn.init import *


from torch.nn.functional import linear


from torch.nn.functional import softmax


from torch.nn.functional import dropout


from torch.optim.lr_scheduler import LambdaLR


import time


class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)
        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """
        return memory


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


TORCH_LABEL_TYPE = torch.long


def _get_valid_embedding(Er, len_q, len_k):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Gets valid embeddings based on max length of RPR attention
    ----------
    """
    len_e = Er.shape[0]
    start = max(0, len_e - len_q)
    return Er[start:, :]


def _skew(qe):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Performs the skew optimized RPR computation (https://arxiv.org/abs/1809.04281)
    ----------
    """
    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz)) == 1).float().flip(0)
    qe = mask * qe
    qe = F.pad(qe, (1, 0, 0, 0, 0, 0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))
    srel = qe[:, 1:, :]
    return srel


def multi_head_attention_forward_rpr(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None, rpr_mat=None):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/functional.html

    Modification to take RPR embedding matrix and perform skew optimized RPR (https://arxiv.org/abs/1809.04281)
    ----------
    """
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    if use_separate_proj_weight is not True:
        if qkv_same:
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:embed_dim * 2])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim * 2:])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if rpr_mat is not None:
        rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
        qe = torch.einsum('hld,md->hlm', q, rpr_mat)
        srel = _skew(qe)
        attn_output_weights += srel
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttentionRPR(Module):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Modification to add RPR embedding Er and call custom multi_head_attention_forward_rpr
    ----------
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, er_len=None):
        super(MultiheadAttentionRPR, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        if er_len is not None:
            self.Er = Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32))
        else:
            self.Er = None
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward_rpr(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, rpr_mat=self.Er)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented.                     Please re-train your model with the new module', UserWarning)
            return multi_head_attention_forward_rpr(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, rpr_mat=self.Er)


class TransformerEncoderLayerRPR(Module):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    Modification to create and call custom MultiheadAttentionRPR
    ----------
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerEncoderLayerRPR, self).__init__()
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderRPR(Module):
    """
    ----------
    Author: Pytorch
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoder

    No modification. Copied here to ensure continued compatibility with other edits.
    ----------
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderRPR, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output


TORCH_CPU_DEVICE = torch.device('cpu')


USE_CUDA = True


def get_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the default device. Default device is CUDA if available and use_cuda is not False, CPU otherwise.
    ----------
    """
    if not USE_CUDA or TORCH_CUDA_DEVICE is None:
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE


class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1, max_sequence=2048, rpr=False):
        super(MusicTransformer, self).__init__()
        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.rpr = rpr
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)
        if not self.rpr:
            self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers, num_decoder_layers=0, dropout=self.dropout, dim_feedforward=self.d_ff, custom_decoder=self.dummy)
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers, num_decoder_layers=0, dropout=self.dropout, dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder)
        self.Wout = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """
        if mask is True:
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1])
        else:
            mask = None
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)
        x_out = x_out.permute(1, 0, 2)
        y = self.Wout(x_out)
        del mask
        return y

    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """
        assert not self.training, 'Cannot generate while in training mode'
        None
        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE)
        cur_i = num_primer
        while cur_i < target_seq_length:
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i - 1, :]
            if beam == 0:
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0, 1)
            if beam_ran <= beam_chance:
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                if next_token == TOKEN_END:
                    None
                    break
            cur_i += 1
            if cur_i % 50 == 0:
                None
        return gen_seq[:, :cur_i]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DummyDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttentionRPR,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothCrossEntropyLoss,
     lambda: ([], {'label_smoothing': 0, 'vocab_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerEncoderLayerRPR,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_gwinndr_MusicTransformer_Pytorch(_paritybench_base):
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

