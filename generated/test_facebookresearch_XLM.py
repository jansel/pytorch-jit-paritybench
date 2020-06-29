import sys
_module = sys.modules[__name__]
del sys
preprocess = _module
src = _module
data = _module
dataset = _module
dictionary = _module
loader = _module
evaluation = _module
evaluator = _module
glue = _module
xnli = _module
logger = _module
model = _module
embedder = _module
memory = _module
memory = _module
query = _module
utils = _module
pretrain = _module
transformer = _module
optim = _module
slurm = _module
trainer = _module
lowercase_and_remove_accent = _module
segment_th = _module
train = _module
translate = _module

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


from logging import getLogger


import copy


import time


from collections import OrderedDict


import numpy as np


import torch


from torch import nn


import torch.nn.functional as F


from scipy.stats import spearmanr


from scipy.stats import pearsonr


import math


import itertools


from torch.nn import functional as F


import torch.nn as nn


from torch.nn.utils import clip_grad_norm_


def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


class GroupedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias
        assert groups > 1
        self.layer = nn.Conv1d(in_features, out_features, bias=bias,
            kernel_size=1, groups=groups)

    def forward(self, input):
        assert input.dim() == 2 and input.size(1) == self.in_features
        return self.layer(input.unsqueeze(2)).squeeze(2)

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.in_features, self.out_features, self.groups, self.bias is not
            None)


class BottleneckResidualConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=
        True, batchnorm=True, groups=1):
        super().__init__()
        hidden_channels = min(input_channels, output_channels)
        assert all(k % 2 == 1 for k in kernel_size)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size,
            padding=[(k // 2) for k in kernel_size], bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels,
            kernel_size, padding=[(k // 2) for k in kernel_size], bias=bias,
            groups=groups)
        self.act = nn.ReLU()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)
        if input_channels == output_channels:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Conv2d(input_channels, output_channels, (1, 
                1), bias=False, groups=groups)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) if self.batchnorm else x
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batchnorm else x
        x = self.act(x + self.residual(input))
        return x


def get_slices(dim, head_id):
    """
    Generate slices of hidden dimensions.
    Used when there are multiple heads and/or different set of keys,
    and that there is no query network.
    """
    if head_id == 0:
        return [(0, dim)]
    offset = dim // 2 ** (head_id + 1)
    starts = np.arange(0, dim, offset)
    slices1 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 0]
    slices2 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 1]
    return slices1 + slices2


class QueryIdentity(nn.Module):

    def __init__(self, input_dim, heads, shuffle_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.shuffle_query = shuffle_hidden
        assert shuffle_hidden is False or heads > 1
        assert shuffle_hidden is False or self.input_dim % 2 ** self.heads == 0
        if shuffle_hidden:
            self.slices = {head_id: get_slices(input_dim, head_id) for
                head_id in range(heads)}

    def forward(self, input):
        """
        Generate queries from hidden states by either
        repeating them or creating some shuffled version.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim(
            ) > 2 else input
        bs = len(input)
        if self.heads == 1:
            query = input
        elif not self.shuffle_query:
            query = input.unsqueeze(1).repeat(1, self.heads, 1)
            query = query.view(bs * self.heads, self.input_dim)
        else:
            query = torch.cat([input[:, a:b] for head_id in range(self.
                heads) for a, b in self.slices[head_id]], 1).view(bs * self
                .heads, self.input_dim)
        assert query.shape == (bs * self.heads, self.input_dim)
        return query


def mlp(sizes, bias=True, batchnorm=True, groups=1):
    """
    Generate a feedforward neural network.
    """
    assert len(sizes) >= 2
    pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        if groups == 1 or i == 0:
            layers.append(nn.Linear(dim_in, groups * dim_out, bias=bias))
        else:
            layers.append(GroupedLinear(groups * dim_in, groups * dim_out,
                bias=bias, groups=groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(groups * dim_out))
        if i < len(pairs) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class QueryMLP(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization,
        multi_query_net, sizes, bias=True, batchnorm=True, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_mlps = mlp(sizes, bias=bias, batchnorm=batchnorm,
                groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_mlps = mlp(sizes_, bias=bias, batchnorm=batchnorm,
                groups=1)
        else:
            self.query_mlps = nn.ModuleList([mlp(sizes, bias=bias,
                batchnorm=batchnorm, groups=1) for _ in range(self.groups)])

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim(
            ) > 2 else input
        bs = len(input)
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_mlps(input)
        else:
            outputs = [m(input) for m in self.query_mlps]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)


def convs(channel_sizes, kernel_sizes, bias=True, batchnorm=True, residual=
    False, groups=1):
    """
    Generate a convolutional neural network.
    """
    assert len(channel_sizes) >= 2
    assert len(channel_sizes) == len(kernel_sizes) + 1
    pairs = [(channel_sizes[i], channel_sizes[i + 1]) for i in range(len(
        channel_sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        ks = kernel_sizes[i], kernel_sizes[i]
        in_group = 1 if i == 0 else groups
        _dim_in = dim_in * in_group
        _dim_out = dim_out * groups
        if not residual:
            layers.append(nn.Conv2d(_dim_in, _dim_out, ks, padding=[(k // 2
                ) for k in ks], bias=bias, groups=in_group))
            if batchnorm:
                layers.append(nn.BatchNorm2d(_dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())
        else:
            layers.append(BottleneckResidualConv2d(_dim_in, _dim_out, ks,
                bias=bias, batchnorm=batchnorm, groups=in_group))
            if i == len(pairs) - 1:
                layers.append(nn.Conv2d(_dim_out, _dim_out, (1, 1), bias=bias))
    return nn.Sequential(*layers)


class QueryConv(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization,
        multi_query_net, sizes, kernel_sizes, bias=True, batchnorm=True,
        residual=False, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        assert len(sizes) == len(kernel_sizes) + 1 >= 2 and all(ks % 2 == 1 for
            ks in kernel_sizes)
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_convs = convs(sizes, kernel_sizes, bias=bias,
                batchnorm=batchnorm, residual=residual, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_convs = convs(sizes_, kernel_sizes, bias=bias,
                batchnorm=batchnorm, residual=residual, groups=1)
        else:
            self.query_convs = nn.ModuleList([convs(sizes, kernel_sizes,
                bias=bias, batchnorm=batchnorm, residual=residual, groups=1
                ) for _ in range(self.groups)])

    def forward(self, input):
        bs, nf, h, w = input.shape
        assert nf == self.input_dim
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_convs(input)
        else:
            outputs = [m(input) for m in self.query_convs]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim, h, w)
        query = query.transpose(1, 3).contiguous().view(bs * w * h * self.
            heads, self.k_dim)
        return query


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    return m


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim
        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim,
                n_classes=params.n_words, cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value, head_bias=True)

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0
        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None
        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (
            dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 
            1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads *
                dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, v)
        context = unshape(context)
        return self.out_lin(context)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in
                    enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (self.worst_score >= best_sum_logprobs / self.max_len **
                self.length_penalty)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


N_MAX_POSITIONS = 512


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for
        j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, (None)]
    if causal:
        attn_mask = alen[(None), (None), :].repeat(bs, slen, 1) <= alen[(
            None), :, (None)]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask


class TransformerModel(nn.Module):
    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index',
        'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim',
        'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, 'use_lang_emb', True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs
        self.dim = params.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_heads = params.n_heads
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=
                self.position_embeddings.weight)
        if params.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=
            self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()
        self.memories = nn.ModuleDict()
        if getattr(params, 'use_memory', False):
            mem_positions = (params.mem_enc_positions if is_encoder else
                params.mem_dec_positions)
            for layer_id, pos in mem_positions:
                assert 0 <= layer_id <= params.n_layers - 1
                assert pos in ['in', 'after']
                self.memories['%i_%s' % (layer_id, pos)] = HashingMemory.build(
                    self.dim, self.dim, params)
        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.
                dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads,
                    self.dim, dropout=self.attention_dropout))
            if '%i_in' % layer_id in self.memories:
                self.ffns.append(None)
            else:
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim,
                    self.dim, dropout=self.dropout, gelu_activation=params.
                    gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception('Unknown mode: %s' % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions
        =None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device
                =lengths.device) < src_len[:, (None)]
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1)
        for i in range(self.n_layers):
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc,
                    cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
            if '%i_in' % i in self.memories:
                tensor = tensor + self.memories['%i_in' % i](tensor)
            else:
                tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            if '%i_after' % i in self.memories:
                tensor = tensor + self.memories['%i_after' % i](tensor)
            tensor *= mask.unsqueeze(-1)
        if cache is not None:
            cache['slen'] += tensor.size(1)
        tensor = tensor.transpose(0, 1)
        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(
            -1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200,
        sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        bs = len(src_len)
        assert src_enc.size(0) == bs
        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(
            max_len, bs)
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)
        cache = {'slen': 0}
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=
                gen_len, positions=positions[:cur_len], langs=langs[:
                cur_len], causal=True, src_enc=src_enc, src_len=src_len,
                cache=cache)
            assert tensor.size() == (1, bs, self.dim), (cur_len, max_len,
                src_enc.size(), tensor.size(), (1, bs, self.dim))
            tensor = tensor.data[(-1), :, :].type_as(src_enc)
            scores = self.pred_layer.get_scores(tensor)
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores /
                    sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)
            generated[cur_len
                ] = next_words * unfinished_sents + self.pad_index * (1 -
                unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size,
        length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.
            shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(
            -1)
        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty,
            early_stopping) for _ in range(bs)]
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1
            ).expand_as(generated)
        langs = positions.clone().fill_(tgt_lang_id)
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1000000000.0
        beam_scores = beam_scores.view(-1)
        cur_len = 1
        cache = {'slen': 0}
        done = [(False) for _ in range(bs)]
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=
                src_len.new(bs * beam_size).fill_(cur_len), positions=
                positions[:cur_len], langs=langs[:cur_len], causal=True,
                src_enc=src_enc, src_len=src_len, cache=cache)
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[(-1), :, :]
            scores = self.pred_layer.get_scores(tensor)
            scores = F.log_softmax(scores, dim=-1)
            assert scores.size() == (bs * beam_size, n_words)
            _scores = scores + beam_scores[:, (None)].expand_as(scores)
            _scores = _scores.view(bs, beam_size * n_words)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size,
                dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 *
                beam_size)
            next_batch_beam = []
            for sent_id in range(bs):
                done[sent_id] = done[sent_id] or generated_hyps[sent_id
                    ].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size
                        )
                    continue
                next_sent_beam = []
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]
                    ):
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, (
                            sent_id * beam_size + beam_id)].clone(), value.
                            item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id *
                            beam_size + beam_id))
                    if len(next_sent_beam) == beam_size:
                        break
                assert len(next_sent_beam
                    ) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])
            generated = generated[:, (beam_idx)]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = cache[k][0][beam_idx], cache[k][1][beam_idx]
            cur_len = cur_len + 1
            if all(done):
                break
        tgt_len = src_len.new(bs)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, (i)] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index
        assert (decoded == self.eos_index).sum() == 2 * bs
        return decoded, tgt_len


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_XLM(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(MultiHeadAttention(*[], **{'n_heads': 4, 'dim': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(TransformerFFN(*[], **{'in_dim': 4, 'dim_hidden': 4, 'out_dim': 4, 'dropout': 0.5, 'gelu_activation': 4}), [torch.rand([4, 4, 4, 4])], {})

