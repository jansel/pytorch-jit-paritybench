import sys
_module = sys.modules[__name__]
del sys
alphafold2_pytorch = _module
alphafold2 = _module
constants = _module
embeds = _module
mlm = _module
reversible = _module
rotary = _module
utils = _module
refinement = _module
setup = _module
test_attention = _module
test_utils = _module
train_end2end = _module
train_pre = _module
datasets = _module
trrosetta = _module
deepspeed = _module
lightning = _module

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


from torch import einsum


from torch.utils.checkpoint import checkpoint


from torch.utils.checkpoint import checkpoint_sequential


from inspect import isfunction


from functools import partial


import torch.nn.functional as F


from math import sqrt


import math


import torch.nn as nn


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


from math import log


from math import pi


import re


import numpy as np


from functools import wraps


import itertools


import string


from torch.optim import Adam


from torch.utils.data import DataLoader


from typing import Callable


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy.linalg as LA


from torch.utils.data import Dataset


class Always(nn.Module):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def exists(val):
    return val is not None


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Attention(nn.Module):

    def __init__(self, dim, seq_len=None, heads=8, dim_head=64, dropout=0.0, gating=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)
        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        if exists(tie_dim):
            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)
            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)
        if exists(attn_bias):
            dots = dots + attn_bias
        if exists(mask):
            mask = default(mask, lambda : torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda : torch.ones(1, k.shape[-2], device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        gates = self.gating(x)
        out = out * gates.sigmoid()
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):

    def __init__(self, dim, heads, row_attn=True, col_attn=True, accept_edges=False, global_query_attn=False, **kwargs):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, **kwargs)
        self.edges_to_attn_bias = nn.Sequential(nn.Linear(dim, heads, bias=False), Rearrange('b i j h -> b h i j')) if accept_edges else None

    def forward(self, x, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'
        b, h, w, d = x.shape
        x = self.norm(x)
        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'
        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'
        x = rearrange(x, input_fold_eq)
        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)
        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x=axial_dim)
        tie_dim = axial_dim if self.global_query_attn else None
        out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
        out = rearrange(out, output_fold_eq, h=h, w=w)
        return out


class TriangleMultiplicativeModule(nn.Module):

    def __init__(self, *, dim, hidden_dim=None, mix='ingoing'):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'
        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        if exists(mask):
            left = left * mask
            right = right * mask
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        left = left * left_gate
        right = right * right_gate
        out = einsum(self.mix_einsum_eq, left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class OuterMean(nn.Module):

    def __init__(self, dim, hidden_dim=None, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')
        if exists(mask):
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.0)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)
        return self.proj_out(outer)


class PairwiseAttentionBlock(nn.Module):

    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0, global_column_attn=False):
        super().__init__()
        self.outer_mean = OuterMean(dim)
        self.triangle_attention_outgoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, accept_edges=True)
        self.triangle_attention_ingoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=False, col_attn=True, accept_edges=True, global_query_attn=global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, mix='ingoing')

    def forward(self, x, mask=None, msa_repr=None, msa_mask=None):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask=msa_mask)
        x = self.triangle_multiply_outgoing(x, mask=mask) + x
        x = self.triangle_multiply_ingoing(x, mask=mask) + x
        x = self.triangle_attention_outgoing(x, edges=x, mask=mask) + x
        x = self.triangle_attention_ingoing(x, edges=x, mask=mask) + x
        return x


class MsaAttentionBlock(nn.Module):

    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, accept_edges=True)
        self.col_attn = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=False, col_attn=True)

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        x = self.col_attn(x, mask=mask) + x
        return x


class EvoformerBlock(nn.Module):

    def __init__(self, *, dim, seq_len, heads, dim_head, attn_dropout, ff_dropout, global_column_attn=False):
        super().__init__()
        self.layer = nn.ModuleList([PairwiseAttentionBlock(dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout, global_column_attn=global_column_attn), FeedForward(dim=dim, dropout=ff_dropout), MsaAttentionBlock(dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout), FeedForward(dim=dim, dropout=ff_dropout)])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer
        m = msa_attn(m, mask=msa_mask, pairwise_repr=x)
        m = msa_ff(m) + m
        x = attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask)
        x = ff(x) + x
        return x, m, mask, msa_mask


class Evoformer(nn.Module):

    def __init__(self, *, depth, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(self, x, m, mask=None, msa_mask=None):
        inp = x, m, mask, msa_mask
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1000000000.0)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


class MLM(nn.Module):

    def __init__(self, dim, num_tokens, mask_id, mask_prob=0.15, random_replace_token_prob=0.1, keep_token_same_prob=0.1, exclude_token_ids=(0,)):
        super().__init__()
        self.to_logits = nn.Linear(dim, num_tokens)
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.exclude_token_ids = exclude_token_ids
        self.keep_token_same_prob = keep_token_same_prob
        self.random_replace_token_prob = random_replace_token_prob

    def noise(self, seq, mask):
        num_msa = seq.shape[1]
        seq = rearrange(seq, 'b n ... -> (b n) ...')
        mask = rearrange(mask, 'b n ... -> (b n) ...')
        excluded_tokens_mask = mask
        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)
        mlm_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.mask_prob)
        replace_token_with_mask = get_mask_subset_with_prob(mlm_mask, 1.0 - self.keep_token_same_prob)
        seq = seq.masked_fill(mlm_mask, self.mask_id)
        random_replace_token_prob_mask = get_mask_subset_with_prob(mlm_mask, (1 - self.keep_token_same_prob) * self.random_replace_token_prob)
        random_tokens = torch.randint(1, constants.NUM_AMINO_ACIDS, seq.shape)
        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)
        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)
        noised_seq = rearrange(noised_seq, '(b n) ... -> b n ...', n=num_msa)
        mlm_mask = rearrange(mlm_mask, '(b n) ... -> b n ...', n=num_msa)
        return noised_seq, mlm_mask

    def forward(self, seq_embed, original_seq, mask):
        logits = self.to_logits(seq_embed)
        seq_logits = logits[mask]
        seq_labels = original_seq[mask]
        loss = F.cross_entropy(seq_logits, seq_labels, reduction='mean')
        return loss


PROTTRAN_EMBED_DIM = 1024


def ids_to_prottran_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []
    for ids in x:
        chars = ' '.join([id2aa[i] for i in ids])
        chars = re.sub('[UZOB]', 'X', chars)
        out.append(chars)
    return out


def get_prottran_embedd(seq, model, tokenizer, device=None):
    fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=-1 if not exists(device) else device.index)
    max_seq_len = seq.shape[1]
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())
    embedding = fe(embedd_inputs)
    embedding = torch.tensor(embedding, device=device)
    return embedding[:, 1:max_seq_len + 1]


class ProtTranEmbedWrapper(nn.Module):

    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2
        self.project_embed = nn.Linear(PROTTRAN_EMBED_DIM, alphafold2.dim)
        self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        self.model = AutoModel.from_pretrained('Rostlab/prot_bert')

    def forward(self, seq, msa, msa_mask=None, **kwargs):
        device = seq.device
        num_msa = msa.shape[1]
        msa_flat = rearrange(msa, 'b m n -> (b m) n')
        seq_embed = get_prottran_embedd(seq, self.model, self.tokenizer, device=device)
        msa_embed = get_prottran_embedd(msa_flat, self.model, self.tokenizer, device=device)
        seq_embed, msa_embed = map(self.project_embed, (seq_embed, msa_embed))
        msa_embed = rearrange(msa_embed, '(b m) n d -> b m n d', m=num_msa)
        return self.alphafold2(seq, msa, seq_embed=seq_embed, msa_embed=msa_embed, msa_mask=msa_mask, **kwargs)


MSA_EMBED_DIM = 768


MSA_MODEL_PATH = ['facebookresearch/esm', 'esm_msa1_t12_100M_UR50S']


def ids_to_embed_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []
    for el in x:
        if isinstance(el, list):
            out.append(ids_to_embed_input(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')
    if all(map(lambda c: isinstance(c, str), out)):
        return None, ''.join(out)
    return out


def get_msa_embedd(msa, embedd_model, batch_converter, device=None):
    """ Returns the MSA_tr embeddings for a protein.
        Inputs: 
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: MSA_tr model (see train_end2end.py for an example)
        * batch_converter: MSA_tr batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA
            * embedd_dim: number of embedding dimensions. 768 for MSA_Transformer
    """
    REPR_LAYER_NUM = 12
    device = seq.device
    max_seq_len = msa.shape[-1]
    embedd_inputs = ids_to_embed_input(msa.cpu().tolist())
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(msa_batch_tokens, repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    token_reps = results['representations'][REPR_LAYER_NUM][..., 1:max_seq_len + 1, :]
    return token_reps


class MSAEmbedWrapper(nn.Module):

    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2
        model, alphabet = torch.hub.load(*MSA_MODEL_PATH)
        batch_converter = alphabet.get_batch_converter()
        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(MSA_EMBED_DIM, alphafold2.dim) if MSA_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa, msa_mask=None, **kwargs):
        assert seq.shape[-1] == msa.shape[-1], 'sequence and msa must have the same length if you wish to use MSA transformer embeddings'
        model, batch_converter, device = self.model, self.batch_converter, seq.device
        seq_and_msa = torch.cat((seq.unsqueeze(1), msa), dim=1)
        if exists(msa_mask):
            num_msa = msa_mask.any(dim=-1).sum(dim=-1).tolist()
            seq_and_msa_list = seq_and_msa.unbind(dim=0)
            num_rows = seq_and_msa.shape[1]
            embeds = []
            for num, batch_el in zip(num_msa, seq_and_msa_list):
                batch_el = rearrange(batch_el, '... -> () ...')
                batch_el = batch_el[:, :num]
                embed = get_msa_embedd(batch_el, model, batch_converter, device=device)
                embed = F.pad(embed, (0, 0, 0, 0, 0, num_rows - num), value=0.0)
                embeds.append(embed)
            embeds = torch.cat(embeds, dim=0)
        else:
            embeds = get_msa_embedd(seq_and_msa, model, batch_converter, device=device)
        embeds = self.project_embed(embeds)
        seq_embed, msa_embed = embeds[:, 0], embeds[:, 1:]
        return self.alphafold2(seq, msa, seq_embed=seq_embed, msa_embed=msa_embed, msa_mask=msa_mask, **kwargs)


ESM_EMBED_DIM = 1280


ESM_MODEL_PATH = ['facebookresearch/esm', 'esm1b_t33_650M_UR50S']


def get_esm_embedd(seq, embedd_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for ESM-1b
            * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
    """
    device = seq.device
    REPR_LAYER_NUM = 33
    max_seq_len = seq.shape[-1]
    embedd_inputs = ids_to_embed_input(seq.cpu().tolist())
    batch_labels, batch_strs, batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(batch_tokens, repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    token_reps = results['representations'][REPR_LAYER_NUM][..., 1:max_seq_len + 1, :].unsqueeze(dim=1)
    return token_reps


class ESMEmbedWrapper(nn.Module):

    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2
        model, alphabet = torch.hub.load(*ESM_MODEL_PATH)
        batch_converter = alphabet.get_batch_converter()
        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(ESM_EMBED_DIM, alphafold2.dim) if ESM_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa=None, **kwargs):
        model, batch_converter, device = self.model, self.batch_converter, seq.device
        seq_embeds = get_esm_embedd(seq, model, batch_converter, device=device)
        seq_embeds = self.project_embed(seq_embeds)
        if msa is not None:
            flat_msa = rearrange(msa, 'b m n -> (b m) n')
            msa_embeds = get_esm_embedd(flat_msa, model, batch_converter, device=device)
            msa_embeds = rearrange(msa_embeds, '(b m) n d -> b m n d')
            msa_embeds = self.project_embed(msa_embeds)
        else:
            msa_embeds = None
        return self.alphafold2(seq, msa, seq_embed=seq_embeds, msa_embed=msa_embeds, **kwargs)


class Deterministic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleSelfAttnBlock(nn.Module):

    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)

    def forward(self, x, m, mask=None, msa_mask=None, seq_shape=None, msa_shape=None, seq_pos_emb=None, msa_pos_emb=None, _reverse=True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim=2)
        m1, m2 = torch.chunk(m, 2, dim=2)
        y1, y2, n1, n2 = None, None, None, None
        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse
        with context():
            y1 = x1 + self.f(x2, shape=seq_shape, record_rng=record_rng, mask=mask, rotary_emb=seq_pos_emb)
            y2 = x2 + self.g(y1, shape=seq_shape, record_rng=record_rng)
            n1 = m1 + self.j(m2, shape=msa_shape, record_rng=record_rng, mask=msa_mask, rotary_emb=msa_pos_emb)
            n2 = m2 + self.k(n1, record_rng=record_rng)
        return torch.cat((y1, y2), dim=2), torch.cat((n1, n2), dim=2)

    def backward_pass(self, y, n, dy, dn, mask=None, msa_mask=None, seq_shape=None, msa_shape=None, seq_pos_emb=None, msa_pos_emb=None, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, shape=seq_shape, set_rng=True)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, shape=seq_shape, set_rng=True, mask=mask, rotary_emb=seq_pos_emb)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        n1, n2 = torch.chunk(n, 2, dim=2)
        del n
        dn1, dn2 = torch.chunk(dn, 2, dim=2)
        del dn
        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng=True)
            torch.autograd.backward(gn1, dn2)
        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1
            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None
        with torch.enable_grad():
            m2.requires_grad = True
            fm2 = self.j(m2, shape=msa_shape, set_rng=True, mask=msa_mask, rotary_emb=msa_pos_emb)
            torch.autograd.backward(fm2, dm1, retain_graph=True)
        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2
            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None
            m = torch.cat([m1, m2.detach()], dim=2)
            dm = torch.cat([dm1, dm2], dim=2)
        return x, m, dx, dm


class ReversibleCrossAttnBlock(nn.Module):

    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)

    def forward(self, x, m, mask=None, msa_mask=None, seq_shape=None, msa_shape=None, seq_to_msa_pos_emb=None, msa_to_seq_pos_emb=None, _reverse=True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim=2)
        m1, m2 = torch.chunk(m, 2, dim=2)
        y1, y2, n1, n2 = None, None, None, None
        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse
        with context():
            y1 = x1 + self.f(x2, m2, record_rng=record_rng, mask=mask, context_mask=msa_mask, shape=seq_shape, context_shape=msa_shape, rotary_emb=seq_to_msa_pos_emb)
            y2 = x2 + self.k(y1, shape=seq_shape, record_rng=record_rng)
            n1 = m1 + self.j(m2, y2, record_rng=record_rng, mask=msa_mask, context_mask=mask, shape=msa_shape, context_shape=seq_shape, rotary_emb=msa_to_seq_pos_emb)
            n2 = m2 + self.g(n1, record_rng=record_rng)
        return torch.cat((y1, y2), dim=2), torch.cat((n1, n2), dim=2)

    def backward_pass(self, y, n, dy, dn, mask=None, msa_mask=None, seq_shape=None, msa_shape=None, seq_to_msa_pos_emb=None, msa_to_seq_pos_emb=None, **kwargs):
        n1, n2 = torch.chunk(n, 2, dim=2)
        del n
        dn1, dn2 = torch.chunk(dn, 2, dim=2)
        del dn
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.g(n1, set_rng=True)
            torch.autograd.backward(gn1, dn2)
        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1
            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None
        with torch.enable_grad():
            m2.requires_grad = True
            y2.requires_grad = True
            fm2 = self.j(m2, y2, set_rng=True, mask=msa_mask, context_mask=mask, shape=msa_shape, context_shape=seq_shape, rotary_emb=msa_to_seq_pos_emb)
            torch.autograd.backward(fm2, dm1)
        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2
            dm2 = dn2 + m2.grad
            dx2 = dy2 + y2.grad
            del dn2
            del dy2
            m2.grad = None
            y2.grad = None
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.k(y1, shape=seq_shape, set_rng=True)
            torch.autograd.backward(gy1, dx2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            m2.requires_grad = True
            fx2 = self.f(x2, m2, set_rng=True, mask=mask, context_mask=msa_mask, shape=seq_shape, context_shape=msa_shape, rotary_emb=seq_to_msa_pos_emb)
            torch.autograd.backward(fx2, dx1)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dx2 + x2.grad
            dm2 = dm2 + m2.grad
            x2.grad = None
            m2.grad = None
        with torch.no_grad():
            m = torch.cat([m1, m2.detach()], dim=2)
            dm = torch.cat([dm1, dm2], dim=2)
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, m, dx, dm


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = *pre_slices, slice(None, index)
    r = *pre_slices, slice(index, None)
    return t[l], t[r]


def irreversible_apply(inputs, ind, blocks, kwargs):
    x, m = split_at_index(1, ind, inputs)
    for block in blocks:
        x, m = block(x, m, _reverse=False, **kwargs)
    return torch.cat((x, m), dim=1)


class ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, inp, ind, blocks, kwargs):
        x, m = split_at_index(1, ind, inp)
        for block in blocks:
            x, m = block(x, m, _reverse=True, **kwargs)
        ctx.blocks = blocks
        ctx.kwargs = kwargs
        ctx.ind = ind
        ctx.save_for_backward(x.detach(), m.detach())
        return torch.cat((x, m), dim=1)

    @staticmethod
    def backward(ctx, d):
        ind = ctx.ind
        blocks = ctx.blocks
        kwargs = ctx.kwargs
        dy, dn = split_at_index(1, ind, d)
        y, n = ctx.saved_tensors
        for block in blocks[::-1]:
            y, n, dy, dn = block.backward_pass(y, n, dy, dn, **kwargs)
        d = torch.cat((dy, dn), dim=1)
        return d, None, None, None


reversible_apply = ReversibleFunction.apply


class ReversibleSequence(nn.Module):

    def __init__(self, input_blocks, block_types):
        super().__init__()
        self.block_types = block_types
        blocks = nn.ModuleList([])
        for block, block_type in zip(input_blocks, block_types):
            if block_type == 'self':
                reversible_klass = ReversibleSelfAttnBlock
            elif block_type == 'cross':
                reversible_klass = ReversibleCrossAttnBlock
            elif block_type == 'conv':
                reversible_klass = ReversibleSelfAttnBlock
            blocks.append(reversible_klass(*block))
        self.blocks = blocks

    def forward(self, seq, msa, seq_shape=None, msa_shape=None, mask=None, msa_mask=None, seq_pos_emb=None, msa_pos_emb=None, seq_to_msa_pos_emb=None, msa_to_seq_pos_emb=None, reverse=True):
        assert exists(msa), 'reversibility does not work with no MSA sequences yet'
        blocks = self.blocks
        seq, msa = list(map(lambda t: torch.cat((t, t), dim=-1), (seq, msa)))
        kwargs = {'mask': mask, 'msa_mask': msa_mask, 'seq_shape': seq_shape, 'msa_shape': msa_shape, 'seq_pos_emb': seq_pos_emb, 'msa_pos_emb': msa_pos_emb, 'seq_to_msa_pos_emb': seq_to_msa_pos_emb, 'msa_to_seq_pos_emb': msa_to_seq_pos_emb}
        fn = reversible_apply if reverse else irreversible_apply
        ind = seq.shape[1]
        inp = torch.cat((seq, msa), dim=1)
        out = fn(inp, ind, blocks, kwargs)
        seq, msa = split_at_index(1, ind, out)
        return list(map(lambda t: reduce(t, 'b n (c d) -> b n d', 'mean', c=2), (seq, msa)))


class DepthWiseConv1d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True, groups=None):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = nn.Sequential(nn.Conv1d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias), nn.Conv1d(dim_in, dim_out, 1, bias=bias))

    def forward(self, x):
        return self.net(x)


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        seq = torch.arange(n, device=device).type_as(self.inv_freq)
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        freqs = repeat(freqs, 'i j -> () i (j r)', r=2)
        return [freqs.sin(), freqs.cos()]


class AxialRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim // 2
        inv_freq = 1.0 / 10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        seq = torch.arange(n, device=device).type_as(self.inv_freq)
        x = einsum('n, d -> n d', seq, self.inv_freq)
        y = einsum('n, d -> n d', seq, self.inv_freq)
        x_sinu = repeat(x, 'i d -> i j d', j=n)
        y_sinu = repeat(y, 'j d -> i j d', i=n)
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)
        sin, cos = map(lambda t: repeat(t, 'i j d -> () (i j) (d r)', r=2), (sin, cos))
        return [sin, cos]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Always,
     lambda: ([], {'val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthWiseConv1d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Deterministic,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLM,
     lambda: ([], {'dim': 4, 'num_tokens': 4, 'mask_id': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (TriangleMultiplicativeModule,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_alphafold2(_paritybench_base):
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

