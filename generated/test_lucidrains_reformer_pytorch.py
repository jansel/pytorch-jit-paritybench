import sys
_module = sys.modules[__name__]
del sys
train = _module
train = _module
reformer_pytorch = _module
autopadder = _module
generative_tools = _module
recorder = _module
reformer_enc_dec = _module
reformer_pytorch = _module
reversible = _module
setup = _module

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


import random


import numpy as np


import torch


import torch.optim as optim


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import re


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import random_split


import logging


import math


from torch import nn


from functools import partial


from torch.nn.utils.rnn import pad_sequence


from collections import defaultdict


from torch.nn import Identity


from torch.autograd import Function


from functools import reduce


from functools import wraps


from itertools import chain


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)


class Autopadder(nn.Module):

    def __init__(self, net):
        super().__init__()
        assert isinstance(net, (LSHSelfAttention, Reformer, ReformerLM)
            ), 'only modules LSHSelfAttention, Reformer, ReformerLM accepted'
        self.net = net
        reformer = net.reformer if isinstance(net, ReformerLM) else net
        self.pad_dim = -1 if isinstance(net, ReformerLM) else -2
        self.bucket_size = reformer.bucket_size
        self.num_mem_kv = reformer.num_mem_kv
        self.full_attn_thres = reformer.full_attn_thres

    def forward(self, x, **kwargs):
        b, t, m, device = *x.shape[:2], self.num_mem_kv, x.device
        keys = kwargs.get('keys')
        input_mask = kwargs.get('input_mask')
        input_attn_mask = kwargs.get('input_attn_mask')
        k_len = 0 if keys is None else keys.shape[1]
        seqlen = t + m + k_len
        if seqlen > self.full_attn_thres:
            if input_mask is None:
                input_mask = torch.full_like(x, True, device=x.device,
                    dtype=torch.bool)
            x = pad_to_multiple(x, seqlen, self.bucket_size * 2, dim=self.
                pad_dim)
            if input_mask is not None:
                new_mask = F.pad(input_mask, (0, x.shape[1] - input_mask.
                    shape[1]), value=False)
                kwargs.update(input_mask=new_mask)
            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_mask = F.pad(input_attn_mask, (0, offset, 0, offset),
                    value=False)
                kwargs.update(input_attn_mask=new_mask)
        out = self.net(x, **kwargs)
        return out[:, 0:t]


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class TrainingWrapper(nn.Module):

    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        assert isinstance(net, ReformerLM
            ), 'generative trainer wrapper can only accept ReformerLM class'
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=
        1.0, filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[(None), :]
        b, t = start_tokens.shape
        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)
        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool,
                device=out.device)
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]
            logits = self.net(x, input_mask=input_mask, **kwargs)[:, (-1), :]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)
            if eos_token is not None and (sample == eos_token).all():
                break
        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        self.net.train(was_training)
        return out

    def forward(self, x, return_loss=False, **kwargs):
        pad = partial(pad_sequence, batch_first=True, padding_value=self.
            pad_value)
        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)
        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.
            ignore_index)
        return loss


class Recorder(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.iter = 0
        self.recordings = defaultdict(list)
        self.net = net
        self.on = True
        self.ejected = False

    def eject(self):
        self.ejected = True
        self.clear()
        self.unwire()
        return self.net

    def wire(self):
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = True
            if isinstance(module, LSHSelfAttention):
                module.callback = self.record

    def unwire(self):
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = False
            if isinstance(module, LSHSelfAttention):
                module.callback = None

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def clear(self):
        del self.recordings
        self.recordings = defaultdict(list)
        self.iter = 0

    def record(self, attn, buckets):
        if not self.on:
            return
        data = {'attn': attn.detach().cpu(), 'buckets': buckets.detach().cpu()}
        self.recordings[self.iter].append(data)

    def forward(self, x, **kwargs):
        assert not self.ejected, 'Recorder has already been ejected and disposed'
        if self.on:
            self.wire()
        out = self.net(x, **kwargs)
        self.iter += 1
        self.unwire()
        return out


ENC_PREFIX = 'enc_'


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x:
        string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]),
        tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


DEC_PREFIX = 'dec_'


def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX,
        kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX,
        kwargs)
    return enc_kwargs, dec_kwargs, kwargs


def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'input_mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['input_mask'])
    return enc_kwargs, dec_kwargs, kwargs


class ReformerEncDec(nn.Module):

    def __init__(self, dim, ignore_index=-100, pad_value=0, **kwargs):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'
        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['return_embeddings'] = True
        dec_kwargs['causal'] = True
        enc_kwargs.setdefault('bucket_size', 64)
        dec_kwargs.setdefault('bucket_size', enc_kwargs['bucket_size'] * 2)
        enc = ReformerLM(**enc_kwargs)
        dec = ReformerLM(**dec_kwargs)
        self.enc = TrainingWrapper(enc, ignore_index=ignore_index,
            pad_value=pad_value)
        self.dec = TrainingWrapper(dec, ignore_index=ignore_index,
            pad_value=pad_value)

    def generate(self, seq_in, seq_out_start, seq_len, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec.generate(seq_out_start, seq_len, keys=enc_keys, **{
            **dec_kwargs, **kwargs})

    def forward(self, seq_in, seq_out, return_loss=False, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec(seq_out, return_loss=return_loss, keys=enc_keys, **
            dec_kwargs)


class MatrixMultiply(nn.Module):

    def __init__(self, tensor, transpose=False, normalize=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x):
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor


class ReZero(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class PreNorm(nn.Module):

    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Chunk(nn.Module):

    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def default(val, default_val):
    return default_val if val is None else val


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):

    def inner_fn(fn):

        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache
            =True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'
            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, (None)].expand(-1, -1, last_dim))


TOKEN_SELF_ATTN_VALUE = -50000.0


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


class LSHAttention(nn.Module):

    def __init__(self, dropout=0.0, bucket_size=64, n_hashes=8, causal=
        False, allow_duplicate_attention=True, attend_across_buckets=True,
        rehash_each_round=True, drop_for_hash_rate=0.0,
        random_rotations_per_head=False, return_attn=False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)
        assert rehash_each_round or allow_duplicate_attention, 'The setting {allow_duplicate_attention=False, rehash_each_round=False} is not implemented.'
        self.causal = causal
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head
        self._return_attn = return_attn
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device
        assert n_buckets % 2 == 0
        rot_size = n_buckets
        rotations_shape = (batch_size if self._random_rotations_per_head else
            1, vecs.shape[-1], self.n_hashes if self._rehash_each_round else
            1, rot_size // 2)
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype,
            device=device).expand(batch_size, -1, -1, -1)
        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs,
            random_rotations)
        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)
            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]
            h, *_ = buckets.shape
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))
        return buckets

    def forward(self, qk, v, query_len=None, input_mask=None,
        input_attn_mask=None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)
        assert seqlen % (self.bucket_size * 2
            ) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'
        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth,
            fetch=is_reverse, set_cache=self.training)
        assert int(buckets.shape[1]) == self.n_hashes * seqlen
        total_hashes = self.n_hashes
        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0
            ).expand_as(buckets)
        buckets_and_t = seqlen * buckets + ticker % seqlen
        buckets_and_t = buckets_and_t.detach()
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker
        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()
        st = sticker % seqlen
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type(bq.type())

        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, (...)], x[:, :-1, (...)]], dim=1)
            return torch.cat([x, x_extra], dim=2)
        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * dim ** -0.5
        masked_value = max_neg_value(dots)
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen -
                input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.
                shape[-2]), value=True)
            dot_attn_indices = (bq_t * seqlen)[:, :, :, (None)] + bkv_t[:,
                :, (None), :]
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]
                ), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, (None)] * mkv[:, :, (None), :]
            dots.masked_fill_(~mask, masked_value)
            del mask
        if self.causal:
            mask = bq_t[:, :, :, (None)] < bkv_t[:, :, (None), :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, (None), :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask
        self_mask = bq_t[:, :, :, (None)] == bkv_t[:, :, (None), :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t //
                seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, (None)] != bkv_buckets[:, :,
                (None), :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([torch.reshape(locs1, (batch_size,
                total_hashes, seqlen)), torch.reshape(locs2, (batch_size,
                total_hashes, seqlen))], 1).permute((0, 2, 1))
            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 *
                total_hashes))
            b_locs1 = b_locs[:, :, :, (None), :total_hashes]
            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)
            dup_counts = bq_locs[:, :, :, (None), :] == bkv_locs[:, :, (
                None), :, :]
            dup_counts = chunked_sum(dup_counts, chunks=total_hashes *
                batch_size)
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-09)
            del dup_counts
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())
        dropped_dots = self.dropout(dots)
        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1))


        class UnsortLogits(Function):

            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad
        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))
        if query_len != seqlen:
            query_slice = slice(None), slice(None), slice(0, query_len)
            o, logits = o[query_slice], logits[query_slice]
        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True)
            )
        out = torch.sum(o * probs, dim=1)
        attn = torch.empty(0, device=device)
        if self._return_attn:
            attn_unsort = (bq_t * seqlen)[:, :, :, (None)] + bkv_t[:, :, (
                None), :]
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long(
                )
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen *
                seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(
                attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes,
                seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1
                )
        return out, attn, buckets


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:ind + t, (...)] for ind in range(forward +
        backward + 1)]
    return torch.cat(tensors, dim=dim)


class LocalAttention(nn.Module):

    def __init__(self, bucket_size, causal=False, look_backward=1,
        look_forward=0, dropout=0.0, shared_qk=False):
        super().__init__()
        assert not (causal and look_forward > 0
            ), 'you cannot look forward if causal'
        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.shared_qk = shared_qk
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask=None):
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        bucket_size, causal, look_backward, look_forward, shared_qk = (self
            .bucket_size, self.causal, self.look_backward, self.
            look_forward, self.shared_qk)
        buckets = t // bucket_size
        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type(q.type())
        ticker = torch.arange(t, device=device, dtype=dtype)[(None), :]
        b_t = ticker.reshape(1, buckets, bucket_size)
        bucket_fn = lambda t: t.reshape(b, buckets, bucket_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))
        look_around_kwargs = {'backward': look_backward, 'forward':
            look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * e ** -0.5
        mask_value = max_neg_value(dots)
        if shared_qk:
            mask = bq_t[:, :, :, (None)] == bq_k[:, :, (None), :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask
        if causal:
            mask = bq_t[:, :, :, (None)] < bq_k[:, :, (None), :]
            dots.masked_fill_(mask, mask_value)
            del mask
        mask = bq_k[:, :, (None), :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask
        if input_mask is not None:
            h = b // input_mask.shape[0]
            input_mask = input_mask.reshape(-1, buckets, bucket_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = mq[:, (None), :, :, (None)] * mk[:, (None), :, (None), :]
            mask = merge_dims(0, 1, mask.expand(-1, h, -1, -1, -1))
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(b, t, e)
        return out


class FullQKAttention(nn.Module):

    def __init__(self, causal=False, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len=None, input_mask=None,
        input_attn_mask=None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len
        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type(q.type())
        dot = torch.einsum('bie,bje->bij', q, qk) * dim ** -0.5
        i = torch.arange(t)
        dot[:, (i), (i)] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, (None)] * input_mask[:, (None), :
                ]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len -
                input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)
        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, (i), (j)] = masked_value
        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)
        out = torch.einsum('bij,bje->bie', dot, v)
        return out, dot, torch.empty(0)


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def process_inputs_chunk(fn, chunks=1, dim=0):

    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), 
            list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]
            ))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = *pre_slices, slice(None, index)
    r = *pre_slices, slice(index, None)
    return t[l], t[r]


class LSHSelfAttention(nn.Module):

    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, causal=
        False, attn_chunks=1, random_rotations_per_head=False,
        attend_across_buckets=True, allow_duplicate_attention=True,
        num_mem_kv=0, one_value_head=False, use_full_attn=False,
        full_attn_thres=None, return_attn=False, post_attn_dropout=0.0,
        dropout=0.0, n_local_attn_heads=0, **kwargs):
        super().__init__()
        assert dim % heads == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'
        self.dim = dim
        self.heads = heads
        self.attn_chunks = default(attn_chunks, 1)
        self.v_head_repeats = heads if one_value_head else 1
        v_dim = dim // self.v_head_repeats
        self.toqk = nn.Linear(dim, dim, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=
            n_hashes, causal=causal, random_rotations_per_head=
            random_rotations_per_head, attend_across_buckets=
            attend_across_buckets, allow_duplicate_attention=
            allow_duplicate_attention, return_attn=return_attn, dropout=
            dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)
        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)
        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim,
            requires_grad=True)) if num_mem_kv > 0 else None
        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(bucket_size=bucket_size * 2,
            causal=causal, dropout=dropout, shared_qk=True, look_forward=1 if
            not causal else 0)
        self.callback = None

    def forward(self, x, keys=None, input_mask=None, input_attn_mask=None,
        context_mask=None, **kwargs):
        device, dtype = x.device, x.dtype
        b, t, e, h, m, l_h = (*x.shape, self.heads, self.num_mem_kv, self.
            n_local_attn_heads)
        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype,
            device=device))
        mem = mem_kv.expand(b, m, e)
        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]
        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres
        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()
        merge_batch_and_heads = partial(merge_dims, 0, 1)
        qk, v = map(merge_heads, (qk, v))
        has_local = l_h > 0
        lsh_h = h - l_h
        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))
        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask
        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h,
                input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask
        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=
            self.attn_chunks)
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)
        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))
        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)
        out = split_heads(out).view(b, t, e)
        out = self.to_out(out)
        return self.post_attn_dropout(out)


class GELU_(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type()
            )
        sinusoid_inp = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[(None), :, :]


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class Reformer(nn.Module):

    def __init__(self, dim, depth, max_seq_len, heads=8, bucket_size=64,
        n_hashes=8, ff_chunks=100, attn_chunks=None, causal=False,
        weight_tie=False, lsh_dropout=0.0, ff_dropout=0.0, ff_activation=
        None, ff_mult=4, ff_glu=False, post_attn_dropout=0.0, layer_dropout
        =0.0, lsh_attend_across_buckets=True, lsh_allow_duplicate_attention
        =True, random_rotations_per_head=False, twin_attention=False,
        use_scale_norm=False, use_rezero=False, use_full_attn=False,
        full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=
        False, n_local_attn_heads=0, pkm_layers=tuple(), pkm_num_keys=128):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv
        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres
        get_attn = lambda : LSHSelfAttention(dim, heads, bucket_size,
            n_hashes, causal=causal, dropout=lsh_dropout, post_attn_dropout
            =post_attn_dropout, attn_chunks=attn_chunks,
            allow_duplicate_attention=lsh_allow_duplicate_attention,
            attend_across_buckets=lsh_attend_across_buckets,
            random_rotations_per_head=random_rotations_per_head, num_mem_kv
            =num_mem_kv, use_full_attn=use_full_attn, full_attn_thres=
            full_attn_thres, one_value_head=one_value_head,
            n_local_attn_heads=n_local_attn_heads)
        get_ff = lambda : Chunk(ff_chunks, FeedForward(dim, dropout=
            ff_dropout, activation=ff_activation, mult=ff_mult, glu=ff_glu),
            along_dim=-2)
        get_pkm = lambda : PKM(dim, num_keys=pkm_num_keys)
        if weight_tie:
            get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff,
                get_pkm))
        blocks = []
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm
        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm,
            norm_type, dim)
        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)
            parallel_net = None
            attn = get_attn()
            if use_pkm:
                parallel_net = get_pkm()
            elif twin_attention:
                parallel_net = get_attn()
            else:
                parallel_net = get_ff()
            f = residual_fn_wrapper(attn)
            g = residual_fn_wrapper(parallel_net)
            blocks.append(nn.ModuleList([f, g]))
        self.layers = ReversibleSequence(nn.ModuleList(blocks),
            layer_dropout=layer_dropout, reverse_thres=reverse_thres,
            send_signal=True)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        arg_route = True, self.twin_attention
        x = self.layers(x, arg_route=arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class ReformerLM(nn.Module):

    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8,
        bucket_size=64, n_hashes=4, ff_chunks=100, attn_chunks=1, causal=
        False, weight_tie=False, lsh_dropout=0.0, ff_dropout=0.0, ff_mult=4,
        ff_activation=None, ff_glu=False, post_attn_dropout=0.0,
        layer_dropout=0.0, random_rotations_per_head=False, twin_attention=
        False, use_scale_norm=False, use_rezero=False, use_full_attn=False,
        full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=
        False, emb_dim=None, return_embeddings=False, weight_tie_embedding=
        False, fixed_position_emb=False, absolute_position_emb=False,
        axial_position_shape=None, n_local_attn_heads=0, pkm_layers=tuple(),
        pkm_num_keys=128):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim
            , dim)
        if absolute_position_emb:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:
            axial_position_shape = default(axial_position_shape, (
                max_seq_len // bucket_size, bucket_size))
            self.pos_emb = AxialPositionalEmbedding(emb_dim,
                axial_position_shape)
        self.reformer = Reformer(dim, depth, max_seq_len, heads=heads,
            bucket_size=bucket_size, n_hashes=n_hashes, ff_chunks=ff_chunks,
            attn_chunks=attn_chunks, causal=causal, weight_tie=weight_tie,
            lsh_dropout=lsh_dropout, ff_mult=ff_mult, ff_activation=
            ff_activation, ff_glu=ff_glu, ff_dropout=ff_dropout,
            post_attn_dropout=0.0, layer_dropout=layer_dropout,
            random_rotations_per_head=random_rotations_per_head,
            twin_attention=twin_attention, use_scale_norm=use_scale_norm,
            use_rezero=use_rezero, use_full_attn=use_full_attn,
            full_attn_thres=full_attn_thres, reverse_thres=reverse_thres,
            num_mem_kv=num_mem_kv, one_value_head=one_value_head,
            n_local_attn_heads=n_local_attn_heads, pkm_layers=pkm_layers,
            pkm_num_keys=pkm_num_keys)
        if return_embeddings:
            self.out = Identity()
            return
        self.out = nn.Sequential(nn.Linear(dim, emb_dim) if emb_dim != dim else
            Identity(), nn.Linear(emb_dim, num_tokens) if not
            weight_tie_embedding else MatrixMultiply(self.token_emb.weight,
            transpose=True, normalize=True))

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())
        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)
        return self.out(x)


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


class ReversibleBlock(nn.Module):

    def __init__(self, f, g, depth=None, send_signal=False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None
        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = False
            f_args['_depth'] = g_args['_depth'] = self.depth
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = True
            f_args['_depth'] = g_args['_depth'] = self.depth
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


class IrreversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)


class _ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):

    def __init__(self, blocks, layer_dropout=0.0, reverse_thres=0,
        send_signal=False):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres
        self.blocks = nn.ModuleList([ReversibleBlock(f, g, depth,
            send_signal) for depth, (f, g) in enumerate(blocks)])
        self.irrev_blocks = nn.ModuleList([IrreversibleBlock(f=f, g=g) for 
            f, g in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks
        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1
                ) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if
                not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x
        return _ReversibleFunction.apply(x, blocks, block_kwargs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lucidrains_reformer_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(AbsolutePositionalEmbedding(*[], **{'dim': 4, 'max_seq_len': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(FeedForward(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FixedPositionalEmbedding(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GELU_(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(ScaleNorm(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

