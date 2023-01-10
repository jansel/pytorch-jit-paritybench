import sys
_module = sys.modules[__name__]
del sys
retro_pytorch = _module
data = _module
optimizer = _module
retrieval = _module
retro_pytorch = _module
training = _module
utils = _module
setup = _module

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


from functools import partial


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.optim import AdamW


from math import ceil


import torch.nn.functional as F


import logging


from torch import nn


from torch import einsum


from torch.utils.data import DataLoader


def exists(val):
    return val is not None


class RMSNorm(nn.Module):

    def __init__(self, dim, *, eps=1e-08, gated=False):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim)) if gated else None

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1) * self.scale
        out = x / norm.clamp(min=self.eps) * self.gamma
        if not exists(self.weight):
            return out
        return out * (x * self.weight).sigmoid()


class PreNorm(nn.Module):

    def __init__(self, dim, fn, norm_klass=RMSNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm_klass(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs) + x


class PostNorm(nn.Module):

    def __init__(self, dim, fn, scale_residual=1, norm_klass=RMSNorm):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = norm_klass(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset=0):
        seq = torch.arange(max_seq_len, device=device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> 1 1 n d')


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(mult * dim)
        self.ff = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.ff(x)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = t * freqs.cos() + rotate_half(t) * freqs.sin()
    return torch.cat((t, t_pass), dim=-1)


def cast_tuple(val, num=1):
    return val if isinstance(val, tuple) else (val,) * num


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):

    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, causal=False, dropout=0.0, null_kv=False):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.null_k = nn.Parameter(torch.randn(inner_dim)) if null_kv else None
        self.null_v = nn.Parameter(torch.randn(inner_dim)) if null_kv else None

    def forward(self, x, mask=None, context=None, pos_emb=None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale
        kv_input = default(context, x)
        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * scale
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num=2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)
        if exists(self.null_k):
            nk, nv = self.null_k, self.null_v
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b=b, h=h), (nk, nv))
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            if exists(self.null_k):
                mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ChunkedCrossAttention(nn.Module):

    def __init__(self, chunk_size, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.cross_attn = Attention(null_kv=True, **kwargs)

    def forward(self, x, *, context_mask=None, context, pos_emb=None):
        chunk_size = self.chunk_size
        b, n, num_chunks, num_retrieved = x.shape[0], x.shape[-2], *context.shape[-4:-2]
        if n < self.chunk_size:
            return torch.zeros_like(x)
        causal_padding = chunk_size - 1
        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value=0.0)
        seq_index = n // chunk_size * chunk_size
        x, x_remainder = x[:, :seq_index], x[:, seq_index:]
        seq_remain_len = x_remainder.shape[-2]
        q_pos_emb, k_pos_emb = pos_emb
        q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value=0.0)
        k_pos_emb = repeat(k_pos_emb, 'b h n d -> b h (r n) d', r=num_retrieved)
        pos_emb = q_pos_emb, k_pos_emb
        x = rearrange(x, 'b (k n) d -> (b k) n d', k=num_chunks)
        context = rearrange(context, 'b k r n d -> (b k) (r n) d')
        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b k r n -> (b k) (r n)')
        out = self.cross_attn(x, context=context, mask=context_mask, pos_emb=pos_emb)
        out = rearrange(out, '(b k) n d -> b (k n) d', b=b)
        out = F.pad(out, (0, 0, causal_padding, -causal_padding + seq_remain_len), value=0.0)
        return out


MIN_DIM_HEAD = 32


class Encoder(nn.Module):

    def __init__(self, dim, *, depth, context_dim=None, causal=False, heads=8, dim_head=64, attn_dropout=0.0, ff_mult=4, ff_dropout=0.0, final_norm=True, cross_attn_layers=None, post_norm=False, output_dim=None, norm_klass=RMSNorm, scale_residual=1.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        rotary_emb_dim = min(dim_head, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        wrapper = partial(PreNorm, dim, norm_klass=norm_klass) if not post_norm else partial(PostNorm, dim, scale_residual=scale_residual, norm_klass=norm_klass)
        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers
            self.layers.append(nn.ModuleList([wrapper(Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, causal=causal)), wrapper(Attention(dim=dim, context_dim=context_dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)) if has_cross_attn else None, wrapper(FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))]))
        self.norm_out = norm_klass(dim) if final_norm and not post_norm else nn.Identity()
        self.project_out = nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()

    def forward(self, x, *, mask=None, chunked_seq):
        device, chunk_size, seq_len = x.device, x.shape[-2], chunked_seq.shape[-2]
        q_pos_emb = self.rotary_pos_emb(chunk_size, device=device)
        k_pos_emb = self.rotary_pos_emb(seq_len, device=device)
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask=mask, pos_emb=q_pos_emb)
            if exists(cross_attn):
                x = cross_attn(x, context=chunked_seq, pos_emb=(q_pos_emb, k_pos_emb))
            x = ff(x)
        x = self.norm_out(x)
        return self.project_out(x)


class Decoder(nn.Module):

    def __init__(self, dim, *, depth, heads=8, dim_head=64, attn_dropout=0.0, ff_mult=4, ff_dropout=0.0, final_norm=True, cross_attn_layers=None, chunk_size=64, post_norm=False, norm_klass=RMSNorm, scale_residual=1.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        rotary_emb_dim = min(dim_head, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        wrapper = partial(PreNorm, dim, norm_klass=norm_klass) if not post_norm else partial(PostNorm, dim, scale_residual=scale_residual, norm_klass=norm_klass)
        self.chunk_size = chunk_size
        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers
            self.layers.append(nn.ModuleList([wrapper(Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, causal=True)), wrapper(ChunkedCrossAttention(chunk_size=chunk_size, dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)) if has_cross_attn else None, wrapper(FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))]))
        self.norm_out = norm_klass(dim) if final_norm and not post_norm else nn.Identity()

    def forward(self, x, *, encoder=None, encoder_retrieved_mask=None, context_mask=None, retrieved=None):
        device, seq_len = x.device, x.shape[-2]
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device=device)
        num_seq_chunks = seq_len // self.chunk_size
        seq_index = num_seq_chunks * self.chunk_size
        if exists(retrieved):
            num_chunks, num_neighbors, chunk_size = retrieved.shape[-4:-1]
            cross_attn_q_pos_emb = self.rotary_pos_emb(self.chunk_size, device=device, offset=self.chunk_size - 1)
            cross_attn_k_pos_emb = self.rotary_pos_emb(chunk_size, device=device)
            cross_attn_pos_emb = cross_attn_q_pos_emb, cross_attn_k_pos_emb
        retrieved_encoded = False
        for attn, cross_attn, ff in self.layers:
            x = attn(x, pos_emb=self_attn_pos_emb)
            if exists(cross_attn) and exists(retrieved):
                if not retrieved_encoded:
                    retrieved = rearrange(retrieved, 'b k r n d -> (b k r) n d')
                    seq_as_context = repeat(x[:, :seq_index], 'b (k n) d -> (b k r) n d', n=self.chunk_size, r=num_neighbors)
                    retrieved = encoder(retrieved, mask=encoder_retrieved_mask, chunked_seq=seq_as_context)
                    retrieved = rearrange(retrieved, '(b k r) n d -> b k r n d', k=num_chunks, r=num_neighbors)
                    retrieved_encoded = True
                x = cross_attn(x, context=retrieved, context_mask=context_mask, pos_emb=cross_attn_pos_emb)
            x = ff(x)
        return self.norm_out(x)


BERT_VOCAB_SIZE = 28996


def deepnorm_init(transformer, beta, module_name_match_list=['.ff.', '.to_v', '.to_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue
        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain=gain)
        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


class RETRO(nn.Module):

    def __init__(self, *, num_tokens=BERT_VOCAB_SIZE, max_seq_len=2048, enc_dim=896, enc_depth=2, enc_cross_attn_layers=None, dec_depth=12, dec_cross_attn_layers=(1, 3, 6, 9), heads=8, dec_dim=768, dim_head=64, enc_attn_dropout=0.0, enc_ff_dropout=0.0, dec_attn_dropout=0.0, dec_ff_dropout=0.0, chunk_size=64, pad_id=0, enc_scale_residual=None, dec_scale_residual=None, norm_klass=None, gated_rmsnorm=False, use_deepnet=False):
        super().__init__()
        assert dim_head >= MIN_DIM_HEAD, f'dimension per head must be greater than {MIN_DIM_HEAD}'
        self.seq_len = max_seq_len
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(num_tokens, enc_dim)
        self.pos_emb = nn.Embedding(max_seq_len, enc_dim)
        self.chunk_size = chunk_size
        self.to_decoder_model_dim = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        norm_klass = default(norm_klass, RMSNorm)
        if use_deepnet:
            enc_scale_residual = default(enc_scale_residual, 0.81 * (enc_depth ** 4 * dec_depth) ** 0.0625)
            dec_scale_residual = default(dec_scale_residual, (3 * dec_depth) ** 0.25)
            norm_klass = nn.LayerNorm
        if gated_rmsnorm:
            norm_klass = partial(RMSNorm, gated=True)
        self.encoder = Encoder(dim=enc_dim, context_dim=dec_dim, depth=enc_depth, attn_dropout=enc_attn_dropout, ff_dropout=enc_ff_dropout, cross_attn_layers=enc_cross_attn_layers, post_norm=use_deepnet, norm_klass=norm_klass, scale_residual=enc_scale_residual, output_dim=dec_dim)
        self.decoder = Decoder(dim=dec_dim, depth=dec_depth, attn_dropout=dec_attn_dropout, ff_dropout=dec_ff_dropout, cross_attn_layers=dec_cross_attn_layers, chunk_size=chunk_size, post_norm=use_deepnet, norm_klass=norm_klass, scale_residual=dec_scale_residual)
        self.to_logits = nn.Linear(dec_dim, num_tokens)
        if use_deepnet:
            deepnorm_init(self.encoder, 0.87 * (enc_depth ** 4 * dec_depth) ** -0.0625)
            deepnorm_init(self.decoder, (12 * dec_depth) ** -0.25)

    def forward_without_retrieval(self, seq):
        embed = self.token_emb(seq)
        embed = embed[:, :self.seq_len]
        pos_emb = self.pos_emb(torch.arange(embed.shape[1], device=embed.device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb
        embed = self.to_decoder_model_dim(embed)
        embed = self.decoder(embed)
        return self.to_logits(embed)

    def forward(self, seq, retrieved=None, return_loss=False):
        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """
        if not exists(retrieved):
            return self.forward_without_retrieval(seq)
        assert not (return_loss and not self.training), 'must be training if returning loss'
        mask = retrieved != self.pad_id
        if retrieved.ndim == 3:
            retrieved = rearrange(retrieved, 'b k n -> b k 1 n')
        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]
        n, num_chunks, num_neighbors, chunk_size, retrieved_shape, device = seq.shape[-1], *retrieved.shape[-3:], retrieved.shape, seq.device
        assert chunk_size >= self.chunk_size, 'chunk size of retrieval input must be greater or equal to the designated chunk_size on RETRO initialization'
        num_seq_chunks = n // self.chunk_size
        assert num_chunks == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {num_chunks} passed in'
        seq_index = num_seq_chunks * self.chunk_size
        embed = self.token_emb(seq)
        retrieved = self.token_emb(retrieved)
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb
        encoder_retrieved_mask = decoder_retrieved_mask = None
        if exists(mask):
            assert mask.shape == retrieved_shape, 'retrieval mask must be of the same shape as the retrieval tokens'
            encoder_retrieved_mask = rearrange(mask, 'b k r n -> (b k r) n')
            decoder_retrieved_mask = mask
        embed = self.to_decoder_model_dim(embed)
        embed = self.decoder(embed, encoder=self.encoder, context_mask=decoder_retrieved_mask, encoder_retrieved_mask=encoder_retrieved_mask, retrieved=retrieved)
        logits = self.to_logits(embed)
        if not return_loss:
            return logits
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=self.pad_id)
        return loss


EOS_ID = 102


def knn_to_retrieved_chunks(knns, chunks_memmap, *, add_continuations, num_chunks, pad_id=0, eos_id=EOS_ID):
    no_neighbor_mask = knns == -1
    knns = np.maximum(knns, 0)
    knn_chunks = chunks_memmap[knns]
    is_last_document_chunk = np.any(knn_chunks == eos_id, axis=-1, keepdims=True)
    retrieved = knn_chunks[..., :-1]
    if add_continuations:
        continuation_indices = np.clip(knns + 1, 0, num_chunks - 1)
        continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
        continuation_chunks *= ~is_last_document_chunk
        retrieved = np.concatenate((retrieved, continuation_chunks), axis=-1)
    retrieved = np.where(~no_neighbor_mask[..., None], retrieved, pad_id)
    return retrieved


class RETRODataset(Dataset):

    def __init__(self, *, num_chunks, chunk_size, seq_len, num_sequences, num_neighbors, chunk_memmap_path, chunk_nn_memmap_path, seq_memmap_path, eos_id=EOS_ID, pad_id=0.0, add_continuations=True):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        num_chunks_with_padding = num_chunks + self.seq_num_chunks
        chunks_shape = num_chunks_with_padding, chunk_size + 1
        knn_shape = num_chunks_with_padding, num_neighbors
        self.add_continuations = add_continuations
        self.get_chunks = partial(memmap, chunk_memmap_path, dtype=np.int32, shape=chunks_shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype=np.int32, shape=knn_shape)
        self.get_seqs = partial(memmap, seq_memmap_path, dtype=np.int32, shape=(num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        with self.get_chunks() as chunks_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:
            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, begin_chunk_index + self.seq_num_chunks)
            chunks = chunks_memmap[chunk_range]
            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))
            seq_mask = np.cumsum(seq_tokens == self.eos_id, axis=0)
            seq_mask = np.pad(seq_mask, (1, 0))[:-1] == 0.0
            seq_tokens = np.where(seq_mask, seq_tokens, 0.0)
            knns = knns_memmap[chunk_range]
            retrieved = knn_to_retrieved_chunks(knns, chunks_memmap, add_continuations=self.add_continuations, eos_id=self.eos_id, num_chunks=self.num_chunks)
        seq_tokens_torch = torch.from_numpy(seq_tokens).long()
        retrieved_torch = torch.from_numpy(retrieved).long()
        return seq_tokens_torch, retrieved_torch


SOS_ID = 101


BERT_MODEL_DIM = 768


EMBEDDING_TMP_SUBFOLDER = 'embeddings'


def get_bert():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL
    return MODEL


@torch.no_grad()
def bert_embed(token_ids, return_cls_repr=False, eps=1e-08, pad_id=0.0):
    model = get_bert()
    mask = token_ids != pad_id
    if torch.cuda.is_available():
        token_ids = token_ids
        mask = mask
    outputs = model(input_ids=token_ids, attention_mask=mask, output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1]
    if return_cls_repr:
        return hidden_state[:, 0]
    if not exists(mask):
        return hidden_state.mean(dim=1)
    mask = mask[:, 1:]
    mask = rearrange(mask, 'b n -> b n 1')
    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean


def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


def chunks_to_embeddings_(*, num_chunks, chunks_memmap_path, embeddings_memmap_path, chunk_size=64, embed_dim=BERT_MODEL_DIM, batch_size=16, use_cls_repr=False, pad_id=0.0):
    chunks_shape = num_chunks, chunk_size + 1
    embed_shape = num_chunks, embed_dim
    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32) as chunks, memmap(embeddings_memmap_path, shape=embed_shape, dtype=np.float32, mode='w+') as embeddings:
        for dim_slice in range_chunked(num_chunks, batch_size=batch_size):
            batch_chunk_npy = chunks[dim_slice]
            batch_chunk = torch.from_numpy(batch_chunk_npy)
            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim=1)
            batch_chunk = batch_chunk[:, :-1]
            batch_embed = bert_embed(batch_chunk, return_cls_repr=use_cls_repr)
            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            None


def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)


def index_embeddings(embeddings_folder, *, index_file='knn.index', index_infos_file='index_infos.json', max_index_memory_usage='100m', current_memory_available='1G'):
    embeddings_path = TMP_PATH / embeddings_folder
    index_path = INDEX_FOLDER_PATH / index_file
    reset_folder_(INDEX_FOLDER_PATH)
    build_index(embeddings=str(embeddings_path), index_path=str(index_path), index_infos_path=str(INDEX_FOLDER_PATH / index_infos_file), metric_type='l2', max_index_memory_usage=max_index_memory_usage, current_memory_available=current_memory_available, make_direct_map=True, should_be_memory_mappable=False, use_gpu=torch.cuda.is_available())
    index = faiss_read_index(index_path)
    return index


def memmap_file_to_chunks_(memmap_path, *, folder, shape, dtype, max_rows_per_file=500):
    rows, _ = shape
    with memmap(memmap_path, shape=shape, dtype=dtype, mode='r') as f:
        root_path = TMP_PATH / folder
        reset_folder_(root_path)
        for ind, dim_slice in enumerate(range_chunked(rows, batch_size=max_rows_per_file)):
            filename = root_path / f'{ind:05d}.npy'
            data_slice = f[dim_slice]
            np.save(str(filename), f[dim_slice])
            None


def chunks_to_index_and_embed(*, num_chunks, chunk_size, chunk_memmap_path, use_cls_repr=False, max_rows_per_file=500, chunks_to_embeddings_batch_size=16, embed_dim=BERT_MODEL_DIM, index_file='knn.index', **index_kwargs):
    embedding_path = f'{chunk_memmap_path}.embedded'
    embed_shape = num_chunks, embed_dim
    chunks_to_embeddings_(num_chunks=num_chunks, chunk_size=chunk_size, chunks_memmap_path=chunk_memmap_path, embeddings_memmap_path=embedding_path, use_cls_repr=use_cls_repr, batch_size=chunks_to_embeddings_batch_size, embed_dim=embed_dim)
    memmap_file_to_chunks_(embedding_path, shape=embed_shape, dtype=np.float32, folder=EMBEDDING_TMP_SUBFOLDER, max_rows_per_file=max_rows_per_file)
    index = index_embeddings(embeddings_folder=EMBEDDING_TMP_SUBFOLDER, index_file=index_file, **index_kwargs)
    embeddings = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode='r')
    return index, embeddings


def chunks_to_precalculated_knn_(*, num_nearest_neighbors, num_chunks, chunk_size, chunk_memmap_path, doc_ids_memmap_path, use_cls_repr=False, max_rows_per_file=500, chunks_to_embeddings_batch_size=16, embed_dim=BERT_MODEL_DIM, num_extra_neighbors=10, force_reprocess=False, index_file='knn.index', **index_kwargs):
    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f'{chunk_path.stem}.knn{chunk_path.suffix}'
    index_path = INDEX_FOLDER_PATH / index_file
    if index_path.exists() and knn_path.exists() and not force_reprocess:
        None
        index = faiss_read_index(index_path)
        return knn_path, index
    index, embeddings = chunks_to_index_and_embed(num_chunks=num_chunks, chunk_size=chunk_size, chunk_memmap_path=chunk_memmap_path, index_file=index_file, **index_kwargs)
    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1
    with memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode='w+') as knns, memmap(doc_ids_memmap_path, shape=(num_chunks,), dtype=np.int32, mode='r') as doc_ids:
        for dim_slice in range_chunked(num_chunks, batch_size=max_rows_per_file):
            query_vector = embeddings[dim_slice]
            distances, indices = index.search(query_vector, k=total_neighbors_to_fetch)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids
            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1000.0, distances)
            indices = np.take_along_axis(indices, np.argsort(distances, axis=1), axis=1)
            knns[dim_slice] = indices[:, :num_nearest_neighbors]
            None
    None
    return knn_path, index


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / temperature + gumbel_noise(t)).argmax(dim=dim)


def is_true_env_flag(env_flag):
    return os.getenv(env_flag, 'false').lower() in ('true', '1', 't')


def knn_chunks_from_seq_chunks(seq_chunks, *, knn, faiss_index, num_chunks, chunk_size, chunks_memmap_path):
    b, device = seq_chunks.shape[0], seq_chunks.device
    ones = torch.ones((b, 1), dtype=torch.bool, device=device)
    sos = ones * SOS_ID
    eos = ones * EOS_ID
    seq_chunks = torch.cat((sos, seq_chunks, eos), dim=1)
    embeds = bert_embed(seq_chunks.cpu())
    _, knn_indices = faiss_index.search(embeds.cpu().numpy(), k=knn)
    with memmap(chunks_memmap_path, dtype=np.int32, shape=(num_chunks + 1, chunk_size + 1)) as chunk_memmap:
        knn_chunks = knn_to_retrieved_chunks(knn_indices, chunk_memmap, add_continuations=True, num_chunks=num_chunks)
        knn_chunks_torch = torch.from_numpy(knn_chunks)
    return knn_chunks_torch


def safe_cat(accum, t, dim=-1):
    if not exists(accum):
        return t
    return torch.cat((accum, t), dim=dim)


def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER


def tokenize(texts, add_special_tokens=True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    tokenizer = get_tokenizer()
    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens=add_special_tokens, padding=True, return_tensors='pt')
    token_ids = encoding.input_ids
    return token_ids


def doc_text_to_chunks_and_seq_indices(*, doc_text, chunk_size=64, seq_len=2048, pad_id=0):
    assert seq_len % chunk_size == 0, 'sequence length must be divisible by chunk size'
    ids = tokenize(doc_text)
    ids = rearrange(ids, '1 ... -> ...')
    text_len = ids.shape[-1]
    padding = chunk_size - (text_len - 1) % chunk_size
    ids = F.pad(ids, (0, padding))
    ids, last_token = ids[:-1], ids[-1:]
    ids = rearrange(ids, '(n c) -> n c', c=chunk_size)
    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim=0)
    all_last_tokens = rearrange(all_last_tokens, 'n -> n 1')
    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim=-1)
    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)
    return chunks_with_extra_token, seq


def text_folder_to_chunks_(*, folder, chunks_memmap_path, seqs_memmap_path, doc_ids_memmap_path, chunk_size=64, seq_len=2048, glob='**/*.txt', max_chunks=1000000, max_seqs=100000):
    paths = sorted([*Path(folder).glob(glob)])
    total_chunks = 0
    total_docs = 0
    total_seqs = 0
    chunks_shape = max_chunks, chunk_size + 1
    seqs_shape = max_seqs,
    doc_ids_shape = max_chunks,
    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32, mode='w+') as chunks_memmap, memmap(seqs_memmap_path, shape=seqs_shape, dtype=np.int32, mode='w+') as seqs_memmap, memmap(doc_ids_memmap_path, shape=doc_ids_shape, dtype=np.int32, mode='w+') as doc_ids_memmap:
        for path in paths:
            None
            chunks, seq = doc_text_to_chunks_and_seq_indices(doc_text=path.read_text(), chunk_size=chunk_size, seq_len=seq_len)
            doc_chunk_len = chunks.shape[0]
            doc_seq_len = seq.shape[0]
            chunks_memmap[total_chunks:total_chunks + doc_chunk_len] = chunks.numpy()
            seqs_memmap[total_seqs:total_seqs + doc_seq_len] = seq.numpy() + total_chunks
            doc_ids_memmap[total_chunks:total_chunks + doc_chunk_len] = np.full((doc_chunk_len,), total_docs)
            total_chunks += doc_chunk_len
            total_seqs += doc_seq_len
            total_docs += 1
    return dict(chunks=total_chunks, docs=total_docs, seqs=total_seqs)


def top_k(logits, thres=0.9):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > 1 - thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


class TrainingWrapper(nn.Module):

    def __init__(self, *, retro, chunk_size, documents_path, knn, glob='**/*.txt', chunks_memmap_path='./train.chunks.dat', seqs_memmap_path='./train.seq.dat', doc_ids_memmap_path='./train.doc_ids.dat', max_chunks=1000000, max_seqs=100000, knn_extra_neighbors=100, processed_stats_json_path='./processed-stats.json', faiss_index_filename='knn.index', **index_kwargs):
        super().__init__()
        assert isinstance(retro, RETRO), 'retro must be instance of RETRO'
        self.retro = retro
        force_reprocess = is_true_env_flag('REPROCESS')
        stats_path = Path(processed_stats_json_path)
        if not stats_path.exists() or force_reprocess:
            self.stats = text_folder_to_chunks_(folder=documents_path, glob=glob, chunks_memmap_path=chunks_memmap_path, seqs_memmap_path=seqs_memmap_path, doc_ids_memmap_path=doc_ids_memmap_path, chunk_size=chunk_size, seq_len=retro.seq_len, max_chunks=max_chunks, max_seqs=max_seqs)
            with open(processed_stats_json_path, 'w') as f:
                json.dump(self.stats, f)
        else:
            None
            self.stats = json.loads(stats_path.read_text())
        num_chunks = self.stats['chunks']
        num_seqs = self.stats['seqs']
        knn_memmap_path, faiss_index = chunks_to_precalculated_knn_(num_chunks=num_chunks, chunk_size=chunk_size, chunk_memmap_path=chunks_memmap_path, doc_ids_memmap_path=doc_ids_memmap_path, num_nearest_neighbors=knn, num_extra_neighbors=knn_extra_neighbors, index_file=faiss_index_filename, force_reprocess=force_reprocess, **index_kwargs)
        self.ds = RETRODataset(num_sequences=num_seqs, num_chunks=num_chunks, num_neighbors=knn, chunk_size=chunk_size, seq_len=retro.seq_len, chunk_memmap_path=chunks_memmap_path, chunk_nn_memmap_path=knn_memmap_path, seq_memmap_path=seqs_memmap_path)
        self.chunk_size = chunk_size
        self.max_seq_len = self.retro.seq_len
        self.fetch_knn_chunks_fn = partial(knn_chunks_from_seq_chunks, knn=knn, chunk_size=chunk_size, num_chunks=num_chunks, chunks_memmap_path=chunks_memmap_path, faiss_index=faiss_index)

    @torch.no_grad()
    @eval_decorator
    def generate(self, start=None, retrieved=None, filter_fn=top_k, filter_thres=0.9, temperature=1.0):
        assert filter_fn in {top_k, top_p}, 'filter function must be either top-k or nucleus'
        device = next(self.retro.parameters()).device
        if not exists(start):
            start = torch.full((1, 1), SOS_ID, device=device).long()
        b, start_seq_len = start.shape
        start = start
        if start_seq_len >= self.chunk_size:
            seq_index = start_seq_len // self.chunk_size * self.chunk_size
            past_seq_chunks = rearrange(start[:, :seq_index], 'b (n c) -> (b n) c', c=self.chunk_size)
            retrieved = self.fetch_knn_chunks_fn(past_seq_chunks)
            retrieved = rearrange(retrieved, '(b n) k c -> b n k c', b=b)
        out = start
        for i in range(start_seq_len - 1, self.max_seq_len):
            logits = self.retro(out, retrieved=retrieved)
            logits = logits[:, i]
            logits = filter_fn(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, temperature=temperature, dim=-1)
            sampled = rearrange(sampled, 'b -> b 1')
            out = torch.cat((out, sampled), dim=1)
            is_eos_tokens = out == EOS_ID
            if is_eos_tokens.any(dim=-1).all():
                shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, self.retro.pad_id)
                break
            curr_seq_len = out.shape[-1]
            if curr_seq_len % self.chunk_size == 0:
                last_chunk = rearrange(out, 'b (c n) -> b c n', n=self.chunk_size)[:, -1]
                knn_chunks = self.fetch_knn_chunks_fn(last_chunk)
                knn_chunks = rearrange(knn_chunks, 'b k r -> b 1 k r')
                retrieved = safe_cat(retrieved, knn_chunks, dim=1)
                None
        return out

    def get_dataloader(self, **kwargs):
        return DataLoader(self.ds, **kwargs)

    def get_optimizer(self, **kwargs):
        return get_optimizer(self.retro.parameters(), **kwargs)

    def forward(self):
        raise NotImplemented


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PostNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_RETRO_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

