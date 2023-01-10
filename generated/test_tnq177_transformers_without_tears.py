import sys
_module = sys.modules[__name__]
del sys
all_constants = _module
configurations = _module
controller = _module
data_manager = _module
layers = _module
main = _module
model = _module
preprocessing = _module
utils = _module

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


import time


import numpy as np


import torch


from torch import nn


from torch.nn import Parameter


import torch.nn.functional as F


import re


import logging


class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module
    I learned a lot from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
    """

    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.dropout = args.att_dropout
        self.use_bias = args.use_bias
        if self.embed_dim % self.num_heads != 0:
            raise ValueError('Required: embed_dim % num_heads == 0')
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        if self.use_bias:
            self.biases = Parameter(torch.Tensor(4 * self.embed_dim))
        mean = 0
        std = (2 / (5 * self.embed_dim)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.biases, 0.0)

    def forward(self, q, k, v, mask, do_proj_qkv=True):

        def _split_heads(tensor):
            bsz, length, embed_dim = tensor.size()
            tensor = tensor.reshape(bsz, length, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, -1, self.head_dim)
            return tensor
        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)
        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)
        att_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        bsz_x_num_heads, src_len, tgt_len = att_weights.size()
        bsz = bsz_x_num_heads // self.num_heads
        att_weights = att_weights.reshape(bsz, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att_weights.masked_fill_(mask, -1000000000.0)
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(bsz_x_num_heads, src_len, tgt_len)
        output = torch.bmm(_att_weights, v)
        output = output.reshape(bsz, self.num_heads, src_len, self.head_dim).transpose(1, 2).reshape(bsz, src_len, -1)
        output = self.proj_o(output)
        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()
        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.embed_dim).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.embed_dim)
            k, v = self._proj(k, start=self.embed_dim, end=3 * self.embed_dim).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)
        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.embed_dim)

    def proj_k(self, k):
        return self._proj(k, start=self.embed_dim, end=2 * self.embed_dim)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.embed_dim, end=3 * self.embed_dim)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.embed_dim)


class FeedForward(nn.Module):
    """FeedForward"""

    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.ff_dim = args.ff_dim
        self.embed_dim = args.embed_dim
        self.use_bias = args.use_bias
        self.in_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=self.use_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=self.use_bias)
        mean = 0
        std = (2 / (self.ff_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-05):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class Encoder(nn.Module):
    """Self-attention Encoder"""

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_enc_layers
        self.pre_act = args.pre_act
        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.num_layers)])
        num_scales = self.num_layers * 2 + 1 if self.pre_act else self.num_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.embed_dim ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.embed_dim) for _ in range(num_scales)])

    def forward(self, src_inputs, src_mask):
        pre_act = self.pre_act
        post_act = not pre_act
        x = F.dropout(src_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            att = self.atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[2 * i]
            ff_scale = self.scales[2 * i + 1]
            residual = x
            x = att_scale(x) if pre_act else x
            x, _ = att(q=x, k=x, v=x, mask=src_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = att_scale(x) if post_act else x
            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_scale(x) if post_act else x
        x = self.scales[-1](x) if pre_act else x
        return x


class Decoder(nn.Module):
    """Self-attention Decoder"""

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_dec_layers
        self.pre_act = args.pre_act
        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.cross_atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.num_layers)])
        num_scales = self.num_layers * 3 + 1 if self.pre_act else self.num_layers * 3
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.embed_dim ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.embed_dim) for _ in range(num_scales)])

    def forward(self, tgt_inputs, tgt_mask, encoder_out, encoder_mask):
        pre_act = self.pre_act
        post_act = not pre_act
        x = F.dropout(tgt_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            att = self.atts[i]
            cross_att = self.cross_atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[3 * i]
            cross_att_scale = self.scales[3 * i + 1]
            ff_scale = self.scales[3 * i + 2]
            residual = x
            x = att_scale(x) if pre_act else x
            x, _ = att(q=x, k=x, v=x, mask=tgt_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = att_scale(x) if post_act else x
            residual = x
            x = cross_att_scale(x) if pre_act else x
            x, _ = cross_att(q=x, k=encoder_out, v=encoder_out, mask=encoder_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = cross_att_scale(x) if post_act else x
            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_scale(x) if post_act else x
        x = self.scales[-1](x) if pre_act else x
        return x

    def beam_step(self, inp, cache):
        bsz, beam_size = cache['encoder_mask'].size()[:2]
        cache['encoder_mask'] = cache['encoder_mask'].reshape(bsz * beam_size, 1, 1, -1)
        for i in range(self.num_layers):
            seq_len = cache[i]['cross_att_k'].size(2)
            cache[i]['cross_att_k'] = cache[i]['cross_att_k'].reshape(bsz * beam_size, seq_len, -1)
            cache[i]['cross_att_v'] = cache[i]['cross_att_v'].reshape(bsz * beam_size, seq_len, -1)
            if cache[i]['att']['k'] is not None:
                seq_len = cache[i]['att']['k'].size(2)
                cache[i]['att']['k'] = cache[i]['att']['k'].reshape(bsz * beam_size, seq_len, -1)
                cache[i]['att']['v'] = cache[i]['att']['v'].reshape(bsz * beam_size, seq_len, -1)
        pre_act = self.pre_act
        post_act = not pre_act
        x = inp
        for i in range(self.num_layers):
            att = self.atts[i]
            cross_att = self.cross_atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[3 * i]
            cross_att_scale = self.scales[3 * i + 1]
            ff_scale = self.scales[3 * i + 2]
            residual = x
            x = att_scale(x) if pre_act else x
            q, k, v = att.proj_qkv(x, x, x)
            if cache[i]['att']['k'] is not None:
                k = torch.cat((cache[i]['att']['k'], k), 1)
                v = torch.cat((cache[i]['att']['v'], v), 1)
            cache[i]['att']['k'] = k
            cache[i]['att']['v'] = v
            x, _ = att(q=q, k=k, v=v, mask=None, do_proj_qkv=False)
            x = residual + x
            x = att_scale(x) if post_act else x
            residual = x
            x = cross_att_scale(x) if pre_act else x
            q = cross_att.proj_q(x)
            k = cache[i]['cross_att_k']
            v = cache[i]['cross_att_v']
            mask = cache['encoder_mask']
            x, _ = cross_att(q=q, k=k, v=v, mask=mask, do_proj_qkv=False)
            x = residual + x
            x = cross_att_scale(x) if post_act else x
            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + x
            x = ff_scale(x) if post_act else x
        x = self.scales[-1](x) if pre_act else x
        cache['encoder_mask'] = cache['encoder_mask'].reshape(bsz, beam_size, 1, 1, -1)
        for i in range(self.num_layers):
            seq_len = cache[i]['cross_att_k'].size(1)
            cache[i]['cross_att_k'] = cache[i]['cross_att_k'].reshape(bsz, beam_size, seq_len, -1)
            cache[i]['cross_att_v'] = cache[i]['cross_att_v'].reshape(bsz, beam_size, seq_len, -1)
            seq_len = cache[i]['att']['k'].size(1)
            cache[i]['att']['k'] = cache[i]['att']['k'].reshape(bsz, beam_size, seq_len, -1)
            cache[i]['att']['v'] = cache[i]['att']['v'].reshape(bsz, beam_size, seq_len, -1)
        return x

    def beam_decode(self, encoder_out, encoder_mask, get_input_fn, logprob_fn, bos_id, eos_id, max_len, beam_size=4, alpha=-1):
        batch_size = encoder_out.size(0)
        inp = get_input_fn(torch.tensor([bos_id] * batch_size).reshape(batch_size, 1), 0)
        cache = {'encoder_mask': encoder_mask.unsqueeze_(1)}
        for i in range(self.num_layers):
            cache[i] = {'att': {'k': None, 'v': None}}
            cache[i]['cross_att_k'] = self.cross_atts[i].proj_k(encoder_out).unsqueeze_(1)
            cache[i]['cross_att_v'] = self.cross_atts[i].proj_v(encoder_out).unsqueeze_(1)
        y = self.beam_step(inp, cache).squeeze_(1)
        probs = logprob_fn(y)
        probs[:, eos_id] = float('-inf')
        all_probs, symbols = torch.topk(probs, beam_size, dim=-1)
        last_probs = all_probs.reshape(batch_size, beam_size)
        last_scores = last_probs.clone()
        all_symbols = symbols.reshape(batch_size, beam_size, 1)
        cache['encoder_mask'] = encoder_mask.expand(-1, beam_size, -1, -1, -1)
        for i in range(self.num_layers):
            cache[i]['att']['k'] = cache[i]['att']['k'].expand(-1, beam_size, -1, -1)
            cache[i]['att']['v'] = cache[i]['att']['v'].expand(-1, beam_size, -1, -1)
            cache[i]['cross_att_k'] = cache[i]['cross_att_k'].expand(-1, beam_size, -1, -1)
            cache[i]['cross_att_v'] = cache[i]['cross_att_v'].expand(-1, beam_size, -1, -1)
        num_classes = probs.size(-1)
        not_eos_mask = (torch.arange(num_classes).reshape(1, -1) != eos_id).type(encoder_mask.type())
        maximum_length = max_len.max().item()
        ret = [None] * batch_size
        batch_idxs = torch.arange(batch_size)
        for time_step in range(1, maximum_length + 1):
            surpass_length = (max_len < time_step) + (time_step == maximum_length)
            finished_decoded = torch.sum((all_symbols[:, :, -1] == eos_id).type(max_len.type()), -1) == beam_size
            finished_sents = surpass_length + finished_decoded
            if finished_sents.any():
                for j in range(finished_sents.size(0)):
                    if finished_sents[j]:
                        ret[batch_idxs[j]] = {'symbols': all_symbols[j].clone(), 'probs': last_probs[j].clone(), 'scores': last_scores[j].clone()}
                all_symbols = all_symbols[~finished_sents]
                last_probs = last_probs[~finished_sents]
                last_scores = last_scores[~finished_sents]
                max_len = max_len[~finished_sents]
                batch_idxs = batch_idxs[~finished_sents]
                cache['encoder_mask'] = cache['encoder_mask'][~finished_sents]
                for i in range(self.num_layers):
                    cache[i]['att']['k'] = cache[i]['att']['k'][~finished_sents]
                    cache[i]['att']['v'] = cache[i]['att']['v'][~finished_sents]
                    cache[i]['cross_att_k'] = cache[i]['cross_att_k'][~finished_sents]
                    cache[i]['cross_att_v'] = cache[i]['cross_att_v'][~finished_sents]
            if finished_sents.all():
                break
            bsz = all_symbols.size(0)
            last_symbols = all_symbols[:, :, -1]
            inps = get_input_fn(last_symbols, time_step).reshape(bsz * beam_size, -1).unsqueeze_(1)
            ys = self.beam_step(inps, cache).squeeze_(1)
            probs = logprob_fn(ys)
            last_probs = last_probs.reshape(-1, 1)
            last_scores = last_scores.reshape(-1, 1)
            length_penalty = 1.0 if alpha == -1 else (5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha
            finished_mask = last_symbols.reshape(-1) == eos_id
            beam_probs = probs.clone()
            if finished_mask.any():
                beam_probs[finished_mask] = last_probs[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_probs[~finished_mask] = last_probs[~finished_mask] + probs[~finished_mask]
            else:
                beam_probs = last_probs + probs
            beam_scores = beam_probs.clone()
            if finished_mask.any():
                beam_scores[finished_mask] = last_scores[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_scores[~finished_mask] = beam_probs[~finished_mask] / length_penalty
            else:
                beam_scores = beam_probs / length_penalty
            beam_probs = beam_probs.reshape(bsz, -1)
            beam_scores = beam_scores.reshape(bsz, -1)
            max_scores, idxs = torch.topk(beam_scores, beam_size, dim=-1)
            parent_idxs = idxs // num_classes
            symbols = (idxs - parent_idxs * num_classes).type(idxs.type())
            last_probs = torch.gather(beam_probs, -1, idxs)
            last_scores = max_scores
            parent_idxs = parent_idxs + torch.arange(bsz).unsqueeze_(1).type(parent_idxs.type()) * beam_size
            parent_idxs = parent_idxs.reshape(-1)
            all_symbols = all_symbols.reshape(bsz * beam_size, -1)[parent_idxs].reshape(bsz, beam_size, -1)
            all_symbols = torch.cat((all_symbols, symbols.unsqueeze_(-1)), -1)
            for i in range(self.num_layers):
                seq_len = cache[i]['att']['k'].size(2)
                cache[i]['att']['k'] = cache[i]['att']['k'].reshape(bsz * beam_size, seq_len, -1)[parent_idxs].reshape(bsz, beam_size, seq_len, -1)
                cache[i]['att']['v'] = cache[i]['att']['v'].reshape(bsz * beam_size, seq_len, -1)[parent_idxs].reshape(bsz, beam_size, seq_len, -1)
        if batch_idxs.size(0) > 0:
            for j in range(batch_idxs.size(0)):
                ret[batch_idxs[j]] = {'symbols': all_symbols[j].clone(), 'probs': last_probs[j].clone(), 'scores': last_scores[j].clone()}
        return ret


class Transformer(nn.Module):
    """Transformer https://arxiv.org/pdf/1706.03762.pdf"""

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        embed_dim = args.embed_dim
        fix_norm = args.fix_norm
        joint_vocab_size = args.joint_vocab_size
        lang_vocab_size = args.lang_vocab_size
        use_bias = args.use_bias
        self.scale = embed_dim ** 0.5
        if args.mask_logit:
            self.logit_mask = None
        else:
            mask = [True] * joint_vocab_size
            mask[ac.BOS_ID] = False
            mask[ac.PAD_ID] = False
            self.logit_mask = torch.tensor(mask).type(torch.bool)
        self.word_embedding = Parameter(torch.Tensor(joint_vocab_size, embed_dim))
        self.lang_embedding = Parameter(torch.Tensor(lang_vocab_size, embed_dim))
        self.out_bias = Parameter(torch.Tensor(joint_vocab_size)) if use_bias else None
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        nn.init.normal_(self.lang_embedding, mean=0, std=embed_dim ** -0.5)
        if fix_norm:
            d = 0.01
            nn.init.uniform_(self.word_embedding, a=-d, b=d)
        else:
            nn.init.normal_(self.word_embedding, mean=0, std=embed_dim ** -0.5)
        if use_bias:
            nn.init.constant_(self.out_bias, 0.0)

    def replace_with_unk(self, toks):
        p = self.args.word_dropout
        if self.training and 0 < p < 1:
            non_pad_mask = toks != ac.PAD_ID
            mask = (torch.rand(toks.size()) <= p).type(non_pad_mask.type())
            mask = mask & non_pad_mask
            toks[mask] = ac.UNK_ID

    def get_input(self, toks, lang_idx, word_embedding, pos_embedding):
        self.replace_with_unk(toks)
        word_embed = F.embedding(toks, word_embedding) * self.scale
        lang_embed = self.lang_embedding[lang_idx].unsqueeze(0).unsqueeze(1)
        pos_embed = pos_embedding[:toks.size(-1), :].unsqueeze(0)
        return word_embed + lang_embed + pos_embed

    def forward(self, src, tgt, targets, src_lang_idx, tgt_lang_idx, logit_mask):
        embed_dim = self.args.embed_dim
        max_len = max(src.size(1), tgt.size(1))
        pos_embedding = ut.get_positional_encoding(embed_dim, max_len)
        word_embedding = F.normalize(self.word_embedding, dim=-1) if self.args.fix_norm else self.word_embedding
        encoder_inputs = self.get_input(src, src_lang_idx, word_embedding, pos_embedding)
        encoder_mask = (src == ac.PAD_ID).unsqueeze(1).unsqueeze(2)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        decoder_inputs = self.get_input(tgt, tgt_lang_idx, word_embedding, pos_embedding)
        decoder_mask = torch.triu(torch.ones((tgt.size(-1), tgt.size(-1))), diagonal=1).type(tgt.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)
        logit_mask = logit_mask == 1 if self.logit_mask is None else self.logit_mask
        logits = self.logit_fn(decoder_outputs, word_embedding, logit_mask)
        neglprobs = F.log_softmax(logits, -1) * logit_mask.type(logits.type()).reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID
        nll_loss = neglprobs.gather(dim=-1, index=targets)[non_pad_mask]
        smooth_loss = neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = -nll_loss.sum()
        smooth_loss = -smooth_loss.sum()
        label_smoothing = self.args.label_smoothing
        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / logit_mask.type(nll_loss.type()).sum()
        else:
            loss = nll_loss
        num_words = non_pad_mask.type(loss.type()).sum()
        opt_loss = loss / num_words
        return {'opt_loss': opt_loss, 'loss': loss, 'nll_loss': nll_loss, 'num_words': num_words}

    def logit_fn(self, decoder_output, softmax_weight, logit_mask):
        logits = F.linear(decoder_output, softmax_weight, bias=self.out_bias)
        logits = logits.reshape(-1, logits.size(-1))
        logits[:, ~logit_mask] = -1000000000.0
        return logits

    def beam_decode(self, src, src_lang_idx, tgt_lang_idx, logit_mask):
        embed_dim = self.args.embed_dim
        max_len = src.size(1) + 51
        pos_embedding = ut.get_positional_encoding(embed_dim, max_len)
        word_embedding = F.normalize(self.word_embedding, dim=-1) if self.args.fix_norm else self.word_embedding
        logit_mask = logit_mask == 1 if self.logit_mask is None else self.logit_mask
        tgt_lang_embed = self.lang_embedding[tgt_lang_idx]
        encoder_inputs = self.get_input(src, src_lang_idx, word_embedding, pos_embedding)
        encoder_mask = (src == ac.PAD_ID).unsqueeze(1).unsqueeze(2)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)

        def get_tgt_inp(tgt, time_step):
            word_embed = F.embedding(tgt.type(src.type()), word_embedding) * self.scale
            pos_embed = pos_embedding[time_step, :].reshape(1, 1, -1)
            return word_embed + tgt_lang_embed + pos_embed

        def logprob_fn(decoder_output):
            logits = self.logit_fn(decoder_output, word_embedding, logit_mask)
            return F.log_softmax(logits, dim=-1)
        max_lengths = torch.sum(src != ac.PAD_ID, dim=-1).type(src.type()) + 50
        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_tgt_inp, logprob_fn, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.args.beam_size, alpha=self.args.beam_alpha)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'args': _mock_config(ff_dropout=0.5, ff_dim=4, embed_dim=4, use_bias=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleNorm,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tnq177_transformers_without_tears(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

