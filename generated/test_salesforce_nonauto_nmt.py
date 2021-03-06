import sys
_module = sys.modules[__name__]
del sys
decode = _module
model = _module
run = _module
generate_fertility = _module
remove = _module
self_learn = _module
train = _module
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


import torch


import numpy as np


import time


from torch.nn import functional as F


from torch.autograd import Variable


from time import gmtime


from time import strftime


from torch import nn


from torch.autograd import Function


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import math


from torchtext import data


from torchtext import datasets


import logging


import random


import string


import uuid


import copy


from collections import OrderedDict


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


TINY = 1e-09


class CosineLinear(Linear):

    def forward(self, x, tau=0.05):
        size = x.size()
        x = x / (x.norm(dim=-1, keepdim=True).expand_as(x) + TINY)
        x = x.contiguous().view(-1, size[-1])
        weight = self.weight / (self.weight.norm(dim=-1, keepdim=True).expand_as(self.weight) + TINY)
        value = F.linear(x, weight)
        value = value.view(*size[:-1], -1) / tau
        return value


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))


class ConvolutionBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, pos=0, w=1):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos
        self.kernel = nn.Conv1d(d_model, d_model, 2 * w + 1, 1, padding=w)

    def forward(self, *x):
        return self.layernorm(self.kernel(x[self.pos].transpose(2, 1)).transpose(2, 1) + self.dropout(self.layer(*x)))


class HighwayBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.gate = Linear(d_model, 1)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        g = F.sigmoid(self.gate(x[0])).expand_as(x[0])
        return self.layernorm(x[0] * g + self.dropout(self.layer(*x)) * (1 - g))


INF = 10000000000.0


def softmax(x):
    if x.dim() == 3:
        return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.softmax(x)


def gumbel_softmax(input, beta=0.5, tau=1.0):
    noise = input.data.new(*input.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return softmax((input + beta * Variable(noise)) / tau)


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal, diag=False, window=-1, noisy=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.diag = diag
        self.window = window
        self.noisy = noisy

    def forward(self, query, key, value=None, mask=None, feedback=None, beta=0, tau=1, weights=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if weights is not None:
            dot_products = dot_products + weights
        if query.dim() == 3 and self.causal and query.size(1) == key.size(1):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0))
        if self.window > 0:
            window_mask = key.data.new(key.size(1), key.size(1)).fill_(1)
            window_mask = (window_mask.triu(self.window + 1) + window_mask.tril(-self.window - 1)) * INF
            dot_products.data.sub_(window_mask.unsqueeze(0))
        if self.diag:
            inds = torch.arange(0, key.size(1)).long().view(1, 1, -1)
            if key.is_cuda:
                inds = inds
            dot_products.data.scatter_(1, inds.expand(dot_products.size(0), 1, inds.size(-1)), -INF)
        if mask is not None:
            if dot_products.dim() == 2:
                dot_products.data -= (1 - mask) * INF
            else:
                dot_products.data -= (1 - mask[:, (None), :]) * INF
        if value is None:
            return dot_products
        logits = dot_products / self.scale
        if not self.noisy:
            probs = softmax(logits)
        else:
            probs = gumbel_softmax(logits, beta=beta, tau=tau)
        if feedback is not None:
            feedback.append(probs.contiguous())
        return matmul(self.dropout(probs), value)


class MultiHead2(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False, diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        if use_wo:
            self.wo = Linear(d_value, d_key, bias=use_wo)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, value, mask=None, feedback=None, weights=None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads
        probs = []
        query, key, value = (x.contiguous().view(B, -1, N, D // N).transpose(2, 1).contiguous().view(B * N, -1, D // N) for x in (query, key, value))
        if mask is not None:
            mask = mask[:, (None), :].expand(B, N, Tk).contiguous().view(B * N, -1)
        outputs = self.attention(query, key, value, mask, probs, beta, tau, weights)
        outputs = outputs.contiguous().view(B, N, -1, D // N).transpose(2, 1).contiguous().view(B, -1, D)
        if feedback is not None:
            feedback.append(probs[0].view(B, N, Tq, Tk))
        if self.use_wo:
            return self.wo(outputs)
        return outputs


class MoEHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False, diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        self.wo = Linear(d_value, d_key, bias=use_wo)
        self.gate = Linear(d_value // n_heads, 1)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, inputs, mask=None, feedback=None, weights=None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(inputs)
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads
        probs = []
        query, key, value = (x.contiguous().view(B, -1, N, D // N).transpose(2, 1).contiguous().view(B * N, -1, D // N) for x in (query, key, value))
        if mask is not None:
            mask = mask[:, (None), :].expand(B, N, Tk).contiguous().view(B * N, -1)
        probs = self.attention(query, key, None, mask, probs, beta, tau, weights)
        mix = matmul(self.attention.dropout(probs), value).contiguous().view(B, N, -1, D // N).transpose(2, 1).contiguous()
        mix = softmax(self.gate(mix))
        probs = (probs.contiguous().view(B, N, Tq, Tk).transpose(2, 1) * mix).sum(-2)
        outputs = matmul(probs, inputs)
        return self.wo(outputs)


class ReorderHead(nn.Module):

    def __init__(self, d_key, n_heads, drop_ratio, causal=False, diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.n_heads = n_heads

    def forward(self, query, key, positions=None, mask=None, feedback=None, beta=0, tau=1):
        B, D = query.size()[0], query.size()[-1]
        T = key.size()[1]
        N = self.n_heads
        query, key = self.wq(query), self.wk(key)
        query, key = (x.contiguous().view(B, -1, N, D // N).transpose(2, 1).contiguous().view(B * N, -1, D // N) for x in (query, key))
        mask = mask[:, (None), :].expand(B, N, T).contiguous().view(B * N, -1)
        probs = self.attention(query, key, None, mask, None, beta, tau).transpose(2, 1)
        return probs


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class LSTMCritic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.out = nn.Linear(args.d_model, args.trg_vocab, bias=False)
        self.bilstm = nn.LSTM(args.d_model, args.d_model, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(2 * args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, 1), nn.Sigmoid())

    def forward(self, trg, mask):
        batchsize = trg.size(0)
        trgs = grad_reverse(trg)
        lens = mask.sum(-1).long()
        lens, indices = torch.sort(lens, dim=0, descending=True)
        if trg.is_cuda:
            with torch.cuda.device_of(trg):
                lens = lens.tolist()
        trgs = pack_padded_sequence(trgs[(indices), :, :], lens, batch_first=True)
        _, (out, _) = self.bilstm(trgs)
        out = out.permute(1, 0, 2).contiguous().view(batchsize, -1)
        return self.mlp(out)


class ConvCritic(nn.Module):

    def __init__(self, args):
        """
        accept a sequence of word-embeddings and classification on that
        """
        super(ConvCritic, self).__init__()
        self.args = args
        D = args.d_model
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.out = nn.Linear(args.d_model, args.trg_vocab, bias=False)
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.mlp = nn.Sequential(nn.Linear(len(Ks) * Co, args.d_model), nn.ReLU(), nn.Linear(args.d_model, 1), nn.Sigmoid())

    def forward(self, x, mask):
        x = grad_reverse(x)
        x = x.transpose(2, 1)
        x = [F.relu(conv(x)) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.mlp(x)


class EncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(MultiHead2(args.d_model, args.d_model, args.n_heads, args.drop_ratio, use_wo=args.use_wo), args.d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(args.d_model, args.d_hidden), args.d_model, args.drop_ratio)

    def forward(self, x, mask=None):
        return self.feedforward(self.selfattn(x, x, x, mask))


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(-2))
        if x.is_cuda:
            positions = positions
        positions = Variable(positions.float())
    else:
        positions = t
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1)
    if x.is_cuda:
        channels = channels
    channels = 1 / 10000 ** Variable(channels)
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)
    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)
    return encodings


class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, diag=False, highway=False, window=-1, positional=False, noisy=False):
        super().__init__()
        self.positional = positional
        self.selfattn = ResidualBlock(MultiHead2(args.d_model, args.d_model, args.n_heads, args.drop_ratio, causal, diag, window, use_wo=args.use_wo), args.d_model, args.drop_ratio)
        self.attention = ResidualBlock(MultiHead2(args.d_model, args.d_model, args.n_heads, args.drop_ratio, noisy=noisy, use_wo=args.use_wo), args.d_model, args.drop_ratio)
        if positional:
            self.pos_selfattn = ResidualBlock(MultiHead2(args.d_model, args.d_model, args.n_heads, args.drop_ratio, causal, diag, window, use_wo=args.use_wo), args.d_model, args.drop_ratio, pos=2)
        self.feedforward = ResidualBlock(FeedForward(args.d_model, args.d_hidden), args.d_model, args.drop_ratio)

    def forward(self, x, encoding, p=None, mask_src=None, mask_trg=None, feedback=None):
        feedback_src = []
        feedback_trg = []
        x = self.selfattn(x, x, x, mask_trg, feedback_trg)
        if self.positional:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_selfattn(pos_encoding, pos_encoding, x, mask_trg, None, weights)
        x = self.feedforward(self.attention(x, encoding, encoding, mask_src, feedback_src))
        if feedback is not None:
            if 'source' not in feedback:
                feedback['source'] = feedback_src
            else:
                feedback['source'] += feedback_src
            if 'target' not in feedback:
                feedback['target'] = feedback_trg
            else:
                feedback['target'] += feedback_trg
        return x


class Encoder(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        if args.share_embeddings:
            self.out = nn.Linear(args.d_model, len(field.vocab))
        else:
            self.embed = nn.Embedding(len(field.vocab), args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings

    def forward(self, x, mask=None):
        if self.share_embeddings:
            x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        else:
            x = self.embed(x)
        x += positional_encodings_like(x)
        encoding = [x]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
            encoding.append(x)
        return encoding


def log_softmax(x):
    if x.dim() == 3:
        return F.log_softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.log_softmax(x)


class Decoder(nn.Module):

    def __init__(self, field, args, causal=True, positional=False, diag=False, highway=False, windows=None, noisy=False, cosine_output=False):
        super().__init__()
        if windows is None:
            windows = [(-1) for _ in range(args.n_layers)]
        self.layers = nn.ModuleList([DecoderLayer(args, causal, diag, highway, windows[i], positional, noisy) for i in range(args.n_layers)])
        self.out = nn.Linear(args.d_model, len(field.vocab))
        self.dropout = nn.Dropout(args.drop_ratio)
        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.positional = positional
        self.orderless = args.input_orderless

    def forward(self, x, encoding, mask_src=None, mask_trg=None, input_embeddings=False, feedback=None, positions=None):
        if not input_embeddings:
            if x.ndimension() == 2:
                x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
            elif x.ndimension() == 3:
                x = x @ self.out.weight * math.sqrt(self.d_model)
        if not self.orderless:
            x += positional_encodings_like(x)
        x = self.dropout(x)
        for l, (layer, enc) in enumerate(zip(self.layers, encoding[1:])):
            x = layer(x, enc, mask_src=mask_src, mask_trg=mask_trg, feedback=feedback)
        return x

    def greedy(self, encoding, mask_src=None, mask_trg=None, feedback=None):
        encoding = encoding[1:]
        B, T, C = encoding[0].size()
        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, T + 1).long().fill_(self.field.vocab.stoi['<init>']))
        hiddens = [Variable(encoding[0].data.new(B, T, C).zero_()) for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B).byte().zero_()
        attentions = []
        for t in range(T):
            torch.cuda.nvtx.mark(f'greedy:{t}')
            hiddens[0][:, (t)] = self.dropout(hiddens[0][:, (t)] + F.embedding(outs[:, (t)], embedW))
            inter_attention = []
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]
                x = self.layers[l].selfattn(hiddens[l][:, t:t + 1], x, x)
                hiddens[l + 1][:, (t)] = self.layers[l].feedforward(self.layers[l].attention(x, encoding[l], encoding[l], mask_src, inter_attention))[:, (0)]
            inter_attention = torch.cat(inter_attention, 1)
            attentions.append(inter_attention)
            _, preds = self.out(hiddens[-1][:, (t)]).max(-1)
            preds[eos_yet] = self.field.vocab.stoi['<pad>']
            eos_yet = eos_yet | (preds.data == self.field.vocab.stoi['<eos>'])
            outs[:, (t + 1)] = preds
            if eos_yet.all():
                break
        if feedback is not None:
            feedback['source'] = torch.cat(attentions, 2)
        return outs[:, 1:t + 2]

    def beam_search(self, encoding, mask_src=None, mask_trg=None, width=2, alpha=0.6):
        encoding = encoding[1:]
        W = width
        B, T, C = encoding[0].size()
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, (None), :].expand(B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, (None), :].expand(B, W, T).contiguous().view(B * W, T)
        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, W, T + 1).long().fill_(self.field.vocab.stoi['<init>']))
        logps = Variable(encoding[0].data.new(B, W).float().fill_(0))
        hiddens = [Variable(encoding[0].data.new(B, W, T, C).zero_()) for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B, W).byte().zero_()
        eos_mask = eos_yet.float().fill_(-INF)[:, :, (None)].expand(B, W, W)
        eos_mask[:, :, (0)] = 0
        for t in range(T):
            hiddens[0][:, :, (t)] = self.dropout(hiddens[0][:, :, (t)] + F.embedding(outs[:, :, (t)], embedW))
            for l in range(len(self.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, (t)] = self.layers[l].feedforward(self.layers[l].attention(x, encoding[l], encoding[l], mask_src)).view(B, W, C)
            topk2_logps, topk2_inds = log_softmax(self.out(hiddens[-1][:, :, (t)])).topk(W, dim=-1)
            topk2_logps = topk2_logps * Variable(eos_yet[:, :, (None)].float() * eos_mask + 1 - eos_yet[:, :, (None)].float())
            topk2_logps = topk2_logps + logps[:, :, (None)]
            if t == 0:
                logps, topk_inds = topk2_logps[:, (0)].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)
            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)
            logps = logps * (1 - Variable(eos_yet.float()) * 1 / (t + 2)).pow(alpha)
            outs = outs.gather(1, topk_beam_inds[:, :, (None)].expand_as(outs))
            outs[:, :, (t + 1)] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, (None), (None)].expand_as(hiddens[0])
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)
            eos_yet = eos_yet | (topk_token_inds.data == self.field.vocab.stoi['<eos>'])
            if eos_yet.all():
                return outs[:, (0), 1:]
        return outs[:, (0), 1:]


class ReOrderer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.wq = Linear(args.d_model, args.d_model, bias=True)
        self.wk = Linear(args.d_model, args.d_model, bias=True)
        self.gate = Linear(args.d_model, 1, bias=True)
        self.scale = math.sqrt(args.d_model)
        self.diag = False

    @staticmethod
    def linear_attention(mask_src, mask_trg):
        max_src_len = mask_src.size(1)
        max_trg_len = mask_trg.size(1)
        src_lens = mask_src.sum(-1).float()
        trg_lens = mask_trg.sum(-1).float()
        steps = src_lens / trg_lens
        index_t = torch.arange(0, max_trg_len)
        if mask_trg.is_cuda:
            index_t = index_t
        index_t = steps[:, (None)] @ index_t[(None), :]
        index_s = torch.arange(0, max_src_len)
        if mask_trg.is_cuda:
            index_s = index_s
        indexxx = (index_s[(None), (None), :] - index_t[:, :, (None)]) ** 2
        indexxx = softmax(Variable(-indexxx / 0.3 - INF * (1 - mask_src[:, (None), :])))
        return indexxx

    def forward(self, key, mask_src, mask_trg):
        l_att = self.linear_attention(mask_src, mask_trg)
        query = matmul(l_att, key)
        gates = F.sigmoid(self.gate(query).expand(query.size(0), mask_trg.size(1), mask_src.size(1)))
        query, key = self.wq(query), self.wk(key)
        dot_products = matmul(query, key.transpose(1, 2))
        if mask_src.ndimension() == 2:
            dot_products.data -= (1 - mask_src[:, (None), :]) * INF
        else:
            dot_products.data -= (1 - mask_src) * INF
        logits = dot_products / self.scale
        probs = softmax(logits)
        probs = (1 - gates) * probs + gates * l_att
        return probs


def topK_search(logits, mask_src, N=100):
    nlogP = -log_softmax(logits).data
    maxL = nlogP.size(-1)
    overmask = torch.cat([mask_src[:, :, (None)], (1 - mask_src[:, :, (None)]).expand(*mask_src.size(), maxL - 1) * INF + mask_src[:, :, (None)]], 2)
    nlogP = nlogP * overmask
    batch_size, src_len, L = logits.size()
    _, R = nlogP.sort(-1)

    def get_score(data, index):
        return data.gather(-1, index).sum(-2)
    heap_scores = torch.ones(batch_size, N) * INF
    heap_inx = torch.zeros(batch_size, src_len, N).long()
    heap_scores[:, :1] = get_score(nlogP, R[:, :, :1])
    if nlogP.is_cuda:
        heap_scores = heap_scores
        heap_inx = heap_inx

    def span(ins):
        inds = torch.eye(ins.size(1)).long()
        if ins.is_cuda:
            inds = inds
        return ins[:, :, (None)].expand(ins.size(0), ins.size(1), ins.size(1)) + inds[(None), :, :]
    for k in range(1, N):
        cur_inx = heap_inx[:, :, (k - 1)]
        I_t = span(cur_inx).clamp(0, L - 1)
        S_t = get_score(nlogP, R.gather(-1, I_t))
        S_t, _inx = torch.cat([heap_scores[:, k:], S_t], 1).sort(1)
        S_t[:, 1:] += (S_t[:, 1:] - S_t[:, :-1] == 0).float() * INF
        S_t, _inx2 = S_t.sort(1)
        I_t = torch.cat([heap_inx[:, :, k:], I_t], 2).gather(2, _inx.gather(1, _inx2)[:, (None), :].expand(batch_size, src_len, _inx.size(-1)))
        heap_scores[:, k:] = S_t[:, :N - k]
        heap_inx[:, :, k:] = I_t[:, :, :N - k]
    output = R.gather(-1, heap_inx)
    output = output.transpose(2, 1).contiguous().view(batch_size * N, src_len)
    output = Variable(output)
    mask_src = mask_src[:, (None), :].expand(batch_size, N, src_len).contiguous().view(batch_size * N, src_len)
    return output, mask_src


class Fertility(nn.Module):

    def __init__(self, args, L=50):
        super().__init__()
        self.wf = Linear(args.d_model, L, bias=True)
        self.max_ratio = 2
        self.L = L
        self.f0 = torch.arange(0, self.L).float()
        self.saved_fertilities = None

    @staticmethod
    def transform(fertilities, mask):
        fertilities = fertilities.data
        fertilities *= mask.long()
        m = fertilities.max()
        L = fertilities.sum(dim=1).max()
        source_indices = torch.arange(0, fertilities.size(1))[(None), :].expand_as(fertilities)
        if fertilities.is_cuda:
            source_indices = source_indices
        zero_mask_zero = fertilities == 0
        source_indices = source_indices * mask - (1 - mask)
        source_indices.masked_fill_(zero_mask_zero, -1)
        source_indices = torch.cat((source_indices, source_indices.new(source_indices.size(0), 1).fill_(-1)), 1).long()
        target_indices = fertilities.new(source_indices.size(0), L + m).fill_(-1)
        start_indices = torch.cat((fertilities.new(fertilities.size(0), 1).zero_(), fertilities.cumsum(1)), 1)
        zero_fertility_mask = torch.cat((zero_mask_zero, fertilities.new(fertilities.size(0), 1).byte().zero_()), 1)
        start_indices.masked_fill_(zero_fertility_mask, L)
        for offset in range(m - 1, -1, -1):
            target_indices.scatter_(1, start_indices + offset, source_indices)
        target_indices = target_indices[:, :-m].long()
        new_mask = (target_indices != -1).float()
        assert new_mask.sum() - fertilities.sum() == 0, '??'
        new_indices = target_indices * new_mask.long()
        return Variable(new_indices), Variable(fertilities), new_mask

    def forward(self, encoding, mask_src=None, mask_trg=None, mode=None, N=1, tau=1, return_samples=False):
        logits = self.wf(encoding)
        if mode is None:
            return logits
        if mode == 'sharp':
            inxxx = torch.arange(0, self.L).float()
            if encoding.is_cuda:
                inxxx = inxxx
            inxxx = Variable(inxxx)
            fertilities = (softmax(logits / 0.5) * inxxx[(None), (None), :]).sum(-1)
            return fertilities
        elif mode == 'argmax':
            fertilities = logits.max(-1)[1]
        elif mode == 'mean':
            f0 = self.f0
            if encoding.is_cuda:
                f0 = f0
            f0 = Variable(f0)
            fertilities = (softmax(logits) * f0[(None), (None), :]).sum(-1).round().clamp(0, self.L - 1).long()
        elif mode == 'reinforce' or mode == 'sample':
            fertilities = softmax(logits / tau).contiguous().view(-1, self.L).multinomial(N, True)
            self.saved_fertilities = fertilities
            B, T, _ = encoding.size()
            if N == 1:
                fertilities = fertilities.contiguous().view(B, T)
            else:
                fertilities = fertilities.contiguous().view(B, T, N)
                fertilities = fertilities.transpose(2, 1).contiguous().view(B * N, T)
                mask_src = mask_src[:, (None), :].expand(B, N, T).contiguous().view(B * N, T)
        elif mode == 'search':
            fertilities, mask_src = topK_search(logits, mask_src, N)
        else:
            raise NotImplementedError
        cumsum = fertilities.cumsum(dim=1).float()
        overflow = cumsum < self.max_ratio * mask_src.size(1)
        fertilities = fertilities * overflow.long()
        new_indices, fertilities, new_mask = self.transform(fertilities, mask_src)
        if not return_samples:
            return logits, (new_indices, new_mask)
        return logits, fertilities, (new_indices, new_mask)


def mask(targets, out, input_mask=None, return_mask=False):
    if input_mask is None:
        input_mask = targets != 1
    out_mask = input_mask.unsqueeze(-1).expand_as(out)
    if return_mask:
        return targets[input_mask], out[out_mask].view(-1, out.size(-1)), the_mask
    return targets[input_mask], out[out_mask].view(-1, out.size(-1))


class Transformer(nn.Module):

    def __init__(self, src, trg, args):
        super().__init__()
        self.encoder = Encoder(src, args)
        self.decoder = Decoder(trg, args)
        self.field = trg
        self.share_embeddings = args.share_embeddings
        if args.share_embeddings:
            self.encoder.out.weight = self.decoder.out.weight

    def denum(self, data, target=True):
        field = self.decoder.field if target else self.encoder.field
        return field.reverse(data.unsqueeze(0))[0]

    def apply_mask(self, inputs, mask, p=1):
        _mask = Variable(mask.long())
        outputs = inputs * _mask + (1 + -1 * _mask) * p
        return outputs

    def apply_mask_cost(self, loss, mask, batched=False):
        loss.data *= mask
        cost = loss.sum() / (mask.sum() + TINY)
        if not batched:
            return cost
        loss = loss.sum(1, keepdim=True) / (TINY + Variable(mask).sum(1, keepdim=True))
        return cost, loss

    def output_decoding(self, outputs):
        field, text = outputs
        if field is 'src':
            return self.encoder.field.reverse(text.data)
        else:
            return self.decoder.field.reverse(text.data)

    def prepare_sources(self, batch, masks=None):
        masks = self.prepare_masks(batch.src) if masks is None else masks
        return batch.src, masks

    def prepare_inputs(self, batch, inputs=None, distillation=False, masks=None):
        if inputs is None:
            if distillation:
                inputs = batch.dec
            else:
                inputs = batch.trg
            decoder_inputs = inputs[:, :-1].contiguous()
            decoder_masks = self.prepare_masks(inputs[:, 1:]) if masks is None else masks
        else:
            if inputs.ndimension() == 2:
                decoder_inputs = Variable(inputs.data.new(inputs.size(0), 1).fill_(self.field.vocab.stoi['<init>']))
                if inputs.size(1) > 1:
                    decoder_inputs = torch.cat((decoder_inputs, inputs[:, :-1]), dim=1)
            else:
                decoder_inputs = Variable(inputs.data.new(inputs.size(0), 1, inputs.size(2))).fill_(0)
                decoder_inputs[:, (self.field.vocab.stoi['<init>'])] = 1
                if inputs.size(1) > 1:
                    decoder_inputs = torch.cat((decoder_inputs, inputs[:, :-1, :]))
            decoder_masks = self.prepare_masks(inputs) if masks is None else masks
        return decoder_inputs, decoder_masks

    def prepare_targets(self, batch, targets=None, distillation=False, masks=None):
        if targets is None:
            if distillation:
                targets = batch.dec[:, 1:].contiguous()
            else:
                targets = batch.trg[:, 1:].contiguous()
        masks = self.prepare_masks(targets) if masks is None else masks
        return targets, masks

    def prepare_masks(self, inputs):
        if inputs.ndimension() == 2:
            masks = (inputs.data != self.field.vocab.stoi['<pad>']).float()
        else:
            masks = (inputs.data[:, :, (self.field.vocab.stoi['<pad>'])] != 1).float()
        return masks

    def encoding(self, encoder_inputs, encoder_masks):
        return self.encoder(encoder_inputs, encoder_masks)

    def quick_prepare(self, batch, distillation=False, inputs=None, targets=None, input_masks=None, target_masks=None, source_masks=None):
        inputs, input_masks = self.prepare_inputs(batch, inputs, distillation, input_masks)
        targets, target_masks = self.prepare_targets(batch, targets, distillation, target_masks)
        sources, source_masks = self.prepare_sources(batch, source_masks)
        encoding = self.encoding(sources, source_masks)
        return inputs, input_masks, targets, target_masks, sources, source_masks, encoding, inputs.size(0)

    def forward(self, encoding, encoder_masks, decoder_inputs, decoder_masks, decoding=False, beam=1, alpha=0.6, return_probs=False, positions=None, feedback=None):
        if return_probs and decoding or not decoding:
            out = self.decoder(decoder_inputs, encoding, encoder_masks, decoder_masks)
        if decoding:
            if beam == 1:
                output = self.decoder.greedy(encoding, encoder_masks, decoder_masks, feedback=feedback)
            else:
                output = self.decoder.beam_search(encoding, encoder_masks, decoder_masks, beam, alpha)
            if return_probs:
                return output, out, softmax(self.decoder.out(out))
            return output
        if return_probs:
            return out, softmax(self.decoder.out(out))
        return out

    def cost(self, decoder_targets, decoder_masks, out=None):
        decoder_targets, out = mask(decoder_targets, out, decoder_masks.byte())
        logits = self.decoder.out(out)
        loss = F.cross_entropy(logits, decoder_targets)
        return loss

    def batched_cost(self, decoder_targets, decoder_masks, probs, batched=False):
        if decoder_targets.ndimension() == 2:
            loss = -torch.log(probs + TINY).gather(2, decoder_targets[:, :, (None)])[:, :, (0)]
        else:
            loss = -(torch.log(probs + TINY) * decoder_targets).sum(-1)
        return self.apply_mask_cost(loss, decoder_masks, batched)


class FastTransformer(Transformer):

    def __init__(self, src, trg, args):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src, args)
        self.decoder = Decoder(trg, args, causal=False, positional=args.positional_attention, diag=args.diag, windows=args.windows)
        self.field = trg
        self.fertility = args.fertility
        self.alignment = None
        self.hard_inputs = args.hard_inputs
        if self.fertility:
            self.reorderer = ReOrderer(args)
            self.predictor = Fertility(args)
            if not args.old:
                self.offsetor = Fertility(args, L=51)

    def predict_offset(self, outs, masks, truth=None):
        abs_pos = torch.arange(0, masks.size(1))[(None), :].expand_as(masks)
        if outs.is_cuda:
            abs_pos = abs_pos
        abs_pos = Variable(abs_pos)
        positions = self.offsetor(outs, mode='sharp') + abs_pos - 25
        positions.data += (1 - masks) * INF
        if truth is None:
            return positions
        logits = self.offsetor(outs)
        truth = (truth - abs_pos.long() + 25).clamp(0, 50)
        truth, logits = mask(truth, logits, masks.byte())
        loss = F.cross_entropy(logits, truth)
        return loss, positions

    def prepare_initial(self, encoding, source=None, mask_src=None, mask_trg=None, post_fer=None, mode='argmax', N=1, tau=1):
        source_embeddings = encoding[0]
        if self.alignment is None:
            input_embed = source_embeddings
        else:
            input_embed = F.embedding(self.alignment[source.data.view(-1)].view(*source.data.size()), self.decoder.out.weight * math.sqrt(self.decoder.d_model))
        loss = None
        if not self.fertility:
            attention = ReOrderer.linear_attention(mask_src, mask_trg)
            reordering = attention.max(-1)[1]
        else:
            if post_fer is not None and post_fer.size(1) < mask_src.size(1):
                post_fer = torch.cat((post_fer, Variable(post_fer.data.new(post_fer.size(0), mask_src.size(1) - post_fer.size(1)).zero_())), 1)
            if post_fer is not None:
                logits_fer = self.predictor(encoding[-1], mask_src)
                reordering, _, mask_trg = self.predictor.transform(post_fer, mask_src)
                fer_targets, logits_fer = mask(post_fer, logits_fer, mask_src.byte())
                loss = F.cross_entropy(logits_fer, fer_targets.clamp(0, self.predictor.L - 1))
            else:
                logits_fer, pred_fer, (reordering, mask_trg) = self.predictor(encoding[-1], mask_src, mode=mode, N=N, tau=tau, return_samples=True)
        if N > 1:
            B, T, D = source_embeddings.size()
            source = source[:, (None), :].expand(B, N, T).contiguous().view(B * N, T)
            input_embed = input_embed[:, (None), :, :].expand(B, N, T, D).contiguous().view(B * N, T, D)
        source_reordering = self.apply_mask(source.gather(1, reordering), mask_trg)
        if not self.hard_inputs and not self.fertility:
            input_embed = matmul(attention, input_embed)
        else:
            input_embed = input_embed.gather(1, reordering[:, :, (None)].expand(*reordering.size(), input_embed.size(2)))
        if post_fer is None:
            return input_embed, source_reordering, mask_trg, loss, pred_fer
        return input_embed, source_reordering, mask_trg, loss

    def forward(self, encoding, encoder_masks, decoder_inputs, decoder_masks, decoding=False, beam=1, alpha=0.6, return_probs=False, positions=None, feedback=None):
        out = self.decoder(decoder_inputs, encoding, encoder_masks, decoder_masks, input_embeddings=True, positions=positions, feedback=feedback)
        if not decoding:
            if not return_probs:
                return out
            return out, softmax(self.decoder.out(out))
        logits = self.decoder.out(out)
        if beam == 1:
            output = self.apply_mask(logits.max(-1)[1], decoder_masks)
        else:
            output, decoder_masks = topK_search(logits, decoder_masks, N=beam)
            output = self.apply_mask(output, decoder_masks)
        if not return_probs:
            return output
        else:
            return output, out, softmax(logits)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'d_key': 4, 'drop_ratio': 0.5, 'causal': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CosineLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecoderLayer,
     lambda: ([], {'args': _mock_config(d_model=4, n_heads=4, drop_ratio=0.5, use_wo=4, d_hidden=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderLayer,
     lambda: ([], {'args': _mock_config(d_model=4, n_heads=4, drop_ratio=0.5, use_wo=4, d_hidden=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'d_model': 4, 'd_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fertility,
     lambda: ([], {'args': _mock_config(d_model=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MoEHead,
     lambda: ([], {'d_key': 4, 'd_value': 4, 'n_heads': 4, 'drop_ratio': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiHead2,
     lambda: ([], {'d_key': 4, 'd_value': 4, 'n_heads': 4, 'drop_ratio': 0.5}),
     lambda: ([torch.rand([4, 64, 4]), torch.rand([4, 64, 4]), torch.rand([4, 16, 4, 4])], {}),
     False),
]

class Test_salesforce_nonauto_nmt(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

