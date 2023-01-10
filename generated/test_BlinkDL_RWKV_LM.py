import sys
_module = sys.modules[__name__]
del sys
src = _module
model = _module
trainer = _module
utils = _module
train = _module
run = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
run = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
verify = _module
run = _module
binidx = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
verify = _module
img_demoAE = _module
run = _module
binidx = _module
dataset = _module
model = _module
model_img = _module
model_run = _module
trainer = _module
utils = _module
train = _module

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


import math


import logging


import torch


import torch.nn as nn


from torch.nn import functional as F


import numpy as np


import torch.optim as optim


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data.dataloader import DataLoader


import random


import time


from torch.utils.data import Dataset


import types


import copy


from torch.utils.cpp_extension import load


from functools import lru_cache


from itertools import accumulate


import torchvision as vision


import torchvision.transforms as transforms


import torch.nn.functional as F


from typing import List


from typing import Dict


from torch.utils.data import DataLoader


def __nop(ob):
    return ob


MyFunction = __nop


MyModule = torch.jit.ScriptModule


class WKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device=gy.device).contiguous()
        gu = torch.zeros((B, C), device=gy.device).contiguous()
        gk = torch.zeros((B, T, C), device=gy.device).contiguous()
        gv = torch.zeros((B, T, C), device=gy.device).contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return None, None, None, gw, gu, gk, gv
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return None, None, None, gw.half(), gu.half(), gk.half(), gv.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16()


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


class RWKV_TimeMix(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd
        attn_sz = args.n_embd
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            zigzag = torch.tensor([((i + 1) % 3 - 1) for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, args.n_embd, bias=False)

    @MyFunction
    def jit_func(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        hidden_sz = 4 * args.n_embd
        self.key = nn.Linear(args.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class RWKV_TinyAttn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_attn = config.rwkv_tiny_attn
        self.n_head = config.rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head
        self.qkv = nn.Linear(config.n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, config.n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        qk = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_size))
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v
        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(qkv)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MHA_rotary(nn.Module):

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.register_buffer('mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)
        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        x = att @ v
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


class MHA_pro(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head
        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.register_buffer('mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)
        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)
        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att * w
        att = self.head_mix(att)
        x = att @ v
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        x = self.output(x) * self.time_gamma[:T, :]
        return x


class RMSNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed


class FixedNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed


class Block(nn.Module):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1, args.my_pos_emb, args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb, 1, args.n_embd)))
        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)
        self.ffn = RWKV_ChannelMix(args, layer_id)
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer('tiny_mask', torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = q @ k.transpose(-2, -1) * args.tiny_att_dim ** -0.5
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 0.0001 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy


RWKV_HEAD_QK_DIM = 0


def RWKV_Init(model, args):
    None
    None
    for mm in model.modules():
        if 'RecursiveScriptModule' in str(type(mm)):
            if mm.original_name not in ['Linear']:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == 'weight':
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in model.named_parameters():
                if id(ww) == id(parameter):
                    break
            shape = ww.shape
            gain = 1.0
            scale = 1.0
            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:
                    scale = 0.0001
                else:
                    scale = 0
            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:
                    scale = 0.5
            if hasattr(m, 'scale_init'):
                scale = m.scale_init
            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


logger = logging.getLogger(__name__)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer('copy_mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.ctx_len = config.ctx_len
        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config)
        except:
            pass
        logger.info('number of parameters: %e', sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1e-05)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [{'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}]
        try:
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            None
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, idx, targets=None):
        idx = idx
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, 'Cannot forward, because len(input) > model ctx_len.'
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = q @ k.transpose(-2, -1) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()
            x = self.head(x) + c
        else:
            x = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, x)


class RWKV_GPT(nn.Module):

    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, vocab_size, n_layer, n_embd, ctx_len):
        global RWKV_CFG
        super().__init__()
        RWKV_CFG.RUN_DEVICE = RUN_DEVICE
        RWKV_CFG.model_type = model_type
        RWKV_CFG.vocab_size = vocab_size
        RWKV_CFG.n_layer = n_layer
        RWKV_CFG.n_embd = n_embd
        RWKV_CFG.ctx_len = ctx_len
        None
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer('copy_mask', torch.tril(torch.ones(ctx_len, ctx_len)))
        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))
        self.eval()

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.ctx_len, 'Cannot forward, because len(input) > model ctx_len.'
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = q @ k.transpose(-2, -1) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).bfloat16()
            x = self.head(x) + c
        else:
            x = self.head(x)
        return x


class R_ENCODER(MyModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.Bxx = nn.BatchNorm2d(dd * 64)
        self.CIN = nn.Conv2d(3, dd, kernel_size=3, padding=1)
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)
        self.B00 = nn.BatchNorm2d(dd * 4)
        self.C00 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.B10 = nn.BatchNorm2d(dd * 16)
        self.C10 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.B20 = nn.BatchNorm2d(dd * 64)
        self.C20 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.COUT = nn.Conv2d(dd * 64, args.my_img_bit, kernel_size=3, padding=1)

    @MyFunction
    def forward(self, img):
        ACT = F.mish
        x = self.CIN(img)
        xx = self.Bxx(F.pixel_unshuffle(x, 8))
        x = x + self.Cx1(ACT(self.Cx0(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))
        x = self.COUT(x + xx)
        return torch.sigmoid(x)


class R_DECODER(MyModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.CIN = nn.Conv2d(args.my_img_bit, dd * 64, kernel_size=3, padding=1)
        self.B00 = nn.BatchNorm2d(dd * 64)
        self.C00 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.B10 = nn.BatchNorm2d(dd * 16)
        self.C10 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.B20 = nn.BatchNorm2d(dd * 4)
        self.C20 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)
        self.COUT = nn.Conv2d(dd, 3, kernel_size=3, padding=1)

    @MyFunction
    def forward(self, code):
        ACT = F.mish
        x = self.CIN(code)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.Cx1(ACT(self.Cx0(x)))
        x = self.COUT(x)
        return torch.sigmoid(x)


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class DISTS(torch.nn.Module):

    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = vision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_buffer('alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_buffer('beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        weights = torch.load('test/DISTS_weights.pt')
        self.alpha.data = weights['alpha']
        self.beta.data = weights['beta']
        for param in self.parameters():
            param.requires_grad = False

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-06
        c2 = 1e-06
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)
            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score


    class ToBinary(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            return torch.floor(x + 0.5)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()


DEBUG_TIME = False


RWKV_RESCALE_LAYER = 6


class RWKV_RNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE
        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(args.ctx_len + 1, -1)[:-1, :]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / 2 ** int(block_id // RWKV_RESCALE_LAYER)
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / 2 ** int(block_id // RWKV_RESCALE_LAYER)
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        None
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                elif self.FLOAT_MODE == 'fp32':
                    w[x] = w[x].float()
                elif self.FLOAT_MODE == 'bf16':
                    w[x] = w[x].bfloat16()
                elif self.FLOAT_MODE == 'fp16':
                    w[x] = w[x].half()
                w[x].requires_grad = False
                if args.RUN_DEVICE == 'cuda' and x != 'emb.weight':
                    w[x] = w[x]
                if 'blocks.' not in x or 'blocks.0.' in x:
                    if print_need_newline:
                        None
                        print_need_newline = False
                    None
                else:
                    print_need_newline = True
                    None
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i + 1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])
        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def FF(self, x, state, i, time_mix_k, time_mix_r, kw, vw, rw):
        if self.FLOAT_MODE == 'bf16':
            xk = x * time_mix_k + state[5 * i + 0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + state[5 * i + 0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5 * i + 0] = x.float()
        elif self.FLOAT_MODE == 'fp16':
            xk = x * time_mix_k + state[5 * i + 0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[5 * i + 0].half() * (1 - time_mix_r)
            state[5 * i + 0] = x.float()
        else:
            xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
            state[5 * i + 0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k
        return r * kv

    @MyFunction
    def SA(self, x, state, i, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        if self.FLOAT_MODE == 'bf16':
            xk = x * time_mix_k + state[5 * i + 1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + state[5 * i + 1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + state[5 * i + 1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5 * i + 1] = x.float()
        elif self.FLOAT_MODE == 'fp16':
            xk = x * time_mix_k + state[5 * i + 1].half() * (1 - time_mix_k)
            xv = x * time_mix_v + state[5 * i + 1].half() * (1 - time_mix_v)
            xr = x * time_mix_r + state[5 * i + 1].half() * (1 - time_mix_r)
            state[5 * i + 1] = x.float()
        else:
            xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
            state[5 * i + 1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv
        if '16' in self.FLOAT_MODE:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v
        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5 * i + 2] = e1 * aa + e2 * vv
        state[5 * i + 3] = e1 * bb + e2
        state[5 * i + 4] = p
        if self.FLOAT_MODE == 'bf16':
            wkv = (a / b).type(torch.bfloat16)
        elif self.FLOAT_MODE == 'fp16':
            wkv = (a / b).half()
        else:
            wkv = a / b
        return ow @ (r * wkv)

    def forward(self, ctx, state, preprocess_only=False):
        with torch.no_grad():
            w = self.w
            args = self.args
            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x
            try:
                pos_emb = w.pos_emb[len(ctx) - 1]
                x = x + pos_emb
            except:
                pass
            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5 * i + 4] -= 1e+30
            for i in range(args.n_layer):
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)
                ww = w.blocks[i].att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i, ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                ww = w.blocks[i].ffn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i, ww.time_mix_k, ww.time_mix_r, ww.key.weight, ww.value.weight, ww.receptance.weight)
                if (i + 1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2
            if preprocess_only:
                return state
            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x
            return x.float(), state


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FixedNorm,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeGLU,
     lambda: ([], {'config': _mock_config(n_ffn=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MHA_pro,
     lambda: ([], {'config': _mock_config(n_attn=4, n_head=4, ctx_len=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MHA_rotary,
     lambda: ([], {'config': _mock_config(n_attn=4, n_head=4, ctx_len=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RMSNorm,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RWKV_ChannelMix,
     lambda: ([], {'args': _mock_config(n_layer=1, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RWKV_TinyAttn,
     lambda: ([], {'config': _mock_config(rwkv_tiny_attn=4, rwkv_tiny_head=4, n_embd=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (R_DECODER,
     lambda: ([], {'args': _mock_config(my_img_bit=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (R_ENCODER,
     lambda: ([], {'args': _mock_config(my_img_bit=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_BlinkDL_RWKV_LM(_paritybench_base):
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

