import sys
_module = sys.modules[__name__]
del sys
attributes = _module
dataloader = _module
generate = _module
musemorphose = _module
transformer_encoder = _module
transformer_helpers = _module
remi2midi = _module
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


import random


import torch


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import time


from copy import deepcopy


from scipy.stats import entropy


from torch import nn


import torch.nn.functional as F


import math


from torch import optim


from scipy.spatial import distance


def generate_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask


class VAETransformerDecoder(nn.Module):

    def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu', cond_mode='in-attn'):
        super(VAETransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.dropout = dropout
        self.activation = activation
        self.cond_mode = cond_mode
        if cond_mode == 'in-attn':
            self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
        elif cond_mode == 'pre-attn':
            self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)
        self.decoder_layers = nn.ModuleList()
        for i in range(n_layer):
            self.decoder_layers.append(nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation))

    def forward(self, x, seg_emb):
        if not hasattr(self, 'cond_mode'):
            self.cond_mode = 'in-attn'
        attn_mask = generate_causal_mask(x.size(0))
        if self.cond_mode == 'in-attn':
            seg_emb = self.seg_emb_proj(seg_emb)
        elif self.cond_mode == 'pre-attn':
            x = torch.cat([x, seg_emb], dim=-1)
            x = self.seg_emb_proj(x)
        out = x
        for i in range(self.n_layer):
            if self.cond_mode == 'in-attn':
                out += seg_emb
            out = self.decoder_layers[i](out, src_mask=attn_mask)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_pos=20480):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos
        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len, bsz=None):
        pos_encoding = self.pe[:seq_len, :]
        if bsz is not None:
            pos_encoding = pos_encoding.expand(seq_len, bsz, -1)
        return pos_encoding


class TokenEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj):
        super(TokenEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.emb_lookup = nn.Embedding(n_token, d_embed)
        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)
        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)
        return inp_emb.mul_(self.emb_scale)


class VAETransformerEncoder(nn.Module):

    def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
        super(VAETransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout
        self.activation = activation
        self.tr_encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.tr_encoder = nn.TransformerEncoder(self.tr_encoder_layer, n_layer)
        self.fc_mu = nn.Linear(d_model, d_vae_latent)
        self.fc_logvar = nn.Linear(d_model, d_vae_latent)

    def forward(self, x, padding_mask=None):
        out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
        hidden_out = out[0, :, :]
        mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)
        return hidden_out, mu, logvar


def bias_init(bias):
    nn.init.constant_(bias, 0.0)


def weight_init_normal(weight, normal_std):
    nn.init.normal_(weight, 0.0, normal_std)


def weight_init_orthogonal(weight, gain):
    nn.init.orthogonal_(weight, gain)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('GRU') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                weight_init_orthogonal(param, 0.01)
            else:
                bias_init(param)


class MuseMorphose(nn.Module):

    def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent, d_embed, n_token, enc_dropout=0.1, enc_activation='relu', dec_dropout=0.1, dec_activation='relu', d_rfreq_emb=32, d_polyph_emb=32, n_rfreq_cls=8, n_polyph_cls=8, is_training=True, use_attr_cls=True, cond_mode='in-attn'):
        super(MuseMorphose, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation
        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation
        self.d_vae_latent = d_vae_latent
        self.n_token = n_token
        self.is_training = is_training
        self.cond_mode = cond_mode
        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.d_embed = d_embed
        self.pe = PositionalEncoding(d_embed)
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        self.encoder = VAETransformerEncoder(enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation)
        self.use_attr_cls = use_attr_cls
        if use_attr_cls:
            self.decoder = VAETransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_polyph_emb + d_rfreq_emb, dropout=dec_dropout, activation=dec_activation, cond_mode=cond_mode)
        else:
            self.decoder = VAETransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent, dropout=dec_dropout, activation=dec_activation, cond_mode=cond_mode)
        if use_attr_cls:
            self.d_rfreq_emb = d_rfreq_emb
            self.d_polyph_emb = d_polyph_emb
            self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
            self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)
        else:
            self.rfreq_attr_emb = None
            self.polyph_attr_emb = None
        self.emb_dropout = nn.Dropout(self.enc_dropout)
        self.apply(weights_init)

    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        std = torch.exp(0.5 * logvar)
        if use_sampling:
            eps = torch.randn_like(std) * sampling_var
        else:
            eps = torch.zeros_like(std)
        return eps * std + mu

    def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.0):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)
        return vae_latent

    def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        if rfreq_cls is not None and polyph_cls is not None:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb
        out = self.decoder(dec_inp, dec_seg_emb_cat)
        out = self.dec_out_proj(out)
        if keep_last_only:
            out = out[-1, ...]
        return out

    def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None, padding_mask=None):
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)
        dec_token_emb = self.token_emb(dec_inp)
        enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1))
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        vae_latent = self.reparameterize(mu, logvar)
        vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)
        dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent)
        for n in range(dec_inp.size(1)):
            for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
                dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]
        if rfreq_cls is not None and polyph_cls is not None and self.use_attr_cls:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb
        dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
        dec_logits = self.dec_out_proj(dec_out)
        return mu, logvar, dec_logits

    def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
        recons_loss = F.cross_entropy(dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), ignore_index=self.n_token - 1, reduction='mean').float()
        kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()
        return {'beta': beta, 'total_loss': recons_loss + beta * kldiv_loss, 'kldiv_loss': kldiv_loss, 'kldiv_raw': kl_before_free_bits, 'recons_loss': recons_loss}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PositionalEncoding,
     lambda: ([], {'d_embed': 4}),
     lambda: ([0], {}),
     True),
    (VAETransformerDecoder,
     lambda: ([], {'n_layer': 1, 'n_head': 4, 'd_model': 4, 'd_ff': 4, 'd_seg_emb': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (VAETransformerEncoder,
     lambda: ([], {'n_layer': 1, 'n_head': 4, 'd_model': 4, 'd_ff': 4, 'd_vae_latent': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_YatingMusic_MuseMorphose(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

