import sys
_module = sys.modules[__name__]
del sys
clean = _module
eval_noisy_performance = _module
purge_ckpts = _module
segan = _module
datasets = _module
se_dataset = _module
vc_dataset = _module
models = _module
core = _module
discriminator = _module
generator = _module
model = _module
modules = _module
ops = _module
spectral_norm = _module
utils = _module
select_speakers = _module
train = _module
weightG_fmt_converter = _module

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


import torch.nn as nn


from torch.utils.data import DataLoader


from scipy.io import wavfile


from torch.autograd import Variable


import numpy as np


import random


import matplotlib


import matplotlib.pyplot as plt


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataloader import default_collate


import scipy.io.wavfile as wavfile


from torch.utils.data import Dataset


from torch.nn.parameter import Parameter


from torch.nn.modules import Module


import torch.nn.functional as F


import math


import torch.nn.utils as nnu


from collections import OrderedDict


from torch.nn.utils.spectral_norm import spectral_norm


from random import shuffle


import torch.optim as optim


import torchvision.utils as vutils


from torch.optim import lr_scheduler


from torch import autograd


from scipy import signal


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch import nn


from torch import Tensor


from torch.nn import Parameter


from scipy.linalg import toeplitz


from scipy.signal import lfilter


from scipy.interpolate import interp1d


import re


class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix))
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest': [], 'current': []}
        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val:
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        None
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        None
        latest += [model_path]
        ckpts['latest'] = latest
        ckpts['current'] = model_path
        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))
        st_dict = {'step': step, 'state_dict': self.model.state_dict()}
        if self.optimizer is not None:
            st_dict['optimizer'] = self.optimizer.state_dict()
        torch.save(st_dict, os.path.join(save_path, 'weights_' + model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        None
        if not os.path.exists(ckpt_path):
            None
            return False
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current']
            return curr_ckpt

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is False:
            if not os.path.exists(ckpt_path):
                None
                return False
        else:
            st_dict = torch.load(os.path.join(save_path, 'weights_' + curr_ckpt))
            if 'state_dict' in st_dict:
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                self.model.load_state_dict(st_dict)
            None
            return True

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and k in allowed_keys and v.size() == model_dict[k].size()}
        None
        None
        None
        if len(pt_dict.keys()) != len(model_dict.keys()):
            None
        model_dict.update(pt_dict)
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                None
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])


class Model(nn.Module):

    def __init__(self, name='BaseModel'):
        super().__init__()
        self.name = name
        self.optim = None

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name
        if not hasattr(self, 'saver') and saver is None:
            self.saver = Saver(self, save_path, optimizer=self.optim, prefix=model_name + '-')
        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, optimizer=self.optim, prefix=model_name + '-')
            self.saver.load_weights()
        else:
            None
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False):
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last)

    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class LayerNorm(nn.Module):

    def __init__(self, *args):
        super().__init__()

    def forward(self, activation):
        if len(activation.size()) == 3:
            ori_size = activation.size()
            activation = activation.view(-1, activation.size(-1))
        else:
            ori_size = None
        means = torch.mean(activation, dim=1, keepdim=True)
        stds = torch.std(activation, dim=1, keepdim=True)
        activation = (activation - means) / stds
        if ori_size is not None:
            activation = activation.view(ori_size)
        return activation


class Conv1DResBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth=3, dilations=[1, 2, 4, 8], stride=4, bias=True, transpose=False, act='prelu'):
        super().__init__()
        self.ninputs = ninputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.dilations = dilations
        self.stride = stride
        self.bias = bias
        self.transpose = transpose
        assert dilations[0] == 1, dilations[0]
        assert len(dilations) > 1, len(dilations)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev_in = ninputs
        for n, d in enumerate(dilations):
            if n == 0:
                curr_stride = stride
            else:
                curr_stride = 1
            if n == 0 or n + 1 >= len(dilations):
                curr_fmaps = fmaps
            else:
                curr_fmaps = fmaps // 4
                curr_fmaps = max(curr_fmaps, 1)
            if n == 0 and transpose:
                p_ = (self.kwidth - 4) // 2
                op_ = 0
                if p_ < 0:
                    op_ = p_ * -1
                    p_ = 0
                self.convs.append(nn.ConvTranspose1d(prev_in, curr_fmaps, kwidth, stride=curr_stride, dilation=d, padding=p_, output_padding=op_, bias=bias))
            else:
                self.convs.append(nn.Conv1d(prev_in, curr_fmaps, kwidth, stride=curr_stride, dilation=d, padding=0, bias=bias))
            self.acts.append(nn.PReLU(curr_fmaps))
            prev_in = curr_fmaps

    def forward(self, x):
        h = x
        res_act = None
        for li, layer in enumerate(self.convs):
            if self.stride > 1 and li == 0:
                pad_tuple = self.kwidth // 2 - 1, self.kwidth // 2
            else:
                p_ = (self.kwidth - 1) * self.dilations[li] // 2
                pad_tuple = p_, p_
            if not (self.transpose and li == 0):
                h = F.pad(h, pad_tuple)
            h = layer(h)
            h = self.acts[li](h)
            if li == 0:
                res_act = h
        return h + res_act


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)


class GConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, ret_linear=False):
        if self.stride > 1:
            P = self.kwidth // 2 - 1, self.kwidth // 2
        else:
            P = self.kwidth // 2, self.kwidth // 2
        x_p = F.pad(x, P, mode='reflect')
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, (getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long()), :]
    return x.view(xsize)


def sinc(band, t_right, cuda=False):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)
    ones = torch.ones(1)
    if cuda:
        ones = ones
    y = torch.cat([y_left, ones, y_right])
    return y


class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs, padding='VALID'):
        super(SincConv, self).__init__()
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + fs / 2 / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = fs / 2 - 100
        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding

    def forward(self, x):
        cuda = x.is_cuda
        filters = torch.zeros((self.N_filt, self.Filt_dim))
        N = self.Filt_dim
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs
        if cuda:
            filters = filters
            t_right = t_right
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)
        n = torch.linspace(0, N, steps=N)
        window = (0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right, cuda)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / torch.max(band_pass)
            if cuda:
                band_pass = band_pass
            filters[(i), :] = band_pass * window
        if self.padding == 'SAME':
            x_p = F.pad(x, (self.Filt_dim // 2, self.Filt_dim // 2), mode='reflect')
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim))
        return out


class Discriminator(Model):

    def __init__(self, ninputs, fmaps, kwidth, poolings, pool_type='none', pool_slen=None, norm_type='bnorm', bias=True, phase_shift=None, sinc_conv=False):
        super().__init__(name='Discriminator')
        self.phase_shift = phase_shift
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if pool_slen is None:
            raise ValueError('Please specify D network pool seq len (pool_slen) in the end of the conv stack: [inp_len // (total_pooling_factor)]')
        ninp = ninputs
        if sinc_conv:
            self.sinc_conv = SincConv(fmaps[0] // 2, 251, 16000.0, padding='SAME')
            inp = fmaps[0]
            fmaps = fmaps[1:]
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool) in enumerate(zip(fmaps, poolings), start=1):
            enc_block = GConv1DBlock(ninp, fmap, kwidth, stride=pool, bias=bias, norm_type=norm_type)
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.pool_type = pool_type
        if pool_type == 'none':
            pool_slen *= fmaps[-1]
            self.fc = nn.Sequential(nn.Linear(pool_slen, 256), nn.PReLU(256), nn.Linear(256, 128), nn.PReLU(128), nn.Linear(128, 1))
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc[0])
                torch.nn.utils.spectral_norm(self.fc[2])
                torch.nn.utils.spectral_norm(self.fc[3])
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool_conv)
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(nn.Conv1d(fmaps[-1], fmaps[-1], 1), nn.PReLU(fmaps[-1]), nn.Conv1d(fmaps[-1], 1, 1))
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.mlp[0])
                torch.nn.utils.spectral_norm(self.mlp[1])
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)

    def forward(self, x):
        h = x
        if hasattr(self, 'sinc_conv'):
            h_l, h_r = torch.chunk(h, 2, dim=1)
            h_l = self.sinc_conv(h_l)
            h_r = self.sinc_conv(h_r)
            h = torch.cat((h_l, h_r), dim=1)
        int_act = {}
        for ii, layer in enumerate(self.enc_blocks):
            if self.phase_shift is not None:
                shift = random.randint(1, self.phase_shift)
                right = random.random() > 0.5
                if right:
                    sp1 = h[:, :, :-shift]
                    sp2 = h[:, :, -shift:]
                    h = torch.cat((sp2, sp1), dim=2)
                else:
                    sp1 = h[:, :, :shift]
                    sp2 = h[:, :, shift:]
                    h = torch.cat((sp2, sp1), dim=2)
            h = layer(h)
            int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gmax':
            h = self.gmax(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gavg':
            h = self.gavg(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'mlp':
            y = self.mlp(h)
        int_act['logit'] = y
        return y, int_act


class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0, merge_mode='sum', kwidth=11, bias=True):
        super().__init__()
        self.merge_mode = merge_mode
        if skip_type == 'alpha' or skip_type == 'constant':
            if skip_init == 'zero':
                alpha_ = torch.zeros(size)
            elif skip_init == 'randn':
                alpha_ = torch.randn(size)
            elif skip_init == 'one':
                alpha_ = torch.ones(size)
            else:
                raise TypeError('Unrecognized alpha init scheme: ', skip_init)
            if skip_type == 'alpha':
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
                self.skip_k.requires_grad = False
        elif skip_type == 'conv':
            if kwidth > 1:
                pad = kwidth // 2
            else:
                pad = 0
            self.skip_k = nn.Conv1d(size, size, kwidth, stride=1, padding=pad, bias=bias)
        else:
            raise TypeError('Unrecognized GSkip scheme: ', skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def __repr__(self):
        if self.skip_type == 'alpha':
            return self._get_name() + '(Alpha(1))'
        elif self.skip_type == 'constant':
            return self._get_name() + '(Constant(1))'
        else:
            return super().__repr__()

    def forward(self, hj, hi):
        if self.skip_type == 'conv':
            sk_h = self.skip_k(hj)
        else:
            skip_k = self.skip_k.repeat(hj.size(0), 1, hj.size(2))
            sk_h = skip_k * hj
        if hasattr(self, 'skip_dropout'):
            sk_h = self.skip_dropout(sk_h)
        if self.merge_mode == 'sum':
            return sk_h + hi
        elif self.merge_mode == 'concat':
            return torch.cat((hi, sk_h), dim=1)
        else:
            raise TypeError('Unrecognized skip merge mode: ', self.merge_mode)


class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth, stride=4, bias=True, norm_type=None, act=None):
        super().__init__()
        pad = max(0, (stride - kwidth) // -2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps, kwidth, stride=stride, padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv, fmaps)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.deconv(x)
        if self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h


class Generator(Model):

    def __init__(self, ninputs, fmaps, kwidth, poolings, dec_fmaps=None, dec_kwidth=None, dec_poolings=None, z_dim=None, no_z=False, skip=True, bias=False, skip_init='one', skip_dropout=0, skip_type='alpha', norm_type=None, skip_merge='sum', skip_kwidth=11, name='Generator'):
        super().__init__(name=name)
        self.skip = skip
        self.bias = bias
        self.no_z = no_z
        self.z_dim = z_dim
        self.enc_blocks = nn.ModuleList()
        assert isinstance(fmaps, list), type(fmaps)
        assert isinstance(poolings, list), type(poolings)
        if isinstance(kwidth, int):
            kwidth = [kwidth] * len(fmaps)
        assert isinstance(kwidth, list), type(kwidth)
        skips = {}
        ninp = ninputs
        for pi, (fmap, pool, kw) in enumerate(zip(fmaps, poolings, kwidth), start=1):
            if skip and pi < len(fmaps):
                gskip = GSkip(skip_type, fmap, skip_init, skip_dropout, merge_mode=skip_merge, kwidth=skip_kwidth, bias=bias)
                l_i = pi - 1
                skips[l_i] = {'alpha': gskip}
                setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            enc_block = GConv1DBlock(ninp, fmap, kw, stride=pool, bias=bias, norm_type=norm_type)
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.skips = skips
        if not no_z and z_dim is None:
            z_dim = fmaps[-1]
        if not no_z:
            ninp += z_dim
        if dec_fmaps is None:
            dec_fmaps = fmaps[::-1][1:] + [1]
        else:
            assert isinstance(dec_fmaps, list), type(dec_fmaps)
        if dec_poolings is None:
            dec_poolings = poolings[:]
        else:
            assert isinstance(dec_poolings, list), type(dec_poolings)
        self.dec_poolings = dec_poolings
        if dec_kwidth is None:
            dec_kwidth = kwidth[:]
        elif isinstance(dec_kwidth, int):
            dec_kwidth = [dec_kwidth] * len(dec_fmaps)
        assert isinstance(dec_kwidth, list), type(dec_kwidth)
        self.dec_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(dec_fmaps, dec_poolings, dec_kwidth), start=1):
            if skip and pi > 1 and pool > 1:
                if skip_merge == 'concat':
                    ninp *= 2
            if pi >= len(dec_fmaps):
                act = 'Tanh'
            else:
                act = None
            if pool > 1:
                dec_block = GDeconv1DBlock(ninp, fmap, kw, stride=pool, norm_type=norm_type, bias=bias, act=act)
            else:
                dec_block = GConv1DBlock(ninp, fmap, kw, stride=1, bias=bias, norm_type=norm_type)
            self.dec_blocks.append(dec_block)
            ninp = fmap

    def forward(self, x, z=None, ret_hid=False):
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)
            if self.skip and l_i < len(self.enc_blocks) - 1:
                skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if not self.no_z:
            if z is None:
                z = torch.randn(hi.size(0), self.z_dim, *hi.size()[2:])
                if hi.is_cuda:
                    z = z
            if len(z.size()) != len(hi.size()):
                raise ValueError('len(z.size) {} != len(hi.size) {}'.format(len(z.size()), len(hi.size())))
            if not hasattr(self, 'z'):
                self.z = z
            hi = torch.cat((z, hi), dim=1)
            if ret_hid:
                hall['enc_zc'] = hi
        else:
            z = None
        enc_layer_idx = len(self.enc_blocks) - 1
        for l_i, dec_layer in enumerate(self.dec_blocks):
            if self.skip and enc_layer_idx in self.skips and self.dec_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                hi = skip_conn['alpha'](skip_conn['tensor'], hi)
            hi = dec_layer(hi)
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi


class CombFilter(nn.Module):

    def __init__(self, ninputs, fmaps, L):
        super().__init__()
        self.L = L
        self.filt = nn.Conv1d(ninputs, fmaps, 2, dilation=L, bias=False)
        r_init_weight = torch.ones(ninputs * fmaps, 2)
        r_init_weight[:, (0)] = torch.rand(r_init_weight.size(0))
        self.filt.weight.data = r_init_weight.view(fmaps, ninputs, 2)

    def forward(self, x):
        x_p = F.pad(x, (self.L, 0))
        y = self.filt(x_p)
        return y


class PostProcessingCombNet(nn.Module):

    def __init__(self, ninputs, fmaps, L=[4, 8, 16, 32]):
        super().__init__()
        filts = nn.ModuleList()
        for l in L:
            filt = CombFilter(ninputs, fmaps // len(L), l)
            filts.append(filt)
        self.filts = filts
        self.W = nn.Linear(fmaps, 1, bias=False)

    def forward(self, x):
        hs = []
        for filt in self.filts:
            h = filt(x)
            hs.append(h)
        hs = torch.cat(hs, dim=1)
        y = self.W(hs.transpose(1, 2)).transpose(1, 2)
        return y


def pos_code(chunk_pos, x):
    pos_dim = x.size(1)
    chunk_size = x.size(2)
    bsz = x.size(0)
    pe = torch.zeros(x.size(0), chunk_size, pos_dim)
    for n in range(bsz):
        cpos = chunk_pos[n].item()
        position = torch.arange(chunk_size * cpos, chunk_size * cpos + chunk_size)
        position = position.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2) * -(math.log(10000.0) / pos_dim))
        pe[(n), :, 0::2] = torch.sin(position * div_term)
        pe[(n), :, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(1, 2)
    if x.is_cuda:
        pe = pe
    x = x + pe
    return x


class Generator1D(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth, activations, lnorm=False, dropout=0.0, pooling=2, z_dim=256, z_all=False, skip=True, skip_blacklist=[], dec_activations=None, cuda=False, bias=False, aal=False, wd=0.0, skip_init='one', skip_dropout=0.0, no_tanh=False, aal_out=False, rnn_core=False, linterp=False, mlpconv=False, dec_kwidth=None, no_z=False, skip_type='alpha', num_spks=None, multilayer_out=False, skip_merge='sum', snorm=False, convblock=False, post_skip=False, pos_code=False, satt=False, dec_fmaps=None, up_poolings=None, post_proc=False, out_gate=False, linterp_mode='linear', hidden_comb=False, big_out_filter=False, z_std=1, freeze_enc=False, skip_kwidth=11, pad_type='constant'):
        super().__init__(name='Generator1D')
        self.dec_kwidth = dec_kwidth
        self.skip_kwidth = skip_kwidth
        self.skip = skip
        self.skip_init = skip_init
        self.skip_dropout = skip_dropout
        self.snorm = snorm
        self.z_dim = z_dim
        self.z_all = z_all
        self.pos_code = pos_code
        self.post_skip = post_skip
        self.big_out_filter = big_out_filter
        self.satt = satt
        self.post_proc = post_proc
        self.pad_type = pad_type
        self.onehot = num_spks is not None
        if self.onehot:
            assert num_spks > 0
        self.num_spks = num_spks
        self.no_z = no_z
        self.do_cuda = cuda
        self.wd = wd
        self.no_tanh = no_tanh
        self.skip_blacklist = skip_blacklist
        self.z_std = z_std
        self.freeze_enc = freeze_enc
        self.gen_enc = nn.ModuleList()
        if aal or aal_out:
            from scipy.signal import cheby1
            from scipy.signal import dlti
            from scipy.signal import dimpulse
            system = dlti(*cheby1(8, 0.05, 0.8 / pooling))
            tout, yout = dimpulse(system)
            filter_h = yout[0]
        if aal:
            self.filter_h = filter_h
        else:
            self.filter_h = None
        if dec_kwidth is None:
            dec_kwidth = kwidth
        if isinstance(activations, str):
            if activations != 'glu':
                activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)
        if not isinstance(pooling, list) or len(pooling) == 1:
            pooling = [pooling] * len(enc_fmaps)
        skips = {}
        for layer_idx, (fmaps, pool, act) in enumerate(zip(enc_fmaps, pooling, activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            if self.skip and layer_idx < len(enc_fmaps) - 1:
                if layer_idx not in self.skip_blacklist:
                    l_i = layer_idx
                    gskip = GSkip(skip_type, fmaps, skip_init, skip_dropout, merge_mode=skip_merge, cuda=self.do_cuda, kwidth=self.skip_kwidth)
                    skips[l_i] = {'alpha': gskip}
                    setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act, padding=None, lnorm=lnorm, dropout=dropout, pooling=pool, enc=True, bias=bias, aal_h=self.filter_h, snorm=snorm, convblock=convblock, satt=self.satt, pad_type=pad_type))
        self.skips = skips
        dec_inp = enc_fmaps[-1]
        if dec_fmaps is None:
            if mlpconv:
                dec_fmaps = enc_fmaps[:-1][::-1] + [16, 8, 1]
                None
                up_poolings = [pooling] * (len(dec_fmaps) - 2) + [1] * 3
                add_activations = [nn.PReLU(16), nn.PReLU(8), nn.PReLU(1)]
                raise NotImplementedError('MLPconv is not useful and should be deleted')
            else:
                dec_fmaps = enc_fmaps[:-1][::-1] + [1]
                up_poolings = pooling[::-1]
            None
            self.up_poolings = up_poolings
        else:
            assert up_poolings is not None
            self.up_poolings = up_poolings
        if rnn_core:
            self.z_all = False
            z_all = False
            self.rnn_core = nn.LSTM(dec_inp, dec_inp // 2, bidirectional=True, batch_first=True)
        elif no_z:
            all_z = False
        else:
            dec_inp += z_dim
        self.gen_dec = nn.ModuleList()
        if dec_activations is None:
            dec_activations = [activations[0]] * len(dec_fmaps)
        elif mlpconv:
            dec_activations = dec_activations[:-1]
            dec_activations += add_activations
        enc_layer_idx = len(enc_fmaps) - 1
        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, dec_activations)):
            if skip and layer_idx > 0 and enc_layer_idx not in skip_blacklist and up_poolings[layer_idx] > 1:
                if skip_merge == 'concat':
                    dec_inp *= 2
                None
            if z_all and layer_idx > 0:
                dec_inp += z_dim
            if self.onehot:
                dec_inp += self.num_spks
            if layer_idx >= len(dec_fmaps) - 1:
                if self.no_tanh:
                    act = None
                else:
                    act = nn.Tanh()
                lnorm = False
                dropout = 0
            if up_poolings[layer_idx] > 1:
                pooling = up_poolings[layer_idx]
                self.gen_dec.append(GBlock(dec_inp, fmaps, dec_kwidth, act, padding=0, lnorm=lnorm, dropout=dropout, pooling=pooling, enc=False, bias=bias, linterp=linterp, linterp_mode=linterp_mode, convblock=convblock, comb=hidden_comb, pad_type=pad_type))
            else:
                self.gen_dec.append(GBlock(dec_inp, fmaps, dec_kwidth, act, lnorm=lnorm, dropout=dropout, pooling=1, padding=0, enc=True, bias=bias, convblock=convblock, pad_type=pad_type))
            dec_inp = fmaps
        if aal_out:
            self.aal_out = nn.Conv1d(1, 1, filter_h.shape[0] + 1, stride=1, padding=filter_h.shape[0] // 2, bias=False)
            None
            aal_t = torch.FloatTensor(filter_h).view(1, 1, -1)
            aal_t = torch.cat((aal_t, torch.zeros(1, 1, 1)), dim=-1)
            self.aal_out.weight.data = aal_t
            None
        if post_proc:
            self.comb_net = PostProcessingCombNet(1, 512)
        if out_gate:
            self.out_gate = OutGate(1, 1)
        if big_out_filter:
            self.out_filter = nn.Conv1d(1, 1, 513, padding=513 // 2)

    def forward(self, x, z=None, ret_hid=False, spkid=None, slice_idx=0, att_weight=0):
        if self.num_spks is not None and spkid is None:
            raise ValueError('Please specify spk ID to network to build OH identifier in decoder')
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi, linear_hi = enc_layer(hi, att_weight=att_weight)
            if self.skip and l_i < len(self.gen_enc) - 1:
                if l_i not in self.skip_blacklist:
                    if self.post_skip:
                        skips[l_i]['tensor'] = hi
                    else:
                        skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if hasattr(self, 'rnn_core'):
            self.z_all = False
            if z is None:
                if self.no_z:
                    h0 = Variable(torch.zeros(2, hi.size(0), hi.size(1) // 2))
                else:
                    h0 = Variable(self.z_std * torch.randn(2, hi.size(0), hi.size(1) // 2))
                c0 = Variable(torch.zeros(2, hi.size(0), hi.size(1) // 2))
                if self.do_cuda:
                    h0 = h0
                    c0 = c0
                z = h0, c0
                if not hasattr(self, 'z'):
                    self.z = z
            hi = hi.transpose(1, 2)
            hi, state = self.rnn_core(hi, z)
            hi = hi.transpose(1, 2)
        else:
            if not self.no_z:
                if z is None:
                    z = Variable(self.z_std * torch.randn(hi.size(0), self.z_dim, *hi.size()[2:]))
                if len(z.size()) != len(hi.size()):
                    raise ValueError('len(z.size) {} != len(hi.size) {}'.format(len(z.size()), len(hi.size())))
                if self.do_cuda:
                    z = z
                if not hasattr(self, 'z'):
                    self.z = z
                hi = torch.cat((z, hi), dim=1)
                if ret_hid:
                    hall['enc_zc'] = hi
            else:
                z = None
            if self.pos_code:
                hi = pos_code(slice_idx, hi)
        if self.freeze_enc:
            hi = hi.detach()
        enc_layer_idx = len(self.gen_enc) - 1
        z_up = z
        if self.onehot:
            spk_oh = Variable(torch.zeros(spkid.size(0), self.num_spks))
            for bidx in range(spkid.size(0)):
                if len(spkid.size()) == 3:
                    spk_id = spkid[bidx, 0].cpu().data[0]
                else:
                    spk_id = spkid[bidx].cpu().data[0]
                spk_oh[bidx, spk_id] = 1
            spk_oh = spk_oh.view(spk_oh.size(0), -1, 1)
            if self.do_cuda:
                spk_oh = spk_oh
        for l_i, dec_layer in enumerate(self.gen_dec):
            if self.skip and enc_layer_idx in self.skips and self.up_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                hi = skip_conn['alpha'](skip_conn['tensor'], hi)
            if l_i > 0 and self.z_all:
                z_up = torch.cat((z_up, z_up), dim=2)
                hi = torch.cat((hi, z_up), dim=1)
            if self.onehot:
                spk_oh_r = spk_oh.repeat(1, 1, hi.size(-1))
                hi = torch.cat((hi, spk_oh_r), dim=1)
            hi, _ = dec_layer(hi, att_weight=att_weight)
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if hasattr(self, 'aal_out'):
            hi = self.aal_out(hi)
        if hasattr(self, 'comb_net'):
            hi = F.tanh(self.comb_net(hi))
        if hasattr(self, 'out_gate'):
            hi = self.out_gate(hi)
        if hasattr(self, 'out_filter'):
            hi = self.out_filter(hi)
        if ret_hid:
            return hi, hall
        else:
            return hi

    def batch_minmax_norm(self, x, out_min=-1, out_max=1):
        mins = torch.min(x, dim=2)[0]
        maxs = torch.max(x, dim=2)[0]
        R = (out_max - out_min) / (maxs - mins)
        R = R.unsqueeze(1)
        x = R * (x - mins.unsqueeze(1)) + out_min
        return x

    def skip_merge(self, skip_conn, hi):
        raise NotImplementedError
        hj = skip_conn['tensor']
        alpha = skip_conn['alpha'].view(1, -1, 1)
        alpha = alpha.repeat(hj.size(0), 1, hj.size(2))
        if 'dropout' in skip_conn:
            alpha = skip_conn['dropout'](alpha)
        return hi + alpha * hj


def PESQ(ref_wav, deg_wav):
    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'
    sf.write(ref_tfl, ref_wav, 16000, subtype='PCM_16')
    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')
    curr_dir = os.getcwd()
    try:
        p = run(['pesqmain'.format(curr_dir), ref_tfl, deg_tfl, '+16000', '+wb'], stdout=PIPE, encoding='ascii')
        res_line = p.stdout.split('\n')[-2]
        results = re.split('\\s+', res_line)
        return results[-1]
    except FileNotFoundError:
        None


def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 1e-19))
    winlength = int(np.round(30 * srate / 1000))
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35
    num_frames = int(clean_length / skiprate - winlength / skiprate)
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []
    for frame_count in range(int(num_frames)):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr


def lpcoeff(speech_frame, model_order):
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:winlength - k]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)
    return acorr, refcoeff, lpparams


def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length
    winlength = round(30 * srate / 1000.0)
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        P = 10
    else:
        P = 16
    num_frames = int(clean_length / skiprate - winlength / skiprate)
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []
    for frame_count in range(num_frames):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[(None), :]
        A_processed = A_processed[(None), :]
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        log_ = np.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.array(distortion)


def wss(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length
    winlength = round(30 * srate / 1000.0)
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25
    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2 * winlength) / np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1
    cent_freq = [50.0, 120, 190, 260, 330, 400, 470, 540, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.3, 1288.72, 1442.54, 1610.7, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63]
    bandwidth = [70.0, 70, 70, 70, 70, 70, 70, 77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136]
    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2 * 2.303))
    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = cent_freq[i] / max_freq * n_fftby2
        all_f0.append(np.floor(f0))
        bw = bandwidth[i] / max_freq * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[(i), :] = np.exp(-11 * ((j - np.floor(f0)) / bw) ** 2 + norm_factor)
        crit_filter[(i), :] = crit_filter[(i), :] * (crit_filter[(i), :] > min_factor)
    num_frames = int(clean_length / skiprate - winlength / skiprate)
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []
    for frame_count in range(num_frames):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
        processed_spec = np.abs(np.fft.fft(processed_frame, n_fft)) ** 2
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[(i), :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * crit_filter[(i), :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit - 1]
        processed_slope = processed_energy[1:num_crit] - processed_energy[:num_crit - 1]
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit - 1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit - 1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit - 1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit - 1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - processed_slope[:num_crit - 1]) ** 2))
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def CompositeEval(ref_wav, deg_wav, log_all=False):
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]
    wss_dist_vec = wss(ref_wav, deg_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    LLR_dist = llr(ref_wav, deg_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])
    snr_mean, segsnr_mean = SSNR(ref_wav, deg_wav, 16000)
    segSNR = np.mean(segsnr_mean)
    pesq_raw = PESQ(ref_wav, deg_wav)
    if 'error!' not in pesq_raw:
        pesq_raw = float(pesq_raw)
    else:
        pesq_raw = -1.0

    def trim_mos(val):
        return min(max(val, 1), 5)
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    if log_all:
        return Csig, Cbak, Covl, pesq_raw, segSNR
    else:
        return Csig, Cbak, Covl


def eval_composite(clean_utt, Genh_utt, noisy_utt=None):
    clean_utt = clean_utt.reshape(-1)
    Genh_utt = Genh_utt.reshape(-1)
    csig, cbak, covl, pesq, ssnr = CompositeEval(clean_utt, Genh_utt, True)
    evals = {'csig': csig, 'cbak': cbak, 'covl': covl, 'pesq': pesq, 'ssnr': ssnr}
    if noisy_utt is not None:
        noisy_utt = noisy_utt.reshape(-1)
        csig, cbak, covl, pesq, ssnr = CompositeEval(clean_utt, noisy_utt, True)
        return evals, {'csig': csig, 'cbak': cbak, 'covl': covl, 'pesq': pesq, 'ssnr': ssnr}
    else:
        return evals


def composite_helper(args):
    return eval_composite(*args)


def de_emphasize(y, coef=0.95):
    if coef <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coef * x[n - 1] + y[n]
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        None
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                p.data.normal_(0.0, 0.02)
    elif classname.find('Conv1d') != -1:
        None
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            None
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        None
        nn.init.xavier_uniform_(m.weight.data)


class SEGAN(Model):

    def __init__(self, opts, name='SEGAN', generator=None, discriminator=None):
        super(SEGAN, self).__init__(name)
        self.save_path = opts.save_path
        self.preemph = opts.preemph
        self.reg_loss = getattr(F, opts.reg_loss)
        if generator is None:
            self.G = Generator(1, opts.genc_fmaps, opts.gkwidth, opts.genc_poolings, opts.gdec_fmaps, opts.gdec_kwidth, opts.gdec_poolings, z_dim=opts.z_dim, no_z=opts.no_z, skip=not opts.no_skip, bias=opts.bias, skip_init=opts.skip_init, skip_type=opts.skip_type, skip_merge=opts.skip_merge, skip_kwidth=opts.skip_kwidth)
        else:
            self.G = generator
        self.G.apply(weights_init)
        None
        if discriminator is None:
            dkwidth = opts.gkwidth if opts.dkwidth is None else opts.dkwidth
            self.D = Discriminator(2, opts.denc_fmaps, dkwidth, poolings=opts.denc_poolings, pool_type=opts.dpool_type, pool_slen=opts.dpool_slen, norm_type=opts.dnorm_type, phase_shift=opts.phase_shift, sinc_conv=opts.sinc_conv)
        else:
            self.D = discriminator
        self.D.apply(weights_init)
        None

    def generate(self, inwav, z=None, device='cpu'):
        self.G.eval()
        N = 16384
        x = np.zeros((1, 1, N))
        c_res = None
        slice_idx = torch.zeros(1)
        for beg_i in range(0, inwav.shape[2], N):
            if inwav.shape[2] - beg_i < N:
                length = inwav.shape[2] - beg_i
                pad = N - length
            else:
                length = N
                pad = 0
            if pad > 0:
                x[0, 0] = torch.cat((inwav[(0), (0), beg_i:beg_i + length], torch.zeros(pad)), dim=0)
            else:
                x[0, 0] = inwav[(0), (0), beg_i:beg_i + length]
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            x = x
            canvas_w, hall = self.infer_G(x, z=z, ret_hid=True)
            nums = []
            for k in hall.keys():
                if 'enc' in k and 'zc' not in k:
                    nums.append(int(k.split('_')[1]))
            g_c = hall['enc_{}'.format(max(nums))]
            if z is None and hasattr(self.G, 'z'):
                z = self.G.z
            if pad > 0:
                canvas_w = canvas_w[(0), (0), :-pad]
            canvas_w = canvas_w.data.cpu().numpy().squeeze()
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
            slice_idx += 1
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, g_c

    def discriminate(self, cwav, nwav):
        self.D.eval()
        d_in = torch.cat((cwav, nwav), dim=1)
        d_veredict, _ = self.D(d_in)
        return d_veredict

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        if ret_hid:
            Genh, hall = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh, hall
        else:
            Genh = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh

    def infer_D(self, x_, ref):
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def gen_train_samples(self, clean_samples, noisy_samples, z_sample, iteration=None):
        if z_sample is not None:
            canvas_w = self.infer_G(noisy_samples, clean_samples, z=z_sample)
        else:
            canvas_w = self.infer_G(noisy_samples, clean_samples)
        sample_dif = noisy_samples - clean_samples
        for m in range(noisy_samples.size(0)):
            m_canvas = de_emphasize(canvas_w[m, 0].cpu().data.numpy(), self.preemph)
            None
            wavfile.write(os.path.join(self.save_path, 'sample_{}-{}.wav'.format(iteration, m)), int(16000.0), m_canvas)
            m_clean = de_emphasize(clean_samples[m, 0].cpu().data.numpy(), self.preemph)
            m_noisy = de_emphasize(noisy_samples[m, 0].cpu().data.numpy(), self.preemph)
            m_dif = de_emphasize(sample_dif[m, 0].cpu().data.numpy(), self.preemph)
            m_gtruth_path = os.path.join(self.save_path, 'gtruth_{}.wav'.format(m))
            if not os.path.exists(m_gtruth_path):
                wavfile.write(os.path.join(self.save_path, 'gtruth_{}.wav'.format(m)), int(16000.0), m_clean)
                wavfile.write(os.path.join(self.save_path, 'noisy_{}.wav'.format(m)), int(16000.0), m_noisy)
                wavfile.write(os.path.join(self.save_path, 'dif_{}.wav'.format(m)), int(16000.0), m_dif)

    def build_optimizers(self, opts):
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0, 0.9))
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr, betas=(0, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))
        return Gopt, Dopt

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None, device='cpu'):
        """ Train the SEGAN """
        self.writer = SummaryWriter(os.path.join(self.save_path, 'train'))
        Gopt, Dopt = self.build_optimizers(opts)
        self.G.optim = Gopt
        self.D.optim = Dopt
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3, optimizer=self.D.optim, prefix='EOE_D-')
        num_batches = len(dloader)
        l1_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = 0
        acum_val_obj = 0
        label = torch.ones(opts.batch_size)
        label = label
        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            self.G.train()
            self.D.train()
            for bidx, batch in enumerate(dloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        l1_weight = max(0, l1_weight)
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per sample?'.format(len(sample)))
                clean = clean.unsqueeze(1)
                noisy = noisy.unsqueeze(1)
                label.resize_(clean.size(0)).fill_(1)
                clean = clean
                noisy = noisy
                if noisy_samples is None:
                    noisy_samples = noisy[:20, :, :].contiguous()
                    clean_samples = clean[:20, :, :].contiguous()
                Dopt.zero_grad()
                total_d_fake_loss = 0
                total_d_real_loss = 0
                Genh = self.infer_G(noisy, clean)
                lab = label
                d_real, _ = self.infer_D(clean, noisy)
                d_real_loss = criterion(d_real.view(-1), lab)
                d_real_loss.backward()
                total_d_real_loss += d_real_loss
                d_fake, _ = self.infer_D(Genh.detach(), noisy)
                lab = label.fill_(0)
                d_fake_loss = criterion(d_fake.view(-1), lab)
                d_fake_loss.backward()
                total_d_fake_loss += d_fake_loss
                Dopt.step()
                d_loss = d_fake_loss + d_real_loss
                Gopt.zero_grad()
                lab = label.fill_(1)
                d_fake_, _ = self.infer_D(Genh, noisy)
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                g_l1_loss = l1_weight * self.reg_loss(Genh, clean)
                g_loss = g_adv_loss + g_l1_loss
                g_loss.backward()
                Gopt.step()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if z_sample is None and not self.G.no_z:
                    z_sample = self.G.z[:20, :, :].contiguous()
                    None
                    z_sample = z_sample
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    d_real_loss_v = d_real_loss.cpu().item()
                    d_fake_loss_v = d_fake_loss.cpu().item()
                    g_adv_loss_v = g_adv_loss.cpu().item()
                    g_l1_loss_v = g_l1_loss.cpu().item()
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, d_fake:{:.4f}, '.format(iteration, bidx, len(dloader), epoch, d_real_loss_v, d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f} l1_w: {:.2f}, btime: {:.4f} s, mbtime: {:.4f} s'.format(g_adv_loss_v, g_l1_loss_v, l1_weight, timings[-1], np.mean(timings))
                    None
                    self.writer.add_scalar('D_real', d_real_loss_v, iteration)
                    self.writer.add_scalar('D_fake', d_fake_loss_v, iteration)
                    self.writer.add_scalar('G_adv', g_adv_loss_v, iteration)
                    self.writer.add_scalar('G_l1', g_l1_loss_v, iteration)
                    self.writer.add_histogram('D_fake__hist', d_fake_.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('D_fake_hist', d_fake.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('D_real_hist', d_real.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')

                    def model_weights_norm(model, total_name):
                        total_GW_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    if not opts.no_train_gen:
                        self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
                iteration += 1
            if va_dloader is not None:
                if len(noisy_evals) == 0:
                    evals_, noisy_evals_ = self.evaluate(opts, va_dloader, log_freq, do_noisy=True)
                    for k, v in noisy_evals_.items():
                        if k not in noisy_evals:
                            noisy_evals[k] = []
                        noisy_evals[k] += v
                        self.writer.add_scalar('noisy-{}'.format(k), noisy_evals[k][-1], epoch)
                else:
                    evals_ = self.evaluate(opts, va_dloader, log_freq, do_noisy=False)
                for k, v in evals_.items():
                    if k not in evals:
                        evals[k] = []
                    evals[k] += v
                    self.writer.add_scalar('Genh-{}'.format(k), evals[k][-1], epoch)
                val_obj = evals['covl'][-1] + evals['pesq'][-1] + evals['ssnr'][-1]
                self.writer.add_scalar('Genh-val_obj', val_obj, epoch)
                if val_obj > best_val_obj:
                    None
                    best_val_obj = val_obj
                    patience = opts.patience
                    self.G.save(self.save_path, iteration, True)
                    self.D.save(self.save_path, iteration, True)
                else:
                    patience -= 1
                    None
                    if patience <= 0:
                        None
                        break
            self.G.save(self.save_path, iteration, saver=eoe_g_saver)
            self.D.save(self.save_path, iteration, saver=eoe_d_saver)

    def evaluate(self, opts, dloader, log_freq, do_noisy=False, max_samples=1, device='cpu'):
        """ Objective evaluation with PESQ, SSNR, COVL, CBAK and CSIG """
        self.G.eval()
        self.D.eval()
        evals = {'pesq': [], 'ssnr': [], 'csig': [], 'cbak': [], 'covl': []}
        pesqs = []
        ssnrs = []
        if do_noisy:
            noisy_evals = {'pesq': [], 'ssnr': [], 'csig': [], 'cbak': [], 'covl': []}
            npesqs = []
            nssnrs = []
        if not hasattr(self, 'pool'):
            self.pool = mp.Pool(opts.eval_workers)
        total_s = 0
        timings = []
        with torch.no_grad():
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per sample?'.format(len(sample)))
                clean = clean
                noisy = noisy.unsqueeze(1)
                clean = clean
                noisy = noisy
                Genh = self.infer_G(noisy).squeeze(1)
                clean_npy = clean.cpu().data.numpy()
                Genh_npy = Genh.cpu().data.numpy()
                clean_npy = np.apply_along_axis(de_emphasize, 0, clean_npy, self.preemph)
                Genh_npy = np.apply_along_axis(de_emphasize, 0, Genh_npy, self.preemph)
                beg_t = timeit.default_timer()
                if do_noisy:
                    noisy_npy = noisy.cpu().data.numpy()
                    noisy_npy = np.apply_along_axis(de_emphasize, 0, noisy_npy, self.preemph)
                    args = [(clean_npy[i], Genh_npy[i], noisy_npy[i]) for i in range(clean.size(0))]
                else:
                    args = [(clean_npy[i], Genh_npy[i], None) for i in range(clean.size(0))]
                map_ret = self.pool.map(composite_helper, args)
                end_t = timeit.default_timer()
                None
                if bidx >= max_samples:
                    break

            def fill_ret_dict(ret_dict, in_dict):
                for k, v in in_dict.items():
                    ret_dict[k].append(v)
            if do_noisy:
                for eval_, noisy_eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                    fill_ret_dict(noisy_evals, noisy_eval_)
                return evals, noisy_evals
            else:
                for eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                return evals


def make_divN(tensor, N, method='zeros'):
    pad_num = tensor.size(1) + N - tensor.size(1) % N - tensor.size(1)
    if method == 'zeros':
        pad = torch.zeros(tensor.size(0), pad_num, tensor.size(-1))
        return torch.cat((tensor, pad), dim=1)
    elif method == 'reflect':
        tensor = tensor.transpose(1, 2)
        return F.pad(tensor, (0, pad_num), 'reflect').transpose(1, 2)
    else:
        raise TypeError('Unrecognized make_divN pad method: ', method)


def wsegan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        None
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                nn.init.xavier_uniform_(p.data)
    elif classname.find('Conv1d') != -1:
        None
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('ConvTranspose1d') != -1:
        None
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        None
        nn.init.xavier_uniform_(m.weight.data)


class WSEGAN(SEGAN):

    def __init__(self, opts, name='WSEGAN', generator=None, discriminator=None):
        self.lbd = 1
        self.critic_iters = 1
        self.misalign_pair = opts.misalign_pair
        self.interf_pair = opts.interf_pair
        self.pow_weight = opts.pow_weight
        self.vanilla_gan = opts.vanilla_gan
        self.n_fft = opts.n_fft
        super(WSEGAN, self).__init__(opts, name, None, None)
        self.G.apply(wsegan_weights_init)
        self.D.apply(wsegan_weights_init)

    def sample_dloader(self, dloader, device='cpu'):
        sample = next(dloader.__iter__())
        batch = sample
        uttname, clean, noisy, slice_idx = batch
        clean = clean.unsqueeze(1)
        noisy = noisy.unsqueeze(1)
        clean = clean
        noisy = noisy
        slice_idx = slice_idx
        return uttname, clean, noisy, slice_idx

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        Genh = self.G(nwav, z=z, ret_hid=ret_hid)
        return Genh

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None, device='cpu'):
        """ Train the SEGAN """
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        Gopt, Dopt = self.build_optimizers(opts)
        self.G.optim = Gopt
        self.D.optim = Dopt
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3, optimizer=self.D.optim, prefix='EOE_D-')
        num_batches = len(dloader)
        l1_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = np.inf
        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader, device)
            bsz = clean.size(0)
            Dopt.zero_grad()
            D_in = torch.cat((clean, noisy), dim=1)
            d_real, _ = self.infer_D(clean, noisy)
            rl_lab = torch.ones(d_real.size())
            if self.vanilla_gan:
                cost = F.binary_cross_entropy_with_logits
            else:
                cost = F.mse_loss
            d_real_loss = cost(d_real, rl_lab)
            Genh = self.infer_G(noisy, clean)
            fake = Genh.detach()
            d_fake, _ = self.infer_D(fake, noisy)
            fk_lab = torch.zeros(d_fake.size())
            d_fake_loss = cost(d_fake, fk_lab)
            d_weight = 0.5
            d_loss = d_fake_loss + d_real_loss
            if self.misalign_pair:
                clean_shuf = list(torch.chunk(clean, clean.size(0), dim=0))
                shuffle(clean_shuf)
                clean_shuf = torch.cat(clean_shuf, dim=0)
                d_fake_shuf, _ = self.infer_D(clean, clean_shuf)
                d_fake_shuf_loss = cost(d_fake_shuf, fk_lab)
                d_weight = 1 / 3
                d_loss += d_fake_shuf_loss
            if self.interf_pair:
                freqs = [250, 1000, 4000]
                amps = [0.01, 0.05, 0.1, 1]
                bsz = clean.size(0)
                squares = []
                t = np.linspace(0, 2, 32000)
                for _ in range(bsz):
                    f_ = random.choice(freqs)
                    a_ = random.choice(amps)
                    sq = a_ * signal.square(2 * np.pi * f_ * t)
                    sq = sq[:clean.size(-1)].reshape((1, -1))
                    squares.append(torch.FloatTensor(sq))
                squares = torch.cat(squares, dim=0).unsqueeze(1)
                if clean.is_cuda:
                    squares = squares
                interf = clean + squares
                d_fake_inter, _ = self.infer_D(interf, noisy)
                d_fake_inter_loss = cost(d_fake_inter, fk_lab)
                d_weight = 1 / 4
                d_loss += d_fake_inter_loss
            d_loss = d_weight * d_loss
            d_loss.backward()
            Dopt.step()
            Gopt.zero_grad()
            d_fake_, _ = self.infer_D(Genh, noisy)
            g_adv_loss = cost(d_fake_, torch.ones(d_fake_.size()))
            clean_stft = torch.stft(clean.squeeze(1), n_fft=min(clean.size(-1), self.n_fft), hop_length=160, win_length=320, normalized=True)
            clean_mod = torch.norm(clean_stft, 2, dim=3)
            clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 1e-19)
            Genh_stft = torch.stft(Genh.squeeze(1), n_fft=min(Genh.size(-1), self.n_fft), hop_length=160, win_length=320, normalized=True)
            Genh_mod = torch.norm(Genh_stft, 2, dim=3)
            Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 1e-19)
            pow_loss = self.pow_weight * F.l1_loss(Genh_mod_pow, clean_mod_pow)
            G_cost = g_adv_loss + pow_loss
            if l1_weight > 0:
                mask = torch.zeros(bsz, 1, Genh.size(2))
                if opts.cuda:
                    mask = mask
                for utt_i, uttn in enumerate(uttname):
                    if 'additive' in uttn:
                        mask[(utt_i), (0), :] = 1.0
                den_loss = l1_weight * F.l1_loss(Genh * mask, clean * mask)
                G_cost += den_loss
            else:
                den_loss = torch.zeros(1)
            G_cost.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not self.G.no_z:
                z_sample = self.G.z[:20, :, :].contiguous()
                None
                z_sample = z_sample
            if iteration % log_freq == 0:
                log = 'Iter {}/{} ({} bpe) d_loss:{:.4f}, g_loss: {:.4f}, pow_loss: {:.4f}, den_loss: {:.4f} '.format(iteration, len(dloader) * opts.epoch, len(dloader), d_loss.item(), G_cost.item(), pow_loss.item(), den_loss.item())
                log += 'btime: {:.4f} s, mbtime: {:.4f} s'.format(timings[-1], np.mean(timings))
                None
                self.writer.add_scalar('D_loss', d_loss.item(), iteration)
                self.writer.add_scalar('G_loss', G_cost.item(), iteration)
                self.writer.add_scalar('G_adv_loss', g_adv_loss.item(), iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(), iteration)
                self.writer.add_histogram('clean_mod_pow', clean_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Genh_mod_pow', Genh_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                if hasattr(self.G, 'skips'):
                    for skip_id, alpha in self.G.skips.items():
                        skip = alpha['alpha']
                        if skip.skip_type == 'alpha':
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id), skip.skip_k.data, iteration, bins='sturges')

                def model_weights_norm(model, total_name):
                    total_GW_norm = 0
                    for k, v in model.named_parameters():
                        if 'weight' in k:
                            W = v.data
                            W_norm = torch.norm(W)
                            self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                            total_GW_norm += W_norm
                    self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)
                model_weights_norm(self.G, 'Gtotal')
                model_weights_norm(self.D, 'Dtotal')
                if not opts.no_train_gen:
                    self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
            if iteration % len(dloader) == 0:
                self.G.save(self.save_path, iteration, saver=eoe_g_saver)
                self.D.save(self.save_path, iteration, saver=eoe_d_saver)

    def generate(self, inwav, z=None):
        self.G.eval()
        ori_len = inwav.size(2)
        p_wav = make_divN(inwav.transpose(1, 2), 1024).transpose(1, 2)
        c_res, hall = self.infer_G(p_wav, z=z, ret_hid=True)
        c_res = c_res[(0), (0), :ori_len].cpu().data.numpy()
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, hall


class AEWSEGAN(WSEGAN):
    """ Auto-Encoder model """

    def __init__(self, opts, name='AEWSEGAN', generator=None, discriminator=None):
        super().__init__(opts, name=name, generator=generator, discriminator=discriminator)
        self.D = None

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None, device='cpu'):
        """ Train the SEGAN """
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0.5, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))
        self.G.optim = Gopt
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        num_batches = len(dloader)
        l2_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = np.inf
        acum_val_obj = 0
        G = self.G
        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader, device)
            bsz = clean.size(0)
            Genh = self.infer_G(noisy, clean)
            Gopt.zero_grad()
            if self.l1_loss:
                loss = F.l1_loss(Genh, clean)
            else:
                loss = F.mse_loss(Genh, clean)
            loss.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not G.no_z:
                z_sample = G.z[:20, :, :].contiguous()
                None
                z_sample = z_sample
            if iteration % log_freq == 0:
                clean_stft = torch.stft(clean.squeeze(1), n_fft=min(clean.size(-1), self.n_fft), hop_length=160, win_length=320, normalized=True)
                clean_mod = torch.norm(clean_stft, 2, dim=3)
                clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 1e-19)
                Genh_stft = torch.stft(Genh.detach().squeeze(1), n_fft=min(Genh.size(-1), self.n_fft), hop_length=160, win_length=320, normalized=True)
                Genh_mod = torch.norm(Genh_stft, 2, dim=3)
                Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 1e-19)
                pow_loss = F.l1_loss(Genh_mod_pow, clean_mod_pow)
                log = 'Iter {}/{} ({} bpe) g_l2_loss:{:.4f}, pow_loss: {:.4f}, '.format(iteration, len(dloader) * opts.epoch, len(dloader), loss.item(), pow_loss.item())
                log += 'btime: {:.4f} s, mbtime: {:.4f} s'.format(timings[-1], np.mean(timings))
                None
                self.writer.add_scalar('g_l2/l1_loss', loss.item(), iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(), iteration)
                self.writer.add_histogram('clean_mod_pow', clean_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Genh_mod_pow', Genh_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                if hasattr(G, 'skips'):
                    for skip_id, alpha in G.skips.items():
                        skip = alpha['alpha']
                        if skip.skip_type == 'alpha':
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id), skip.skip_k.data, iteration, bins='sturges')

                def model_weights_norm(model, total_name):
                    total_GW_norm = 0
                    for k, v in model.named_parameters():
                        if 'weight' in k:
                            W = v.data
                            W_norm = torch.norm(W)
                            self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                            total_GW_norm += W_norm
                    self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)
                if not opts.no_train_gen:
                    self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
                if va_dloader is not None:
                    if len(noisy_evals) == 0:
                        sd, nsd = self.evaluate(opts, va_dloader, log_freq, do_noisy=True)
                        self.writer.add_scalar('noisy_SD', nsd, iteration)
                    else:
                        sd = self.evaluate(opts, va_dloader, log_freq, do_noisy=False)
                    self.writer.add_scalar('Genh_SD', sd, iteration)
                    None
                    if sd < best_val_obj:
                        self.G.save(self.save_path, iteration, True)
                        best_val_obj = sd
            if iteration % len(dloader) == 0:
                self.G.save(self.save_path, iteration, saver=eoe_g_saver)


class ResBlock1D(nn.Module):

    def __init__(self, num_inputs, hidden_size, kwidth, dilation=1, bias=True, norm_type=None, hid_act=nn.ReLU(inplace=True), out_act=None, skip_init=0):
        super().__init__()
        self.entry_conv = nn.Conv1d(num_inputs, hidden_size, 1, bias=bias)
        self.entry_norm = build_norm_layer(norm_type, self.entry_conv, hidden_size)
        self.entry_act = hid_act
        self.mid_conv = nn.Conv1d(hidden_size, hidden_size, kwidth, dilation=dilation, bias=bias)
        self.mid_norm = build_norm_layer(norm_type, self.mid_conv, hidden_size)
        self.mid_act = hid_act
        self.exit_conv = nn.Conv1d(hidden_size, num_inputs, 1, bias=bias)
        self.exit_norm = build_norm_layer(norm_type, self.exit_conv, num_inputs)
        if out_act is None:
            out_act = hid_act
        self.exit_act = out_act
        self.kwidth = kwidth
        self.dilation = dilation
        self.skip_alpha = nn.Parameter(torch.FloatTensor([skip_init]))

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.entry_conv(x)
        h = self.forward_norm(h, self.entry_norm)
        h = self.entry_act(h)
        kw_2 = self.kwidth // 2
        P = kw_2 + kw_2 * (self.dilation - 1)
        h_p = F.pad(h, (P, P), mode='reflect')
        h = self.mid_conv(h_p)
        h = self.forward_norm(h, self.mid_norm)
        h = self.mid_act(h)
        h = self.exit_conv(h)
        h = self.forward_norm(h, self.exit_norm)
        y = self.exit_act(self.skip_alpha * x + h)
        return y


class ResARModule(nn.Module):

    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, bias=True, norm_type=None, act=None):
        super().__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation, bias=bias)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1, bias=bias)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.conv_1x1_skip, ninp)
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1, bias=bias)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.conv_1x1_res, res_fmaps)

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        kw__1 = self.kwidth - 1
        P = kw__1 + kw__1 * (self.dilation - 1)
        x_p = F.pad(x, (P, 0))
        h = self.dil_conv(x_p)
        h = self.forward_norm(h, self.dil_norm)
        h = self.act(h)
        a = h
        h = self.conv_1x1_skip(h)
        h = self.forward_norm(h, self.conv_1x1_skip_norm)
        y = x + h
        sh = self.conv_1x1_res(a)
        sh = self.forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CombFilter,
     lambda: ([], {'ninputs': 4, 'fmaps': 4, 'L': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (Conv1DResBlock,
     lambda: ([], {'ninputs': 4, 'fmaps': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (GConv1DBlock,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'kwidth': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GDeconv1DBlock,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'kwidth': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (LayerNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PostProcessingCombNet,
     lambda: ([], {'ninputs': 4, 'fmaps': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ResARModule,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'res_fmaps': 4, 'kwidth': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
]

class Test_santi_pdp_segan_pytorch(_paritybench_base):
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

