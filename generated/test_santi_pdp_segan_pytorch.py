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


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


from scipy.io import wavfile


from torch.autograd import Variable


import numpy as np


import random


from torch.nn.parameter import Parameter


from torch.nn.modules import Module


import torch.nn.functional as F


import math


import torch.nn.utils as nnu


from collections import OrderedDict


from torch.nn.utils.spectral_norm import spectral_norm


from random import shuffle


import torch.optim as optim


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

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''
        ):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix)
            )
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
                        print('Removing old ckpt {}'.format(os.path.join(
                            save_path, 'weights_' + todel)))
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')
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
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
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
                print('[!] No weights to be loaded')
                return False
        else:
            st_dict = torch.load(os.path.join(save_path, 'weights_' +
                curr_ckpt))
            if 'state_dict' in st_dict:
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file, map_location=lambda storage, loc:
            storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and 
            k in allowed_keys and v.size() == model_dict[k].size()}
        print('Current Model keys: ', len(list(model_dict.keys())))
        print('Loading Pt Model keys: ', len(list(pt_dict.keys())))
        print('Loading matching keys: ', list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            print('WARNING: LOADING DIFFERENT NUM OF KEYS')
        model_dict.update(pt_dict)
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
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
            self.saver = Saver(self, save_path, optimizer=self.optim,
                prefix=model_name + '-')
        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, optimizer=self.optim,
                    prefix=model_name + '-')
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

    def __init__(self, ninputs, fmaps, kwidth=3, dilations=[1, 2, 4, 8],
        stride=4, bias=True, transpose=False, act='prelu'):
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
                self.convs.append(nn.ConvTranspose1d(prev_in, curr_fmaps,
                    kwidth, stride=curr_stride, dilation=d, padding=p_,
                    output_padding=op_, bias=bias))
            else:
                self.convs.append(nn.Conv1d(prev_in, curr_fmaps, kwidth,
                    stride=curr_stride, dilation=d, padding=0, bias=bias))
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


class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0,
        merge_mode='sum', kwidth=11, bias=True):
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
            self.skip_k = nn.Conv1d(size, size, kwidth, stride=1, padding=
                pad, bias=bias)
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


class ResBlock1D(nn.Module):

    def __init__(self, num_inputs, hidden_size, kwidth, dilation=1, bias=
        True, norm_type=None, hid_act=nn.ReLU(inplace=True), out_act=None,
        skip_init=0):
        super().__init__()
        self.entry_conv = nn.Conv1d(num_inputs, hidden_size, 1, bias=bias)
        self.entry_norm = build_norm_layer(norm_type, self.entry_conv,
            hidden_size)
        self.entry_act = hid_act
        self.mid_conv = nn.Conv1d(hidden_size, hidden_size, kwidth,
            dilation=dilation, bias=bias)
        self.mid_norm = build_norm_layer(norm_type, self.mid_conv, hidden_size)
        self.mid_act = hid_act
        self.exit_conv = nn.Conv1d(hidden_size, num_inputs, 1, bias=bias)
        self.exit_norm = build_norm_layer(norm_type, self.exit_conv, num_inputs
            )
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


class GConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, norm_type=None
        ):
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


class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth, stride=4, bias=True, norm_type=
        None, act=None):
        super().__init__()
        pad = max(0, (stride - kwidth) // -2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps, kwidth, stride=stride,
            padding=pad)
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


class ResARModule(nn.Module):

    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, bias=True,
        norm_type=None, act=None):
        super().__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation,
            bias=bias)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1, bias=bias)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.
            conv_1x1_skip, ninp)
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1, bias=bias)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.
            conv_1x1_res, res_fmaps)

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


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, (getattr(torch.arange(x.size(1) -
        1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long()), :]
    return x.view(xsize)


def sinc(band, t_right, cuda=False):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band *
        t_right)
    y_left = flip(y_right, 0)
    ones = torch.ones(1)
    if cuda:
        ones = ones.to('cuda')
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
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.
            freq_scale))
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding

    def forward(self, x):
        cuda = x.is_cuda
        filters = torch.zeros((self.N_filt, self.Filt_dim))
        N = self.Filt_dim
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)
            ) / self.fs
        if cuda:
            filters = filters
            t_right = t_right
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + 
            min_band / self.freq_scale)
        n = torch.linspace(0, N, steps=N)
        window = (0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / torch.max(band_pass)
            if cuda:
                band_pass = band_pass
            filters[(i), :] = band_pass * window
        if self.padding == 'SAME':
            x_p = F.pad(x, (self.Filt_dim // 2, self.Filt_dim // 2), mode=
                'reflect')
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim))
        return out


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
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data),
                u.data))
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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_santi_pdp_segan_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(CombFilter(*[], **{'ninputs': 4, 'fmaps': 4, 'L': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Conv1DResBlock(*[], **{'ninputs': 4, 'fmaps': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GConv1DBlock(*[], **{'ninp': 4, 'fmaps': 4, 'kwidth': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(GDeconv1DBlock(*[], **{'ninp': 4, 'fmaps': 4, 'kwidth': 4}), [torch.rand([4, 4, 64])], {})

    def test_004(self):
        self._check(LayerNorm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(PostProcessingCombNet(*[], **{'ninputs': 4, 'fmaps': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ResARModule(*[], **{'ninp': 4, 'fmaps': 4, 'res_fmaps': 4, 'kwidth': 4, 'dilation': 1}), [torch.rand([4, 4, 64])], {})

