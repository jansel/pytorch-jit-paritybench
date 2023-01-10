import sys
_module = sys.modules[__name__]
del sys
src = _module
dataset = _module
e3d_lstm = _module
trainer = _module
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


import numpy as np


import torch


import torch.utils.data


from functools import reduce


import copy


import torch.nn as nn


import torch.nn.functional as F


from functools import lru_cache


from torch.utils.data import DataLoader


import math


import matplotlib.pyplot as plt


import torch.nn.init as init


import uuid


class ConvDeconv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode='nearest')


class E3DLSTMCell(nn.Module):

    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()
        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)
        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)
        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)
        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size
        self.layer_norm = nn.LayerNorm(memory_shape)
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)
        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)
        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)
        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)
        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        c_history_flatten = c_history.view(batch_size, -1, channels)
        scores = torch.einsum('bxc,byc->bxy', r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)
        return torch.einsum('bxy,byc->bxc', attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        scaling_factor = 1 / reduce(operator.mul, r.shape[-3:], 1) ** 0.5
        scores = torch.einsum('bctwh,lbctwh->bl', r, c_history) * scaling_factor
        attention = F.softmax(scores, dim=0)
        return torch.einsum('bl,lbctwh->bctwh', attention, c_history)

    def forward(self, x, c_history, m, h):
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))
        recall = self.self_attention_fast(r, c_history)
        c = i * g + self.layer_norm(c_history[-1] + recall)
        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))
        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(LR(self.weight_xo(x) + self.weight_ho(h) + self.weight_co(c) + self.weight_mo(m)))
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        return c_history, m, h

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)
        return c_history, m, h


class E3DLSTM(nn.Module):

    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
        super().__init__()
        self._tau = tau
        self._cells = []
        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            input_shape[0] = hidden_size
            self._cells.append(cell)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input):
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []
        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(batch_size, self._tau, input.device)
                    c_history_states.append(c_history)
                    h_states.append(h)
                c_history, m, h = cell(x, c_history_states[cell_idx], m, h_states[cell_idx])
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                x = h
            outputs.append(h)
        return torch.cat(outputs, dim=1)


class SlidingWindowDataset(torch.utils.data.Dataset):

    def __init__(self, data, window=1, horizon=1, transform=None, dtype=torch.float):
        super().__init__()
        self._data = data
        self._window = window
        self._horizon = horizon
        self._dtype = dtype
        self._transform = transform

    def __getitem__(self, index):
        x = self._data[index:index + self._window]
        y = self._data[index + self._window:index + self._window + self._horizon]
        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)
        if self._transform:
            x = self._transform(x)
            y = self._transform(y)
        return torch.from_numpy(x).type(self._dtype), torch.from_numpy(y).type(self._dtype)

    def __len__(self):
        return self._data.shape[0] - self._window - self._horizon + 1


def h5_virtual_file(filenames, name='data'):
    """
    Assembles a virtual h5 file from multiples
    """
    vsources = []
    total_t = 0
    for path in filenames:
        data = h5py.File(path, 'r').get(name)
        t, *features_shape = data.shape
        total_t += t
        vsources.append(h5py.VirtualSource(path, name, shape=(t, *features_shape)))
    layout = h5py.VirtualLayout(shape=(total_t, *features_shape), dtype=data.dtype)
    cursor = 0
    for vsource in vsources:
        indices = (slice(cursor, cursor + vsource.shape[0]),) + (slice(None),) * (len(vsource.shape) - 1)
        layout[indices] = vsource
        cursor += vsource.shape[0]
    f = h5py.File(f'{uuid.uuid4()}.h5', 'w', libver='latest')
    f.create_virtual_dataset(name, layout)
    return f


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def window(seq, size=2, stride=1):
    """Returns a sliding window (of width n) over data from the iterable
       E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
    """
    it = iter(seq)
    result = []
    for elem in it:
        result.append(elem)
        if len(result) == size:
            yield result
            result = result[stride:]


class TaxiBJTrainer(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float
        self.num_epoch = 100
        self.batch_size = 32
        self.input_time_window = 4
        self.output_time_horizon = 1
        self.temporal_stride = 1
        self.temporal_frames = 2
        self.time_steps = (self.input_time_window - self.temporal_frames + 1) // self.temporal_stride
        input_shape = 2, self.temporal_frames, 32, 32
        output_shape = 2, self.output_time_horizon, 32, 32
        self.tau = 2
        hidden_size = 64
        kernel = 2, 5, 5
        lstm_layers = 4
        self.encoder = E3DLSTM(input_shape, hidden_size, lstm_layers, kernel, self.tau).type(dtype)
        self.decoder = nn.Conv3d(hidden_size * self.time_steps, output_shape[0], kernel, padding=(0, 2, 2)).type(dtype)
        self
        params = self.parameters(recurse=True)
        self.optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0)
        self.apply(weights_init())

    def forward(self, input_seq):
        return self.decoder(self.encoder(input_seq))

    def loss(self, input_seq, target):
        output = self(input_seq)
        l2_loss = F.mse_loss(output * 255, target * 255)
        l1_loss = F.l1_loss(output * 255, target * 255)
        return l1_loss, l2_loss

    @property
    @lru_cache(maxsize=1)
    def data(self):
        taxibj_dir = './data/TaxiBJ/'
        f = h5_virtual_file([f'{taxibj_dir}BJ13_M32x32_T30_InOut.h5', f'{taxibj_dir}BJ14_M32x32_T30_InOut.h5', f'{taxibj_dir}BJ15_M32x32_T30_InOut.h5', f'{taxibj_dir}BJ16_M32x32_T30_InOut.h5'])
        return f.get('data')

    def get_trainloader(self, raw_data, shuffle=True):
        dataset = SlidingWindowDataset(raw_data, self.input_time_window, self.output_time_horizon, lambda t: t / 255)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    def validate(self, val_dataloader):
        self.eval()
        sum_l1_loss = 0
        sum_l2_loss = 0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                frames_seq = []
                for indices in window(range(self.input_time_window), self.temporal_frames, self.temporal_stride):
                    frames_seq.append(input[:, :, indices[0]:indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0)
                target = target
                l1_loss, l2_loss = self.loss(input, target)
                sum_l1_loss += l1_loss
                sum_l2_loss += l2_loss
        None

    def resume_train(self, ckpt_path='./taxibj_trainer.pt', resume=False):
        train_dataloader = self.get_trainloader(self.data[:-672])
        val_dataloader = self.get_trainloader(self.data[-672:], False)
        if resume:
            checkpoint = torch.load(self, ckpt_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            epoch = 0
        while epoch < self.num_epoch:
            epoch += 1
            for i, (input, target) in enumerate(train_dataloader):
                frames_seq = []
                for indices in window(range(self.input_time_window), self.temporal_frames, self.temporal_stride):
                    frames_seq.append(input[:, :, indices[0]:indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0)
                target = target
                self.train()
                self.optimizer.zero_grad()
                l1_loss, l2_loss = self.loss(input, target)
                loss = l1_loss + l2_loss
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    None
            torch.save({'epoch': epoch, 'state_dict': self.state_dict()}, ckpt_path)
            self.validate(val_dataloader)

