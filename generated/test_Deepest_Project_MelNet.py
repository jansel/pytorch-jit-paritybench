import sys
_module = sys.modules[__name__]
del sys
wavloader = _module
inference = _module
loss = _module
model = _module
rnn = _module
tier = _module
tts = _module
upsample = _module
text = _module
cleaners = _module
en_numbers = _module
english = _module
ko_dictionary = _module
symbols = _module
trainer = _module
audio = _module
constant = _module
gmm = _module
hparams = _module
plotting = _module
reconstruct = _module
tierutil = _module
train = _module
utils = _module
validation = _module
writer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import random


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Normal


import math


import itertools


class GMMLoss(nn.Module):

    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi, audio_lengths):
        x = nn.utils.rnn.pack_padded_sequence(x.unsqueeze(-1).transpose(1, 2), audio_lengths, batch_first=True, enforce_sorted=False).data
        mu = nn.utils.rnn.pack_padded_sequence(mu.transpose(1, 2), audio_lengths, batch_first=True, enforce_sorted=False).data
        std = nn.utils.rnn.pack_padded_sequence(std.transpose(1, 2), audio_lengths, batch_first=True, enforce_sorted=False).data
        pi = nn.utils.rnn.pack_padded_sequence(pi.transpose(1, 2), audio_lengths, batch_first=True, enforce_sorted=False).data
        log_prob = Normal(loc=mu, scale=std.exp()).log_prob(x)
        log_distrib = log_prob + F.log_softmax(pi, dim=-1)
        loss = -torch.logsumexp(log_distrib, dim=-1).mean()
        return loss


f_div = {(1): 1, (2): 1, (3): 2, (4): 2, (5): 4, (6): 4, (7): 8}


t_div = {(1): 1, (2): 1, (3): 2, (4): 2, (5): 4, (6): 4}


class TierUtil:

    def __init__(self, hp):
        self.hp = hp
        self.n_mels = hp.audio.n_mels
        self.f_div = f_div[hp.model.tier]
        self.t_div = t_div[hp.model.tier]

    def cut_divide_tiers(self, x, tierNo):
        x = x[:, :x.shape[-1] - x.shape[-1] % self.t_div]
        M, T = x.shape
        assert M % self.f_div == 0, 'freq(mel) dimension should be divisible by %d, got %d.' % (self.f_div, M)
        assert T % self.t_div == 0, 'time dimension should be divisible by %d, got %d.' % (self.t_div, T)
        tiers = list()
        for i in range(self.hp.model.tier, max(1, tierNo - 1), -1):
            if i % 2 == 0:
                tiers.append(x[1::2, :])
                x = x[::2, :]
            else:
                tiers.append(x[:, 1::2])
                x = x[:, ::2]
        tiers.append(x)
        if tierNo == 1:
            return tiers[-1], tiers[-1].copy()
        else:
            return tiers[-1], tiers[-2]

    def interleave(self, x, y, tier):
        """
            implements eq. (25)
            x: x^{<g}
            y: x^{g}
            tier: g+1
        """
        assert x.size() == y.size(), 'two inputs for interleave should be identical: got %s, %s' % (x.size(), y.size())
        B, M, T = x.size()
        if tier % 2 == 0:
            temp = x.new_zeros(B, M, 2 * T)
            temp[:, :, 0::2] = x
            temp[:, :, 1::2] = y
        else:
            temp = x.new_zeros(B, 2 * M, T)
            temp[:, 0::2, :] = x
            temp[:, 1::2, :] = y
        return temp


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


class HParam(Dotdict):

    def __init__(self, file):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


def load_hparam_str(hp_str):
    path = os.path.join('temp-restore.yaml')
    with open(path, 'w') as f:
        f.write(hp_str)
    ret = HParam(path)
    os.remove(path)
    return ret


def get_pi_indices(pi):
    cumsum = torch.cumsum(pi.cpu(), dim=-1)
    rand = torch.rand(pi.shape[:-1] + (1,))
    indices = (cumsum < rand).sum(dim=-1)
    return indices.flatten().detach().numpy()


def sample_gmm(mu, std, pi):
    std = std.exp()
    pi = pi.softmax(dim=-1)
    indices = get_pi_indices(pi)
    mu = mu.reshape(-1, mu.shape[-1])
    mu = mu[np.arange(mu.shape[0]), indices].reshape(std.shape[:-1])
    std = std.reshape(-1, std.shape[-1])
    std = std[np.arange(std.shape[0]), indices].reshape(mu.shape)
    return torch.normal(mu, std).reshape_as(mu).clamp(0.0, 1.0).to(mu.device)


EOS = '~'


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([('@' + s) for s in text.split()])


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


_curly_re = re.compile('(.*?)\\{(.+?)\\}(.*)')


PAD = '_'


def sequence_to_text(sequence, skip_eos_and_pad=False, combine_jamo=False):
    """Converts a sequence of IDs back to a string"""
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            if not skip_eos_and_pad or s not in [EOS, PAD]:
                result += s
    result = result.replace('}{', ' ')
    if combine_jamo:
        return jamo_to_korean(result)
    else:
        return result


def _text_to_sequence(text, cleaner_names, as_token):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

        Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

        Returns:
            List of integers corresponding to the symbols in the text
    """
    sequence = []
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
    sequence.append(_symbol_to_id[EOS])
    if as_token:
        return sequence_to_text(sequence, combine_jamo=True)
    else:
        return np.array(sequence, dtype=np.int32)


def text_to_sequence(text, cleaner_names=['korean_cleaners'], as_token=False):
    return _text_to_sequence(text, cleaner_names, as_token)


class MelNet(nn.Module):

    def __init__(self, hp, args, infer_hp):
        super(MelNet, self).__init__()
        self.hp = hp
        self.args = args
        self.infer_hp = infer_hp
        self.f_div = f_div[hp.model.tier + 1]
        self.t_div = t_div[hp.model.tier]
        self.n_mels = hp.audio.n_mels
        self.tierutil = TierUtil(hp)
        if infer_hp.conditional:
            self.tiers = [TTS(hp=hp, freq=hp.audio.n_mels // self.f_div * f_div[1], layers=hp.model.layers[0])] + [Tier(hp=hp, freq=hp.audio.n_mels // self.f_div * f_div[tier], layers=hp.model.layers[tier - 1], tierN=tier) for tier in range(2, hp.model.tier + 1)]
        else:
            self.tiers = [Tier(hp=hp, freq=hp.audio.n_mels // self.f_div * f_div[tier], layers=hp.model.layers[tier - 1], tierN=tier) for tier in range(1, hp.model.tier + 1)]
        self.tiers = nn.ModuleList([None] + [nn.DataParallel(tier) for tier in self.tiers])

    def forward(self, x, tier_num):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num
        return self.tiers[tier_num](x)

    def sample(self, condition):
        x = None
        seq = torch.from_numpy(text_to_sequence(condition)).long().unsqueeze(0)
        input_lengths = torch.LongTensor([seq[0].shape[0]])
        audio_lengths = torch.LongTensor([0])
        tqdm.write('Tier 1')
        for t in tqdm(range(self.args.timestep // self.t_div)):
            audio_lengths += 1
            if x is None:
                x = torch.zeros((1, self.n_mels // self.f_div, 1))
            else:
                x = torch.cat([x, torch.zeros((1, self.n_mels // self.f_div, 1))], dim=-1)
            for m in tqdm(range(self.n_mels // self.f_div)):
                torch.synchronize()
                if self.infer_hp.conditional:
                    mu, std, pi, _ = self.tiers[1](x, seq, input_lengths, audio_lengths)
                else:
                    mu, std, pi = self.tiers[1](x, audio_lengths)
                temp = sample_gmm(mu, std, pi)
                x[:, (m), (t)] = temp[:, (m), (t)]
        for tier in tqdm(range(2, self.hp.model.tier + 1)):
            tqdm.write('Tier %d' % tier)
            mu, std, pi = self.tiers[tier](x)
            temp = sample_gmm(mu, std, pi)
            x = self.tierutil.interleave(x, temp, tier + 1)
        return x

    def load_tiers(self):
        for idx, chkpt_path in enumerate(self.infer_hp.checkpoints):
            checkpoint = torch.load(chkpt_path)
            hp = load_hparam_str(checkpoint['hp_str'])
            if self.hp != hp:
                None
            self.tiers[idx + 1].load_state_dict(checkpoint['model'])


class DelayedRNN(nn.Module):

    def __init__(self, hp):
        super(DelayedRNN, self).__init__()
        self.num_hidden = hp.model.hidden
        self.t_delay_RNN_x = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.t_delay_RNN_yz = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True, bidirectional=True)
        self.c_RNN = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.f_delay_RNN = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.W_t = nn.Linear(3 * self.num_hidden, self.num_hidden)
        self.W_c = nn.Linear(self.num_hidden, self.num_hidden)
        self.W_f = nn.Linear(self.num_hidden, self.num_hidden)

    def flatten_rnn(self):
        self.t_delay_RNN_x.flatten_parameters()
        self.t_delay_RNN_yz.flatten_parameters()
        self.c_RNN.flatten_parameters()
        self.f_delay_RNN.flatten_parameters()

    def forward(self, input_h_t, input_h_f, input_h_c, audio_lengths):
        self.flatten_rnn()
        B, M, T, D = input_h_t.size()
        h_t_x_temp = input_h_t.view(-1, T, D)
        h_t_x_packed = nn.utils.rnn.pack_padded_sequence(h_t_x_temp, audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1), batch_first=True, enforce_sorted=False)
        h_t_x, _ = self.t_delay_RNN_x(h_t_x_packed)
        h_t_x, _ = nn.utils.rnn.pad_packed_sequence(h_t_x, batch_first=True, total_length=T)
        h_t_x = h_t_x.view(B, M, T, D)
        h_t_yz_temp = input_h_t.transpose(1, 2).contiguous()
        h_t_yz_temp = h_t_yz_temp.view(-1, M, D)
        h_t_yz, _ = self.t_delay_RNN_yz(h_t_yz_temp)
        h_t_yz = h_t_yz.view(B, T, M, 2 * D)
        h_t_yz = h_t_yz.transpose(1, 2)
        h_t_concat = torch.cat((h_t_x, h_t_yz), dim=3)
        output_h_t = input_h_t + self.W_t(h_t_concat)
        h_c_temp = nn.utils.rnn.pack_padded_sequence(input_h_c, audio_lengths, batch_first=True, enforce_sorted=False)
        h_c_temp, _ = self.c_RNN(h_c_temp)
        h_c_temp, _ = nn.utils.rnn.pad_packed_sequence(h_c_temp, batch_first=True, total_length=T)
        output_h_c = input_h_c + self.W_c(h_c_temp)
        h_c_expanded = output_h_c.unsqueeze(1)
        h_f_sum = input_h_f + output_h_t + h_c_expanded
        h_f_sum = h_f_sum.transpose(1, 2).contiguous()
        h_f_sum = h_f_sum.view(-1, M, D)
        h_f_temp, _ = self.f_delay_RNN(h_f_sum)
        h_f_temp = h_f_temp.view(B, T, M, D)
        h_f_temp = h_f_temp.transpose(1, 2)
        output_h_f = input_h_f + self.W_f(h_f_temp)
        return output_h_t, output_h_f, output_h_c


class Tier(nn.Module):

    def __init__(self, hp, freq, layers, tierN):
        super(Tier, self).__init__()
        num_hidden = hp.model.hidden
        self.hp = hp
        self.tierN = tierN
        if tierN == 1:
            self.W_t_0 = nn.Linear(1, num_hidden)
            self.W_f_0 = nn.Linear(1, num_hidden)
            self.W_c_0 = nn.Linear(freq, num_hidden)
            self.layers = nn.ModuleList([DelayedRNN(hp) for _ in range(layers)])
        else:
            self.W_t = nn.Linear(1, num_hidden)
            self.layers = nn.ModuleList([UpsampleRNN(hp) for _ in range(layers)])
        self.K = hp.model.gmm
        self.pi_softmax = nn.Softmax(dim=3)
        self.W_theta = nn.Linear(num_hidden, 3 * self.K)

    def forward(self, x, audio_lengths):
        if self.tierN == 1:
            h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
            h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
            h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
            for layer in self.layers:
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)
        else:
            h_f = self.W_t(x.unsqueeze(-1))
            for layer in self.layers:
                h_f = layer(h_f, audio_lengths)
        theta_hat = self.W_theta(h_f)
        mu = theta_hat[(...), :self.K]
        std = theta_hat[(...), self.K:2 * self.K]
        pi = theta_hat[(...), 2 * self.K:]
        return mu, std, pi


class Attention(nn.Module):

    def __init__(self, hp):
        super(Attention, self).__init__()
        self.M = hp.model.gmm
        self.rnn_cell = nn.LSTMCell(input_size=2 * hp.model.hidden, hidden_size=hp.model.hidden)
        self.W_g = nn.Linear(hp.model.hidden, 3 * self.M)

    def attention(self, h_i, memory, ksi):
        phi_hat = self.W_g(h_i)
        ksi = ksi + torch.exp(phi_hat[:, :self.M])
        beta = torch.exp(phi_hat[:, self.M:2 * self.M])
        alpha = F.softmax(phi_hat[:, 2 * self.M:3 * self.M], dim=-1)
        u = memory.new_tensor(np.arange(memory.size(1)), dtype=torch.float)
        u_R = u + 1.5
        u_L = u + 0.5
        term1 = torch.sum(alpha.unsqueeze(-1) * torch.sigmoid((u_R - ksi.unsqueeze(-1)) / beta.unsqueeze(-1)), keepdim=True, dim=1)
        term2 = torch.sum(alpha.unsqueeze(-1) * torch.sigmoid((u_L - ksi.unsqueeze(-1)) / beta.unsqueeze(-1)), keepdim=True, dim=1)
        weights = term1 - term2
        context = torch.bmm(weights, memory)
        termination = 1 - term1.squeeze(1)
        return context, weights, termination, ksi

    def forward(self, input_h_c, memory):
        B, T, D = input_h_c.size()
        context = input_h_c.new_zeros(B, D)
        h_i, c_i = input_h_c.new_zeros(B, D), input_h_c.new_zeros(B, D)
        ksi = input_h_c.new_zeros(B, self.M)
        contexts, weights = [], []
        for i in range(T):
            x = torch.cat([input_h_c[:, (i)], context.squeeze(1)], dim=-1)
            h_i, c_i = self.rnn_cell(x, (h_i, c_i))
            context, weight, termination, ksi = self.attention(h_i, memory, ksi)
            contexts.append(context)
            weights.append(weight)
        contexts = torch.cat(contexts, dim=1) + input_h_c
        alignment = torch.cat(weights, dim=1)
        return contexts, alignment


en_symbols = PAD + EOS + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "


class TTS(nn.Module):

    def __init__(self, hp, freq, layers):
        super(TTS, self).__init__()
        self.hp = hp
        self.W_t_0 = nn.Linear(1, hp.model.hidden)
        self.W_f_0 = nn.Linear(1, hp.model.hidden)
        self.W_c_0 = nn.Linear(freq, hp.model.hidden)
        self.layers = nn.ModuleList([DelayedRNN(hp) for _ in range(layers)])
        self.K = hp.model.gmm
        self.W_theta = nn.Linear(hp.model.hidden, 3 * self.K)
        if self.hp.data.name == 'KSS':
            self.embedding_text = nn.Embedding(len(symbols), hp.model.hidden)
        elif self.hp.data.name == 'Blizzard':
            self.embedding_text = nn.Embedding(len(en_symbols), hp.model.hidden)
        else:
            raise NotImplementedError
        self.text_lstm = nn.LSTM(input_size=hp.model.hidden, hidden_size=hp.model.hidden // 2, batch_first=True, bidirectional=True)
        self.attention = Attention(hp)

    def text_encode(self, text, text_lengths):
        total_length = text.size(1)
        embed = self.embedding_text(text)
        packed = nn.utils.rnn.pack_padded_sequence(embed, text_lengths, batch_first=True, enforce_sorted=False)
        memory, _ = self.text_lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(memory, batch_first=True, total_length=total_length)
        return unpacked

    def forward(self, x, text, text_lengths, audio_lengths):
        memory = self.text_encode(text, text_lengths)
        h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
        h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
        h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) // 2:
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)
            else:
                h_c, alignment = self.attention(h_c, memory)
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)
        theta_hat = self.W_theta(h_f)
        mu = theta_hat[(...), :self.K]
        std = theta_hat[(...), self.K:2 * self.K]
        pi = theta_hat[(...), 2 * self.K:]
        return mu, std, pi, alignment


class UpsampleRNN(nn.Module):

    def __init__(self, hp):
        super(UpsampleRNN, self).__init__()
        self.num_hidden = hp.model.hidden
        self.rnn_x = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True, bidirectional=True)
        self.rnn_y = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True, bidirectional=True)
        self.W = nn.Linear(4 * self.num_hidden, self.num_hidden)

    def flatten_parameters(self):
        self.rnn_x.flatten_parameters()
        self.rnn_y.flatten_parameters()

    def forward(self, inp, audio_lengths):
        self.flatten_parameters()
        B, M, T, D = inp.size()
        inp_temp = inp.view(-1, T, D)
        inp_temp = nn.utils.rnn.pack_padded_sequence(inp_temp, audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn_x(inp_temp)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T)
        x = x.view(B, M, T, 2 * D)
        y, _ = self.rnn_y(inp.transpose(1, 2).contiguous().view(-1, M, D))
        y = y.view(B, T, M, 2 * D).transpose(1, 2).contiguous()
        z = torch.cat([x, y], dim=-1)
        output = inp + self.W(z)
        return output

