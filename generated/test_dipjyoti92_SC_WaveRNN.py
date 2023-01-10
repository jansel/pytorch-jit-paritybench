import sys
_module = sys.modules[__name__]
del sys
gen_tacotron = _module
gen_wavernn = _module
hparams = _module
deepmind_version = _module
fatchord_version = _module
tacotron = _module
preprocess = _module
train_tacotron = _module
train_wavernn = _module
utils = _module
dataset = _module
display = _module
distribution = _module
dsp = _module
files = _module
paths = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
recipes = _module
symbols = _module

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


import torch.nn.functional as F


import numpy as np


from torch import optim


import time


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data.sampler import Sampler


device = torch.device('cuda')


class Spk_UpsampleNetwork(nn.Module):

    def forward(self, s_e, dim):
        s = s_e.unsqueeze(dim=1).detach().cpu().numpy()
        s = np.tile(s, (dim, 1))
        s = torch.from_numpy(s).float()
        return s


class ResBlock(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):

    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):

    def __init__(self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = 1, scale * 2 + 1
            padding = 0, scale
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.0) - 1.0


def decode_mu_law(y, mu, from_labels=True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def progbar(i, n, size=16):
    done = i * size // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def stream(message):
    sys.stdout.write(f'\r{message}')


class WaveRNN(nn.Module):

    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors, feat_dims, compute_dims, res_out_dims, res_blocks, hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError('Unknown model mode value - ', self.mode)
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.spk_upsample = Spk_UpsampleNetwork()
        self.I = nn.Linear(feat_dims + self.aux_dims + 256 + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def forward(self, x, mels, spk_embd):
        self.step += 1
        bsize, up_dim = x.size(0), x.size(1)
        h1 = torch.zeros(1, bsize, self.rnn_dims)
        h2 = torch.zeros(1, bsize, self.rnn_dims)
        mels, aux = self.upsample(mels)
        s_e = self.spk_upsample(spk_embd, up_dim)
        aux_idx = [(self.aux_dims * i) for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        x = torch.cat([x.unsqueeze(-1), mels, a1, s_e], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, spk_embd, save_path, batched, target, overlap, mu_law):
        mu_law = mu_law if self.mode == 'RAW' else False
        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        with torch.no_grad():
            mels = mels
            spk_embd = spk_embd
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))
            s_e = self.spk_upsample(spk_embd, mels.size(1))
            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)
                s_e = self.fold_with_overlap(s_e, target, overlap)
            b_size, seq_len, _ = mels.size()
            h1 = torch.zeros(b_size, self.rnn_dims)
            h2 = torch.zeros(b_size, self.rnn_dims)
            x = torch.zeros(b_size, 1)
            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]
            for i in range(seq_len):
                m_t = mels[:, i, :]
                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)
                s_e_t = s_e[:, i, :]
                x = torch.cat([x, m_t, a1_t, s_e_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                logits = self.fc3(x)
                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1)
                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1.0
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError('Unknown model mode value - ', self.mode)
                if i % 100 == 0:
                    self.gen_display(i, seq_len, b_size, start)
        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]
        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        save_wav(output, save_path)
        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i * b_size}/{seq_len * b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        """ Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """
        _, total_len, features = x.size()
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')
        folded = torch.zeros(num_folds, target + 2 * overlap, features)
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]
        return folded

    def xfade_and_unfold(self, y, target, overlap):
        """ Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        """
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros(silence_len, dtype=np.float64)
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out
        unfolded = np.zeros(total_len, dtype=np.float64)
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]
        return unfolded

    def get_step(self):
        return self.step.data.item()

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            None

    def restore(self, path):
        if not os.path.exists(path):
            None
            self.save(path)
        else:
            None
            self.load(path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            None


class HighwayNetwork(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.0)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1.0 - g) * x
        return y


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):

    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)
        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, x):
        residual = x
        seq_len = x.size(-1)
        conv_bank = []
        for conv in self.conv1d_bank:
            c = conv(x)
            conv_bank.append(c[:, :, :seq_len])
        conv_bank = torch.cat(conv_bank, dim=1)
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = x + residual
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)
        x, _ = self.rnn(x)
        return x


class PreNet(nn.Module):

    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class Encoder(nn.Module):

    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.spk_upsample = Spk_UpsampleNetwork()
        self.pre_net = PreNet(embed_dims * 2)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels, proj_channels=[cbhg_channels, cbhg_channels], num_highways=num_highways)

    def forward(self, x, s_e):
        x = self.embedding(x)
        _, dim, _ = x.shape
        s_embd = self.spk_upsample(s_e, dim)
        x = torch.cat([x, s_embd], dim=2)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        return x


class Attention(nn.Module):

    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):
        query_proj = self.W(query).unsqueeze(1)
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)
        return scores.transpose(1, 2)


class LSA(nn.Module):

    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t)
        self.attention = torch.zeros(b, t)

    def forward(self, encoder_seq_proj, query, t):
        if t == 0:
            self.init_attention(encoder_seq_proj)
        processed_query = self.W(query).unsqueeze(1)
        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))
        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        self.attention = scores
        self.cumulative += self.attention
        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__()
        self.max_r = 20
        self.r = None
        self.generating = False
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)

    def zoneout(self, prev, current, p=0.1):
        mask = torch.zeros(prev.size()).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t):
        batch_size = encoder_seq.size(0)
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states
        prenet_out = self.prenet(prenet_in)
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        context_vec = scores @ encoder_seq
        context_vec = context_vec.squeeze(1)
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if not self.generating:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if not self.generating:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = attn_hidden, rnn1_hidden, rnn2_hidden
        cell_states = rnn1_cell, rnn2_cell
        return mels, scores, hidden_states, cell_states, context_vec


class Tacotron(nn.Module):

    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways, dropout):
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, decoder_dims, lstm_dims)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)
        self.init_model()
        self.num_params()
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.r = nn.Parameter(torch.tensor(0).long(), requires_grad=False)

    def set_r(self, r):
        self.r.data = torch.tensor(r)
        self.decoder.r = r

    def get_r(self):
        return self.r.item()

    def forward(self, x, m, s_e, generate_gta=False):
        self.step += 1
        if generate_gta:
            self.encoder.eval()
            self.postnet.eval()
            self.decoder.generating = True
        else:
            self.encoder.train()
            self.postnet.train()
            self.decoder.generating = False
        batch_size, _, steps = m.size()
        attn_hidden = torch.zeros(batch_size, self.decoder_dims)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims)
        hidden_states = attn_hidden, rnn1_hidden, rnn2_hidden
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims)
        cell_states = rnn1_cell, rnn2_cell
        go_frame = torch.zeros(batch_size, self.n_mels)
        context_vec = torch.zeros(batch_size, self.decoder_dims)
        encoder_seq = self.encoder(x, s_e)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        mel_outputs, attn_scores = [], []
        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()
        return mel_outputs, linear, attn_scores

    def generate(self, x, steps=2000):
        self.encoder.eval()
        self.postnet.eval()
        self.decoder.generating = True
        batch_size = 1
        x = torch.LongTensor(x).unsqueeze(0)
        attn_hidden = torch.zeros(batch_size, self.decoder_dims)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims)
        hidden_states = attn_hidden, rnn1_hidden, rnn2_hidden
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims)
        cell_states = rnn1_cell, rnn2_cell
        go_frame = torch.zeros(batch_size, self.n_mels)
        context_vec = torch.zeros(batch_size, self.decoder_dims)
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        mel_outputs, attn_scores = [], []
        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            if (mel_frames < -3.8).all() and t > 10:
                break
        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]
        self.encoder.train()
        self.postnet.train()
        self.decoder.generating = False
        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            None

    def restore(self, path):
        if not os.path.exists(path):
            None
            self.save(path)
        else:
            None
            self.load(path)
            self.decoder.r = self.r.item()

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'attn_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNormConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (HighwayNetwork,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSA,
     lambda: ([], {'attn_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), 0], {}),
     True),
    (PreNet,
     lambda: ([], {'in_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'dims': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Spk_UpsampleNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
]

class Test_dipjyoti92_SC_WaveRNN(_paritybench_base):
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

