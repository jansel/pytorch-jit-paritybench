import sys
_module = sys.modules[__name__]
del sys
dataset = _module
generate = _module
model = _module
preprocess = _module
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


import numpy as np


import torch


from random import randint


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


class Vocoder(nn.Module):

    def __init__(self, mel_channels, conditioning_channels, embedding_dim, rnn_channels, fc_channels, bits, hop_length):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2 ** bits
        self.hop_length = hop_length
        self.rnn1 = nn.GRU(mel_channels, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, mels):
        sample_frames = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (sample_frames - audio_slice_frames) // 2
        mels, _ = self.rnn1(mels)
        mels = mels[:, pad:pad + audio_slice_frames, :]
        mels = F.interpolate(mels.transpose(1, 2), scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)
        x = self.embedding(x)
        x, _ = self.rnn2(torch.cat((x, mels), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, mel):
        self.eval()
        output = []
        cell = get_gru_cell(self.rnn2)
        with torch.no_grad():
            mel, _ = self.rnn1(mel)
            mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
            mel = mel.transpose(1, 2)
            batch_size, sample_size, _ = mel.size()
            h = torch.zeros(batch_size, self.rnn_channels, device=mel.device)
            x = torch.zeros(batch_size, device=mel.device).fill_(self.quantization_channels // 2).long()
            for m in tqdm(torch.unbind(mel, dim=1), leave=False):
                x = self.embedding(x)
                h = cell(torch.cat((x, m), dim=1), h)
                x = F.relu(self.fc1(h))
                logits = self.fc2(x)
                posterior = F.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(posterior)
                x = dist.sample()
                output.append(2 * x.float().item() / (self.quantization_channels - 1.0) - 1.0)
        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        self.train()
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Vocoder,
     lambda: ([], {'mel_channels': 4, 'conditioning_channels': 4, 'embedding_dim': 4, 'rnn_channels': 4, 'fc_channels': 4, 'bits': 4, 'hop_length': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_bshall_UniversalVocoding(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

