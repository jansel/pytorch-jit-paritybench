import sys
_module = sys.modules[__name__]
del sys
data = _module
data_manager = _module
data_sets = _module
transforms = _module
eval = _module
evaluate = _module
infer = _module
net = _module
audio = _module
base_model = _module
loss = _module
metric = _module
model = _module
run = _module
train = _module
base_trainer = _module
trainer = _module
utils = _module
logger = _module
util = _module
visualization = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.utils.data as data


from torch.utils.data.dataloader import default_collate


from torchvision import transforms


import torch.nn as nn


from torchaudio.transforms import Spectrogram


from torchaudio.transforms import MelSpectrogram


from torchaudio.transforms import ComplexNorm


from torchaudio.transforms import TimeStretch


from torchaudio.transforms import AmplitudeToDB


from torch.distributions import Uniform


import logging


import torch.nn.functional as F


import math


from torchvision.utils import make_grid


class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):
        super(SpecNormalization, self).__init__()
        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x

    def z_transform(self, x):
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x):
        return self._norm(x)


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, config=''):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.classes = None

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)


class RandomTimeStretch(TimeStretch):

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):
        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1.0 - max_perc, 1 + max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate


def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length


class MelspectrogramStretch(MelSpectrogram):

    def __init__(self, hop_length=None, sample_rate=44100, num_mels=128, fft_length=2048, norm='whiten', stretch_param=[0.4, 0.4]):
        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, n_fft=fft_length, hop_length=hop_length, n_mels=num_mels)
        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, pad=self.pad, power=None, normalized=False)
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1], self.hop_length, self.n_fft // 2 + 1, fixed_rate=None)
        self.complex_norm = ComplexNorm(power=2.0)
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)
        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft // 2)
            lengths = lengths.long()
        if torch.rand(1)[0] <= self.prob and self.training:
            x, rate = self.random_stretch(x)
            lengths = (lengths.float() / rate).long() + 1
        x = self.complex_norm(x)
        x = self.mel_scale(x)
        x = self.norm(x)
        if lengths is not None:
            return x, lengths
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AudioCRNN(BaseModel):

    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1
        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, num_mels=128, fft_length=2048, norm='whiten', stretch_param=[0.4, 0.4])
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):

        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        for name, layer in self.net['convs'].named_children():
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size, layer.stride])
                lengths = ((lengths + 2 * p - k) // s + 1).long()
        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):
        x, lengths, _ = batch
        xt = x.float().transpose(1, 2)
        xt, lengths = self.spec(xt, lengths)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)
        x = xt.transpose(1, -1)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        x = self._many_to_one(x, lengths)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward(x)
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()
            return self.classes[max_ind], out[:, (max_ind)].item()


class AudioCNN(AudioCRNN):

    def forward(self, batch):
        x, _, _ = batch
        x = x.float().transpose(1, 2)
        x = self.spec(x)
        x = self.net['convs'](x)
        x = x.view(x.size(0), -1)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x


class AudioRNN(AudioCRNN):

    def forward(self, batch):
        x, lengths, _ = batch
        x = x.float().transpose(1, 2)
        x, lengths = self.spec(x, lengths)
        x = x.transpose(1, -1)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        x = self._many_to_one(x, lengths)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SpecNormalization,
     lambda: ([], {'norm_type': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ksanjeevan_crnn_audio_classification(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

