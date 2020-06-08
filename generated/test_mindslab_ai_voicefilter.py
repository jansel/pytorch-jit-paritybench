import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
generator = _module
inference = _module
embedder = _module
model = _module
trainer = _module
adabound = _module
audio = _module
evaluation = _module
hparams = _module
plotting = _module
train = _module
writer = _module

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


import torch.nn.functional as F


import math


class LinearNorm(nn.Module):

    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hp.embedder.lstm_hidden, hp.embedder.
            emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):

    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp.embedder.num_mels, hp.embedder.lstm_hidden,
            num_layers=hp.embedder.lstm_layers, batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        mels = mel.unfold(1, self.hp.embedder.window, self.hp.embedder.stride)
        mels = mels.permute(1, 2, 0)
        x, _ = self.lstm(mels)
        x = x[:, (-1), :]
        x = self.proj(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        x = x.sum(0) / x.size(0)
        return x


class VoiceFilter(nn.Module):

    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, 'stft-related dimension mismatch'
        self.conv = nn.Sequential(nn.ZeroPad2d((3, 3, 0, 0)), nn.Conv2d(1, 
            64, kernel_size=(1, 7), dilation=(1, 1)), nn.BatchNorm2d(64),
            nn.ReLU(), nn.ZeroPad2d((0, 0, 3, 3)), nn.Conv2d(64, 64,
            kernel_size=(7, 1), dilation=(1, 1)), nn.BatchNorm2d(64), nn.
            ReLU(), nn.ZeroPad2d(2), nn.Conv2d(64, 64, kernel_size=(5, 5),
            dilation=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), nn.ZeroPad2d((
            2, 2, 4, 4)), nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2,
            1)), nn.BatchNorm2d(64), nn.ReLU(), nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), nn.
            BatchNorm2d(64), nn.ReLU(), nn.ZeroPad2d((2, 2, 16, 16)), nn.
            Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), nn.
            BatchNorm2d(64), nn.ReLU(), nn.ZeroPad2d((2, 2, 32, 32)), nn.
            Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), nn.
            BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 8, kernel_size=(1, 1),
            dilation=(1, 1)), nn.BatchNorm2d(8), nn.ReLU())
        self.lstm = nn.LSTM(8 * hp.audio.num_freq + hp.embedder.emb_dim, hp
            .model.lstm_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hp.model.lstm_dim, hp.model.fc1_dim)
        self.fc2 = nn.Linear(hp.model.fc1_dim, hp.model.fc2_dim)

    def forward(self, x, dvec):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        x = torch.cat((x, dvec), dim=2)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mindslab_ai_voicefilter(_paritybench_base):
    pass
