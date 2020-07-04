import sys
_module = sys.modules[__name__]
del sys
radio_transformer_networks = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


USE_CUDA = False


class RadioTransformerNetwork(nn.Module):

    def __init__(self, in_channels, compressed_dim):
        super(RadioTransformerNetwork, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True), nn.Linear(in_channels, compressed_dim))
        self.decoder = nn.Sequential(nn.Linear(compressed_dim,
            compressed_dim), nn.ReLU(inplace=True), nn.Linear(
            compressed_dim, in_channels))

    def decode_signal(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.in_channels ** 2 * (x / x.norm(dim=-1)[:, (None)])
        training_signal_noise_ratio = 5.01187
        communication_rate = 1
        noise = Variable(torch.randn(*x.size()) / (2 * communication_rate *
            training_signal_noise_ratio) ** 0.5)
        if USE_CUDA:
            noise = noise
        x += noise
        x = self.decoder(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gram_ai_radio_transformer_networks(_paritybench_base):
    pass
