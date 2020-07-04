import sys
_module = sys.modules[__name__]
del sys
beamdecode = _module
data = _module
decoder = _module
_init_path = _module
embedding = _module
record = _module
train = _module
feature = _module
models = _module
base = _module
conv = _module
trainable = _module
train = _module

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


import torch.nn.functional as F


import numpy as np


from torch.nn.utils import remove_weight_norm


import torch.nn as nn


from torch.nn.utils import weight_norm


import torch.optim as optim


class MASRModel(nn.Module):

    def __init__(self, **config):
        super().__init__()
        self.config = config

    @classmethod
    def load(cls, path):
        package = torch.load(path)
        state_dict = package['state_dict']
        config = package['config']
        m = cls(**config)
        m.load_state_dict(state_dict)
        return m

    def to_train(self):
        self.__class__.__bases__ = TrainableModel,
        return self

    def predict(self, *args):
        raise NotImplementedError()

    def _default_decode(self, yp, yp_lens):
        idxs = yp.argmax(1)
        texts = []
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ''
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):
                    text += self.vocabulary[i.item()]
                last = i
            texts.append(text)
        return texts

    def decode(self, *outputs):
        return self._default_decode(*outputs)


class ConvBlock(nn.Module):

    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_seracc_masr(_paritybench_base):
    pass
