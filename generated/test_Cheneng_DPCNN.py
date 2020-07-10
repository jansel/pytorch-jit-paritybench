import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
dataset = _module
main = _module
BasicModule = _module
DPCNN = _module
model = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils import data


import torch


import torch.autograd as autograd


import torch.nn as nn


import torch.optim as optim


import torch.utils.data as data


import torch.nn.functional as F


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, 2)

    def forward(self, x):
        batch = x.shape[0]
        x = self.conv_region_embedding(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        while x.size()[-2] > 2:
            x = self._block(x)
        x = x.view(batch, 2 * self.channel_size)
        x = self.linear_out(x)
        return x

    def _block(self, x):
        x = self.padding_pool(x)
        px = self.pooling(x)
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = x + px
        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DPCNN,
     lambda: ([], {'config': _mock_config(word_embedding_dimension=4)}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_Cheneng_DPCNN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

