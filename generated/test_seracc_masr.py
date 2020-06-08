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


import torch.nn.functional as F


import numpy as np


from torch.nn.utils import remove_weight_norm


import torch.nn as nn


from torch.nn.utils import weight_norm


import torch.optim as optim


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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_seracc_masr(_paritybench_base):
    pass
