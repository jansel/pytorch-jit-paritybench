import sys
_module = sys.modules[__name__]
del sys
dev = _module
test_logger = _module
trainer = _module
plugins = _module
accuracy = _module
constant = _module
logger = _module
loss = _module
monitor = _module
plugin = _module
progress = _module
saver = _module
time = _module
visdom_logger = _module

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


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


class ShallowMLP(nn.Module):

    def __init__(self, shape, force_no_cuda=False):
        super(ShallowMLP, self).__init__()
        self.in_shape = shape[0]
        self.hidden_shape = shape[1]
        self.out_shape = shape[2]
        self.fc1 = nn.Linear(self.in_shape, self.hidden_shape)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_shape, self.out_shape)
        self.use_cuda = torch.cuda.is_available() and not force_no_cuda
        if self.use_cuda:
            self = self

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_alexsax_pytorch_visdom(_paritybench_base):
    pass
    def test_000(self):
        self._check(ShallowMLP(*[], **{'shape': [4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

