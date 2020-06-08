import sys
_module = sys.modules[__name__]
del sys
generate_5k = _module
train_mnist = _module
attack = _module
attack_mnist = _module
visualize_adv_examples = _module
visualize_imnet = _module
visualize_mnist = _module

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


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch.optim as optim


import random


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.r = nn.Parameter(data=torch.zeros(1, 784), requires_grad=True)

    def forward(self, x):
        x = x + self.r
        x = torch.clamp(x, 0, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def classify(self, x):
        outputs = self.forward(x)
        outputs = outputs / torch.norm(outputs)
        max_val, max_idx = torch.max(outputs, 1)
        return int(max_idx.data.numpy()), float(max_val.data.numpy())


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_akshaychawla_Adversarial_Examples_in_PyTorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(Net(*[], **{}), [torch.rand([784, 784])], {})
