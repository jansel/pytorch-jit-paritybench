import sys
_module = sys.modules[__name__]
del sys
prepare = _module
linear_regression = _module
ipython_config = _module
jupyter_notebook_config_py2 = _module
jupyter_notebook_config_py3 = _module
benchmark = _module
mnist = _module
mnist = _module
tensorboardx = _module
mnist = _module
tensorboardx = _module
mnist = _module
nn = _module
symeig = _module
vision = _module
dataset = _module
keras_mnist_test = _module
mnist_eager = _module
mnist_estimator = _module
mnist_session = _module
parsers = _module
tf_keras_mnist = _module
mnist_cnn = _module
cpu_and_gpu = _module
setup = _module
floydker = _module
build = _module
list_cmd = _module
render = _module
test = _module
utils = _module
test_utils = _module

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


import torch.optim as optim


from torch.autograd import Variable


import numpy as np


from torch.autograd.variable import Variable


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import torch as th


from torch.autograd.function import Function


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_floydhub_dockerfiles(_paritybench_base):
    pass
