import sys
_module = sys.modules[__name__]
del sys
example1 = _module
example2_adv_example = _module
example2_gradient = _module
example3 = _module
example4 = _module
example5 = _module
example6 = _module
example6_features = _module
example6_gradient = _module
example6_squeezenet = _module

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


import numpy as np


import random


import torch.utils.data as data


from torch.autograd import Variable


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


def myGradientHook(grad):
    grad_np = grad.data.numpy()
    print('Here is the gradient right before the first FC layer:')
    print(grad_np)
    print('Gradient shape:')
    print(grad_np.shape)
    grad[(0), :] = 0


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
        x.register_hook(myGradientHook)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


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
        if self.training:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if self.training:
            x = F.dropout(x, training=True)
        x = self.fc2(x)
        nans = np.sum(np.isnan(x.detach().cpu().numpy()))
        infs = np.sum(np.isinf(x.detach().cpu().numpy()))
        if nans > 0:
            None
        if infs > 0:
            None
        return F.log_softmax(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(30)
        self.fc1 = nn.Linear(750, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), 2))
        x = F.elu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.elu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.elu(F.max_pool2d(self.bn4(self.conv4(x)), 2))
        x = x.view(-1, 750)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(512, 2, kernel_size=1)
        self.avgpool = nn.AvgPool2d(13)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = self.conv(x)
        x = self.avgpool(x)
        x = F.log_softmax(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_soravux_pytorch_tutorial(_paritybench_base):
    pass

    def test_000(self):
        self._check(Net(*[], **{}), [torch.rand([4, 512, 64, 64])], {})
