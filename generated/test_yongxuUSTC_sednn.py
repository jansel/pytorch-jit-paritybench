import sys
_module = sys.modules[__name__]
del sys
config = _module
get_csv = _module
istft = _module
main_train_sednn_keras_v2 = _module
prepare_fea = _module
preprocessing = _module
spectrogram_to_wave = _module
wav_reconstruction = _module
data_generator = _module
evaluate = _module
main_dnn = _module
prepare_data = _module
crash = _module
stft = _module
test9 = _module
tmp01 = _module
tmp01b = _module
tmp01c = _module
tmp01d = _module
tmp01e = _module
tmp01f = _module
tmp02 = _module
tmp02b = _module
tmp03 = _module
tmp03b = _module
tmp04 = _module
tmp04b = _module

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


import numpy as np


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


import torch.optim as optim


from torch.autograd import Variable


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
        self.bn3 = nn.BatchNorm2d(n_hid)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.bn1(self.fc1(x))), p=drop_p, training=
            self.training)
        x3 = F.dropout(F.relu(self.bn2(self.fc2(x2))), p=drop_p, training=
            self.training)
        x4 = F.dropout(F.relu(self.bn3(self.fc3(x3))), p=drop_p, training=
            self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 4096
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 1024
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid * 2, n_hid)
        self.fc4 = nn.Linear(n_hid * 3, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x3 = torch.cat((x3, x2), dim=-1)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x4 = torch.cat((x4, x3), dim=-1)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 1024
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid * 2, n_hid)
        self.fc4 = nn.Linear(n_hid * 3, n_hid)
        self.fc5 = nn.Linear(n_hid * 4, n_hid)
        self.fc6 = nn.Linear(n_hid * 5, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x3 = torch.cat((x3, x2), dim=-1)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x4 = torch.cat((x4, x3), dim=-1)
        x5 = F.dropout(F.relu(self.fc4(x4)), p=drop_p, training=self.training)
        x5 = torch.cat((x5, x4), dim=-1)
        x6 = F.dropout(F.relu(self.fc5(x5)), p=drop_p, training=self.training)
        x6 = torch.cat((x6, x5), dim=-1)
        x7 = self.fc6(x6)
        return x7


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yongxuUSTC_sednn(_paritybench_base):
    pass
    def test_000(self):
        self._check(DNN(*[], **{'stack_num': 4, 'n_freq': 4}), [torch.rand([4, 4, 4])], {})

