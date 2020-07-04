import sys
_module = sys.modules[__name__]
del sys
captcha_cnn_model = _module
captcha_gen = _module
captcha_predict = _module
captcha_setting = _module
captcha_test = _module
captcha_train = _module
my_dataset = _module
one_hot_encoding = _module

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


import torch.nn as nn


import torch


from torch.autograd import Variable


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding
            =1), nn.BatchNorm2d(32), nn.Dropout(0.5), nn.ReLU(), nn.
            MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3,
            padding=1), nn.BatchNorm2d(64), nn.Dropout(0.5), nn.ReLU(), nn.
            MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,
            padding=1), nn.BatchNorm2d(64), nn.Dropout(0.5), nn.ReLU(), nn.
            MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(captcha_setting.IMAGE_WIDTH // 8 *
            (captcha_setting.IMAGE_HEIGHT // 8) * 64, 1024), nn.Dropout(0.5
            ), nn.ReLU())
        self.rfc = nn.Sequential(nn.Linear(1024, captcha_setting.
            MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dee1024_pytorch_captcha_recognition(_paritybench_base):
    pass
