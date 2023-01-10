import sys
_module = sys.modules[__name__]
del sys
extract_audio = _module
create_phases = _module
vad = _module
DataProviderDevelopment = _module
development = _module
loader_fn = _module
train_softmax = _module
train_softmax_not_dataloader = _module
DataProviderEnrollment = _module
DataProviderEnrollment_once = _module
enrollment = _module
DataProviderEvaluation = _module
evaluation = _module
PlotHIST = _module
PlotPR = _module
PlotROC = _module
ROC_PR_curve = _module
calculate_roc = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torchvision


import torchvision.transforms as transforms


from scipy.io.wavfile import read


import scipy.io.wavfile as wav


import numpy as np


import random


from random import shuffle


import torch.multiprocessing as multiprocessing


from torch._C import _set_worker_signal_handlers


from torch._C import _remove_worker_pids


from torch._C import _error_if_any_worker_fails


import functools


import collections


import re


from torch._six import string_classes


import torch.nn.init as init


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import time


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import BatchSampler


from torch.optim.lr_scheduler import StepLR


from sklearn.preprocessing import Normalizer


from sklearn.metrics.pairwise import cosine_similarity


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)
        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1211)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv21_bn(self.conv21(x)))
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.conv31_bn(self.conv31(x)))
        x = F.relu(self.conv32_bn(self.conv32(x)))
        x = F.relu(self.conv41_bn(self.conv41(x)))
        x = x.view(-1, 128 * 4 * 6 * 2)
        x = self.fc1_bn(self.fc1(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 128, 128])], {}),
     True),
]

class Test_astorfi_3D_convolutional_speaker_recognition_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

