import sys
_module = sys.modules[__name__]
del sys
ChexnetTrainer = _module
DatasetGenerator = _module
DensenetModels = _module
HeatmapGenerator = _module
Main = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.nn.functional as tfunc


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as func


class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount,
            classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet169, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount,
            classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x


class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet201, self).__init__()
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount,
            classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zoogzog_chexnet(_paritybench_base):
    pass
