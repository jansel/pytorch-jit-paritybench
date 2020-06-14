import sys
_module = sys.modules[__name__]
del sys
VBMF = _module
dataset = _module
decompositions = _module
main = _module

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


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data as data


from torch.autograd import Variable


import torch.nn.functional as F


import time


from itertools import chain


class ModifiedVGG16Model(torch.nn.Module):

    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(25088, 4096
            ), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jacobgil_pytorch_tensor_decompositions(_paritybench_base):
    pass
