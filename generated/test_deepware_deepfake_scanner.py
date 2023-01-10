import sys
_module = sys.modules[__name__]
del sys
scan = _module

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


import time


from collections import defaultdict


import torch


import torch.nn as nn


import numpy as np


from sklearn.cluster import DBSCAN


from torchvision import transforms


import torchvision.transforms.functional as TF


from torch.cuda.amp import autocast


class EffNet(nn.Module):

    def __init__(self, arch='b3'):
        super(EffNet, self).__init__()
        fc_size = {'b1': 1280, 'b2': 1408, 'b3': 1536, 'b4': 1792, 'b5': 2048, 'b6': 2304, 'b7': 2560}
        assert arch in fc_size.keys()
        effnet_model = getattr(effnet, 'tf_efficientnet_%s_ns' % arch)
        self.encoder = effnet_model()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(fc_size[arch], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Ensemble(nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = []
        for i, model in enumerate(self.models):
            y = model(x)
            preds.append(y)
        final = torch.mean(torch.stack(preds), dim=0)
        return final


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Ensemble,
     lambda: ([], {'models': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_deepware_deepfake_scanner(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

