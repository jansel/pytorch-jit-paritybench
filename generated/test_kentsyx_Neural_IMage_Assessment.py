import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
main = _module
model = _module
test = _module

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


import torch.autograd as autograd


import torch.optim as optim


import torch.nn as nn


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(nn.Dropout(p=0.75), nn.Linear(
            in_features=25088, out_features=num_classes), nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kentsyx_Neural_IMage_Assessment(_paritybench_base):
    pass
