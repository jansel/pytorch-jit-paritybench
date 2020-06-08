import sys
_module = sys.modules[__name__]
del sys
losses = _module
metrics = _module
models = _module
base = _module
mnist = _module
utils = _module

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


from torch.nn import functional as F


from torch import nn


class BaseModel(nn.Module):
    """An abstract class representing a model architecture.

    Any model definition should subclass `BaseModel`.
    """

    def __init__(self):
        super().__init__()

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())

    def forward(self, x):
        raise NotImplementedError


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kevinzakka_pytorch_goodies(_paritybench_base):
    pass
