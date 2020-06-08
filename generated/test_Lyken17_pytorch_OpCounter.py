import sys
_module = sys.modules[__name__]
del sys
date_extraction = _module
evaluate_famous_models = _module
evaluate_rnn_models = _module
setup = _module
thop = _module
profile = _module
rnn_hooks = _module
utils = _module
vision = _module
basic_hooks = _module
efficientnet = _module

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


import logging


from torch.nn.modules.conv import _ConvNd


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Lyken17_pytorch_OpCounter(_paritybench_base):
    pass
