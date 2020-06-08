import sys
_module = sys.modules[__name__]
del sys
convert = _module
convert_gn = _module
resnet = _module
synset = _module

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


import re


import numpy as np


import torch


import torch.nn.functional as F


from collections import OrderedDict


import torch.nn as nn


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ruotianluo_pytorch_resnet(_paritybench_base):
    pass
