import sys
_module = sys.modules[__name__]
del sys
grad_cam = _module
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


from collections import Sequence


import numpy as np


import torch


import torch.nn as nn


from torch.nn import functional as F


import copy


import torch.nn.functional as F


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kazuto1011_grad_cam_pytorch(_paritybench_base):
    pass
