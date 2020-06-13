import sys
_module = sys.modules[__name__]
del sys
fid_score = _module
inception = _module

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


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torch.nn as nn


import torch.nn.functional as F


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mseitzer_pytorch_fid(_paritybench_base):
    pass
