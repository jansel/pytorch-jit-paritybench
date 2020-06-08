import sys
_module = sys.modules[__name__]
del sys
inception_score = _module

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


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


import torch.utils.data


import numpy as np


from scipy.stats import entropy


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sbarratt_inception_score_pytorch(_paritybench_base):
    pass
