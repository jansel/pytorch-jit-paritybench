import sys
_module = sys.modules[__name__]
del sys
convai_evaluation = _module
example_entry = _module
interact = _module
test_special_tokens = _module
train = _module
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


import random


import logging


from collections import defaultdict


from functools import partial


import torch


import torch.nn.functional as F


from itertools import chain


import warnings


import math


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_huggingface_transfer_learning_conv_ai(_paritybench_base):
    pass
