import sys
_module = sys.modules[__name__]
del sys
bert = _module

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


import numpy as np


import torch


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_real_brilliant_bert_chinese_pytorch(_paritybench_base):
    pass
