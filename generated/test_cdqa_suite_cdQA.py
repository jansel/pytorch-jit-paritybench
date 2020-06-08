import sys
_module = sys.modules[__name__]
del sys
api = _module
cdqa = _module
pipeline = _module
cdqa_sklearn = _module
reader = _module
bertqa_sklearn = _module
retriever = _module
retriever_sklearn = _module
text_transformers = _module
vectorizers = _module
utils = _module
converters = _module
download = _module
evaluation = _module
filters = _module
setup = _module
tests = _module
test_converters = _module
test_evaluation = _module
test_pipeline = _module
test_processor = _module

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


import collections


import logging


import math


import random


import uuid


import numpy as np


import torch


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


from torch.optim.lr_scheduler import LambdaLR


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cdqa_suite_cdQA(_paritybench_base):
    pass
