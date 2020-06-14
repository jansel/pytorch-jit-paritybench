import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
augmentations = _module
dataloader = _module
main = _module
models = _module
model = _module
test = _module
utils = _module
progress_bar = _module

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


import time


import torch


import numpy as np


import warnings


from torch import nn


from torch import optim


from collections import OrderedDict


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_spytensor_pytorch_image_classification(_paritybench_base):
    pass
