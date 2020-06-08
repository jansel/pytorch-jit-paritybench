import sys
_module = sys.modules[__name__]
del sys
dataset = _module
defaults = _module
demo = _module
model = _module
test = _module
train = _module

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


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.nn.functional as F


import torch.nn as nn


from torch.utils.data import DataLoader


from collections import OrderedDict


from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yu4u_age_estimation_pytorch(_paritybench_base):
    pass
