import sys
_module = sys.modules[__name__]
del sys
convert = _module
convert_gn = _module
resnet = _module
synset = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
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


import torchvision.models as models


import torch.nn.functional as F


from collections import OrderedDict


from torchvision import transforms as trn


import torch.nn as nn


import torchvision.models.resnet


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ruotianluo_pytorch_resnet(_paritybench_base):
    pass
