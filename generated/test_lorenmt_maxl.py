import sys
_module = sys.modules[__name__]
del sys
create_dataset = _module
model_vgg_human = _module
model_vgg_maxl = _module
model_vgg_maxl_firstorder = _module
model_vgg_single = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


from torch.utils.data.dataset import Dataset


from torchvision import transforms


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import check_integrity


import numpy as np


import torch


import torch.nn as nn


import torchvision.transforms as transforms


import torch.optim as optim


import torch.nn.functional as F


from collections import OrderedDict


import torch.utils.data.sampler as sampler


import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


psi = [5] * 20

