import sys
_module = sys.modules[__name__]
del sys
coglm_strategy = _module
cogview2_completion = _module
cogview2_text2image = _module
comp_pipeline = _module
base_completion = _module
patch_completion = _module
predict = _module
pretrain_coglm = _module
sr_pipeline = _module
direct_sr = _module
dsr_model = _module
dsr_sampling = _module
iterative_sr = _module
itersr_model = _module
itersr_sampling = _module
sr_group = _module

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


import math


import random


import torch


import numpy as np


import torch.nn.functional as F


from functools import partial


from torchvision.io import read_image


from torchvision import transforms


from typing import List


from torchvision.utils import save_image


from torchvision.utils import make_grid

