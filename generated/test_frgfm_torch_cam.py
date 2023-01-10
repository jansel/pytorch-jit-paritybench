import sys
_module = sys.modules[__name__]
del sys
collect_env = _module
verify_labels = _module
app = _module
conf = _module
cam_example = _module
eval_latency = _module
eval_perf = _module
setup = _module
conftest = _module
test_methods_activation = _module
test_methods_core = _module
test_methods_gradient = _module
test_methods_utils = _module
test_metrics = _module
test_utils = _module
torchcam = _module
methods = _module
_utils = _module
activation = _module
core = _module
gradient = _module
metrics = _module
utils = _module

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


import re


from collections import namedtuple


from typing import Any


from typing import Set


from typing import Tuple


import matplotlib.pyplot as plt


import torch


from torchvision import models


from torchvision.transforms.functional import normalize


from torchvision.transforms.functional import resize


from torchvision.transforms.functional import to_pil_image


from torchvision.transforms.functional import to_tensor


import math


import time


import numpy as np


from functools import partial


from torch.utils.data import SequentialSampler


from torchvision.datasets import ImageFolder


from torchvision.transforms import transforms as T


from torch import nn


from torchvision.models import mobilenet_v2


from torchvision.models import mobilenet_v3_small


from typing import List


from typing import Optional


from torch import Tensor


import logging


from typing import Union


import torch.nn.functional as F


from typing import Callable


from typing import Dict

