import sys
_module = sys.modules[__name__]
del sys
awsio = _module
python = _module
lib = _module
io = _module
s3 = _module
s3dataset = _module
s3_cv_iterable_example = _module
s3_cv_iterable_shuffle_example = _module
s3_cv_map_example = _module
s3_cv_transform = _module
s3_imagenet_example = _module
s3_nlp_iterable_example = _module
setup = _module
test_integration = _module
test_read_datasets = _module
test_regions = _module
test_s3dataset = _module
test_s3iterabledataset = _module
test_utils = _module
get_version = _module

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


from torch.utils.data import IterableDataset


from torch.utils.data import Dataset


import torch


import torch.distributed as dist


import random


from itertools import chain


from torch.utils.data import DataLoader


from itertools import islice


from torchvision import transforms


import time


import warnings


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.models as models


import numpy as np


import math


from collections import defaultdict

