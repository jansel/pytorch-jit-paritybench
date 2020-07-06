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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import warnings


import numpy as np


import torch


from sklearn.base import BaseEstimator


import collections


import logging


import math


import random


import uuid


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


from torch.optim.lr_scheduler import LambdaLR


from sklearn.base import TransformerMixin


from collections import Counter


import string


import re

