import sys
_module = sys.modules[__name__]
del sys
lecture_summarizer = _module
server = _module
setup = _module
BertParent = _module
LectureService = _module
SummarizationService = _module
UdacityParser = _module
summarizer = _module
attempt = _module
lecture_searcher = _module
qa_start = _module
tables = _module
test = _module
tests = _module

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


import logging


import torch


import numpy as np


from numpy import ndarray


from typing import List


import uuid


import collections


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


import math

