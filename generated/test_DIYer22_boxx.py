import sys
_module = sys.modules[__name__]
del sys
boxx = _module
cp = _module
g = _module
gg = _module
p = _module
script = _module
tool = _module
toolFunction = _module
toolGui = _module
toolIo = _module
toolLog = _module
toolMarkdown = _module
toolStructObj = _module
toolSystem = _module
toolTools = _module
undetermined = _module
ylcompat = _module
yldb = _module
dbPublicFuncation = _module
yldf = _module
ylmysql = _module
ylsqlite = _module
ylimg = _module
showImgsInBrowser = _module
ylimgTool = _module
ylimgVideoAndGif = _module
yllab = _module
ylml = _module
ylDete = _module
ylmlEvalu = _module
ylmlTest = _module
ylmlTrain = _module
ylsci = _module
ylnp = _module
ylstatistics = _module
ylvector = _module
ylsys = _module
ylth = _module
pthnan = _module
ylweb = _module
generate_table_of_contents_for_ipynb = _module
g_call = _module
loga = _module
mapmp = _module
show = _module
torch_data = _module
tree = _module
w = _module
setup = _module
testAll = _module
toolTest = _module
yldbTest = _module
ylimgTest = _module

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
xrange = range
wraps = functools.wraps


import time


import math


import re


from collections import defaultdict


import inspect


import types


import numpy as np


from functools import reduce


from collections import OrderedDict


from functools import wraps


from torch.autograd import Variable


import torch.utils.data


from torch.nn import Conv2d


from torch.nn import Linear


from torch.nn import ConvTranspose2d


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn import Tanh


from torch.nn import Softmax2d


from torch.nn import CrossEntropyLoss


from torch.nn import DataParallel


from torch.nn import MSELoss


from torch.nn import MaxPool2d


from torch.nn import AvgPool2d


from torch.nn import Module


from torch.nn import functional


from torch.nn import Sequential


import torchvision


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torch


from torchvision import datasets


from torchvision import transforms

