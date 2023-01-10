import sys
_module = sys.modules[__name__]
del sys
LM_extractor = _module
visualization = _module
LR = _module
LR_tf = _module
MLP_LM = _module
MLP_combined_features = _module
MLP_psycho_features = _module
SVM_LM = _module
SVM_combined_features = _module
SVM_psycho_features = _module
NRC_features_extractor = _module
NRC_vad_features_extractor = _module
affectivespace_features_extractor = _module
hourglass_features_extractor = _module
mairesse_processor = _module
readability_features_extractor = _module
author_100recent = _module
data_utils = _module
dataset_processors = _module
gen_utils = _module
linguistic_features_utils = _module

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


import numpy as np


import pandas as pd


import re


import time


import torch


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import math

