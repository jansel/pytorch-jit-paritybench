import sys
_module = sys.modules[__name__]
del sys
hmtlPredictor = _module
predictionFormatter = _module
server = _module
evaluate = _module
fine_tune = _module
hmtl = _module
common = _module
util = _module
dataset_readers = _module
coref_ace = _module
dataset_utils = _module
ace = _module
mention_ace = _module
ner_ontonotes = _module
relation_ace = _module
models = _module
coref_custom = _module
hmtl = _module
layerCoref = _module
layerEmdCoref = _module
layerEmdRelation = _module
layerNer = _module
layerNerEmd = _module
layerNerEmdCoref = _module
layerNerEmdRelation = _module
layerRelation = _module
relation_extraction = _module
modules = _module
seq2seq_encoders = _module
stacked_gru = _module
text_field_embedders = _module
shortcut_connect_text_field_embedder = _module
tasks = _module
task = _module
training = _module
metrics = _module
conll_coref_full_scores = _module
relation_f1_measure = _module
multi_task_trainer = _module
sampler_multi_task_trainer = _module
html_senteval = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from typing import List


from typing import Dict


from typing import Any


from typing import Iterable


import torch


import torch.nn.functional as F


import math


import re


import logging


import itertools


from copy import deepcopy


from typing import Tuple


from typing import Optional


import torch.nn as nn


from torch.autograd import Variable


from torch.nn import Dropout


from torch.nn import Linear


from torch.nn import GRU


from typing import Set


from collections import defaultdict


import time


import random


import torch.optim.lr_scheduler


import numpy as np

