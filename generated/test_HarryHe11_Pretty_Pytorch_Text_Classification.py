import sys
_module = sys.modules[__name__]
del sys
bert_optimizer = _module
Bert = _module
run = _module
text_cleaner = _module
twitter_preprocessor = _module
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


import math


import torch


from torch.optim import Optimizer


from torch.optim.optimizer import required


from torch.nn.utils import clip_grad_norm_


import logging


import abc


import torch.nn as nn


import time


import numpy as np


import pandas as pd


import random


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooler_output = self.bert(context, attention_mask=mask)
        pooled = self.dropout(pooler_output)
        logits = self.fc(pooled)
        return logits

