import sys
_module = sys.modules[__name__]
del sys
tncc_finetune = _module
wcm_finetune = _module
wcm_zeroshot = _module
ynat_finetune = _module

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


import time


import logging


import numpy as np


import torch


import torch.nn as nn


from torch.nn.modules import padding


import torch.optim as optim


import torch.nn.functional as F


from sklearn.metrics import f1_score


from sklearn.metrics import classification_report


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


import random


class CINO_Model(nn.Module):

    def __init__(self, cino_path, class_num):
        super().__init__()
        self.config = XLMRobertaConfig.from_pretrained(cino_path)
        self.cino = XLMRobertaModel.from_pretrained(cino_path)
        for param in self.cino.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.config.hidden_size, class_num)

    def forward(self, input_ids, attention_mask):
        output = self.cino(input_ids, attention_mask)[1]
        logits = self.fc(output)
        return logits

