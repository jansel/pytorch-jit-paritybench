import sys
_module = sys.modules[__name__]
del sys
LSP_train = _module
data_config = _module
data_loader = _module
demo = _module
demo_utils = _module
batch_eval = _module
dstc = _module
extract_human = _module
metrics = _module
tokenizers = _module
util = _module
env = _module
distributed = _module
eval_utils = _module
train_utils = _module
lsp_model = _module
modeling_gpt2 = _module
optim = _module
prepro = _module
pycocoevalcap = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
eval = _module
meteor = _module
rouge = _module
tokenizer = _module
ptbtokenizer = _module
reddit = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import random


import torch


from math import ceil


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


import logging


from collections import defaultdict


import copy


import torch.nn as nn


from torch.nn import CrossEntropyLoss


from torch.optim import Optimizer


from torch.nn.utils import clip_grad_norm_


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_microsoft_DialoGPT(_paritybench_base):
    pass
