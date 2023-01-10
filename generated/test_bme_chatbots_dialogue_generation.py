import sys
_module = sys.modules[__name__]
del sys
setup = _module
src = _module
data = _module
evaluate = _module
interact = _module
model = _module
train = _module

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


import torch


import random


from itertools import zip_longest


from itertools import chain


from copy import deepcopy


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torch.utils.data.distributed import DistributedSampler


from math import ceil


from torch.nn.functional import softmax


from torch.nn.modules import ModuleList


from torch.nn.modules import Module


from torch.nn import Linear


from torch.nn import Parameter


from collections import namedtuple


from torch.utils.checkpoint import checkpoint


import logging


import numpy as np


from collections import OrderedDict


from collections import defaultdict


from functools import partial


from torch.nn.functional import log_softmax


from torch.nn.functional import nll_loss


from torch.nn.functional import cross_entropy


from torch.distributed import all_reduce


from torch.distributed import ReduceOp


from torch.distributed import barrier


from torch.optim.lr_scheduler import LambdaLR


from torch.nn.utils import clip_grad_norm_


from torch.nn.parallel import DistributedDataParallel


class CkptGPT2Layer(Module):
    """
    Wrapper class to perform checkpointing of
    Block ( GPT2Layer ) modue.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward_wrapper(self, *args, **kwargs):
        """
        Converts the forward function's output
        to a tuple.
        """
        return tuple(self.module.forward(*args, **kwargs))

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        return checkpoint(self.forward_wrapper, x, layer_past, attention_mask, head_mask)


class CkptXLNetLayer(Module):
    """
    Wrapper class to perform checkpointing of
    XLNetLayer modue.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        return checkpoint(self.module.forward, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems, target_mapping, head_mask)

