import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
architext = _module
grounded_program_synthesis = _module
lang = _module
train_trlx = _module
ilql_sentiments = _module
ppo_sentiments = _module
randomwalks = _module
ilql_randomwalks = _module
ppo_randomwalks = _module
randomwalks = _module
simulacra = _module
setup = _module
tests = _module
test_configs = _module
test_ppo = _module
test_utils = _module
trlx = _module
data = _module
accelerate_base_datatypes = _module
configs = _module
ilql_types = _module
method_configs = _module
ppo_types = _module
orchestrator = _module
offline_orchestrator = _module
ppo_orchestrator = _module
pipeline = _module
offline_pipeline = _module
ppo_pipeline = _module
ray_tune = _module
train_funcs = _module
wandb = _module
sweep = _module
trainer = _module
accelerate_base_trainer = _module
accelerate_ilql_trainer = _module
accelerate_ppo_trainer = _module
nn = _module
ilql_models = _module
ppo_models = _module
utils = _module
loading = _module
modeling = _module

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


from typing import List


import numpy as np


from typing import Iterable


from typing import Callable


from time import time


import random


from abc import abstractmethod


from abc import abstractstaticmethod


from typing import Any


from typing import Dict


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


import time


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


import torch.nn.functional as F


from typing import cast


import uuid


import inspect


from collections import defaultdict


from copy import deepcopy


from functools import reduce


from itertools import chain


from torch import nn


import torch.nn as nn


from enum import Enum


from torch.optim.lr_scheduler import ChainedScheduler


from torch.optim.lr_scheduler import LinearLR


from typing import MutableMapping


import functools


import torch.distributed as dist


def register_method(name):
    """Decorator used register a method config
    Args:
        name: Name of the method
    """

    def register_class(cls, name):
        _METHODS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls
    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)
    cls = name
    name = cls.__name__
    register_class(cls, name.lower())
    return cls


def make_head(n_embd: int, out: int) ->nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(nn.Linear(n_embd, n_embd * 2), nn.ReLU(), nn.Linear(n_embd * 2, out))


def rgetattr(obj, attr: str, *args) ->object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split('.')
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def findattr(obj, attrs: Tuple[str]) ->Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f'Could not find an attribute from `{attrs}` in `{obj}`')


def hf_get_causal_hidden_layers(model: nn.Module) ->Tuple[nn.Module]:
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = 'transformer.h', 'model.decoder.layers', 'gpt_neox.layers'
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int=0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_causal_hidden_layers(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def hf_get_causal_final_norm(model: nn.Module) ->float:
    """Returns the final (layer) norm of the specified model.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = 'transformer.ln_f', 'model.decoder.final_layer_norm', 'gpt_neox.final_layer_norm'
    return findattr(model, norm_attrs)

