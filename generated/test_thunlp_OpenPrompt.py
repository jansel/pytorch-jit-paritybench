import sys
_module = sys.modules[__name__]
del sys
conf = _module
cli = _module
openprompt = _module
config = _module
ZH = _module
closed_QA = _module
coreference = _module
entity_typing = _module
generation = _module
nli = _module
paraphrase = _module
processor = _module
reading_comprehensation = _module
relation = _module
sentiment = _module
summarization = _module
topic_classification = _module
data_utils = _module
conditional_generation_dataset = _module
data_processor = _module
data_sampler = _module
fewglue_dataset = _module
huggingface_dataset = _module
lama_dataset = _module
nli_dataset = _module
relation_classification_dataset = _module
text_classification_dataset = _module
typing_dataset = _module
utils = _module
default_config = _module
lm_bff_trainer = _module
pipeline_base = _module
plms = _module
lm = _module
mlm = _module
seq2seq = _module
utils = _module
prompt_base = _module
prompts = _module
automatic_verbalizer = _module
generation_verbalizer = _module
knowledgeable_verbalizer = _module
manual_template = _module
manual_verbalizer = _module
mixed_template = _module
one2one_verbalizer = _module
prefix_tuning_template = _module
prompt_generator = _module
prototypical_verbalizer = _module
ptr_prompts = _module
ptuning_prompts = _module
soft_template = _module
soft_verbalizer = _module
protoverb_trainer = _module
trainer = _module
calibrate = _module
crossfit_metrics = _module
cuda = _module
logging = _module
metrics = _module
reproduciblity = _module
utils = _module
setup = _module
test_condition_generation_dataset = _module
test_lama_dataset = _module
test_nli_dataset = _module
test_relation_classification_dataset = _module
test_text_classification_dataset = _module
test_typing_dataset = _module
text_fewglue = _module
TokenizerWrapper = _module
basic = _module
dataloader = _module
modeling = _module
template = _module
verbalizer = _module

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


from collections import defaultdict


from collections import namedtuple


from typing import *


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data.dataset import Subset


from typing import Union


import random


from abc import ABC


from abc import abstractmethod


from collections import Counter


from typing import List


from typing import Dict


from typing import Callable


from typing import Sequence


from torch.utils.data import dataset


import copy


import torch


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.sampler import RandomSampler


import torch.nn as nn


from torch.utils.data import DataLoader


import itertools


import warnings


from typing import Tuple


from typing import Optional


import torch.nn.functional as F


from functools import partial


import re


import string


from torch import nn


from torch.nn.parallel import DataParallel


from inspect import Parameter


from torch.nn.parameter import Parameter


from torch.nn.parallel.data_parallel import DataParallel


from torch.utils.data.dataset import Dataset


from math import ceil


import inspect


import time


logger = logging.getLogger()


class InputFeatures(dict):
    """
    The class for input to the PLM and Prompts. To make users explicitly know the available keys,
    we define a dict with a set of predefined possible keys. The default value to any key is None.
    When use it as a dict, all the keys whose values are None are invisible.

    This class support most of the dict's operation (See Examples). It can also be consumed by
    pytorch's default_collate in DataLoader.
    Also a :py:meth:`to_tensor()` method is build to convert the values into torch.Tensor for torch's input.

    Examples:

    ..  code-block:: python

        in_feat = InputFeatures(**{'input_ids':[1,4,5], 'soft_token_ids': [3,4,5]})  # init from dict
        print(in_feat.keys())       # ['input_ids, 'soft_token_ids']
        in_feat['label'] = 3        # can assign value like normal dict
        print(in_feat.keys())       # ['input_ids','label', 'soft_token_ids'] (Note that it's also ordered)
        print(in_feat['label'])     # 3
        in_feat['alice'] = 0        # KeyError: Key alice not in predefined set of keys
        in_feat.values()            # [[1,4,5], 3, [3,4,5]]  (Note that it's also ordered)
        [in_feat[key] for key in in_feat]   # [[1,4,5], 3, [3,4,5]]
        new_dict= {**in_feat, 'new_key':2}  # new_dict is {'input_ids': [1, 4, 5], 'label': 3, 'soft_token_ids': [3, 4, 5], 'new_key': 2}

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label', 'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids', 'past_key_values', 'loss_ids']
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label', 'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids', 'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len']
    non_tensorable_keys = []

    def __init__(self, input_ids: Optional[Union[List, torch.Tensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, attention_mask: Optional[Union[List[int], torch.Tensor]]=None, token_type_ids: Optional[Union[List[int], torch.Tensor]]=None, label: Optional[Union[int, torch.Tensor]]=None, decoder_input_ids: Optional[Union[List, torch.Tensor]]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, soft_token_ids: Optional[Union[List, torch.Tensor]]=None, past_key_values: Optional[torch.Tensor]=None, loss_ids: Optional[Union[List, torch.Tensor]]=None, guid: Optional[str]=None, tgt_text: Optional[str]=None, use_cache: Optional[bool]=None, encoded_tgt_text: Optional[str]=None, input_ids_len: Optional[int]=None, **kwargs):
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len
        for k in kwargs.keys():
            logger.warning('Your are passing an unexpected key words: {} to InputFeatures, might yield unexpected behaviours!'.format(k))
            setattr(self, k, kwargs[k])

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)

    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())

    def __len__(self):
        return len(self.keys())

    def to_tensor(self, device: str='cuda'):
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self

    def to(self, device: str='cuda:0'):
        """move the tensor keys to runtime device, such as gpu:0
        """
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value)
        return self

    def cuda(self, device: str='cuda:0'):
        """mimic the tensor behavior
        """
        return self

    def to_json_string(self, keep_none=False):
        """Serializes this instance to a JSON string."""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                data[key] = value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + '\n'

    def keys(self, keep_none=False) ->List[str]:
        """get all keys of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[str]`: keys of the InputFeatures
        """
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]

    def to_dict(self, keep_none=False) ->Dict[str, Any]:
        """get the dict of mapping from keys to values of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`Dict[str, Any]`: dict of mapping from keys to values of the InputFeatures
        """
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] = value
            elif value is None and keep_none:
                data[key] = None
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError('Key {} not in predefined set of keys'.format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) ->List[Any]:
        """get the values with respect to the keys  of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[Any]`: the values with respect to the keys of the InputFeatures
        """
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)

    def items(self):
        """get the (key, value) pairs  of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[Any]`: the (key, value) pairs of the InputFeatures
        """
        return [(key, self.__getitem__(key)) for key in self.keys()]

    @staticmethod
    def collate_fct(batch: List):
        """
        This function is used to collate the input_features.

        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        """
        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == 'encoded_tgt_text':
                return_dict[key] = [d[key] for d in batch]
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    None
        return InputFeatures(**return_dict)


class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self, guid=None, text_a='', text_b='', label=None, meta: Optional[Dict]=None, tgt_text: Optional[Union[str, List[str]]]=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) ->List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) ->None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_cfg_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            None
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict


def signature(f):
    """Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.
    
    Args:
        f (:obj:`function`) : the function to get the input arguments.
    
    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s

    """
    sig = inspect.signature(f)
    args = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    varargs = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_POSITIONAL]
    varargs = varargs[0] if varargs else None
    keywords = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD]
    keywords = keywords[0] if keywords else None
    defaults = [p.default for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not p.empty] or None
    argspec = namedtuple('Signature', ['args', 'defaults', 'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)

