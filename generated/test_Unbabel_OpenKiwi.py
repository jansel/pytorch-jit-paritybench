import sys
_module = sys.modules[__name__]
del sys
conf = _module
kiwi = _module
__main__ = _module
cli = _module
better_argparse = _module
main = _module
models = _module
linear = _module
nuqe = _module
predictor = _module
predictor_estimator = _module
quetch = _module
opts = _module
pipelines = _module
evaluate = _module
jackknife = _module
predict = _module
train = _module
constants = _module
data = _module
builders = _module
corpus = _module
fields = _module
alignment_field = _module
qe_field = _module
sequence_labels_field = _module
fieldsets = _module
extend_vocabs_fieldset = _module
fieldset = _module
iterators = _module
qe_dataset = _module
tokenizers = _module
utils = _module
vectors = _module
vocabulary = _module
lib = _module
search = _module
loggers = _module
metrics = _module
functions = _module
stats = _module
label_dictionary = _module
linear_model = _module
linear_trainer = _module
linear_word_qe_decoder = _module
linear_word_qe_features = _module
linear_word_qe_sentence = _module
sequence_parts = _module
sparse_feature_vector = _module
sparse_vector = _module
structured_classifier = _module
structured_decoder = _module
linear_word_qe_classifier = _module
model = _module
modules = _module
attention = _module
scorer = _module
nuqe = _module
quetch = _module
utils = _module
predictors = _module
linear_tester = _module
trainers = _module
callbacks = _module
linear_word_qe_trainer = _module
trainer = _module
convert_parsed_to_postags = _module
extract_columns = _module
merge_target_and_gaps_preds = _module
stack_probabilities_for_linear = _module
conftest = _module
test_data_builders = _module
test_linear = _module
test_metrics = _module
test_model = _module
test_model_config = _module
test_nuqe = _module
test_nuqe_18 = _module
test_predest = _module
test_predictor = _module
test_qe_dataset = _module
test_quetch = _module
test_quetch_18 = _module
test_write_read_config_files = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import logging


from abc import ABCMeta


from abc import abstractmethod


import torch


import torch.nn as nn


from torch import nn


from collections import OrderedDict


import torch.nn.functional as F


import numpy as np


from torch.autograd import Function


from torch.nn.utils.rnn import pack_padded_sequence as pack


from torch.nn.utils.rnn import pad_packed_sequence as unpack


class ModelConfig:
    __metaclass__ = ABCMeta

    def __init__(self, vocabs):
        """Model Configuration Base Class.

        Args:
        vocabs: Dictionary Mapping Field Names to Vocabularies.
                Must contain 'source' and 'target' keys
        """
        self.source_vocab_size = len(vocabs[const.SOURCE])
        self.target_vocab_size = len(vocabs[const.TARGET])

    @classmethod
    def from_dict(cls, config_dict, vocabs):
        """Create config from a saved state_dict.
           Args:
             config_dict: A dictionary that is the return value of
                          a call to the `state_dict()` method of `cls`
             vocab: See `ModelConfig.__init__`
        """
        config = cls(vocabs)
        config.update(config_dict)
        return config

    def update(self, other_config):
        """Updates the config object with the values of `other_config`
           Args:
             other_config: The `dict` or `ModelConfig` object to update with.
        """
        config_dict = dict()
        if isinstance(self, other_config.__class__):
            config_dict = other_config.__dict__
        elif isinstance(other_config, dict):
            config_dict = other_config
        self.__dict__.update(config_dict)

    def state_dict(self):
        """Return the __dict__ for serialization.
        """
        self.__dict__['__version__'] = kiwi.__version__
        return self.__dict__


def load_torch_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError('Torch file not found: {}'.format(file_path))
    file_dict = torch.load(str(file_path), map_location=lambda storage, loc: storage)
    if isinstance(file_dict, Path):
        linked_path = file_dict
        if not linked_path.exists():
            relative_path = file_path.with_name(file_dict.name) / const.MODEL_FILE
            if relative_path.exists():
                linked_path = relative_path
        return load_torch_file(linked_path)
    return file_dict


class Attention(nn.Module):
    """Generic Attention Implementation.
       Module computes a convex combination of a set of values based on the fit
       of their keys with a query.
    """

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.mask = None

    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        scores = self.scorer(query, keys)
        scores = scores - scores.mean(1, keepdim=True)
        scores = torch.exp(scores)
        if self.mask is not None:
            scores = self.mask * scores
        convex = scores / scores.sum(1, keepdim=True)
        return torch.einsum('bs,bsi->bi', [convex, values])

    def set_mask(self, mask):
        self.mask = mask


class Scorer(nn.Module):
    """Score function for Attention module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, keys):
        """Computes Scores for each key given the query.
           args:
                 query:  FloatTensor batch x n
                 keys:   FloatTensor batch x seq_length x m
           ret:
                 scores: FloatTensor batch x seq_length
        """
        raise NotImplementedError

