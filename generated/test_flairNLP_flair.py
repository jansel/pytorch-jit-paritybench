import sys
_module = sys.modules[__name__]
del sys
flair = _module
data = _module
data_fetcher = _module
datasets = _module
base = _module
document_classification = _module
sequence_labeling = _module
text_image = _module
text_text = _module
treebanks = _module
embeddings = _module
base = _module
document = _module
image = _module
legacy = _module
token = _module
file_utils = _module
hyperparameter = _module
param_selection = _module
parameter = _module
inference_utils = _module
models = _module
language_model = _module
sequence_tagger_model = _module
similarity_learning_model = _module
text_classification_model = _module
text_regression_model = _module
nn = _module
optim = _module
samplers = _module
trainers = _module
language_model_trainer = _module
trainer = _module
training_utils = _module
visual = _module
activations = _module
manifold = _module
ner_html = _module
training_curves = _module
predict = _module
setup = _module
conftest = _module
test_data = _module
test_datasets = _module
test_embeddings = _module
test_hyperparameter = _module
test_language_model = _module
test_sequence_tagger = _module
test_text_classifier = _module
test_text_regressor = _module
test_utils = _module
test_visual = _module
train = _module

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


from abc import abstractmethod


from typing import Union


from typing import List


from torch.nn import ParameterList


from torch.nn import Parameter


import torch


import logging


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


from torch.nn import Sequential


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import ReLU


from torch.nn import MaxPool2d


from torch.nn import Dropout2d


from torch.nn import AdaptiveAvgPool2d


from torch.nn import AdaptiveMaxPool2d


from torch.nn import TransformerEncoderLayer


from torch.nn import TransformerEncoder


from typing import Tuple


from typing import Dict


from collections import Counter


from functools import lru_cache


import re


import numpy as np


import torch.nn as nn


import math


from torch.optim import Optimizer


from typing import Optional


from typing import Callable


import torch.nn


from torch.nn.parameter import Parameter


from torch.utils.data import DataLoader


from torch import nn


import itertools


import warnings


import random


from torch import cuda


from torch.utils.data import Dataset


from torch.optim.sgd import SGD


import copy


import inspect


from torch.utils.data.dataset import ConcatDataset


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, value: str, score: float=1.0):
        self.value = value
        self.score = score
        super().__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != '':
            raise ValueError(
                'Incorrect label value provided. Label value needs to be set.')
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if 0.0 <= score <= 1.0:
            self._score = score
        else:
            self._score = 1.0

    def to_dict(self):
        return {'value': self.value, 'confidence': self.score}

    def __str__(self):
        return f'{self._value} ({round(self._score, 4)})'

    def __repr__(self):
        return f'{self._value} ({round(self._score, 4)})'


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
    def to(self, device: str, pin_memory: bool=False):
        pass

    @abstractmethod
    def clear_embeddings(self, embedding_names: List[str]=None):
        pass

    def add_label(self, label_type: str, value: str, score: float=1.0):
        if label_type not in self.annotation_layers:
            self.annotation_layers[label_type] = [Label(value, score)]
        else:
            self.annotation_layers[label_type].append(Label(value, score))
        return self

    def set_label(self, label_type: str, value: str, score: float=1.0):
        self.annotation_layers[label_type] = [Label(value, score)]
        return self

    def get_labels(self, label_type: str=None):
        if label_type is None:
            return self.labels
        return self.annotation_layers[label_type
            ] if label_type in self.annotation_layers else []

    @property
    def labels(self) ->List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels


class Image(DataPoint):

    def __init__(self, data=None, imageURL=None):
        super().__init__()
        self.data = data
        self._embeddings: Dict = {}
        self.imageURL = imageURL

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self):
        image_repr = self.data.size() if self.data else ''
        image_url = self.imageURL if self.imageURL else ''
        return f'Image: {image_repr} {image_url}'

    def get_embedding(self) ->torch.tensor:
        embeddings = [self._embeddings[embed] for embed in sorted(self.
            _embeddings.keys())]
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.tensor([], device=flair.device)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.
            keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True)

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]


log = logging.getLogger('flair')


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors.
    This method was proposed by Liu et al. (2019) in the paper:
    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)

    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.
    It can be found under:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.
    """

    def __init__(self, mixture_size: int, trainable: bool=False) ->None:
        """
        Inits scalar mix implementation.
        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
        :param mixture_size: size of mixtures (usually the number of layers)
        """
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        initial_scalar_parameters = [0.0] * mixture_size
        self.scalar_parameters = ParameterList([Parameter(torch.tensor([
            initial_scalar_parameters[i]], dtype=torch.float, device=flair.
            device), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.tensor([1.0], dtype=torch.float,
            device=flair.device), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) ->torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error(
                '{} tensors were passed, but the module was initialized to mix {} tensors.'
                .format(len(tensors), self.mixture_size))
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for
            parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class Result(object):

    def __init__(self, main_score: float, log_header: str, log_line: str,
        detailed_results: str):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.
                dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.
                dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.
            dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.
            dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_flairNLP_flair(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LockedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SimilarityLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(WordDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

