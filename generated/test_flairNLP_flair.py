import sys
_module = sys.modules[__name__]
del sys
run_ner = _module
flair = _module
data = _module
datasets = _module
base = _module
biomedical = _module
document_classification = _module
entity_linking = _module
ocr = _module
relation_extraction = _module
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
transformer = _module
file_utils = _module
hyperparameter = _module
param_selection = _module
parameter = _module
inference_utils = _module
models = _module
clustering = _module
entity_linker_model = _module
language_model = _module
lemmatizer_model = _module
multitask_model = _module
pairwise_classification_model = _module
regexp_tagger = _module
relation_classifier_model = _module
relation_extractor_model = _module
sequence_tagger_model = _module
sequence_tagger_utils = _module
bioes = _module
crf = _module
viterbi = _module
tars_model = _module
text_classification_model = _module
text_regression_model = _module
word_tagger_model = _module
nn = _module
decoder = _module
distance = _module
cosine = _module
euclidean = _module
hyperbolic = _module
dropout = _module
model = _module
multitask = _module
recurrent = _module
optim = _module
samplers = _module
splitter = _module
tokenization = _module
trainers = _module
language_model_trainer = _module
trainer = _module
training_utils = _module
visual = _module
activations = _module
manifold = _module
ner_html = _module
training_curves = _module
tree_printer = _module
setup = _module
tests = _module
conftest = _module
embedding_test_utils = _module
test_document_transform_word_embeddings = _module
test_flair_embeddings = _module
test_stacked_embeddings = _module
test_transformer_document_embeddings = _module
test_transformer_word_embeddings = _module
test_word_embeddings = _module
model_test_utils = _module
test_entity_linker = _module
test_relation_classifier = _module
test_relation_extractor = _module
test_sequence_tagger = _module
test_tars_classifier = _module
test_tars_ner = _module
test_text_classifier = _module
test_text_regressor = _module
test_word_tagger = _module
test_corpus_dictionary = _module
test_datasets = _module
test_datasets_biomedical = _module
test_hyperparameter = _module
test_labels = _module
test_language_model = _module
test_lemmatizer = _module
test_models = _module
test_multitask = _module
test_sentence = _module
test_tars = _module
test_tokenize_sentence = _module
test_trainer = _module
test_utils = _module
test_visual = _module

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


import inspect


import logging


import torch


import logging.config


import re


import typing


from abc import ABC


from abc import abstractmethod


from collections import Counter


from collections import defaultdict


from collections import namedtuple


from functools import lru_cache


from typing import Dict


from typing import Iterable


from typing import List


from typing import Optional


from typing import Union


from typing import cast


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.dataset import Subset


from typing import Generic


import torch.utils.data.dataloader


import copy


from typing import Any


from torch.utils.data import ConcatDataset


import numpy as np


from typing import Sequence


from torch.nn import Parameter


from torch.nn import ParameterList


from sklearn.feature_extraction.text import TfidfVectorizer


from torch.nn import RNNBase


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


from torch.nn import AdaptiveAvgPool2d


from torch.nn import AdaptiveMaxPool2d


from torch.nn import Conv2d


from torch.nn import Dropout2d


from torch.nn import Linear


from torch.nn import MaxPool2d


from torch.nn import ReLU


from torch.nn import Sequential


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


from typing import Tuple


from torch import nn


import random


import warnings


from typing import Type


from torch.jit import ScriptModule


from typing import Callable


import math


import torch.nn as nn


from torch import logsumexp


from torch.optim import Optimizer


from math import inf


import itertools


from typing import Iterator


from typing import NamedTuple


from typing import Set


from torch.utils.data.dataset import Dataset


import torch.nn


from torch.nn.functional import softmax


from collections import OrderedDict


from sklearn.metrics.pairwise import cosine_similarity


from sklearn.preprocessing import minmax_scale


from torch import Tensor


from torch.nn.modules.loss import _Loss


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import required


from torch.utils.data.sampler import Sampler


import time


from torch import cuda


from torch.optim import AdamW


from torch.optim.sgd import SGD


from torch.utils.data import DataLoader


from inspect import signature


from torch.optim.lr_scheduler import OneCycleLR


from enum import Enum


from functools import reduce


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import mean_absolute_error


from sklearn.metrics import mean_squared_error


from torch.optim import SGD


from torch.optim import Adam


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, data_point: 'DataPoint', value: str, score: float=1.0):
        self._value = value
        self._score = score
        self.data_point: DataPoint = data_point
        super().__init__()

    def set_value(self, value: str, score: float=1.0):
        self.value = value
        self.score = score

    @property
    def value(self) ->str:
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != '':
            raise ValueError('Incorrect label value provided. Label value needs to be set.')
        else:
            self._value = value

    @property
    def score(self) ->float:
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    def to_dict(self):
        return {'value': self.value, 'confidence': self.score}

    def __str__(self):
        return f'{self.data_point.unlabeled_identifier}{flair._arrow}{self._value} ({round(self._score, 4)})'

    @property
    def shortstring(self):
        return f'"{self.data_point.text}"/{self._value}'

    def __repr__(self):
        return f"'{self.data_point.unlabeled_identifier}'/'{self._value}' ({round(self._score, 4)})"

    def __eq__(self, other):
        return self.value == other.value and self.score == other.score and self.data_point == other.data_point

    def __hash__(self):
        return hash(self.__repr__())

    def __lt__(self, other):
        return self.data_point < other.data_point

    @property
    def labeled_identifier(self):
        return f'{self.data_point.unlabeled_identifier}/{self.value}'

    @property
    def unlabeled_identifier(self):
        return f'{self.data_point.unlabeled_identifier}'


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._metadata: Dict[str, typing.Any] = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    def set_embedding(self, name: str, vector: torch.Tensor):
        self._embeddings[name] = vector

    def get_embedding(self, names: Optional[List[str]]=None) ->torch.Tensor:
        if names and len(names) == 1:
            if names[0] in self._embeddings:
                return self._embeddings[names[0]]
            else:
                return torch.tensor([], device=flair.device)
        embeddings = self.get_each_embedding(names)
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            return torch.tensor([], device=flair.device)

    def get_each_embedding(self, embedding_names: Optional[List[str]]=None) ->List[torch.Tensor]:
        embeddings = []
        for embed_name in sorted(self._embeddings.keys()):
            if embedding_names and embed_name not in embedding_names:
                continue
            embed = self._embeddings[embed_name]
            embeddings.append(embed)
        return embeddings

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.pin_memory()
                else:
                    self._embeddings[name] = vector

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def has_label(self, type) ->bool:
        if type in self.annotation_layers:
            return True
        else:
            return False

    def add_metadata(self, key: str, value: typing.Any) ->None:
        self._metadata[key] = value

    def get_metadata(self, key: str) ->typing.Any:
        return self._metadata[key]

    def has_metadata(self, key: str) ->bool:
        return key in self._metadata

    def add_label(self, typename: str, value: str, score: float=1.0):
        if typename not in self.annotation_layers:
            self.annotation_layers[typename] = [Label(self, value, score)]
        else:
            self.annotation_layers[typename].append(Label(self, value, score))
        return self

    def set_label(self, typename: str, value: str, score: float=1.0):
        self.annotation_layers[typename] = [Label(self, value, score)]
        return self

    def remove_labels(self, typename: str):
        if typename in self.annotation_layers.keys():
            del self.annotation_layers[typename]

    def get_label(self, label_type: str=None, zero_tag_value='O'):
        if len(self.get_labels(label_type)) == 0:
            return Label(self, zero_tag_value)
        return self.get_labels(label_type)[0]

    def get_labels(self, typename: str=None):
        if typename is None:
            return self.labels
        return self.annotation_layers[typename] if typename in self.annotation_layers else []

    @property
    def labels(self) ->List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels

    @property
    @abstractmethod
    def unlabeled_identifier(self):
        raise NotImplementedError

    def _printout_labels(self, main_label=None, add_score: bool=True):
        all_labels = []
        keys = [main_label] if main_label is not None else self.annotation_layers.keys()
        if add_score:
            for key in keys:
                all_labels.extend([f'{label.value} ({round(label.score, 4)})' for label in self.get_labels(key) if label.data_point == self])
            labels = '; '.join(all_labels)
            if labels != '':
                labels = flair._arrow + labels
        else:
            for key in keys:
                all_labels.extend([f'{label.value}' for label in self.get_labels(key) if label.data_point == self])
            labels = '/'.join(all_labels)
            if labels != '':
                labels = '/' + labels
        return labels

    def __str__(self) ->str:
        return self.unlabeled_identifier + self._printout_labels()

    @property
    @abstractmethod
    def start_position(self) ->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def end_position(self) ->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def text(self):
        raise NotImplementedError

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score

    def __lt__(self, other):
        return self.start_position < other.start_position

    def __eq__(self, other):
        return self.unlabeled_identifier == other.unlabeled_identifier

    def __hash__(self):
        return hash(self.unlabeled_identifier)

    def __len__(self):
        raise NotImplementedError


DT = typing.TypeVar('DT', bound=DataPoint)


class Embeddings(torch.nn.Module, Generic[DT]):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    def __init__(self):
        """Set some attributes that would otherwise result in errors. Overwrite these in your embedding class."""
        if not hasattr(self, 'name'):
            self.name: str = 'unnamed_embedding'
        if not hasattr(self, 'static_embeddings'):
            self.static_embeddings = False
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_type(self) ->str:
        raise NotImplementedError

    def embed(self, data_points: Union[DT, List[DT]]) ->List[DT]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""
        if not isinstance(data_points, list):
            data_points = [data_points]
        if not self._everything_embedded(data_points) or not self.static_embeddings:
            self._add_embeddings_internal(data_points)
        return data_points

    def _everything_embedded(self, data_points: Sequence[DT]) ->bool:
        for data_point in data_points:
            if self.name not in data_point._embeddings.keys():
                return False
        return True

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[DT]):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    def get_names(self) ->List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        return [self.name]

    def get_named_embeddings_dict(self) ->Dict:
        return {self.name: self}

    @staticmethod
    def get_instance_parameters(locals: dict) ->dict:
        class_definition = locals.get('__class__')
        instance_parameter_names = set(inspect.signature(class_definition.__init__).parameters)
        instance_parameter_names.remove('self')
        instance_parameter_names.add('__class__')
        instance_parameters = {class_attribute: attribute_value for class_attribute, attribute_value in locals.items() if class_attribute in instance_parameter_names}
        return instance_parameters


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
        self.scalar_parameters = ParameterList([Parameter(torch.tensor([initial_scalar_parameters[i]], dtype=torch.float, device=flair.device), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.tensor([1.0], dtype=torch.float, device=flair.device), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) ->torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error('{} tensors were passed, but the module was initialized to mix {} tensors.'.format(len(tensors), self.mixture_size))
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter in self.scalar_parameters]), dim=0)
        normed_weights_split = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights_split, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


rnn_layers = {'lstm': (nn.LSTM, 2), 'gru': (nn.GRU, 1)}


def create_recurrent_layer(layer_type, initial_size, hidden_size, nlayers, dropout=0, **kwargs):
    layer_type = layer_type.lower()
    assert layer_type in rnn_layers.keys()
    module, hidden_count = rnn_layers[layer_type]
    if nlayers == 1:
        dropout = 0
    return module(initial_size, hidden_size, nlayers, dropout=dropout, **kwargs), hidden_count


class CRF(torch.nn.Module):
    """
    Conditional Random Field Implementation according to sgrvinod (https://github.com/sgrvinod).
    Classifier which predicts single tag / class / label for given word based on not just the word,
    but also on previous seen annotations.
    """

    def __init__(self, tag_dictionary, tagset_size: int, init_from_state_dict: bool):
        """
        :param tag_dictionary: tag dictionary in order to find ID for start and stop tags
        :param tagset_size: number of tag from tag dictionary
        :param init_from_state_dict: whether we load pretrained model from state dict
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = torch.nn.Parameter(torch.randn(tagset_size, tagset_size))
        if not init_from_state_dict:
            self.transitions.detach()[tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            self.transitions.detach()[:, tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000
        self

    def forward(self, features: torch.Tensor) ->torch.Tensor:
        """
        Forward propagation of Conditional Random Field.
        :param features: output from RNN / Linear layer in shape (batch size, seq len, hidden size)
        :return: CRF scores (emission scores for each token + transitions prob from previous state) in
        shape (batch_size, seq len, tagset size, tagset size)
        """
        batch_size, seq_len = features.size()[:2]
        emission_scores = features
        emission_scores = emission_scores.unsqueeze(-1).expand(batch_size, seq_len, self.tagset_size, self.tagset_size)
        crf_scores = emission_scores + self.transitions.unsqueeze(0).unsqueeze(0)
        return crf_scores


def dot_product(a: torch.Tensor, b: torch.Tensor, normalize=False):
    """
    Computes dot product for pairs of vectors.
    :param normalize: Vectors are normalized (leads to cosine similarity)
    :return: Matrix with res[i][j]  = dot_product(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    if normalize:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))


class CosineDistance(torch.nn.Module):

    def forward(self, a, b):
        return -dot_product(a, b, normalize=True)


class EuclideanDistance(nn.Module):
    """Implement a EuclideanDistance object."""

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        _dist = [torch.sum((mat_1 - mat_2[i]) ** 2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist


EPSILON = 1e-05


def arccosh(x):
    """Compute the arcosh, numerically stable."""
    x = torch.clamp(x, min=1 + EPSILON)
    a = torch.log(x)
    b = torch.log1p(torch.sqrt(x * x - 1) / x)
    return a + b


class HyperbolicDistance(nn.Module):
    """Implement a HyperbolicDistance object."""

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        mat_1_x_0 = torch.sqrt(1 + mat_1.pow(2).sum(dim=1, keepdim=True))
        mat_2_x_0 = torch.sqrt(1 + mat_2.pow(2).sum(dim=1, keepdim=True))
        left = mat_1_x_0.mm(mat_2_x_0.t())
        right = mat_1[:, 1:].mm(mat_2[:, 1:].t())
        return arccosh(left - right).pow(2)


class LogitCosineDistance(torch.nn.Module):

    def forward(self, a, b):
        return torch.logit(0.5 - 0.5 * dot_product(a, b, normalize=True))


class NegativeScaledDotProduct(torch.nn.Module):

    def forward(self, a, b):
        sqrt_d = torch.sqrt(torch.tensor(a.size(-1)))
        return -dot_product(a, b, normalize=False) / sqrt_d


class PrototypicalDecoder(torch.nn.Module):

    def __init__(self, num_prototypes: int, embeddings_size: int, prototype_size: Optional[int]=None, distance_function: str='euclidean', use_radius: Optional[bool]=False, min_radius: Optional[int]=0, unlabeled_distance: Optional[float]=None, unlabeled_idx: Optional[int]=None, learning_mode: Optional[str]='joint', normal_distributed_initial_prototypes: bool=False):
        super().__init__()
        if not prototype_size:
            prototype_size = embeddings_size
        self.prototype_size = prototype_size
        self.metric_space_decoder: Optional[torch.nn.Linear] = None
        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size, prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)
        self.prototype_vectors = torch.nn.Parameter(torch.ones(num_prototypes, prototype_size), requires_grad=True)
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(num_prototypes, prototype_size)))
        self.prototype_radii: Optional[torch.nn.Parameter] = None
        if use_radius:
            self.prototype_radii = torch.nn.Parameter(torch.ones(num_prototypes), requires_grad=True)
        self.min_radius = min_radius
        self.learning_mode = learning_mode
        assert (unlabeled_idx is None) == (unlabeled_distance is None), "'unlabeled_idx' and 'unlabeled_distance' should either both be set or both not be set."
        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_distance = unlabeled_distance
        self._distance_function = distance_function
        self.distance: Optional[torch.nn.Module] = None
        if distance_function.lower() == 'hyperbolic':
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == 'cosine':
            self.distance = CosineDistance()
        elif distance_function.lower() == 'logit_cosine':
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == 'euclidean':
            self.distance = EuclideanDistance()
        elif distance_function.lower() == 'dot_product':
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f'Distance function {distance_function} not found.')
        self

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded):
        if self.learning_mode == 'learn_only_map_and_prototypes':
            embedded = embedded.detach()
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded
        prot = self.prototype_vectors
        radii = self.prototype_radii
        if self.learning_mode == 'learn_only_prototypes':
            encoded = encoded.detach()
        if self.learning_mode == 'learn_only_embeddings_and_map':
            prot = prot.detach()
            if radii is not None:
                radii = radii.detach()
        distance = self.distance(encoded, prot)
        if radii is not None:
            distance /= self.min_radius + torch.nn.functional.softplus(radii)
        if self.unlabeled_distance:
            distance[..., self.unlabeled_idx] = self.unlabeled_distance
        scores = -distance
        return scores


class EuclideanMean(nn.Module):
    """Implement a EuclideanMean object."""

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        return data.mean(0)


def mdot(x, y):
    """Compute the inner product."""
    m = x.new_ones(1, x.size(1))
    m[0, 0] = -1
    return torch.sum(m * x * y, 1, keepdim=True)


def norm(x):
    """Compute the norm"""
    n = torch.sqrt(torch.abs(mdot(x, x)))
    return n


def exp_map(x, y):
    """Perform the exp step."""
    n = torch.clamp(norm(y), min=EPSILON)
    return torch.cosh(n) * x + torch.sinh(n) / n * y


def dist(x, y):
    """Get the hyperbolic distance between x and y."""
    return arccosh(-mdot(x, y))


def log_map(x, y):
    """Perform the log step."""
    d = dist(x, y)
    return d / torch.sinh(d) * (y - torch.cosh(d) * x)


def project(x):
    """Project onto the hyeprboloid embedded in in n+1 dimensions."""
    return torch.cat([torch.sqrt(1.0 + torch.sum(x * x, 1, keepdim=True)), x], 1)


class HyperbolicMean(nn.Module):
    """Compute the mean point in the hyperboloid model."""

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        n_iter = 5 if self.training else 100
        projected = project(data)
        mean = torch.mean(projected, 0, keepdim=True)
        mean = mean / norm(mean)
        r = 0.01
        for i in range(n_iter):
            g = -2 * torch.mean(log_map(mean, projected), 0, keepdim=True)
            mean = exp_map(mean, -r * g)
            mean = mean / norm(mean)
        return mean.squeeze()[1:]


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
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
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
        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


class Result(object):

    def __init__(self, main_score: float, log_header: str, log_line: str, detailed_results: str, loss: float, classification_report: dict={}):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results
        self.classification_report = classification_report
        self.loss: float = loss

    def __str__(self):
        return f"{str(self.detailed_results)}\nLoss: {self.loss}'"


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CRF,
     lambda: ([], {'tag_dictionary': 4, 'tagset_size': 4, 'init_from_state_dict': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CosineDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (EuclideanDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EuclideanMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HyperbolicDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (HyperbolicMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 5])], {}),
     False),
    (LockedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogitCosineDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (NegativeScaledDotProduct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (PrototypicalDecoder,
     lambda: ([], {'num_prototypes': 4, 'embeddings_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WordDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_flairNLP_flair(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

