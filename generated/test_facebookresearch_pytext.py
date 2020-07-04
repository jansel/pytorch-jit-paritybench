import sys
_module = sys.modules[__name__]
del sys
data_processor = _module
source = _module
tensorizer = _module
atis = _module
server = _module
metric = _module
model = _module
my_tagging_task = _module
output = _module
pytext = _module
builtin_task = _module
common = _module
constants = _module
utils = _module
config = _module
component = _module
config_adapter = _module
contextual_intent_slot = _module
doc_classification = _module
field_config = _module
module_config = _module
pair_classification = _module
pytext_config = _module
query_document_pairwise_ranking = _module
serialize = _module
component_test = _module
config_adapter_test = _module
pytext_all_config_test = _module
pytext_config_test = _module
serialize_test = _module
data = _module
batch_sampler = _module
bert_tensorizer = _module
data_handler = _module
data_structures = _module
annotation = _module
node = _module
annotation_test = _module
dense_retrieval_tensorizer = _module
disjoint_multitask_data = _module
disjoint_multitask_data_handler = _module
dynamic_pooling_batcher = _module
featurizer = _module
simple_featurizer = _module
packed_lm_data = _module
roberta_tensorizer = _module
sources = _module
conllu = _module
data_source = _module
dense_retrieval = _module
pandas = _module
session = _module
squad = _module
tsv = _module
squad_for_bert_tensorizer = _module
squad_tensorizer = _module
tensorizers = _module
test = _module
batch_sampler_test = _module
data_test = _module
dynamic_pooling_batcher_test = _module
pandas_data_source_test = _module
round_robin_batchiterator_test = _module
simple_featurizer_test = _module
tensorizers_test = _module
tokenizers_test = _module
tsv_data_source_test = _module
utils_test = _module
token_tensorizer = _module
tokenizers = _module
tokenizer = _module
xlm_constants = _module
xlm_dictionary = _module
xlm_tensorizer = _module
make_config_docs = _module
conf = _module
exporters = _module
custom_exporters = _module
exporter = _module
new_text_model_exporter_test = _module
text_model_exporter_test = _module
fields = _module
char_field = _module
contextual_token_embedding_field = _module
dict_field = _module
field = _module
char_field_test = _module
contextual_token_embedding_field_test = _module
dict_field_test = _module
field_test = _module
text_field_with_special_unk = _module
loss = _module
loss = _module
label_smoothing_loss_test = _module
main = _module
metric_reporters = _module
channel = _module
classification_metric_reporter = _module
compositional_metric_reporter = _module
dense_retrieval_metric_reporter = _module
disjoint_multitask_metric_reporter = _module
intent_slot_detection_metric_reporter = _module
language_model_metric_reporter = _module
metric_reporter = _module
pairwise_ranking_metric_reporter = _module
regression_metric_reporter = _module
seq2seq_compositional = _module
seq2seq_metric_reporter = _module
seq2seq_utils = _module
squad_metric_reporter = _module
compositional_metric_reporter_test = _module
intent_slot_metric_reporter_test = _module
language_model_metric_reporter_test = _module
tensorboard_test = _module
word_tagging_metric_reporter = _module
metrics = _module
dense_retrieval_metrics = _module
intent_slot_metrics = _module
language_model_metrics = _module
seq2seq_metrics = _module
squad_metrics = _module
basic_metrics_test = _module
intent_slot_metrics_test = _module
metrics_test_base = _module
models = _module
bert_classification_models = _module
bert_regression_model = _module
crf = _module
decoders = _module
decoder_base = _module
intent_slot_model_decoder = _module
mlp_decoder = _module
mlp_decoder_query_response = _module
multilabel_decoder = _module
disjoint_multitask_model = _module
distributed_model = _module
doc_model = _module
embeddings = _module
char_embedding = _module
contextual_token_embedding = _module
dict_embedding = _module
embedding_base = _module
embedding_list = _module
scriptable_embedding_list = _module
word_embedding = _module
word_seq_embedding = _module
ensembles = _module
bagging_doc_ensemble = _module
bagging_intent_slot_ensemble = _module
ensemble = _module
joint_model = _module
language_models = _module
lmlstm = _module
masked_lm = _module
masking_utils = _module
model = _module
module = _module
output_layers = _module
distance_output_layer = _module
doc_classification_output_layer = _module
doc_regression_output_layer = _module
intent_slot_output_layer = _module
lm_output_layer = _module
multi_label_classification_layer = _module
output_layer_base = _module
pairwise_ranking_output_layer = _module
squad_output_layer = _module
word_tagging_output_layer = _module
pair_classification_model = _module
qna = _module
bert_squad_qa = _module
dr_qa = _module
query_document_pairwise_ranking_model = _module
representations = _module
attention = _module
augmented_lstm = _module
bilstm = _module
bilstm_doc_attention = _module
bilstm_doc_slot_attention = _module
bilstm_slot_attn = _module
biseqcnn = _module
contextual_intent_slot_rep = _module
deepcnn = _module
docnn = _module
huggingface_bert_sentence_encoder = _module
jointcnn_rep = _module
ordered_neuron_lstm = _module
pair_rep = _module
pass_through = _module
pooling = _module
pure_doc_attention = _module
representation_base = _module
seq_rep = _module
slot_attention = _module
sparse_transformer_sentence_encoder = _module
stacked_bidirectional_rnn = _module
augmented_lstm_test = _module
ordered_neuron_lstm_test = _module
transformer_test = _module
traced_transformer_encoder = _module
transformer = _module
multihead_attention = _module
positional_embedding = _module
residual_mlp = _module
sentence_encoder = _module
transformer = _module
transformer_sentence_encoder = _module
transformer_sentence_encoder_base = _module
roberta = _module
semantic_parsers = _module
rnng = _module
rnng_constant = _module
rnng_data_structures = _module
rnng_parser = _module
seq_models = _module
attention = _module
base = _module
rnn_decoder = _module
rnn_encoder = _module
rnn_encoder_decoder = _module
seq2seq_model = _module
seq2seq_output_layer = _module
seqnn = _module
bilstm_test = _module
crf_test = _module
dict_embedding_test = _module
embedding_list_test = _module
module_test = _module
output_layer_test = _module
personalized_doc_model_test = _module
rnng_test = _module
scripted_seq2seq_generator_test = _module
transformer_sentence_encoder_test = _module
word_embedding_test = _module
word_seq_embedding_test = _module
word_model = _module
optimizer = _module
activations = _module
fairseq_fp16_utils = _module
fp16_optimizer = _module
lamb = _module
optimizers = _module
radam = _module
scheduler = _module
sparsifiers = _module
blockwise_sparsifier = _module
sparsifier = _module
sparsifier_test = _module
swa = _module
fp16optimizer_test = _module
test_swa = _module
resources = _module
task = _module
disjoint_multitask = _module
new_task = _module
tasks = _module
torchscript = _module
seq2seq = _module
beam_decode = _module
beam_search = _module
decoder = _module
encoder = _module
export_model = _module
scripted_seq2seq_generator = _module
seq2seq_rnn_decoder_utils = _module
bert = _module
normalizer = _module
xlm = _module
test_tensorizer = _module
test_tokenizer = _module
test_vocab = _module
bpe = _module
vocab = _module
trainers = _module
ensemble_trainer = _module
hogwild_trainer = _module
trainer = _module
training_state = _module
ascii_table = _module
config_utils = _module
cuda = _module
distributed = _module
documentation = _module
file_io = _module
label = _module
lazy = _module
meter = _module
mobile_onnx = _module
model = _module
onnx = _module
path = _module
precision = _module
tensor = _module
ascii_table_test = _module
embeddings_utils_test = _module
label_test = _module
lazy_test = _module
timing_test = _module
timing = _module
torch = _module
usage = _module
workflow = _module
setup = _module
tests = _module
data_utils = _module
main_test = _module
model_utils_test = _module
module_load_save_test = _module
predictor_test = _module
seq2seq_model_tests = _module
task_load_save_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


import collections


import enum


from typing import Any


from typing import Dict


from typing import List


from typing import Tuple


from typing import Type


from typing import Union


import copy


from typing import Optional


import numpy as np


from collections import Counter


from torchtext.vocab import Vocab


from torch import nn


from scipy.special import logsumexp


from torch.utils.tensorboard import SummaryWriter


import math


import time


from torch import optim


import torch.nn as nn


import torch.jit as jit


import torch.onnx.operators


from typing import Iterable


from torch.nn import ModuleList


from inspect import signature


from copy import deepcopy


import torch.jit


from torch.jit import quantized


from enum import IntEnum


from enum import unique


from torch import jit


import functools


import itertools


from scipy.special import comb


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.onnx


from enum import Enum


import torch.cuda


from torch.nn import functional as F


import re


from torch.serialization import default_restore_location


from typing import Sized


import torch as torch


from torch import Tensor


from itertools import chain


from torch.optim import Optimizer as PT_Optimizer


import warnings


from collections import defaultdict


from torch.autograd import Variable


from torch import sparse


from torch.utils import data


import torch.jit.quantized


import torch.multiprocessing as mp


from torchtext.data import Iterator


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


class ConfigBaseMeta(type):

    def annotations_and_defaults(cls):
        annotations = OrderedDict()
        defaults = {}
        for base in reversed(cls.__bases__):
            if base is ConfigBase:
                continue
            annotations.update(getattr(base, '__annotations__', {}))
            defaults.update(getattr(base, '_field_defaults', {}))
        annotations.update(vars(cls).get('__annotations__', {}))
        defaults.update({k: getattr(cls, k) for k in annotations if hasattr
            (cls, k)})
        return annotations, defaults

    @property
    def __annotations__(cls):
        annotations, _ = cls.annotations_and_defaults()
        return annotations
    _field_types = __annotations__

    @property
    def _fields(cls):
        return cls.__annotations__.keys()

    @property
    def _field_defaults(cls):
        _, defaults = cls.annotations_and_defaults()
        return defaults


class ConfigBase(metaclass=ConfigBaseMeta):

    def items(self):
        return self._asdict().items()

    def _asdict(self):
        return {k: getattr(self, k) for k in type(self).__annotations__}

    def _replace(self, **kwargs):
        args = self._asdict()
        args.update(kwargs)
        return type(self)(**args)

    def __init__(self, **kwargs):
        """Configs can be constructed by specifying values by keyword.
        If a keyword is supplied that isn't in the config, or if a config requires
        a value that isn't specified and doesn't have a default, a TypeError will be
        raised."""
        specified = kwargs.keys() | type(self)._field_defaults.keys()
        required = type(self).__annotations__.keys()
        unspecified_fields = required - specified
        if unspecified_fields:
            raise TypeError(
                f'Failed to specify {unspecified_fields} for {type(self)}')
        overspecified_fields = specified - required
        if overspecified_fields:
            raise TypeError(
                f'Specified non-existent fields {overspecified_fields} for {type(self)}'
                )
        vars(self).update(kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in sorted(self._asdict().items()):
            lines += f'{key}: {val}'.split('\n')
        return '\n    '.join(lines)

    def __eq__(self, other):
        """Mainly a convenience utility for unit testing."""
        return type(self) == type(other) and self._asdict() == other._asdict()


CUDA_ENABLED = False


def FloatTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


class ComponentType(enum.Enum):
    TASK = 'task'
    COLUMN = 'column'
    DATA_TYPE = 'data_type'
    DATA_HANDLER = 'data_handler'
    DATA_SOURCE = 'data_source'
    TOKENIZER = 'tokenizer'
    TENSORIZER = 'tensorizer'
    BATCHER = 'batcher'
    BATCH_SAMPLER = 'batch_sampler'
    FEATURIZER = 'featurizer'
    TRAINER = 'trainer'
    LOSS = 'loss'
    OPTIMIZER = 'optimizer'
    SCHEDULER = 'scheduler'
    MODEL = 'model'
    MODEL2 = 'model2'
    MODULE = 'module'
    PREDICTOR = 'predictor'
    EXPORTER = 'exporter'
    METRIC_REPORTER = 'metric_reporter'
    SPARSIFIER = 'sparsifier'
    MASKING_FUNCTION = 'masking_function'


class RegistryError(Exception):
    pass


class Registry:
    _registered_components: Dict[ComponentType, Dict[Type, Type]
        ] = collections.defaultdict(dict)

    @classmethod
    def add(cls, component_type: ComponentType, cls_to_add: Type,
        config_cls: Type):
        component = cls._registered_components[component_type]
        if config_cls in component:
            raise RegistryError(
                f"Cannot add {cls_to_add} to {component_type} for task_config type {config_cls}; it's already registered for {component[config_cls]}"
                )
        component[config_cls] = cls_to_add

    @classmethod
    def get(cls, component_type: ComponentType, config_cls: Type) ->Type:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        if config_cls not in cls._registered_components[component_type]:
            raise RegistryError(
                f'unregistered config class {config_cls.__name__} for {component_type}'
                )
        return cls._registered_components[component_type][config_cls]

    @classmethod
    def values(cls, component_type: ComponentType) ->Tuple[Type, ...]:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        return tuple(cls._registered_components[component_type].values())

    @classmethod
    def configs(cls, component_type: ComponentType) ->Tuple[Type, ...]:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        return tuple(cls._registered_components[component_type].keys())

    @classmethod
    def subconfigs(cls, config_cls: Type) ->Tuple[Type, ...]:
        return tuple(sub_cls for sub_cls in cls.configs(config_cls.
            __COMPONENT_TYPE__) if issubclass(sub_cls.__COMPONENT__,
            config_cls.__COMPONENT__))


class ComponentMeta(type):

    def __new__(metacls, typename, bases, namespace):
        if 'Config' not in namespace:
            parent_config = next((base.Config for base in bases if hasattr(
                base, 'Config')), None)
            if parent_config is not None:


                class Config(parent_config):
                    pass
            else:


                class Config(ConfigBase):
                    pass
            namespace['Config'] = Config
        component_type = next((base.__COMPONENT_TYPE__ for base in bases if
            hasattr(base, '__COMPONENT_TYPE__')), namespace.get(
            '__COMPONENT_TYPE__'))
        new_cls = super().__new__(metacls, typename, bases, namespace)
        new_cls.Config.__COMPONENT_TYPE__ = component_type
        new_cls.Config.__name__ = f'{typename}.Config'
        new_cls.Config.__COMPONENT__ = new_cls
        new_cls.Config.__EXPANSIBLE__ = namespace.get('__EXPANSIBLE__')
        if component_type:
            Registry.add(component_type, new_cls, new_cls.Config)
        return new_cls

    def __dir__(cls):
        """Jit doesnt allow scripting of attributes whose classname includes "."

        Example Repro:

        class OldModule(Module):
            class Config(ConfigBase):
                a: int = 5

            @classmethod
            def from_config(cls, config: Config):
                return cls(config.a)

            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, b: int) -> int:
                return b + self.a

        m = OldModule.from_config(OldModule.Config())
        jit.script(m)

            > RuntimeError: Could not get qualified name for class 'OldModule.Config':
        'OldModule.Config' is not a valid identifier

        print(m.Config.__name__)
            > OldModule.Config

        At the sametime, we dont need to script the config classes because they
        are not needed during inference time. Hence in this workaround we skip
        the config classes.

        Ideal solution is that when building models they should be inheriting
        from nn.Module only and not Component. This requires significant changes
        to the way models are created in PyText.

        """
        result = super().__dir__()
        return [r for r in result if not (isinstance(getattr(cls, r, None),
            type) and issubclass(getattr(cls, r, None), ConfigBase))]


class Component(metaclass=ComponentMeta):


    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(config, *args, **kwargs)

    def __init__(self, config=None, *args, **kwargs):
        self.config = config


class Loss(Component):
    """Base class for loss functions"""
    __COMPONENT_TYPE__ = ComponentType.LOSS

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config)

    def __call__(self, logit, targets, reduce=True):
        raise NotImplementedError


class AUCPRHingeLoss(nn.Module, Loss):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5     TensorFlow Implementation:     https://github.com/tensorflow/models/tree/master/research/global_objectives    """


    class Config(ConfigBase):
        """
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        """
        precision_range_lower: float = 0.0
        precision_range_upper: float = 1.0
        num_classes: int = 1
        num_anchors: int = 20

    def __init__(self, config, weights=None, *args, **kwargs):
        """Args:
            config: Config containing `precision_range_lower`, `precision_range_upper`,
                `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)
        Loss.__init__(self, config)
        self.num_classes = self.config.num_classes
        self.num_anchors = self.config.num_anchors
        self.precision_range = (self.config.precision_range_lower, self.
            config.precision_range_upper)
        self.precision_values, self.delta = (loss_utils.
            range_to_anchors_and_delta(self.precision_range, self.num_anchors))
        self.biases = nn.Parameter(FloatTensor(self.config.num_classes,
            self.config.num_anchors).zero_())
        self.lambdas = nn.Parameter(FloatTensor(self.config.num_classes,
            self.config.num_anchors).data.fill_(1.0))

    def forward(self, logits, targets, reduce=True, size_average=True,
        weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)
        if self.num_classes != C:
            raise ValueError('num classes is %d while logits width is %d' %
                (self.num_classes, C))
        labels, weights = AUCPRHingeLoss._prepare_labels_weights(logits,
            targets, weights=weights)
        lambdas = loss_utils.lagrange_multiplier(self.lambdas)
        hinge_loss = loss_utils.weighted_hinge_loss(labels.unsqueeze(-1), 
            logits.unsqueeze(-1) - self.biases, positive_weights=1.0 + 
            lambdas * (1.0 - self.precision_values), negative_weights=
            lambdas * self.precision_values)
        class_priors = loss_utils.build_class_priors(labels, weights=weights)
        lambda_term = class_priors.unsqueeze(-1) * (lambdas * (1.0 - self.
            precision_values))
        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]
        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        labels = FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).
            data, 1)
        if weights is None:
            weights = FloatTensor(N).data.fill_(1.0)
        if weights.dim() == 1:
            weights.unsqueeze_(-1)
        return labels, weights


class FCModelWithNanAndInfWts(nn.Module):
    """ Simple FC model
    """

    def __init__(self):
        super(FCModelWithNanAndInfWts, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc1.weight.data.fill_(float('NaN'))
        self.fc2.weight.data.fill_(float('Inf'))

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


def log_class_usage(klass):
    identifier = 'PyText'
    if klass and hasattr(klass, '__name__'):
        identifier += f'.{klass.__name__}'
    torch._C._log_api_usage_once(identifier)


class CRF(nn.Module):
    """
    Compute the log-likelihood of the input assuming a conditional random field
    model.

    Args:
        num_tags: The number of tags
    """

    def __init__(self, num_tags: int, ignore_index: int,
        default_label_pad_index: int) ->None:
        if num_tags <= 0:
            raise ValueError(f'Invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags + 2, num_tags +
            2))
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        self.reset_parameters()
        self.ignore_index = ignore_index
        self.default_label_pad_index = default_label_pad_index
        log_class_usage(__class__)

    def reset_parameters(self) ->None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, (self.start_tag)] = -10000
        self.transitions.data[(self.end_tag), :] = -10000

    def get_transitions(self):
        return self.transitions.data

    def set_transitions(self, transitions: torch.Tensor=None):
        self.transitions.data = transitions

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, reduce:
        bool=True) ->torch.Tensor:
        """
        Compute log-likelihood of input.

        Args:
            emissions: Emission values for different tags for each input. The
                expected shape is batch_size * seq_len * num_labels. Padding is
                should be on the right side of the input.
            tags: Actual tags for each token in the input. Expected shape is
                batch_size * seq_len
        """
        mask = self._make_mask_from_targets(tags)
        numerator = self._compute_joint_llh(emissions, tags, mask)
        denominator = self._compute_log_partition_function(emissions, mask)
        llh = numerator - denominator
        return llh if not reduce else torch.mean(llh)

    @jit.export
    def decode(self, emissions: torch.Tensor, seq_lens: torch.Tensor
        ) ->torch.Tensor:
        """
        Given a set of emission probabilities, return the predicted tags.

        Args:
            emissions: Emission probabilities with expected shape of
                batch_size * seq_len * num_labels
            seq_lens: Length of each input.
        """
        mask = self._make_mask_from_seq_lens(seq_lens)
        result = self._viterbi_decode(emissions, mask)
        return result

    def _compute_joint_llh(self, emissions: torch.Tensor, tags: torch.
        Tensor, mask: torch.Tensor) ->torch.Tensor:
        seq_len = emissions.shape[1]
        llh = self.transitions[self.start_tag, tags[:, (0)]].unsqueeze(1)
        llh += emissions[:, (0), :].gather(1, tags[:, (0)].view(-1, 1)) * mask[
            :, (0)].unsqueeze(1)
        for idx in range(1, seq_len):
            old_state, new_state = tags[:, (idx - 1)].view(-1, 1), tags[:,
                (idx)].view(-1, 1)
            emission_scores = emissions[:, (idx), :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, (idx)
                ].unsqueeze(1)
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))
        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1
            )
        return llh.squeeze(1)

    def _compute_log_partition_function(self, emissions: torch.Tensor, mask:
        torch.Tensor) ->torch.Tensor:
        seq_len = emissions.shape[1]
        log_prob = emissions[:, (0)].clone()
        log_prob += self.transitions[(self.start_tag), :self.start_tag
            ].unsqueeze(0)
        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, (idx)].unsqueeze(1)
            broadcast_transitions = self.transitions[:self.start_tag, :self
                .start_tag].unsqueeze(0)
            broadcast_logprob = log_prob.unsqueeze(2)
            score = (broadcast_logprob + broadcast_emissions +
                broadcast_transitions)
            score = torch.logsumexp(score, 1)
            log_prob = score * mask[:, (idx)].unsqueeze(1) + log_prob.squeeze(1
                ) * (1 - mask[:, (idx)].unsqueeze(1))
        log_prob += self.transitions[:self.start_tag, (self.end_tag)
            ].unsqueeze(0)
        return torch.logsumexp(log_prob.squeeze(1), 1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor
        ) ->torch.Tensor:
        tensor_device = emissions.device
        seq_len = emissions.shape[1]
        mask = mask
        log_prob = emissions[:, (0)].clone()
        log_prob += self.transitions[(self.start_tag), :self.start_tag
            ].unsqueeze(0)
        end_scores = log_prob + self.transitions[:self.start_tag, (self.
            end_tag)].unsqueeze(0)
        best_scores_list: List[torch.Tensor] = []
        empty_data: List[int] = []
        best_paths_list = [torch.tensor(empty_data, device=tensor_device).
            long()]
        best_scores_list.append(end_scores.unsqueeze(1))
        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, (idx)].unsqueeze(1)
            broadcast_transmissions = self.transitions[:self.start_tag, :
                self.start_tag].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)
            score = (broadcast_emissions + broadcast_transmissions +
                broadcast_log_prob)
            max_scores, max_score_indices = torch.max(score, 1)
            best_paths_list.append(max_score_indices.unsqueeze(1))
            end_scores = max_scores + self.transitions[:self.start_tag, (
                self.end_tag)].unsqueeze(0)
            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores
        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)
        _, max_indices_from_scores = torch.max(best_scores, 2)
        valid_index_tensor = torch.tensor(0, device=tensor_device).long()
        if self.ignore_index == self.default_label_pad_index:
            padding_tensor = valid_index_tensor
        else:
            padding_tensor = torch.tensor(self.ignore_index, device=
                tensor_device).long()
        labels = max_indices_from_scores[:, (seq_len - 1)]
        labels = self._mask_tensor(labels, 1 - mask[:, (seq_len - 1)],
            padding_tensor)
        all_labels = labels.unsqueeze(1).long()
        for idx in range(seq_len - 2, -1, -1):
            indices_for_lookup = all_labels[:, (-1)].clone()
            indices_for_lookup = self._mask_tensor(indices_for_lookup, 
                indices_for_lookup == self.ignore_index, valid_index_tensor)
            indices_from_prev_pos = best_paths[:, (idx), :].gather(1,
                indices_for_lookup.view(-1, 1).long()).squeeze(1)
            indices_from_prev_pos = self._mask_tensor(indices_from_prev_pos,
                1 - mask[:, (idx + 1)], padding_tensor)
            indices_from_max_scores = max_indices_from_scores[:, (idx)]
            indices_from_max_scores = self._mask_tensor(indices_from_max_scores
                , mask[:, (idx + 1)], padding_tensor)
            labels = torch.where(indices_from_max_scores == self.
                ignore_index, indices_from_prev_pos, indices_from_max_scores)
            labels = self._mask_tensor(labels, 1 - mask[:, (idx)],
                padding_tensor)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)
        return torch.flip(all_labels, [1])

    def _make_mask_from_targets(self, targets):
        mask = targets.ne(self.ignore_index).float()
        return mask

    def _make_mask_from_seq_lens(self, seq_lens):
        seq_lens = seq_lens.view(-1, 1)
        max_len = torch.max(seq_lens)
        range_tensor = torch.arange(max_len, device=seq_lens.device).unsqueeze(
            0)
        range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.
            size(1))
        mask = (range_tensor < seq_lens).float()
        return mask

    def _mask_tensor(self, score_tensor, mask_condition, mask_value):
        masked_tensor = torch.where(mask_condition, mask_value, score_tensor)
        return masked_tensor

    def export_to_caffe2(self, workspace, init_net, predict_net,
        logits_output_name):
        """
        Exports the crf layer to caffe2 by manually adding the necessary operators
        to the init_net and predict net.

        Args:
            init_net: caffe2 init net created by the current graph
            predict_net: caffe2 net created by the current graph
            workspace: caffe2 current workspace
            output_names: current output names of the caffe2 net
            py_model: original pytorch model object

        Returns:
            string: The updated predictions blob name
        """
        crf_transitions = init_net.AddExternalInput(init_net.NextName())
        workspace.FeedBlob(str(crf_transitions), self.get_transitions().numpy()
            )
        logits_squeezed = predict_net.Squeeze(logits_output_name, dims=[0])
        new_logits = apply_crf(init_net, predict_net, crf_transitions,
            logits_squeezed, self.num_tags)
        new_logits = predict_net.ExpandDims(new_logits, dims=[0])
        predict_net.Copy(new_logits, logits_output_name)
        return logits_output_name


class Stage(Enum):
    TRAIN = 'Training'
    EVAL = 'Evaluation'
    TEST = 'Test'
    OTHERS = 'Others'


class DistributedModel(nn.parallel.DistributedDataParallel):
    """
    Wrapper model class to train models in distributed data parallel manner.
    The way to use this class to train your module in distributed manner is::

        distributed_model = DistributedModel(
            module=model,
            device_ids=[device_id0, device_id1],
            output_device=device_id0,
            broadcast_buffers=False,
        )


    where, `model` is the object of the actual model class you want to train in
    distributed manner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_class_usage(__class__)

    def __getattr__(self, name):
        wrapped_module = super().__getattr__('module')
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def cpu(self):
        wrapped_module = super().__getattr__('module')
        return wrapped_module.cpu()

    def state_dict(self, *args, **kwargs):
        wrapped_module = super().__getattr__('module')
        return wrapped_module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        wrapped_module = super().__getattr__('module')
        return wrapped_module.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        """
        Override to set stage
        """
        super().train(mode)
        self._set_module_stage(Stage.TRAIN)

    def eval(self, stage=Stage.TEST):
        """
        Override to set stage
        """
        super().eval()
        self._set_module_stage(stage)

    def _set_module_stage(self, stage):
        wrapped_module = super().__getattr__('module')
        if hasattr(wrapped_module, 'stage'):
            wrapped_module.stage = stage


class Highway(nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: int, num_layers: int=1):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for
            _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = F.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x


class EmbeddingBase(Module):
    """Base class for token level embedding modules.

    Args:
        embedding_dim (int): Size of embedding vector.

    Attributes:
        num_emb_modules (int): Number of ways to embed a token.
        embedding_dim (int): Size of embedding vector.

    """
    __EXPANSIBLE__ = True

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.num_emb_modules = 1
        self.embedding_dim = embedding_dim
        log_class_usage(__class__)

    def get_param_groups_for_optimizer(self) ->List[Dict[str, nn.Parameter]]:
        """
        Organize module parameters into param_groups (or layers), so the optimizer
        and / or schedulers can have custom behavior per layer.
        """
        return [dict(self.named_parameters())]

    def visualize(self, summary_writer: SummaryWriter):
        """
        Overridden in sub classes to implement Tensorboard visualization of
        embedding space
        """
        pass


class EmbeddingList(EmbeddingBase, ModuleList):
    """
    There are more than one way to embed a token and this module provides a way
    to generate a list of sub-embeddings, concat embedding tensors into a single
    Tensor or return a tuple of Tensors that can be used by downstream modules.

    Args:
        embeddings (Iterable[EmbeddingBase]): A sequence of embedding modules to
        embed a token.
        concat (bool): Whether to concatenate the embedding vectors emitted from
        `embeddings` modules.

    Attributes:
        num_emb_modules (int): Number of flattened embeddings in `embeddings`,
            e.g: ((e1, e2), e3) has 3 in total
        input_start_indices (List[int]): List of indices of the sub-embeddings
            in the embedding list.
        concat (bool): Whether to concatenate the embedding vectors emitted from
            `embeddings` modules.
        embedding_dim: Total embedding size, can be a single int or tuple of
            int depending on concat setting
    """

    def __init__(self, embeddings: Iterable[EmbeddingBase], concat: bool
        ) ->None:
        EmbeddingBase.__init__(self, 0)
        embeddings = list(filter(None, embeddings))
        self.num_emb_modules = sum(emb.num_emb_modules for emb in embeddings)
        embeddings_list, input_start_indices = [], []
        start = 0
        for emb in embeddings:
            if emb.embedding_dim > 0:
                embeddings_list.append(emb)
                input_start_indices.append(start)
            start += emb.num_emb_modules
        ModuleList.__init__(self, embeddings_list)
        self.input_start_indices = input_start_indices
        self.concat = concat
        assert len(self) > 0, 'must have at least 1 sub embedding'
        embedding_dims = tuple(emb.embedding_dim for emb in self)
        self.embedding_dim = sum(embedding_dims) if concat else embedding_dims
        log_class_usage(__class__)

    def forward(self, *emb_input) ->Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Get embeddings from all sub-embeddings and either concatenate them
        into one Tensor or return them in a tuple.

        Args:
            *emb_input (type): Sequence of token level embeddings to combine.
                The inputs should match the size of configured embeddings. Each
                of them is either a Tensor or a tuple of Tensors.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: If `concat` is True then
                a Tensor is returned by concatenating all embeddings. Otherwise
                all embeddings are returned in a tuple.

        """
        if self.num_emb_modules != len(emb_input):
            raise Exception(
                f'expecting {self.num_emb_modules} embeddings, ' +
                f'but got {len(emb_input)} input')
        tensors = []
        for emb, start in zip(self, self.input_start_indices):
            end = start + emb.num_emb_modules
            input = emb_input[start:end]
            if len(input) == 1:
                if isinstance(input[0], list) or isinstance(input[0], tuple):
                    [input] = input
            emb_tensor = emb(*input)
            tensors.append(emb_tensor)
        if self.concat:
            return torch.cat(tensors, 2)
        else:
            return tuple(tensors) if len(tensors) > 1 else tensors[0]

    def get_param_groups_for_optimizer(self) ->List[Dict[str, nn.Parameter]]:
        """
        Organize child embedding parameters into param_groups (or layers), so the
        optimizer and / or schedulers can have custom behavior per layer. The
        param_groups from each child embedding are aligned at the first (lowest)
        param_group.
        """
        param_groups: List[Dict[str, nn.Parameter]] = []
        for module_name, embedding_module in self.named_children():
            child_params = embedding_module.get_param_groups_for_optimizer()
            for i, child_param_group in enumerate(child_params):
                if i >= len(param_groups):
                    param_groups.append({})
                for param_name, param in child_param_group.items():
                    param_name = '%s.%s' % (module_name, param_name)
                    param_groups[i][param_name] = param
        return param_groups

    def visualize(self, summary_writer: SummaryWriter):
        for child in self:
            child.visualize(summary_writer)


class lazy_property(object):
    """
    More or less copy-pasta: http://stackoverflow.com/a/6849299
    Meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self._fget = fget
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, obj_cls_type):
        if obj is None:
            return None
        value = self._fget(obj)
        setattr(obj, self.__name__, value)
        return value


class Tensorizer(Component):
    """Tensorizers are a component that converts from batches of
    `pytext.data.type.DataType` instances to tensors. These tensors will eventually
    be inputs to the model, but the model is aware of the tensorizers and can arrange
    the tensors they create to conform to its model.

    Tensorizers have an initialize function. This function allows the tensorizer to
    read through the training dataset to build up any data that it needs for
    creating the model. Commonly this is valuable for things like inferring a
    vocabulary from the training set, or learning the entire set of training labels,
    or slot labels, etc.
    """
    __COMPONENT_TYPE__ = ComponentType.TENSORIZER
    __EXPANSIBLE__ = True
    __TENSORIZER_SCRIPT_IMPL__ = None


    class Config(Component.Config):
        is_input: bool = True

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.is_input)

    def __init__(self, is_input: bool=True):
        self.is_input = is_input
        log_class_usage(__class__)

    @property
    def column_schema(self):
        """Generic types don't pickle well pre-3.7, so we don't actually want
        to store the schema as an attribute. We're already storing all of the
        columns anyway, so until there's a better solution, schema is a property."""
        return []

    def numberize(self, row):
        raise NotImplementedError

    def prepare_input(self, row):
        """ Return preprocessed input tensors/blob for caffe2 prediction net."""
        return self.numberize(row)

    def sort_key(self, row):
        raise NotImplementedError

    def tensorize(self, batch):
        """Tensorizer knows how to pad and tensorize a batch of it's own output."""
        return batch

    def initialize(self, from_scratch=True):
        """
        The initialize function is carefully designed to allow us to read through the
        training dataset only once, and not store it in memory. As such, it can't itself
        manually iterate over the data source. Instead, the initialize function is a
        coroutine, which is sent row data. This should look roughly like::

            # set up variables here
            ...
            try:
                # start reading through data source
                while True:
                    # row has type Dict[str, types.DataType]
                    row = yield
                    # update any variables, vocabularies, etc.
                    ...
            except GeneratorExit:
                # finalize your initialization, set instance variables, etc.
                ...

        See `WordTokenizer.initialize` for a more concrete example.
        """
        return
        yield

    @lazy_property
    def tensorizer_script_impl(self):
        raise NotImplementedError

    def __getstate__(self):
        state = copy.copy(vars(self))
        state.pop('tensorizer_script_impl', None)
        return state

    def stringify(self, token_indices):
        res = ''
        if hasattr(self, 'vocab'):
            res = ' '.join([self.vocab._vocab[index] for index in
                token_indices])
            if hasattr(self, 'tokenizer'):
                if hasattr(self.tokenizer, 'decode'):
                    res = self.tokenizer.decode(res)
        return res

    def torchscriptify(self):
        return self.tensorizer_script_impl.torchscriptify()


def _assert_tensorizer_type(t):
    if t is not type(None) and not issubclass(t, Tensorizer.Config):
        raise TypeError(
            f'ModelInput configuration should only include tensorizers: {t}')


class ModelInputMeta(ConfigBaseMeta):

    def __new__(metacls, typename, bases, namespace):
        annotations = namespace.get('__annotations__', {})
        for t in annotations.values():
            if getattr(t, '__origin__', '') is Union:
                for ut in t.__args__:
                    _assert_tensorizer_type(ut)
            else:
                _assert_tensorizer_type(t)
        return super().__new__(metacls, typename, bases, namespace)


class ModelInputBase(ConfigBase, metaclass=ModelInputMeta):
    """Base class for model inputs."""


FP16_ENABLED = False


def maybe_float(tensor):
    if FP16_ENABLED and tensor.type().split('.')[-1] == 'HalfTensor':
        return tensor.float()
    else:
        return tensor


class ModuleConfig(ConfigBase):
    load_path: Optional[str] = None
    save_path: Optional[str] = None
    freeze: bool = False
    shared_module_key: Optional[str] = None


class Module(nn.Module, Component):
    """Generic module class that serves as base class for all PyText modules.

    Args:
        config (type): Module's `config` object. Specific contents of this object
            depends on the module. Defaults to None.

    """
    Config = ModuleConfig
    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) ->None:
        nn.Module.__init__(self)
        Component.__init__(self, config)
        log_class_usage(__class__)

    def freeze(self) ->None:
        for param in self.parameters():
            param.requires_grad = False


class ClassificationScores(jit.ScriptModule):

    def __init__(self, classes, score_function):
        super().__init__()
        self.classes = jit.Attribute(classes, List[str])
        self.score_function = score_function

    @jit.script_method
    def forward(self, logits: torch.Tensor):
        scores = self.score_function(logits)
        results = jit.annotate(List[Dict[str, float]], [])
        for example_scores in scores.chunk(len(scores)):
            example_scores = example_scores.squeeze(dim=0)
            example_response = jit.annotate(Dict[str, float], {})
            for i in range(len(self.classes)):
                example_response[self.classes[i]] = float(example_scores[i]
                    .item())
            results.append(example_response)
        return results


class IntentSlotScores(nn.Module):

    def __init__(self, doc_scores: jit.ScriptModule, word_scores: jit.
        ScriptModule):
        super().__init__()
        self.doc_scores = doc_scores
        self.word_scores = word_scores
        log_class_usage(__class__)

    def forward(self, logits: Tuple[torch.Tensor, torch.Tensor], context:
        Dict[str, torch.Tensor]) ->Tuple[List[Dict[str, float]], List[List[
        Dict[str, float]]]]:
        d_logits, w_logits = logits
        if 'token_indices' in context:
            w_logits = torch.gather(w_logits, 1, context['token_indices'].
                unsqueeze(2).expand(-1, -1, w_logits.size(-1)))
        d_results = self.doc_scores(d_logits)
        w_results = self.word_scores(w_logits, context)
        return d_results, w_results


class MultiLabelClassificationScores(nn.Module):

    def __init__(self, scores: List[jit.ScriptModule]):
        super().__init__()
        self.scores = nn.ModuleList(scores)
        log_class_usage(__class__)

    def forward(self, logits: List[torch.Tensor]) ->List[List[Dict[str, float]]
        ]:
        results: List[List[Dict[str, float]]] = []
        for idx, sc in enumerate(self.scores):
            logit = logits[idx]
            flattened_logit = logit.view(-1, logit.size()[-1])
            results.append(sc(flattened_logit))
        return results


@jit.script
def _get_prediction_from_scores(scores: torch.Tensor, classes: List[str]
    ) ->List[List[Dict[str, float]]]:
    """
    Given scores for a batch, get the prediction for each word in the form of a
    List[List[Dict[str, float]]] for callers of the torchscript model to consume.
    The outer list iterates over batches of sentences and the inner iterates
    over each token in the sentence. The dictionary consists of
    `label:score` for each word.

    Example:

    Assuming slot labels are [No-Label, Number, Name]
    Utterances: [[call john please], [Brightness 25]]
    Output could look like:
    [
        [
            { No-Label: -0.1, Number: -1.5, Name: -9.01},
            { No-Label: -2.1, Number: -1.5, Name: -0.01},
            { No-Label: -0.1, Number: -1.5, Name: -2.01},
        ],
        [
            { No-Label: -0.1, Number: -1.5, Name: -9.01},
            { No-Label: -2.1, Number: -0.5, Name: -7.01},
            { No-Label: -0.1, Number: -1.5, Name: -2.01},
        ]
    ]
    """
    results: List[List[Dict[str, float]]] = []
    for sentence_scores in scores.chunk(len(scores)):
        sentence_scores = sentence_scores.squeeze(0)
        sentence_response: List[Dict[str, float]] = []
        for word_scores in sentence_scores.chunk(len(sentence_scores)):
            word_scores = word_scores.squeeze(0)
            word_response: Dict[str, float] = {}
            for i in range(len(classes)):
                word_response[classes[i]] = float(word_scores[i].item())
            sentence_response.append(word_response)
        results.append(sentence_response)
    return results


class WordTaggingScores(nn.Module):
    classes: List[str]

    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        log_class_usage(__class__)

    def forward(self, logits: torch.Tensor, context: Optional[Dict[str,
        torch.Tensor]]=None) ->List[List[Dict[str, float]]]:
        scores: torch.Tensor = F.log_softmax(logits, 2)
        return _get_prediction_from_scores(scores, self.classes)


class DotProductSelfAttention(Module):
    """
    Given vector w and token vectors = {t1, t2, ..., t_n}, compute self attention
    weights to weighs the tokens
    * a_j = softmax(w . t_j)
    """


    class Config(Module.Config):
        input_dim: int = 32

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.input_dim)

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        log_class_usage(__class__)

    def forward(self, tokens, tokens_mask):
        """
        Input:
            x: batch_size * seq_len * input_dim
            x_mask: batch_size * seq_len (1 for padding, 0 for true)
        Output:
            alpha: batch_size * seq_len
        """
        scores = self.linear(tokens).squeeze(2)
        scores.data.masked_fill_(tokens_mask.data, -float('inf'))
        return F.softmax(scores, dim=-1)


class SequenceAlignedAttention(Module):
    """
    Given sequences P and Q, computes attention weights for each element in P by
    matching Q with each element in P.
    * a_i_j = softmax(p_i . q_j) where softmax is computed by summing over q_j
    """


    class Config(Module.Config):
        proj_dim: int = 32

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.proj_dim)

    def __init__(self, proj_dim):
        super().__init__()
        self.linear = nn.Linear(proj_dim, proj_dim)
        self.proj_dim = proj_dim
        log_class_usage(__class__)

    def forward(self, p: torch.Tensor, q: torch.Tensor, q_mask: torch.Tensor):
        """
        Input:
            p: batch_size * p_seq_len * dim
            q: batch_size * q_seq_len * dim
            q_mask: batch_size * q_seq_len (1 for padding, 0 for true)
        Output:
            matched_seq: batch_size * doc_seq_len * dim
        """
        p_transform = F.relu(self.linear(p))
        q_transform = F.relu(self.linear(q))
        attn_scores = p_transform.bmm(q_transform.transpose(2, 1))
        q_mask = q_mask.unsqueeze(1).expand(attn_scores.size())
        attn_scores.data.masked_fill_(q_mask.data, -float('inf'))
        attn_scores_flattened = F.softmax(attn_scores.view(-1, q.size(1)),
            dim=-1)
        return attn_scores_flattened.view(-1, p.size(1), q.size(1))


class MultiplicativeAttention(Module):
    """
    Given sequence P and vector q, computes attention weights for each element
    in P by matching q with each element in P using multiplicative attention.
    * a_i = softmax(p_i . W . q)
    """


    class Config(Module.Config):
        p_hidden_dim: int = 32
        q_hidden_dim: int = 32
        normalize: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.p_hidden_dim, config.q_hidden_dim, config.normalize)

    def __init__(self, p_hidden_dim, q_hidden_dim, normalize):
        super().__init__()
        self.normalize = normalize
        self.linear = nn.Linear(p_hidden_dim, q_hidden_dim)
        log_class_usage(__class__)

    def forward(self, p_seq: torch.Tensor, q: torch.Tensor, p_mask: torch.
        Tensor):
        """
        Input:
            p_seq: batch_size * p_seq_len * p_hidden_dim
            q: batch_size * q_hidden_dim
            p_mask: batch_size * p_seq_len (1 for padding, 0 for true)
        Output:
            attn_scores: batch_size * p_seq_len
        """
        Wq = self.linear(q) if self.linear is not None else q
        pWq = p_seq.bmm(Wq.unsqueeze(2)).squeeze(2)
        pWq.data.masked_fill_(p_mask.data, -float('inf'))
        attn_scores = F.softmax(pWq, dim=-1) if self.normalize else pWq.exp()
        return attn_scores


class AugmentedLSTMCell(nn.Module):
    """
    `AugmentedLSTMCell` implements a AugmentedLSTM cell.
    Args:
        embed_dim (int): The number of expected features in the input.
        lstm_dim (int): Number of features in the hidden state of the LSTM.
        Defaults to 32.
        use_highway (bool): If `True` we append a highway network to the
        outputs of the LSTM.
        use_bias (bool): If `True` we use a bias in our LSTM calculations, otherwise
        we don't.

    Attributes:
        input_linearity (nn.Module): Fused weight matrix which
            computes a linear function over the input.
        state_linearity (nn.Module): Fused weight matrix which
            computes a linear function over the states.
    """

    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool,
        use_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias
        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim
            self.input_linearity = nn.Linear(self.embed_dim, self.
                _highway_inp_proj_end, bias=self.use_bias)
            self.state_linearity = nn.Linear(self.lstm_dim, self.
                _highway_inp_proj_start, bias=True)
        else:
            self.input_linearity = nn.Linear(self.embed_dim, 4 * self.
                lstm_dim, bias=self.use_bias)
            self.state_linearity = nn.Linear(self.lstm_dim, 4 * self.
                lstm_dim, bias=True)
        self.reset_parameters()
        log_class_usage(__class__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, states=Tuple[torch.Tensor, torch.
        Tensor], variational_dropout_mask: Optional[torch.Tensor]=None
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Warning: DO NOT USE THIS LAYER DIRECTLY, INSTEAD USE the AugmentedLSTM class

        Args:
            x (torch.Tensor): Input tensor of shape
                (bsize x input_dim).
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (bsize x nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Returned states. Shape of each state is (bsize x nhid).

        """
        hidden_state, memory_state = states
        projected_input = self.input_linearity(x)
        projected_state = self.state_linearity(hidden_state)
        (input_gate) = (forget_gate) = (memory_init) = (output_gate) = (
            highway_gate) = None
        if self.use_highway:
            fused_op = projected_input[:, :5 * self.lstm_dim] + projected_state
            fused_chunked = torch.chunk(fused_op, 5, 1)
            (input_gate, forget_gate, memory_init, output_gate, highway_gate
                ) = fused_chunked
            highway_gate = torch.sigmoid(highway_gate)
        else:
            fused_op = projected_input + projected_state
            input_gate, forget_gate, memory_init, output_gate = torch.chunk(
                fused_op, 4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        memory_init = torch.tanh(memory_init)
        output_gate = torch.sigmoid(output_gate)
        memory = input_gate * memory_init + forget_gate * memory_state
        timestep_output: torch.Tensor = output_gate * torch.tanh(memory)
        if self.use_highway:
            highway_input_projection = projected_input[:, self.
                _highway_inp_proj_start:self._highway_inp_proj_end]
            timestep_output = highway_gate * timestep_output + (1 -
                highway_gate) * highway_input_projection
        if variational_dropout_mask is not None and self.training:
            timestep_output = timestep_output * variational_dropout_mask
        return timestep_output, memory


class AugmentedLSTMUnidirectional(nn.Module):
    """
    `AugmentedLSTMUnidirectional` implements a one-layer single directional
    AugmentedLSTM layer. AugmentedLSTM is an LSTM which optionally
    appends an optional highway network to the output layer. Furthermore the
    dropout controlls the level of variational dropout done.

    Args:
        embed_dim (int): The number of expected features in the input.
        lstm_dim (int): Number of features in the hidden state of the LSTM.
            Defaults to 32.
        go_forward (bool): Whether to compute features left to right (forward)
            or right to left (backward).
        recurrent_dropout_probability (float): Variational dropout probability
            to use. Defaults to 0.0.
        use_highway (bool): If `True` we append a highway network to the
            outputs of the LSTM.
        use_input_projection_bias (bool): If `True` we use a bias in
            our LSTM calculations, otherwise we don't.

    Attributes:
        cell (AugmentedLSTMCell): AugmentedLSTMCell that is applied at every timestep.
    """

    def __init__(self, embed_dim: int, lstm_dim: int, go_forward: bool=True,
        recurrent_dropout_probability: float=0.0, use_highway: bool=True,
        use_input_projection_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.cell = AugmentedLSTMCell(self.embed_dim, self.lstm_dim, self.
            use_highway, use_input_projection_bias)
        log_class_usage(__class__)

    def get_dropout_mask(self, dropout_probability: float,
        tensor_for_masking: torch.Tensor) ->torch.Tensor:
        binary_mask = torch.rand(tensor_for_masking.size()
            ) > dropout_probability
        dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
        return dropout_mask

    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.
        Tensor, torch.Tensor]]=None) ->Tuple[PackedSequence, Tuple[torch.
        Tensor, torch.Tensor]]:
        """
        Warning: DO NOT USE THIS LAYER DIRECTLY, INSTEAD USE the AugmentedLSTM class

        Given an input batch of sequential data such as word embeddings, produces
        a single layer unidirectional AugmentedLSTM representation of the sequential
        input and new state tensors.

        Args:
            inputs (PackedSequence): Input tensor of shape
                (bsize x seq_len x input_dim).
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the initial hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (1 x bsize x num_directions * nhid). Defaults to `None`.

        Returns:
            Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                AgumentedLSTM representation of input and the
                state of the LSTM `t = seq_len`.
                Shape of representation is (bsize x seq_len x representation_dim).
                Shape of each state is (1 x bsize x nhid).

        """
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs,
            batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(batch_size,
            total_timesteps, self.lstm_dim)
        if states is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size,
                self.lstm_dim)
            full_batch_previous_state = sequence_tensor.data.new_zeros(
                batch_size, self.lstm_dim)
        else:
            full_batch_previous_state = states[0].squeeze(0)
            full_batch_previous_memory = states[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = self.get_dropout_mask(self.
                recurrent_dropout_probability, full_batch_previous_memory)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = (timestep if self.go_forward else total_timesteps -
                timestep - 1)
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths
                    ) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:
                current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:
                current_length_index + 1].clone()
            timestep_input = sequence_tensor[0:current_length_index + 1, (
                index)]
            timestep_output, memory = self.cell(timestep_input, (
                previous_state, previous_memory), dropout_mask[0:
                current_length_index + 1] if dropout_mask is not None else None
                )
            full_batch_previous_memory = full_batch_previous_memory.data.clone(
                )
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1
                ] = timestep_output
            output_accumulator[0:current_length_index + 1, (index), :
                ] = timestep_output
        output_accumulator = pack_padded_sequence(output_accumulator,
            batch_lengths, batch_first=True)
        final_state = full_batch_previous_state.unsqueeze(0
            ), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


class ContextualWordConvolution(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes:
        List[int]):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, k,
            padding=k - 1) for k in kernel_sizes])
        token_rep_size = len(kernel_sizes) * out_channels
        self.fc = nn.Linear(token_rep_size, token_rep_size)
        log_class_usage

    def forward(self, words: torch.Tensor):
        words = words.transpose(1, 2)
        conv_outs = [F.relu(conv(words)) for conv in self.convs]
        mp_outs = [self.max_pool(co).squeeze(2) for co in conv_outs]
        return self.fc(torch.cat(mp_outs, 1))


class Trim1d(nn.Module):
    """
    Trims a 1d convolutional output. Used to implement history-padding
    by removing excess padding from the right.

    """

    def __init__(self, trim):
        super(Trim1d, self).__init__()
        self.trim = trim

    def forward(self, x):
        return x[:, :, :-self.trim].contiguous()


class SeparableConv1d(nn.Module):
    """
    Implements a 1d depthwise separable convolutional layer. In regular convolutional
    layers, the input channels are mixed with each other to produce each output channel.
    Depthwise separable convolutions decompose this process into two smaller
    convolutions -- a depthwise and pointwise convolution.

    The depthwise convolution spatially convolves each input channel separately,
    then the pointwise convolution projects this result into a new channel space.
    This process reduces the number of FLOPS used to compute a convolution and also
    exhibits a regularization effect. The general behavior -- including the input
    parameters -- is equivalent to `nn.Conv1d`.

    `bottleneck` controls the behavior of the pointwise convolution. Instead of
    upsampling directly, we split the pointwise convolution into two pieces: the first
    convolution downsamples into a (sufficiently small) low dimension and the
    second convolution upsamples into the target (higher) dimension. Creating this
    bottleneck significantly cuts the number of parameters with minimal loss
    in performance.

    """

    def __init__(self, input_channels: int, output_channels: int,
        kernel_size: int, padding: int, dilation: int, bottleneck: int):
        super(SeparableConv1d, self).__init__()
        conv_layers = [nn.Conv1d(input_channels, input_channels,
            kernel_size, padding=padding, dilation=dilation, groups=
            input_channels)]
        if bottleneck > 0:
            conv_layers.extend([nn.Conv1d(input_channels, bottleneck, 1),
                nn.Conv1d(bottleneck, output_channels, 1)])
        else:
            conv_layers.append(nn.Conv1d(input_channels, output_channels, 1))
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)


class OrderedNeuronLSTMLayer(Module):

    def __init__(self, embed_dim: int, lstm_dim: int, padding_value: float,
        dropout: float) ->None:
        super().__init__()
        self.lstm_dim = lstm_dim
        self.padding_value = padding_value
        self.dropout = nn.Dropout(dropout)
        total_size = embed_dim + lstm_dim
        self.f_gate = nn.Linear(total_size, lstm_dim)
        self.i_gate = nn.Linear(total_size, lstm_dim)
        self.o_gate = nn.Linear(total_size, lstm_dim)
        self.c_hat_gate = nn.Linear(total_size, lstm_dim)
        self.master_forget_no_cumax_gate = nn.Linear(total_size, lstm_dim)
        self.master_input_no_cumax_gate = nn.Linear(total_size, lstm_dim)
        log_class_usage(__class__)

    def forward(self, embedded_tokens: torch.Tensor, states: Tuple[torch.
        Tensor, torch.Tensor], seq_lengths: List[int]) ->Tuple[torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]]:
        hidden, context = states
        batch_size = hidden.size(0)
        all_context = []
        all_hidden = []
        if self.dropout.p > 0.0:
            embedded_tokens = self.dropout(embedded_tokens)
        for batch in embedded_tokens:
            combined = torch.cat((batch, hidden), 1)
            ft = self.f_gate(combined).sigmoid()
            it = self.i_gate(combined).sigmoid()
            ot = self.o_gate(combined).sigmoid()
            c_hat = self.c_hat_gate(combined).tanh()
            master_forget_no_cumax = self.master_forget_no_cumax_gate(combined)
            master_forget = torch.cumsum(F.softmax(master_forget_no_cumax,
                dim=1), dim=1)
            master_input_no_cumax = self.master_input_no_cumax_gate(combined)
            master_input = torch.cumsum(F.softmax(master_input_no_cumax,
                dim=1), dim=1)
            wt = master_forget * master_input
            f_hat_t = ft * wt + (master_forget - wt)
            i_hat_t = it * wt + (master_input - wt)
            context = f_hat_t * context + i_hat_t * c_hat
            hidden = ot * context
            all_context.append(context)
            all_hidden.append(hidden)
        state_hidden = []
        state_context = []
        for i in range(batch_size):
            seq_length = seq_lengths[i]
            state_hidden.append(all_hidden[seq_length - 1][i])
            state_context.append(all_context[seq_length - 1][i])
        return torch.stack(all_hidden), (torch.stack(state_hidden), torch.
            stack(state_context))


class SelfAttention(Module):


    class Config(ConfigBase):
        attn_dimension: int = 64
        dropout: float = 0.4

    def __init__(self, config: Config, n_input: int) ->None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.n_input = n_input
        self.n_attn = config.attn_dimension
        self.ws1 = nn.Linear(n_input, self.n_attn, bias=False)
        self.ws2 = nn.Linear(self.n_attn, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()
        log_class_usage(__class__)

    def init_weights(self, init_range: float=0.1) ->None:
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None
        ) ->torch.Tensor:
        size = torch.onnx.operators.shape_as_tensor(inputs)
        flat_2d_shape = torch.cat((torch.LongTensor([-1]), size[2].view(1)))
        compressed_emb = torch.onnx.operators.reshape_from_tensor_shape(inputs,
            flat_2d_shape)
        hbar = self.tanh(self.ws1(self.dropout(compressed_emb)))
        alphas = self.ws2(hbar)
        alphas = torch.onnx.operators.reshape_from_tensor_shape(alphas,
            size[:2])
        alphas = self.softmax(alphas)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)


class MaxPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None
        ) ->torch.Tensor:
        return torch.max(inputs, 1)[0]


class MeanPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor
        ) ->torch.Tensor:
        return torch.sum(inputs, 1) / seq_lengths.unsqueeze(1).float()


class NoPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None
        ) ->torch.Tensor:
        return inputs


class BoundaryPool(Module):


    class Config(ConfigBase):
        boundary_type: str = 'first'

    def __init__(self, config: Config, n_input: int) ->None:
        super().__init__(config)
        self.boundary_type = config.boundary_type
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None
        ) ->torch.Tensor:
        max_len = inputs.size()[1]
        if self.boundary_type == 'first':
            return inputs[:, (0), :]
        elif self.boundary_type == 'last':
            assert max_len > 1
            return inputs[:, (max_len - 1), :]
        elif self.boundary_type == 'firstlast':
            assert max_len > 1
            return torch.cat((inputs[:, (0), :], inputs[:, (max_len - 1), :
                ]), dim=1)
        else:
            raise Exception('Unknown configuration type {}'.format(self.
                boundary_type))


class LastTimestepPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor
        ) ->torch.Tensor:
        if torch._C._get_tracing_state():
            assert inputs.shape[0] == 1
            return inputs[:, (-1), :]
        bsz, _, dim = inputs.shape
        idx = seq_lengths.unsqueeze(1).expand(bsz, dim).unsqueeze(1)
        return inputs.gather(1, idx - 1).squeeze(1)


class SlotAttentionType(Enum):
    NO_ATTENTION = 'no_attention'
    CONCAT = 'concat'
    MULTIPLY = 'multiply'
    DOT = 'dot'


class SlotAttention(Module):


    class Config(ConfigBase):
        attn_dimension: int = 64
        attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION

    def __init__(self, config: Config, n_input: int, batch_first: bool=True
        ) ->None:
        super().__init__()
        self.batch_first = batch_first
        self.attention_type = config.attention_type
        if self.attention_type == SlotAttentionType.CONCAT:
            self.attention_add = nn.Sequential(nn.Linear(2 * n_input,
                config.attn_dimension, bias=False), nn.Tanh(), nn.Linear(
                config.attn_dimension, 1, bias=False))
        elif self.attention_type == SlotAttentionType.MULTIPLY:
            self.attention_mult = nn.Linear(n_input, n_input, bias=False)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        if isinstance(inputs, PackedSequence):
            inputs, lengths = pad_packed_sequence(inputs, batch_first=self.
                batch_first)
        size = inputs.size()
        exp_inputs_2 = inputs.unsqueeze(1).expand(size[0], size[1], size[1],
            size[2])
        if self.attention_type == SlotAttentionType.CONCAT:
            exp_inputs_1 = inputs.unsqueeze(2).expand(size[0], size[1],
                size[1], size[2])
            catted = torch.cat((exp_inputs_1, exp_inputs_2), 3)
            attn_weights_add = F.softmax(self.attention_add(catted).squeeze
                (3), dim=2).unsqueeze(2)
            context_add = torch.matmul(attn_weights_add, exp_inputs_2).squeeze(
                2)
            output = torch.cat((inputs, context_add), 2)
        elif self.attention_type == SlotAttentionType.MULTIPLY or self.attention_type == SlotAttentionType.DOT:
            attended = (inputs if self.attention_type == SlotAttentionType.
                DOT else self.attention_mult(inputs))
            attn_weights_mult = F.softmax(torch.matmul(inputs, torch.
                transpose(attended, 1, 2)), dim=2).unsqueeze(2)
            context_mult = torch.matmul(attn_weights_mult, exp_inputs_2
                ).squeeze(2)
            output = torch.cat((inputs, context_mult), 2)
        else:
            output = inputs
        return output


class RnnType(Enum):
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'


RNN_TYPE_DICT = {RnnType.RNN: nn.RNN, RnnType.LSTM: nn.LSTM, RnnType.GRU:
    nn.GRU}


class StackedBidirectionalRNN(Module):
    """
    `StackedBidirectionalRNN` implements a multi-layer bidirectional RNN with an
    option to return outputs from all the layers of RNN.

    Args:
        config (Config): Configuration object of type BiLSTM.Config.
        embed_dim (int): The number of expected features in the input.
        padding_value (float): Value for the padded elements. Defaults to 0.0.

    Attributes:
        padding_value (float): Value for the padded elements.
        dropout (nn.Dropout): Dropout layer preceding the LSTM.
        lstm (nn.LSTM): LSTM layer that operates on the inputs.
        representation_dim (int): The calculated dimension of the output features
            of BiLSTM.
    """


    class Config(Module.Config):
        """
        Configuration class for `StackedBidirectionalRNN`.

        Attributes:
            hidden_size (int): Number of features in the hidden state of the RNN.
                Defaults to 32.
            num_layers (int): Number of recurrent layers. Eg. setting `num_layers=2`
                would mean stacking two RNNs together to form a stacked RNN,
                with the second RNN taking in the outputs of the first RNN and
                computing the final result. Defaults to 1.
            dropout (float): Dropout probability to use. Defaults to 0.4.
            bidirectional (bool): If `True`, becomes a bidirectional RNN. Defaults
                to `True`.
            rnn_type (str): Which RNN type to use. Options: "rnn", "lstm", "gru".
            concat_layers (bool): Whether to concatenate the outputs of each layer
                of stacked RNN.
        """
        hidden_size: int = 32
        num_layers: int = 1
        dropout: float = 0.0
        bidirectional: bool = True
        rnn_type: RnnType = RnnType.LSTM
        concat_layers: bool = True

    def __init__(self, config: Config, input_size: int, padding_value:
        float=0.0):
        super().__init__()
        self.num_layers = config.num_layers
        self.dropout = nn.Dropout(config.dropout)
        self.concat_layers = config.concat_layers
        self.padding_value = padding_value
        self.rnns = nn.ModuleList()
        rnn_module = RNN_TYPE_DICT.get(config.rnn_type)
        assert rnn_module is not None, 'rnn_cell cannot be None'
        for i in range(config.num_layers):
            input_size = input_size if i == 0 else 2 * config.hidden_size
            self.rnns.append(rnn_module(input_size, config.hidden_size,
                num_layers=1, bidirectional=config.bidirectional))
        self.representation_dim = (config.num_layers if config.
            concat_layers else 1) * config.hidden_size * (2 if config.
            bidirectional else 1)
        log_class_usage(__class__)

    def forward(self, tokens, tokens_mask):
        """
        Args:
            tokens: batch, max_seq_len, hidden_size
            tokens_mask: batch, max_seq_len (1 for padding, 0 for true)
        Output:
            tokens_encoded: batch, max_seq_len, hidden_size * num_layers if
                concat_layers = True else batch, max_seq_len, hidden_size
        """
        seq_lengths = tokens_mask.eq(0).long().sum(1)
        seq_lengths_sorted, idx_of_sorted = torch.sort(seq_lengths, dim=0,
            descending=True)
        tokens_sorted = tokens.index_select(0, idx_of_sorted)
        packed_tokens = nn.utils.rnn.pack_padded_sequence(tokens_sorted,
            seq_lengths_sorted, batch_first=True)
        outputs = [packed_tokens]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            rnn_input = nn.utils.rnn.PackedSequence(self.dropout(rnn_input.
                data), rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        outputs = outputs[1:]
        for i in range(len(outputs)):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(outputs[i],
                padding_value=self.padding_value, batch_first=True)[0]
        output = torch.cat(outputs, 2) if self.concat_layers else outputs[-1]
        _, idx_of_original = torch.sort(idx_of_sorted, dim=0)
        output = output.index_select(0, idx_of_original)
        max_seq_len = tokens_mask.size(1)
        batch_size, output_seq_len, output_dim = output.size()
        if output_seq_len != max_seq_len:
            padding = torch.zeros(batch_size, max_seq_len - output_seq_len,
                output_dim).type(output.data.type())
            output = torch.cat([output, padding], 1)
        return output


class MultiheadSelfAttention(nn.Module):
    """
    This is a TorchScriptable implementation of MultiheadAttention from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.

    The default constructor values match those required to import the public
    RoBERTa weights. Unless you are pretraining your own model, there's no need to
    change them.
    """

    def __init__(self, embed_dim: int, num_heads: int, scaling: float=0.125,
        dropout: float=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        log_class_usage(__class__)

    def forward(self, query, key_padding_mask):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x source_length, where padding elements are indicated by 1s.
        """
        target_length, batch_size, embed_dim = query.size()
        mask_batch_size, source_length = key_padding_mask.size()
        assert embed_dim == self.embed_dim
        assert batch_size == mask_batch_size, 'query and key_padding_mask batch sizes differed'
        projection = self.input_projection(query)
        q, k, v = projection.chunk(3, dim=-1)
        q *= self.scaling
        batch_heads = batch_size * self.num_heads
        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        assert k.size(1) == source_length
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.shape) == [batch_heads, target_length,
            source_length]
        attn_weights = attn_weights.view(batch_size, self.num_heads,
            target_length, source_length)
        attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(
            1).unsqueeze(2), float('-inf'))
        attn_weights = attn_weights.view(batch_heads, target_length,
            source_length)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32
            ).type_as(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.shape) == [batch_heads, target_length, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(target_length,
            batch_size, embed_dim)
        attn = self.output_projection(attn)
        return attn


def make_positions(tensor, pad_index: int):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at pad_index+1. Padding symbols are ignored.
    """
    masked = tensor.ne(pad_index).long()
    return torch.cumsum(masked, dim=1) * masked + pad_index


class PositionalEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on pad_index
    or by setting pad_index to None and ensuring that the appropriate
    position ids are passed to the forward function.

    This is a TorchScriptable implementation of PositionalEmbedding from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, pad_index:
        Optional[int]=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, pad_index)
        self.pad_index = pad_index
        log_class_usage(__class__)

    def forward(self, input):
        """Input is expected to be of size [batch_size x sequence_length]."""
        positions = make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings


class GeLU(nn.Module):
    """Component class to wrap F.gelu."""

    def forward(self, input):
        return F.gelu(input.float()).type_as(input)


class ResidualMLP(nn.Module):
    """A square MLP component which can learn a bias on an input vector.
    This MLP in particular defaults to using GeLU as its activation function
    (this can be changed by passing a different activation function),
    and retains a residual connection to its original input to help with gradient
    propogation.

    Unlike pytext's MLPDecoder it doesn't currently allow adding a LayerNorm
    in between hidden layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout:
        float=0.1, activation=GeLU):
        super().__init__()
        modules = []
        for last_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.extend([nn.Linear(last_dim, dim), activation(), nn.
                Dropout(dropout)])
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        modules.extend([nn.Linear(last_dim, input_dim), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*modules)
        log_class_usage(__class__)

    def forward(self, input):
        bias = self.mlp(input)
        return input + bias


def merge_input_projection(state):
    """
    New checkpoints of fairseq multihead attention split in_projections into
    k,v,q projections. This function merge them back to to make it compatible.
    """
    items_to_add = {}
    keys_to_remove = []
    bias_suffix = ['q_proj.bias', 'k_proj.bias', 'v_proj.bias']
    weight_suffix = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight']

    def override_state(k, suffix, new_suffix, idx):
        new_key = k[:-len(suffix)] + new_suffix
        dim = state[k].shape[0]
        if new_key not in items_to_add:
            items_to_add[new_key] = torch.zeros_like(state[k]).repeat(3, 1
                ) if len(state[k].shape) > 1 else torch.zeros_like(state[k]
                ).repeat(3)
        items_to_add[new_key][idx * dim:(idx + 1) * dim] = state[k]
        keys_to_remove.append(k)
    for k in state.keys():
        for idx, suffix in enumerate(weight_suffix):
            if k.endswith(suffix):
                override_state(k, suffix, 'in_proj_weight', idx)
        for idx, suffix in enumerate(bias_suffix):
            if k.endswith(suffix):
                override_state(k, suffix, 'in_proj_bias', idx)
    for k in keys_to_remove:
        del state[k]
    for key, value in items_to_add.items():
        state[key] = value
    return state


def remove_state_keys(state, keys_regex):
    """Remove keys from state that match a regex"""
    regex = re.compile(keys_regex)
    return {k: v for k, v in state.items() if not regex.findall(k)}


def rename_state_keys(state, keys_regex, replacement):
    """Rename keys from state that match a regex; replacement can use capture groups"""
    regex = re.compile(keys_regex)
    return {(k if not regex.findall(k) else regex.sub(replacement, k)): v for
        k, v in state.items()}


def rename_component_from_root(state, old_name, new_name):
    """Rename keys from state using full python paths"""
    return rename_state_keys(state, '^' + old_name.replace('.', '\\.') +
        '.?(.*)$', new_name + '.\\1')


def translate_roberta_state_dict(state_dict):
    """Translate the public RoBERTa weights to ones which match SentenceEncoder."""
    new_state = rename_component_from_root(state_dict,
        'decoder.sentence_encoder', 'transformer')
    new_state = rename_state_keys(new_state, 'embed_tokens', 'token_embedding')
    new_state = rename_state_keys(new_state, 'embed_positions',
        'positional_embedding.embedding')
    new_state = rename_state_keys(new_state, 'emb_layer_norm',
        'embedding_layer_norm')
    new_state = rename_state_keys(new_state, 'self_attn', 'attention')
    new_state = merge_input_projection(new_state)
    new_state = rename_state_keys(new_state, '_proj.(.*)', 'put_projection.\\1'
        )
    new_state = rename_state_keys(new_state, 'fc1', 'residual_mlp.mlp.0')
    new_state = rename_state_keys(new_state, 'fc2', 'residual_mlp.mlp.3')
    new_state = remove_state_keys(new_state, '^sentence_')
    new_state = remove_state_keys(new_state, '_classification_head.')
    new_state = remove_state_keys(new_state, '^decoder\\.lm_head')
    new_state = remove_state_keys(new_state, 'segment_embedding')
    return new_state


DEFAULT_EMBEDDING_DIM = 768


DEFAULT_NUM_ATTENTION_HEADS = 12


class TransformerLayer(nn.Module):

    def __init__(self, embedding_dim: int=DEFAULT_EMBEDDING_DIM, attention:
        Optional[MultiheadSelfAttention]=None, residual_mlp: Optional[
        ResidualMLP]=None, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention or MultiheadSelfAttention(embedding_dim,
            num_heads=DEFAULT_NUM_ATTENTION_HEADS)
        self.residual_mlp = residual_mlp or ResidualMLP(embedding_dim,
            hidden_dims=[embedding_dim * 4])
        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        log_class_usage(__class__)

    def forward(self, input, key_padding_mask):
        attention = self.attention(input, key_padding_mask)
        attention = self.dropout(attention)
        biased_input = input + attention
        biased_input = self.attention_layer_norm(biased_input)
        biased = self.residual_mlp(biased_input)
        return self.final_layer_norm(biased)


DEFAULT_MAX_SEQUENCE_LENGTH = 514


DEFAULT_NUM_LAYERS = 12


DEFAULT_PADDING_IDX = 1


DEFAULT_VOCAB_SIZE = 50265


class Transformer(nn.Module):

    def __init__(self, vocab_size: int=DEFAULT_VOCAB_SIZE, embedding_dim:
        int=DEFAULT_EMBEDDING_DIM, padding_idx: int=DEFAULT_PADDING_IDX,
        max_seq_len: int=DEFAULT_MAX_SEQUENCE_LENGTH, layers: List[
        TransformerLayer]=(), dropout: float=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim,
            padding_idx)
        self.layers = nn.ModuleList(layers or [TransformerLayer(
            embedding_dim) for _ in range(DEFAULT_NUM_LAYERS)])
        self.positional_embedding = PositionalEmbedding(max_seq_len,
            embedding_dim, padding_idx)
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        log_class_usage(__class__)

    def forward(self, tokens: torch.Tensor) ->List[torch.Tensor]:
        padding_mask = tokens.eq(self.padding_idx)
        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)
        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        padded_normed = normed * (1 - padding_mask.unsqueeze(-1).type_as(
            normed))
        encoded = padded_normed.transpose(0, 1)
        states = [encoded]
        for layer in self.layers:
            encoded = layer(encoded, padding_mask)
            states.append(encoded)
        return states


@torch.jit.script
def reverse_tensor_list(int_list: List[torch.Tensor]) ->List[torch.Tensor]:
    l_len = len(int_list)
    res = []
    for idx in range(l_len):
        res.append(int_list[l_len - idx - 1])
    return res


@torch.jit.script
def xaviervar(size: List[int], device: str):
    t = torch.empty(size, device=device)
    t = torch.nn.init.xavier_normal_(t)
    return t


class CompositionalNN(torch.jit.ScriptModule):
    """
    Combines a list / sequence of embeddings into one using a biLSTM
    """
    __constants__ = ['lstm_dim', 'linear_seq']

    def __init__(self, lstm_dim: int):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.lstm_fwd = nn.LSTM(lstm_dim, lstm_dim, num_layers=1)
        self.lstm_rev = nn.LSTM(lstm_dim, lstm_dim, num_layers=1)
        self.linear_seq = nn.Sequential(nn.Linear(2 * lstm_dim, lstm_dim),
            nn.Tanh())

    @torch.jit.script_method
    def forward(self, x: List[torch.Tensor], device: str='cpu') ->torch.Tensor:
        """
        Embed the sequence. If the input corresponds to [IN:GL where am I at]:
        - x will contain the embeddings of [at I am where IN:GL] in that order.
        - Forward LSTM will embed the sequence [IN:GL where am I at].
        - Backward LSTM will embed the sequence [IN:GL at I am where].
        The final hidden states are concatenated and then projected.

        Args:
            x: Embeddings of the input tokens in *reversed* order
        Shapes:
            x: (1, lstm_dim) each
            return value: (1, lstm_dim)
        """
        lstm_hidden_fwd = xaviervar([1, 1, self.lstm_dim], device=device
            ), xaviervar([1, 1, self.lstm_dim], device=device)
        lstm_hidden_rev = xaviervar([1, 1, self.lstm_dim], device=device
            ), xaviervar([1, 1, self.lstm_dim], device=device)
        nonterminal_element = x[-1]
        reversed_rest = x[:-1]
        fwd_input = [nonterminal_element] + reverse_tensor_list(reversed_rest)
        rev_input = [nonterminal_element] + reversed_rest
        stacked_fwd = self.lstm_fwd(torch.stack(fwd_input), lstm_hidden_fwd)[0
            ][0]
        stacked_rev = self.lstm_rev(torch.stack(rev_input), lstm_hidden_rev)[0
            ][0]
        combined = torch.cat([stacked_fwd, stacked_rev], dim=1)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


class CompositionalSummationNN(torch.jit.ScriptModule):
    """
    Simpler version of CompositionalNN
    """
    __constants__ = ['lstm_dim', 'linear_seq']

    def __init__(self, lstm_dim: int):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.linear_seq = nn.Sequential(nn.Linear(lstm_dim, lstm_dim), nn.
            Tanh())

    @torch.jit.script_method
    def forward(self, x: List[torch.Tensor], device: str='cpu') ->torch.Tensor:
        combined = torch.sum(torch.cat(x, dim=0), dim=0, keepdim=True)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


def create_src_lengths_mask(batch_size: int, src_lengths):
    """
    Generate boolean mask to prevent attention beyond the end of source

    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths

    Outputs:
      [batch_size, max_src_len]
    """
    max_srclen = src_lengths.max()
    src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking: bool=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class DotAttention(nn.Module):

    def __init__(self, decoder_hidden_state_dim, context_dim,
        force_projection=False, src_length_masking=True):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim
        self.input_proj = None
        if force_projection or decoder_hidden_state_dim != context_dim:
            self.input_proj = nn.Linear(decoder_hidden_state_dim,
                context_dim, bias=True)
        self.src_length_masking = src_length_masking
        log_class_usage(__class__)

    def forward(self, decoder_state, source_hids, src_lengths):
        source_hids = source_hids.transpose(0, 1)
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        attn_scores = torch.bmm(source_hids, decoder_state.unsqueeze(2)
            ).squeeze(2)
        normalized_masked_attn_scores = masked_softmax(attn_scores,
            src_lengths, self.src_length_masking)
        attn_weighted_context = (source_hids *
            normalized_masked_attn_scores.unsqueeze(2)).contiguous().sum(1)
        return attn_weighted_context, normalized_masked_attn_scores.t()


class PyTextSeq2SeqModule(Module):
    instance_id: str = None

    def __init__(self):
        super().__init__()
        self.assign_id()
        log_class_usage(__class__)

    def assign_id(self):
        global global_counter
        self.instance_id = '.'.join([type(self).__name__, str(global_counter)])
        global_counter = global_counter + 1


class PlaceholderIdentity(nn.Module):

    def forward(self, x, incremental_state: Optional[Dict[str, Tensor]]=None):
        return x


class PlaceholderAttentionIdentity(nn.Module):

    def forward(self, query, key, value, need_weights: bool=None,
        key_padding_mask: Optional[Tensor]=None, incremental_state:
        Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Tensor]]:
        optional_attention: Optional[Tensor] = None
        return query, optional_attention

    def reorder_incremental_state(self, incremental_state: Dict[str, Tensor
        ], new_order: Tensor):
        pass


class BiLSTM(torch.nn.Module):
    """Wrapper for nn.LSTM

    Differences include:
    * weight initialization
    * the bidirectional option makes the first layer bidirectional only
    (and in that case the hidden dim is divided by 2)
    """

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = torch.nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(self, num_layers, bidirectional, embed_dim, hidden_dim,
        dropout):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0, 'hidden_dim should be even if bidirectional'
        self.hidden_dim = hidden_dim
        self.layers = torch.nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, 'hidden_dim must be even if bidirectional (to be divided evenly between directions)'
            self.layers.append(BiLSTM.LSTM(embed_dim if layer == 0 else
                hidden_dim, hidden_dim // 2 if is_layer_bidirectional else
                hidden_dim, num_layers=1, dropout=dropout, bidirectional=
                is_layer_bidirectional))
        log_class_usage(__class__)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor,
        enforce_sorted: bool=True):
        bsz = embeddings.size()[1]
        packed_input = pack_padded_sequence(embeddings, lengths,
            enforce_sorted=enforce_sorted)
        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
                c0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
            else:
                h0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)
                c0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0)
                )
            if self.bidirectional and i == 0:
                h_last = torch.cat((h_last[(0), :, :], h_last[(1), :, :]),
                    dim=1)
                c_last = torch.cat((c_last[(0), :, :], c_last[(1), :, :]),
                    dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)
            final_hiddens.append(h_last)
            final_cells.append(c_last)
            packed_input = current_output
        final_hidden_size_list: List[int] = final_hiddens[0].size()
        final_hidden_size: Tuple[int, int] = (final_hidden_size_list[0],
            final_hidden_size_list[1])
        final_hiddens = torch.cat(final_hiddens, dim=0).view(self.
            num_layers, *final_hidden_size)
        final_cell_size_list: List[int] = final_cells[0].size()
        final_cell_size: Tuple[int, int] = (final_cell_size_list[0],
            final_cell_size_list[1])
        final_cells = torch.cat(final_cells, dim=0).view(self.num_layers, *
            final_cell_size)
        unpacked_output, _ = pad_packed_sequence(packed_input)
        return unpacked_output, final_hiddens, final_cells


class GeLU(nn.Module):
    """
    Implements Gaussian Error Linear Units (GELUs).

    Reference:
    Gaussian Error Linear Units (GELUs). Dan Hendrycks, Kevin Gimpel.
    Technical Report, 2017. https://arxiv.org/pdf/1606.08415.pdf
    """

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return torch.ops._caffe2.Gelu(x, True)
        else:
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
                0.044715 * (x * x * x))))


class BeamDecode(torch.nn.Module):
    """
    Decodes the output of Beam Search to get the top hypotheses
    """

    def __init__(self, eos_token_id, length_penalty, nbest, beam_size,
        stop_at_eos):
        super().__init__()
        self.eos_token_id: int = eos_token_id
        self.length_penalty: float = length_penalty
        self.nbest: int = nbest
        self.beam_size: int = beam_size
        self.stop_at_eos: bool = stop_at_eos

    @torch.no_grad()
    def forward(self, beam_tokens: Tensor, beam_scores: Tensor,
        token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int
        ) ->List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:
        self._check_dimensions(beam_tokens, beam_scores, token_weights,
            beam_prev_indices, num_steps)
        end_states = self._get_all_end_states(beam_tokens, beam_scores,
            beam_prev_indices, num_steps)
        outputs = torch.jit.annotate(List[Tuple[Tensor, float, List[float],
            Tensor, Tensor]], [])
        for state_idx in range(len(end_states)):
            state = end_states[state_idx]
            hypothesis_score = float(state[0])
            beam_indices = self._get_output_steps_to_beam_indices(state,
                beam_prev_indices)
            beam_output = torch.jit.annotate(List[Tensor], [])
            token_level_scores = torch.jit.annotate(List[float], [])
            position = int(state[1])
            hyp_index = int(state[2])
            best_indices = torch.tensor([position, hyp_index])
            back_alignment_weights = []
            assert position + 1 == len(beam_indices)
            pos = 1
            prev_beam_index = -1
            while pos < len(beam_indices):
                beam_index = beam_indices[pos]
                beam_output.append(beam_tokens[pos][beam_index])
                if pos == 1:
                    token_level_scores.append(float(beam_scores[pos][
                        beam_index]))
                else:
                    token_level_scores.append(float(beam_scores[pos][
                        beam_index]) - float(beam_scores[pos - 1][
                        prev_beam_index]))
                back_alignment_weights.append(token_weights[pos][beam_index
                    ].detach())
                prev_beam_index = beam_index
                pos += 1
            outputs.append((torch.stack(beam_output), hypothesis_score,
                token_level_scores, torch.stack(back_alignment_weights, dim
                =1), best_indices))
        return outputs

    def _get_output_steps_to_beam_indices(self, end_state: Tensor,
        beam_prev_indices: Tensor) ->List[int]:
        """
        Returns a mapping from each output position and the beam index that was
        picked from the beam search results.
        """
        present_position = int(end_state[1])
        beam_index = int(end_state[2])
        beam_indices = torch.jit.annotate(List[int], [])
        while present_position >= 0:
            beam_indices.insert(0, beam_index)
            beam_index = beam_prev_indices[present_position][beam_index]
            present_position = present_position - 1
        return beam_indices

    def _add_to_end_states(self, end_states: List[Tensor], min_score: float,
        state: Tensor, min_index: int) ->Tuple[List[Tensor], float, int]:
        """
        Maintains a list of atmost `nbest` highest end states
        """
        if len(end_states) < self.nbest:
            end_states.append(state)
            if state[0] <= min_score:
                min_score = state[0]
                min_index = len(end_states) - 1
        elif state[0] > min_score:
            end_states[min_index] = state
            min_index = -1
            min_score = 65504.0
            for idx in range(len(end_states)):
                s = end_states[idx]
                if s[0] <= min_score:
                    min_index = idx
                    min_score = s[0]
        return end_states, min_score, min_index

    def _get_all_end_states(self, beam_tokens: Tensor, beam_scores: Tensor,
        beam_prev_indices: Tensor, num_steps: int) ->Tensor:
        """
        Return all end states and hypothesis scores for those end states.
        """
        min_score = 65504.0
        min_index = -1
        end_states = torch.jit.annotate(List[Tensor], [])
        prev_hypo_is_finished = torch.zeros(self.beam_size).byte()
        position = 1
        while position <= num_steps:
            hypo_is_finished = torch.zeros(self.beam_size, dtype=torch.bool)
            for hyp_index in range(self.beam_size):
                prev_pos = beam_prev_indices[position][hyp_index]
                hypo_is_finished[hyp_index] = prev_hypo_is_finished[prev_pos]
                if hypo_is_finished[hyp_index] == 0:
                    if beam_tokens[position][hyp_index
                        ] == self.eos_token_id or position == num_steps:
                        if self.stop_at_eos:
                            hypo_is_finished[hyp_index] = 1
                        hypo_score = float(beam_scores[position][hyp_index])
                        if self.length_penalty != 0:
                            hypo_score = (hypo_score / position ** self.
                                length_penalty)
                        end_states, min_score, min_index = (self.
                            _add_to_end_states(end_states, min_score, torch
                            .tensor([hypo_score, float(position), float(
                            hyp_index)]), min_index))
            prev_hypo_is_finished = hypo_is_finished
            position = position + 1
        end_states = torch.stack(end_states)
        _, sorted_end_state_indices = end_states[:, (0)].sort(dim=0,
            descending=True)
        end_states = end_states[(sorted_end_state_indices), :]
        return end_states

    def _check_dimensions(self, beam_tokens: Tensor, beam_scores: Tensor,
        token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int
        ) ->None:
        assert beam_tokens.size(1
            ) == self.beam_size, 'Dimension of beam_tokens : {} and beam size : {} are not consistent'.format(
            beam_tokens.size(), self.beam_size)
        assert beam_scores.size(1
            ) == self.beam_size, 'Dimension of beam_scores : {} and beam size : {} are not consistent'.format(
            beam_scores.size(), self.beam_size)
        assert token_weights.size(1
            ) == self.beam_size, 'Dimension of token_weights : {} and beam size : {} are not consistent'.format(
            token_weights.size(), self.beam_size)
        assert beam_prev_indices.size(1
            ) == self.beam_size, 'Dimension of beam_prev_indices : {} and beam size : {} '
        """are not consistent""".format(beam_prev_indices.size(), self.
            beam_size)
        assert beam_tokens.size(0
            ) <= num_steps + 1, 'Dimension of beam_tokens : {} and num_steps : {} are not consistent'.format(
            beam_tokens.size(), num_steps)
        assert beam_scores.size(0
            ) <= num_steps + 1, 'Dimension of beam_scores : {} and num_steps : {} are not consistent'.format(
            beam_scores.size(), num_steps)
        assert token_weights.size(0
            ) <= num_steps + 1, 'Dimension of token_weights : {} and num_steps : {} are not consistent'.format(
            token_weights.size(), num_steps)
        assert beam_prev_indices.size(0
            ) <= num_steps + 1, 'Dimension of beam_prev_indices : {} and num_steps : {} are not consistent'.format(
            beam_prev_indices.size(), num_steps)


@torch.jit.script
def get_first_decoder_step_input(beam_size: int=5, eos_token_id: int=0,
    src_length: int=1) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor]:
    prev_tokens = torch.full([beam_size], eos_token_id, dtype=torch.long)
    prev_scores = torch.full([beam_size], 1, dtype=torch.float)
    prev_hypos = torch.full([beam_size], 0, dtype=torch.long)
    attention_weights = torch.full([beam_size, src_length], 1, dtype=torch.
        float)
    return prev_tokens, prev_scores, prev_hypos, attention_weights


class BeamSearch(nn.Module):

    def __init__(self, model_list, tgt_dict_eos, beam_size: int=2, quantize:
        bool=False, record_attention: bool=False):
        super().__init__()
        self.models = model_list
        self.target_dict_eos = tgt_dict_eos
        self.beam_size = beam_size
        self.record_attention = record_attention
        encoder_ens = EncoderEnsemble(self.models, self.beam_size)
        if quantize:
            encoder_ens = torch.quantization.quantize_dynamic(encoder_ens,
                {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.encoder_ens = torch.jit.script(encoder_ens)
        decoder_ens = DecoderBatchedStepEnsemble(self.models, beam_size,
            record_attention=record_attention)
        if quantize:
            decoder_ens = torch.quantization.quantize_dynamic(decoder_ens,
                {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.decoder_ens = torch.jit.script(decoder_ens)

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor,
        num_steps: int, dict_feat: Optional[Tuple[torch.Tensor, torch.
        Tensor, torch.Tensor]]=None, contextual_token_embedding: Optional[
        torch.Tensor]=None):
        self.decoder_ens.reset_incremental_states()
        decoder_ip = self.encoder_ens(src_tokens, src_lengths, dict_feat,
            contextual_token_embedding)
        prev_token, prev_scores, prev_hypos_indices, attention_weights = (
            get_first_decoder_step_input(self.beam_size, self.
            target_dict_eos, src_lengths[0]))
        all_tokens_list = [prev_token]
        all_scores_list = [prev_scores]
        all_prev_indices_list = [prev_hypos_indices]
        all_attentions_list: List[torch.Tensor] = []
        if self.record_attention:
            all_attentions_list.append(attention_weights)
        for i in range(num_steps):
            (prev_token, prev_scores, prev_hypos_indices, attention_weights,
                decoder_ip) = (self.decoder_ens(prev_token, prev_scores, i +
                1, decoder_ip))
            all_tokens_list.append(prev_token)
            all_scores_list.append(prev_scores)
            all_prev_indices_list.append(prev_hypos_indices)
            if self.record_attention:
                all_attentions_list.append(attention_weights)
        all_tokens = torch.stack(all_tokens_list)
        all_scores = torch.stack(all_scores_list)
        all_prev_indices = torch.stack(all_prev_indices_list)
        if self.record_attention:
            all_attn_weights = torch.stack(all_attentions_list)
        else:
            all_attn_weights = torch.zeros(num_steps + 1, self.beam_size,
                src_tokens.size(1))
        return all_tokens, all_scores, all_attn_weights, all_prev_indices


class DecoderBatchedStepEnsemble(nn.Module):
    """
    This method should have a common interface such that it can be called after
    the encoder as well as after the decoder
    """
    incremental_states: Dict[str, Dict[str, Tensor]]

    def __init__(self, models, beam_size, record_attention=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.incremental_states = {}
        self.beam_size = beam_size
        self.record_attention = record_attention

    @torch.jit.export
    def reset_incremental_states(self):
        for idx, _model in enumerate(self.models):
            self.incremental_states[str(idx)] = {}

    def forward(self, prev_tokens: Tensor, prev_scores: Tensor, timestep:
        int, decoder_ips: List[Dict[str, Tensor]]) ->Tuple[Tensor, Tensor,
        Tensor, Tensor, List[Dict[str, Tensor]]]:
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """
        prev_tokens = prev_tokens.unsqueeze(1)
        log_probs_per_model = torch.jit.annotate(List[Tensor], [])
        attn_weights_per_model = torch.jit.annotate(List[Tensor], [])
        futures = torch.jit.annotate(List[Tuple[Tensor, Dict[str, Tensor]]], []
            )
        for idx, model in enumerate(self.models):
            decoder_ip = decoder_ips[idx]
            incremental_state = self.incremental_states[str(idx)]
            fut = model.decoder(prev_tokens, decoder_ip, incremental_state=
                incremental_state, timestep=timestep)
            futures.append(fut)
        for idx, _model in enumerate(self.models):
            fut = futures[idx]
            log_probs, features = fut
            log_probs_per_model.append(log_probs)
            if 'attn_scores' in features:
                attn_weights_per_model.append(features['attn_scores'])
        best_scores, best_tokens, prev_hypos, attention_weights = (self.
            beam_search_aggregate_topk(log_probs_per_model,
            attn_weights_per_model, prev_scores, self.beam_size, self.
            record_attention))
        for model_state_ptr, model in enumerate(self.models):
            incremental_state = self.incremental_states[str(model_state_ptr)]
            model.decoder.reorder_incremental_state(incremental_state,
                prev_hypos)
        return (best_tokens, best_scores, prev_hypos, attention_weights,
            decoder_ips)

    def beam_search_aggregate_topk(self, log_probs_per_model: List[torch.
        Tensor], attn_weights_per_model: List[torch.Tensor], prev_scores:
        torch.Tensor, beam_size: int, record_attention: bool):
        average_log_probs = torch.mean(torch.cat(log_probs_per_model, dim=1
            ), dim=1, keepdim=True)
        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(average_log_probs
            .squeeze(1), k=beam_size)
        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)
        best_scores, best_indices = torch.topk(total_scores_flat, k=beam_size)
        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices
            ).view(-1)
        prev_hypos = best_indices // beam_size
        if record_attention:
            average_attn_weights = torch.mean(torch.cat(
                attn_weights_per_model, dim=1), dim=1, keepdim=True)
            attention_weights = average_attn_weights.index_select(dim=0,
                index=prev_hypos)
            attention_weights = attention_weights.squeeze_(1)
        else:
            attention_weights = torch.zeros(beam_size,
                attn_weights_per_model[0].size(2))
        return best_scores, best_tokens, prev_hypos, attention_weights


class EncoderEnsemble(nn.Module):
    """
    This class will call the encoders from all the models in the ensemble.
    It will process the encoder output to prepare input for each decoder step
    input
    """

    def __init__(self, models, beam_size):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.beam_size = beam_size

    def forward(self, src_tokens: Tensor, src_lengths: Tensor, dict_feat:
        Optional[Tuple[Tensor, Tensor, Tensor]]=None,
        contextual_token_embedding: Optional[Tensor]=None) ->List[Dict[str,
        Tensor]]:
        src_tokens_seq_first = src_tokens.t()
        futures = torch.jit.annotate(List[Dict[str, Tensor]], [])
        for model in self.models:
            embedding_input = [[src_tokens_seq_first]]
            if dict_feat is not None:
                embedding_input.append(list(dict_feat))
            if contextual_token_embedding is not None:
                embedding_input.append([contextual_token_embedding])
            embeddings = model.source_embeddings(embedding_input)
            futures.append(model.encoder(src_tokens_seq_first, embeddings,
                src_lengths))
        return self.prepare_decoderstep_ip(futures)

    def prepare_decoderstep_ip(self, futures: List[Dict[str, Tensor]]) ->List[
        Dict[str, Tensor]]:
        outputs = torch.jit.annotate(List[Dict[str, Tensor]], [])
        for idx, model in enumerate(self.models):
            encoder_out = futures[idx]
            tiled_encoder_out = model.encoder.tile_encoder_out(self.
                beam_size, encoder_out)
            outputs.append(tiled_encoder_out)
        return outputs


@torch.jit.script
def list_membership(item: int, list: List[int]):
    item_present = False
    for i in list:
        if item == i:
            item_present = True
    return item_present


class ScriptVocabulary(torch.jit.ScriptModule):

    def __init__(self, vocab_list, unk_idx: int=0, pad_idx: int=-1, bos_idx:
        int=-1, eos_idx: int=-1, mask_idx: int=-1):
        super().__init__()
        self.vocab = torch.jit.Attribute(vocab_list, List[str])
        self.unk_idx = torch.jit.Attribute(unk_idx, int)
        self.pad_idx = torch.jit.Attribute(pad_idx, int)
        self.eos_idx = torch.jit.Attribute(eos_idx, int)
        self.bos_idx = torch.jit.Attribute(bos_idx, int)
        self.mask_idx = torch.jit.Attribute(mask_idx, int)
        self.idx = torch.jit.Attribute({word: i for i, word in enumerate(
            vocab_list)}, Dict[str, int])
        pad_token = vocab_list[pad_idx] if pad_idx >= 0 else '__PAD__'
        self.pad_token = torch.jit.Attribute(pad_token, str)

    @torch.jit.script_method
    def lookup_indices_1d(self, values: List[str]) ->List[int]:
        result = torch.jit.annotate(List[int], [])
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    @torch.jit.script_method
    def lookup_indices_2d(self, values: List[List[str]]) ->List[List[int]]:
        result = torch.jit.annotate(List[List[int]], [])
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

    @torch.jit.script_method
    def lookup_words_1d(self, values: torch.Tensor, filter_token_list: List
        [int]=(), possible_unk_token: Optional[str]=None) ->List[str]:
        """If possible_unk_token is not None, then all UNK id's will be replaced
        by possible_unk_token instead of the default UNK string which is <UNK>.
        This is a simple way to resolve UNK's when there's a correspondence
        between source and target translations.
        """
        result = torch.jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                result.append(self.lookup_word(value, possible_unk_token))
        return result

    @torch.jit.script_method
    def lookup_words_1d_cycle_heuristic(self, values: torch.Tensor,
        filter_token_list: List[int], ordered_unks_token: List[str]) ->List[str
        ]:
        """This function is a extension of the possible_unk_token heuristic
        in lookup_words_1d, which fails in the case when multiple unks are
        available. The way we deal with this is we increment every unk token in
        ordered_unks_token everytime we substitute an unk token. This solves a
        substantial amount of queries with multiple unk tokens.
        """
        unk_idx = 0
        unk_idx_length = torch.jit.annotate(int, len(ordered_unks_token))
        unk_copy = torch.jit.annotate(bool, unk_idx_length != 0)
        vocab_length = torch.jit.annotate(int, len(self.vocab))
        result = torch.jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                if value < vocab_length and value != self.unk_idx:
                    result.append(self.vocab[value])
                elif not unk_copy:
                    result.append(self.vocab[self.unk_idx])
                else:
                    unk_value = ordered_unks_token[unk_idx % unk_idx_length]
                    result.append(unk_value)
                    unk_idx += 1
        return result

    @torch.jit.script_method
    def lookup_word(self, idx: int, possible_unk_token: Optional[str]=None):
        if idx < len(self.vocab) and idx != self.unk_idx:
            return self.vocab[idx]
        else:
            return self.vocab[self.unk_idx
                ] if possible_unk_token is None else possible_unk_token


@torch.jit.script
def get_single_unk_token(src_tokens: List[str], word_ids: List[int],
    copy_unk_token: bool, unk_idx: int):
    """Returns the string representation of the first UNK
       we get in our source utterance. We can then use this string instead of
       writing "<UNK>" in our decoding.
    """
    if copy_unk_token:
        for i, x in enumerate(word_ids):
            if x == unk_idx:
                return src_tokens[i]
    return None


class Seq2SeqJIT(torch.nn.Module):

    def __init__(self, src_dict, tgt_dict, sequence_generator,
        filter_eos_bos, copy_unk_token=False, dictfeat_dict=None):
        super().__init__()
        self.source_vocab = ScriptVocabulary(src_dict._vocab, src_dict.
            get_unk_index(), bos_idx=src_dict.get_bos_index(-1), eos_idx=
            src_dict.get_eos_index(-1))
        self.target_vocab = ScriptVocabulary(tgt_dict._vocab, tgt_dict.
            get_unk_index(), bos_idx=tgt_dict.get_bos_index(), eos_idx=
            tgt_dict.get_eos_index())
        if dictfeat_dict:
            self.dictfeat_vocab = ScriptVocabulary(dictfeat_dict._vocab,
                pad_idx=dictfeat_dict.idx[src_dict[src_dict.get_pad_index()]])
        else:
            self.dictfeat_vocab = ScriptVocabulary([])
        self.sequence_generator = sequence_generator
        self.copy_unk_token: bool = copy_unk_token
        self.unk_idx: int = self.source_vocab.unk_idx
        self.filter_eos_bos: bool = filter_eos_bos

    def prepare_generator_inputs(self, word_ids: List[int], dict_feat:
        Optional[Tuple[List[str], List[float], List[int]]]=None,
        contextual_token_embedding: Optional[List[float]]=None) ->Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.
        Tensor]], Optional[torch.Tensor], torch.Tensor]:
        src_len = len(word_ids)
        dict_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = None
        if dict_feat is not None:
            dict_tokens, dict_weights, dict_lengths = dict_feat
            dict_ids = self.dictfeat_vocab.lookup_indices_1d(dict_tokens)
            dict_tensors = torch.tensor([dict_ids]), torch.tensor([
                dict_weights], dtype=torch.float), torch.tensor([dict_lengths])
        contextual_embedding_tensor: Optional[torch.Tensor] = None
        if contextual_token_embedding is not None:
            assert len(contextual_token_embedding) % src_len == 0 and len(
                contextual_token_embedding
                ) > 0, f'Incorrect size for contextual embeddings: {len(contextual_token_embedding)}, Expected a non-zero multiple of input token count {src_len} '
            contextual_embedding_tensor = torch.tensor([
                contextual_token_embedding], dtype=torch.float)
        return torch.tensor(word_ids).reshape(-1, 1
            ), dict_tensors, contextual_embedding_tensor, torch.tensor([
            src_len])

    def forward(self, src_tokens: List[str], dict_feat: Optional[Tuple[List
        [str], List[float], List[int]]]=None, contextual_token_embedding:
        Optional[List[float]]=None) ->List[Tuple[List[str], float, List[float]]
        ]:
        word_ids = self.source_vocab.lookup_indices_1d(src_tokens)
        single_unk_token: Optional[str] = get_single_unk_token(src_tokens,
            word_ids, self.copy_unk_token, self.unk_idx)
        words, dict_tensors, contextual_embedding_tensor, src_lengths = (self
            .prepare_generator_inputs(word_ids, dict_feat,
            contextual_token_embedding))
        hypos_etc = self.sequence_generator(words, dict_tensors,
            contextual_embedding_tensor, src_lengths)
        hypos_list: List[Tuple[List[str], float, List[float]]] = []
        filter_token_list: List[int] = []
        if self.filter_eos_bos:
            filter_token_list = [self.target_vocab.bos_idx, self.
                target_vocab.eos_idx]
        for seq in hypos_etc:
            hyopthesis = seq[0]
            stringified = self.target_vocab.lookup_words_1d(hyopthesis,
                filter_token_list=filter_token_list, possible_unk_token=
                single_unk_token)
            hypos_list.append((stringified, seq[1], seq[2]))
        return hypos_list


class VectorNormalizer(torch.nn.Module):
    """Performs in-place normalization over all features of a dense feature
    vector by doing (x - mean)/stddev for each x in the feature vector.

    This is a ScriptModule so that the normalize function can be called at
    training time in the tensorizer, as well as at inference time by using it in
    your torchscript forward function. To use this in your tensorizer
    update_meta_data must be called once per row in your initialize function,
    and then calculate_feature_stats must be called upon the last time it runs.
    See usage in FloatListTensorizer for an example.

    Setting do_normalization=False will make the normalize function an identity
    function.
    """

    def __init__(self, dim: int, do_normalization: bool=True):
        super().__init__()
        self.num_rows = 0
        self.feature_sums = [0] * dim
        self.feature_squared_sums = [0] * dim
        self.do_normalization = do_normalization
        self.feature_avgs = [0.0] * dim
        self.feature_stddevs = [1.0] * dim

    def __getstate__(self):
        return {'num_rows': self.num_rows, 'feature_sums': self.
            feature_sums, 'feature_squared_sums': self.feature_squared_sums,
            'do_normalization': self.do_normalization, 'feature_avgs': self
            .feature_avgs, 'feature_stddevs': self.feature_stddevs}

    def __setstate__(self, state):
        self.num_rows = state['num_rows']
        self.feature_sums = state['feature_sums']
        self.feature_squared_sums = state['feature_squared_sums']
        self.do_normalization = state['do_normalization']
        self.feature_avgs = state['feature_avgs']
        self.feature_stddevs = state['feature_stddevs']

    def forward(self):
        pass

    def update_meta_data(self, vec):
        if self.do_normalization:
            self.num_rows += 1
            for i in range(len(vec)):
                self.feature_sums[i] += vec[i]
                self.feature_squared_sums[i] += vec[i] ** 2

    def calculate_feature_stats(self):
        if self.do_normalization:
            self.feature_avgs = [(x / self.num_rows) for x in self.feature_sums
                ]
            self.feature_stddevs = [((self.feature_squared_sums[i] / self.
                num_rows - self.feature_avgs[i] ** 2) ** 0.5) for i in
                range(len(self.feature_squared_sums))]

    def normalize(self, vec: List[List[float]]):
        if self.do_normalization:
            for i in range(len(vec)):
                for j in range(len(vec[i])):
                    vec[i][j] -= self.feature_avgs[j]
                    vec[i][j] /= self.feature_stddevs[j
                        ] if self.feature_stddevs[j] != 0 else 1.0
        return vec


class Infer:
    """A value which can be inferred from a forward pass. Infer objects should
    be passed as arguments or keyword arguments to Lazy objects; see Lazy
    documentation for more details.
    """

    def __init__(self, resolve_fn):
        """resolve_fn is called by Lazy on the arguments of the first forward pass
        to the Lazy module, and the Infer object will be replaced in the call by the
        output of this function. It should have the same signature as the
        Lazy-wrapped Module's forward function."""
        self.resolve = resolve_fn

    @classmethod
    def dimension(cls, dim):
        """A helper for creating Infer arguments looking at specific dimensions."""
        return cls(lambda input: input.size()[dim])


class UninitializedLazyModuleError(Exception):
    """A lazy module was used improperly."""


class Lazy(nn.Module):
    """
    A module which is able to infer some of its parameters from the inputs to
    its first forward pass. Lazy wraps any other nn.Module, and arguments can be passed
    that will be used to construct that wrapped Module after the first forward pass.
    If any of these arguments are Infer objects, those arguments will be replaced by
    calling the callback of the Infer object on the forward pass input.

    For instance,
    >>> Lazy(nn.Linear, Infer(lambda input: input.size(-1)), 4)
    Lazy()

    takes its in_features dimension from the last dimension of the input to its forward
    pass. This can be simplified to

    >>> Lazy(nn.Linear, Infer.dimension(-1), 4)

    or a partial can be created, for instance

    >>> LazyLinear = Lazy.partial(nn.Linear, Infer.dimension(-1))
    >>> LazyLinear(4)
    Lazy()

    Finally, these Lazy objects explicitly forbid treating themselves normally;
    they must instead be replaced by calling `init_lazy_modules`
    on your model before training. For instance,

    >>> ll = lazy.Linear(4)
    >>> seq = nn.Sequential(ll)
    >>> seq
    Sequential(
        0: Lazy(),
    )
    >>> init_lazy_modules(seq, torch.rand(1, 2)
    Sequential(
        0: Linear(in_features=2, out_features=4, bias=True)
    )
    """

    def __init__(self, module_class, *args, **kwargs):
        super().__init__()
        self._module = None
        self._module_class = module_class
        self._args = args
        self._kwargs = kwargs

    @classmethod
    def partial(cls, module_class, *args, **kwargs):
        return functools.partial(cls, module_class, *args, **kwargs)

    @property
    def _parameters(self):
        raise UninitializedLazyModuleError(
            'Must call init_lazy_modules before getting parameters')

    @_parameters.setter
    def _parameters(self, value):
        return None

    def __setattr__(self, name, value):
        return object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        if not self._module:
            constructor_args = [(arg if not isinstance(arg, Infer) else arg
                .resolve(*args, **kwargs)) for arg in self._args]
            constructor_kwargs = {key: (arg if not isinstance(arg, Infer) else
                arg.resolve(*args, **kwargs)) for key, arg in self._kwargs.
                items()}
            self._module = self._module_class(*constructor_args, **
                constructor_kwargs)
        return self._module(*args, **kwargs)

    def resolve(self):
        """Must make a call to forward before calling this function; returns the
        full nn.Module object constructed using inferred arguments/dimensions."""
        if not self._module:
            raise UninitializedLazyModuleError(
                'Must call forward before calling resolve on a lazy module')
        return self._module


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_pytext(_paritybench_base):
    pass
    def test_000(self):
        self._check(BiLSTM(*[], **{'num_layers': 1, 'bidirectional': 4, 'embed_dim': 4, 'hidden_dim': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(FCModelWithNanAndInfWts(*[], **{}), [torch.rand([10, 10])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GeLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Highway(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MaxPool(*[], **{'config': _mock_config(), 'n_input': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(MeanPool(*[], **{'config': _mock_config(), 'n_input': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(MultiLabelClassificationScores(*[], **{'scores': [_mock_layer()]}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(NoPool(*[], **{'config': _mock_config(), 'n_input': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(PlaceholderAttentionIdentity(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(PlaceholderIdentity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(SelfAttention(*[], **{'config': _mock_config(dropout=0.5, attn_dimension=4), 'n_input': 4}), [torch.rand([4, 4, 4])], {})

    def test_011(self):
        self._check(SeparableConv1d(*[], **{'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1, 'bottleneck': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_012(self):
        self._check(SlotAttention(*[], **{'config': _mock_config(attention_type=4), 'n_input': 4}), [torch.rand([4, 4, 4])], {})

    def test_013(self):
        self._check(Transformer(*[], **{}), [torch.zeros([4, 4], dtype=torch.int64)], {})

    def test_014(self):
        self._check(Trim1d(*[], **{'trim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(VectorNormalizer(*[], **{'dim': 4}), [], {})

