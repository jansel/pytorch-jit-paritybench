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
utils = _module
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
utils = _module
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
serialize = _module
tasks = _module
torchscript = _module
module = _module
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
roberta = _module
tensorizer = _module
xlm = _module
test_tensorizer = _module
test_tokenizer = _module
test_vocab = _module
bpe = _module
tokenizer = _module
utils = _module
vocab = _module
trainers = _module
ensemble_trainer = _module
hogwild_trainer = _module
trainer = _module
training_state = _module
utils = _module
ascii_table = _module
config_utils = _module
cuda = _module
distributed = _module
documentation = _module
embeddings = _module
file_io = _module
label = _module
lazy = _module
loss = _module
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
utils_test = _module
timing = _module
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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from typing import List


import torch


import torch.nn.functional as F


import collections


import enum


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Type


from typing import Union


from collections import OrderedDict


from typing import Optional


import math


from copy import deepcopy


from typing import Generator


from typing import Iterable


from typing import MutableMapping


from typing import Set


from torchtext import data as textdata


import itertools


import copy


import numpy as np


import re


from typing import NamedTuple


from collections import Counter


from logging import getLogger


from typing import Callable


from torchtext.vocab import Vocab


from torchtext import vocab


from typing import Mapping


from itertools import chain


from torchtext.data import Dataset


from torch import nn


from scipy.special import logsumexp


from torch.multiprocessing.spawn import spawn


from torch.utils.tensorboard import SummaryWriter


from enum import Enum


from itertools import zip_longest


import time


from torch import optim


import torch.nn as nn


import torch.jit as jit


from torch import jit


import torch.onnx.operators


from torch.nn import ModuleList


from inspect import signature


import torch.jit


from torch.jit import quantized


from enum import IntEnum


from enum import unique


import functools


from scipy.special import comb


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.onnx


import torch.cuda


from torch.nn import functional as F


from torch.serialization import default_restore_location


from typing import Sized


import torch as torch


from torch import Tensor


import random


import string


from torch.onnx import ExportTypes


from torch.onnx import OperatorExportTypes


from torch.optim import Optimizer as PT_Optimizer


from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR


from torch.optim.lr_scheduler import CyclicLR as TorchCyclicLR


from torch.optim.lr_scheduler import ExponentialLR as TorchExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR as TorchStepLR


from torch.optim.lr_scheduler import _LRScheduler


import warnings


from collections import defaultdict


from torch.autograd import Variable


from torch import sparse


from torch.utils import data


import abc


import logging


import torch.jit.quantized


import torch.multiprocessing as mp


from torchtext.data import Iterator


import torch.distributed as dist_c10d


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


import numpy


from typing import get_type_hints


class ScriptBatchInput(NamedTuple):
    """A batch of inputs for TorchScript Module(bundle of Tensorizer and Model)
    texts or tokens is required but multually exclusive
    Args:
        texts: a batch of raw text inputs
        tokens: a batch of pre-tokenized inputs
        languages: language for each input in the batch
    """
    texts: Optional[List[List[str]]]
    tokens: Optional[List[List[List[str]]]]
    languages: Optional[List[List[str]]]


class TensorizerScriptImpl(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device: str = ''

    @torch.jit.export
    def set_device(self, device: str):
        self.device = device

    def batch_size(self, inputs: ScriptBatchInput) ->int:
        texts: Optional[List[List[str]]] = inputs.texts
        tokens: Optional[List[List[List[str]]]] = inputs.tokens
        if texts is not None:
            return len(texts)
        elif tokens is not None:
            return len(tokens)
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    def row_size(self, inputs: ScriptBatchInput) ->int:
        texts: Optional[List[List[str]]] = inputs.texts
        tokens: Optional[List[List[List[str]]]] = inputs.tokens
        if texts is not None:
            return len(texts[0])
        elif tokens is not None:
            return len(tokens[0])
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    def get_texts_by_index(self, texts: Optional[List[List[str]]], index: int) ->Optional[List[str]]:
        if texts is None or len(texts) == 0:
            return None
        return texts[index]

    def get_tokens_by_index(self, tokens: Optional[List[List[List[str]]]], index: int) ->Optional[List[List[str]]]:
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[index]

    def tokenize(self, *args, **kwargs):
        """
        This functions will receive the inputs from Clients, usually there are
        two possible inputs
        1) a row of texts: List[str]
        2) a row of pre-processed tokens: List[List[str]]

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    def numberize(self, *args, **kwargs):
        """
        This functions will receive the outputs from function: tokenize() or
        will be called directly from PyTextTensorizer function: numberize().

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    def tensorize(self, *args, **kwargs):
        """
        This functions will receive a list(e.g a batch) of outputs
        from function numberize(), padding and convert to output tensors.

        Override this function to be TorchScriptable, e.g you need to declare
        concrete input arguments with type hints.
        """
        raise NotImplementedError

    @torch.jit.ignore
    def tensorize_wrapper(self, *args, **kwargs):
        """
        This functions will receive a list(e.g a batch) of outputs
        from function numberize(), padding and convert to output tensors.

        It will be called in PyText Tensorizer during training time, this
        function is not torchscriptiable because it depends on cuda.device().
        """
        with to_device(self, cuda.device()):
            return self.tensorize(*args, **kwargs)

    @torch.jit.ignore
    def torchscriptify(self):
        return torch.jit.script(self)


class ScriptTokenizerBase(torch.jit.ScriptModule):

    @torch.jit.script_method
    def tokenize(self, input: str) ->List[Tuple[str, int, int]]:
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
        1) a single text
        2) a token

        Returns a list of tokens with start and end indices in original input.
        """
        raise NotImplementedError


class ScriptDoNothingTokenizer(ScriptTokenizerBase):

    @torch.jit.script_method
    def tokenize(self, raw_token: str) ->List[Tuple[str, int, int]]:
        return [(raw_token, -1, -1)]


@torch.jit.script
def list_membership(item: int, list: List[int]):
    item_present = False
    for i in list:
        if item == i:
            item_present = True
    return item_present


class ScriptVocabulary(torch.jit.ScriptModule):

    def __init__(self, vocab_list, unk_idx: int=0, pad_idx: int=-1, bos_idx: int=-1, eos_idx: int=-1, mask_idx: int=-1):
        super().__init__()
        self.vocab = torch.jit.Attribute(vocab_list, List[str])
        self.unk_idx = torch.jit.Attribute(unk_idx, int)
        self.pad_idx = torch.jit.Attribute(pad_idx, int)
        self.eos_idx = torch.jit.Attribute(eos_idx, int)
        self.bos_idx = torch.jit.Attribute(bos_idx, int)
        self.mask_idx = torch.jit.Attribute(mask_idx, int)
        self.idx = torch.jit.Attribute({word: i for i, word in enumerate(vocab_list)}, Dict[str, int])
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
    def lookup_words_1d(self, values: torch.Tensor, filter_token_list: List[int]=(), possible_unk_token: Optional[str]=None) ->List[str]:
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
    def lookup_words_1d_cycle_heuristic(self, values: torch.Tensor, filter_token_list: List[int], ordered_unks_token: List[str]) ->List[str]:
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
            return self.vocab[self.unk_idx] if possible_unk_token is None else possible_unk_token


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
        defaults.update({k: getattr(cls, k) for k in annotations if hasattr(cls, k)})
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
            raise TypeError(f'Failed to specify {unspecified_fields} for {type(self)}')
        overspecified_fields = specified - required
        if overspecified_fields:
            raise TypeError(f'Specified non-existent fields {overspecified_fields} for {type(self)}')
        vars(self).update(kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in sorted(self._asdict().items()):
            lines += f'{key}: {val}'.split('\n')
        return '\n    '.join(lines)

    def __eq__(self, other):
        """Mainly a convenience utility for unit testing."""
        return type(self) == type(other) and self._asdict() == other._asdict()


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
    _registered_components: Dict[ComponentType, Dict[Type, Type]] = collections.defaultdict(dict)

    @classmethod
    def add(cls, component_type: ComponentType, cls_to_add: Type, config_cls: Type):
        component = cls._registered_components[component_type]
        if config_cls in component:
            raise RegistryError(f"Cannot add {cls_to_add} to {component_type} for task_config type {config_cls}; it's already registered for {component[config_cls]}")
        component[config_cls] = cls_to_add

    @classmethod
    def get(cls, component_type: ComponentType, config_cls: Type) ->Type:
        if component_type not in cls._registered_components:
            raise RegistryError(f"type {component_type} doesn't exist")
        if config_cls not in cls._registered_components[component_type]:
            raise RegistryError(f'unregistered config class {config_cls.__name__} for {component_type}')
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
        return tuple(sub_cls for sub_cls in cls.configs(config_cls.__COMPONENT_TYPE__) if issubclass(sub_cls.__COMPONENT__, config_cls.__COMPONENT__))


class ComponentMeta(type):

    def __new__(metacls, typename, bases, namespace):
        if 'Config' not in namespace:
            parent_config = next((base.Config for base in bases if hasattr(base, 'Config')), None)
            if parent_config is not None:


                class Config(parent_config):
                    pass
            else:


                class Config(ConfigBase):
                    pass
            namespace['Config'] = Config
        component_type = next((base.__COMPONENT_TYPE__ for base in bases if hasattr(base, '__COMPONENT_TYPE__')), namespace.get('__COMPONENT_TYPE__'))
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
        return [r for r in result if not (isinstance(getattr(cls, r, None), type) and issubclass(getattr(cls, r, None), ConfigBase))]


class Component(metaclass=ComponentMeta):


    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(config, *args, **kwargs)

    def __init__(self, config=None, *args, **kwargs):
        self.config = config


class Token(NamedTuple):
    value: str
    start: int
    end: int


class Tokenizer(Component):
    """A simple regex-splitting tokenizer."""
    __COMPONENT_TYPE__ = ComponentType.TOKENIZER
    __EXPANSIBLE__ = True


    class Config(Component.Config):
        split_regex: str = '\\s+'
        lowercase: bool = True

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.split_regex, config.lowercase)

    def __init__(self, split_regex='\\s+', lowercase=True):
        super().__init__(None)
        self.split_regex = split_regex
        self.lowercase = lowercase

    def tokenize(self, input: str) ->List[Token]:
        tokens = []
        start = 0
        tokenize_input = input.lower() if self.lowercase else input
        for match in re.finditer(self.split_regex, tokenize_input):
            split_start, split_end = match.span()
            tokens.append(Token(tokenize_input[start:split_start], start, split_start))
            start = split_end
        tokens.append(Token(tokenize_input[start:len(input)], start, len(input)))
        return [token for token in tokens if token.value]

    def torchscriptify(self):
        raise NotImplementedError

    def decode(self, sentence: str):
        return sentence


class VocabLookup(torch.jit.ScriptModule):
    """
    TorchScript implementation of lookup_tokens() in pytext/data/tensorizers.py
    """

    def __init__(self, vocab: ScriptVocabulary):
        super().__init__()
        self.vocab = vocab

    @torch.jit.script_method
    def forward(self, tokens: List[Tuple[str, int, int]], bos_idx: Optional[int]=None, eos_idx: Optional[int]=None, use_eos_token_for_bos: bool=False, max_seq_len: int=2 ** 30) ->Tuple[List[int], List[int], List[int]]:
        """Convert tokens into ids by doing vocab look-up.

        Convert tokens into ids by doing vocab look-up. It will also append
        bos & eos index into token_ids if needed. A token is represented by
        a Tuple[str, int, int], which is [token, start_index, end_index].

        Args:
            tokens: List of tokens with start and end position in the original
                text. start and end index could be optional (e.g value is -1)
            bos_idx: index of begin of sentence, optional.
            eos_idx: index of end of sentence, optional.
            use_eos_token_for_bos: use eos index as bos.
            max_seq_len: maximum tokens length.
        """
        if bos_idx is None:
            bos_idx = -1
        if eos_idx is None:
            eos_idx = -1
        text_tokens: List[str] = []
        start_idxs: List[int] = []
        end_idxs: List[int] = []
        max_seq_len = max_seq_len - (1 if bos_idx >= 0 else 0) - (1 if eos_idx >= 0 else 0)
        for i in range(min(len(tokens), max_seq_len)):
            token: Tuple[str, int, int] = tokens[i]
            text_tokens.append(token[0])
            start_idxs.append(token[1])
            end_idxs.append(token[2])
        token_ids: List[int] = self.vocab.lookup_indices_1d(text_tokens)
        if bos_idx >= 0:
            if use_eos_token_for_bos:
                bos_idx = eos_idx
            token_ids = [bos_idx] + token_ids
            start_idxs = [-1] + start_idxs
            end_idxs = [-1] + end_idxs
        if eos_idx >= 0:
            token_ids.append(eos_idx)
            start_idxs.append(-1)
            end_idxs.append(-1)
        return token_ids, start_idxs, end_idxs


class SpecialToken(str):

    def __eq__(self, other):
        return isinstance(other, SpecialToken) and super().__eq__(other)
    __hash__ = str.__hash__


BOS = SpecialToken('__BEGIN_OF_SENTENCE__')


EOS = SpecialToken('__END_OF_SENTENCE__')


MASK = SpecialToken('__MASK__')


PAD = SpecialToken('__PAD__')


UNK = SpecialToken('__UNKNOWN__')


def should_iter(i):
    """Whether or not an object looks like a python iterable (not including strings)."""
    return hasattr(i, '__iter__') and not isinstance(i, str) and not (isinstance(i, torch.Tensor) and (i.dim() == 0 or len(i) == 0))


class Vocabulary:
    """A mapping from indices to vocab elements."""

    def __init__(self, vocab_list: List[str], counts: List=None, replacements: Optional[Dict[str, str]]=None, unk_token: str=UNK, pad_token: str=PAD, bos_token: str=BOS, eos_token: str=EOS, mask_token: str=MASK):
        self._vocab = vocab_list
        self.counts = counts
        self.idx = {word: i for i, word in enumerate(vocab_list)}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        if replacements:
            self.replace_tokens(replacements)
        self.unk_token_counter = [0, 0]
        self.unk_example_counter = [0, 0]
        self.messages_printed = 0

    def replace_tokens(self, replacements):
        """Replace tokens in vocab with given replacement.
           Used for replacing special strings for special tokens.
           e.g. '[UNK]' for UNK"""
        for token, replacement in replacements.items():
            idx = self.idx.pop(token, len(self._vocab))
            if idx == len(self._vocab):
                self._vocab.append(replacement)
                self.counts.append(1)
            else:
                self._vocab[idx] = replacement
            self.idx[replacement] = idx

    def lookup_all(self, nested_values):
        res, unk_counter, total = self.lookup_all_internal(nested_values)
        self.unk_token_counter[0] += unk_counter
        self.unk_token_counter[1] += total
        self.unk_example_counter[1] += 1
        if total > 3 and unk_counter / total > 0.75:
            self.unk_example_counter[0] += 1
            if self.unk_example_counter[0] % 100 == 0 and self.messages_printed < 200:
                self.messages_printed += 1
                c1, c2 = self.unk_token_counter
                None
                None
                c1, c2 = self.unk_example_counter
                None
                None
                None
        return res

    def lookup_all_internal(self, nested_values):
        """
        Look up a value or nested container of values in the vocab index.
        The return value will have the same shape as the input, with all values
        replaced with their respective indicies.
        """

        def lookup(value):
            if self.unk_token in self.idx:
                unk_idx = self.get_unk_index()
                v = self.idx.get(value, unk_idx)
                return v, 1 if v == unk_idx else 0, 1
            else:
                return self.idx[value], 0, 1
        if not should_iter(nested_values):
            return lookup(nested_values)
        else:
            indices = []
            unks = 0
            total = 0
            for value in nested_values:
                v, unk, t = self.lookup_all_internal(value)
                indices.append(v)
                unks += unk
                total += t
            return indices, unks, total

    def get_unk_index(self, value=None):
        if value is None:
            return self.idx[self.unk_token]
        else:
            return self.idx.get(self.unk_token, value)

    def get_pad_index(self, value=None):
        if value is None:
            return self.idx[self.pad_token]
        else:
            return self.idx.get(self.pad_token, value)

    def get_mask_index(self, value=None):
        if value is None:
            return self.idx[self.mask_token]
        else:
            return self.idx.get(self.mask_token, value)

    def get_bos_index(self, value=None):
        if value is None:
            return self.idx[self.bos_token]
        else:
            return self.idx.get(self.bos_token, value)

    def get_eos_index(self, value=None):
        if value is None:
            return self.idx[self.eos_token]
        else:
            return self.idx.get(self.eos_token, value)

    def __getitem__(self, item):
        return self._vocab[item]

    def __len__(self):
        return len(self._vocab)


@torch.jit.script
def list_max(l: List[int]):
    max_value = l[0]
    for i in range(len(l) - 1):
        max_value = max(max_value, l[i + 1])
    return max_value


@torch.jit.script
def pad_2d(batch: List[List[int]], seq_lens: List[int], pad_idx: int) ->List[List[int]]:
    pad_to_length = list_max(seq_lens)
    for sentence in batch:
        for _ in range(pad_to_length - len(sentence)):
            sentence.append(pad_idx)
    return batch


class TokenTensorizerScriptImpl(TensorizerScriptImpl):

    def __init__(self, add_bos_token: bool, add_eos_token: bool, use_eos_token_for_bos: bool, max_seq_len: int, vocab: Vocabulary, tokenizer: Optional[Tokenizer]):
        super().__init__()
        if tokenizer is not None and hasattr(tokenizer, 'torchscriptify'):
            try:
                self.tokenizer = tokenizer.torchscriptify()
            except NotImplementedError:
                self.tokenizer = None
        else:
            self.tokenizer = None
        self.do_nothing_tokenizer = ScriptDoNothingTokenizer()
        self.vocab = ScriptVocabulary(list(vocab), pad_idx=vocab.get_pad_index(), bos_idx=vocab.get_bos_index() if add_bos_token else -1, eos_idx=vocab.get_eos_index() if add_eos_token else -1)
        self.vocab_lookup_1d = VocabLookup(self.vocab)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len

    def get_texts_by_index(self, texts: Optional[List[List[str]]], index: int) ->Optional[str]:
        if texts is None or len(texts) == 0:
            return None
        return texts[index][0]

    def get_tokens_by_index(self, tokens: Optional[List[List[List[str]]]], index: int) ->Optional[List[str]]:
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[index][0]

    def _lookup_tokens_1d(self, tokens: List[Tuple[str, int, int]]) ->Tuple[List[int], List[int], List[int]]:
        return self.vocab_lookup_1d(tokens, bos_idx=self.vocab.bos_idx if self.add_bos_token else None, eos_idx=self.vocab.eos_idx if self.add_eos_token else None, use_eos_token_for_bos=self.use_eos_token_for_bos, max_seq_len=self.max_seq_len)

    def tokenize(self, row_text: Optional[str], row_pre_tokenized: Optional[List[str]]) ->List[Tuple[str, int, int]]:
        tokens: List[Tuple[str, int, int]] = []
        if row_text is not None:
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(row_text)
        elif row_pre_tokenized is not None:
            for token in row_pre_tokenized:
                tokens.extend(self.do_nothing_tokenizer.tokenize(token))
        return tokens

    def numberize(self, text_tokens: List[Tuple[str, int, int]]) ->Tuple[List[int], int, List[Tuple[int, int]]]:
        token_indices: List[int] = []
        token_starts: List[int] = []
        token_ends: List[int] = []
        token_indices, token_starts, token_ends = self._lookup_tokens_1d(text_tokens)
        token_ranges: List[Tuple[int, int]] = []
        for s, e in zip(token_starts, token_ends):
            token_ranges.append((s, e))
        return token_indices, len(token_indices), token_ranges

    def tensorize(self, tokens_2d: List[List[int]], seq_lens_1d: List[int], positions_2d: List[List[Tuple[int, int]]]) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_indices_tensor: torch.Tensor = torch.tensor(pad_2d(tokens_2d, seq_lens=seq_lens_1d, pad_idx=self.vocab.pad_idx), dtype=torch.long)
        token_starts_2d: List[List[int]] = []
        token_ends_2d: List[List[int]] = []
        for position_list in positions_2d:
            token_starts_2d.append([x[0] for x in position_list])
            token_ends_2d.append([x[1] for x in position_list])
        token_positions_tensor = torch.stack([torch.tensor(pad_2d(token_starts_2d, seq_lens=seq_lens_1d, pad_idx=-1), dtype=torch.long), torch.tensor(pad_2d(token_ends_2d, seq_lens=seq_lens_1d, pad_idx=-1), dtype=torch.long)], dim=2)
        return token_indices_tensor, torch.tensor(seq_lens_1d, dtype=torch.long), token_positions_tensor

    def forward(self, inputs: ScriptBatchInput) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[Tuple[int, int]]] = []
        for idx in range(self.batch_size(inputs)):
            tokens: List[Tuple[str, int, int]] = self.tokenize(self.get_texts_by_index(inputs.texts, idx), self.get_tokens_by_index(inputs.tokens, idx))
            numberized: Tuple[List[int], int, List[Tuple[int, int]]] = self.numberize(tokens)
            tokens_2d.append(numberized[0])
            seq_lens_1d.append(numberized[1])
            positions_2d.append(numberized[2])
        return self.tensorize(tokens_2d, seq_lens_1d, positions_2d)


CUDA_ENABLED = False


def FloatTensor(*args):
    if CUDA_ENABLED:
        return torch.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


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
        self.precision_range = self.config.precision_range_lower, self.config.precision_range_upper
        self.precision_values, self.delta = loss_utils.range_to_anchors_and_delta(self.precision_range, self.num_anchors)
        self.biases = nn.Parameter(FloatTensor(self.config.num_classes, self.config.num_anchors).zero_())
        self.lambdas = nn.Parameter(FloatTensor(self.config.num_classes, self.config.num_anchors).data.fill_(1.0))

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
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
            raise ValueError('num classes is %d while logits width is %d' % (self.num_classes, C))
        labels, weights = AUCPRHingeLoss._prepare_labels_weights(logits, targets, weights=weights)
        lambdas = loss_utils.lagrange_multiplier(self.lambdas)
        hinge_loss = loss_utils.weighted_hinge_loss(labels.unsqueeze(-1), logits.unsqueeze(-1) - self.biases, positive_weights=1.0 + lambdas * (1.0 - self.precision_values), negative_weights=lambdas * self.precision_values)
        class_priors = loss_utils.build_class_priors(labels, weights=weights)
        lambda_term = class_priors.unsqueeze(-1) * (lambdas * (1.0 - self.precision_values))
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
        labels = FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
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

    def __init__(self, num_tags: int, ignore_index: int, default_label_pad_index: int) ->None:
        if num_tags <= 0:
            raise ValueError(f'Invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags + 2, num_tags + 2))
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

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, reduce: bool=True) ->torch.Tensor:
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
    def decode(self, emissions: torch.Tensor, seq_lens: torch.Tensor) ->torch.Tensor:
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

    def _compute_joint_llh(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        seq_len = emissions.shape[1]
        llh = self.transitions[self.start_tag, tags[:, (0)]].unsqueeze(1)
        llh += emissions[:, (0), :].gather(1, tags[:, (0)].view(-1, 1)) * mask[:, (0)].unsqueeze(1)
        for idx in range(1, seq_len):
            old_state, new_state = tags[:, (idx - 1)].view(-1, 1), tags[:, (idx)].view(-1, 1)
            emission_scores = emissions[:, (idx), :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, (idx)].unsqueeze(1)
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))
        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1)
        return llh.squeeze(1)

    def _compute_log_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        seq_len = emissions.shape[1]
        log_prob = emissions[:, (0)].clone()
        log_prob += self.transitions[(self.start_tag), :self.start_tag].unsqueeze(0)
        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, (idx)].unsqueeze(1)
            broadcast_transitions = self.transitions[:self.start_tag, :self.start_tag].unsqueeze(0)
            broadcast_logprob = log_prob.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions
            score = torch.logsumexp(score, 1)
            log_prob = score * mask[:, (idx)].unsqueeze(1) + log_prob.squeeze(1) * (1 - mask[:, (idx)].unsqueeze(1))
        log_prob += self.transitions[:self.start_tag, (self.end_tag)].unsqueeze(0)
        return torch.logsumexp(log_prob.squeeze(1), 1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        tensor_device = emissions.device
        seq_len = emissions.shape[1]
        mask = mask
        log_prob = emissions[:, (0)].clone()
        log_prob += self.transitions[(self.start_tag), :self.start_tag].unsqueeze(0)
        end_scores = log_prob + self.transitions[:self.start_tag, (self.end_tag)].unsqueeze(0)
        best_scores_list: List[torch.Tensor] = []
        empty_data: List[int] = []
        best_paths_list = [torch.tensor(empty_data, device=tensor_device).long()]
        best_scores_list.append(end_scores.unsqueeze(1))
        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, (idx)].unsqueeze(1)
            broadcast_transmissions = self.transitions[:self.start_tag, :self.start_tag].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)
            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob
            max_scores, max_score_indices = torch.max(score, 1)
            best_paths_list.append(max_score_indices.unsqueeze(1))
            end_scores = max_scores + self.transitions[:self.start_tag, (self.end_tag)].unsqueeze(0)
            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores
        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)
        _, max_indices_from_scores = torch.max(best_scores, 2)
        valid_index_tensor = torch.tensor(0, device=tensor_device).long()
        if self.ignore_index == self.default_label_pad_index:
            padding_tensor = valid_index_tensor
        else:
            padding_tensor = torch.tensor(self.ignore_index, device=tensor_device).long()
        labels = max_indices_from_scores[:, (seq_len - 1)]
        labels = self._mask_tensor(labels, 1 - mask[:, (seq_len - 1)], padding_tensor)
        all_labels = labels.unsqueeze(1).long()
        for idx in range(seq_len - 2, -1, -1):
            indices_for_lookup = all_labels[:, (-1)].clone()
            indices_for_lookup = self._mask_tensor(indices_for_lookup, indices_for_lookup == self.ignore_index, valid_index_tensor)
            indices_from_prev_pos = best_paths[:, (idx), :].gather(1, indices_for_lookup.view(-1, 1).long()).squeeze(1)
            indices_from_prev_pos = self._mask_tensor(indices_from_prev_pos, 1 - mask[:, (idx + 1)], padding_tensor)
            indices_from_max_scores = max_indices_from_scores[:, (idx)]
            indices_from_max_scores = self._mask_tensor(indices_from_max_scores, mask[:, (idx + 1)], padding_tensor)
            labels = torch.where(indices_from_max_scores == self.ignore_index, indices_from_prev_pos, indices_from_max_scores)
            labels = self._mask_tensor(labels, 1 - mask[:, (idx)], padding_tensor)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)
        return torch.flip(all_labels, [1])

    def _make_mask_from_targets(self, targets):
        mask = targets.ne(self.ignore_index).float()
        return mask

    def _make_mask_from_seq_lens(self, seq_lens):
        seq_lens = seq_lens.view(-1, 1)
        max_len = torch.max(seq_lens)
        range_tensor = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
        range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
        mask = (range_tensor < seq_lens).float()
        return mask

    def _mask_tensor(self, score_tensor, mask_condition, mask_value):
        masked_tensor = torch.where(mask_condition, mask_value, score_tensor)
        return masked_tensor

    def export_to_caffe2(self, workspace, init_net, predict_net, logits_output_name):
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
        workspace.FeedBlob(str(crf_transitions), self.get_transitions().numpy())
        logits_squeezed = predict_net.Squeeze(logits_output_name, dims=[0])
        new_logits = apply_crf(init_net, predict_net, crf_transitions, logits_squeezed, self.num_tags)
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
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
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

    def __init__(self, embeddings: Iterable[EmbeddingBase], concat: bool) ->None:
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
            raise Exception(f'expecting {self.num_emb_modules} embeddings, ' + f'but got {len(emb_input)} input')
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


class ScriptableEmbeddingList(EmbeddingBase):
    """

    This class is a Torchscript-friendly version of
    pytext.models.embeddings.EmbeddingList. The main differences are that it
    requires input arguments to be passed in as a list of Tensors, since
    Torchscript does not allow variable arguments, and that it only supports
    concat mode, since Torchscript does not support return value variance.

    """


    class Wrapper1(torch.nn.Module):

        def __init__(self, embedding: EmbeddingBase):
            super().__init__()
            self._embedding = embedding

        def forward(self, xs: List[torch.Tensor]):
            return self._embedding(xs[0])


    class Wrapper3(torch.nn.Module):

        def __init__(self, embedding: EmbeddingBase):
            super().__init__()
            self._embedding = embedding

        def forward(self, xs: List[torch.Tensor]):
            return self._embedding(xs[0], xs[1], xs[2])

    @staticmethod
    def _adapt_embedding(embedding: torch.nn.Module) ->torch.nn.Module:
        param_count = len(signature(embedding.forward).parameters)
        if param_count == 1:
            return ScriptableEmbeddingList.Wrapper1(embedding)
        elif param_count == 3:
            return ScriptableEmbeddingList.Wrapper3(embedding)
        raise AssertionError(f'Unsupported parameter count {param_count}. If a new embedding class has been added, you will need to add support in this class.')

    def __init__(self, embeddings: Iterable[EmbeddingBase]):
        EmbeddingBase.__init__(self, 0)
        embeddings = list(filter(None, embeddings))
        self.num_emb_modules = sum(emb.num_emb_modules for emb in embeddings)
        embeddings_list: List[EmbeddingBase] = []
        input_start_indices: List[int] = []
        start = 0
        embedding_dim = 0
        for emb in embeddings:
            if emb.embedding_dim > 0:
                embeddings_list.append(emb)
                input_start_indices.append(start)
                embedding_dim += emb.embedding_dim
            start += emb.num_emb_modules
        self.embeddings_list = torch.nn.ModuleList(map(ScriptableEmbeddingList._adapt_embedding, embeddings_list))
        self.input_start_indices: Tuple[int] = tuple(input_start_indices)
        assert len(self.embeddings_list) > 0, 'must have at least 1 sub embedding'
        self.embedding_dim = embedding_dim

    def forward(self, emb_input: List[List[torch.Tensor]]) ->torch.Tensor:
        """
        Get embeddings from all sub-embeddings and either concatenate them
        into one Tensor or return them in a tuple.

        Args:
            emb_input (type): Sequence of token level embeddings to combine.
                The inputs should match the size of configured embeddings. Each
                of them is a List of Tensors.

        Returns:
            torch.Tensor: a Tensor is returned by concatenating all embeddings.

        """
        if self.num_emb_modules != len(emb_input):
            raise Exception(f'expecting {self.num_emb_modules} embeddings, ' + f'but got {len(emb_input)} input')
        tensors = []
        for emb, start in zip(self.embeddings_list, self.input_start_indices):
            tensors.append(emb(emb_input[start]))
        return torch.cat(tensors, 2)

    def get_param_groups_for_optimizer(self) ->List[Dict[str, torch.nn.Parameter]]:
        """
        Organize child embedding parameters into param_groups (or layers), so the
        optimizer and / or schedulers can have custom behavior per layer. The
        param_groups from each child embedding are aligned at the first (lowest)
        param_group.
        """
        param_groups: List[Dict[str, torch.nn.Parameter]] = []
        for module_name, embedding_module in self.embeddings_list.named_children():
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


class FieldMeta:
    vocab: Vocab
    vocab_size: int
    vocab_export_name: str
    pad_token_idx: int
    unk_token_idx: int
    init_token_idx: int
    eos_token_idx: int
    nesting_meta: Any
    dummy_model_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]


class EmbedInitStrategy(Enum):
    RANDOM = 'random'
    ZERO = 'zero'


class PackageFileName:
    SERIALIZED_EMBED = 'pretrained_embed_pt_serialized'
    RAW_EMBED = 'pretrained_embed_raw'


def append_dialect(word: str, dialect: str) ->str:
    if word.endswith('-{}'.format(dialect)):
        return word
    else:
        return f'{word}-{dialect}'


def get_pytext_home():
    internal_home = os.path.realpath(os.path.join(__file__, '../../'))
    oss_home = os.path.realpath(os.path.join(__file__, '../../../'))
    default_home = ''
    if PathManager.exists(os.path.join(internal_home, 'tests')):
        default_home = internal_home
    elif PathManager.exists(os.path.join(oss_home, 'tests')):
        default_home = oss_home
    else:
        default_home = os.path.dirname(__file__)
    pytext_home = os.environ.get('PYTEXT_HOME', default_home)
    return pytext_home


def get_absolute_path(file_path: str) ->str:
    if os.path.isabs(file_path):
        return file_path
    absolute_path = os.path.realpath(os.path.join(PYTEXT_HOME, file_path))
    if PathManager.exists(absolute_path):
        return absolute_path
    return file_path


class PretrainedEmbedding(object):
    """
    Utility class for loading/caching/initializing word embeddings
    """

    def __init__(self, embeddings_path: str=None, lowercase_tokens: bool=True, skip_header: bool=True, delimiter: str=' ') ->None:
        self.lowercase_tokens = lowercase_tokens
        if embeddings_path:
            embeddings_path = get_absolute_path(embeddings_path)
            if PathManager.isdir(embeddings_path):
                serialized_embed_path = os.path.join(embeddings_path, PackageFileName.SERIALIZED_EMBED)
                raw_embeddings_path = os.path.join(embeddings_path, PackageFileName.RAW_EMBED)
            elif PathManager.isfile(embeddings_path):
                serialized_embed_path = ''
                raw_embeddings_path = embeddings_path
            else:
                raise FileNotFoundError(f"{embeddings_path} not found. Can't load pretrained embeddings.")
            if PathManager.isfile(serialized_embed_path):
                try:
                    self.load_cached_embeddings(serialized_embed_path)
                except Exception:
                    None
                    self.load_pretrained_embeddings(raw_embeddings_path, lowercase_tokens=lowercase_tokens, skip_header=skip_header, delimiter=delimiter)
            else:
                self.load_pretrained_embeddings(raw_embeddings_path, lowercase_tokens=lowercase_tokens, skip_header=skip_header, delimiter=delimiter)
        else:
            self.embed_vocab = []
            self.stoi = {}
            self.embedding_vectors = None

    def filter_criteria(self, token: str) ->bool:
        return True

    def normalize_token(self, token: str) ->str:
        """
        Apply normalizations to the input token for the
        embedding space
        """
        if self.lowercase_tokens:
            return token.lower()
        else:
            return token

    def load_pretrained_embeddings(self, raw_embeddings_path: str, append: bool=False, dialect: str=None, lowercase_tokens: bool=True, skip_header: bool=True, delimiter: str=' ') ->None:
        """
        Loading raw embeddings vectors from file in the format:
        num_words dim
        word_i v0, v1, v2, ...., v_dim
        word_2 v0, v1, v2, ...., v_dim
        ....
        Optionally appends _dialect to every token in the vocabulary
        (for XLU embeddings).
        """
        chunk_vocab = []

        def iter_parser(skip_header: int=0, delimiter: str=' ', dtype: type=np.float32):
            """ Iterator to load numpy 1-d array from multi-row text file,
            where format is assumed to be:
                word_i v0, v1, v2, ...., v_dim
                word_2 v0, v1, v2, ...., v_dim
            The iterator will omit the first column (vocabulary) and via closure
            store values into the 'chunk_vocab' list.
            """
            tokens = set()
            with PathManager.open(raw_embeddings_path, 'r') as txtfile:
                for _ in range(skip_header):
                    next(txtfile)
                for line in txtfile:
                    split_line = line.rstrip('\r\n ').split(delimiter)
                    token = split_line[0]
                    token = self.normalize_token(token)
                    if token not in tokens and self.filter_criteria(token):
                        chunk_vocab.append(token)
                        for item in split_line[1:]:
                            yield dtype(item)
        t = time.time()
        skip_header = 1 if skip_header else 0
        embed_array = np.fromiter(iter_parser(skip_header=skip_header, delimiter=delimiter), dtype=np.float32)
        embed_matrix = embed_array.reshape((len(chunk_vocab), -1))
        None
        if not append:
            self.embed_vocab = []
            self.stoi = {}
        if dialect is not None:
            chunk_vocab = [append_dialect(word, dialect) for word in chunk_vocab]
        self.embed_vocab.extend(chunk_vocab)
        self.stoi = {word: i for i, word in enumerate(chunk_vocab)}
        if append and self.embedding_vectors is not None:
            self.embedding_vectors = torch.cat((self.embedding_vectors, torch.Tensor(embed_matrix)))
        else:
            self.embedding_vectors = torch.Tensor(embed_matrix)

    def cache_pretrained_embeddings(self, cache_path: str) ->None:
        """
        Cache the processed embedding vectors and vocab to a file for faster
        loading
        """
        t = time.time()
        None
        with PathManager.open(cache_path, 'wb') as f:
            torch.save((self.embed_vocab, self.stoi, self.embedding_vectors), f)
        None

    def load_cached_embeddings(self, cache_path: str) ->None:
        """
        Load cached embeddings from file
        """
        with PathManager.open(cache_path, 'rb') as f:
            self.embed_vocab, self.stoi, self.embedding_vectors = torch.load(f)

    def initialize_embeddings_weights(self, str_to_idx: Dict[str, int], unk: str, embed_dim: int, init_strategy: EmbedInitStrategy) ->torch.Tensor:
        """
        Initialize embeddings weights of shape (len(str_to_idx), embed_dim) from the
        pretrained embeddings vectors. Words that are not in the pretrained
        embeddings list will be initialized according to `init_strategy`.
        :param str_to_idx: a dict that maps words to indices that the model expects
        :param unk: unknown token
        :param embed_dim: the embeddings dimension
        :param init_strategy: method of initializing new tokens
        :returns: a float tensor of dimension (vocab_size, embed_dim)
        """
        pretrained_embeds_weight = torch.Tensor(len(str_to_idx), embed_dim)
        if init_strategy == EmbedInitStrategy.RANDOM:
            pretrained_embeds_weight.normal_(0, 1)
        elif init_strategy == EmbedInitStrategy.ZERO:
            pretrained_embeds_weight.fill_(0.0)
        else:
            raise ValueError("Unknown embedding initialization strategy '{}'".format(init_strategy))
        assert self.embedding_vectors is not None and self.embed_vocab is not None
        assert pretrained_embeds_weight.shape[-1] == self.embedding_vectors.shape[-1], f"shape of pretrained_embeds_weight {pretrained_embeds_weight.shape[-1]}         and embedding_vectors {self.embedding_vectors.shape[-1]} doesn't match!"
        unk_idx = str_to_idx[unk]
        oov_count = 0
        for word, idx in str_to_idx.items():
            if word in self.stoi and idx != unk_idx:
                pretrained_embeds_weight[idx] = self.embedding_vectors[self.stoi[word]]
            else:
                oov_count += 1
        None
        return pretrained_embeds_weight


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
            res = ' '.join([self.vocab._vocab[index] for index in token_indices])
            if hasattr(self, 'tokenizer'):
                if hasattr(self.tokenizer, 'decode'):
                    res = self.tokenizer.decode(res)
        return res

    def torchscriptify(self):
        return self.tensorizer_script_impl.torchscriptify()


class ModuleConfig(ConfigBase):
    load_path: Optional[str] = None
    save_path: Optional[str] = None
    freeze: bool = False
    shared_module_key: Optional[str] = None


class WordFeatConfig(ModuleConfig):
    embed_dim: int = 100
    freeze: bool = False
    embedding_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM
    embedding_init_range: Optional[List[float]] = None
    embeddding_init_std: Optional[float] = 0.02
    export_input_names: List[str] = ['tokens_vals']
    pretrained_embeddings_path: str = ''
    vocab_file: str = ''
    vocab_size: int = 0
    vocab_from_train_data: bool = True
    vocab_from_all_data: bool = False
    vocab_from_pretrained_embeddings: bool = False
    lowercase_tokens: bool = True
    min_freq: int = 1
    mlp_layer_dims: Optional[List[int]] = []
    padding_idx: Optional[int] = None
    cpu_only: bool = False
    skip_header: bool = True
    delimiter: str = ' '


class WordEmbedding(EmbeddingBase):
    """
    A word embedding wrapper module around `torch.nn.Embedding` with options to
    initialize the word embedding weights and add MLP layers acting on each word.

    Note: Embedding weights for UNK token are always initialized to zeros.

    Args:
        num_embeddings (int): Total number of words/tokens (vocabulary size).
        embedding_dim (int): Size of embedding vector.
        embeddings_weight (torch.Tensor): Pretrained weights to initialize the
            embedding table with.
        init_range (List[int]): Range of uniform distribution to initialize the
            weights with if `embeddings_weight` is None.
        unk_token_idx (int): Index of UNK token in the word vocabulary.
        mlp_layer_dims (List[int]): List of layer dimensions (if any) to add
            on top of the embedding lookup.

    """
    Config = WordFeatConfig

    @classmethod
    def from_config(cls, config: WordFeatConfig, metadata: Optional[FieldMeta]=None, tensorizer: Optional[Tensorizer]=None, init_from_saved_state: Optional[bool]=False):
        """Factory method to construct an instance of WordEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (WordFeatConfig): Configuration object specifying all the
            parameters of WordEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of WordEmbedding.

        """
        if tensorizer is not None:
            if config.vocab_from_pretrained_embeddings:
                raise ValueError('In new data design, to add tokens from a pretrained embeddings file to the vocab, specify `vocab_file` in the token tensorizer.')
            embeddings_weight = None
            if config.pretrained_embeddings_path and not init_from_saved_state:
                pretrained_embedding = PretrainedEmbedding(config.pretrained_embeddings_path, lowercase_tokens=config.lowercase_tokens, skip_header=config.skip_header, delimiter=config.delimiter)
                embeddings_weight = pretrained_embedding.initialize_embeddings_weights(tensorizer.vocab.idx, tensorizer.vocab.unk_token, config.embed_dim, config.embedding_init_strategy)
            num_embeddings = len(tensorizer.vocab)
            unk_token_idx = tensorizer.vocab.get_unk_index()
            vocab = tensorizer.vocab
            vocab_pad_idx = vocab.get_pad_index(value=-1)
            if vocab_pad_idx == -1:
                vocab_pad_idx = None
        else:
            num_embeddings = metadata.vocab_size
            embeddings_weight = metadata.pretrained_embeds_weight
            unk_token_idx = metadata.unk_token_idx
            vocab = metadata.vocab
            vocab_pad_idx = None
        return cls(num_embeddings=num_embeddings, embedding_dim=config.embed_dim, embeddings_weight=embeddings_weight, init_range=config.embedding_init_range, init_std=config.embeddding_init_std, unk_token_idx=unk_token_idx, mlp_layer_dims=config.mlp_layer_dims, padding_idx=config.padding_idx or vocab_pad_idx, vocab=vocab)

    def __init__(self, num_embeddings: int, embedding_dim: int=300, embeddings_weight: Optional[torch.Tensor]=None, init_range: Optional[List[int]]=None, init_std: Optional[float]=None, unk_token_idx: int=0, mlp_layer_dims: List[int]=(), padding_idx: Optional[int]=None, vocab: Optional[List[str]]=None) ->None:
        output_embedding_dim = mlp_layer_dims[-1] if mlp_layer_dims else embedding_dim
        EmbeddingBase.__init__(self, embedding_dim=output_embedding_dim)
        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim, _weight=embeddings_weight, padding_idx=padding_idx)
        if embeddings_weight is None:
            if init_range:
                self.word_embedding.weight.data.uniform_(init_range[0], init_range[1])
            if init_std:
                self.word_embedding.weight.data.normal_(mean=0.0, std=init_std)
        self.word_embedding.weight.data[unk_token_idx].fill_(0.0)
        if mlp_layer_dims is None:
            mlp_layer_dims = []
        self.mlp = nn.Sequential(*(nn.Sequential(nn.Linear(m, n), nn.ReLU()) for m, n in zip([embedding_dim] + list(mlp_layer_dims), mlp_layer_dims)))
        self.vocab = vocab
        self.padding_idx = padding_idx
        log_class_usage(__class__)

    def __getattr__(self, name):
        if name == 'weight':
            return self.word_embedding.weight
        return super().__getattr__(name)

    def forward(self, input):
        return self.mlp(self.word_embedding(input))

    def freeze(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False

    def visualize(self, summary_writer: SummaryWriter):
        if self.vocab:
            summary_writer.add_embedding(self.word_embedding.weight, metadata=self.vocab)


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

    def __init__(self, num_layers, bidirectional, embed_dim, hidden_dim, dropout):
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
            self.layers.append(BiLSTM.LSTM(embed_dim if layer == 0 else hidden_dim, hidden_dim // 2 if is_layer_bidirectional else hidden_dim, num_layers=1, dropout=dropout, bidirectional=is_layer_bidirectional))
        log_class_usage(__class__)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor, enforce_sorted: bool=True):
        bsz = embeddings.size()[1]
        packed_input = pack_padded_sequence(embeddings, lengths, enforce_sorted=enforce_sorted)
        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
                c0 = embeddings.new_full((2, bsz, self.hidden_dim // 2), 0)
            else:
                h0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)
                c0 = embeddings.new_full((1, bsz, self.hidden_dim), 0)
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0))
            if self.bidirectional and i == 0:
                h_last = torch.cat((h_last[(0), :, :], h_last[(1), :, :]), dim=1)
                c_last = torch.cat((c_last[(0), :, :], c_last[(1), :, :]), dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)
            final_hiddens.append(h_last)
            final_cells.append(c_last)
            packed_input = current_output
        final_hidden_size_list: List[int] = final_hiddens[0].size()
        final_hidden_size: Tuple[int, int] = (final_hidden_size_list[0], final_hidden_size_list[1])
        final_hiddens = torch.cat(final_hiddens, dim=0).view(self.num_layers, *final_hidden_size)
        final_cell_size_list: List[int] = final_cells[0].size()
        final_cell_size: Tuple[int, int] = (final_cell_size_list[0], final_cell_size_list[1])
        final_cells = torch.cat(final_cells, dim=0).view(self.num_layers, *final_cell_size)
        unpacked_output, _ = pad_packed_sequence(packed_input)
        return unpacked_output, final_hiddens, final_cells


def _assert_tensorizer_type(t):
    if t is not type(None) and not issubclass(t, Tensorizer.Config):
        raise TypeError(f'ModelInput configuration should only include tensorizers: {t}')


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


class BaseModel(nn.Module, Component):
    """
    Base model class which inherits from nn.Module. Also has a stage flag to
    indicate it's in `train`, `eval`, or `test` stage.
    This is because the built-in train/eval flag in PyTorch can't distinguish eval
    and test, which is required to support some use cases.
    """
    __EXPANSIBLE__ = True
    __COMPONENT_TYPE__ = ComponentType.MODEL
    SUPPORT_FP16_OPTIMIZER = False


    class Config(Component.Config):


        class ModelInput(ModelInputBase):
            pass
        inputs: ModelInput = ModelInput()

    def __init__(self, stage: Stage=Stage.TRAIN) ->None:
        nn.Module.__init__(self)
        self.stage = stage
        self.module_list: List[nn.Module] = []
        self.find_unused_parameters = True
        log_class_usage(__class__)

    def train(self, mode=True):
        """Override to explicitly maintain the stage (train, eval, test)."""
        super().train(mode)
        self.stage = Stage.TRAIN

    def eval(self, stage=Stage.TEST):
        """Override to explicitly maintain the stage (train, eval, test)."""
        super().eval()
        self.stage = stage

    def contextualize(self, context):
        """Add additional context into model. `context` can be anything that
        helps maintaining/updating state. For example, it is used by
        :class:`~DisjointMultitaskModel` for changing the task that should be
        trained with a given iterator.
        """
        self.context = context

    def get_loss(self, logit, target, context):
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, target=None, context=None, *args):
        return self.output_layer.get_pred(logit, target, context)

    def save_modules(self, base_path: str='', suffix: str=''):
        """Save each sub-module in separate files for reusing later."""

        def save(module):
            save_path = getattr(module, 'save_path', None)
            if save_path:
                path = os.path.join(base_path, module.save_path + suffix)
                None
                with PathManager.open(path, 'wb') as save_file:
                    if isinstance(module, torch.jit.ScriptModule):
                        module.save(save_file)
                    else:
                        torch.save(module.state_dict(), save_file)
        self.apply(save)

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_'):
                module.prepare_for_onnx_export_(**kwargs)
        self.apply(apply_prepare_for_onnx_export_)

    def quantize(self):
        """Quantize the model during export."""
        torch.quantization.quantize_dynamic(self, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

    def get_param_groups_for_optimizer(self) ->List[Dict[str, List[nn.Parameter]]]:
        """
        Returns a list of parameter groups of the format {"params": param_list}.
        The parameter groups loosely correspond to layers and are ordered from low
        to high. Currently, only the embedding layer can provide multiple param groups,
        and other layers are put into one param group. The output of this method
        is passed to the optimizer so that schedulers can change learning rates
        by layer.
        """
        non_emb_params = dict(self.named_parameters())
        model_params = [non_emb_params]
        embedding = getattr(self, 'embedding', None)
        if embedding is not None:
            emb_params_by_layer = self.embedding.get_param_groups_for_optimizer()
            for emb_params in emb_params_by_layer:
                for name in emb_params:
                    del non_emb_params['embedding.%s' % name]
            model_params = emb_params_by_layer + model_params
            print_str = 'Model has %d param groups (%d from embedding module) for optimizer'
            None
        model_params = [{'params': params.values()} for params in model_params]
        return model_params

    @classmethod
    def train_batch(cls, model, batch, state=None):
        model_inputs = model.arrange_model_inputs(batch)
        model_context = model.arrange_model_context(batch)
        targets = model.arrange_targets(batch)
        model_outputs = model(*model_inputs)
        if state:
            if model_context is None:
                model_context = {'stage': state.stage}
            else:
                model_context['stage'] = state.stage
        loss = maybe_float(model.get_loss(model_outputs, targets, model_context))
        predictions, scores = model.get_pred(model_outputs, context=model_context)
        metric_data = predictions, targets, scores, loss, model_inputs
        return loss, metric_data

    def arrange_model_inputs(self, tensor_dict):
        pass

    def arrange_targets(self, tensor_dict):
        pass

    def arrange_model_context(self, tensor_dict):
        return None

    def onnx_trace_input(self, tensor_dict):
        return self.arrange_model_inputs(tensor_dict)

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        pass

    def arrange_caffe2_model_inputs(self, tensor_dict):
        """
        Generate inputs for exported caffe2 model, default behavior is flatten the
        input tuples
        """
        model_inputs = self.arrange_model_inputs(tensor_dict)
        flat_model_inputs = []
        for model_input in model_inputs:
            if isinstance(model_input, tuple):
                flat_model_inputs.extend(model_input)
            else:
                flat_model_inputs.append(model_input)
        return flat_model_inputs

    def get_num_examples_from_batch(self, batch):
        pass


class CommonMetadata:
    features: Dict[str, FieldMeta]
    target: FieldMeta
    dataset_sizes: Dict[str, int]


class DecoderBase(Module):
    """Base class for all decoder modules.

    Args:
        config (ConfigBase): Configuration object.

    Attributes:
        in_dim (int): Dimension of input Tensor passed to the decoder.
        out_dim (int): Dimension of output Tensor produced by the decoder.

    """

    def __init__(self, config: ConfigBase):
        super().__init__(config)
        self.input_dim = 0
        self.target_dim = 0
        self.num_decoder_modules = 0
        log_class_usage(__class__)

    def forward(self, *input):
        raise NotImplementedError()

    def get_decoder(self):
        """Returns the decoder module.
        """
        raise NotImplementedError()

    def get_in_dim(self) ->int:
        """Returns the dimension of the input Tensor that the decoder accepts.
        """
        return self.in_dim

    def get_out_dim(self) ->int:
        """Returns the dimension of the input Tensor that the decoder emits.
        """
        return self.out_dim


class CNNParams(ConfigBase):
    kernel_num: int = 100
    kernel_sizes: List[int] = [3, 4]
    weight_norm: bool = False
    dilated: bool = False
    causal: bool = False


class CharFeatConfig(ModuleConfig):
    embed_dim: int = 100
    sparse: bool = False
    cnn: CNNParams = CNNParams()
    highway_layers: int = 0
    projection_dim: Optional[int] = None
    export_input_names: List[str] = ['char_vals']
    vocab_from_train_data: bool = True
    max_word_length: int = 20
    min_freq: int = 1


class ContextualTokenEmbeddingConfig(ConfigBase):
    embed_dim: int = 0
    model_paths: Optional[Dict[str, str]] = None
    export_input_names: List[str] = ['contextual_token_embedding']


class PoolingType(Enum):
    MEAN = 'mean'
    MAX = 'max'
    LOGSUMEXP = 'logsumexp'
    NONE = 'none'


class DictFeatConfig(ModuleConfig):
    embed_dim: int = 100
    sparse: bool = False
    pooling: PoolingType = PoolingType.MEAN
    export_input_names: List[str] = ['dict_vals', 'dict_weights', 'dict_lens']
    vocab_from_train_data: bool = True
    mobile: bool = False


class FloatVectorConfig(ConfigBase):
    dim: int = 0
    export_input_names: List[str] = ['float_vec_vals']
    dim_error_check: bool = False


class FeatureConfig(ModuleConfig):
    word_feat: WordFeatConfig = WordFeatConfig()
    seq_word_feat: Optional[WordFeatConfig] = None
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    dense_feat: Optional[FloatVectorConfig] = None
    contextual_token_embedding: Optional[ContextualTokenEmbeddingConfig] = None


class ModelInput:
    TEXT = 'word_feat'
    DICT = 'dict_feat'
    CHAR = 'char_feat'
    CONTEXTUAL_TOKEN_EMBEDDING = 'contextual_token_embedding'
    SEQ = 'seq_word_feat'
    DENSE = 'dense_feat'


class RepresentationBase(Module):

    def __init__(self, config):
        super().__init__(config)
        self.representation_dim = None

    def forward(self, *inputs):
        raise NotImplementedError()

    def get_representation_dim(self):
        return self.representation_dim

    def _preprocess_inputs(self, inputs):
        raise NotImplementedError()


def create_component(component_type: ComponentType, config: Any, *args, **kwargs):
    config_cls = type(config)
    cls = Registry.get(component_type, config_cls)
    try:
        return cls.from_config(config, *args, **kwargs)
    except TypeError as e:
        raise Exception(f"Can't create component {cls}: {str(e)}")


def _create_module_from_registry(module_config, *args, **kwargs):
    return create_component(ComponentType.MODULE, module_config, *args, **kwargs)


def create_module(module_config, *args, create_fn=_create_module_from_registry, **kwargs):
    """Create module object given the module's config object. It depends on the
    global shared module registry. Hence, your module must be available for the
    registry. This entails that your module must be imported somewhere in the
    code path during module creation (ideally in your model class) for the module
    to be visible for registry.

    Args:
        module_config (type): Module config object.
        create_fn (type): The function to use for creating the module. Use this
            parameter if your module creation requires custom code and pass your
            function here. Defaults to `_create_module_from_registry()`.

    Returns:
        type: Description of returned object.

    """
    shared_module_key = getattr(module_config, 'shared_module_key', None)
    typed_shared_module_key = shared_module_key, type(module_config)
    load_path = getattr(module_config, 'load_path', None)
    is_torchscript_load_path = load_path and zipfile.is_zipfile(PathManager.get_local_path(load_path))
    module = SHARED_MODULE_REGISTRY.get(typed_shared_module_key)
    if not module:
        if is_torchscript_load_path:
            with PathManager.open(load_path, 'rb') as load_file:
                module = torch.jit.load(load_file)
        else:
            module = create_fn(module_config, *args, **kwargs)
    name = type(module).__name__
    if load_path and not is_torchscript_load_path:
        None
        with PathManager.open(load_path, 'rb') as load_file:
            module.load_state_dict(torch.load(load_file, map_location='cpu'))
    if getattr(module_config, 'freeze', False):
        None
        module.freeze()
    if shared_module_key:
        SHARED_MODULE_REGISTRY[typed_shared_module_key] = module
    module.save_path = getattr(module_config, 'save_path', None)
    return module


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
                example_response[self.classes[i]] = float(example_scores[i].item())
            results.append(example_response)
        return results


class RegressionScores(torch.jit.ScriptModule):

    def __init__(self, squash_to_unit_range: bool):
        super().__init__()
        self.squash_to_unit_range = torch.jit.Attribute(squash_to_unit_range, bool)

    @torch.jit.script_method
    def forward(self, logits: torch.Tensor) ->List[float]:
        prediction = logits.squeeze(dim=1)
        if self.squash_to_unit_range:
            prediction = torch.sigmoid(prediction)
        scores: List[float] = prediction.tolist()
        return scores


class IntentSlotScores(nn.Module):

    def __init__(self, doc_scores: jit.ScriptModule, word_scores: jit.ScriptModule):
        super().__init__()
        self.doc_scores = doc_scores
        self.word_scores = word_scores
        log_class_usage(__class__)

    def forward(self, logits: Tuple[torch.Tensor, torch.Tensor], context: Dict[str, torch.Tensor]) ->Tuple[List[Dict[str, float]], List[List[Dict[str, float]]]]:
        d_logits, w_logits = logits
        if 'token_indices' in context:
            w_logits = torch.gather(w_logits, 1, context['token_indices'].unsqueeze(2).expand(-1, -1, w_logits.size(-1)))
        d_results = self.doc_scores(d_logits)
        w_results = self.word_scores(w_logits, context)
        return d_results, w_results


class MultiLabelClassificationScores(nn.Module):

    def __init__(self, scores: List[jit.ScriptModule]):
        super().__init__()
        self.scores = nn.ModuleList(scores)
        log_class_usage(__class__)

    def forward(self, logits: List[torch.Tensor]) ->List[List[Dict[str, float]]]:
        results: List[List[Dict[str, float]]] = []
        for idx, sc in enumerate(self.scores):
            logit = logits[idx]
            flattened_logit = logit.view(-1, logit.size()[-1])
            results.append(sc(flattened_logit))
        return results


class CrossEntropyLoss(Loss):


    class Config(ConfigBase):
        pass

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, logits, targets, reduce=True):
        return F.nll_loss(F.log_softmax(logits, 1, dtype=torch.float32), targets, weight=self.weight, ignore_index=self.ignore_index, reduction='mean' if reduce else 'none')


class KLDivergenceCELoss(Loss):


    class Config(ConfigBase):
        temperature: float = 1.0
        hard_weight: float = 0.0

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        assert ignore_index < 0
        assert 0.0 <= config.hard_weight < 1.0
        self.weight = weight
        self.t = config.temperature
        self.hard_weight = config.hard_weight

    def __call__(self, logits, targets, reduce=True, combine_loss=True):
        """
        Computes Kullback-Leibler divergence loss for multiclass classification
        probability distribution computed by CrossEntropyLoss loss.
        For, KL-divergence, batchmean is the right way to reduce, not just mean.
        """
        hard_targets, _, soft_targets_logits = targets
        soft_targets = F.softmax(soft_targets_logits.float() / self.t, dim=1)
        soft_targets = soft_targets.clamp(1e-10, 1 - 1e-10)
        log_probs = F.log_softmax(logits / self.t, 1)
        if self.weight is not None:
            soft_loss = F.kl_div(log_probs, soft_targets, reduction='none') * self.weight
            soft_loss = torch.sum(soft_loss, dim=1).mean() if reduce else torch.sum(soft_loss, dim=1)
        else:
            soft_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean' if reduce else 'none')
        soft_loss *= self.t ** 2
        hard_loss = 0.0
        if self.hard_weight > 0.0:
            hard_loss = F.cross_entropy(logits, hard_targets, reduction='mean' if reduce else 'none', weight=self.weight)
        return (1.0 - self.hard_weight) * soft_loss + self.hard_weight * hard_loss if combine_loss else (soft_loss, hard_loss)


def create_loss(loss_config, *args, **kwargs):
    return create_component(ComponentType.LOSS, loss_config, *args, **kwargs)


@jit.script
def _get_prediction_from_scores(scores: torch.Tensor, classes: List[str]) ->List[List[Dict[str, float]]]:
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

    def forward(self, logits: torch.Tensor, context: Optional[Dict[str, torch.Tensor]]=None) ->List[List[Dict[str, float]]]:
        scores: torch.Tensor = F.log_softmax(logits, 2)
        return _get_prediction_from_scores(scores, self.classes)


@jit.script
def _rearrange_output(logit, pred):
    """
    Rearrange the word logits so that the decoded word has the highest valued
    logits by swapping the indices predicted with those with maximum logits.
    """
    max_logits, max_logit_indices = torch.max(logit, 2, keepdim=True)
    pred_indices = pred.unsqueeze(2)
    pred_logits = torch.gather(logit, 2, pred_indices)
    logit_rearranged = logit.scatter(2, pred_indices, max_logits)
    logit_rearranged.scatter_(2, max_logit_indices, pred_logits)
    return logit_rearranged


class CRFWordTaggingScores(WordTaggingScores):

    def __init__(self, classes: List[str], crf):
        super().__init__(classes)
        self.crf = crf
        self.crf.eval()
        log_class_usage(__class__)

    def forward(self, logits: torch.Tensor, context: Dict[str, torch.Tensor]) ->List[List[Dict[str, float]]]:
        assert 'seq_lens' in context
        pred = self.crf.decode(logits, context['seq_lens'])
        logits_rearranged = _rearrange_output(logits, pred)
        scores: torch.Tensor = F.log_softmax(logits_rearranged, 2)
        return _get_prediction_from_scores(scores, self.classes)


class BinaryCrossEntropyLoss(Loss):


    class Config(ConfigBase):
        reweight_negative: bool = True
        reduce: bool = True

    def __call__(self, logits, targets, reduce=True):
        """
        Computes 1-vs-all binary cross entropy loss for multiclass
        classification.
        """
        targets = FloatTensor(targets.size(0), logits.size(1)).zero_().scatter_(1, targets.unsqueeze(1).data, 1) if len(logits.size()) > 1 else targets.float()
        """
        `F.binary_cross_entropy` or `torch.nn.BCELoss.` requires the
        output of the previous function be already a FloatTensor.
        """
        loss = F.binary_cross_entropy_with_logits(precision.maybe_float(logits), targets, reduction='none')
        if self.config.reweight_negative:
            weights = targets + (1.0 - targets) / max(1, targets.size(1) - 1.0)
            loss = loss * weights
        return loss.sum(-1).mean() if reduce else loss.sum(-1)


class KLDivergenceBCELoss(Loss):


    class Config(ConfigBase):
        temperature: float = 1.0
        hard_weight: float = 0.0

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        assert 0.0 <= config.hard_weight < 1.0
        self.ignore_index = ignore_index
        self.weight = weight
        self.t = config.temperature
        self.hard_weight = config.hard_weight

    def __call__(self, logits, targets, reduce=True):
        """
        Computes Kullback-Leibler divergence loss for multiclass classification
        probability distribution computed by BinaryCrossEntropyLoss loss
        """
        hard_targets, _, soft_targets_logits = targets
        soft_targets = F.sigmoid(FloatTensor(soft_targets_logits) / self.t).clamp(1e-20, 1 - 1e-20)
        probs = F.sigmoid(logits / self.t).clamp(1e-20, 1 - 1e-20)
        probs_neg = probs.neg().add(1).clamp(1e-20, 1 - 1e-20)
        soft_targets_neg = soft_targets.neg().add(1).clamp(1e-20, 1 - 1e-20)
        if self.weight is not None:
            soft_loss = F.kl_div(probs.log(), soft_targets, reduction='none') * self.weight + F.kl_div(probs_neg.log(), soft_targets_neg, reduction='none') * self.weight
            if reduce:
                soft_loss = soft_loss.mean()
        else:
            soft_loss = F.kl_div(probs.log(), soft_targets, reduction='mean' if reduce else 'none') + F.kl_div(probs_neg.log(), soft_targets_neg, reduction='mean' if reduce else 'none')
        soft_loss *= self.t ** 2
        hard_loss = 0.0
        if self.hard_weight > 0.0:
            one_hot_targets = FloatTensor(hard_targets.size(0), logits.size(1)).zero_().scatter_(1, hard_targets.unsqueeze(1).data, 1)
            hard_loss = F.binary_cross_entropy_with_logits(logits, one_hot_targets, reduction='mean' if reduce else 'none', weight=self.weight)
        return (1.0 - self.hard_weight) * soft_loss + self.hard_weight * hard_loss


class LabelSmoothedCrossEntropyLoss(Loss):


    class Config(ConfigBase):
        beta: float = 0.1
        from_logits: bool = True
        use_entropy: bool = False

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        if weight is not None:
            assert torch.sum(torch.abs(weight - 1.0)) < 1e-07
        self.ignore_index = ignore_index
        self.weight = weight
        self.beta = config.beta
        self.from_logits = config.from_logits
        self.use_entropy = config.use_entropy

    def __call__(self, logits, targets, reduce=True):
        """
        If use_entropy is False, returns the cross-entropy loss alongwith the KL divergence of the
        discrete uniform distribution with the logits. Refer to section 3.2
        If use_entopy is True, uses the entropy of the output distribution as
        the smoothing loss (i.e., higher entropy, better). Refer to section 3
        https://arxiv.org/pdf/1701.06548.pdf
        """
        if self.use_entropy:
            probs = F.softmax(logits, dim=1)
            log_probs = torch.log(probs)
            label_smoothing_loss = torch.sum(log_probs * probs, dim=1)
        else:
            log_probs = F.log_softmax(logits, dim=1) if self.from_logits else logits
            label_smoothing_loss = -1 * log_probs.mean(dim=1)
        if reduce:
            non_ignored = targets != self.ignore_index
            if non_ignored.any():
                label_smoothing_loss = torch.mean(label_smoothing_loss[non_ignored])
            else:
                label_smoothing_loss = torch.tensor(0.0, device=logits.device)
        cross_entropy_loss = F.nll_loss(log_probs, targets, ignore_index=self.ignore_index, reduction='mean' if reduce else 'none', weight=self.weight)
        return (1.0 - self.beta) * cross_entropy_loss + self.beta * label_smoothing_loss


class Padding:
    WORD_LABEL_PAD = 'PAD_LABEL'
    WORD_LABEL_PAD_IDX = 0
    DEFAULT_LABEL_PAD_IDX = -1


def device():
    return 'cuda:{}'.format(torch.cuda.current_device()) if CUDA_ENABLED else 'cpu'


def tensor(data, dtype):
    return torch.tensor(data, dtype=dtype, device=device())


def get_label_weights(vocab_dict: Dict[str, int], label_weights: Dict[str, float]):
    pruned_label_weights = {vocab_dict[k]: v for k, v in label_weights.items() if k in vocab_dict}
    if len(pruned_label_weights) != len(label_weights):
        filtered_labels = [k for k in label_weights if k not in vocab_dict]
        None
    if len(pruned_label_weights) == 0:
        return None
    weights_tensor = [1] * len(vocab_dict)
    for k, v in pruned_label_weights.items():
        weights_tensor[k] = v
    return tensor(weights_tensor, dtype=torch.float)


class ConfigParseError(Exception):
    pass


class MissingValueError(ConfigParseError):
    pass


class MultiLabelSoftMarginLoss(Loss):


    class Config(ConfigBase):
        pass

    def __call__(self, m_out, targets, reduce=True):
        """
        Computes multi-label classification loss
        see details in torch.nn.MultiLabelSoftMarginLoss
        """
        num_classes = m_out.size()[1]
        target_labels = targets[0]
        tmp_target_labels = target_labels + 1
        n_hot_targets = FloatTensor(target_labels.size(0), num_classes + 1).zero_().scatter_(1, tmp_target_labels, 1)[:, 1:]
        """
        `F.multilabel_soft_margin_loss` or `torch.nn.MultiLabelSoftMarginLoss.`
        requires the
        output of the previous function be already a FloatTensor.
        """
        loss = F.multilabel_soft_margin_loss(precision.maybe_float(m_out), n_hot_targets, reduction='mean')
        return loss


class Activation(Enum):
    RELU = 'relu'
    LEAKYRELU = 'leakyrelu'
    TANH = 'tanh'
    GELU = 'gelu'
    GLU = 'glu'


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
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x * x * x))))


def get_activation(name, dim=1):
    if name == Activation.RELU:
        return nn.ReLU()
    elif name == Activation.LEAKYRELU:
        return nn.LeakyReLU()
    elif name == Activation.TANH:
        return nn.Tanh()
    elif name == Activation.GELU:
        return GeLU()
    elif name == Activation.GLU:
        return nn.GLU(dim=dim)
    else:
        raise RuntimeError(f'{name} is not supported')


class CosineEmbeddingLoss(Loss):


    class Config(ConfigBase):
        margin: float = 0.0

    def __init__(self, config, *args, **kwargs):
        self.margin = config.margin

    def __call__(self, embeddings, targets, reduce=True):
        if len(embeddings) != 2:
            raise ValueError(f'Number of embeddings must be 2. Found {len(embeddings)} embeddings.')
        return F.cosine_embedding_loss(embeddings[0], embeddings[1], targets, margin=self.margin, reduction='mean' if reduce else 'none')


class MAELoss(Loss):
    """
    Mean absolute error or L1 loss, for regression tasks.
    """


    class Config(ConfigBase):
        pass

    def __call__(self, predictions, targets, reduce=True):
        return F.l1_loss(predictions, targets, reduction='mean' if reduce else 'none')


class MSELoss(Loss):
    """
    Mean squared error or L2 loss, for regression tasks.
    """


    class Config(ConfigBase):
        pass

    def __call__(self, predictions, targets, reduce=True):
        return F.mse_loss(predictions, targets, reduction='mean' if reduce else 'none')


class NLLLoss(Loss):

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, log_probs, targets, reduce=True):
        return F.nll_loss(log_probs, targets, ignore_index=self.ignore_index, reduction='mean' if reduce else 'none', weight=self.weight)


@unique
class OutputScore(IntEnum):
    raw_cosine = 1
    norm_cosine = 2
    sigmoid_cosine = 3


def get_norm_cosine_scores(cosine_sim_scores):
    pos_scores = (cosine_sim_scores + 1.0) / 2.0
    neg_scores = 1.0 - pos_scores
    return pos_scores, neg_scores


def get_sigmoid_scores(cosine_sim_scores):
    pos_scores = torch.sigmoid(cosine_sim_scores)
    neg_scores = 1.0 - pos_scores
    return pos_scores, neg_scores


class LastTimestepPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) ->torch.Tensor:
        if torch._C._get_tracing_state():
            assert inputs.shape[0] == 1
            return inputs[:, (-1), :]
        bsz, _, dim = inputs.shape
        idx = seq_lengths.unsqueeze(1).expand(bsz, dim).unsqueeze(1)
        return inputs.gather(1, idx - 1).squeeze(1)


class MaxPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None) ->torch.Tensor:
        return torch.max(inputs, 1)[0]


class MeanPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) ->torch.Tensor:
        return torch.sum(inputs, 1) / seq_lengths.unsqueeze(1).float()


class NoPool(Module):

    def __init__(self, config: Module.Config, n_input: int) ->None:
        super().__init__(config)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None) ->torch.Tensor:
        return inputs


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

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None) ->torch.Tensor:
        size = torch.onnx.operators.shape_as_tensor(inputs)
        flat_2d_shape = torch.cat((torch.LongTensor([-1]), size[2].view(1)))
        compressed_emb = torch.onnx.operators.reshape_from_tensor_shape(inputs, flat_2d_shape)
        hbar = self.tanh(self.ws1(self.dropout(compressed_emb)))
        alphas = self.ws2(hbar)
        alphas = torch.onnx.operators.reshape_from_tensor_shape(alphas, size[:2])
        alphas = self.softmax(alphas)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)


BOL = SpecialToken('__BEGIN_OF_LIST__')


EOL = SpecialToken('__END_OF_LIST__')


PAD_INDEX = 1


UNK_INDEX = 0


class VocabBuilder:
    """Helper class for aggregating and building `Vocabulary` objects."""

    def __init__(self, delimiter=' '):
        self._counter = Counter()
        self.use_unk = True
        self.unk_index = UNK_INDEX
        self.use_pad = True
        self.pad_index = PAD_INDEX
        self.use_bos = False
        self.bos_index = 2
        self.use_eos = False
        self.eos_index = 3
        self.use_bol = False
        self.bol_index = 4
        self.use_eol = False
        self.eol_index = 5
        self.use_mask = False
        self.mask_index = 6
        self.unk_token = UNK
        self.pad_token = PAD
        self.bos_token = BOS
        self.eos_token = EOS
        self.mask_token = MASK
        self.delimiter = delimiter

    def add_all(self, values) ->None:
        """Count a value or nested container of values in the vocabulary."""
        if should_iter(values):
            for value in values:
                self.add_all(value)
        elif values not in [None, '']:
            self.add(values)

    def add(self, value) ->None:
        """Count a single value in the vocabulary."""
        self._counter[value] += 1

    def add_from_file(self, file_pointer, skip_header_line, lowercase_tokens, size):
        vocab_from_file = set()
        if skip_header_line:
            next(file_pointer)
        for i, line in enumerate(file_pointer):
            if size and len(vocab_from_file) == size:
                None
                break
            token = line.split(self.delimiter)[0].strip()
            if lowercase_tokens:
                token = token.lower()
            vocab_from_file.add(token)
        self.add_all(sorted(vocab_from_file))

    def make_vocab(self) ->Vocabulary:
        """Build a Vocabulary object from the values seen by the builder."""
        tokens_to_insert: List[Tuple[int, object]] = []
        if self.use_unk:
            tokens_to_insert.append((self.unk_index, self.unk_token))
            del self._counter[self.unk_token]
        if self.use_pad:
            tokens_to_insert.append((self.pad_index, self.pad_token))
            del self._counter[self.pad_token]
        if self.use_bos:
            tokens_to_insert.append((self.bos_index, self.bos_token))
            del self._counter[self.bos_token]
        if self.use_eos:
            tokens_to_insert.append((self.eos_index, self.eos_token))
            del self._counter[self.eos_token]
        if self.use_bol:
            tokens_to_insert.append((self.bol_index, BOL))
            del self._counter[BOL]
        if self.use_eol:
            tokens_to_insert.append((self.eol_index, EOL))
            del self._counter[EOL]
        if self.use_mask:
            tokens_to_insert.append((self.mask_index, MASK))
            del self._counter[MASK]
        vocab_list = list(self._counter)
        for index, token in sorted(tokens_to_insert):
            vocab_list.insert(index, token)
        return Vocabulary(vocab_list, counts=self._counter, unk_token=self.unk_token, pad_token=self.pad_token, bos_token=self.bos_token, eos_token=self.eos_token, mask_token=self.mask_token)

    def truncate_to_vocab_size(self, vocab_size=-1, min_counts=-1) ->None:
        if len(self._counter) > vocab_size > 0:
            self._counter = Counter(dict(self._counter.most_common(vocab_size)))
        if len(self._counter) > 0 and min_counts > 0:
            self._counter = Counter({k: v for k, v in self._counter.items() if v >= min_counts})


def _infer_pad_shape(nested_lists):
    """Return the minimal tensor shape which could contain the input data."""
    yield len(nested_lists)
    while nested_lists and all(should_iter(i) for i in nested_lists):
        yield precision.pad_length(max(len(nested) for nested in nested_lists))
        nested_lists = list(itertools.chain.from_iterable(nested_lists))


def _make_nested_padding(pad_shape, pad_token):
    """Create nested lists of pad_token of shape pad_shape."""
    result = [pad_token]
    for dimension in reversed(pad_shape):
        result = [result * dimension]
    return result[0]


def pad(nested_lists, pad_token, pad_shape=None):
    """Pad the input lists with the pad token. If pad_shape is provided, pad to that
    shape, otherwise infer the input shape and pad out to a square tensor shape."""
    if pad_shape is None:
        pad_shape = list(_infer_pad_shape(nested_lists))
    if not pad_shape:
        return nested_lists
    dimension, *rest = pad_shape
    result = [pad(nested, pad_token, rest) for nested in nested_lists]
    result += [_make_nested_padding(rest, pad_token)] * (dimension - len(result))
    return result


def pad_and_tensorize(batch, pad_token=0, pad_shape=None, dtype=torch.long):
    batch = list(batch)
    if not batch:
        return torch.Tensor()
    return cuda.tensor(pad(batch, pad_token=pad_token, pad_shape=pad_shape), dtype=dtype)


class LabelTensorizer(Tensorizer):
    """Numberize labels. Label can be used as either input or target """
    __EXPANSIBLE__ = True


    class Config(Tensorizer.Config):
        column: str = 'label'
        allow_unknown: bool = False
        pad_in_vocab: bool = False
        label_vocab: Optional[List[str]] = None
        is_input: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.allow_unknown, config.pad_in_vocab, config.label_vocab, config.is_input)

    def __init__(self, label_column: str='label', allow_unknown: bool=False, pad_in_vocab: bool=False, label_vocab: Optional[List[str]]=None, is_input: bool=Config.is_input):
        self.label_column = label_column
        self.pad_in_vocab = pad_in_vocab
        self.vocab_builder = VocabBuilder()
        self.vocab_builder.use_pad = pad_in_vocab
        self.vocab_builder.use_unk = allow_unknown
        self.vocab = None
        self.pad_idx = -1
        if label_vocab:
            self.vocab_builder.add_all(label_vocab)
            self.vocab, self.pad_idx = self._create_vocab()
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.label_column, str)]

    def initialize(self, from_scratch=True):
        """
        Look through the dataset for all labels and create a vocab map for them.
        """
        if self.vocab and from_scratch:
            return
        try:
            while True:
                row = yield
                labels = row[self.label_column]
                self.vocab_builder.add_all(labels)
        except GeneratorExit:
            self.vocab, self.pad_idx = self._create_vocab()

    def _create_vocab(self):
        vocab = self.vocab_builder.make_vocab()
        pad_idx = vocab.get_pad_index() if self.pad_in_vocab else Padding.DEFAULT_LABEL_PAD_IDX
        return vocab, pad_idx

    def numberize(self, row):
        """Numberize labels."""
        return self.vocab.lookup_all(row[self.label_column])

    def tensorize(self, batch):
        return pad_and_tensorize(batch, self.pad_idx)


class VocabFileConfig(Component.Config):
    filepath: str = ''
    skip_header_line: bool = False
    lowercase_tokens: bool = False
    size_limit: int = 0


class VocabConfig(Component.Config):
    build_from_data: bool = True
    size_from_data: int = 0
    min_counts: int = 0
    vocab_files: List[VocabFileConfig] = []


def tokenize(text: str=None, pre_tokenized: List[Token]=None, tokenizer: Tokenizer=None, bos_token: Optional[str]=None, eos_token: Optional[str]=None, pad_token: str=PAD, use_eos_token_for_bos: bool=False, max_seq_len: int=2 ** 30):
    tokenized = pre_tokenized or tokenizer.tokenize(text)[:max_seq_len - (bos_token is not None) - (eos_token is not None)]
    if bos_token:
        if use_eos_token_for_bos:
            bos_token = eos_token
        tokenized = [Token(bos_token, -1, -1)] + tokenized
    if eos_token:
        tokenized.append(Token(eos_token, -1, -1))
    if not tokenized:
        tokenized = [Token(pad_token, -1, -1)]
    tokenized_texts, start_idx, end_idx = zip(*((t.value, t.start, t.end) for t in tokenized))
    return tokenized_texts, start_idx, end_idx


def lookup_tokens(text: str=None, pre_tokenized: List[Token]=None, tokenizer: Tokenizer=None, vocab: Vocabulary=None, bos_token: Optional[str]=None, eos_token: Optional[str]=None, pad_token: str=PAD, use_eos_token_for_bos: bool=False, max_seq_len: int=2 ** 30):
    tokenized_texts, start_idx, end_idx = tokenize(text, pre_tokenized, tokenizer, bos_token, eos_token, pad_token, use_eos_token_for_bos, max_seq_len)
    tokens = vocab.lookup_all(tokenized_texts)
    return tokens, start_idx, end_idx


class TokenTensorizer(Tensorizer):
    """Convert text to a list of tokens. Do this based on a tokenizer configuration,
    and build a vocabulary for numberization. Finally, pad the batch to create
    a square tensor of the correct size.
    """


    class Config(Tensorizer.Config):
        column: str = 'text'
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        add_bos_token: bool = False
        add_eos_token: bool = False
        use_eos_token_for_bos: bool = False
        max_seq_len: Optional[int] = None
        vocab: VocabConfig = VocabConfig()
        vocab_file_delimiter: str = ' '

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(text_column=config.column, tokenizer=tokenizer, add_bos_token=config.add_bos_token, add_eos_token=config.add_eos_token, use_eos_token_for_bos=config.use_eos_token_for_bos, max_seq_len=config.max_seq_len, vocab_config=config.vocab, vocab_file_delimiter=config.vocab_file_delimiter, is_input=config.is_input)

    def __init__(self, text_column, tokenizer=None, add_bos_token=Config.add_bos_token, add_eos_token=Config.add_eos_token, use_eos_token_for_bos=Config.use_eos_token_for_bos, max_seq_len=Config.max_seq_len, vocab_config=None, vocab=None, vocab_file_delimiter=' ', is_input=Config.is_input):
        self.text_column = text_column
        self.tokenizer = tokenizer or Tokenizer()
        self.vocab = vocab
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len or 2 ** 30
        self.vocab_builder = None
        self.vocab_config = vocab_config or VocabConfig()
        self.vocab_file_delimiter = vocab_file_delimiter
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.text_column, str)]

    def _tokenize(self, text=None, pre_tokenized=None):
        return tokenize(text=text, pre_tokenized=pre_tokenized, tokenizer=self.tokenizer, bos_token=self.vocab.bos_token if self.add_bos_token else None, eos_token=self.vocab.eos_token if self.add_eos_token else None, pad_token=self.vocab.pad_token, use_eos_token_for_bos=self.use_eos_token_for_bos, max_seq_len=self.max_seq_len)

    def _lookup_tokens(self, text=None, pre_tokenized=None):
        return lookup_tokens(text=text, pre_tokenized=pre_tokenized, tokenizer=self.tokenizer, vocab=self.vocab, bos_token=self.vocab.bos_token if self.add_bos_token else None, eos_token=self.vocab.eos_token if self.add_eos_token else None, pad_token=self.vocab.pad_token, use_eos_token_for_bos=self.use_eos_token_for_bos, max_seq_len=self.max_seq_len)

    def _reverse_lookup(self, token_ids):
        return [self.vocab[id] for id in token_ids]

    def initialize(self, vocab_builder=None, from_scratch=True):
        """Build vocabulary based on training corpus."""
        if self.vocab and from_scratch:
            if self.vocab_config.build_from_data or self.vocab_config.vocab_files:
                None
            return
        if not self.vocab_config.build_from_data and not self.vocab_config.vocab_files:
            raise ValueError(f"To create token tensorizer for '{self.text_column}', either `build_from_data` or `vocab_files` must be set.")
        if not self.vocab_builder:
            self.vocab_builder = vocab_builder or VocabBuilder(delimiter=self.vocab_file_delimiter)
            self.vocab_builder.use_bos = self.add_bos_token
            self.vocab_builder.use_eos = self.add_eos_token
        if not self.vocab_config.build_from_data:
            self._add_vocab_from_files()
            self.vocab = self.vocab_builder.make_vocab()
            return
        try:
            while True:
                row = yield
                raw_text = row[self.text_column]
                tokenized = self.tokenizer.tokenize(raw_text)
                self.vocab_builder.add_all([t.value for t in tokenized])
        except GeneratorExit:
            self.vocab_builder.truncate_to_vocab_size(self.vocab_config.size_from_data, self.vocab_config.min_counts)
            self._add_vocab_from_files()
            self.vocab = self.vocab_builder.make_vocab()

    def _add_vocab_from_files(self):
        for vocab_file in self.vocab_config.vocab_files:
            with PathManager.open(vocab_file.filepath) as f:
                self.vocab_builder.add_from_file(f, vocab_file.skip_header_line, vocab_file.lowercase_tokens, vocab_file.size_limit)

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        tokens, start_idx, end_idx = self._lookup_tokens(row[self.text_column])
        token_ranges = list(zip(start_idx, end_idx))
        return tokens, len(tokens), token_ranges

    def prepare_input(self, row):
        """Tokenize, look up in vocabulary, return tokenized_texts in raw text"""
        tokenized_texts, start_idx, end_idx = self._tokenize(row[self.text_column])
        token_ranges = list(zip(start_idx, end_idx))
        return list(tokenized_texts), len(tokenized_texts), token_ranges

    def tensorize(self, batch):
        tokens, seq_lens, token_ranges = zip(*batch)
        return pad_and_tensorize(tokens, self.vocab.get_pad_index()), pad_and_tensorize(seq_lens), pad_and_tensorize(token_ranges)

    def sort_key(self, row):
        return row[1]


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


GLOVE_840B_300D = '/mnt/vol/pytext/users/kushall/pretrained/glove.840B.300d.txt'


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

    def forward(self, p_seq: torch.Tensor, q: torch.Tensor, p_mask: torch.Tensor):
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
        attn_scores_flattened = F.softmax(attn_scores.view(-1, q.size(1)), dim=-1)
        return attn_scores_flattened.view(-1, p.size(1), q.size(1))


class BERTInitialTokenizer(Tokenizer):
    """
    Basic initial tokenization for BERT.  This is run prior to word piece, does
    white space tokenization in addition to lower-casing and accent removal
    if specified.
    """


    class Config(Tokenizer.Config):
        """Config for this class."""

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = BasicTokenizer(do_lower_case=config.lowercase)
        return cls(basic_tokenizer)

    def __init__(self, basic_tokenizer) ->None:
        self.tokenizer = basic_tokenizer
        log_class_usage(__class__)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        if self.tokenizer.do_lower_case:
            text = self.tokenizer._run_strip_accents(text.lower())
        tokens = self.tokenizer.tokenize(text)
        end = 0
        result = []
        for token in tokens:
            start = text.find(token, end)
            if start == -1:
                start = end
            end = start + len(token)
            result.append(Token(token, start, end))
        return result


def load_vocab(file_path):
    """
    Given a file, prepare the vocab dictionary where each line is the value and
    (line_no - 1) is the key
    """
    vocab = {}
    with PathManager.open(file_path, 'r') as file_contents:
        for idx, word in enumerate(file_contents):
            vocab[str(idx)] = word.strip()
    return vocab


class WordPieceTokenizer(Tokenizer):
    """Word piece tokenizer for BERT models."""


    class Config(ConfigBase):
        basic_tokenizer: BERTInitialTokenizer.Config = BERTInitialTokenizer.Config()
        wordpiece_vocab_path: str = '/mnt/vol/nlp_technologies/bert/uncased_L-12_H-768_A-12/vocab.txt'

    def __init__(self, wordpiece_vocab, basic_tokenizer, wordpiece_tokenizer) ->None:
        self.vocab = wordpiece_vocab
        self.basic_tokenizer = basic_tokenizer
        self.wordpiece_tokenizer = wordpiece_tokenizer
        log_class_usage(__class__)

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = create_component(ComponentType.TOKENIZER, config.basic_tokenizer)
        vocab = load_vocab(config.wordpiece_vocab_path)
        wordpiece_tokenizer = WordpieceTokenizer(vocab=vocab)
        return cls(vocab, basic_tokenizer, wordpiece_tokenizer)

    def tokenize(self, input_str: str) ->List[Token]:
        tokens = []
        for token in self.basic_tokenizer.tokenize(input_str):
            start = token.start
            for sub_token in self.wordpiece_tokenizer.tokenize(token.value):
                piece_len = len(sub_token) if not sub_token.startswith('##') else len(sub_token) - 2
                if sub_token == '[UNK]':
                    piece_len = len(token.value)
                end = start + piece_len
                tokens.append(Token(sub_token, start, end))
                start = end
        return [token for token in tokens if token.value]


class SquadTensorizer(TokenTensorizer):
    """Produces inputs and answer spans for Squad."""
    __EXPANSIBLE__ = True
    SPAN_PAD_IDX = -100


    class Config(TokenTensorizer.Config):
        doc_column: str = 'doc'
        ques_column: str = 'question'
        answers_column: str = 'answers'
        answer_starts_column: str = 'answer_starts'
        tokenizer: Tokenizer.Config = Tokenizer.Config(split_regex='\\W+')
        max_ques_seq_len: int = 64
        max_doc_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        vocab = None
        if isinstance(tokenizer, WordPieceTokenizer):
            None
            replacements = {'[UNK]': UNK, '[PAD]': PAD, '[CLS]': BOS, '[SEP]': EOS, '[MASK]': MASK}
            vocab = Vocabulary([token for token, _ in tokenizer.vocab.items()], replacements=replacements)
        doc_tensorizer = TokenTensorizer(text_column=config.doc_column, tokenizer=tokenizer, vocab=vocab, max_seq_len=config.max_doc_seq_len)
        ques_tensorizer = TokenTensorizer(text_column=config.ques_column, tokenizer=tokenizer, vocab=vocab, max_seq_len=config.max_ques_seq_len)
        return cls(doc_tensorizer=doc_tensorizer, ques_tensorizer=ques_tensorizer, doc_column=config.doc_column, ques_column=config.ques_column, answers_column=config.answers_column, answer_starts_column=config.answer_starts_column, tokenizer=tokenizer, vocab=vocab, **kwargs)

    def __init__(self, doc_tensorizer: TokenTensorizer, ques_tensorizer: TokenTensorizer, doc_column: str=Config.doc_column, ques_column: str=Config.ques_column, answers_column: str=Config.answers_column, answer_starts_column: str=Config.answer_starts_column, **kwargs):
        super().__init__(text_column=None, **kwargs)
        self.ques_tensorizer = ques_tensorizer
        self.doc_tensorizer = doc_tensorizer
        self.doc_column = doc_column
        self.ques_column = ques_column
        self.answers_column = answers_column
        self.answer_starts_column = answer_starts_column

    def initialize(self, vocab_builder=None, from_scratch=True):
        """Build vocabulary based on training corpus."""
        if isinstance(self.tokenizer, WordPieceTokenizer):
            return
        if not self.vocab_builder or from_scratch:
            self.vocab_builder = vocab_builder or VocabBuilder()
            self.vocab_builder.pad_index = 0
            self.vocab_builder.unk_index = 1
        ques_initializer = self.ques_tensorizer.initialize(self.vocab_builder, from_scratch)
        doc_initializer = self.doc_tensorizer.initialize(self.vocab_builder, from_scratch)
        ques_initializer.send(None)
        doc_initializer.send(None)
        try:
            while True:
                row = yield
                ques_initializer.send(row)
                doc_initializer.send(row)
        except GeneratorExit:
            self.vocab = self.vocab_builder.make_vocab()

    def _lookup_tokens(self, text, source_is_doc=True):
        return self.doc_tensorizer._lookup_tokens(text) if source_is_doc else self.ques_tensorizer._lookup_tokens(text)

    def numberize(self, row):
        assert len(self.vocab) == len(self.ques_tensorizer.vocab)
        assert len(self.vocab) == len(self.doc_tensorizer.vocab)
        ques_tokens, _, _ = self.ques_tensorizer._lookup_tokens(row[self.ques_column])
        doc_tokens, orig_start_idx, orig_end_idx = self.doc_tensorizer._lookup_tokens(row[self.doc_column])
        start_idx_map = {}
        end_idx_map = {}
        for token_idx, (start_idx, end_idx) in enumerate(zip(orig_start_idx, orig_end_idx)):
            start_idx_map[start_idx] = token_idx
            end_idx_map[end_idx] = token_idx
        answer_start_token_indices = [start_idx_map.get(raw_idx, self.SPAN_PAD_IDX) for raw_idx in row[self.answer_starts_column]]
        answer_end_token_indices = [end_idx_map.get(raw_idx + len(answer), self.SPAN_PAD_IDX) for raw_idx, answer in zip(row[self.answer_starts_column], row[self.answers_column])]
        if not (answer_start_token_indices and answer_end_token_indices) or self._only_pad(answer_start_token_indices) or self._only_pad(answer_end_token_indices):
            answer_start_token_indices = [self.SPAN_PAD_IDX]
            answer_end_token_indices = [self.SPAN_PAD_IDX]
        return doc_tokens, len(doc_tokens), ques_tokens, len(ques_tokens), answer_start_token_indices, answer_end_token_indices

    def tensorize(self, batch):
        doc_tokens, doc_seq_len, ques_tokens, ques_seq_len, answer_start_idx, answer_end_idx = zip(*batch)
        doc_tokens = pad_and_tensorize(doc_tokens, self.vocab.get_pad_index())
        doc_mask = (doc_tokens == self.vocab.get_pad_index()).byte()
        ques_tokens = pad_and_tensorize(ques_tokens, self.vocab.get_pad_index())
        ques_mask = (ques_tokens == self.vocab.get_pad_index()).byte()
        answer_start_idx = pad_and_tensorize(answer_start_idx, self.SPAN_PAD_IDX)
        answer_end_idx = pad_and_tensorize(answer_end_idx, self.SPAN_PAD_IDX)
        return doc_tokens, pad_and_tensorize(doc_seq_len), doc_mask, ques_tokens, pad_and_tensorize(ques_seq_len), ques_mask, answer_start_idx, answer_end_idx

    def sort_key(self, row):
        raise NotImplementedError('SquadTensorizer.sort_key() should not be called.')

    def _only_pad(self, token_id_list: List[int]) ->bool:
        for token_id in token_id_list:
            if token_id != self.SPAN_PAD_IDX:
                return False
        return True


class RnnType(Enum):
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'


RNN_TYPE_DICT = {RnnType.RNN: nn.RNN, RnnType.LSTM: nn.LSTM, RnnType.GRU: nn.GRU}


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

    def __init__(self, config: Config, input_size: int, padding_value: float=0.0):
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
            self.rnns.append(rnn_module(input_size, config.hidden_size, num_layers=1, bidirectional=config.bidirectional))
        self.representation_dim = (config.num_layers if config.concat_layers else 1) * config.hidden_size * (2 if config.bidirectional else 1)
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
        seq_lengths_sorted, idx_of_sorted = torch.sort(seq_lengths, dim=0, descending=True)
        tokens_sorted = tokens.index_select(0, idx_of_sorted)
        packed_tokens = nn.utils.rnn.pack_padded_sequence(tokens_sorted, seq_lengths_sorted, batch_first=True)
        outputs = [packed_tokens]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            rnn_input = nn.utils.rnn.PackedSequence(self.dropout(rnn_input.data), rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        outputs = outputs[1:]
        for i in range(len(outputs)):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(outputs[i], padding_value=self.padding_value, batch_first=True)[0]
        output = torch.cat(outputs, 2) if self.concat_layers else outputs[-1]
        _, idx_of_original = torch.sort(idx_of_sorted, dim=0)
        output = output.index_select(0, idx_of_original)
        max_seq_len = tokens_mask.size(1)
        batch_size, output_seq_len, output_dim = output.size()
        if output_seq_len != max_seq_len:
            padding = torch.zeros(batch_size, max_seq_len - output_seq_len, output_dim).type(output.data.type())
            output = torch.cat([output, padding], 1)
        return output


class PairwiseRankingLoss(Loss):
    """
    Given embeddings for a query, positive response and negative response
    computes pairwise ranking hinge loss
    """


    class Config(ConfigBase):
        margin: float = 1.0

    @staticmethod
    def get_similarities(embeddings):
        pos_embed, neg_embed, query_embed = embeddings
        pos_similarity = F.cosine_similarity(query_embed, pos_embed)
        neg_similarity = F.cosine_similarity(query_embed, neg_embed)
        return pos_similarity, neg_similarity, query_embed.size(0)

    def __call__(self, logits, targets, reduce=True):
        pos_similarity, neg_similarity, batch_size = self.get_similarities(logits)
        targets_local = FloatTensor(batch_size)
        targets_local.fill_(1)
        return F.margin_ranking_loss(pos_similarity, neg_similarity, targets_local, self.config.margin)


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

    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool, use_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias
        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim
            self.input_linearity = nn.Linear(self.embed_dim, self._highway_inp_proj_end, bias=self.use_bias)
            self.state_linearity = nn.Linear(self.lstm_dim, self._highway_inp_proj_start, bias=True)
        else:
            self.input_linearity = nn.Linear(self.embed_dim, 4 * self.lstm_dim, bias=self.use_bias)
            self.state_linearity = nn.Linear(self.lstm_dim, 4 * self.lstm_dim, bias=True)
        self.reset_parameters()
        log_class_usage(__class__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, states=Tuple[torch.Tensor, torch.Tensor], variational_dropout_mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
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
        input_gate = forget_gate = memory_init = output_gate = highway_gate = None
        if self.use_highway:
            fused_op = projected_input[:, :5 * self.lstm_dim] + projected_state
            fused_chunked = torch.chunk(fused_op, 5, 1)
            input_gate, forget_gate, memory_init, output_gate, highway_gate = fused_chunked
            highway_gate = torch.sigmoid(highway_gate)
        else:
            fused_op = projected_input + projected_state
            input_gate, forget_gate, memory_init, output_gate = torch.chunk(fused_op, 4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        memory_init = torch.tanh(memory_init)
        output_gate = torch.sigmoid(output_gate)
        memory = input_gate * memory_init + forget_gate * memory_state
        timestep_output: torch.Tensor = output_gate * torch.tanh(memory)
        if self.use_highway:
            highway_input_projection = projected_input[:, self._highway_inp_proj_start:self._highway_inp_proj_end]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection
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

    def __init__(self, embed_dim: int, lstm_dim: int, go_forward: bool=True, recurrent_dropout_probability: float=0.0, use_highway: bool=True, use_input_projection_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.cell = AugmentedLSTMCell(self.embed_dim, self.lstm_dim, self.use_highway, use_input_projection_bias)
        log_class_usage(__class__)

    def get_dropout_mask(self, dropout_probability: float, tensor_for_masking: torch.Tensor) ->torch.Tensor:
        binary_mask = torch.rand(tensor_for_masking.size()) > dropout_probability
        dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
        return dropout_mask

    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) ->Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
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
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.lstm_dim)
        if states is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.lstm_dim)
            full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size, self.lstm_dim)
        else:
            full_batch_previous_state = states[0].squeeze(0)
            full_batch_previous_memory = states[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = self.get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_memory)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:current_length_index + 1].clone()
            timestep_input = sequence_tensor[0:current_length_index + 1, (index)]
            timestep_output, memory = self.cell(timestep_input, (previous_state, previous_memory), dropout_mask[0:current_length_index + 1] if dropout_mask is not None else None)
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, (index), :] = timestep_output
        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        final_state = full_batch_previous_state.unsqueeze(0), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


class ContextualWordConvolution(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, k, padding=k - 1) for k in kernel_sizes])
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

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, padding: int, dilation: int, bottleneck: int):
        super(SeparableConv1d, self).__init__()
        conv_layers = [nn.Conv1d(input_channels, input_channels, kernel_size, padding=padding, dilation=dilation, groups=input_channels)]
        if bottleneck > 0:
            conv_layers.extend([nn.Conv1d(input_channels, bottleneck, 1), nn.Conv1d(bottleneck, output_channels, 1)])
        else:
            conv_layers.append(nn.Conv1d(input_channels, output_channels, 1))
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)


class OrderedNeuronLSTMLayer(Module):

    def __init__(self, embed_dim: int, lstm_dim: int, padding_value: float, dropout: float) ->None:
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

    def forward(self, embedded_tokens: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor], seq_lengths: List[int]) ->Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            master_forget = torch.cumsum(F.softmax(master_forget_no_cumax, dim=1), dim=1)
            master_input_no_cumax = self.master_input_no_cumax_gate(combined)
            master_input = torch.cumsum(F.softmax(master_input_no_cumax, dim=1), dim=1)
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
        return torch.stack(all_hidden), (torch.stack(state_hidden), torch.stack(state_context))


class BoundaryPool(Module):


    class Config(ConfigBase):
        boundary_type: str = 'first'

    def __init__(self, config: Config, n_input: int) ->None:
        super().__init__(config)
        self.boundary_type = config.boundary_type
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor=None) ->torch.Tensor:
        max_len = inputs.size()[1]
        if self.boundary_type == 'first':
            return inputs[:, (0), :]
        elif self.boundary_type == 'last':
            assert max_len > 1
            return inputs[:, (max_len - 1), :]
        elif self.boundary_type == 'firstlast':
            assert max_len > 1
            return torch.cat((inputs[:, (0), :], inputs[:, (max_len - 1), :]), dim=1)
        else:
            raise Exception('Unknown configuration type {}'.format(self.boundary_type))


class SlotAttentionType(Enum):
    NO_ATTENTION = 'no_attention'
    CONCAT = 'concat'
    MULTIPLY = 'multiply'
    DOT = 'dot'


class SlotAttention(Module):


    class Config(ConfigBase):
        attn_dimension: int = 64
        attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION

    def __init__(self, config: Config, n_input: int, batch_first: bool=True) ->None:
        super().__init__()
        self.batch_first = batch_first
        self.attention_type = config.attention_type
        if self.attention_type == SlotAttentionType.CONCAT:
            self.attention_add = nn.Sequential(nn.Linear(2 * n_input, config.attn_dimension, bias=False), nn.Tanh(), nn.Linear(config.attn_dimension, 1, bias=False))
        elif self.attention_type == SlotAttentionType.MULTIPLY:
            self.attention_mult = nn.Linear(n_input, n_input, bias=False)
        log_class_usage(__class__)

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        if isinstance(inputs, PackedSequence):
            inputs, lengths = pad_packed_sequence(inputs, batch_first=self.batch_first)
        size = inputs.size()
        exp_inputs_2 = inputs.unsqueeze(1).expand(size[0], size[1], size[1], size[2])
        if self.attention_type == SlotAttentionType.CONCAT:
            exp_inputs_1 = inputs.unsqueeze(2).expand(size[0], size[1], size[1], size[2])
            catted = torch.cat((exp_inputs_1, exp_inputs_2), 3)
            attn_weights_add = F.softmax(self.attention_add(catted).squeeze(3), dim=2).unsqueeze(2)
            context_add = torch.matmul(attn_weights_add, exp_inputs_2).squeeze(2)
            output = torch.cat((inputs, context_add), 2)
        elif self.attention_type == SlotAttentionType.MULTIPLY or self.attention_type == SlotAttentionType.DOT:
            attended = inputs if self.attention_type == SlotAttentionType.DOT else self.attention_mult(inputs)
            attn_weights_mult = F.softmax(torch.matmul(inputs, torch.transpose(attended, 1, 2)), dim=2).unsqueeze(2)
            context_mult = torch.matmul(attn_weights_mult, exp_inputs_2).squeeze(2)
            output = torch.cat((inputs, context_mult), 2)
        else:
            output = inputs
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

    def __init__(self, embed_dim: int, num_heads: int, scaling: float=0.125, dropout: float=0.1):
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
        assert list(attn_weights.shape) == [batch_heads, target_length, source_length]
        attn_weights = attn_weights.view(batch_size, self.num_heads, target_length, source_length)
        attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = attn_weights.view(batch_heads, target_length, source_length)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.shape) == [batch_heads, target_length, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(target_length, batch_size, embed_dim)
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

    def __init__(self, num_embeddings: int, embedding_dim: int, pad_index: Optional[int]=None):
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


class ResidualMLP(nn.Module):
    """A square MLP component which can learn a bias on an input vector.
    This MLP in particular defaults to using GeLU as its activation function
    (this can be changed by passing a different activation function),
    and retains a residual connection to its original input to help with gradient
    propogation.

    Unlike pytext's MLPDecoder it doesn't currently allow adding a LayerNorm
    in between hidden layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float=0.1, activation=GeLU):
        super().__init__()
        modules = []
        for last_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.extend([nn.Linear(last_dim, dim), activation(), nn.Dropout(dropout)])
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        modules.extend([nn.Linear(last_dim, input_dim), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*modules)
        log_class_usage(__class__)

    def forward(self, input):
        bias = self.mlp(input)
        return input + bias


DEFAULT_EMBEDDING_DIM = 768


DEFAULT_MAX_SEQUENCE_LENGTH = 514


DEFAULT_NUM_LAYERS = 12


DEFAULT_PADDING_IDX = 1


DEFAULT_VOCAB_SIZE = 50265


DEFAULT_NUM_ATTENTION_HEADS = 12


class TransformerLayer(nn.Module):

    def __init__(self, embedding_dim: int=DEFAULT_EMBEDDING_DIM, attention: Optional[MultiheadSelfAttention]=None, residual_mlp: Optional[ResidualMLP]=None, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention or MultiheadSelfAttention(embedding_dim, num_heads=DEFAULT_NUM_ATTENTION_HEADS)
        self.residual_mlp = residual_mlp or ResidualMLP(embedding_dim, hidden_dims=[embedding_dim * 4])
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


class Transformer(nn.Module):

    def __init__(self, vocab_size: int=DEFAULT_VOCAB_SIZE, embedding_dim: int=DEFAULT_EMBEDDING_DIM, padding_idx: int=DEFAULT_PADDING_IDX, max_seq_len: int=DEFAULT_MAX_SEQUENCE_LENGTH, layers: List[TransformerLayer]=(), dropout: float=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.layers = nn.ModuleList(layers or [TransformerLayer(embedding_dim) for _ in range(DEFAULT_NUM_LAYERS)])
        self.positional_embedding = PositionalEmbedding(max_seq_len, embedding_dim, padding_idx)
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        log_class_usage(__class__)

    def forward(self, tokens: torch.Tensor) ->List[torch.Tensor]:
        padding_mask = tokens.eq(self.padding_idx)
        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)
        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        padded_normed = normed * (1 - padding_mask.unsqueeze(-1).type_as(normed))
        encoded = padded_normed.transpose(0, 1)
        states = [encoded]
        for layer in self.layers:
            encoded = layer(encoded, padding_mask)
            states.append(encoded)
        return states


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
            items_to_add[new_key] = torch.zeros_like(state[k]).repeat(3, 1) if len(state[k].shape) > 1 else torch.zeros_like(state[k]).repeat(3)
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
    return {(k if not regex.findall(k) else regex.sub(replacement, k)): v for k, v in state.items()}


def rename_component_from_root(state, old_name, new_name):
    """Rename keys from state using full python paths"""
    return rename_state_keys(state, '^' + old_name.replace('.', '\\.') + '.?(.*)$', new_name + '.\\1')


def translate_roberta_state_dict(state_dict):
    """Translate the public RoBERTa weights to ones which match SentenceEncoder."""
    new_state = rename_component_from_root(state_dict, 'decoder.sentence_encoder', 'transformer')
    new_state = rename_state_keys(new_state, 'embed_tokens', 'token_embedding')
    new_state = rename_state_keys(new_state, 'embed_positions', 'positional_embedding.embedding')
    new_state = rename_state_keys(new_state, 'emb_layer_norm', 'embedding_layer_norm')
    new_state = rename_state_keys(new_state, 'self_attn', 'attention')
    new_state = merge_input_projection(new_state)
    new_state = rename_state_keys(new_state, '_proj.(.*)', 'put_projection.\\1')
    new_state = rename_state_keys(new_state, 'fc1', 'residual_mlp.mlp.0')
    new_state = rename_state_keys(new_state, 'fc2', 'residual_mlp.mlp.3')
    new_state = remove_state_keys(new_state, '^sentence_')
    new_state = remove_state_keys(new_state, '_classification_head.')
    new_state = remove_state_keys(new_state, '^decoder\\.lm_head')
    new_state = remove_state_keys(new_state, 'segment_embedding')
    return new_state


class SentenceEncoder(nn.Module):
    """
    This is a TorchScriptable implementation of RoBERTa from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa model, and within that
    is restructured and rewritten to be able to be compiled by TorchScript for
    production use cases.

    This SentenceEncoder can load in the public RoBERTa weights directly with
    `load_roberta_state_dict`, which will translate the keys as they exist in
    the publicly released RoBERTa to the correct structure for this implementation.
    The default constructor value will have the same size and shape as that model.

    To use RoBERTa with this, download the RoBERTa public weights as `roberta.weights`

    >>> encoder = SentenceEncoder()
    >>> weights = torch.load("roberta.weights")
    >>> encoder.load_roberta_state_dict(weights)

    Within this you will still need to preprocess inputs using fairseq and the publicly
    released vocabs, and finally place this encoder in a model alongside say an MLP
    output layer to do classification.
    """

    def __init__(self, transformer: Optional[Transformer]=None):
        super().__init__()
        self.transformer = transformer or Transformer()
        log_class_usage(__class__)

    def forward(self, tokens):
        all_layers = self.extract_features(tokens)
        last_layer = all_layers[-1]
        return last_layer.transpose(0, 1)

    def extract_features(self, tokens):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        return self.transformer(tokens)

    def load_roberta_state_dict(self, state_dict):
        return self.load_state_dict(translate_roberta_state_dict(state_dict))


class PoolingMethod(Enum):
    """
    Pooling Methods are chosen from the "Feature-based Approachs" section in
    https://arxiv.org/pdf/1810.04805.pdf
    """
    AVG_CONCAT_LAST_4_LAYERS = 'avg_concat_last_4_layers'
    AVG_SECOND_TO_LAST_LAYER = 'avg_second_to_last_layer'
    AVG_LAST_LAYER = 'avg_last_layer'
    AVG_SUM_LAST_4_LAYERS = 'avg_sum_last_4_layers'
    CLS_TOKEN = 'cls_token'
    NO_POOL = 'no_pool'


class BERTTensorizerBase(Tensorizer):
    """
    Base Tensorizer class for all BERT style models including XLM,
    RoBERTa and XLM-R.
    """
    __EXPANSIBLE__ = True


    class Config(Tensorizer.Config):
        columns: List[str] = ['text']
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        base_tokenizer: Optional[Tokenizer.Config] = None
        vocab_file: str = ''
        max_seq_len: int = 256

    def __init__(self, columns: List[str]=Config.columns, vocab: Vocabulary=None, tokenizer: Tokenizer=None, max_seq_len: int=Config.max_seq_len, base_tokenizer: Tokenizer=None) ->None:
        super().__init__()
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.base_tokenizer = base_tokenizer
        self.max_seq_len = max_seq_len
        self.bos_token = self.vocab.bos_token

    @property
    def column_schema(self):
        return [(column, str) for column in self.columns]

    @lazy_property
    def tensorizer_script_impl(self):
        return self.__TENSORIZER_SCRIPT_IMPL__(tokenizer=self.tokenizer, vocab=self.vocab, max_seq_len=self.max_seq_len)

    def numberize(self, row: Dict) ->Tuple[Any, ...]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model.
        """
        per_sentence_tokens = [self.tokenizer.tokenize(row[column]) for column in self.columns]
        return self.tensorizer_script_impl.numberize(per_sentence_tokens)

    def tensorize(self, batch) ->Tuple[torch.Tensor, ...]:
        """
        Convert instance level vectors into batch level tensors.
        """
        return self.tensorizer_script_impl.tensorize_wrapper(*zip(*batch))

    def initialize(self, vocab_builder=None, from_scratch=True):
        return
        yield

    def sort_key(self, row):
        return row[2]


@torch.jit.script
def long_tensor_2d(shape: Tuple[int, int], fill_value: int=0) ->torch.Tensor:
    """Return a new 2d torch.LongTensor with size according to shape.
    The values of this tensor will be fill_value."""
    outer = torch.jit.annotate(List[List[int]], [])
    inner = torch.jit.annotate(List[int], [])
    for _i in range(shape[1]):
        inner.append(fill_value)
    for _i in range(shape[0]):
        outer.append(inner)
    return torch.tensor(outer, dtype=torch.long)


@torch.jit.script
def pad_2d_mask(input: List[List[int]], pad_value: int=0) ->Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list to a 2d tensor. Returns a pair of tensors, the padded tensor
    as well as a mask tensor. The mask tensor has the same shape as the padded tensor,
    with a 1 in the position of non-pad values and a 0 in the position of pads."""
    max_len = 0
    for i in input:
        max_len = max(max_len, len(i))
    tensor = long_tensor_2d((len(input), max_len), pad_value)
    mask = long_tensor_2d((len(input), max_len), 0)
    for i in range(len(input)):
        for j in range(len(input[i])):
            tensor[i][j] = input[i][j]
            mask[i][j] = 1
    return tensor, mask


class BERTTensorizerBaseScriptImpl(TensorizerScriptImpl):

    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = ScriptVocabulary(list(vocab), pad_idx=vocab.get_pad_index(), bos_idx=vocab.get_bos_index(-1), eos_idx=vocab.get_eos_index(-1), unk_idx=vocab.get_unk_index())
        self.vocab_lookup = VocabLookup(self.vocab)
        self.max_seq_len = max_seq_len

    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]], max_seq_len: Optional[int]=None) ->Tuple[List[int], List[int], List[int]]:
        """
        This function knows how to call lookup_tokens with the correct
        settings for this model. The default behavior is to wrap the
        numberized text with distinct BOS and EOS tokens. The resulting
        vector would look something like this:
        [BOS, token1_id, . . . tokenN_id, EOS]

        The function also takes an optional seq_len parameter which is
        used to customize truncation in case we have multiple text fields.
        By default max_seq_len is used. It's upto the numberize function of
        the class to decide how to use the seq_len param.

        For example:
        - In the case of sentence pair classification, we might want both
        pieces of text have the same length which is half of the
        max_seq_len supported by the model.
        - In the case of QA, we might want to truncate the context by a
        seq_len which is longer than what we use for the question.

        Args:
            tokens: a list of tokens represent a sentence, each token represented
            by token string, start and end indices.

        Returns:
            tokens_ids: List[int], a list of token ids represent a sentence.
            start_indices: List[int], each token start indice in the sentence.
            end_indices: List[int], each token end indice in the sentence.
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return self.vocab_lookup(tokens, bos_idx=self.vocab.bos_idx, eos_idx=self.vocab.eos_idx, use_eos_token_for_bos=False, max_seq_len=max_seq_len)

    def _wrap_numberized_tokens(self, numberized_tokens: List[int], idx: int) ->List[int]:
        """
        If a class has a non-standard way of generating the final numberized text
        (eg: BERT) then a class specific version of wrap_numberized_text function
        should be implemented. This allows us to share the numberize
        function across classes without having to copy paste code. The default
        implementation doesnt do anything.
        """
        return numberized_tokens

    def numberize(self, per_sentence_tokens: List[List[Tuple[str, int, int]]]) ->Tuple[List[int], List[int], int, List[int]]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model.

        Args:
            per_sentence_tokens: list of tokens per sentence level in one row,
            each token represented by token string, start and end indices.

        Returns:
            tokens: List[int], a list of token ids, concatenate all
            sentences token ids.
            segment_labels: List[int], denotes each token belong to
            which sentence.
            seq_len: int, tokens length
            positions: List[int], token positions
        """
        tokens: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        positions: List[int] = []
        for idx, single_sentence_tokens in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(single_sentence_tokens)[0]
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)
            tokens.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(tokens)
        positions = [i for i in range(seq_len)]
        return tokens, segment_labels, seq_len, positions

    def tensorize(self, tokens_2d: List[List[int]], segment_labels_2d: List[List[int]], seq_lens_1d: List[int], positions_2d: List[List[int]]) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert instance level vectors into batch level tensors.
        """
        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels = torch.tensor(pad_2d(segment_labels_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long)
        positions = torch.tensor(pad_2d(positions_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long)
        if self.device == '':
            return tokens, pad_mask, segment_labels, positions
        else:
            return tokens, pad_mask, segment_labels, positions

    def tokenize(self, row_text: Optional[List[str]], row_pre_tokenized: Optional[List[List[str]]]) ->List[List[Tuple[str, int, int]]]:
        """
        This function convert raw inputs into tokens, each token is represented
        by token(str), start and end indices in the raw inputs. There are two
        possible inputs to this function depends if the tokenized in implemented
        in TorchScript or not.

        Case 1: Tokenizer has a full TorchScript implementation, the input will
        be a list of sentences (in most case it is single sentence or a pair).

        Case 2: Tokenizer have partial or no TorchScript implementation, in most
        case, the tokenizer will be host in Yoda, the input will be a list of
        pre-processed tokens.

        Returns:
            per_sentence_tokens: tokens per setence level, each token is
            represented by token(str), start and end indices.
        """
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        if row_text is not None:
            for text in row_text:
                per_sentence_tokens.append(self.tokenizer.tokenize(text))
        elif row_pre_tokenized is not None:
            for sentence_pre_tokenized in row_pre_tokenized:
                sentence_tokens: List[Tuple[str, int, int]] = []
                for token in sentence_pre_tokenized:
                    sentence_tokens.extend(self.tokenizer.tokenize(token))
                per_sentence_tokens.append(sentence_tokens)
        return per_sentence_tokens

    def forward(self, inputs: ScriptBatchInput) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wire up tokenize(), numberize() and tensorize() functions for data
        processing.
        When export to TorchScript, the wrapper module should choose to use
        texts or pre_tokenized based on the TorchScript tokenizer
        implementation (e.g use external tokenizer such as Yoda or not).
        """
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[int]] = []
        for idx in range(self.batch_size(inputs)):
            tokens: List[List[Tuple[str, int, int]]] = self.tokenize(self.get_texts_by_index(inputs.texts, idx), self.get_tokens_by_index(inputs.tokens, idx))
            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(tokens)
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_lens_1d.append(numberized[2])
            positions_2d.append(numberized[3])
        return self.tensorize(tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d)

    def torchscriptify(self):
        if not isinstance(self.tokenizer, torch.jit.ScriptModule):
            self.tokenizer = self.tokenizer.torchscriptify()
        return super().torchscriptify()


RoBERTaTensorizerScriptImpl = BERTTensorizerBaseScriptImpl


BOS_WORD = '<s>'


EOS_WORD = '</s>'


PAD_WORD = '<pad>'


SPECIAL_WORD = '<special%i>'


SPECIAL_WORDS = 10


UNK_WORD = '<unk>'


logger = logging.getLogger(__name__)


class Dictionary(object):

    def __init__(self, id2word, word2id, counts):
        assert len(id2word) == len(word2id) == len(counts)
        self.id2word = id2word
        self.word2id = word2id
        self.counts = counts
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        assert all(self.id2word[4 + i] == SPECIAL_WORD % i for i in range(SPECIAL_WORDS))
        assert len(self.id2word) == len(self.word2id) == len(self.counts)
        assert set(self.word2id.keys()) == set(self.counts.keys())
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i
        last_count = 1e+18
        for i in range(4 + SPECIAL_WORDS, len(self.id2word) - 1):
            count = self.counts[self.id2word[i]]
            assert count <= last_count
            last_count = count

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.unk_index)

    def max_vocab(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info('Maximum vocabulary size: %i. Dictionary size: %i -> %i (removed %i words).' % (max_vocab, init_size, len(self), init_size - len(self)))

    def min_count(self, min_count):
        """
        Threshold on the word frequency counts.
        """
        assert min_count >= 0
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if self.counts[self.id2word[k]] >= min_count or k < 4 + SPECIAL_WORDS}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info('Minimum frequency count: %i. Dictionary size: %i -> %i (removed %i words).' % (min_count, init_size, len(self), init_size - len(self)))

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert PathManager.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(SPECIAL_WORDS):
            word2id[SPECIAL_WORD % i] = 4 + i
        counts = {k: (0) for k in word2id.keys()}
        f = PathManager.open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            if len(line) != 2:
                skipped += 1
                continue
            assert len(line) == 2, (i, line)
            assert line[1].isdigit(), (i, line)
            if line[0] in word2id:
                skipped += 1
                None
                continue
            if not line[1].isdigit():
                skipped += 1
                None
                continue
            word2id[line[0]] = 4 + SPECIAL_WORDS + i - skipped
            counts[line[0]] = int(line[1])
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, counts)
        logger.info('Read %i words from the vocabulary file.' % len(dico))
        if skipped > 0:
            logger.warning('Skipped %i empty lines!' % skipped)
        return dico

    @staticmethod
    def index_data(path, bin_path, dico):
        """
        Index sentences with a dictionary.
        """
        if bin_path is not None and PathManager.isfile(bin_path):
            None
            data = torch.load(bin_path)
            assert dico == data['dico']
            return data
        positions = []
        sentences = []
        unk_words = {}
        f = PathManager.open(path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                None
            s = line.rstrip().split()
            if len(s) == 0:
                None
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w, no_unk=False)
                if 0 <= word_id < 4 + SPECIAL_WORDS and word_id != 3:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(1)
        f.close()
        positions = np.int64(positions)
        if len(dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception('Dictionary is too big.')
        assert sentences.min() >= 0
        data = {'dico': dico, 'positions': positions, 'sentences': sentences, 'unk_words': unk_words}
        if bin_path is not None:
            None
            torch.save(data, bin_path, pickle_protocol=4)
        return data


def build_fairseq_vocab(vocab_file: str, dictionary_class: Dictionary=Dictionary, special_token_replacements: Dict[str, SpecialToken]=None, max_vocab: int=-1, min_count: int=-1, tokens_to_add: Optional[List[str]]=None) ->Vocabulary:
    """
    Function builds a PyText vocabulary for models pre-trained using Fairseq
    modules. The dictionary class can take any Fairseq Dictionary class
    and is used to load the vocab file.
    """
    dictionary = dictionary_class.load(vocab_file)
    if min_count > 0 or max_vocab > 0:
        dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
    if tokens_to_add:
        for token in tokens_to_add:
            dictionary.add_symbol(token)
    return Vocabulary(dictionary.symbols, dictionary.count, replacements=special_token_replacements)


class ScriptTensorizer(torch.jit.ScriptModule):

    def __init__(self):
        super().__init__()
        self.device = torch.jit.Attribute('', str)

    @torch.jit.script_method
    def set_device(self, device: str):
        self.device = device

    @torch.jit.script_method
    def tokenize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]):
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)
        """
        raise NotImplementedError

    @torch.jit.script_method
    def numberize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]):
        """
        Process a single line of raw inputs into numberized result, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling tokenize(), add special
        tokens and vocab lookup.
        """
        raise NotImplementedError

    @torch.jit.script_method
    def tensorize(self, texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[List[str]]]]=None):
        """
        Process raw inputs into model input tensors, it supports two input
        formats:
            1) multiple rows of texts (single sentence or a pair)
            2) multiple rows of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling numberize() and also
        padding the numberized result.
        """
        raise NotImplementedError

    @torch.jit.script_method
    def batch_size(self, texts: Optional[List[List[str]]], tokens: Optional[List[List[List[str]]]]) ->int:
        if texts is not None:
            return len(texts)
        elif tokens is not None:
            return len(tokens)
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    @torch.jit.script_method
    def row_size(self, texts_list: Optional[List[List[str]]]=None, tokens_list: Optional[List[List[List[str]]]]=None) ->int:
        if texts_list is not None:
            return len(texts_list[0])
        elif tokens_list is not None:
            return len(tokens_list[0])
        else:
            raise RuntimeError('Empty input for both texts and tokens.')

    @torch.jit.script_method
    def get_texts_by_index(self, texts: Optional[List[List[str]]], index: int) ->Optional[List[str]]:
        if texts is None:
            return None
        return texts[index]

    @torch.jit.script_method
    def get_tokens_by_index(self, tokens: Optional[List[List[List[str]]]], index: int) ->Optional[List[List[str]]]:
        if tokens is None:
            return None
        return tokens[index]


class ScriptBERTTensorizerBase(ScriptTensorizer):

    def __init__(self, tokenizer: torch.jit.ScriptModule, vocab: ScriptVocabulary, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_lookup = VocabLookup(vocab)
        self.max_seq_len = torch.jit.Attribute(max_seq_len, int)

    @torch.jit.script_method
    def tokenize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]) ->List[List[Tuple[str, int, int]]]:
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        if text_row is not None:
            for text in text_row:
                per_sentence_tokens.append(self.tokenizer.tokenize(text))
        elif token_row is not None:
            for sentence_raw_tokens in token_row:
                sentence_tokens: List[Tuple[str, int, int]] = []
                for raw_token in sentence_raw_tokens:
                    sentence_tokens.extend(self.tokenizer.tokenize(raw_token))
                per_sentence_tokens.append(sentence_tokens)
        return per_sentence_tokens

    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]]) ->List[int]:
        raise NotImplementedError

    @torch.jit.script_method
    def _wrap_numberized_tokens(self, token_ids: List[int], idx: int) ->List[int]:
        return token_ids

    @torch.jit.script_method
    def numberize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]) ->Tuple[List[int], List[int], int, List[int]]:
        token_ids: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        positions: List[int] = []
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = self.tokenize(text_row, token_row)
        for idx, per_sentence_token in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(per_sentence_token)
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)
            token_ids.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(token_ids)
        positions = [i for i in range(seq_len)]
        return token_ids, segment_labels, seq_len, positions

    @torch.jit.script_method
    def tensorize(self, texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[List[str]]]]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_len_2d: List[int] = []
        positions_2d: List[List[int]] = []
        for idx in range(self.batch_size(texts, tokens)):
            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(self.get_texts_by_index(texts, idx), self.get_tokens_by_index(tokens, idx))
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])
            positions_2d.append(numberized[3])
        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels = torch.tensor(pad_2d(segment_labels_2d, seq_lens=seq_len_2d, pad_idx=self.vocab.pad_idx), dtype=torch.long)
        positions = torch.tensor(pad_2d(positions_2d, seq_lens=seq_len_2d, pad_idx=self.vocab.pad_idx), dtype=torch.long)
        if self.device == '':
            return tokens, pad_mask, segment_labels, positions
        else:
            return tokens, pad_mask, segment_labels, positions


class ScriptRoBERTaTensorizer(ScriptBERTTensorizerBase):

    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]]) ->List[int]:
        return self.vocab_lookup(tokens, bos_idx=self.vocab.bos_idx, eos_idx=self.vocab.eos_idx, use_eos_token_for_bos=False, max_seq_len=self.max_seq_len)[0]


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
        self.linear_seq = nn.Sequential(nn.Linear(2 * lstm_dim, lstm_dim), nn.Tanh())

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
        lstm_hidden_fwd = xaviervar([1, 1, self.lstm_dim], device=device), xaviervar([1, 1, self.lstm_dim], device=device)
        lstm_hidden_rev = xaviervar([1, 1, self.lstm_dim], device=device), xaviervar([1, 1, self.lstm_dim], device=device)
        nonterminal_element = x[-1]
        reversed_rest = x[:-1]
        fwd_input = [nonterminal_element] + reverse_tensor_list(reversed_rest)
        rev_input = [nonterminal_element] + reversed_rest
        stacked_fwd = self.lstm_fwd(torch.stack(fwd_input), lstm_hidden_fwd)[0][0]
        stacked_rev = self.lstm_rev(torch.stack(rev_input), lstm_hidden_rev)[0][0]
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
        self.linear_seq = nn.Sequential(nn.Linear(lstm_dim, lstm_dim), nn.Tanh())

    @torch.jit.script_method
    def forward(self, x: List[torch.Tensor], device: str='cpu') ->torch.Tensor:
        combined = torch.sum(torch.cat(x, dim=0), dim=0, keepdim=True)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


class Element:
    """
    Generic element representing a token / non-terminal / sub-tree on a stack.
    Used to compute valid actions in the RNNG parser.
    """

    def __init__(self, node: Any) ->None:
        self.node = node

    def __eq__(self, other) ->bool:
        return self.node == other.node

    def __repr__(self) ->str:
        return str(self.node)


class StackLSTM(Sized):
    """
    The Stack LSTM from Dyer et al: https://arxiv.org/abs/1505.08075
    """

    def __init__(self, lstm: nn.LSTM):
        """
        Shapes:
            initial_state: (lstm_layers, 1, lstm_hidden_dim) each
        """
        self.lstm = lstm
        initial_state = FloatTensor(lstm.num_layers, 1, lstm.hidden_size).fill_(0), FloatTensor(lstm.num_layers, 1, lstm.hidden_size).fill_(0)
        self.stack = [(initial_state, (self._lstm_output(initial_state), Element('Root')))]

    def _lstm_output(self, state: Tuple[torch.Tensor, torch.Tensor]) ->torch.Tensor:
        """
        Shapes:
            state: (lstm_layers, 1, lstm_hidden_dim) each
            return value: (1, lstm_hidden_dim)
        """
        return state[0][-1]

    def push(self, expression: torch.Tensor, element: Element) ->None:
        """
        Shapes:
            expression: (1, lstm_input_dim)
        """
        old_top_state = self.stack[-1][0]
        _, new_top_state = self.lstm(expression.unsqueeze(0), old_top_state)
        self.stack.append((new_top_state, (self._lstm_output(new_top_state), element)))

    def pop(self) ->Tuple[torch.Tensor, Element]:
        """
        Pops and returns tuple of output embedding (1, lstm_hidden_dim) and element
        """
        return self.stack.pop()[1]

    def embedding(self) ->torch.Tensor:
        """
        Shapes:
            return value: (1, lstm_hidden_dim)
        """
        assert len(self.stack) > 0, 'stack size must be greater than 0'
        top_state = self.stack[-1][0]
        return self._lstm_output(top_state)

    def element_from_top(self, index: int) ->Element:
        return self.stack[-(index + 1)][1][1]

    def __len__(self) ->int:
        return len(self.stack) - 1

    def __str__(self) ->str:
        return '->'.join([str(x[1][1]) for x in self.stack])

    def copy(self):
        other = StackLSTM(self.lstm)
        other.stack = list(self.stack)
        return other


class ParserState:
    """
    Maintains state of the Parser. Useful for beam search
    """

    def __init__(self, parser=None):
        if not parser:
            return
        self.buffer_stackrnn = StackLSTM(parser.buff_rnn)
        self.stack_stackrnn = StackLSTM(parser.stack_rnn)
        self.action_stackrnn = StackLSTM(parser.action_rnn)
        self.predicted_actions_idx = []
        self.action_scores = []
        self.num_open_NT = 0
        self.is_open_NT: List[bool] = []
        self.found_unsupported = False
        self.action_p = torch.Tensor()
        self.neg_prob = 0

    def finished(self):
        return len(self.stack_stackrnn) == 1 and len(self.buffer_stackrnn) == 0

    def copy(self):
        other = ParserState()
        other.buffer_stackrnn = self.buffer_stackrnn.copy()
        other.stack_stackrnn = self.stack_stackrnn.copy()
        other.action_stackrnn = self.action_stackrnn.copy()
        other.predicted_actions_idx = self.predicted_actions_idx.copy()
        other.action_scores = self.action_scores.copy()
        other.num_open_NT = self.num_open_NT
        other.is_open_NT = self.is_open_NT.copy()
        other.neg_prob = self.neg_prob
        other.found_unsupported = self.found_unsupported
        other.action_p = self.action_p.detach()
        return other

    def __gt__(self, other):
        return self.neg_prob > other.neg_prob

    def __eq__(self, other):
        return self.neg_prob == other.neg_prob


CLOSE = ']'


INTENT_PREFIX = 'IN:'


COMBINATION_INTENT_LABEL = INTENT_PREFIX + 'COMBINE'


SLOT_PREFIX = 'SL:'


COMBINATION_SLOT_LABEL = SLOT_PREFIX + 'COMBINE'


ESCAPE = '\\'


class Node_Info:
    """
    This class extracts the essential information for a mode, for use in rules.
    """

    def __init__(self, node):
        self.label = node.label
        self.tokens = node.list_tokens()
        self.parent_label = self.get_parent(node)
        self.children = []
        for a in node.children:
            if type(a) == Slot or type(a) == Intent:
                self.children.append(a.label)
        self.token_indices = node.get_token_indices()
        self.ancestors = [a.label for a in node.list_ancestors()]
        self.descendents = [d.label for d in node.list_nonTerminals()]
        self.prior_token = None
        prior = node.get_prior_token()
        if prior:
            self.prior_token = prior.label
        if type(node) == Token:
            self.label_type = 'TOKEN'
        elif type(node) == Intent:
            self.label_type = 'INTENT'
        elif type(node) == Slot:
            self.label_type = 'SLOT'
        self.same_span = self.get_same_span(node)

    def get_same_span(self, node):
        if node.parent:
            if set(node.parent.list_tokens()) == set(node.list_tokens()):
                return True
        return False

    def get_parent(self, node):
        if node.parent and type(node.parent) != Root:
            return node.parent.label
        return None

    def __str__(self):
        result = []
        result.append('Info:')
        result.append('Label: ' + self.label)
        result.append('Tokens: ' + ' '.join(self.tokens))
        result.append('Token Indicies: ' + ', '.join([str(i) for i in self.token_indices]))
        result.append('Prior Token: ' + str(self.prior_token))
        result.append('Parent: ' + str(self.parent_label))
        result.append('Children: ' + ', '.join(self.children))
        result.append('Ancestors: ' + ', '.join(self.ancestors))
        result.append('Descendents: ' + ', '.join(self.descendents))
        result.append('Label Type: ' + str(self.label_type))
        result.append('Same Span: ' + str(self.same_span))
        return '\n'.join(result)


OPEN = '['


class Token_Info:
    """
    This class extracts the essential information for a token for use in rules.
    """

    def __init__(self, node):
        self.token_word = node.label
        self.parent_label = self.get_parent(node)
        self.ancestors = [a.label for a in node.list_ancestors()]
        self.prior_token = None
        prior = node.get_prior_token()
        if prior:
            self.prior_token = prior.label
        self.next_token = None
        next_token = node.get_next_token()
        if next_token:
            self.next_token = next_token.label

    def get_parent(self, node):
        if node.parent and type(node.parent) != Root:
            return node.parent.label
        return None

    def __str__(self):
        result = []
        result.append('Token Info:')
        result.append('Token Word: ' + self.token_word)
        result.append('Previous Token: ' + str(self.prior_token))
        result.append('Next Token: ' + str(self.next_token))
        result.append('Parent: ' + str(self.parent_label))
        result.append('Ancestors: ' + ', '.join(self.ancestors))
        return '\n'.join(result)


def escape_brackets(string: str) ->str:
    return re.sub(f'([\\{OPEN}\\{CLOSE}\\{ESCAPE}])', f'\\{ESCAPE}\\1', string)


class Node:

    def __init__(self, label):
        self.label = label
        self.children = []
        self.parent = None

    def list_ancestors(self):
        ancestors = []
        if self.parent:
            if type(self.parent) != Root:
                ancestors.append(self.parent)
                ancestors += self.parent.list_ancestors()
        return ancestors

    def validate_node(self, *args):
        if self.children:
            for child in self.children:
                child.validate_node(*args)

    def list_tokens(self):
        tokens = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    tokens.append(child.label)
                else:
                    tokens += child.list_tokens()
        return tokens

    def get_token_span(self):
        """
        0 indexed
        Like array slicing: For the first 3 tokens, returns 0, 3
        """
        indices = self.get_token_indices()
        if len(indices) > 0:
            return min(indices), max(indices) + 1
        else:
            return None

    def get_token_indices(self):
        indices = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    indices.append(child.index)
                else:
                    indices += child.get_token_indices()
        return indices

    def list_nonTerminals(self):
        """
        Returns all Intent and Slot nodes subordinate to this node
        """
        non_terminals = []
        for child in self.children:
            if type(child) != Root and type(child) != Token:
                non_terminals.append(child)
                non_terminals += child.list_nonTerminals()
        return non_terminals

    def list_terminals(self):
        """
        Returns all Token nodes
        """
        terminals = []
        for child in self.children:
            if type(child) == Token:
                terminals.append(child)
            else:
                terminals += child.list_terminals()
        return terminals

    def get_info(self):
        if type(self) == Token:
            return Token_Info(self)
        return Node_Info(self)

    def flat_str(self):
        string = ''
        if type(self) == Intent or type(self) == Slot:
            string = OPEN
        if type(self) != Root:
            string += escape_brackets(str(self.label)) + ' '
        if self.children:
            for child in self.children:
                string += child.flat_str()
        if type(self) == Intent or type(self) == Slot:
            string += CLOSE + ' '
        return string

    def children_flat_str_spans(self):
        string = str(self.get_token_span()) + ':'
        if self.children:
            for child in self.children:
                string += child.flat_str()
        return string

    def __str__(self):
        string = self._recursive_str('', '')
        return string

    def _recursive_str(self, string, spacer):
        string = spacer + str(self.label) + '\n'
        spacer += '\t'
        if self.children:
            for child in self.children:
                string += child._recursive_str(string, spacer)
        return string

    def __eq__(self, other):
        return self.label == other.label and self.children == other.children


class Intent(Node):

    def __init__(self, label):
        super().__init__(label)

    def validate_node(self, *args):
        super().validate_node(*args)
        for child in self.children:
            if type(child) == Intent or type(child) == Root:
                raise TypeError('An intent child must be a slot or token: ' + self.label)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.label == other.label and self.children == other.children


OOD_TOKEN = 'outOfDomain'


REDUCE = 'REDUCE'


SEQLOGICAL_LOTV_TOKEN = '0'


SHIFT = 'SHIFT'


class Annotation:

    def __init__(self, annotation_string: str, utterance: str='', brackets: str=OPEN + CLOSE, combination_labels: bool=True, add_dict_feat: bool=False, accept_flat_intents_slots: bool=False) ->None:
        super(Annotation, self).__init__()
        self.OPEN = brackets[0]
        self.CLOSE = brackets[1]
        self.combination_labels = combination_labels
        parts = annotation_string.rstrip().split('\t')
        if len(parts) == 5:
            [_, _, utterance, sparse_feat, self.seqlogical] = parts
        elif len(parts) == 1:
            [self.seqlogical] = parts
        else:
            raise ValueError('Cannot parse annotation_string')
        self.tree = Tree(self.build_tree(accept_flat_intents_slots), combination_labels, utterance)
        self.root: Root = self.tree.root

    def build_tree(self, accept_flat_intents_slots: bool=False):
        root = Root()
        node_stack: List[Any] = [root]
        curr_chars: List[str] = []
        token_count = 0
        expecting_label = False
        it = iter(self.seqlogical)
        while True:
            char = next(it, None)
            if char is None:
                break
            if char.isspace() or char in (OPEN, CLOSE):
                if curr_chars:
                    word = ''.join(curr_chars)
                    curr_chars = []
                    parent = node_stack[-1]
                    if expecting_label:
                        if word.startswith(INTENT_PREFIX):
                            node: Union[Intent, Slot, Token] = Intent(word)
                        elif word.startswith(SLOT_PREFIX):
                            node = Slot(word)
                        elif word == OOD_TOKEN:
                            node = Intent(word)
                        elif accept_flat_intents_slots:
                            if isinstance(parent, (Root, Slot)):
                                node = Intent(word)
                            elif isinstance(parent, Intent):
                                node = Slot(word)
                            else:
                                raise ValueError('The previous node in node_stack is not of type Root, Intent or Slot.')
                        else:
                            raise ValueError(f'Label {word} must start with IN: or SL:')
                        node_stack.append(node)
                        expecting_label = False
                    else:
                        if isinstance(parent, Root):
                            raise ValueError('Root cannot have a Token as child.')
                        node = Token(word, token_count)
                        token_count += 1
                    parent.children.append(node)
                    node.parent = parent
                if char in (OPEN, CLOSE):
                    if expecting_label:
                        raise ValueError("Invalid tree. No label found after '['.")
                    if char == OPEN:
                        expecting_label = True
                    else:
                        node_stack.pop()
            else:
                if char == ESCAPE:
                    char = next(it, None)
                    if char not in (OPEN, CLOSE, ESCAPE):
                        raise ValueError(f"Escape '{ESCAPE}' followed by none of '{OPEN}', '{CLOSE}', or '{ESCAPE}'.")
                curr_chars.append(char)
        if len(node_stack) != 1:
            raise ValueError('Invalid tree.')
        if len(root.children) > 1 and self.combination_labels:
            comb_intent = Intent(COMBINATION_INTENT_LABEL)
            node_stack.insert(1, comb_intent)
            for child in root.children:
                if type(child) == Intent:
                    comb_slot = Slot(COMBINATION_SLOT_LABEL)
                    comb_slot.parent = comb_intent
                    comb_slot.children.append(child)
                    comb_intent.children.append(comb_slot)
                    child.parent = comb_slot
                else:
                    child.parent = comb_intent
                    comb_intent.children.append(child)
            comb_intent.parent = root
            root.children = [comb_intent]
        return root

    def __str__(self):
        """
        A tab-indented version of the tree.
        strip() removes an extra final newline added during recursion
        """
        return self.tree.__str__()

    def __eq__(self, other):
        return self.tree == other.tree


def is_intent_nonterminal(node_label: str) ->bool:
    return node_label.startswith(INTENT_PREFIX)


def is_slot_nonterminal(node_label: str) ->bool:
    return node_label.startswith(SLOT_PREFIX)


def is_unsupported(node_label: str) ->bool:
    return is_intent_nonterminal(node_label) and node_label.lower().find('unsupported', 0) > 0


def is_valid_nonterminal(node_label: str) ->bool:
    return node_label.startswith(INTENT_PREFIX) or node_label.startswith(SLOT_PREFIX)


class AnnotationNumberizer(Tensorizer):
    """
    Not really a Tensorizer (since it does not create tensors) but technically
    serves the same function. This class parses Annotations in the format below
    and extracts the actions (type List[List[int]])
    ::

        [IN:GET_ESTIMATED_DURATION How long will it take to [SL:METHOD_TRAVEL
        drive ] from [SL:SOURCE Chicago ] to [SL:DESTINATION Mississippi ] ]

    Extraction algorithm is handled by Annotation class. We only care about
    the list of actions, which before vocab index lookups would look like:
    ::

        [
            IN:GET_ESTIMATED_DURATION, SHIFT, SHIFT, SHIFT, SHIFT, SHIFT, SHIFT,
            SL:METHOD_TRAVEL, SHIFT, REDUCE,
            SHIFT,
            SL:SOURCE, SHIFT, REDUCE,
            SHIFT,
            SL:DESTINATION, SHIFT, REDUCE,
        ]

    """


    class Config(Tensorizer.Config):
        column: str = 'seqlogical'

    @classmethod
    def from_config(cls, config: Config):
        return cls(column=config.column, is_input=config.is_input)

    def __init__(self, column: str=Config.column, vocab=None, is_input: bool=Config.is_input):
        self.column = column
        self.vocab = vocab
        self.vocab_builder = None
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.column, str)]

    def initialize(self, vocab_builder=None, from_scratch=True):
        """Build vocabulary based on training corpus."""
        if self.vocab and from_scratch:
            return
        if not self.vocab_builder:
            self.vocab_builder = vocab_builder or VocabBuilder()
            self.vocab_builder.use_unk = False
            self.vocab_builder.use_pad = False
        try:
            while True:
                row = yield
                annotation = Annotation(row[self.column])
                actions = annotation.tree.to_actions()
                self.vocab_builder.add_all(actions)
        except GeneratorExit:
            self.vocab = self.vocab_builder.make_vocab()
            self.shift_idx = self.vocab.idx[SHIFT]
            self.reduce_idx = self.vocab.idx[REDUCE]

            def filterVocab(fn):
                return [token for nt, token in self.vocab.idx.items() if fn(nt)]
            self.ignore_subNTs_roots = filterVocab(is_unsupported)
            self.valid_NT_idxs = filterVocab(is_valid_nonterminal)
            self.valid_IN_idxs = filterVocab(is_intent_nonterminal)
            self.valid_SL_idxs = filterVocab(is_slot_nonterminal)

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        annotation = Annotation(row[self.column])
        return self.vocab.lookup_all(annotation.tree.to_actions())

    def tensorize(self, batch):
        return batch


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

    def __init__(self, decoder_hidden_state_dim, context_dim, force_projection=False, src_length_masking=True):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim
        self.input_proj = None
        if force_projection or decoder_hidden_state_dim != context_dim:
            self.input_proj = nn.Linear(decoder_hidden_state_dim, context_dim, bias=True)
        self.src_length_masking = src_length_masking
        log_class_usage(__class__)

    def forward(self, decoder_state, source_hids, src_lengths):
        source_hids = source_hids.transpose(0, 1)
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        attn_scores = torch.bmm(source_hids, decoder_state.unsqueeze(2)).squeeze(2)
        normalized_masked_attn_scores = masked_softmax(attn_scores, src_lengths, self.src_length_masking)
        attn_weighted_context = (source_hids * normalized_masked_attn_scores.unsqueeze(2)).contiguous().sum(1)
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


def prepare_full_key(instance_id: str, key: str, secondary_key: Optional[str]=None):
    if secondary_key is not None:
        return instance_id + '.' + key + '.' + secondary_key
    else:
        return instance_id + '.' + key


class PyTextIncrementalDecoderComponent(PyTextSeq2SeqModule):

    def get_incremental_state(self, incremental_state: Dict[str, Tensor], key: str) ->Optional[Tensor]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = prepare_full_key(self.instance_id, key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state: Dict[str, Tensor], key: str, value):
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = prepare_full_key(self.instance_id, key)
            incremental_state[full_key] = value

    def reorder_incremental_state(self, incremental_state: Dict[str, Tensor], new_order: Tensor):
        pass


class PlaceholderIdentity(nn.Module):

    def forward(self, x, incremental_state: Optional[Dict[str, Tensor]]=None):
        return x


class PlaceholderAttentionIdentity(nn.Module):

    def forward(self, query, key, value, need_weights: bool=None, key_padding_mask: Optional[Tensor]=None, incremental_state: Optional[Dict[str, Tensor]]=None) ->Tuple[Tensor, Optional[Tensor]]:
        optional_attention: Optional[Tensor] = None
        return query, optional_attention

    def reorder_incremental_state(self, incremental_state: Dict[str, Tensor], new_order: Tensor):
        pass


class DecoderWithLinearOutputProjection(PyTextSeq2SeqModule):
    """
    Common super class for decoder networks with output projection layers.
    """

    def __init__(self, out_vocab_size, out_embed_dim=512):
        super().__init__()
        self.linear_projection = nn.Linear(out_embed_dim, out_vocab_size)
        self.reset_parameters()
        log_class_usage(__class__)

    def reset_parameters(self):
        nn.init.uniform_(self.linear_projection.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear_projection.bias)

    def forward(self, input_tokens, encoder_out: Dict[str, torch.Tensor], incremental_state: Optional[Dict[str, torch.Tensor]]=None, timestep: int=0) ->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, features = self.forward_unprojected(input_tokens, encoder_out, incremental_state)
        logits = self.linear_projection(x)
        return logits, features

    def forward_unprojected(self, input_tokens, encoder_out, incremental_state=None):
        """Forward pass through the decoder without output projection."""
        raise NotImplementedError()


class RNNDecoderBase(PyTextIncrementalDecoderComponent):
    """
    RNN decoder with multihead attention. Attention is calculated using encoder
    output and output of decoder's first RNN layerself. Attention is applied
    after first RNN layer and concatenated to input of subsequent layers.
    """


    class Config(ConfigBase):
        encoder_hidden_dim: int = 512
        embed_dim: int = 512
        hidden_dim: int = 512
        out_embed_dim: int = 512
        cell_type: str = 'lstm'
        num_layers: int = 1
        dropout_in: float = 0.1
        dropout_out: float = 0.1
        attention_type: str = 'dot'
        attention_heads: int = 8
        first_layer_attention: bool = False
        averaging_encoder: bool = False

    @classmethod
    def from_config(cls, config, out_vocab_size, target_embedding):
        return cls(out_vocab_size, target_embedding, **config._asdict())

    def __init__(self, embed_tokens, encoder_hidden_dim, embed_dim, hidden_dim, out_embed_dim, cell_type, num_layers, dropout_in, dropout_out, attention_type, attention_heads, first_layer_attention, averaging_encoder):
        encoder_hidden_dim = max(1, encoder_hidden_dim)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.attention_type = attention_type
        self.attention_heads = attention_heads
        self.first_layer_attention = first_layer_attention
        self.embed_tokens = embed_tokens
        self.hidden_dim = hidden_dim
        self.averaging_encoder = averaging_encoder
        if cell_type == 'lstm':
            cell_class = torch.nn.LSTMCell
        else:
            raise RuntimeError('Cell type not supported')
        self.change_hidden_dim = hidden_dim != encoder_hidden_dim
        if self.change_hidden_dim:
            hidden_init_fc_list = []
            cell_init_fc_list = []
            for _ in range(num_layers):
                hidden_init_fc_list.append(nn.Linear(encoder_hidden_dim, hidden_dim))
                cell_init_fc_list.append(nn.Linear(encoder_hidden_dim, hidden_dim))
            self.hidden_init_fc_list = nn.ModuleList(hidden_init_fc_list)
            self.cell_init_fc_list = nn.ModuleList(cell_init_fc_list)
        else:
            self.hidden_init_fc_list = nn.ModuleList([])
            self.cell_init_fc_list = nn.ModuleList([])
        if attention_type == 'dot':
            self.attention = DotAttention(decoder_hidden_state_dim=hidden_dim, context_dim=encoder_hidden_dim)
        else:
            raise RuntimeError(f'Attention type {attention_type} not supported')
        self.combined_output_and_context_dim = self.attention.context_dim + hidden_dim
        layers = []
        for layer in range(num_layers):
            if layer == 0:
                cell_input_dim = embed_dim
            else:
                cell_input_dim = hidden_dim
            if self.first_layer_attention or layer == 0:
                cell_input_dim += self.attention.context_dim
            layers.append(cell_class(input_size=cell_input_dim, hidden_size=hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)
        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = nn.Linear(self.combined_output_and_context_dim, out_embed_dim)
        else:
            self.additional_fc = PlaceholderIdentity()
        log_class_usage(__class__)

    def forward_unprojected(self, input_tokens, encoder_out: Dict[str, torch.Tensor], incremental_state: Optional[Dict[str, torch.Tensor]]=None) ->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if incremental_state is not None and len(incremental_state) > 0:
            input_tokens = input_tokens[:, -1:]
        bsz, seqlen = input_tokens.size()
        encoder_outs = encoder_out['unpacked_output']
        src_lengths = encoder_out['src_lengths']
        x = self.embed_tokens([[input_tokens]])
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = x.transpose(0, 1)
        cached_state = self._get_cached_state(incremental_state)
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            if incremental_state is None:
                incremental_state = {}
            self._init_prev_states(encoder_out, incremental_state)
            init_state = self._get_cached_state(incremental_state)
            assert init_state is not None
            prev_hiddens, prev_cells, input_feed = init_state
        outs = []
        attn_scores_per_step: List[torch.Tensor] = []
        next_hiddens: List[torch.Tensor] = []
        next_cells: List[torch.Tensor] = []
        for j in range(seqlen):
            step_input = torch.cat((x[(j), :, :], input_feed), dim=1)
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(step_input, (prev_hiddens[i], prev_cells[i]))
                if self.first_layer_attention and i == 0:
                    input_feed, step_attn_scores = self.attention(hidden, encoder_outs, src_lengths)
                layer_output = F.dropout(hidden, p=self.dropout_out, training=self.training)
                step_input = layer_output
                if self.first_layer_attention:
                    step_input = torch.cat((step_input, input_feed), dim=1)
                next_hiddens.append(hidden)
                next_cells.append(cell)
            if not self.first_layer_attention:
                input_feed, step_attn_scores = self.attention(hidden, encoder_outs, src_lengths)
                attn_scores_per_step.append(step_attn_scores)
            combined_output_and_context = torch.cat((hidden, input_feed), dim=1)
            outs.append(combined_output_and_context)
            prev_hiddens = torch.stack(next_hiddens, 0)
            prev_cells = torch.stack(next_cells, 0)
            next_hiddens = []
            next_cells = []
        attn_scores = torch.stack(attn_scores_per_step, dim=1)
        attn_scores = attn_scores.transpose(0, 2)
        self._set_cached_state(incremental_state, (prev_hiddens, prev_cells, input_feed))
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.combined_output_and_context_dim)
        x = x.transpose(1, 0)
        x = self.additional_fc(x)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        return x, {'attn_scores': attn_scores, 'src_tokens': encoder_out['src_tokens'], 'src_lengths': encoder_out['src_lengths']}

    def reorder_incremental_state(self, incremental_state: Dict[str, torch.Tensor], new_order):
        """Reorder buffered internal state (for incremental generation)."""
        assert incremental_state is not None
        hiddens = self.get_incremental_state(incremental_state, 'cached_hiddens')
        assert hiddens is not None
        cells = self.get_incremental_state(incremental_state, 'cached_cells')
        assert cells is not None
        feeds = self.get_incremental_state(incremental_state, 'cached_feeds')
        assert feeds is not None
        self.set_incremental_state(incremental_state, 'cached_hiddens', hiddens.index_select(1, new_order))
        self.set_incremental_state(incremental_state, 'cached_cells', cells.index_select(1, new_order))
        self.set_incremental_state(incremental_state, 'cached_feeds', feeds.index_select(0, new_order))

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(100000.0)

    def _init_prev_states(self, encoder_out: Dict[str, torch.Tensor], incremental_state: Dict[str, torch.Tensor]) ->None:
        encoder_output = encoder_out['unpacked_output']
        final_hiddens = encoder_out['final_hiddens']
        prev_cells = encoder_out['final_cells']
        if self.averaging_encoder:
            prev_hiddens = torch.stack([torch.mean(encoder_output, 0)] * self.num_layers, dim=0)
        else:
            prev_hiddens = final_hiddens
        if self.change_hidden_dim:
            transformed_hiddens: List[torch.Tensor] = []
            transformed_cells: List[torch.Tensor] = []
            i: int = 0
            for hidden_init_fc, cell_init_fc in zip(self.hidden_init_fc_list, self.cell_init_fc_list):
                transformed_hiddens.append(hidden_init_fc(prev_hiddens[i]))
                transformed_cells.append(cell_init_fc(prev_cells[i]))
                i += 1
            use_hiddens = torch.stack(transformed_hiddens, dim=0)
            use_cells = torch.stack(transformed_cells, dim=0)
        else:
            use_hiddens = prev_hiddens
            use_cells = prev_cells
        assert self.attention.context_dim
        initial_attn_context = torch.zeros(self.attention.context_dim, device=encoder_output.device)
        batch_size = encoder_output.size(1)
        self.set_incremental_state(incremental_state, 'cached_hiddens', use_hiddens)
        self.set_incremental_state(incremental_state, 'cached_cells', use_cells)
        self.set_incremental_state(incremental_state, 'cached_feeds', initial_attn_context.expand(batch_size, self.attention.context_dim))

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return fairseq_utils.log_softmax(logits, dim=-1)
        else:
            return fairseq_utils.softmax(logits, dim=-1)

    def _get_cached_state(self, incremental_state: Optional[Dict[str, torch.Tensor]]):
        if incremental_state is None or len(incremental_state) == 0:
            return None
        hiddens = self.get_incremental_state(incremental_state, 'cached_hiddens')
        assert hiddens is not None
        cells = self.get_incremental_state(incremental_state, 'cached_cells')
        assert cells is not None
        feeds = self.get_incremental_state(incremental_state, 'cached_feeds')
        assert feeds is not None
        return hiddens, cells, feeds

    def _set_cached_state(self, incremental_state: Optional[Dict[str, torch.Tensor]], state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) ->None:
        if incremental_state is None:
            return
        hiddens, cells, feeds = state
        self.set_incremental_state(incremental_state, 'cached_hiddens', hiddens)
        self.set_incremental_state(incremental_state, 'cached_cells', cells)
        self.set_incremental_state(incremental_state, 'cached_feeds', feeds)


class RNNDecoder(RNNDecoderBase, DecoderWithLinearOutputProjection):

    def __init__(self, out_vocab_size, embed_tokens, encoder_hidden_dim, embed_dim, hidden_dim, out_embed_dim, cell_type, num_layers, dropout_in, dropout_out, attention_type, attention_heads, first_layer_attention, averaging_encoder):
        DecoderWithLinearOutputProjection.__init__(self, out_vocab_size, out_embed_dim=out_embed_dim)
        RNNDecoderBase.__init__(self, embed_tokens, encoder_hidden_dim, embed_dim, hidden_dim, out_embed_dim, cell_type, num_layers, dropout_in, dropout_out, attention_type, attention_heads, first_layer_attention, averaging_encoder)
        log_class_usage(__class__)


class LSTMSequenceEncoder(PyTextSeq2SeqModule):
    """RNN encoder using nn.LSTM for cuDNN support / ONNX exportability."""


    class Config(ConfigBase):
        embed_dim: int = 512
        hidden_dim: int = 512
        num_layers: int = 1
        dropout_in: float = 0.1
        dropout_out: float = 0.1
        bidirectional: bool = False

    def __init__(self, embed_dim, hidden_dim, num_layers, dropout_in, dropout_out, bidirectional):
        super().__init__()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers: int = num_layers
        self.word_dim = embed_dim
        self.bilstm = BiLSTM(num_layers=num_layers, bidirectional=bidirectional, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout_out)
        log_class_usage(__class__)

    @classmethod
    def from_config(cls, config):
        return cls(**config._asdict())

    def forward(self, src_tokens: torch.Tensor, embeddings: torch.Tensor, src_lengths) ->Dict[str, torch.Tensor]:
        x = F.dropout(embeddings, p=self.dropout_in, training=self.training)
        x = x.transpose(0, 1)
        unpacked_output, final_hiddens, final_cells = self.bilstm(embeddings=x, lengths=src_lengths)
        return {'unpacked_output': unpacked_output, 'final_hiddens': final_hiddens, 'final_cells': final_cells, 'src_lengths': src_lengths, 'src_tokens': src_tokens, 'embeddings': embeddings}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(100000.0)

    def tile_encoder_out(self, beam_size: int, encoder_out: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        tiled_encoder_out = encoder_out['unpacked_output'].expand(-1, beam_size, -1)
        hiddens = encoder_out['final_hiddens']
        tiled_hiddens: List[torch.Tensor] = []
        for i in range(self.num_layers):
            tiled_hiddens.append(hiddens[i].expand(beam_size, -1))
        cells = encoder_out['final_cells']
        tiled_cells: List[torch.Tensor] = []
        for i in range(self.num_layers):
            tiled_cells.append(cells[i].expand(beam_size, -1))
        return {'unpacked_output': tiled_encoder_out, 'final_hiddens': torch.stack(tiled_hiddens, dim=0), 'final_cells': torch.stack(tiled_cells, dim=0), 'src_lengths': encoder_out['src_lengths'], 'src_tokens': encoder_out['src_tokens']}


class RNNModel(PyTextSeq2SeqModule):


    class Config(ConfigBase):
        encoder: LSTMSequenceEncoder.Config = LSTMSequenceEncoder.Config()
        decoder: RNNDecoder.Config = RNNDecoder.Config()

    def __init__(self, encoder, decoder, source_embeddings):
        super().__init__()
        self.source_embeddings = source_embeddings
        self.encoder = encoder
        self.decoder = decoder
        log_class_usage(__class__)

    def forward(self, src_tokens: torch.Tensor, additional_features: List[List[torch.Tensor]], src_lengths, prev_output_tokens, incremental_state: Optional[Dict[str, torch.Tensor]]=None):
        embeddings = self.source_embeddings([[src_tokens]] + additional_features)
        encoder_out = self.encoder(src_tokens, embeddings, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, incremental_state)
        return decoder_out

    @classmethod
    def from_config(cls, config: Config, source_vocab, source_embedding, target_vocab, target_embedding):
        out_vocab_size = len(target_vocab)
        encoder = create_module(config.encoder)
        decoder = create_module(config.decoder, out_vocab_size, target_embedding)
        return cls(encoder, decoder, source_embedding)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    def max_decoder_positions(self):
        return max(self.encoder.max_positions(), self.decoder.max_positions())


class ByteTokenTensorizer(Tensorizer):
    """Turn words into 2-dimensional tensors of int8 bytes. Words are padded to
    `max_byte_len`. Also computes sequence lengths (1-D tensor) and token lengths
    (2-D tensor). 0 is the pad byte.
    """
    NUM_BYTES = 256


    class Config(Tensorizer.Config):
        column: str = 'text'
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        max_seq_len: Optional[int] = None
        max_byte_len: int = 15
        offset_for_non_padding: int = 0
        add_bos_token: bool = False
        add_eos_token: bool = False
        use_eos_token_for_bos: bool = False

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(text_column=config.column, tokenizer=tokenizer, max_seq_len=config.max_seq_len, max_byte_len=config.max_byte_len, offset_for_non_padding=config.offset_for_non_padding, add_bos_token=config.add_bos_token, add_eos_token=config.add_eos_token, use_eos_token_for_bos=config.use_eos_token_for_bos, is_input=config.is_input)

    def __init__(self, text_column, tokenizer=None, max_seq_len=Config.max_seq_len, max_byte_len=Config.max_byte_len, offset_for_non_padding=Config.offset_for_non_padding, add_bos_token=Config.add_bos_token, add_eos_token=Config.add_eos_token, use_eos_token_for_bos=Config.use_eos_token_for_bos, is_input=Config.is_input):
        self.text_column = text_column
        self.tokenizer = tokenizer or Tokenizer()
        self.max_seq_len = max_seq_len or 2 ** 30
        self.max_byte_len = max_byte_len
        self.offset_for_non_padding = offset_for_non_padding
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.text_column, str)]

    def numberize(self, row):
        """Convert text to bytes, pad batch."""
        tokens = self.tokenizer.tokenize(row[self.text_column])[:self.max_seq_len - self.add_bos_token - self.add_eos_token]
        if self.add_bos_token:
            bos = EOS if self.use_eos_token_for_bos else BOS
            tokens = [Token(bos, -1, -1)] + tokens
        if self.add_eos_token:
            tokens.append(Token(EOS, -1, -1))
        if not tokens:
            tokens = [Token(PAD, -1, -1)]
        bytes = [self._numberize_token(token)[:self.max_byte_len] for token in tokens]
        token_lengths = len(tokens)
        byte_lengths = [len(token_bytes) for token_bytes in bytes]
        return bytes, token_lengths, byte_lengths

    def _numberize_token(self, token):
        return [(c + self.offset_for_non_padding) for c in token.value.encode()]

    def tensorize(self, batch, pad_token=0):
        bytes, token_lengths, byte_lengths = zip(*batch)
        pad_shape = len(batch), precision.pad_length(max(len(l) for l in byte_lengths)), self.max_byte_len
        return pad_and_tensorize(bytes, pad_shape=pad_shape, pad_token=pad_token), pad_and_tensorize(token_lengths), pad_and_tensorize(byte_lengths)

    def sort_key(self, row):
        return len(row[0])


class ContextualTokenEmbedding(EmbeddingBase):
    """Module for providing token embeddings from a pretrained model."""
    Config = ContextualTokenEmbeddingConfig

    @classmethod
    def from_config(cls, config: ContextualTokenEmbeddingConfig, *args, **kwargs):
        return cls(config.embed_dim)

    def forward(self, embedding: torch.Tensor) ->torch.Tensor:
        embedding_shape = torch.onnx.operators.shape_as_tensor(embedding)
        if embedding_shape[1].item() % self.embedding_dim != 0:
            raise ValueError(f'Input embedding_dim {embedding_shape[1]} is not a' + f' multiple of specified embedding_dim {self.embedding_dim}')
        num_tokens = embedding_shape[1] // self.embedding_dim
        new_embedding_shape = torch.cat((torch.tensor([-1], dtype=torch.long), num_tokens.view(1), torch.tensor([self.embedding_dim], dtype=torch.long)))
        return torch.onnx.operators.reshape_from_tensor_shape(embedding, new_embedding_shape)


class DictEmbedding(EmbeddingBase):
    """
    Module for dictionary feature embeddings for tokens. Dictionary features are
    also known as gazetteer features. These are per token discrete features that
    the module learns embeddings for.
    Example: For the utterance *Order coffee from Starbucks*, the dictionary
    features could be
    ::

        [
            {"tokenIdx": 1, "features": {"drink/beverage": 0.8, "music/song": 0.2}},
            {"tokenIdx": 3, "features": {"store/coffee_shop": 1.0}}
        ]

    ::
    Thus, for a given token there can be more than one dictionary features each
    of which has a confidence score. The final embedding for a token is the
    weighted average of the dictionary embeddings followed by a pooling operation
    such that the module produces an embedding vector per token.

    Args:
        num_embeddings (int): Total number of dictionary features (vocabulary size).
        embed_dim (int): Size of embedding vector.
        pooling_type (PoolingType): Type of pooling for combining the dictionary
            feature embeddings.

    Attributes:
        pooling_type (PoolingType): Type of pooling for combining the dictionary
            feature embeddings.

    """
    Config = DictFeatConfig

    @classmethod
    def from_config(cls, config: DictFeatConfig, metadata: Optional[FieldMeta]=None, labels: Optional[Vocabulary]=None, tensorizer: Optional[Tensorizer]=None):
        """Factory method to construct an instance of DictEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (DictFeatConfig): Configuration object specifying all the
            parameters of DictEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of DictEmbedding.

        """
        vocab_size = len(tensorizer.vocab) if tensorizer is not None else len(labels) if labels is not None else metadata.vocab_size
        tensorizer_vocab_exists = tensorizer and tensorizer.vocab
        pad_index = tensorizer.vocab.get_pad_index() if tensorizer_vocab_exists else PAD_INDEX
        unk_index = tensorizer.vocab.get_unk_index() if tensorizer_vocab_exists else UNK_INDEX
        return cls(num_embeddings=vocab_size, embed_dim=config.embed_dim, pooling_type=config.pooling, pad_index=pad_index, unk_index=unk_index, mobile=config.mobile)

    def __init__(self, num_embeddings: int, embed_dim: int, pooling_type: PoolingType, pad_index: int=PAD_INDEX, unk_index: int=UNK_INDEX, mobile: bool=False) ->None:
        super().__init__(embed_dim)
        self.unk_index = unk_index
        self.pad_index = pad_index
        self.embedding = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.pad_index)
        self.pooling_type = str(pooling_type)
        self.mobile = mobile
        log_class_usage(__class__)

    def find_and_replace(self, tensor: torch.Tensor, find_val: int, replace_val: int) ->torch.Tensor:
        """
        `torch.where` is not supported for mobile ONNX, this hack allows a mobile
        exported version of `torch.where` which is computationally more expensive
        """
        if self.mobile:
            mask = torch.eq(tensor, find_val)
            return tensor * (1 - mask.long()) + mask * replace_val
        else:
            return torch.where(tensor == find_val, torch.full_like(tensor, replace_val), tensor)

    def forward(self, feats: torch.Tensor, weights: torch.Tensor, lengths: torch.Tensor) ->torch.Tensor:
        """Given a batch of sentences such containing dictionary feature ids per
        token, produce token embedding vectors for each sentence in the batch.

        Args:
            feats (torch.Tensor): Batch of sentences with dictionary feature ids.
                shape: [bsz, seq_len * max_feat_per_token]
            weights (torch.Tensor): Batch of sentences with dictionary feature
                weights for the dictionary features.
                shape: [bsz, seq_len * max_feat_per_token]
            lengths (torch.Tensor): Batch of sentences with the number of
                dictionary features per token.
                shape: [bsz, seq_len]

        Returns:
            torch.Tensor: Embedded batch of sentences. Dimension:
            batch size X maximum sentence length, token embedding size.
            Token embedding size = `embed_dim` passed to the constructor.

        """
        batch_size = torch.onnx.operators.shape_as_tensor(feats)[0]
        max_toks = torch.onnx.operators.shape_as_tensor(lengths)[1]
        if self.unk_index != self.pad_index:
            feats = self.find_and_replace(feats, self.unk_index, self.pad_index)
        dict_emb = self.embedding(feats)
        weighted_embds = dict_emb * weights.unsqueeze(2)
        new_emb_shape = torch.cat((batch_size.view(1), max_toks.view(1), torch.tensor([-1]).long(), torch.tensor([weighted_embds.size()[-1]]).long()))
        weighted_embds = torch.onnx.operators.reshape_from_tensor_shape(weighted_embds, new_emb_shape)
        if self.pooling_type == 'mean':
            reduced_embeds = torch.sum(weighted_embds, dim=2) / lengths.unsqueeze(2).float()
        else:
            reduced_embeds, _ = torch.max(weighted_embds, dim=2)
        return reduced_embeds


Gazetteer = List[Dict[str, Dict[str, float]]]


class GazetteerTensorizer(Tensorizer):
    """
    Create 3 tensors for dict features.

    - idx: index of feature in token order.
    - weights: weight of feature in token order.
    - lens: number of features per token.

    For each input token, there will be the same number of `idx` and `weights` entries.
    (equal to the max number of features any token has in this row). The values
    in `lens` will tell how many of these features are actually used per token.

    Input format for the dict column is json and should be a list of dictionaries
    containing the "features" and their weight for each relevant "tokenIdx". Example:
    ::

        text: "Order coffee from Starbucks please"
        dict: [
            {"tokenIdx": 1, "features": {"drink/beverage": 0.8, "music/song": 0.2}},
            {"tokenIdx": 3, "features": {"store/coffee_shop": 1.0}}
        ]

    if we assume this vocab
    ::

        vocab = {
            UNK: 0, PAD: 1,
            "drink/beverage": 2, "music/song": 3, "store/coffee_shop": 4
        }

    this example will result in those tensors:
    ::

        idx =     [1,   1,   2,   3,   1,   1,   4,   1,   1,   1]
        weights = [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        lens =    [1,        2,        1,        1,        1]

    """


    class Config(Tensorizer.Config):
        text_column: str = 'text'
        dict_column: str = 'dict'
        tokenizer: Tokenizer.Config = Tokenizer.Config()

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(config.text_column, config.dict_column, tokenizer, config.is_input)

    def __init__(self, text_column: str=Config.text_column, dict_column: str=Config.dict_column, tokenizer: Tokenizer=None, is_input: bool=Config.is_input):
        self.text_column = text_column
        self.dict_column = dict_column
        self.tokenizer = tokenizer or Tokenizer()
        self.vocab_builder = VocabBuilder()
        self.vocab = None
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.text_column, str), (self.dict_column, Gazetteer)]

    def initialize(self, from_scratch=True):
        """
        Look through the dataset for all dict features to create vocab.
        """
        if self.vocab and from_scratch:
            return
        try:
            while True:
                row = yield
                for token_dict in row[self.dict_column]:
                    self.vocab_builder.add_all(token_dict['features'])
        except GeneratorExit:
            self.vocab = self.vocab_builder.make_vocab()

    def numberize(self, row):
        """
        Numberize dict features. Fill in for tokens with no features with
        PAD and weight 0.0. All tokens need to have at least one entry.
        Tokens with more than one feature will have multiple idx and weight
        added in sequence.
        """
        num_tokens = len(self.tokenizer.tokenize(row[self.text_column]))
        num_labels = max(len(t['features']) for t in row[self.dict_column])
        res_idx = [self.vocab.get_pad_index()] * (num_labels * num_tokens)
        res_weights = [0.0] * (num_labels * num_tokens)
        res_lens = [1] * num_tokens
        for dict_feature in row[self.dict_column]:
            idx = dict_feature['tokenIdx']
            feats = dict_feature['features']
            pos = idx * num_labels
            res_lens[idx] = len(feats)
            for label, weight in feats.items():
                res_idx[pos] = self.vocab.lookup_all(label)
                res_weights[pos] = weight
                pos += 1
        return res_idx, res_weights, res_lens

    def tensorize(self, batch):
        feats, weights, lengths = zip(*batch)
        lengths_flattened = [l for l_list in lengths for l in l_list]
        seq_lens = [len(l_list) for l_list in lengths]
        max_ex_len = max(seq_lens)
        max_feat_len = max(lengths_flattened)
        all_lengths, all_feats, all_weights = [], [], []
        for i, seq_len in enumerate(seq_lens):
            ex_feats, ex_weights, ex_lengths = [], [], []
            feats_lengths, feats_vals, feats_weights = lengths[i], feats[i], weights[i]
            max_feat_len_example = max(feats_lengths)
            r_offset = 0
            for _ in feats_lengths:
                ex_feats.extend(feats_vals[r_offset:r_offset + max_feat_len_example])
                ex_feats.extend([self.vocab.get_pad_index()] * (max_feat_len - max_feat_len_example))
                ex_weights.extend(feats_weights[r_offset:r_offset + max_feat_len_example])
                ex_weights.extend([0.0] * (max_feat_len - max_feat_len_example))
                r_offset += max_feat_len_example
            ex_lengths.extend(feats_lengths)
            ex_padding = (max_ex_len - seq_len) * max_feat_len
            ex_feats.extend([self.vocab.get_pad_index()] * ex_padding)
            ex_weights.extend([0.0] * ex_padding)
            ex_lengths.extend([1] * (max_ex_len - seq_len))
            all_feats.append(ex_feats)
            all_weights.append(ex_weights)
            all_lengths.append(ex_lengths)
        return cuda.tensor(all_feats, torch.long), cuda.tensor(all_weights, torch.float), cuda.tensor(all_lengths, torch.long)


def GetTensor(tensor):
    if CUDA_ENABLED:
        return tensor
    else:
        return tensor


class BeamDecode(torch.nn.Module):
    """
    Decodes the output of Beam Search to get the top hypotheses
    """

    def __init__(self, eos_token_id, length_penalty, nbest, beam_size, stop_at_eos):
        super().__init__()
        self.eos_token_id: int = eos_token_id
        self.length_penalty: float = length_penalty
        self.nbest: int = nbest
        self.beam_size: int = beam_size
        self.stop_at_eos: bool = stop_at_eos

    @torch.no_grad()
    def forward(self, beam_tokens: Tensor, beam_scores: Tensor, token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int) ->List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:
        self._check_dimensions(beam_tokens, beam_scores, token_weights, beam_prev_indices, num_steps)
        end_states = self._get_all_end_states(beam_tokens, beam_scores, beam_prev_indices, num_steps)
        outputs = torch.jit.annotate(List[Tuple[Tensor, float, List[float], Tensor, Tensor]], [])
        for state_idx in range(len(end_states)):
            state = end_states[state_idx]
            hypothesis_score = float(state[0])
            beam_indices = self._get_output_steps_to_beam_indices(state, beam_prev_indices)
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
                    token_level_scores.append(float(beam_scores[pos][beam_index]))
                else:
                    token_level_scores.append(float(beam_scores[pos][beam_index]) - float(beam_scores[pos - 1][prev_beam_index]))
                back_alignment_weights.append(token_weights[pos][beam_index].detach())
                prev_beam_index = beam_index
                pos += 1
            outputs.append((torch.stack(beam_output), hypothesis_score, token_level_scores, torch.stack(back_alignment_weights, dim=1), best_indices))
        return outputs

    def _get_output_steps_to_beam_indices(self, end_state: Tensor, beam_prev_indices: Tensor) ->List[int]:
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

    def _add_to_end_states(self, end_states: List[Tensor], min_score: float, state: Tensor, min_index: int) ->Tuple[List[Tensor], float, int]:
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

    def _get_all_end_states(self, beam_tokens: Tensor, beam_scores: Tensor, beam_prev_indices: Tensor, num_steps: int) ->Tensor:
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
                    if beam_tokens[position][hyp_index] == self.eos_token_id or position == num_steps:
                        if self.stop_at_eos:
                            hypo_is_finished[hyp_index] = 1
                        hypo_score = float(beam_scores[position][hyp_index])
                        if self.length_penalty != 0:
                            hypo_score = hypo_score / position ** self.length_penalty
                        end_states, min_score, min_index = self._add_to_end_states(end_states, min_score, torch.tensor([hypo_score, float(position), float(hyp_index)]), min_index)
            prev_hypo_is_finished = hypo_is_finished
            position = position + 1
        end_states = torch.stack(end_states)
        _, sorted_end_state_indices = end_states[:, (0)].sort(dim=0, descending=True)
        end_states = end_states[(sorted_end_state_indices), :]
        return end_states

    def _check_dimensions(self, beam_tokens: Tensor, beam_scores: Tensor, token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int) ->None:
        assert beam_tokens.size(1) == self.beam_size, 'Dimension of beam_tokens : {} and beam size : {} are not consistent'.format(beam_tokens.size(), self.beam_size)
        assert beam_scores.size(1) == self.beam_size, 'Dimension of beam_scores : {} and beam size : {} are not consistent'.format(beam_scores.size(), self.beam_size)
        assert token_weights.size(1) == self.beam_size, 'Dimension of token_weights : {} and beam size : {} are not consistent'.format(token_weights.size(), self.beam_size)
        assert beam_prev_indices.size(1) == self.beam_size, 'Dimension of beam_prev_indices : {} and beam size : {} '
        """are not consistent""".format(beam_prev_indices.size(), self.beam_size)
        assert beam_tokens.size(0) <= num_steps + 1, 'Dimension of beam_tokens : {} and num_steps : {} are not consistent'.format(beam_tokens.size(), num_steps)
        assert beam_scores.size(0) <= num_steps + 1, 'Dimension of beam_scores : {} and num_steps : {} are not consistent'.format(beam_scores.size(), num_steps)
        assert token_weights.size(0) <= num_steps + 1, 'Dimension of token_weights : {} and num_steps : {} are not consistent'.format(token_weights.size(), num_steps)
        assert beam_prev_indices.size(0) <= num_steps + 1, 'Dimension of beam_prev_indices : {} and num_steps : {} are not consistent'.format(beam_prev_indices.size(), num_steps)


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

    def forward(self, prev_tokens: Tensor, prev_scores: Tensor, timestep: int, decoder_ips: List[Dict[str, Tensor]]) ->Tuple[Tensor, Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]:
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """
        prev_tokens = prev_tokens.unsqueeze(1)
        log_probs_per_model = torch.jit.annotate(List[Tensor], [])
        attn_weights_per_model = torch.jit.annotate(List[Tensor], [])
        futures = torch.jit.annotate(List[Tuple[Tensor, Dict[str, Tensor]]], [])
        for idx, model in enumerate(self.models):
            decoder_ip = decoder_ips[idx]
            incremental_state = self.incremental_states[str(idx)]
            fut = model.decoder(prev_tokens, decoder_ip, incremental_state=incremental_state, timestep=timestep)
            futures.append(fut)
        for idx, _model in enumerate(self.models):
            fut = futures[idx]
            log_probs, features = fut
            log_probs_per_model.append(log_probs)
            if 'attn_scores' in features:
                attn_weights_per_model.append(features['attn_scores'])
        best_scores, best_tokens, prev_hypos, attention_weights = self.beam_search_aggregate_topk(log_probs_per_model, attn_weights_per_model, prev_scores, self.beam_size, self.record_attention)
        for model_state_ptr, model in enumerate(self.models):
            incremental_state = self.incremental_states[str(model_state_ptr)]
            model.decoder.reorder_incremental_state(incremental_state, prev_hypos)
        return best_tokens, best_scores, prev_hypos, attention_weights, decoder_ips

    def beam_search_aggregate_topk(self, log_probs_per_model: List[torch.Tensor], attn_weights_per_model: List[torch.Tensor], prev_scores: torch.Tensor, beam_size: int, record_attention: bool):
        average_log_probs = torch.mean(torch.cat(log_probs_per_model, dim=1), dim=1, keepdim=True)
        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(average_log_probs.squeeze(1), k=beam_size)
        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)
        best_scores, best_indices = torch.topk(total_scores_flat, k=beam_size)
        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices).view(-1)
        prev_hypos = best_indices // beam_size
        if record_attention:
            average_attn_weights = torch.mean(torch.cat(attn_weights_per_model, dim=1), dim=1, keepdim=True)
            attention_weights = average_attn_weights.index_select(dim=0, index=prev_hypos)
            attention_weights = attention_weights.squeeze_(1)
        else:
            attention_weights = torch.zeros(beam_size, attn_weights_per_model[0].size(2))
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

    def forward(self, src_tokens: Tensor, src_lengths: Tensor, dict_feat: Optional[Tuple[Tensor, Tensor, Tensor]]=None, contextual_token_embedding: Optional[Tensor]=None) ->List[Dict[str, Tensor]]:
        src_tokens_seq_first = src_tokens.t()
        futures = torch.jit.annotate(List[Dict[str, Tensor]], [])
        for model in self.models:
            embedding_input = [[src_tokens_seq_first]]
            if dict_feat is not None:
                embedding_input.append(list(dict_feat))
            if contextual_token_embedding is not None:
                embedding_input.append([contextual_token_embedding])
            embeddings = model.source_embeddings(embedding_input)
            futures.append(model.encoder(src_tokens_seq_first, embeddings, src_lengths))
        return self.prepare_decoderstep_ip(futures)

    def prepare_decoderstep_ip(self, futures: List[Dict[str, Tensor]]) ->List[Dict[str, Tensor]]:
        outputs = torch.jit.annotate(List[Dict[str, Tensor]], [])
        for idx, model in enumerate(self.models):
            encoder_out = futures[idx]
            tiled_encoder_out = model.encoder.tile_encoder_out(self.beam_size, encoder_out)
            outputs.append(tiled_encoder_out)
        return outputs


@torch.jit.script
def get_first_decoder_step_input(beam_size: int=5, eos_token_id: int=0, src_length: int=1) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prev_tokens = torch.full([beam_size], eos_token_id, dtype=torch.long)
    prev_scores = torch.full([beam_size], 1, dtype=torch.float)
    prev_hypos = torch.full([beam_size], 0, dtype=torch.long)
    attention_weights = torch.full([beam_size, src_length], 1, dtype=torch.float)
    return prev_tokens, prev_scores, prev_hypos, attention_weights


class BeamSearch(nn.Module):

    def __init__(self, model_list, tgt_dict_eos, beam_size: int=2, quantize: bool=False, record_attention: bool=False):
        super().__init__()
        self.models = model_list
        self.target_dict_eos = tgt_dict_eos
        self.beam_size = beam_size
        self.record_attention = record_attention
        encoder_ens = EncoderEnsemble(self.models, self.beam_size)
        if quantize:
            encoder_ens = torch.quantization.quantize_dynamic(encoder_ens, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.encoder_ens = torch.jit.script(encoder_ens)
        decoder_ens = DecoderBatchedStepEnsemble(self.models, beam_size, record_attention=record_attention)
        if quantize:
            decoder_ens = torch.quantization.quantize_dynamic(decoder_ens, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.decoder_ens = torch.jit.script(decoder_ens)

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor, num_steps: int, dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None, contextual_token_embedding: Optional[torch.Tensor]=None):
        self.decoder_ens.reset_incremental_states()
        decoder_ip = self.encoder_ens(src_tokens, src_lengths, dict_feat, contextual_token_embedding)
        prev_token, prev_scores, prev_hypos_indices, attention_weights = get_first_decoder_step_input(self.beam_size, self.target_dict_eos, src_lengths[0])
        all_tokens_list = [prev_token]
        all_scores_list = [prev_scores]
        all_prev_indices_list = [prev_hypos_indices]
        all_attentions_list: List[torch.Tensor] = []
        if self.record_attention:
            all_attentions_list.append(attention_weights)
        for i in range(num_steps):
            prev_token, prev_scores, prev_hypos_indices, attention_weights, decoder_ip = self.decoder_ens(prev_token, prev_scores, i + 1, decoder_ip)
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
            all_attn_weights = torch.zeros(num_steps + 1, self.beam_size, src_tokens.size(1))
        return all_tokens, all_scores, all_attn_weights, all_prev_indices


@torch.jit.script
def get_target_length(src_len: int, targetlen_cap: int, targetlen_a: float, targetlen_b: float, targetlen_c: float) ->int:
    target_length = int(min(targetlen_cap, src_len * targetlen_a * targetlen_a + src_len * targetlen_b + targetlen_c))
    assert target_length > 0, 'Target length cannot be less than 0 src_len:' + str(src_len) + ' target_length:' + str(target_length)
    return target_length


class ScriptedSequenceGenerator(Module):


    class Config(ConfigBase):
        beam_size: int = 2
        targetlen_cap: int = 100
        targetlen_a: float = 0
        targetlen_b: float = 2
        targetlen_c: float = 2
        quantize: bool = True
        length_penalty: float = 0.25
        nbest: int = 2
        stop_at_eos: bool = True
        record_attention: bool = False

    @classmethod
    def from_config(cls, config, models, trg_dict_eos):
        return cls(models, trg_dict_eos, config)

    def __init__(self, models, trg_dict_eos, config):
        super().__init__()
        self.targetlen_cap = config.targetlen_cap
        self.targetlen_a: float = float(config.targetlen_a)
        self.targetlen_b: float = float(config.targetlen_b)
        self.targetlen_c: float = float(config.targetlen_c)
        self.beam_search = BeamSearch(models, trg_dict_eos, beam_size=config.beam_size, quantize=config.quantize, record_attention=config.record_attention)
        self.beam_decode = BeamDecode(eos_token_id=trg_dict_eos, length_penalty=config.length_penalty, nbest=config.nbest, beam_size=config.beam_size, stop_at_eos=config.stop_at_eos)

    def forward(self, src_tokens: torch.Tensor, dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], contextual_token_embedding: Optional[torch.Tensor], src_lengths: torch.Tensor) ->List[Tuple[torch.Tensor, float, List[float], torch.Tensor, torch.Tensor]]:
        target_length = get_target_length(src_lengths.item(), self.targetlen_cap, self.targetlen_a, self.targetlen_b, self.targetlen_c)
        all_tokens, all_scores, all_weights, all_prev_indices = self.beam_search(src_tokens, src_lengths, target_length, dict_feat, contextual_token_embedding)
        return self.beam_decode(all_tokens, all_scores, all_weights, all_prev_indices, target_length)

    @torch.jit.export
    def generate_hypo(self, tensors: Dict[str, torch.Tensor]):
        actual_src_tokens = tensors['src_tokens'].t()
        dict_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if 'dict_tokens' in tensors:
            dict_feat = tensors['dict_tokens'], tensors['dict_weights'], tensors['dict_lengths']
        contextual_token_embedding: Optional[torch.Tensor] = None
        if 'contextual_token_embedding' in tensors:
            contextual_token_embedding = tensors['contextual_token_embedding']
        hypos_etc = self.forward(actual_src_tokens, dict_feat, contextual_token_embedding, tensors['src_lengths'])
        predictions = [[pred for pred, _, _, _, _ in hypos_etc]]
        scores = [[score for _, score, _, _, _ in hypos_etc]]
        return predictions, scores


@torch.jit.script
def get_single_unk_token(src_tokens: List[str], word_ids: List[int], copy_unk_token: bool, unk_idx: int):
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

    def __init__(self, src_dict, tgt_dict, sequence_generator, filter_eos_bos, copy_unk_token=False, dictfeat_dict=None):
        super().__init__()
        self.source_vocab = ScriptVocabulary(src_dict._vocab, src_dict.get_unk_index(), bos_idx=src_dict.get_bos_index(-1), eos_idx=src_dict.get_eos_index(-1))
        self.target_vocab = ScriptVocabulary(tgt_dict._vocab, tgt_dict.get_unk_index(), bos_idx=tgt_dict.get_bos_index(), eos_idx=tgt_dict.get_eos_index())
        if dictfeat_dict:
            self.dictfeat_vocab = ScriptVocabulary(dictfeat_dict._vocab, pad_idx=dictfeat_dict.idx[src_dict[src_dict.get_pad_index()]])
        else:
            self.dictfeat_vocab = ScriptVocabulary([])
        self.sequence_generator = sequence_generator
        self.copy_unk_token: bool = copy_unk_token
        self.unk_idx: int = self.source_vocab.unk_idx
        self.filter_eos_bos: bool = filter_eos_bos

    def prepare_generator_inputs(self, word_ids: List[int], dict_feat: Optional[Tuple[List[str], List[float], List[int]]]=None, contextual_token_embedding: Optional[List[float]]=None) ->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Optional[torch.Tensor], torch.Tensor]:
        src_len = len(word_ids)
        dict_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if dict_feat is not None:
            dict_tokens, dict_weights, dict_lengths = dict_feat
            dict_ids = self.dictfeat_vocab.lookup_indices_1d(dict_tokens)
            dict_tensors = torch.tensor([dict_ids]), torch.tensor([dict_weights], dtype=torch.float), torch.tensor([dict_lengths])
        contextual_embedding_tensor: Optional[torch.Tensor] = None
        if contextual_token_embedding is not None:
            assert len(contextual_token_embedding) % src_len == 0 and len(contextual_token_embedding) > 0, f'Incorrect size for contextual embeddings: {len(contextual_token_embedding)}, Expected a non-zero multiple of input token count {src_len} '
            contextual_embedding_tensor = torch.tensor([contextual_token_embedding], dtype=torch.float)
        return torch.tensor(word_ids).reshape(-1, 1), dict_tensors, contextual_embedding_tensor, torch.tensor([src_len])

    def forward(self, src_tokens: List[str], dict_feat: Optional[Tuple[List[str], List[float], List[int]]]=None, contextual_token_embedding: Optional[List[float]]=None) ->List[Tuple[List[str], float, List[float]]]:
        word_ids = self.source_vocab.lookup_indices_1d(src_tokens)
        single_unk_token: Optional[str] = get_single_unk_token(src_tokens, word_ids, self.copy_unk_token, self.unk_idx)
        words, dict_tensors, contextual_embedding_tensor, src_lengths = self.prepare_generator_inputs(word_ids, dict_feat, contextual_token_embedding)
        hypos_etc = self.sequence_generator(words, dict_tensors, contextual_embedding_tensor, src_lengths)
        hypos_list: List[Tuple[List[str], float, List[float]]] = []
        filter_token_list: List[int] = []
        if self.filter_eos_bos:
            filter_token_list = [self.target_vocab.bos_idx, self.target_vocab.eos_idx]
        for seq in hypos_etc:
            hyopthesis = seq[0]
            stringified = self.target_vocab.lookup_words_1d(hyopthesis, filter_token_list=filter_token_list, possible_unk_token=single_unk_token)
            hypos_list.append((stringified, seq[1], seq[2]))
        return hypos_list


class ScriptModule(torch.jit.ScriptModule):

    @torch.jit.script_method
    def set_device(self, device: str):
        self.tensorizer.set_device(device)


@torch.jit.script
def squeeze_1d(inputs: Optional[List[str]]) ->Optional[List[List[str]]]:
    result: Optional[List[List[str]]] = None
    if inputs is not None:
        result = torch.jit.annotate(List[List[str]], [])
        for line in inputs:
            result.append([line])
    return result


@torch.jit.script
def resolve_texts(texts: Optional[List[str]]=None, multi_texts: Optional[List[List[str]]]=None) ->Optional[List[List[str]]]:
    if texts is not None:
        return squeeze_1d(texts)
    return multi_texts


@torch.jit.script
def squeeze_2d(inputs: Optional[List[List[str]]]) ->Optional[List[List[List[str]]]]:
    result: Optional[List[List[List[str]]]] = None
    if inputs is not None:
        result = torch.jit.annotate(List[List[List[str]]], [])
        for line in inputs:
            result.append([line])
    return result


class ScriptPyTextModule(ScriptModule):

    def __init__(self, model: torch.jit.ScriptModule, output_layer: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__()
        self.model = model
        self.output_layer = output_layer
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def forward(self, texts: Optional[List[str]]=None, multi_texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[str]]]=None, languages: Optional[List[str]]=None):
        inputs: ScriptBatchInput = ScriptBatchInput(texts=resolve_texts(texts, multi_texts), tokens=squeeze_2d(tokens), languages=squeeze_1d(languages))
        input_tensors = self.tensorizer(inputs)
        logits = self.model(input_tensors)
        return self.output_layer(logits)


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
        return {'num_rows': self.num_rows, 'feature_sums': self.feature_sums, 'feature_squared_sums': self.feature_squared_sums, 'do_normalization': self.do_normalization, 'feature_avgs': self.feature_avgs, 'feature_stddevs': self.feature_stddevs}

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
            self.feature_avgs = [(x / self.num_rows) for x in self.feature_sums]
            self.feature_stddevs = [((self.feature_squared_sums[i] / self.num_rows - self.feature_avgs[i] ** 2) ** 0.5) for i in range(len(self.feature_squared_sums))]

    def normalize(self, vec: List[List[float]]):
        if self.do_normalization:
            for i in range(len(vec)):
                for j in range(len(vec[i])):
                    vec[i][j] -= self.feature_avgs[j]
                    vec[i][j] /= self.feature_stddevs[j] if self.feature_stddevs[j] != 0 else 1.0
        return vec


class ScriptPyTextModuleWithDense(ScriptPyTextModule):

    def __init__(self, model: torch.jit.ScriptModule, output_layer: torch.jit.ScriptModule, tensorizer: ScriptTensorizer, normalizer: VectorNormalizer):
        super().__init__(model, output_layer, tensorizer)
        self.normalizer = normalizer

    @torch.jit.script_method
    def forward(self, dense_feat: List[List[float]], texts: Optional[List[str]]=None, multi_texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[str]]]=None, languages: Optional[List[str]]=None):
        inputs: ScriptBatchInput = ScriptBatchInput(texts=resolve_texts(texts, multi_texts), tokens=squeeze_2d(tokens), languages=squeeze_1d(languages))
        input_tensors = self.tensorizer(inputs)
        dense_feat = self.normalizer.normalize(dense_feat)
        logits = self.model(input_tensors, torch.tensor(dense_feat, dtype=torch.float))
        return self.output_layer(logits)


class ScriptPyTextEmbeddingModule(ScriptModule):

    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer):
        super().__init__()
        self.model = model
        self.tensorizer = tensorizer

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors).cpu()

    @torch.jit.script_method
    def forward(self, texts: Optional[List[str]]=None, multi_texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[str]]]=None, languages: Optional[List[str]]=None, dense_feat: Optional[List[List[float]]]=None) ->torch.Tensor:
        inputs: ScriptBatchInput = ScriptBatchInput(texts=resolve_texts(texts, multi_texts), tokens=squeeze_2d(tokens), languages=squeeze_1d(languages))
        return self._forward(inputs)


class ScriptPyTextEmbeddingModuleIndex(ScriptPyTextEmbeddingModule):

    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer, index: int=0):
        super().__init__(model, tensorizer)
        self.index = torch.jit.Attribute(index, int)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput):
        input_tensors = self.tensorizer(inputs)
        return self.model(input_tensors)[self.index].cpu()


class ScriptPyTextEmbeddingModuleWithDense(ScriptPyTextEmbeddingModule):

    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer, normalizer: VectorNormalizer, concat_dense: bool=False):
        super().__init__(model, tensorizer)
        self.normalizer = normalizer
        self.concat_dense = torch.jit.Attribute(concat_dense, bool)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != '':
            dense_tensor = dense_tensor
        return self.model(input_tensors, dense_tensor).cpu()

    @torch.jit.script_method
    def forward(self, texts: Optional[List[str]]=None, multi_texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[str]]]=None, languages: Optional[List[str]]=None, dense_feat: Optional[List[List[float]]]=None) ->torch.Tensor:
        if dense_feat is None:
            raise RuntimeError('Expect dense feature.')
        inputs: ScriptBatchInput = ScriptBatchInput(texts=resolve_texts(texts, multi_texts), tokens=squeeze_2d(tokens), languages=squeeze_1d(languages))
        dense_feat = self.normalizer.normalize(dense_feat)
        dense_tensor = torch.tensor(dense_feat, dtype=torch.float)
        sentence_embedding = self._forward(inputs, dense_tensor)
        if self.concat_dense:
            return torch.cat([sentence_embedding, dense_tensor], 1)
        else:
            return sentence_embedding


class ScriptPyTextEmbeddingModuleWithDenseIndex(ScriptPyTextEmbeddingModuleWithDense):

    def __init__(self, model: torch.jit.ScriptModule, tensorizer: ScriptTensorizer, normalizer: VectorNormalizer, index: int=0, concat_dense: bool=True):
        super().__init__(model, tensorizer, normalizer, concat_dense)
        self.index = torch.jit.Attribute(index, int)

    @torch.jit.script_method
    def _forward(self, inputs: ScriptBatchInput, dense_tensor: torch.Tensor):
        input_tensors = self.tensorizer(inputs)
        if self.tensorizer.device != '':
            dense_tensor = dense_tensor
        return self.model(input_tensors, dense_tensor)[self.index].cpu()


class ScriptXLMTensorizer(ScriptTensorizer):

    def __init__(self, tokenizer: torch.jit.ScriptModule, token_vocab: ScriptVocabulary, language_vocab: ScriptVocabulary, max_seq_len: int, default_language: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_vocab = token_vocab
        self.language_vocab = language_vocab
        self.token_vocab_lookup = VocabLookup(token_vocab)
        self.language_vocab_lookup = VocabLookup(language_vocab)
        self.max_seq_len = torch.jit.Attribute(max_seq_len, int)
        self.default_language = torch.jit.Attribute(default_language, str)

    @torch.jit.script_method
    def tokenize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]], language_row: List[str]) ->Tuple[List[List[Tuple[str, int, int]]], List[List[Tuple[str, int, int]]]]:
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        per_sentence_languages: List[List[Tuple[str, int, int]]] = []
        if text_row is not None:
            """
            Tokenize every single text into a list of tokens.
            For example:
            text_row = ["hello world", "this is sentence"]
            per_sentence_tokens = [["hello", "world"], ["this", "is", "sentence"]]
            """
            for idx, text in enumerate(text_row):
                sentence_tokens: List[Tuple[str, int, int]] = self.tokenizer.tokenize(text)
                sentence_languages: List[Tuple[str, int, int]] = [(language_row[idx], token[1], token[2]) for token in sentence_tokens]
                per_sentence_tokens.append(sentence_tokens)
                per_sentence_languages.append(sentence_languages)
        elif token_row is not None:
            """
            Tokenize every single token into a sub tokens. (example: BPE)
            For example:
            token_row = [["hello", "world"], ["this", "is", "sentence"]]
            per_sentence_tokens = [
                ["he", "llo" "wo", "rld"], ["th", "is", "is", "sen", "tence"]
            ]
            """
            for idx, sentence_raw_tokens in enumerate(token_row):
                sentence_tokens: List[Tuple[str, int, int]] = []
                sentence_languages: List[Tuple[str, int, int]] = []
                for raw_token in sentence_raw_tokens:
                    sub_tokens: List[Tuple[str, int, int]] = self.tokenizer.tokenize(raw_token)
                    sub_languages: List[Tuple[str, int, int]] = [(language_row[idx], token[1], token[2]) for token in sub_tokens]
                    sentence_tokens.extend(sub_tokens)
                    sentence_languages.extend(sub_languages)
                per_sentence_tokens.append(sentence_tokens)
                per_sentence_languages.append(sentence_languages)
        return per_sentence_tokens, per_sentence_languages

    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]], languages: List[Tuple[str, int, int]], max_seq_len: int) ->Tuple[List[int], List[int]]:
        token_ids: List[int] = self.token_vocab_lookup(tokens, bos_idx=self.token_vocab.eos_idx, eos_idx=self.token_vocab.eos_idx, use_eos_token_for_bos=True, max_seq_len=max_seq_len)[0]
        language_special_idx: int = self.language_vocab.idx.get(languages[0][0], self.language_vocab.unk_idx)
        language_ids = self.language_vocab_lookup(languages, bos_idx=language_special_idx, eos_idx=language_special_idx, use_eos_token_for_bos=True, max_seq_len=max_seq_len)[0]
        return token_ids, language_ids

    @torch.jit.script_method
    def numberize(self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]], language_row: List[str]) ->Tuple[List[int], List[int], int, List[int]]:
        per_sentence_tokens, per_sentence_languages = self.tokenize(text_row, token_row, language_row)
        token_ids: List[int] = []
        language_ids: List[int] = []
        max_seq_len: int = self.max_seq_len // len(per_sentence_tokens)
        for idx in range(len(per_sentence_tokens)):
            lookup_token_ids, lookup_language_ids = self._lookup_tokens(per_sentence_tokens[idx], per_sentence_languages[idx], max_seq_len)
            token_ids.extend(lookup_token_ids)
            language_ids.extend(lookup_language_ids)
        seq_len: int = len(token_ids)
        positions: List[int] = [i for i in range(seq_len)]
        return token_ids, language_ids, seq_len, positions

    @torch.jit.script_method
    def tensorize(self, texts: Optional[List[List[str]]]=None, tokens: Optional[List[List[List[str]]]]=None, languages: Optional[List[List[str]]]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size: int = self.batch_size(texts, tokens)
        row_size: int = self.row_size(texts, tokens)
        if languages is None:
            languages = [[self.default_language] * row_size] * batch_size
        tokens_2d: List[List[int]] = []
        languages_2d: List[List[int]] = []
        seq_len_2d: List[int] = []
        positions_2d: List[List[int]] = []
        for idx in range(batch_size):
            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(self.get_texts_by_index(texts, idx), self.get_tokens_by_index(tokens, idx), languages[idx])
            tokens_2d.append(numberized[0])
            languages_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])
            positions_2d.append(numberized[3])
        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.token_vocab.pad_idx)
        languages = torch.tensor(pad_2d(languages_2d, seq_lens=seq_len_2d, pad_idx=0), dtype=torch.long)
        positions = torch.tensor(pad_2d(positions_2d, seq_lens=seq_len_2d, pad_idx=0), dtype=torch.long)
        if self.device == '':
            return tokens, pad_mask, languages, positions
        else:
            return tokens, pad_mask, languages, positions


@torch.jit.script
def utf8_chars(s: str) ->List[str]:
    """An implementation of UTF8 character iteration in TorchScript.
    There are no bitwise operations in torchscript, so we compare directly to
    integer values. There isn't a lot of validation, for instance if you pass
    in an improperly encoded string with an out-of-place continuation byte,
    or with a non-left-to-right byte order, you'll get unexpected results
    and likely throw. Torch itself takes in unicode strings and encodes them
    as UTF8, so that should be actively hard to do.

    The logic is simple: looking at the current start-of-character byte.
    If its high bit is 0, it's a 1-byte character. Otherwise, the number of
    bytes is the number of leading 1s in its binary representation, so
    find that number by comparing it directly to ints with the appropriate
    representation, then append that many bytes as a character and move past
    them to the next start byte.
    """
    chars = torch.jit.annotate(List[str], [])
    i = 0
    while i < len(s):
        byte = ord(s[i])
        if byte < 128:
            chars.append(s[i])
            i += 1
        else:
            if byte < 224:
                num_bytes = 2
            elif byte < 240:
                num_bytes = 3
            elif byte < 248:
                num_bytes = 4
            elif byte < 252:
                num_bytes = 5
            elif byte < 254:
                num_bytes = 6
            elif byte < 255:
                num_bytes = 7
            else:
                num_bytes = 8
            chars.append(s[i:i + num_bytes])
            i += num_bytes
    return chars


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
        raise UninitializedLazyModuleError('Must call init_lazy_modules before getting parameters')

    @_parameters.setter
    def _parameters(self, value):
        return None

    def __setattr__(self, name, value):
        return object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        if not self._module:
            constructor_args = [(arg if not isinstance(arg, Infer) else arg.resolve(*args, **kwargs)) for arg in self._args]
            constructor_kwargs = {key: (arg if not isinstance(arg, Infer) else arg.resolve(*args, **kwargs)) for key, arg in self._kwargs.items()}
            self._module = self._module_class(*constructor_args, **constructor_kwargs)
        return self._module(*args, **kwargs)

    def resolve(self):
        """Must make a call to forward before calling this function; returns the
        full nn.Module object constructed using inferred arguments/dimensions."""
        if not self._module:
            raise UninitializedLazyModuleError('Must call forward before calling resolve on a lazy module')
        return self._module


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiLSTM,
     lambda: ([], {'num_layers': 1, 'bidirectional': 4, 'embed_dim': 4, 'hidden_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (ContextualTokenEmbedding,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextualWordConvolution,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DictEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embed_dim': 4, 'pooling_type': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FCModelWithNanAndInfWts,
     lambda: ([], {}),
     lambda: ([torch.rand([10, 10])], {}),
     True),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Highway,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaxPool,
     lambda: ([], {'config': _mock_config(), 'n_input': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanPool,
     lambda: ([], {'config': _mock_config(), 'n_input': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiLabelClassificationScores,
     lambda: ([], {'scores': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoPool,
     lambda: ([], {'config': _mock_config(), 'n_input': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PlaceholderAttentionIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PlaceholderIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'config': _mock_config(dropout=0.5, attn_dimension=4), 'n_input': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SentenceEncoder,
     lambda: ([], {}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (SeparableConv1d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1, 'bottleneck': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (SlotAttention,
     lambda: ([], {'config': _mock_config(attention_type=4), 'n_input': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Transformer,
     lambda: ([], {}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (Trim1d,
     lambda: ([], {'trim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VectorNormalizer,
     lambda: ([], {'dim': 4}),
     lambda: ([], {}),
     True),
    (WordEmbedding,
     lambda: ([], {'num_embeddings': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
]

class Test_facebookresearch_pytext(_paritybench_base):
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

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

