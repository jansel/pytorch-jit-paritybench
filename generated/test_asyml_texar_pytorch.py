import sys
_module = sys.modules[__name__]
del sys
conf = _module
bert_classifier_main = _module
bert_classifier_using_executor_main = _module
bert_with_hypertuning_main = _module
config_classifier = _module
config_data = _module
download_glue_data = _module
prepare_data = _module
utils = _module
data_utils = _module
model_utils = _module
config_train = _module
gpt2_generate_main = _module
gpt2_train_main = _module
classifier_main = _module
config_kim = _module
sst_data_preprocessor = _module
config_iwslt14 = _module
config_model = _module
config_model_full = _module
config_toy_copy = _module
seq2seq_attn = _module
bleu_main = _module
config_iwslt15 = _module
config_wmt14 = _module
model = _module
transformer_main = _module
preprocess = _module
config_lstm_ptb = _module
config_lstm_yahoo = _module
config_trans_ptb = _module
config_trans_yahoo = _module
vae_train = _module
config_data_imdb = _module
config_data_stsb = _module
dataset = _module
processor = _module
xlnet_classification_main = _module
xlnet_generation_ipython = _module
xlnet_generation_main = _module
setup = _module
attention_mechanism_test = _module
attention_mechanism_utils_test = _module
cell_wrappers_test = _module
layers_test = _module
optimization_test = _module
regularizers_test = _module
data_iterators_test = _module
large_file_test = _module
mono_text_data_test = _module
multi_aligned_data_test = _module
paired_text_data_test = _module
record_data_test = _module
sampler_test = _module
scalar_data_test = _module
embedding_test = _module
bert_tokenizer_test = _module
bert_tokenizer_utils_test = _module
gpt2_tokenizer_test = _module
roberta_tokenizer_test = _module
sentencepiece_tokenizer_test = _module
t5_tokenizer_test = _module
xlnet_tokenizer_test = _module
vocabulary_test = _module
bleu_moses_test = _module
bleu_test = _module
bleu_transformer_test = _module
metrics_test = _module
hyperparams_test = _module
adv_losses_test = _module
entropy_test = _module
mle_losses_test = _module
pg_losses_test = _module
rewards_test = _module
bert_classifier_test = _module
conv_classifiers_test = _module
gpt2_classifier_test = _module
rnn_classifiers_test = _module
roberta_classifier_test = _module
xlnet_classifier_test = _module
connectors_test = _module
decoder_helpers_test = _module
gpt2_decoder_test = _module
rnn_decoders_test = _module
transformer_decoders_test = _module
xlnet_decoder_test = _module
embedder_utils_test = _module
embedders_test = _module
t5_encoder_decoder_test = _module
bert_encoder_test = _module
conv_encoders_test = _module
gpt2_encoder_test = _module
rnn_encoders_test = _module
roberta_encoder_test = _module
transformer_encoder_test = _module
xlnet_encoder_test = _module
conv_networks_test = _module
networks_test = _module
bert_test = _module
gpt2_test = _module
roberta_test = _module
t5_test = _module
t5_utils_test = _module
xlnet_test = _module
xlnet_utils_test = _module
xlnet_regressor_test = _module
condition_test = _module
executor_test = _module
classification_test = _module
generation_test = _module
regression_test = _module
summary_test = _module
average_recorder_test = _module
beam_search_test = _module
rnn_test = _module
shapes_test = _module
utils_test = _module
texar = _module
torch = _module
core = _module
attention_mechanism = _module
attention_mechanism_utils = _module
cell_wrappers = _module
layers = _module
optimization = _module
regularizers = _module
custom = _module
activation = _module
distributions = _module
initializers = _module
data = _module
data_base = _module
data_iterators = _module
data_iterators_utils = _module
dataset_utils = _module
mono_text_data = _module
multi_aligned_data = _module
paired_text_data = _module
record_data = _module
sampler = _module
scalar_data = _module
text_data_base = _module
embedding = _module
tokenizers = _module
bert_tokenizer = _module
bert_tokenizer_utils = _module
gpt2_tokenizer = _module
gpt2_tokenizer_utils = _module
roberta_tokenizer = _module
sentencepiece_tokenizer = _module
t5_tokenizer = _module
tokenizer_base = _module
xlnet_tokenizer = _module
vocabulary = _module
evals = _module
bleu = _module
bleu_moses = _module
bleu_transformer = _module
metrics = _module
hyperparams = _module
losses = _module
adv_losses = _module
entropy = _module
losses_utils = _module
mle_losses = _module
pg_losses = _module
rewards = _module
module_base = _module
modules = _module
classifiers = _module
bert_classifier = _module
classifier_base = _module
conv_classifiers = _module
gpt2_classifier = _module
rnn_classifiers = _module
roberta_classifier = _module
xlnet_classifier = _module
connectors = _module
connector_base = _module
connectors = _module
decoders = _module
decoder_base = _module
decoder_helpers = _module
gpt2_decoder = _module
rnn_decoder_base = _module
rnn_decoders = _module
t5_decoder = _module
transformer_decoders = _module
xlnet_decoder = _module
embedders = _module
embedder_base = _module
embedder_utils = _module
embedders = _module
position_embedders = _module
encoder_decoders = _module
encoder_decoder_base = _module
t5_encoder_decoder = _module
encoders = _module
bert_encoder = _module
conv_encoders = _module
encoder_base = _module
gpt2_encoder = _module
multihead_attention = _module
rnn_encoders = _module
roberta_encoder = _module
t5_encoder = _module
transformer_encoder = _module
xlnet_encoder = _module
networks = _module
conv_networks = _module
network_base = _module
pretrained = _module
bert = _module
gpt2 = _module
pretrained_base = _module
roberta = _module
t5 = _module
t5_utils = _module
xlnet = _module
xlnet_utils = _module
regressors = _module
regressor_base = _module
xlnet_regressor = _module
run = _module
action = _module
condition = _module
executor = _module
executor_utils = _module
metric = _module
base_metric = _module
classification = _module
generation = _module
regression = _module
summary = _module
average_recorder = _module
beam_search = _module
dtypes = _module
exceptions = _module
nest = _module
rnn = _module
shapes = _module
test = _module
transformer_attentions = _module
types = _module
utils = _module
utils_io = _module
version = _module

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


import functools


import logging


from typing import Any


import torch


import torch.nn.functional as F


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


from torch import nn


from torch.nn import functional as F


import torch.nn as nn


import math


from torch import Tensor


from torch.optim.lr_scheduler import ExponentialLR


import numpy as np


from abc import ABC


from typing import Callable


from typing import NamedTuple


from typing import TypeVar


from torch.autograd import Function


from typing import Generic


import copy


from typing import Type


from typing import Iterable


from torch.nn.utils import clip_grad_norm_


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import Optimizer


from typing import ItemsView


from typing import Iterator


from typing import KeysView


from torch.distributions.distribution import Distribution


from abc import abstractmethod


from typing import overload


from torch.distributions import Categorical


from torch.distributions import Gumbel


import warnings


import itertools


import random


import re


from collections import OrderedDict


from collections import defaultdict


from typing import IO


from typing import Sequence


from typing import Set


from typing import no_type_check


from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


from collections import Counter


from typing import Counter as CounterType


from typing import Mapping


import collections


import inspect


from functools import lru_cache


from typing import Collection


from typing import MutableMapping


from typing import cast


from torch.nn.modules.conv import _ConvNd


class Seq2SeqAttn(nn.Module):

    def __init__(self, train_data):
        super().__init__()
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size
        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        self.source_embedder = tx.modules.WordEmbedder(vocab_size=self.
            source_vocab_size, hparams=config_model.embedder)
        self.target_embedder = tx.modules.WordEmbedder(vocab_size=self.
            target_vocab_size, hparams=config_model.embedder)
        self.encoder = tx.modules.BidirectionalRNNEncoder(input_size=self.
            source_embedder.dim, hparams=config_model.encoder)
        self.decoder = tx.modules.AttentionRNNDecoder(token_embedder=self.
            target_embedder, encoder_output_size=self.encoder.cell_fw.
            hidden_size + self.encoder.cell_bw.hidden_size, input_size=self
            .target_embedder.dim, vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)

    def forward(self, batch, mode):
        enc_outputs, _ = self.encoder(inputs=self.source_embedder(batch[
            'source_text_ids']), sequence_length=batch['source_length'])
        memory = torch.cat(enc_outputs, dim=2)
        if mode == 'train':
            helper_train = self.decoder.create_helper(decoding_strategy=
                'train_greedy')
            training_outputs, _, _ = self.decoder(memory=memory,
                memory_sequence_length=batch['source_length'], helper=
                helper_train, inputs=batch['target_text_ids'][:, :-1],
                sequence_length=batch['target_length'] - 1)
            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(labels
                =batch['target_text_ids'][:, 1:], logits=training_outputs.
                logits, sequence_length=batch['target_length'] - 1)
            return mle_loss
        else:
            start_tokens = memory.new_full(batch['target_length'].size(),
                self.bos_token_id, dtype=torch.int64)
            infer_outputs = self.decoder(start_tokens=start_tokens,
                end_token=self.eos_token_id, memory=memory,
                memory_sequence_length=batch['source_length'], beam_width=
                config_model.beam_width)
            return infer_outputs


class LabelSmoothingLoss(nn.Module):
    """With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.

    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size
        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self, output: torch.Tensor, target: torch.Tensor,
        label_lengths: torch.LongTensor) ->torch.Tensor:
        """Compute the label smoothing loss.

        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = output.size(), target.size()
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])
        return tx.losses.sequence_softmax_cross_entropy(labels=model_prob,
            logits=output, sequence_length=label_lengths,
            average_across_batch=False, sum_over_timesteps=False)


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError('loc must be at least one-dimensional.')
    return Independent(Normal(loc, scale_diag), 1)


def kl_divergence(means: Tensor, logvars: Tensor) ->Tensor:
    """Compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - means ** 2 - torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)
    return torch.sum(kl_cost)


State = TypeVar('State')


class RNNCellBase(nn.Module, Generic[State]):
    """The base class for RNN cells in our framework. Major differences over
    :torch_nn:`RNNCell` are two-fold:

    1. Holds an :torch_nn:`Module` which could either be a built-in
       RNN cell or a wrapped cell instance. This design allows
       :class:`RNNCellBase` to serve as the base class for both vanilla
       cells and wrapped cells.

    2. Adds :meth:`zero_state` method for initialization of hidden states,
       which can also be used to implement batch-specific initialization
       routines.
    """

    def __init__(self, cell: Union[nn.RNNCellBase, 'RNNCellBase']):
        super().__init__()
        if not isinstance(cell, nn.Module):
            raise ValueError(
                "Type of parameter 'cell' must be derived fromnn.Module, and has 'input_size' and 'hidden_size'attributes."
                )
        self._cell = cell

    @property
    def input_size(self) ->int:
        """The number of expected features in the input."""
        return self._cell.input_size

    @property
    def hidden_size(self) ->int:
        """The number of features in the hidden state."""
        return self._cell.hidden_size

    @property
    def _param(self) ->nn.Parameter:
        """Convenience method to access a parameter under the module. Useful
        when creating tensors of the same attributes using `param.new_*`.
        """
        return next(self.parameters())

    def init_batch(self):
        """Perform batch-specific initialization routines. For most cells this
        is a no-op.
        """
        pass

    def zero_state(self, batch_size: int) ->State:
        """Return zero-filled state tensor(s).

        Args:
            batch_size: int, the batch size.

        Returns:
            State tensor(s) initialized to zeros. Note that different subclasses
            might return tensors of different shapes and structures.
        """
        self.init_batch()
        if isinstance(self._cell, nn.RNNCellBase):
            state = self._param.new_zeros(batch_size, self.hidden_size,
                requires_grad=False)
        else:
            state = self._cell.zero_state(batch_size)
        return state

    def forward(self, input: torch.Tensor, state: Optional[State]=None
        ) ->Tuple[torch.Tensor, State]:
        """
        Returns:
            A tuple of (output, state). For single layer RNNs, output is
            the same as state.
        """
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        return self._cell(input, state)


class MaxReducePool1d(nn.Module):
    """A subclass of :torch_nn:`Module`.
    Max Pool layer for 1D inputs. The same as :torch_nn:`MaxPool1d` except that
    the pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        output, _ = torch.max(input, dim=2)
        return output


class AvgReducePool1d(nn.Module):
    """A subclass of :torch_nn:`Module`.
    Avg Pool layer for 1D inputs. The same as :torch_nn:`AvgPool1d` except that
    the pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return torch.mean(input, dim=2)


def _type_name(value):
    return type(value).__name__


class HParams:
    """A class that maintains hyperparameters for configuring Texar modules.
    The class has several useful features:

    - **Auto-completion of missing values.** Users can specify only a subset of
      hyperparameters they care about. Other hyperparameters will automatically
      take the default values. The auto-completion performs **recursively** so
      that hyperparameters taking `dict` values will also be auto-completed
      **All Texar modules** provide a :meth:`default_hparams` containing
      allowed hyperparameters and their default values. For example:

        .. code-block:: python

            ## Recursive auto-completion
            default_hparams = {"a": 1, "b": {"c": 2, "d": 3}}
            hparams = {"b": {"c": 22}}
            hparams_ = HParams(hparams, default_hparams)
            hparams_.todict() == {"a": 1, "b": {"c": 22, "d": 3}}
                # "a" and "d" are auto-completed

            ## All Texar modules have built-in `default_hparams`
            hparams = {"dropout_rate": 0.1}
            emb = tx.modules.WordEmbedder(hparams=hparams, ...)
            emb.hparams.todict() == {
                "dropout_rate": 0.1,  # provided value
                "dim": 100            # default value
                ...
            }

    - **Automatic type-check.** For most hyperparameters, provided value must
      have the same or compatible dtype with the default value. :class:`HParams`
      does necessary type-check, and raises Error if improper dtype is provided.
      Also, hyperparameters not listed in `default_hparams` are not allowed,
      except for `"kwargs"` as detailed below.

    - **Flexible dtype for specified hyperparameters.**  Some hyperparameters
      may allow different dtypes of values.

        - Hyperparameters named `"type"` are not type-checked.
          For example, in :func:`~texar.torch.core.get_rnn_cell`, hyperparameter
          `"type"` can take value of an RNNCell class, its string name of module
          path, or an RNNCell class instance. (String name or module path is
          allowed so that users can specify the value in YAML configuration
          files.)

        - For other hyperparameters, list them in the `"@no_typecheck"` field
          in :meth:`default_hparams` to skip type-check. For example, in
          :class:`~texar.torch.modules.Conv1DNetwork`, hyperparameter
          `"kernel_size"` can be set to either a `list` of `int`\\ s or simply
          an `int`.

    - **Special flexibility of keyword argument hyperparameters.**
      Hyperparameters named ``"kwargs"`` are used as keyword arguments for a
      class constructor or a function call. Such hyperparameters take a `dict`,
      and users can add arbitrary valid keyword arguments to the dict.
      For example:

        .. code-block:: python

            default_rnn_cell_hparams = {
                "type": "LSTMCell",
                "kwargs": {"num_units": 256}
                # Other hyperparameters
                ...
            }
            my_hparams = {
                "kwargs" {
                    "num_units": 123,
                    # Other valid keyword arguments for LSTMCell constructor
                    "forget_bias": 0.0
                    "activation": "torch.nn.functional.relu"
                }
            }
            _ = HParams(my_hparams, default_rnn_cell_hparams)

    - **Rich interfaces.** An :class:`HParams` instance provides rich interfaces
      for accessing, updating, or adding hyperparameters.

        .. code-block:: python

            hparams = HParams(my_hparams, default_hparams)
            # Access
            hparams.type == hparams["type"]
            # Update
            hparams.type = "GRUCell"
            hparams.kwargs = { "num_units": 100 }
            hparams.kwargs.num_units == 100
            # Add new
            hparams.add_hparam("index", 1)
            hparams.index == 1

            # Convert to `dict` (recursively)
            type(hparams.todic()) == dict

            # I/O
            pickle.dump(hparams, "hparams.dump")
            with open("hparams.dump", 'rb') as f:
                hparams_loaded = pickle.load(f)


    Args:
        hparams: A `dict` or an :class:`HParams` instance containing
            hyperparameters. If `None`, all hyperparameters are set to default
            values.
        default_hparams (dict): Hyperparameters with default values. If `None`,
            Hyperparameters are fully defined by :attr:`hparams`.
        allow_new_hparam (bool): If `False` (default), :attr:`hparams` cannot
            contain hyperparameters that are not included in
            :attr:`default_hparams`, except for the case of :attr:`"kwargs"` as
            above.
    """

    def __init__(self, hparams: Optional[Union['HParams', Dict[str, Any]]],
        default_hparams: Optional[Dict[str, Any]], allow_new_hparam: bool=False
        ):
        if isinstance(hparams, HParams):
            hparams = hparams.todict()
        if default_hparams is not None:
            parsed_hparams = self._parse(hparams, default_hparams,
                allow_new_hparam)
        else:
            parsed_hparams = self._parse(hparams, hparams)
        super().__setattr__('_hparams', parsed_hparams)

    @staticmethod
    def _parse(hparams: Optional[Dict[str, Any]], default_hparams: Optional
        [Dict[str, Any]], allow_new_hparam: bool=False):
        """Parses hyperparameters.

        Args:
            hparams (dict): Hyperparameters. If `None`, all hyperparameters are
                set to default values.
            default_hparams (dict): Hyperparameters with default values.
                If `None`,Hyperparameters are fully defined by :attr:`hparams`.
            allow_new_hparam (bool): If `False` (default), :attr:`hparams`
                cannot contain hyperparameters that are not included in
                :attr:`default_hparams`, except the case of :attr:`"kwargs"`.

        Return:
            A dictionary of parsed hyperparameters. Returns `None` if both
            :attr:`hparams` and :attr:`default_hparams` are `None`.

        Raises:
            ValueError: If :attr:`hparams` is not `None` and
                :attr:`default_hparams` is `None`.
            ValueError: If :attr:`default_hparams` contains "kwargs" not does
                not contains "type".
        """
        if hparams is None and default_hparams is None:
            return None
        if hparams is None:
            return HParams._parse(default_hparams, default_hparams)
        if default_hparams is None:
            raise ValueError(
                '`default_hparams` cannot be `None` if `hparams` is not `None`.'
                )
        no_typecheck_names = default_hparams.get('@no_typecheck', [])
        if 'kwargs' in default_hparams and 'type' not in default_hparams:
            raise ValueError(
                "Ill-defined hyperparameter structure: 'kwargs' must accompany with 'type'."
                )
        parsed_hparams = copy.deepcopy(default_hparams)
        for name, value in default_hparams.items():
            if name not in hparams and isinstance(value, dict):
                if name == 'kwargs' and 'type' in hparams and hparams['type'
                    ] != default_hparams['type']:
                    parsed_hparams[name] = HParams({}, {})
                else:
                    parsed_hparams[name] = HParams(value, value)
        for name, value in hparams.items():
            if name not in default_hparams:
                if allow_new_hparam:
                    parsed_hparams[name] = HParams._parse_value(value, name)
                    continue
                raise ValueError(
                    "Unknown hyperparameter: %s. Only hyperparameters named 'kwargs' hyperparameters can contain new entries undefined in default hyperparameters."
                     % name)
            if value is None:
                parsed_hparams[name] = HParams._parse_value(parsed_hparams[
                    name])
            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = HParams._parse_value(value)
                continue
            if isinstance(value, dict):
                if name not in no_typecheck_names and not isinstance(
                    default_value, dict):
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" % (
                        name, _type_name(default_value), _type_name(value)))
                if name == 'kwargs':
                    if 'type' in hparams and hparams['type'
                        ] != default_hparams['type']:
                        parsed_hparams[name] = HParams(value, value)
                    else:
                        parsed_hparams[name] = HParams(value, default_value,
                            allow_new_hparam=True)
                elif name in no_typecheck_names:
                    parsed_hparams[name] = HParams(value, value)
                else:
                    parsed_hparams[name] = HParams(value, default_value,
                        allow_new_hparam)
                continue
            if name == 'type' and 'kwargs' in default_hparams:
                parsed_hparams[name] = value
                continue
            if name in no_typecheck_names:
                parsed_hparams[name] = value
            elif isinstance(value, type(default_value)):
                parsed_hparams[name] = value
            elif callable(value) and callable(default_value):
                parsed_hparams[name] = value
            else:
                try:
                    parsed_hparams[name] = type(default_value)(value)
                except TypeError:
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" % (
                        name, _type_name(default_value), _type_name(value)))
        return parsed_hparams

    @staticmethod
    def _parse_value(value: Any, name: Optional[str]=None) ->Any:
        if isinstance(value, dict) and (name is None or name != 'kwargs'):
            return HParams(value, None)
        else:
            return value

    def __getattr__(self, name: str) ->Any:
        """Retrieves the value of the hyperparameter.
        """
        if name == '_hparams':
            return super().__getattribute__('_hparams')
        if name not in self._hparams:
            raise AttributeError('Unknown hyperparameter: %s' % name)
        return self._hparams[name]

    def __getitem__(self, name: str) ->Any:
        """Retrieves the value of the hyperparameter.
        """
        return self.__getattr__(name)

    def __setattr__(self, name: str, value: Any):
        """Sets the value of the hyperparameter.
        """
        if name not in self._hparams:
            raise ValueError(
                'Unknown hyperparameter: %s. Only the `kwargs` hyperparameters can contain new entries undefined in default hyperparameters.'
                 % name)
        self._hparams[name] = self._parse_value(value, name)

    def items(self) ->ItemsView[str, Any]:
        """Returns the list of hyperparameter `(name, value)` pairs.
        """
        return self._hparams.items()

    def keys(self) ->KeysView[str]:
        """Returns the list of hyperparameter names.
        """
        return self._hparams.keys()

    def __iter__(self) ->Iterator[Tuple[str, Any]]:
        for name, value in self._hparams.items():
            yield name, value

    def __len__(self) ->int:
        return len(self._hparams)

    def __contains__(self, name) ->bool:
        return name in self._hparams

    def __str__(self) ->str:
        """Return a string of the hyperparameters.
        """
        hparams_dict = self.todict()
        return json.dumps(hparams_dict, sort_keys=True, indent=2)

    def get(self, name: str, default: Optional[Any]=None) ->Any:
        """Returns the hyperparameter value for the given name. If name is not
        available then returns :attr:`default`.

        Args:
            name (str): the name of hyperparameter.
            default: the value to be returned in case name does not exist.
        """
        try:
            return self.__getattr__(name)
        except AttributeError:
            return default

    def add_hparam(self, name: str, value: Any):
        """Adds a new hyperparameter.
        """
        if name in self._hparams or hasattr(self, name):
            raise ValueError('Hyperparameter name already exists: %s' % name)
        self._hparams[name] = self._parse_value(value, name)

    def todict(self) ->Dict[str, Any]:
        """Returns a copy of hyperparameters as a dictionary.
        """
        dict_ = copy.deepcopy(self._hparams)
        for name, value in self._hparams.items():
            if isinstance(value, HParams):
                dict_[name] = value.todict()
        return dict_


def is_str(x):
    """Returns `True` if :attr:`x` is either a str or unicode.
    Returns `False` otherwise.
    """
    return isinstance(x, str)


def get_layer(hparams: Union[HParams, Dict[str, Any]]) ->nn.Module:
    """Makes a layer instance.

    The layer must be an instance of :torch_nn:`Module`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with
            structure:

            .. code-block:: python

                {
                    "type": "LayerClass",
                    "kwargs": {
                        # Keyword arguments of the layer class
                        # ...
                    }
                }

            Here:

            `"type"`: str or layer class or layer instance
                The layer type. This can be

                - The string name or full module path of a layer class. If
                  the class name is provided, the class must be in module
                  :torch_nn:`Module`, :mod:`texar.torch.core`, or
                  :mod:`texar.torch.custom`.
                - A layer class.
                - An instance of a layer class.

                For example

                .. code-block:: python

                    "type": "Conv1D"                               # class name
                    "type": "texar.torch.core.MaxReducePooling1D"  # module path
                    "type": "my_module.MyLayer"                    # module path
                    "type": torch.nn.Module.Linear                 # class
                    "type": Conv1D(filters=10, kernel_size=2)  # cell instance
                    "type": MyLayer(...)                       # cell instance

            `"kwargs"`: dict
                A dictionary of keyword arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable, or a `str` of
                  the name or module path to the activation function.
                - Arguments named "\\*_regularizer" and "\\*_initializer" can be a
                  class instance, or a `dict` of hyperparameters of respective
                  regularizers and initializers. See
                - Arguments named "\\*_constraint" can be a callable, or a `str`
                  of the name or full path to the constraint function.

    Returns:
        A layer instance. If ``hparams["type"]`` is a layer instance, returns it
        directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :torch_nn:`Module`.
    """
    if hparams is None:
        raise ValueError('`hparams` must not be `None`.')
    layer_type = hparams['type']
    if not is_str(layer_type) and not isinstance(layer_type, type):
        layer = layer_type
    else:
        layer_modules = ['torch.nn', 'texar.torch.core', 'texar.torch.custom']
        layer_class: Type[nn.Module] = utils.check_or_get_class(layer_type,
            layer_modules)
        if isinstance(hparams, dict):
            if (layer_class.__name__ == 'Linear' and 'in_features' not in
                hparams['kwargs']):
                raise ValueError(
                    '"in_features" should be specified for "torch.nn.{}"'.
                    format(layer_class.__name__))
            elif layer_class.__name__ in ['Conv1d', 'Conv2d', 'Conv3d'
                ] and 'in_channels' not in hparams['kwargs']:
                raise ValueError(
                    '"in_channels" should be specified for "torch.nn.{}"'.
                    format(layer_class.__name__))
            default_kwargs: Dict[str, Any] = {}
            default_hparams = {'type': layer_type, 'kwargs': default_kwargs}
            hparams = HParams(hparams, default_hparams)
        if layer_type == 'Sequential':
            names: List[str] = []
            layer = nn.Sequential()
            sub_hparams = hparams.kwargs.layers
            for hparam in sub_hparams:
                sub_layer = get_layer(hparam)
                name = utils.uniquify_str(sub_layer.__class__.__name__, names)
                names.append(name)
                layer.add_module(name=name, module=sub_layer)
        else:
            layer = utils.get_instance(layer_type, hparams.kwargs.todict(),
                layer_modules)
    if not isinstance(layer, nn.Module):
        raise ValueError('layer must be an instance of `torch.nn.Module`.')
    return layer


class MergeLayer(nn.Module):
    """A subclass of :torch_nn:`Module`.
    A layer that consists of multiple layers in parallel. Input is fed to
    each of the parallel layers, and the outputs are merged with a
    specified mode.

    Args:
        layers (list, optional): A list of :torch_docs:`torch.nn.Module
            <nn.html#module>` instances, or a list of hyperparameter
            dictionaries each of which specifies `"type"` and `"kwargs"` of each
            layer (see the `hparams` argument of :func:`get_layer`).

            If `None`, this layer degenerates to a merging operator that merges
            inputs directly.
        mode (str): Mode of the merge op. This can be:

            - :attr:`'concat'`: Concatenates layer outputs along one dim.
              Tensors must have the same shape except for the dimension
              specified in `dim`, which can have different sizes.
            - :attr:`'elemwise_sum'`: Outputs element-wise sum.
            - :attr:`'elemwise_mul'`: Outputs element-wise product.
            - :attr:`'sum'`: Computes the sum of layer outputs along the
              dimension given by `dim`. For example, given `dim=1`,
              two tensors of shape `[a, b]` and `[a, c]` respectively
              will result in a merged tensor of shape `[a]`.
            - :attr:`'mean'`: Computes the mean of layer outputs along the
              dimension given in `dim`.
            - :attr:`'prod'`: Computes the product of layer outputs along the
              dimension given in `dim`.
            - :attr:`'max'`: Computes the maximum of layer outputs along the
              dimension given in `dim`.
            - :attr:`'min'`: Computes the minimum of layer outputs along the
              dimension given in `dim`.
            - :attr:`'and'`: Computes the `logical and` of layer outputs along
              the dimension given in `dim`.
            - :attr:`'or'`: Computes the `logical or` of layer outputs along
              the dimension given in `dim`.
            - :attr:`'logsumexp'`: Computes
              log(sum(exp(elements across the dimension of layer outputs)))
        dim (int): The dim to use in merging. Ignored in modes
            :attr:`'elemwise_sum'` and :attr:`'elemwise_mul'`.
    """
    _functions: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {'sum'
        : torch.sum, 'mean': torch.mean, 'prod': torch.prod, 'max': lambda
        tensors, dim: torch.max(tensors, dim)[0], 'min': lambda tensors,
        dim: torch.min(tensors, dim)[0], 'and': torch.all, 'or': torch.any,
        'logsumexp': torch.logsumexp}

    def __init__(self, layers: Optional[List[nn.Module]]=None, mode: str=
        'concat', dim: Optional[int]=None):
        super().__init__()
        self._mode = mode
        self._dim = dim
        self._layers: Optional[nn.ModuleList] = None
        if layers is not None:
            if len(layers) == 0:
                raise ValueError(
                    "'layers' must be either None or a non-empty list.")
            self._layers = nn.ModuleList()
            for layer in layers:
                if isinstance(layer, nn.Module):
                    self._layers.append(layer)
                else:
                    self._layers.append(get_layer(hparams=layer))

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Feed input to every containing layer and merge the outputs.

        Args:
            input: The input tensor.

        Returns:
            The merged tensor.
        """
        layer_outputs: List[torch.Tensor]
        if self._layers is None:
            layer_outputs = input
            if not isinstance(layer_outputs, (list, tuple)):
                layer_outputs = [layer_outputs]
        else:
            layer_outputs = []
            for layer in self._layers:
                layer_output = layer(input)
                layer_outputs.append(layer_output)
        dim = self._dim if self._dim is not None else -1
        if self._mode == 'concat':
            outputs = torch.cat(tensors=layer_outputs, dim=dim)
        elif self._mode == 'elemwise_sum':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = torch.add(outputs, layer_outputs[i])
        elif self._mode == 'elemwise_mul':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = torch.mul(outputs, layer_outputs[i])
        elif self._mode in self._functions:
            _concat = torch.cat(tensors=layer_outputs, dim=dim)
            outputs = self._functions[self._mode](_concat, dim)
        else:
            raise ValueError("Unknown merge mode: '%s'" % self._mode)
        return outputs

    @property
    def layers(self) ->Optional[nn.ModuleList]:
        """The list of parallel layers.
        """
        return self._layers


class Flatten(nn.Module):
    """Flatten layer to flatten a tensor after convolution."""

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return input.view(input.size()[0], -1)


class Identity(nn.Module):
    """Identity activation layer."""

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return input


class BertGELU(nn.Module):
    """Bert uses GELU as the activation function for the position-wise network.
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GPTGELU(nn.Module):
    """For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class ModuleBase(nn.Module, ABC):
    """Base class inherited by modules that are configurable through
    hyperparameters.

    This is a subclass of :torch_nn:`Module`.

    A Texar module inheriting :class:`~texar.torch.ModuleBase` is
    **configurable through hyperparameters**. That is, each module defines
    allowed hyperparameters and default values. Hyperparameters not
    specified by users will take default values.

    Args:
        hparams (dict, optional): Hyperparameters of the module. See
            :meth:`default_hparams` for the structure and default values.
    """

    def __init__(self, hparams: Optional[Union[HParams, Dict[str, Any]]]=None):
        super().__init__()
        if not hasattr(self, '_hparams'):
            self._hparams = HParams(hparams, self.default_hparams())
        elif hparams is not None:
            raise ValueError(
                '`self._hparams` is already assigned, but `hparams` argument is not None.'
                )

    @staticmethod
    def default_hparams() ->Dict[str, Any]:
        """Returns a `dict` of hyperparameters of the module with default
        values. Used to replace the missing values of input `hparams`
        during module construction.

        .. code-block:: python

            {
                "name": "module"
            }
        """
        return {'name': 'module'}

    @property
    def trainable_variables(self) ->List[nn.Parameter]:
        """The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self.parameters() if x.requires_grad]

    @property
    def hparams(self) ->HParams:
        """An :class:`~texar.torch.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams

    @property
    def output_size(self):
        """The feature size of :meth:`forward` output tensor(s),
        usually it is equal to the last dimension value of the output
        tensor size.
        """
        raise NotImplementedError


class T5LayerNorm(nn.Module):
    """ Custom LayerNorm for T5 with no mean subtraction and no bias.
    """

    def __init__(self, input_size: int, eps: float=1e-05):
        super().__init__()
        self.w = nn.Parameter(torch.ones(input_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * x


class PositionalEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, embed_dim: int):
        super().__init__()
        freq_seq = torch.arange(0.0, embed_dim, 2.0)
        inv_freq = 1 / 10000 ** (freq_seq / embed_dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: torch.Tensor) ->torch.Tensor:
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        pos_embed = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_embed


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_asyml_texar_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(MaxReducePool1d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(AvgReducePool1d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MergeLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(BertGELU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(GPTGELU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(T5LayerNorm(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

