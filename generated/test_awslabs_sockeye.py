import sys
_module = sys.modules[__name__]
del sys
add_tag_to_lines = _module
remove_tag_from_translations = _module
genseqcopy = _module
setup = _module
sockeye = _module
arguments = _module
average = _module
beam_search = _module
checkpoint_decoder = _module
config = _module
constants = _module
convert_deepspeed = _module
data_io = _module
decoder = _module
embeddings = _module
encoder = _module
evaluate = _module
generate_decoder_states = _module
inference = _module
initial_setup = _module
knn = _module
layers = _module
lexicon = _module
log = _module
loss = _module
lr_scheduler = _module
model = _module
nvs = _module
optimizers = _module
output_handler = _module
prepare_data = _module
quantize = _module
rerank = _module
score = _module
scoring = _module
test_utils = _module
train = _module
training = _module
transformer = _module
translate = _module
utils = _module
vocab = _module
sockeye_contrib = _module
benchmark_to_output = _module
benchmark_to_percentiles = _module
plot_metrics = _module
rouge = _module
generate_graphs = _module
test = _module
test_integration = _module
common = _module
integration = _module
test_backwards_compatibility = _module
test_seq_copy_int = _module
system = _module
test_seq_copy_sys = _module
unit = _module
test_arguments = _module
test_average = _module
test_beam_search = _module
test_bleu = _module
test_chrf = _module
test_config = _module
test_data_io = _module
test_decoder = _module
test_deepspeed = _module
test_encoder = _module
test_fixed_param_strategy = _module
test_inference = _module
test_knn = _module
test_layers = _module
test_lexicon = _module
test_loss = _module
test_lr_scheduler = _module
test_output_handler = _module
test_params = _module
test_quantize = _module
test_reranking = _module
test_rouge = _module
test_scoring = _module
test_transformer = _module
test_translate = _module
test_utils = _module
test_vocab = _module

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


import types


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Tuple


from typing import Optional


import itertools


import logging


from typing import Iterable


import torch


import functools


from abc import abstractmethod


from abc import ABC


from typing import Union


import numpy as np


import torch as pt


import random


import time


from itertools import chain


import collections


import math


from collections import OrderedDict


from typing import cast


from typing import Iterator


from typing import Sequence


from typing import Sized


from typing import Set


import torch.distributed


from itertools import islice


from typing import Type


import copy


from functools import partial


from typing import Generator


import torch.nn.functional as F


import logging.config


from math import sqrt


from functools import lru_cache


import torch.distributed.elastic.multiprocessing.errors


from collections import deque


from collections import defaultdict


from itertools import starmap


from typing import TypeVar


import numpy as onp


from math import ceil


from math import pow


import re


class UpdateScores(pt.nn.Module):
    """
    A Module that updates the scores from the decoder step with accumulated scores.
    Finished hypotheses receive their accumulated score for C.PAD_ID.
    Hypotheses at maximum length are forced to produce C.EOS_ID.
    All other options are set to infinity.
    """

    def __init__(self, prevent_unk: bool=False):
        super().__init__()
        self.prevent_unk = prevent_unk
        assert C.PAD_ID == 0, 'This block only works with PAD_ID == 0'

    def forward(self, target_dists, finished, scores_accumulated, lengths, max_lengths, pad_dist, eos_dist):
        if self.prevent_unk:
            target_dists[:, C.UNK_ID] = np.inf
        scores = target_dists + scores_accumulated
        pad_dist = scores_accumulated + pad_dist
        scores = pt.where(finished.unsqueeze(1), pad_dist, scores)
        lengths = lengths + ~finished
        below_max_length = lengths < max_lengths
        scores = pt.where(pt.logical_or(below_max_length, finished).unsqueeze(1), scores, eos_dist + scores)
        return scores, lengths


class LengthPenalty(pt.nn.Module):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float=1.0, beta: float=0.0) ->None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.0) ** self.alpha

    def forward(self, lengths):
        if self.alpha == 0.0:
            if isinstance(lengths, (int, float)):
                return 1.0
            else:
                return pt.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


class BrevityPenalty(pt.nn.Module):
    """
    Calculates the logarithmic brevity penalty as:
      weight * log min(1, exp(1 - ref_len / hyp_len)) = weight * min(0, 1 - ref_len / hyp_len).

    :param weight: Linear weight.
    """

    def __init__(self, weight: float=0.0) ->None:
        super().__init__()
        self.weight = weight

    def forward(self, hyp_lengths, reference_lengths):
        if self.weight == 0.0:
            if isinstance(hyp_lengths, (int, float)):
                return 0.0
            else:
                return pt.zeros_like(hyp_lengths)
        else:
            if isinstance(hyp_lengths, (int, float)):
                log_bp = min(0.0, 1.0 - reference_lengths / hyp_lengths)
            else:
                log_bp = pt.minimum(pt.zeros_like(hyp_lengths, dtype=pt.float), 1.0 - reference_lengths / hyp_lengths.float())
            return self.weight * log_bp


class CandidateScorer(pt.nn.Module):

    def __init__(self, length_penalty_alpha: float=1.0, length_penalty_beta: float=0.0, brevity_penalty_weight: float=0.0) ->None:
        super().__init__()
        self._lp = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
        self._bp = None
        if brevity_penalty_weight > 0.0:
            self._bp = BrevityPenalty(weight=brevity_penalty_weight)

    def forward(self, scores, lengths, reference_lengths):
        lp = self._lp(lengths)
        if self._bp is not None:
            bp = self._bp(lengths, reference_lengths)
        else:
            bp = 0.0
        if isinstance(scores, (int, float)):
            return scores / lp - bp
        else:
            if isinstance(lp, pt.Tensor):
                lp = lp
            if isinstance(bp, pt.Tensor):
                bp = bp
            return (scores.squeeze(1) / lp - bp).unsqueeze(1)

    def unnormalize(self, scores, lengths, reference_lengths):
        bp = 0.0 if self._bp is None else self._bp(lengths, reference_lengths)
        if isinstance(scores, (int, float)):
            return (scores + bp) * self._lp(lengths)
        else:
            return ((scores.squeeze(1) + bp) * self._lp(lengths)).unsqueeze(1)


class SortNormalizeAndUpdateFinished(pt.nn.Module):
    """
    A Module for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self, pad_id: int, eos_id: int, scorer: CandidateScorer, expect_factors: bool) ->None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        self._scorer = scorer
        self.expect_factors = expect_factors

    def forward(self, best_hyp_indices, best_word_indices, finished, scores_accumulated, lengths, reference_lengths, *factor_args):
        finished = finished.index_select(0, best_hyp_indices)
        lengths = lengths.index_select(0, best_hyp_indices)
        reference_lengths = reference_lengths.index_select(0, best_hyp_indices)
        all_finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        newly_finished = pt.logical_xor(all_finished, finished).unsqueeze(1)
        scores_accumulated = pt.where(newly_finished, self._scorer(scores_accumulated, lengths, reference_lengths), scores_accumulated)
        finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        best_word_indices = best_word_indices.unsqueeze(1)
        scores = [scores_accumulated]
        if self.expect_factors:
            factors, factor_scores_accumulated = factor_args
            f_sorted = factors.index_select(0, best_hyp_indices)
            factor_scores, factor_indices = f_sorted[:, :, 0], f_sorted[:, :, 1]
            updated_factor_scores = factor_scores_accumulated.index_select(0, best_hyp_indices) + factor_scores
            best_word_indices = pt.cat((best_word_indices, factor_indices.int()), dim=1)
            scores.append(updated_factor_scores)
        return best_word_indices, finished, scores, lengths, reference_lengths


class TopK(pt.nn.Module):
    """
    Batch-wise topk operation.
    Forward method uses imperative shape inference, since both batch_size and vocab_size are dynamic
    during translation (due to variable batch size and potential vocabulary selection).

    NOTE: This module wouldn't support dynamic batch sizes when traced!
    """

    def __init__(self, k: int) ->None:
        """
        :param k: The number of smallest scores to return.
        """
        super().__init__()
        self.k = k

    def forward(self, scores):
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        batch_times_beam, vocab_size = scores.size()
        batch_size = pt.div(batch_times_beam, self.k, rounding_mode='trunc')
        scores = scores.view(batch_size, self.k * vocab_size)
        values, indices = pt.topk(scores, k=self.k, dim=1, largest=False, sorted=True)
        values, indices = values.view(-1, 1), indices.view(-1)
        best_hyp_indices, best_word_indices = indices.div(vocab_size, rounding_mode='floor'), indices.fmod(vocab_size)
        return best_hyp_indices, best_word_indices, values


class SampleK(pt.nn.Module):
    """
    A Module for selecting a random word from each hypothesis according to its distribution.
    """

    def __init__(self, n: int) ->None:
        super().__init__()
        self.n = n

    def forward(self, scores, target_dists, finished):
        """
        Choose an extension of each hypothesis from its softmax distribution.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param target_dists: The non-cumulative target distributions (ignored).
        :param finished: The list of finished hypotheses.
        :return: The row indices, column indices, and values of the sampled words.
        """
        target_dists = pt.exp(-target_dists)
        if self.n != 0:
            _, indices = pt.topk(target_dists, k=self.n, dim=1, largest=True, sorted=True)
            target_dists = pt.scatter(pt.zeros_like(target_dists), 1, indices, target_dists)
            target_dists = target_dists / target_dists.sum(1, keepdim=True)
        best_word_indices = pt.multinomial(target_dists, 1).squeeze(1)
        best_word_indices = best_word_indices.masked_fill(finished, 0)
        values = scores.gather(dim=1, index=best_word_indices.long().unsqueeze(1))
        best_hyp_indices = pt.arange(0, best_word_indices.size()[0])
        return best_hyp_indices, best_word_indices, values


class RepeatStates(pt.nn.Module):

    def __init__(self, beam_size: int, state_structure: List):
        super().__init__()
        self.beam_size = beam_size
        self.flat_structure = functools.reduce(operator.add, state_structure)

    def forward(self, *states):
        repeated_states = []
        assert len(states) == len(self.flat_structure), 'Number of states do not match the defined state structure'
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE or state_format == C.MASK_STATE:
                repeat_axis = 0
            elif state_format == C.DECODER_STATE or state_format == C.ENCODER_STATE:
                repeat_axis = 1
            else:
                raise ValueError('Provided state format %s not recognized.' % state_format)
            repeated_state = state.repeat_interleave(repeats=self.beam_size, dim=repeat_axis)
            repeated_states.append(repeated_state)
        return repeated_states


class SortStates(pt.nn.Module):

    def __init__(self, state_structure):
        super().__init__()
        self.flat_structure = functools.reduce(operator.add, state_structure)

    def forward(self, best_hyp_indices, *states):
        sorted_states = []
        assert len(states) == len(self.flat_structure), 'Number of states do not match the defined state structure'
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE:
                sorted_state = state.index_select(0, best_hyp_indices)
            elif state_format == C.DECODER_STATE:
                sorted_state = state.index_select(1, best_hyp_indices)
            elif state_format == C.ENCODER_STATE or state_format == C.MASK_STATE:
                sorted_state = state
            else:
                raise ValueError('Provided state format %s not recognized.' % state_format)
            sorted_states.append(sorted_state)
        return sorted_states


logger = logging.getLogger(__name__)


class Search(pt.nn.Module):

    def __init__(self, dtype: pt.dtype, bos_id: int, eos_id: int, device: pt.device, num_source_factors: int, num_target_factors: int, skip_nvs: bool=False, nvs_thresh: float=0.5):
        super().__init__()
        self.dtype = dtype
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.device = device
        self.num_source_factors = num_source_factors
        self.num_target_factors = num_target_factors
        self.skip_nvs = skip_nvs
        self.nvs_thresh = nvs_thresh
        self.output_vocab_sizes = utils.OnlineMeanAndVariance()

    def update_output_vocab_size(self, size: Union[float, int]):
        self.output_vocab_sizes.update(size)

    def log_search_stats(self):
        logger.debug(f'decoder softmax size: {self.output_vocab_sizes.mean:.1f} (avg)')


class GreedyTop1(pt.nn.Module):
    """
    Implements picking the highest scoring next word with support for vocabulary selection and target factors.
    """

    def forward(self, scores: pt.Tensor, vocab_slice_ids: Optional[pt.Tensor]=None, target_factors: Optional[pt.Tensor]=None) ->pt.Tensor:
        best_word_index = pt.argmin(scores, dim=-1, keepdim=True)
        if vocab_slice_ids is not None:
            best_word_index = vocab_slice_ids.index_select(0, best_word_index.squeeze(1)).unsqueeze(1)
        if target_factors is not None:
            factor_index = target_factors[:, :, 1].int()
            best_word_index = pt.cat((best_word_index, factor_index), dim=1)
        return best_word_index


class SafeLoaderWithTuple(yaml.SafeLoader):

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


class TaggedYamlObjectMetaclass(yaml.YAMLObjectMetaclass):

    def __init__(cls, name, bases, kwds):
        cls.yaml_tag = '!' + name
        new_kwds = {}
        new_kwds.update(kwds)
        new_kwds['yaml_tag'] = '!' + name
        super().__init__(name, bases, new_kwds)


class _Inference(ABC):

    @abstractmethod
    def state_structure(self):
        raise NotImplementedError()

    @abstractmethod
    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: pt.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def decode_step(self, step_input: pt.Tensor, states: List, vocab_slice_ids: Optional[pt.Tensor]=None, target_prefix_factor_mask: Optional[pt.Tensor]=None, factor_vocab_size: Optional[int]=None):
        raise NotImplementedError()

    @property
    def model_output_vocab_size(self):
        raise NotImplementedError()

    @property
    def model_output_factor_vocab_size(self):
        raise NotImplementedError()


class Encoder(pt.nn.Module):
    """
    Generic encoder interface.
    """

    @abstractmethod
    def get_num_hidden(self) ->int:
        """
        :return: The representation size of this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) ->int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len

    def get_max_seq_len(self) ->Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        return None


class LHUC(pt.nn.Module):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """

    def __init__(self, num_hidden: int, dtype: Optional[pt.dtype]=None) ->None:
        super().__init__()
        self.weight = pt.nn.Parameter(pt.empty(num_hidden, dtype=dtype))

    def forward(self, data: pt.Tensor) ->pt.Tensor:
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data


class OutputLayer(pt.nn.Module):
    """
    Final output layer of seq2seq models. Supports vocabulary selection that caches reduced weight/bias
    across multiple invocations if selected vocabulary ids do not change.

    :param hidden_size: Input hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight: Optional shared weight Parameter.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self, hidden_size: int, vocab_size: int, weight: Optional[pt.nn.Parameter]=None, dtype: Optional[pt.dtype]=None) ->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = hidden_size
        self.out_features = vocab_size
        if weight is None:
            self.weight = pt.nn.Parameter(pt.empty(vocab_size, hidden_size, dtype=dtype))
        else:
            self.weight = weight
        self.bias = pt.nn.Parameter(pt.empty(vocab_size, dtype=dtype))
        self.previous_slice_ids = pt.empty(0)
        self.reduced_weight = pt.empty(0)
        self.reduced_bias = pt.empty(0)

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={} dtype={}'.format(self.in_features, self.out_features, self.bias is not None, self.weight.dtype)

    def _is_new_slice(self, x: pt.Tensor) ->bool:
        if x.size() != self.previous_slice_ids.size() or pt.any(x != self.previous_slice_ids):
            return True
        return False

    def _take_slice(self, vocab_slice_ids: pt.Tensor) ->Tuple[pt.Tensor, pt.Tensor]:
        weight = self.weight[vocab_slice_ids]
        bias = self.bias[vocab_slice_ids]
        return weight, bias

    def forward(self, data: pt.Tensor, vocab_slice_ids: Optional[pt.Tensor]=None) ->pt.Tensor:
        if vocab_slice_ids is not None:
            if self._is_new_slice(vocab_slice_ids):
                self.previous_slice_ids = vocab_slice_ids
                weight, bias = self.reduced_weight, self.reduced_bias = self._take_slice(vocab_slice_ids)
            else:
                weight, bias = self.reduced_weight, self.reduced_bias
        else:
            weight, bias = self.weight, self.bias
        return F.linear(data, weight, bias)


class KNN(pt.nn.Module):
    """
    An alternative output layer that can produce a output distribution over the vocabulary
    by using the decoder hidden state to query into an index.
    For more details, see: https://arxiv.org/abs/2010.00710.

    :param keys_index: faiss index used for k-NN query.
    :param vals: a list of word indexes that maps key ids to their corresponding vocabulary ids.
    :param vocab_size: the size of the output vocabulary.
    :param k: number of candidates to be retrieved by k-nearest neighbors query.
    :param temperature: temperature that controls the smoothness of the output distribution.
    :param state_store: an optional state store object that is used to compute the exact distance
                        between the query and the index.
    """

    def __init__(self, keys_index: 'faiss.Index', vals: np.memmap, vocab_size: int, k=64, temperature=10, state_store: Optional[np.memmap]=None) ->None:
        super().__init__()
        self.keys_index = keys_index
        self.vals = vals
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.state_store = state_store

    def forward(self, data: pt.Tensor):
        distances, indices = self.keys_index.search(data.cpu().numpy().astype(np.float32), self.k)
        y = self.vals[(indices + 1) % len(self.vals)]
        y[y == C.BOS_ID] = C.EOS_ID
        if self.state_store is not None:
            raw_keys = pt.from_numpy(self.state_store[indices])
            distances = pt.norm(data.unsqueeze(1) - raw_keys, p=2, dim=-1)
        else:
            distances = np.sqrt(distances)
            distances = pt.from_numpy(distances)
        y = pt.from_numpy(y).long()
        probs = pt.exp(-distances / self.temperature)
        full_probs = pt.zeros((data.shape[0], self.vocab_size), device=data.device)
        full_probs.scatter_add_(src=probs, index=y.squeeze(2), dim=-1)
        z = pt.sum(full_probs, dim=-1).unsqueeze(-1)
        z[z < C.KNN_EPSILON] = C.KNN_EPSILON
        full_probs.div_(z)
        return full_probs


class LengthRatio(pt.nn.Module):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self, hidden_size: int, num_layers: int, dtype: Optional[pt.dtype]=None) ->None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        modules = []
        for _ in range(num_layers - 1):
            modules.append(pt.nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype))
            modules.append(pt.nn.Tanh())
        modules.append(pt.nn.Linear(in_features=hidden_size, out_features=1, dtype=dtype))
        modules.append(pt.nn.Softplus())
        self.layers = pt.nn.Sequential(*modules)

    def forward(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor) ->pt.Tensor:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        mask = pt.arange(source_encoded.size()[1], device=source_encoded_length.device)[None, :, None] >= source_encoded_length[:, None, None]
        source_masked = source_encoded.masked_fill(mask, 0.0)
        data = source_masked.sum(dim=1, keepdim=False) / source_encoded_length.unsqueeze(1)
        data = self.layers(data).squeeze(1)
        return data


@pt.jit.script
def interleaved_matmul_encdec_qk(q: pt.Tensor, kv: pt.Tensor, heads: int) ->pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_qk with PyTorch.

    :param q: (qlen, batch, hidden)
    :param kv: (kvlen, batch, hidden * 2) -- interleaved
    :param heads: number of attention heads
    :return: (batch * heads, qlen, klen)
    """
    qlen, batch, hidden = q.size()
    head_dim = hidden // heads
    q = q.contiguous().view(qlen, batch * heads, head_dim).transpose(0, 1)
    q = q * head_dim ** -0.5
    tmp = kv.reshape(-1, batch, heads, 2, head_dim)
    k = tmp[:, :, :, 0, :]
    k = k.permute(1, 2, 3, 0)
    k = k.reshape(batch * heads, head_dim, -1)
    return pt.bmm(q, k)


@pt.jit.script
def interleaved_matmul_encdec_valatt(kv: pt.Tensor, att: pt.Tensor, heads: int) ->pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_valatt with PyTorch.
    There is probably something to be gained by using views more
    efficiently but this is placeholder code anyway.

    :param kv: (kvlen, batch, hidden * 2)
    :param att: (batch * heads, qlen, kvlen)
    :param heads: number of attention heads
    :return: (qlen, batch, hidden)
    """
    kvlen, batch, hidden2 = kv.size()
    hidden = hidden2 // 2
    head_dim = hidden // heads
    tmp = kv.reshape(kvlen, batch, heads, 2, -1)
    v = tmp[:, :, :, 1, :]
    v = v.permute(1, 2, 0, 3)
    v = v.reshape(-1, kvlen, head_dim)
    output = pt.bmm(att, v)
    output = output.transpose(0, 1).contiguous().view(-1, batch, hidden)
    return output


class DotAttentionCell(pt.nn.Module):

    def __init__(self, dropout: float=0.0, heads: int=1) ->None:
        super().__init__()
        self.dropout = pt.nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.heads = heads

    def forward(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor]=None):
        """
        :param queries: Query tensor of shape (query_length, batch_size, hidden)
        :param key_values: Interleaved Key & value tensor of shape (key/value_length, batch_size, hidden * 2)
        :param mask: Optional boolean tensor for attention masking of shape (batch * heads, <qlen>, <kvlen>).
                     If this is cross-attention, <qlen> dimension can be 1 for broadcasting,
                     i.e. (batch * heads, 1, kvlen). For self-attention on the decoder side an autoregressive mask
                     should be provided of shape (1, len, len) or (len, len).
                     Value of this mask is True for positions that should be masked out (padding positions),
                     False for valid positions.
        """
        logits = interleaved_matmul_encdec_qk(queries, key_values, heads=self.heads)
        if mask is not None:
            logits = logits.masked_fill(mask, -C.LARGE_VALUES[logits.dtype])
        probs = F.softmax(logits, dim=-1)
        probs = self.dropout(probs) if self.dropout is not None else probs
        return interleaved_matmul_encdec_valatt(key_values, probs, heads=self.heads)


def clamp_to_dtype_min_max(data: pt.Tensor) ->pt.Tensor:
    """
    Clamp a tensor's values to the min and max for its dtype. This effectively
    pushes overflowed (infinite) values back into the finite range.

    See: https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
    """
    return pt.clamp(data, min=pt.finfo(data.dtype).min, max=pt.finfo(data.dtype).max)


class MultiHeadAttentionBase(pt.nn.Module):
    """
    Base class for Multi-head attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(self, depth_att: int=512, heads: int=8, depth_out: int=512, dropout: float=0.0, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0, 'Number of heads (%d) must divide attention depth (%d)' % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads
        self.clamp_to_dtype = clamp_to_dtype
        self.dot_att = DotAttentionCell(dropout=dropout, heads=heads)
        self.ff_out = pt.nn.Linear(in_features=depth_att, out_features=depth_out, bias=False, dtype=dtype)

    def _attend(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor]=None) ->pt.Tensor:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (queries_length, batch_size, depth).
        :param key_values: Keys/Values. Shape: (keys_values_length, batch_size, depth * 2).
        :param mask: Optional boolean attention mask. See DotAttentionCell for shape requirements.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """
        contexts = self.dot_att(queries=queries, key_values=key_values, mask=mask)
        contexts = self.ff_out(contexts)
        if self.clamp_to_dtype:
            contexts = clamp_to_dtype_min_max(contexts)
        return contexts


class AutoregressiveLayer(pt.nn.Module):

    @property
    @abstractmethod
    def num_state_tensors(self) ->int:
        """ Number of state tensors returned by the layer """
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) ->bool:
        """ Whether the layer makes use of a mask tensor or not """
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size) ->Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: pt.Tensor, previous_states: pt.Tensor, *args) ->Tuple:
        """
        :param inputs: layer input
        :param previous_states: Previous states array or list of arrays
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError


class MultiHeadSelfAttention(MultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(self, depth_att: int=512, heads: int=8, depth_out: int=512, dropout: float=0.0, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.depth_att = depth_att
        self.ff_in = pt.nn.Linear(in_features=depth_att, out_features=depth_att * 3, bias=False, dtype=dtype)
        self._drop_p = dropout
        self.kv_interleaved = False

    def separate_kv(self):
        """ write kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention) """
        assert self.kv_interleaved
        with pt.no_grad():
            kv = self.ff_in.weight.data[self.depth:, :]
            k, v = kv.view(self.heads, 2 * self.depth_per_head, self.depth).split(self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self.depth)
            v = v.reshape(self.depth, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self):
        """ write kv input projection parameters in interleaved format (compatible with interleaved matmul) """
        assert not self.kv_interleaved
        with pt.no_grad():
            _, k, v = self.ff_in.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self.depth)
        self.kv_interleaved = True

    def train(self, mode: bool=True):
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and not self.kv_interleaved:
            self.interleave_kv()
        return super().train(mode)

    @property
    def num_state_tensors(self) ->int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) ->bool:
        """ Whether the layer makes use of a mask tensor or not """
        return True

    def get_state_shape(self, batch_size: int) ->Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return 0, batch_size, self.depth_out * 2

    def forward(self, inputs: pt.Tensor, previous_states: Optional[pt.Tensor]=None, mask: Optional[pt.Tensor]=None, **args) ->Tuple[pt.Tensor, pt.Tensor]:
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a tensor of shape (max_length, batch, output_depth).

        :param inputs: Input Data. Shape: (length, batch, input_depth).
        :param previous_states: Optional list with two tensors - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :return: tensor of shape (max_length, batch, output_depth).
        """
        if self.training:
            assert not self.kv_interleaved
            contexts, _ = F.multi_head_attention_forward(query=inputs, key=inputs, value=inputs, embed_dim_to_check=self.depth, num_heads=self.heads, in_proj_weight=self.ff_in.weight, in_proj_bias=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=self._drop_p, out_proj_weight=self.ff_out.weight, out_proj_bias=self.ff_out.bias, training=self.training, key_padding_mask=None, need_weights=False, attn_mask=mask, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None)
            return contexts, contexts
        else:
            proj = self.ff_in(inputs)
            queries, states = proj.split((self.depth_att, 2 * self.depth_att), dim=2)
            if previous_states is not None:
                states = pt.cat((previous_states, states), dim=0)
            return self._attend(queries=queries, key_values=states, mask=mask), states


class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(self, depth_att: int=512, heads: int=8, depth_out: int=512, dropout: float=0.0, depth_key_value: int=512, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.ff_q = pt.nn.Linear(in_features=depth_out, out_features=depth_att, bias=False, dtype=dtype)
        self.ff_kv = pt.nn.Linear(in_features=depth_key_value, out_features=depth_att * 2, bias=False, dtype=dtype)
        self._drop_p = dropout
        self._depth_key_value = depth_key_value
        self.kv_interleaved = False

    def separate_kv(self):
        """Writes kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention). """
        assert self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.view(self.heads, 2 * self.depth_per_head, self._depth_key_value).split(self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self._depth_key_value)
            v = v.reshape(self.depth, self._depth_key_value)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self):
        """Writes kv input projection parameters in interleaved format (compatible with interleaved matmul). """
        assert not self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self._depth_key_value)
        self.kv_interleaved = True

    def train(self, mode: bool=True):
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and not self.kv_interleaved:
            self.interleave_kv()
        return super().train(mode)

    def forward(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor]=None, projected_memory_kv: Optional[pt.Tensor]=None) ->pt.Tensor:
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns an tensor of shape (max_length, batch, output_depth).

        :param queries: Query tensor. Shape: (queries_length, batch, input_depth).
        :param key_values: Memory data to attend to. Shape: (key_values_length, batch, input_depth).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :param projected_memory_kv: Optional previously projected memory keys and values.
        :return: tensor of shape (query_seq_len, batch, output_depth).
        """
        if self.training:
            assert not self.kv_interleaved
            assert projected_memory_kv is None, 'caching not supported in training'
            contexts, _ = F.multi_head_attention_forward(query=queries, key=key_values, value=key_values, embed_dim_to_check=self.depth, num_heads=self.heads, in_proj_weight=None, in_proj_bias=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=self._drop_p, out_proj_weight=self.ff_out.weight, out_proj_bias=self.ff_out.bias, training=self.training, key_padding_mask=None, need_weights=False, attn_mask=mask, use_separate_proj_weight=True, q_proj_weight=self.ff_q.weight, k_proj_weight=self.ff_kv.weight[:self.depth, :], v_proj_weight=self.ff_kv.weight[self.depth:, :])
            return contexts
        else:
            queries = self.ff_q(queries)
            key_values = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(key_values)
            return self._attend(queries=queries, key_values=key_values, mask=mask)


class PositionalEmbeddings(pt.nn.Module):
    """
    Takes an encoded sequence and adds sinusoidal or learned positional embeddings as in Vaswani et al, 2017 to it.

    :param weight_type: type of embeddings, fixed or learned.
    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self, weight_type: str, num_embed: int, max_seq_len: int, scale_up_input: bool, scale_down_positions: bool, dtype: Optional[pt.dtype]=None) ->None:
        utils.check_condition(num_embed % 2 == 0, 'Positional embeddings require an even embedding size it is however %d.' % num_embed)
        super().__init__()
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions
        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            weight = get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                weight *= self.num_embed ** -0.5
            if dtype is not None:
                weight = weight
            self.weight = pt.nn.Parameter(weight, requires_grad=False)
        elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight = pt.nn.Parameter(pt.empty(self.max_seq_len, self.num_embed, dtype=dtype))
        else:
            raise ValueError("weight_type '%s' is not supported!" % self.weight_type)

    def forward(self, data: pt.Tensor, steps: Optional[pt.Tensor]=None) ->pt.Tensor:
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)

        :return: Data with positional embeddings added
        """
        if steps is None:
            pos_embedding = self.weight.unsqueeze(0)[:, :data.size()[1]]
        else:
            steps = pt.clip(steps, max=self.max_seq_len - 1)
            pos_embedding = F.embedding(steps, self.weight)
        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_embedding = pos_embedding.detach()
        if self.scale_up_input:
            data = data * self.num_embed ** 0.5
        return data + pos_embedding


class SSRU(AutoregressiveLayer):
    """
    Simpler Simple Recurrent Unit

    Kim et al, "From Research to Production and Back: Ludicrously Fast Neural Machine Translation" WNGT 2019

    Variant of an LSTM cell aimed at reducing computational dependency across time steps.
    Formally described as:

    (1) f[t] = sigmoid(W1[t] * x[t] + b[t])
    (2) c[t] = f[t] . c[t-1] + (1 - f[t]) . W2[t] * x[t]
    (3) h = ReLU(c[t])

    where:
        . represents elementwise multiplication;
        x[t] is the input at time step t;
        f[t] is the output of the forget gate at time step t;
        c[t] is the cell state at time step t;
        h is the output of the unit.

    :param model_size: number of hidden units
    :param inference_only: flag used to indicate execution at inference time.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(self, model_size: int, inference_only: bool, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__()
        self.model_size = model_size
        self.inference_only = inference_only
        self.clamp_to_dtype = clamp_to_dtype
        self.cell_state_transform = self._inference_cell_state_transform if inference_only else self._training_cell_state_transform
        self.forget_gate = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=True, dtype=dtype)
        self.forget_gate_act = pt.nn.Sigmoid()
        self.linear = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=False, dtype=dtype)
        self.relu = pt.nn.ReLU(inplace=False)

    @property
    def num_state_tensors(self) ->int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) ->bool:
        """ Whether the layer makes use of a mask tensor or not """
        return False

    def get_state_shape(self, batch_size: int) ->Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return 1, batch_size, self.model_size

    @staticmethod
    @pt.jit.script_if_tracing
    def _training_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) ->Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at training time"""
        steps = weighted_inputs.size()[0]
        cell_state = previous_cell_state.squeeze(0)
        states = []
        for t in range(steps):
            cell_state = forget_rates[t, :, :] * cell_state + weighted_inputs[t, :, :]
            states.append(cell_state)
        states = pt.stack(states, dim=0)
        return states, cell_state.unsqueeze(0)

    @staticmethod
    def _inference_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) ->Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs
        return new_step_state, new_step_state

    def forward(self, inputs: pt.Tensor, previous_states: pt.Tensor, **args) ->Tuple[pt.Tensor, pt.Tensor]:
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates = self.forget_gate_act(self.forget_gate(inputs))
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)
        cell_state, last_step_state = self.cell_state_transform(previous_states, weighted_inputs, forget_rates)
        cell_state = self.relu(cell_state)
        if self.clamp_to_dtype:
            cell_state = clamp_to_dtype_min_max(cell_state)
        return cell_state, last_step_state


class Loss(pt.nn.Module):
    """
    Generic Loss interface.
    A loss has a name, a configuration, and stores information about the output and label it requires from the model(s),
    as well as a weight (default 1.0) and a method to create the corresponding metric.
    """

    def __init__(self, name: str, output_name: str, label_name: str, weight: float=1.0, metric_prefix: str='') ->None:
        super().__init__()
        self._name = name
        self._output_name = output_name
        self._label_name = label_name
        self._weight = weight
        self._metric = None
        self._metric_prefix = metric_prefix
        logger.info("Loss: %s | weight=%.2f | metric: %s (%s) | output_name: '%s' | label_name: '%s'", self._name, self.weight, self.metric.name, self.metric.short_name, self.output_name, self.label_name)

    def __call__(self, outputs: Dict[str, Any], labels: Dict[str, Any]):
        """
        Loss retrieves the required output and label.
        """
        utils.check_condition(self.output_name in outputs, "output '%s' not found. Loss requires this output key" % self.output_name)
        utils.check_condition(self.label_name in labels, "label '%s' not found. Loss requires this label key" % self.output_name)
        output = outputs[self.output_name]
        label = labels[self.label_name]
        return super().__call__(output, label)

    @abstractmethod
    def create_metric(self) ->'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        raise NotImplementedError()

    @property
    def metric(self) ->'LossMetric':
        if self._metric is None:
            self._metric = self.create_metric()
        return self._metric

    @property
    def weight(self) ->float:
        return self._weight

    @property
    def name(self) ->str:
        return self._name

    @property
    def output_name(self) ->str:
        return self._output_name

    @property
    def label_name(self) ->str:
        return self._label_name


class LossMetric(ABC):

    def __init__(self, name: str, short_name: Optional[str]=None, prefix: str='') ->None:
        self._name = prefix + name
        self._short_name = prefix + short_name if short_name else self._name
        self._sum = 0.0
        self._num_inst = 0.0

    def __repr__(self):
        return '%s(%.2f/%.2f=%.2f)' % (self.name, self._sum, self._num_inst, self.get())

    def __str__(self):
        return '%s=%f' % (self.short_name, self.get())

    @property
    def name(self):
        return self._name

    @property
    def short_name(self) ->str:
        return self._short_name

    def update(self, loss, num_samples):
        self._sum += loss
        self._num_inst += num_samples

    def get(self) ->float:
        return self._sum / self._num_inst if self._num_inst else float('nan')

    def reset(self):
        self._sum = 0.0
        self._num_inst = 0.0


class DynamicBCEWithLogitsLoss(pt.nn.BCEWithLogitsLoss):
    """ A version of BCEWithLogitsLoss where the pos_weight can be supplied dynamically in the `forward` call. """

    def __init__(self, weight: Optional[pt.Tensor]=None, size_average=None, reduce=None, reduction: str='mean', pos_weight: Optional[pt.Tensor]=None) ->None:
        super().__init__(reduction=reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[pt.Tensor]
        self.pos_weight: Optional[pt.Tensor]

    def forward(self, input: pt.Tensor, target: pt.Tensor, pos_weight: Optional[pt.Tensor]=None) ->pt.Tensor:
        if pos_weight is None:
            pos_weight = self.pos_weight
        return pt.nn.functional.binary_cross_entropy_with_logits(input, target, self.weight, pos_weight=pos_weight, reduction=self.reduction)


@pt.jit.script
def _label_to_bow(label: pt.Tensor, num_labels: int):
    bow = pt.zeros(label.shape[0], num_labels, device=label.device)
    bow[pt.arange(0, label.shape[0], dtype=pt.int64)[:, np.newaxis], label.long()] = 1.0
    return bow


class ModelWithLoss(torch.nn.Module):
    """
    Wraps a SockeyeModel and its Losses in a single module. The SockeyeModel
    can be JIT traced (ScriptModule).

    :param model: SockeyeModel (untraced or traced).
    :param losses: List of Loss objects.

    :return: Tuple of summed loss, list of loss values, and list of number of
             samples.
    """

    def __init__(self, model: torch.nn.Module, losses: List[loss.Loss]) ->None:
        super().__init__()
        self.model = model
        self.losses = losses

    def forward(self, source: torch.Tensor, source_length: torch.Tensor, target: torch.Tensor, target_length: torch.Tensor, labels: Dict[str, torch.Tensor]) ->Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        model_outputs = self.model(source, source_length, target, target_length)
        if utils.using_deepspeed():
            model_outputs = {output_name: output for output_name, output in model_outputs.items()}
        loss_outputs = [loss_function(model_outputs, labels) for loss_function in self.losses]
        loss_values, num_samples = zip(*loss_outputs)
        sum_losses = sum(loss_values) if len(loss_values) > 1 else loss_values[0]
        return sum_losses, loss_values, num_samples


class TransformerFeedForward(pt.nn.Module):

    def __init__(self, num_hidden: int, num_model: int, act_type: str, dropout: float, use_glu: bool=False, inference_only: bool=False, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__()
        self.dropout = dropout
        self.use_glu = use_glu
        self.clamp_to_dtype = clamp_to_dtype
        self.ff1 = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)
        self.act = sockeye.layers.get_activation(act_type, inplace=inference_only)
        if self.use_glu:
            self.linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)
        if self.dropout > 0.0:
            self.drop = pt.nn.Dropout(p=self.dropout, inplace=inference_only)
        self.ff2 = pt.nn.Linear(in_features=num_hidden, out_features=num_model, dtype=dtype)

    def forward(self, x):
        h = self.ff1(x)
        h = self.act(h)
        if self.use_glu:
            h = h * self.linear(x)
        if self.dropout > 0.0:
            h = self.drop(h)
        y = self.ff2(h)
        if self.clamp_to_dtype:
            y = sockeye.layers.clamp_to_dtype_min_max(y)
        return y


class TransformerProcessBlock(pt.nn.Module):
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self, sequence: str, dropout: float, num_hidden: int=0, dtype: Optional[pt.dtype]=None, clamp_to_dtype: bool=False) ->None:
        super().__init__()
        self.sequence = sequence
        self.clamp_to_dtype = clamp_to_dtype
        self.layer_norm = None
        if 'n' in sequence:
            self.layer_norm = pt.nn.LayerNorm(num_hidden, eps=1e-06, dtype=dtype)
        self.dropout = dropout
        if dropout > 0.0:
            self.drop = pt.nn.Dropout(p=dropout)

    def forward(self, data: pt.Tensor, prev: Optional[pt.Tensor]=None) ->pt.Tensor:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data
        if prev is None:
            assert 'r' not in self.sequence, 'Residual connection not allowed if no previous value given.'
        for step in self.sequence:
            if step == 'r':
                data = data + prev
            elif step == 'n':
                data = self.layer_norm(data)
            elif step == 'd':
                if self.dropout > 0.0:
                    data = self.drop(data)
            else:
                raise ValueError('Unknown step in sequence: %s' % step)
        if self.clamp_to_dtype:
            data = sockeye.layers.clamp_to_dtype_min_max(data)
        return data


class AutoRegressiveMask(pt.nn.Module):

    def forward(self, x: pt.Tensor) ->pt.Tensor:
        """ Input tensor with length on dimension 1 """
        mask = pt.full((x.shape[1], x.shape[1]), fill_value=1, device=x.device, dtype=pt.bool)
        mask = pt.triu(mask, diagonal=1)
        return mask.detach()


class DummyLoss(sockeye.loss.Loss):

    def forward(self, outputs, labels):
        return (outputs + labels) * self.weight

    def create_metric(self):
        return sockeye.loss.LossMetric('test_metric')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AutoRegressiveMask,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BrevityPenalty,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CandidateScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DynamicBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GreedyTop1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LHUC,
     lambda: ([], {'num_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LengthPenalty,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutputLayer,
     lambda: ([], {'hidden_size': 4, 'vocab_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSRU,
     lambda: ([], {'model_size': 4, 'inference_only': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TopK,
     lambda: ([], {'k': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_awslabs_sockeye(_paritybench_base):
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

