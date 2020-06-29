import sys
_module = sys.modules[__name__]
del sys
allennlp = _module
__main__ = _module
commands = _module
evaluate = _module
find_learning_rate = _module
predict = _module
print_results = _module
subcommand = _module
test_install = _module
train = _module
common = _module
checks = _module
file_utils = _module
from_params = _module
lazy = _module
logging = _module
params = _module
plugins = _module
registrable = _module
tee = _module
testing = _module
model_test_case = _module
test_case = _module
tqdm = _module
util = _module
data = _module
batch = _module
dataloader = _module
dataset_readers = _module
babi = _module
conll2003 = _module
dataset_reader = _module
dataset_utils = _module
span_utils = _module
interleaving_dataset_reader = _module
sequence_tagging = _module
sharded_dataset_reader = _module
text_classification_json = _module
fields = _module
adjacency_field = _module
array_field = _module
field = _module
flag_field = _module
index_field = _module
label_field = _module
list_field = _module
metadata_field = _module
multilabel_field = _module
namespace_swapping_field = _module
sequence_field = _module
sequence_label_field = _module
span_field = _module
text_field = _module
instance = _module
samplers = _module
bucket_batch_sampler = _module
max_tokens_batch_sampler = _module
token_indexers = _module
elmo_indexer = _module
pretrained_transformer_indexer = _module
pretrained_transformer_mismatched_indexer = _module
single_id_token_indexer = _module
spacy_indexer = _module
token_characters_indexer = _module
token_indexer = _module
tokenizers = _module
character_tokenizer = _module
letters_digits_tokenizer = _module
pretrained_transformer_tokenizer = _module
sentence_splitter = _module
spacy_tokenizer = _module
token = _module
tokenizer = _module
whitespace_tokenizer = _module
vocabulary = _module
interpret = _module
attackers = _module
attacker = _module
hotflip = _module
input_reduction = _module
utils = _module
saliency_interpreters = _module
integrated_gradient = _module
saliency_interpreter = _module
simple_gradient = _module
smooth_gradient = _module
models = _module
archival = _module
basic_classifier = _module
model = _module
simple_tagger = _module
modules = _module
attention = _module
additive_attention = _module
attention = _module
bilinear_attention = _module
cosine_attention = _module
dot_product_attention = _module
linear_attention = _module
augmented_lstm = _module
bimpm_matching = _module
conditional_random_field = _module
elmo = _module
elmo_lstm = _module
encoder_base = _module
feedforward = _module
gated_sum = _module
highway = _module
input_variational_dropout = _module
layer_norm = _module
lstm_cell_with_projection = _module
masked_layer_norm = _module
matrix_attention = _module
bilinear_matrix_attention = _module
cosine_matrix_attention = _module
dot_product_matrix_attention = _module
linear_matrix_attention = _module
matrix_attention = _module
maxout = _module
residual_with_layer_dropout = _module
sampled_softmax_loss = _module
scalar_mix = _module
seq2seq_encoders = _module
compose_encoder = _module
feedforward_encoder = _module
gated_cnn_encoder = _module
pass_through_encoder = _module
pytorch_seq2seq_wrapper = _module
pytorch_transformer_wrapper = _module
seq2seq_encoder = _module
seq2vec_encoders = _module
bert_pooler = _module
boe_encoder = _module
cls_pooler = _module
cnn_encoder = _module
cnn_highway_encoder = _module
pytorch_seq2vec_wrapper = _module
seq2vec_encoder = _module
softmax_loss = _module
span_extractors = _module
bidirectional_endpoint_span_extractor = _module
endpoint_span_extractor = _module
self_attentive_span_extractor = _module
span_extractor = _module
stacked_alternating_lstm = _module
stacked_bidirectional_lstm = _module
text_field_embedders = _module
basic_text_field_embedder = _module
text_field_embedder = _module
time_distributed = _module
token_embedders = _module
bag_of_word_counts_token_embedder = _module
elmo_token_embedder = _module
embedding = _module
empty_embedder = _module
pass_through_token_embedder = _module
pretrained_transformer_embedder = _module
pretrained_transformer_mismatched_embedder = _module
token_characters_encoder = _module
token_embedder = _module
nn = _module
activations = _module
beam_search = _module
chu_liu_edmonds = _module
initializers = _module
regularizers = _module
regularizer = _module
regularizer_applicator = _module
util = _module
predictors = _module
predictor = _module
sentence_tagger = _module
text_classifier = _module
tools = _module
archive_surgery = _module
create_elmo_embeddings_from_vocab = _module
inspect_cache = _module
training = _module
checkpointer = _module
learning_rate_schedulers = _module
cosine = _module
learning_rate_scheduler = _module
noam = _module
polynomial_decay = _module
slanted_triangular = _module
metric_tracker = _module
metrics = _module
attachment_scores = _module
auc = _module
average = _module
bleu = _module
boolean_accuracy = _module
categorical_accuracy = _module
covariance = _module
entropy = _module
evalb_bracketing_scorer = _module
f1_measure = _module
fbeta_measure = _module
mean_absolute_error = _module
metric = _module
pearson_correlation = _module
perplexity = _module
rouge = _module
sequence_accuracy = _module
span_based_f1_measure = _module
spearman_correlation = _module
unigram_recall = _module
momentum_schedulers = _module
inverted_triangular = _module
momentum_scheduler = _module
moving_average = _module
no_op_trainer = _module
optimizers = _module
scheduler = _module
tensorboard_writer = _module
trainer = _module
util = _module
version = _module
benchmarks = _module
character_tokenizer_bench = _module
resume_daemon = _module
run_with_beaker = _module
build_docs_config = _module
check_links = _module
get_version = _module
py2md = _module
resume_daemon_test = _module
basic_example = _module
py2md_test = _module
train_fixtures = _module
setup = _module
test_fixtures = _module
d = _module
tests = _module
evaluate_test = _module
find_learning_rate_test = _module
main_test = _module
no_op_train_test = _module
predict_test = _module
print_results_test = _module
test_install_test = _module
train_test = _module
file_utils_test = _module
from_params_test = _module
logging_test = _module
params_test = _module
plugins_test = _module
registrable_test = _module
util_test = _module
babi_reader_test = _module
dataset_reader_test = _module
span_utils_test = _module
interleaving_dataset_reader_test = _module
lazy_dataset_reader_test = _module
sequence_tagging_test = _module
sharded_dataset_reader_test = _module
text_classification_json_test = _module
dataset_test = _module
adjacency_field_test = _module
array_field_test = _module
field_test = _module
flag_field_test = _module
index_field_test = _module
label_field_test = _module
list_field_test = _module
metadata_field_test = _module
multilabel_field_test = _module
sequence_label_field_test = _module
span_field_test = _module
text_field_test = _module
instance_test = _module
bucket_batch_sampler_test = _module
max_tokens_batch_sampler_test = _module
sampler_test = _module
character_token_indexer_test = _module
elmo_indexer_test = _module
pretrained_transformer_indexer_test = _module
pretrained_transformer_mismatched_indexer_test = _module
single_id_token_indexer_test = _module
spacy_indexer_test = _module
character_tokenizer_test = _module
letters_digits_tokenizer_test = _module
pretrained_transformer_tokenizer_test = _module
sentence_splitter_test = _module
spacy_tokenizer_test = _module
vocabulary_test = _module
hotflip_test = _module
input_reduction_test = _module
integrated_gradient_test = _module
simple_gradient_test = _module
smooth_gradient_test = _module
archival_test = _module
basic_classifier_test = _module
model_test = _module
simple_tagger_test = _module
test_model_test_case = _module
additive_attention_test = _module
bilinear_attention_test = _module
cosine_attention_test = _module
dot_product_attention_test = _module
linear_attention_test = _module
augmented_lstm_test = _module
bimpm_matching_test = _module
conditional_random_field_test = _module
elmo_test = _module
encoder_base_test = _module
feedforward_test = _module
gated_sum_test = _module
highway_test = _module
lstm_cell_with_projection_test = _module
masked_layer_norm_test = _module
bilinear_matrix_attention_test = _module
cosine_matrix_attention_test = _module
dot_product_matrix_attention_test = _module
linear_matrix_attention_test = _module
maxout_test = _module
residual_with_layer_dropout_test = _module
sampled_softmax_loss_test = _module
scalar_mix_test = _module
seq2seq_encoder_test = _module
compose_encoder_test = _module
feedforward_encoder_test = _module
gated_cnn_encoder_test = _module
pass_through_encoder_test = _module
pytorch_seq2seq_wrapper_test = _module
pytorch_transformer_wrapper_test = _module
seq2vec_encoder_test = _module
boe_encoder_test = _module
cls_pooler_test = _module
cnn_encoder_test = _module
cnn_highway_encoder_test = _module
pytorch_seq2vec_wrapper_test = _module
bidirectional_endpoint_span_extractor_test = _module
endpoint_span_extractor_test = _module
self_attentive_span_extractor_test = _module
stacked_alternating_lstm_test = _module
stacked_bidirectional_lstm_test = _module
stacked_elmo_lstm_test = _module
basic_text_field_embedder_test = _module
time_distributed_test = _module
bag_of_word_counts_token_embedder_test = _module
elmo_token_embedder_test = _module
embedding_test = _module
pass_through_embedder_test = _module
pretrained_transformer_embedder_test = _module
pretrained_transformer_mismatched_embedder_test = _module
token_characters_encoder_test = _module
beam_search_test = _module
chu_liu_edmonds_test = _module
initializers_test = _module
pretrained_model_initializer_test = _module
regularizers_test = _module
util_test = _module
predictor_test = _module
sentence_tagger_test = _module
text_classifier_test = _module
checkpointer_test = _module
cosine_test = _module
learning_rate_scheduler_test = _module
slanted_triangular_test = _module
attachment_scores_test = _module
auc_test = _module
bleu_test = _module
boolean_accuracy_test = _module
categorical_accuracy_test = _module
covariance_test = _module
entropy_test = _module
evalb_bracketing_scorer_test = _module
f1_measure_test = _module
fbeta_measure_test = _module
mean_absolute_error_test = _module
pearson_correlation_test = _module
rouge_test = _module
sequence_accuracy_test = _module
span_based_f1_measure_test = _module
spearman_correlation_test = _module
unigram_recall_test = _module
inverted_triangular_test = _module
moving_average_test = _module
no_op_trainer_test = _module
optimizer_test = _module
trainer_test = _module
tutorials = _module
tagger = _module
basic_allennlp_test = _module
version_test = _module

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


import copy


from typing import Any


from typing import Dict


from typing import Iterable


from typing import Set


from typing import Union


import torch


from numpy.testing import assert_allclose


import logging


import random


from itertools import islice


from itertools import zip_longest


from typing import Callable


from typing import Generator


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import TypeVar


import numpy


import torch.distributed as dist


from copy import deepcopy


from typing import NamedTuple


from torch.nn import Module


from typing import Type


from torch.nn.modules.linear import Linear


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import math


from torch.nn import Parameter


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn as nn


import warnings


from torch.nn.modules import Dropout


from typing import Sequence


import numpy as np


from torch.nn import ParameterList


from torch import nn


import torch.nn


from torch.nn import Conv1d


from torch.nn import Linear


import itertools


import re


from typing import cast


from typing import BinaryIO


from torch.nn.functional import embedding


import torch.nn.init


from collections import defaultdict


import time


import torch.optim.lr_scheduler


from torch.nn.parallel import DistributedDataParallel


from torch.nn.utils import clip_grad_norm_


from collections import Counter


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from collections import OrderedDict


from numpy.testing import assert_almost_equal


from torch.autograd import Variable


from torch.nn.modules.rnn import LSTM


from torch.nn import LSTM


from torch.nn import RNN


import inspect


from torch.nn import GRU


from torch.nn import Embedding


from numpy.testing import assert_array_almost_equal


from math import isclose


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


DataArray = TypeVar('DataArray', torch.Tensor, Dict[str, torch.Tensor],
    Dict[str, Dict[str, torch.Tensor]])


def _is_encodable(value: str) ->bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    return value == '' or value.encode('utf-8', 'ignore') != b''


def _environment_variables() ->Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if
        _is_encodable(value)}


def _is_dict_free(obj: Any) ->bool:
    """
    Returns False if obj is a dict, or if it's a list with an element that _has_dict.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all(_is_dict_free(item) for item in obj)
    else:
        return True


def _replace_none(params: Any) ->Any:
    if params == 'None':
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int], gain:
    float=1.0) ->None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.

    # Parameters

    tensor : `torch.Tensor`, required.
        A tensor to initialize.
    split_sizes : `List[int]`, required.
        A list of length `tensor.ndim()` specifying the size of the
        blocks along that particular dimension. E.g. `[10, 20]` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : `float`, optional (default = `1.0`)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any(a % b != 0 for a, b in zip(sizes, split_sizes)):
        raise ConfigurationError(
            'tensor dimensions must be divisible by their respective split_sizes. Found size: {} and split_sizes: {}'
            .format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(
        sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple(slice(start_index, start_index + step) for 
            start_index, step in index_and_step_tuples)
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].
            contiguous(), gain=gain)


class AugmentedLSTMCell(torch.nn.Module):
    """
    `AugmentedLSTMCell` implements a AugmentedLSTM cell.

    # Parameters

    embed_dim : `int`
        The number of expected features in the input.
    lstm_dim : `int`
        Number of features in the hidden state of the LSTM.
    use_highway : `bool`, optional (default = `True`)
        If `True` we append a highway network to the outputs of the LSTM.
    use_bias : `bool`, optional (default = `True`)
        If `True` we use a bias in our LSTM calculations, otherwise we don't.

    # Attributes

    input_linearity : `nn.Module`
        Fused weight matrix which computes a linear function over the input.
    state_linearity : `nn.Module`
        Fused weight matrix which computes a linear function over the states.
    """

    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool=
        True, use_bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias
        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim
            self.input_linearity = torch.nn.Linear(self.embed_dim, self.
                _highway_inp_proj_end, bias=self.use_bias)
            self.state_linearity = torch.nn.Linear(self.lstm_dim, self.
                _highway_inp_proj_start, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.embed_dim, 4 * self
                .lstm_dim, bias=self.use_bias)
            self.state_linearity = torch.nn.Linear(self.lstm_dim, 4 * self.
                lstm_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.lstm_dim,
            self.embed_dim])
        block_orthogonal(self.state_linearity.weight.data, [self.lstm_dim,
            self.lstm_dim])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.lstm_dim:2 * self.lstm_dim].fill_(
            1.0)

    def forward(self, x: torch.Tensor, states=Tuple[torch.Tensor, torch.
        Tensor], variational_dropout_mask: Optional[torch.BoolTensor]=None
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        !!! Warning
            DO NOT USE THIS LAYER DIRECTLY, instead use the AugmentedLSTM class

        # Parameters

        x : `torch.Tensor`
            Input tensor of shape (bsize x input_dim).
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing
            the hidden state and the cell state of each element in
            the batch. Each of these tensors have a dimension of
            (bsize x nhid). Defaults to `None`.

        # Returns

        `Tuple[torch.Tensor, torch.Tensor]`
            Returned states. Shape of each state is (bsize x nhid).

        """
        hidden_state, memory_state = states
        if variational_dropout_mask is not None and self.training:
            hidden_state = hidden_state * variational_dropout_mask
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
        return timestep_output, memory


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.
    Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    # Parameters

    dropout_probability : `float`, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : `torch.Tensor`, required.


    # Returns

    `torch.FloatTensor`
        A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
        This scaling ensures expected values and variances of the output of applying this mask
        and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability
        ).to(tensor_for_masking.device)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class AugmentedLstm(torch.nn.Module):
    """
    `AugmentedLstm` implements a one-layer single directional
    AugmentedLSTM layer. AugmentedLSTM is an LSTM which optionally
    appends an optional highway network to the output layer. Furthermore the
    dropout controls the level of variational dropout done.

    # Parameters

    input_size : `int`
        The number of expected features in the input.
    hidden_size : `int`
        Number of features in the hidden state of the LSTM.
        Defaults to 32.
    go_forward : `bool`
        Whether to compute features left to right (forward)
        or right to left (backward).
    recurrent_dropout_probability : `float`
        Variational dropout probability to use. Defaults to 0.0.
    use_highway : `bool`
        If `True` we append a highway network to the outputs of the LSTM.
    use_input_projection_bias : `bool`
        If `True` we use a bias in our LSTM calculations, otherwise we don't.

    # Attributes

    cell : `AugmentedLSTMCell`
        `AugmentedLSTMCell` that is applied at every timestep.

    """

    def __init__(self, input_size: int, hidden_size: int, go_forward: bool=
        True, recurrent_dropout_probability: float=0.0, use_highway: bool=
        True, use_input_projection_bias: bool=True):
        super().__init__()
        self.embed_dim = input_size
        self.lstm_dim = hidden_size
        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.cell = AugmentedLSTMCell(self.embed_dim, self.lstm_dim, self.
            use_highway, use_input_projection_bias)

    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.
        Tensor, torch.Tensor]]=None) ->Tuple[PackedSequence, Tuple[torch.
        Tensor, torch.Tensor]]:
        """
        Warning: Would be better to use the BiAugmentedLstm class in a regular model

        Given an input batch of sequential data such as word embeddings, produces a single layer unidirectional
        AugmentedLSTM representation of the sequential input and new state tensors.

        # Parameters

        inputs : `PackedSequence`
            `bsize` sequences of shape `(len, input_dim)` each, in PackedSequence format
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing the initial hidden state and
            the cell state of each element in the batch. Each of these tensors have a dimension of
            (1 x bsize x nhid). Defaults to `None`.

        # Returns

        `Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]`
            AugmentedLSTM representation of input and the state of the LSTM `t = seq_len`.
            Shape of representation is (bsize x seq_len x representation_dim).
            Shape of each state is (1 x bsize x nhid).

        """
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError(
                'inputs must be PackedSequence but got %s' % type(inputs))
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
            dropout_mask = get_dropout_mask(self.
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


class BiAugmentedLstm(torch.nn.Module):
    """
    `BiAugmentedLstm` implements a generic AugmentedLSTM representation layer.
    BiAugmentedLstm is an LSTM which optionally appends an optional highway network to the output layer.
    Furthermore the dropout controls the level of variational dropout done.

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required.
        The dimension of the outputs of the LSTM.
    num_layers : `int`
        Number of recurrent layers. Eg. setting `num_layers=2`
        would mean stacking two LSTMs together to form a stacked LSTM,
        with the second LSTM taking in the outputs of the first LSTM and
        computing the final result. Defaults to 1.
    bias : `bool`
        If `True` we use a bias in our LSTM calculations, otherwise we don't.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        Variational dropout probability to use.
    bidirectional : `bool`
        If `True`, becomes a bidirectional LSTM. Defaults to `True`.
    padding_value : `float`, optional (default = `0.0`)
        Value for the padded elements. Defaults to 0.0.
    use_highway : `bool`, optional (default = `True`)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)

    # Returns

    output_accumulator : `PackedSequence`
        The outputs of the LSTM for each timestep. A tensor of shape (batch_size, max_timesteps, hidden_size) where
        for a given batch element, all outputs past the sequence length for that batch are zero tensors.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1,
        bias: bool=True, recurrent_dropout_probability: float=0.0,
        bidirectional: bool=False, padding_value: float=0.0, use_highway:
        bool=True) ->None:
        super().__init__()
        self.input_size = input_size
        self.padding_value = padding_value
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = use_highway
        self.use_bias = bias
        num_directions = int(self.bidirectional) + 1
        self.forward_layers = torch.nn.ModuleList()
        if self.bidirectional:
            self.backward_layers = torch.nn.ModuleList()
        lstm_embed_dim = self.input_size
        for _ in range(self.num_layers):
            self.forward_layers.append(AugmentedLstm(lstm_embed_dim, self.
                hidden_size, go_forward=True, recurrent_dropout_probability
                =self.recurrent_dropout_probability, use_highway=self.
                use_highway, use_input_projection_bias=self.use_bias))
            if self.bidirectional:
                self.backward_layers.append(AugmentedLstm(lstm_embed_dim,
                    self.hidden_size, go_forward=False,
                    recurrent_dropout_probability=self.
                    recurrent_dropout_probability, use_highway=self.
                    use_highway, use_input_projection_bias=self.use_bias))
            lstm_embed_dim = self.hidden_size * num_directions
        self.representation_dim = lstm_embed_dim

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.
        Tensor, torch.Tensor]]=None) ->Tuple[torch.Tensor, Tuple[torch.
        Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a AugmentedLSTM representation of the sequential input and new state
        tensors.

        # Parameters

        inputs : `PackedSequence`, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing
            the initial hidden state and the cell state of each element in
            the batch. Each of these tensors have a dimension of
            (bsize x num_layers x num_directions * nhid). Defaults to `None`.

        # Returns

        `Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
            AgumentedLSTM representation of input and
            the state of the LSTM `t = seq_len`.
            Shape of representation is (bsize x seq_len x representation_dim).
            Shape of each state is (bsize x num_layers * num_directions x nhid).

        """
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError(
                'inputs must be PackedSequence but got %s' % type(inputs))
        if self.bidirectional:
            return self._forward_bidirectional(inputs, states)
        return self._forward_unidirectional(inputs, states)

    def _forward_bidirectional(self, inputs: PackedSequence, states:
        Optional[Tuple[torch.Tensor, torch.Tensor]]):
        output_sequence = inputs
        final_h = []
        final_c = []
        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                'Initial states were passed to forward() but the number of initial states does not match the number of layers.'
                )
        else:
            hidden_states = list(zip(states[0].chunk(self.num_layers, 0),
                states[1].chunk(self.num_layers, 0)))
        for i, state in enumerate(hidden_states):
            if state:
                forward_state = state[0].chunk(2, -1)
                backward_state = state[1].chunk(2, -1)
            else:
                forward_state = backward_state = None
            forward_layer = self.forward_layers[i]
            backward_layer = self.backward_layers[i]
            forward_output, final_forward_state = forward_layer(output_sequence
                , forward_state)
            backward_output, final_backward_state = backward_layer(
                output_sequence, backward_state)
            forward_output, lengths = pad_packed_sequence(forward_output,
                batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output,
                batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(output_sequence, lengths,
                batch_first=True)
            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])
        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = final_h, final_c
        output_sequence, batch_lengths = pad_packed_sequence(output_sequence,
            padding_value=self.padding_value, batch_first=True)
        output_sequence = pack_padded_sequence(output_sequence,
            batch_lengths, batch_first=True)
        return output_sequence, final_state_tuple

    def _forward_unidirectional(self, inputs: PackedSequence, states:
        Optional[Tuple[torch.Tensor, torch.Tensor]]):
        output_sequence = inputs
        final_h = []
        final_c = []
        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                'Initial states were passed to forward() but the number of initial states does not match the number of layers.'
                )
        else:
            hidden_states = list(zip(states[0].chunk(self.num_layers, 0),
                states[1].chunk(self.num_layers, 0)))
        for i, state in enumerate(hidden_states):
            forward_layer = self.forward_layers[i]
            forward_output, final_forward_state = forward_layer(output_sequence
                , state)
            output_sequence = forward_output
            final_h.append(final_forward_state[0])
            final_c.append(final_forward_state[1])
        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = final_h, final_c
        output_sequence, batch_lengths = pad_packed_sequence(output_sequence,
            padding_value=self.padding_value, batch_first=True)
        output_sequence = pack_padded_sequence(output_sequence,
            batch_lengths, batch_first=True)
        return output_sequence, final_state_tuple


def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor
    ) ->torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError('Does not support torch.bool')
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def masked_max(vector: torch.Tensor, mask: torch.BoolTensor, dim: int,
    keepdim: bool=False) ->torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.
        dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError('Only supports floating point dtypes.')
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 0.0001
    else:
        raise TypeError('Does not support dtype ' + str(dtype))


def masked_mean(vector: torch.Tensor, mask: torch.BoolTensor, dim: int,
    keepdim: bool=False) ->torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(
        torch.float))


def masked_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int=-
    1, memory_efficient: bool=False) ->torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) +
                tiny_value_of_dtype(result.dtype))
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(
                vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def multi_perspective_match(vector1: torch.Tensor, vector2: torch.Tensor,
    weight: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate multi-perspective cosine matching between time-steps of vectors
    of the same length.

    # Parameters

    vector1 : `torch.Tensor`
        A tensor of shape `(batch, seq_len, hidden_size)`
    vector2 : `torch.Tensor`
        A tensor of shape `(batch, seq_len or 1, hidden_size)`
    weight : `torch.Tensor`
        A tensor of shape `(num_perspectives, hidden_size)`

    # Returns

    `torch.Tensor` :
        Shape `(batch, seq_len, 1)`.
    `torch.Tensor` :
        Shape `(batch, seq_len, num_perspectives)`.
    """
    assert vector1.size(0) == vector2.size(0)
    assert weight.size(1) == vector1.size(2) == vector1.size(2)
    similarity_single = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)
    weight = weight.unsqueeze(0).unsqueeze(0)
    vector1 = weight * vector1.unsqueeze(2)
    vector2 = weight * vector2.unsqueeze(2)
    similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)
    return similarity_single, similarity_multi


def multi_perspective_match_pairwise(vector1: torch.Tensor, vector2: torch.
    Tensor, weight: torch.Tensor) ->torch.Tensor:
    """
    Calculate multi-perspective cosine matching between each time step of
    one vector and each time step of another vector.

    # Parameters

    vector1 : `torch.Tensor`
        A tensor of shape `(batch, seq_len1, hidden_size)`
    vector2 : `torch.Tensor`
        A tensor of shape `(batch, seq_len2, hidden_size)`
    weight : `torch.Tensor`
        A tensor of shape `(num_perspectives, hidden_size)`

    # Returns

    `torch.Tensor` :
        A tensor of shape `(batch, seq_len1, seq_len2, num_perspectives)` consisting
        multi-perspective matching results
    """
    num_perspectives = weight.size(0)
    weight = weight.unsqueeze(0).unsqueeze(2)
    vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspectives, -1, -1
        )
    vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspectives, -1, -1
        )
    vector1_norm = vector1.norm(p=2, dim=3, keepdim=True)
    vector2_norm = vector2.norm(p=2, dim=3, keepdim=True)
    mul_result = torch.matmul(vector1, vector2.transpose(2, 3))
    norm_value = vector1_norm * vector2_norm.transpose(2, 3)
    return (mul_result / norm_value.clamp(min=tiny_value_of_dtype(
        norm_value.dtype))).permute(0, 2, 3, 1)


VITERBI_DECODING = Tuple[List[int], float]


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    # Parameters

    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]]=
        None, include_start_end_transitions: bool=True) ->None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))
        if constraints is None:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(
                1.0)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(
                0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask,
            requires_grad=False)
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.BoolTensor
        ) ->torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]
        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = util.logsumexp(inner, 1) * mask[i].view(batch_size, 1
                ) + alpha * (~mask[i]).view(batch_size, 1)
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        return util.logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor,
        mask: torch.BoolTensor) ->torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1),
                next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)
                ).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[
                i]
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(
            0)
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0,
                last_tags)
        else:
            last_transition_score = 0.0
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch
        .BoolTensor=None) ->torch.Tensor:
        """
        Computes the log likelihood.
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.BoolTensor=
        None, top_k: int=None) ->Union[List[VITERBI_DECODING], List[List[
        VITERBI_DECODING]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)

        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=
                logits.device)
        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False
        _, max_seq_length, num_tags = logits.size()
        logits, mask = logits.data, mask.data
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)
        constrained_transitions = self.transitions * self._constraint_mask[:
            num_tags, :num_tags] + -10000.0 * (1 - self._constraint_mask[:
            num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        if self.include_start_end_transitions:
            transitions[(start_tag), :num_tags
                ] = self.start_transitions.detach() * self._constraint_mask[(
                start_tag), :num_tags].data + -10000.0 * (1 - self.
                _constraint_mask[(start_tag), :num_tags].detach())
            transitions[:num_tags, (end_tag)] = self.end_transitions.detach(
                ) * self._constraint_mask[:num_tags, (end_tag)
                ].data + -10000.0 * (1 - self._constraint_mask[:num_tags, (
                end_tag)].detach())
        else:
            transitions[(start_tag), :num_tags] = -10000.0 * (1 - self.
                _constraint_mask[(start_tag), :num_tags].detach())
            transitions[:num_tags, (end_tag)] = -10000.0 * (1 - self.
                _constraint_mask[:num_tags, (end_tag)].detach())
        best_paths = []
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)
        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero().squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:sequence_length + 1, :num_tags] = masked_prediction
            tag_sequence[sequence_length + 1, end_tag] = 0.0
            viterbi_paths, viterbi_scores = util.viterbi_decode(tag_sequence
                =tag_sequence[:sequence_length + 2], transition_matrix=
                transitions, top_k=top_k)
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths,
                viterbi_scores):
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)
        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]
        return best_paths


logger = logging.getLogger(__name__)


def remove_sentence_boundaries(tensor: torch.Tensor, mask: torch.BoolTensor
    ) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps, dim)`
    this returns a tensor of shape `(batch_size, timesteps - 2, dim)` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., `mask[:, 0]` is assumed
    to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of `add_sentence_boundary_token_ids`.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`

    # Returns

    tensor_without_boundary_tokens : `torch.Tensor`
        The tensor after removing the boundary tokens of shape `(batch_size, timesteps - 2, dim)`
    new_mask : `torch.BoolTensor`
        The new mask for the tensor of shape `(batch_size, timesteps - 2)`.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[(i), :j - 2, :] = tensor[(i), 1:
                j - 1, :]
            new_mask[(i), :j - 2] = True
    return tensor_without_boundary_tokens, new_mask


def _make_bos_eos(character: int, padding_character: int,
    beginning_of_word_character: int, end_of_word_character: int,
    max_word_length: int):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.

    We allow to add optional additional special tokens with designated
    character ids with `tokens_to_add`.
    """
    max_word_length = 50
    beginning_of_sentence_character = 256
    end_of_sentence_character = 257
    beginning_of_word_character = 258
    end_of_word_character = 259
    padding_character = 260
    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character, padding_character,
        beginning_of_word_character, end_of_word_character, max_word_length)
    end_of_sentence_characters = _make_bos_eos(end_of_sentence_character,
        padding_character, beginning_of_word_character,
        end_of_word_character, max_word_length)
    bos_token = '<S>'
    eos_token = '</S>'

    def __init__(self, tokens_to_add: Dict[str, int]=None) ->None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) ->List[int]:
        if word in self.tokens_to_add:
            char_ids = [ELMoCharacterMapper.padding_character
                ] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:
                ELMoCharacterMapper.max_word_length - 2]
            char_ids = [ELMoCharacterMapper.padding_character
                ] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1
                ] = ELMoCharacterMapper.end_of_word_character
        return [(c + 1) for c in char_ids]

    def __eq__(self, other) ->bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


def add_sentence_boundary_token_ids(tensor: torch.Tensor, mask: torch.
    BoolTensor, sentence_begin_token: Any, sentence_end_token: Any) ->Tuple[
    torch.Tensor, torch.BoolTensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps)` or
    `(batch_size, timesteps, dim)` this returns a tensor of shape
    `(batch_size, timesteps + 2)` or `(batch_size, timesteps + 2, dim)` respectively.

    Returns both the new tensor and updated mask.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps)` or `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`
    sentence_begin_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `<S>` id. For 3D input, a tensor with length dim.
    sentence_end_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `</S>` id. For 3D input, a tensor with length dim.

    # Returns

    tensor_with_boundary_tokens : `torch.Tensor`
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : `torch.BoolTensor`
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, (0)] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[(i), (0), :] = sentence_begin_token
            tensor_with_boundary_tokens[(i), (j + 1), :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(dim=-1) > 0
    else:
        raise ValueError(
            'add_sentence_boundary_token_ids only accepts 2D and 3D input')
    return tensor_with_boundary_tokens, new_mask


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


RnnStateStorage = Tuple[torch.Tensor, ...]


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A batch first Pytorch tensor.
    sequence_lengths : `torch.LongTensor`, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    # Returns

    sorted_tensor : `torch.FloatTensor`
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : `torch.LongTensor`
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : `torch.LongTensor`
        Indices into the sorted_tensor such that
        `sorted_tensor.index_select(0, restoration_indices) == original_tensor`
    permutation_index : `torch.LongTensor`
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths,
        torch.Tensor):
        raise ConfigurationError(
            'Both the tensor and sequence lengths must be torch.Tensors.')
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0,
        descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=
        sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return (sorted_tensor, sorted_sequence_lengths, restoration_indices,
        permutation_index)


class _EncoderBase(torch.nn.Module):
    """
    This abstract class serves as a base for the 3 `Encoder` abstractions in AllenNLP.
    - [`Seq2SeqEncoders`](./seq2seq_encoders/seq2seq_encoder.md)
    - [`Seq2VecEncoders`](./seq2vec_encoders/seq2vec_encoder.md)

    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool=False) ->None:
        super().__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence,
        Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor],
        RnnState]], inputs: torch.Tensor, mask: torch.BoolTensor,
        hidden_state: Optional[RnnState]=None):
        """
        This function exists because Pytorch RNNs require that their inputs be sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a `PackedSequence` and some `hidden_state`, which can either be a
        tuple of tensors or a tensor.

        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module, which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.

        # Parameters

        module : `Callable[RnnInputs, RnnOutputs]`
            A function to run on the inputs, where
            `RnnInputs: [PackedSequence, Optional[RnnState]]` and
            `RnnOutputs: Tuple[Union[PackedSequence, torch.Tensor], RnnState]`.
            In most cases, this is a `torch.nn.Module`.
        inputs : `torch.Tensor`, required.
            A tensor of shape `(batch_size, sequence_length, embedding_size)` representing
            the inputs to the Encoder.
        mask : `torch.BoolTensor`, required.
            A tensor of shape `(batch_size, sequence_length)`, representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : `Optional[RnnState]`, (default = `None`).
            A single tensor of shape (num_layers, batch_size, hidden_size) representing the
            state of an RNN with or a tuple of
            tensors of shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden state and memory
            state of an LSTM-like RNN.

        # Returns

        module_output : `Union[torch.Tensor, PackedSequence]`.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
            The batch size dimension will be equal to `num_valid`, as sequences of zero
            length are clipped off before the module is called, as Pytorch cannot handle
            zero length sequences.
        final_states : `Optional[RnnState]`
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : `torch.LongTensor`
            A tensor of shape `(batch_size,)`, describing the re-indexing required to transform
            the outputs back to their original batch order.
        """
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, (0)]).int().item()
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        (sorted_inputs, sorted_sequence_lengths, restoration_indices,
            sorting_indices) = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:
            num_valid, :, :], sorted_sequence_lengths[:num_valid].data.
            tolist(), batch_first=True)
        if not self.stateful:
            if hidden_state is None:
                initial_states: Any = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:,
                    :num_valid, :].contiguous() for state in hidden_state]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid,
                sorting_indices)
        module_output, final_states = module(packed_sequence_input,
            initial_states)
        return module_output, final_states, restoration_indices

    def _get_initial_states(self, batch_size: int, num_valid: int,
        sorting_indices: torch.LongTensor) ->Optional[RnnState]:
        """
        Returns an initial state for use in an RNN. Additionally, this method handles
        the batch size changing across calls by mutating the state to append initial states
        for new elements in the batch. Finally, it also handles sorting the states
        with respect to the sequence lengths of elements in the batch and removing rows
        which are completely padded. Importantly, this `mutates` the state if the
        current batch size is larger than when it was previously called.

        # Parameters

        batch_size : `int`, required.
            The batch size can change size across calls to stateful RNNs, so we need
            to know if we need to expand or shrink the states before returning them.
            Expanded states will be set to zero.
        num_valid : `int`, required.
            The batch may contain completely padded sequences which get removed before
            the sequence is passed through the encoder. We also need to clip these off
            of the state too.
        sorting_indices `torch.LongTensor`, required.
            Pytorch RNNs take sequences sorted by length. When we return the states to be
            used for a given call to `module.forward`, we need the states to match up to
            the sorted sequences, so before returning them, we sort the states using the
            same indices used to sort the sequences.

        # Returns

        This method has a complex return type because it has to deal with the first time it
        is called, when it has no state, and the fact that types of RNN have heterogeneous
        states.

        If it is the first time the module has been called, it returns `None`, regardless
        of the type of the `Module`.

        Otherwise, for LSTMs, it returns a tuple of `torch.Tensors` with shape
        `(num_layers, num_valid, state_size)` and `(num_layers, num_valid, memory_size)`
        respectively, or for GRUs, it returns a single `torch.Tensor` of shape
        `(num_layers, num_valid, state_size)`.
        """
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = state.new_zeros(state.size(0), num_states_to_concat,
                    state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(state[:, :batch_size, :] for
                state in self._states)
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1,
                sorting_indices)
            return sorted_state[:, :num_valid, :].contiguous()
        else:
            sorted_states = [state.index_select(1, sorting_indices) for
                state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :].contiguous() for state in
                sorted_states)

    def _update_states(self, final_states: RnnStateStorage,
        restoration_indices: torch.LongTensor) ->None:
        """
        After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detaches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.

        # Parameters

        final_states : `RnnStateStorage`, required.
            The hidden states returned as output from the RNN.
        restoration_indices : `torch.LongTensor`, required.
            The indices that invert the sorting used in `sort_and_run_forward`
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        new_unsorted_states = [state.index_select(1, restoration_indices) for
            state in final_states]
        if self._states is None:
            self._states = tuple(state.data for state in new_unsorted_states)
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [(state[(0), :, :].sum(-1) != 0.0).float()
                .view(1, new_state_batch_size, 1) for state in
                new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states,
                    new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :
                        ] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :
                        ] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states,
                    new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())
            self._states = tuple(new_states)

    def reset_states(self, mask: torch.BoolTensor=None) ->None:
        """
        Resets the internal states of a stateful encoder.

        # Parameters

        mask : `torch.BoolTensor`, optional.
            A tensor of shape `(batch_size,)` indicating which states should
            be reset. If not provided, all states will be reset.
        """
        if mask is None:
            self._states = None
        else:
            mask_batch_size = mask.size(0)
            mask = mask.view(1, mask_batch_size, 1)
            new_states = []
            for old_state in self._states:
                old_state_batch_size = old_state.size(1)
                if old_state_batch_size != mask_batch_size:
                    raise ValueError(
                        f'Trying to reset states using mask with incorrect batch size. Expected batch size: {old_state_batch_size}. Provided batch size: {mask_batch_size}.'
                        )
                new_state = ~mask * old_state
                new_states.append(new_state.detach())
            self._states = tuple(new_states)


class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.

        # Parameters

        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`

        # Returns

        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0],
            input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.
            training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


class LayerNorm(torch.nn.Module):
    """
    An implementation of [Layer Normalization](
    https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5).

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    # Parameters

    dimension : `int`, required.
        The dimension of the layer output to normalize.

    # Returns

    The normalized layer output.
    """

    def __init__(self, dimension: int) ->None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, tensor: torch.Tensor):
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        return self.gamma * (tensor - mean) / (std + util.
            tiny_value_of_dtype(std.dtype)) + self.beta


class LstmCellWithProjection(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.

    [0]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required.
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required.
        The dimension of the outputs of the LSTM.
    cell_size : `int`, required.
        The dimension of the memory cell used for the LSTM.
    go_forward : `bool`, optional (default = `True`)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
        [0]. Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value : `float`, optional, (default = `None`)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value : `float`, optional, (default = `None`)
        The magnitude with which to clip the memory cell.

    # Returns

    output_accumulator : `torch.FloatTensor`
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state : `Tuple[torch.FloatTensor, torch.FloatTensor]`
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(self, input_size: int, hidden_size: int, cell_size: int,
        go_forward: bool=True, recurrent_dropout_probability: float=0.0,
        memory_cell_clip_value: Optional[float]=None,
        state_projection_clip_value: Optional[float]=None) ->None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size,
            bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size,
            bias=True)
        self.state_projection = torch.nn.Linear(cell_size, hidden_size,
            bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size,
            self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size,
            self.hidden_size])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size
            ].fill_(1.0)

    def forward(self, inputs: torch.FloatTensor, batch_lengths: List[int],
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        # Parameters

        inputs : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : `List[int]`, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The `state` has shape (1, batch_size, hidden_size) and the
            `memory` has shape (1, batch_size, cell_size).

        # Returns

        output_accumulator : `torch.FloatTensor`
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : `Tuple[torch.FloatTensor, torch.FloatTensor]`
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The `state` has shape (1, batch_size, hidden_size) and the
            `memory` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]
        output_accumulator = inputs.new_zeros(batch_size, total_timesteps,
            self.hidden_size)
        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.
                cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.
                hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.
                recurrent_dropout_probability, full_batch_previous_state)
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
            timestep_input = inputs[0:current_length_index + 1, (index)]
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            input_gate = torch.sigmoid(projected_input[:, 0 * self.
                cell_size:1 * self.cell_size] + projected_state[:, 0 * self
                .cell_size:1 * self.cell_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.
                cell_size:2 * self.cell_size] + projected_state[:, 1 * self
                .cell_size:2 * self.cell_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.cell_size:
                3 * self.cell_size] + projected_state[:, 2 * self.cell_size
                :3 * self.cell_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.
                cell_size:4 * self.cell_size] + projected_state[:, 3 * self
                .cell_size:4 * self.cell_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            if self.memory_cell_clip_value:
                memory = torch.clamp(memory, -self.memory_cell_clip_value,
                    self.memory_cell_clip_value)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)
            timestep_output = self.state_projection(
                pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output, -self.
                    state_projection_clip_value, self.
                    state_projection_clip_value)
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0:
                    current_length_index + 1]
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1
                ] = timestep_output
            output_accumulator[0:current_length_index + 1, (index)
                ] = timestep_output
        final_state = full_batch_previous_state.unsqueeze(0
            ), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float=0.1) ->None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size

    def forward(self, tensor: torch.Tensor, mask: torch.BoolTensor
        ) ->torch.Tensor:
        broadcast_mask = mask.unsqueeze(-1)
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt((masked_centered * masked_centered).sum() /
            num_elements + util.tiny_value_of_dtype(tensor.dtype))
        return self.gamma * (tensor - mean) / (std + util.
            tiny_value_of_dtype(tensor.dtype)) + self.beta


class ResidualWithLayerDropout(torch.nn.Module):
    """
    A residual connection with the layer dropout technique [Deep Networks with Stochastic
    Depth](https://arxiv.org/pdf/1603.09382.pdf).

    This module accepts the input and output of a layer, decides whether this layer should
    be stochastically dropped, returns either the input or output + input. During testing,
    it will re-calibrate the outputs of this layer by the expected number of times it
    participates in training.
    """

    def __init__(self, undecayed_dropout_prob: float=0.5) ->None:
        super().__init__()
        if undecayed_dropout_prob < 0 or undecayed_dropout_prob > 1:
            raise ValueError(
                f'undecayed dropout probability has to be between 0 and 1, but got {undecayed_dropout_prob}'
                )
        self.undecayed_dropout_prob = undecayed_dropout_prob

    def forward(self, layer_input: torch.Tensor, layer_output: torch.Tensor,
        layer_index: int=None, total_layers: int=None) ->torch.Tensor:
        """
        Apply dropout to this layer, for this whole mini-batch.
        dropout_prob = layer_index / total_layers * undecayed_dropout_prob if layer_idx and
        total_layers is specified, else it will use the undecayed_dropout_prob directly.

        # Parameters

        layer_input `torch.FloatTensor` required
            The input tensor of this layer.
        layer_output `torch.FloatTensor` required
            The output tensor of this layer, with the same shape as the layer_input.
        layer_index `int`
            The layer index, starting from 1. This is used to calcuate the dropout prob
            together with the `total_layers` parameter.
        total_layers `int`
            The total number of layers.

        # Returns

        output : `torch.FloatTensor`
            A tensor with the same shape as `layer_input` and `layer_output`.
        """
        if layer_index is not None and total_layers is not None:
            dropout_prob = (1.0 * self.undecayed_dropout_prob * layer_index /
                total_layers)
        else:
            dropout_prob = 1.0 * self.undecayed_dropout_prob
        if self.training:
            if torch.rand(1) < dropout_prob:
                return layer_input
            else:
                return layer_output + layer_input
        else:
            return (1 - dropout_prob) * layer_output + layer_input


def _choice(num_words: int, num_samples: int) ->Tuple[np.ndarray, int]:
    """
    Chooses `num_samples` samples without replacement from [0, ..., num_words).
    Returns a tuple (samples, num_tries).
    """
    num_tries = 0
    num_chosen = 0

    def get_buffer() ->np.ndarray:
        log_samples = np.random.rand(num_samples) * np.log(num_words + 1)
        samples = np.exp(log_samples).astype('int64') - 1
        return np.clip(samples, a_min=0, a_max=num_words - 1)
    sample_buffer = get_buffer()
    buffer_index = 0
    samples: Set[int] = set()
    while num_chosen < num_samples:
        num_tries += 1
        sample_id = sample_buffer[buffer_index]
        if sample_id not in samples:
            samples.add(sample_id)
            num_chosen += 1
        buffer_index += 1
        if buffer_index == num_samples:
            sample_buffer = get_buffer()
            buffer_index = 0
    return np.array(list(samples)), num_tries


class SampledSoftmaxLoss(torch.nn.Module):
    """
    Based on the default log_uniform_candidate_sampler in tensorflow.

    !!! NOTE
        num_words DOES NOT include padding id.

    !!! NOTE
        In all cases except (tie_embeddings=True and use_character_inputs=False)
        the weights are dimensioned as num_words and do not include an entry for the padding (0) id.
        For the (tie_embeddings=True and use_character_inputs=False) case,
        then the embeddings DO include the extra 0 padding, to be consistent with the word embedding layer.

    # Parameters

    num_words, `int`, required
        The number of words in the vocabulary
    embedding_dim, `int`, required
        The dimension to softmax over
    num_samples, `int`, required
        During training take this many samples. Must be less than num_words.
    sparse, `bool`, optional (default = `False`)
        If this is true, we use a sparse embedding matrix.
    unk_id, `int`, optional (default = `None`)
        If provided, the id that represents unknown characters.
    use_character_inputs, `bool`, optional (default = `True`)
        Whether to use character inputs
    use_fast_sampler, `bool`, optional (default = `False`)
        Whether to use the fast cython sampler.
    """

    def __init__(self, num_words: int, embedding_dim: int, num_samples: int,
        sparse: bool=False, unk_id: int=None, use_character_inputs: bool=
        True, use_fast_sampler: bool=False) ->None:
        super().__init__()
        self.tie_embeddings = False
        assert num_samples < num_words
        if use_fast_sampler:
            raise ConfigurationError('fast sampler is not implemented')
        else:
            self.choice_func = _choice
        if sparse:
            self.softmax_w = torch.nn.Embedding(num_embeddings=num_words,
                embedding_dim=embedding_dim, sparse=True)
            self.softmax_w.weight.data.normal_(mean=0.0, std=1.0 / np.sqrt(
                embedding_dim))
            self.softmax_b = torch.nn.Embedding(num_embeddings=num_words,
                embedding_dim=1, sparse=True)
            self.softmax_b.weight.data.fill_(0.0)
        else:
            self.softmax_w = torch.nn.Parameter(torch.randn(num_words,
                embedding_dim) / np.sqrt(embedding_dim))
            self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))
        self.sparse = sparse
        self.use_character_inputs = use_character_inputs
        if use_character_inputs:
            self._unk_id = unk_id
        self._num_samples = num_samples
        self._embedding_dim = embedding_dim
        self._num_words = num_words
        self.initialize_num_words()

    def initialize_num_words(self):
        if self.sparse:
            num_words = self.softmax_w.weight.size(0)
        else:
            num_words = self.softmax_w.size(0)
        self._num_words = num_words
        self._log_num_words_p1 = np.log(num_words + 1)
        self._probs = (np.log(np.arange(num_words) + 2) - np.log(np.arange(
            num_words) + 1)) / self._log_num_words_p1

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor,
        target_token_embedding: torch.Tensor=None) ->torch.Tensor:
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0)
        if not self.training:
            return self._forward_eval(embeddings, targets)
        else:
            return self._forward_train(embeddings, targets,
                target_token_embedding)

    def _forward_train(self, embeddings: torch.Tensor, targets: torch.
        Tensor, target_token_embedding: torch.Tensor) ->torch.Tensor:
        sampled_ids, target_expected_count, sampled_expected_count = (self.
            log_uniform_candidate_sampler(targets, choice_func=self.
            choice_func))
        long_targets = targets.long()
        long_targets.requires_grad_(False)
        all_ids = torch.cat([long_targets, sampled_ids], dim=0)
        if self.sparse:
            all_ids_1 = all_ids.unsqueeze(1)
            all_w = self.softmax_w(all_ids_1).squeeze(1)
            all_b = self.softmax_b(all_ids_1).squeeze(2).squeeze(1)
        else:
            all_w = torch.nn.functional.embedding(all_ids, self.softmax_w)
            all_b = torch.nn.functional.embedding(all_ids, self.softmax_b.
                unsqueeze(1)).squeeze(1)
        batch_size = long_targets.size(0)
        true_w = all_w[:batch_size, :]
        sampled_w = all_w[batch_size:, :]
        true_b = all_b[:batch_size]
        sampled_b = all_b[batch_size:]
        true_logits = (true_w * embeddings).sum(dim=1) + true_b - torch.log(
            target_expected_count + util.tiny_value_of_dtype(
            target_expected_count.dtype))
        sampled_logits = torch.matmul(embeddings, sampled_w.t()
            ) + sampled_b - torch.log(sampled_expected_count + util.
            tiny_value_of_dtype(sampled_expected_count.dtype))
        true_in_sample_mask = sampled_ids == long_targets.unsqueeze(1)
        masked_sampled_logits = sampled_logits.masked_fill(true_in_sample_mask,
            -10000.0)
        logits = torch.cat([true_logits.unsqueeze(1), masked_sampled_logits
            ], dim=1)
        log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        nll_loss = -1.0 * log_softmax[:, (0)].sum()
        return nll_loss

    def _forward_eval(self, embeddings: torch.Tensor, targets: torch.Tensor
        ) ->torch.Tensor:
        if self.sparse:
            w = self.softmax_w.weight
            b = self.softmax_b.weight.squeeze(1)
        else:
            w = self.softmax_w
            b = self.softmax_b
        log_softmax = torch.nn.functional.log_softmax(torch.matmul(
            embeddings, w.t()) + b, dim=-1)
        if self.tie_embeddings and not self.use_character_inputs:
            targets_ = targets + 1
        else:
            targets_ = targets
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(),
            reduction='sum')

    def log_uniform_candidate_sampler(self, targets, choice_func=_choice):
        np_sampled_ids, num_tries = choice_func(self._num_words, self.
            _num_samples)
        sampled_ids = torch.from_numpy(np_sampled_ids)
        target_probs = torch.log((targets.float() + 2.0) / (targets.float() +
            1.0)) / self._log_num_words_p1
        target_expected_count = -1.0 * (torch.exp(num_tries * torch.log1p(-
            target_probs)) - 1.0)
        sampled_probs = torch.log((sampled_ids.float() + 2.0) / (
            sampled_ids.float() + 1.0)) / self._log_num_words_p1
        sampled_expected_count = -1.0 * (torch.exp(num_tries * torch.log1p(
            -sampled_probs)) - 1.0)
        sampled_ids.requires_grad_(False)
        target_expected_count.requires_grad_(False)
        sampled_expected_count.requires_grad_(False)
        return sampled_ids, target_expected_count, sampled_expected_count


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.

    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(self, mixture_size: int, do_layer_norm: bool=False,
        initial_scalar_parameters: List[float]=None, trainable: bool=True
        ) ->None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError(
                'Length of initial_scalar_parameters {} differs from mixture_size {}'
                .format(initial_scalar_parameters, mixture_size))
        self.scalar_parameters = ParameterList([Parameter(torch.FloatTensor
            ([initial_scalar_parameters[i]]), requires_grad=trainable) for
            i in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=
            trainable)

    def forward(self, tensors: List[torch.Tensor], mask: torch.BoolTensor=None
        ) ->torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.

        When `do_layer_norm=False` the `mask` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError(
                '{} tensors were passed, but the module was initialized to mix {} tensors.'
                .format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2
                ) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + util.
                tiny_value_of_dtype(variance.dtype))
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for
            parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)
        else:
            broadcast_mask = mask.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask) * input_dim
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                    broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)


class ResidualBlock(torch.nn.Module):

    def __init__(self, input_dim: int, layers: Sequence[Sequence[int]],
        direction: str, do_weight_norm: bool=True, dropout: float=0.0) ->None:
        super().__init__()
        self.dropout = dropout
        self._convolutions = torch.nn.ModuleList()
        last_dim = input_dim
        for k, layer in enumerate(layers):
            if len(layer) == 2:
                conv = torch.nn.Conv1d(last_dim, layer[1] * 2, layer[0],
                    stride=1, padding=layer[0] - 1, bias=True)
            elif len(layer) == 3:
                assert layer[0] == 2, 'only support kernel = 2 for now'
                conv = torch.nn.Conv1d(last_dim, layer[1] * 2, layer[0],
                    stride=1, padding=layer[2], dilation=layer[2], bias=True)
            else:
                raise ValueError('each layer must have length 2 or 3')
            if k == 0:
                conv_dropout = dropout
            else:
                conv_dropout = 0.0
            std = math.sqrt(4 * (1.0 - conv_dropout) / (layer[0] * last_dim))
            conv.weight.data.normal_(0, std=std)
            conv.bias.data.zero_()
            if do_weight_norm:
                conv = torch.nn.utils.weight_norm(conv, name='weight', dim=0)
            self._convolutions.append(conv)
            last_dim = layer[1]
        assert last_dim == input_dim
        if direction not in ('forward', 'backward'):
            raise ConfigurationError(f'invalid direction: {direction}')
        self._direction = direction

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = x
        timesteps = x.size(2)
        for k, convolution in enumerate(self._convolutions):
            if k == 0 and self.dropout > 0:
                out = torch.nn.functional.dropout(out, self.dropout, self.
                    training)
            conv_out = convolution(out)
            dims_to_remove = conv_out.size(2) - timesteps
            if dims_to_remove > 0:
                if self._direction == 'forward':
                    conv_out = conv_out.narrow(2, 0, timesteps)
                else:
                    conv_out = conv_out.narrow(2, dims_to_remove, timesteps)
            out = torch.nn.functional.glu(conv_out, dim=1)
        return (out + x) * math.sqrt(0.5)


class SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """

    def __init__(self, num_words: int, embedding_dim: int) ->None:
        super().__init__()
        self.tie_embeddings = False
        self.softmax_w = torch.nn.Parameter(torch.randn(embedding_dim,
            num_words) / np.sqrt(embedding_dim))
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor
        ) ->torch.Tensor:
        probs = torch.nn.functional.log_softmax(torch.matmul(embeddings,
            self.softmax_w) + self.softmax_b, dim=-1)
        return torch.nn.functional.nll_loss(probs, targets.long(),
            reduction='sum')


TensorPair = Tuple[torch.Tensor, torch.Tensor]


class StackedAlternatingLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in [Deep Semantic Role Labelling - What works and what's next][0].

    [0]: https://www.aclweb.org/anthology/P17-1044.pdf
    [1]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required
        The dimension of the outputs of the LSTM.
    num_layers : `int`, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks][1].
    use_input_projection_bias : `bool`, optional (default = `True`)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    # Returns

    output_accumulator : `PackedSequence`
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
        recurrent_dropout_probability: float=0.0, use_highway: bool=True,
        use_input_projection_bias: bool=True) ->None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            go_forward = layer_index % 2 == 0
            layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway, use_input_projection_bias=
                use_input_projection_bias)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self, inputs: PackedSequence, initial_state: Optional[
        TensorPair]=None) ->Tuple[Union[torch.Tensor, PackedSequence],
        TensorPair]:
        """
        # Parameters

        inputs : `PackedSequence`, required.
            A batch first `PackedSequence` to run the stacked LSTM over.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        # Returns

        output_sequence : `PackedSequence`
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: `Tuple[torch.Tensor, torch.Tensor]`
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.
                lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                'Initial states were passed to forward() but the number of initial states does not match the number of layers.'
                )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                initial_state[1].split(1, 0)))
        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, 'layer_{}'.format(i))
            output_sequence, final_state = layer(output_sequence, state)
            final_states.append(final_state)
        final_hidden_state, final_cell_state = tuple(torch.cat(state_list, 
            0) for state_list in zip(*final_states))
        return output_sequence, (final_hidden_state, final_cell_state)


class StackedBidirectionalLstm(torch.nn.Module):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states and outputs of each layer apart
    from the last layer of the LSTM. Note that this will be slower, as it
    doesn't use CUDNN.

    [0]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required
        The dimension of the outputs of the LSTM.
    num_layers : `int`, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The recurrent dropout probability to be used in a dropout scheme as
        stated in [A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks][0].
    layer_dropout_probability : `float`, optional (default = `0.0`)
        The layer wise dropout probability to be used in a dropout scheme as
        stated in [A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks][0].
    use_highway : `bool`, optional (default = `True`)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
        recurrent_dropout_probability: float=0.0, layer_dropout_probability:
        float=0.0, use_highway: bool=True) ->None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            forward_layer = AugmentedLstm(lstm_input_size, hidden_size,
                go_forward=True, recurrent_dropout_probability=
                recurrent_dropout_probability, use_highway=use_highway,
                use_input_projection_bias=False)
            backward_layer = AugmentedLstm(lstm_input_size, hidden_size,
                go_forward=False, recurrent_dropout_probability=
                recurrent_dropout_probability, use_highway=use_highway,
                use_input_projection_bias=False)
            lstm_input_size = hidden_size * 2
            self.add_module('forward_layer_{}'.format(layer_index),
                forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index),
                backward_layer)
            layers.append([forward_layer, backward_layer])
        self.lstm_layers = layers
        self.layer_dropout = InputVariationalDropout(layer_dropout_probability)

    def forward(self, inputs: PackedSequence, initial_state: Optional[
        TensorPair]=None) ->Tuple[PackedSequence, TensorPair]:
        """
        # Parameters

        inputs : `PackedSequence`, required.
            A batch first `PackedSequence` to run the stacked LSTM over.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (num_layers, batch_size, output_dimension * 2).

        # Returns

        output_sequence : `PackedSequence`
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: `torch.Tensor`
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers * 2, batch_size, hidden_size * 2).
        """
        if initial_state is None:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.
                lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                'Initial states were passed to forward() but the number of initial states does not match the number of layers.'
                )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                initial_state[1].split(1, 0)))
        output_sequence = inputs
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(i))
            backward_layer = getattr(self, 'backward_layer_{}'.format(i))
            forward_output, final_forward_state = forward_layer(output_sequence
                , state)
            backward_output, final_backward_state = backward_layer(
                output_sequence, state)
            forward_output, lengths = pad_packed_sequence(forward_output,
                batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output,
                batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            if i < self.num_layers - 1:
                output_sequence = self.layer_dropout(output_sequence)
            output_sequence = pack_padded_sequence(output_sequence, lengths,
                batch_first=True)
            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])
        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = final_h, final_c
        return output_sequence, final_state_tuple


class _Net1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_2 = torch.nn.Linear(10, 5)

    def forward(self, inputs):
        pass


class _Net2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_3 = torch.nn.Linear(10, 5)

    def forward(self, inputs):
        pass


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_allenai_allennlp(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(InputVariationalDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(LayerNorm(*[], **{'dimension': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(LstmCellWithProjection(*[], **{'input_size': 4, 'hidden_size': 4, 'cell_size': 4}), [torch.rand([4, 4, 4]), [4, 4, 4, 4]], {})

    @_fails_compile()
    def test_003(self):
        self._check(MaskedLayerNorm(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(ResidualWithLayerDropout(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(ScalarMix(*[], **{'mixture_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(_Net1(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(_Net2(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

