import sys
_module = sys.modules[__name__]
del sys
addons = _module
paired = _module
reader_conllcased = _module
reader_pandas = _module
reader_parallel_classify = _module
reporting_xpctl = _module
vec_text = _module
analyze_calibration = _module
bio_to_iobes = _module
convert_huggingface_checkpoints = _module
iob_to_bio = _module
iob_to_iobes = _module
iobes_result_to_bio = _module
transformer_utils = _module
baseline = _module
bleu = _module
confusion = _module
conlleval = _module
data = _module
embeddings = _module
mime_type = _module
model = _module
progress = _module
pytorch = _module
classify = _module
model = _module
train = _module
embeddings = _module
lm = _module
model = _module
train = _module
optz = _module
remote = _module
seq2seq = _module
decoders = _module
encoders = _module
model = _module
train = _module
tagger = _module
model = _module
train = _module
torchy = _module
transformer = _module
reader = _module
reporting = _module
services = _module
tensorflow_serving = _module
apis = _module
classification_pb2 = _module
classification_pb2_grpc = _module
get_model_metadata_pb2 = _module
get_model_metadata_pb2_grpc = _module
inference_pb2 = _module
inference_pb2_grpc = _module
input_pb2 = _module
input_pb2_grpc = _module
model_management_pb2 = _module
model_management_pb2_grpc = _module
model_pb2 = _module
model_pb2_grpc = _module
model_service_pb2 = _module
model_service_pb2_grpc = _module
predict_pb2 = _module
predict_pb2_grpc = _module
prediction_service_pb2 = _module
prediction_service_pb2_grpc = _module
regression_pb2 = _module
regression_pb2_grpc = _module
tf = _module
training = _module
datasets = _module
distributed = _module
eager = _module
feed = _module
utils = _module
v1 = _module
v2 = _module
tfy = _module
utils = _module
vectorizers = _module
version = _module
w2v = _module
eight_mile = _module
calibration = _module
metrics = _module
calibration_error = _module
plot = _module
confidence_histogram = _module
reliability_diagram = _module
embeddings = _module
layers = _module
optz = _module
serialize = _module
layers = _module
utils = _module
setup = _module
mead = _module
clean = _module
eval = _module
export = _module
exporters = _module
preprocessors = _module
exporters = _module
tasks = _module
preproc_exporters = _module
trainer = _module
bump = _module
compare_calibrations = _module
download_all = _module
lr_compare = _module
lr_find = _module
lr_visualize = _module
speed_test = _module
report = _module
run = _module
speed_tests = _module
test_bump = _module
test_beam_pytorch = _module
test_beam_tensorflow = _module
test_bleu = _module
test_calc_feats = _module
test_cm = _module
test_conll = _module
test_crf_pytorch = _module
test_crf_tensorflow = _module
test_decay = _module
test_decoders_pytorch = _module
test_decoders_tensorflow = _module
test_embeddings = _module
test_hash_utils = _module
test_iobes = _module
test_label_first_data_utils = _module
test_layers_pytorch = _module
test_layers_tf1 = _module
test_layers_tf2 = _module
test_lr_sched_tf1 = _module
test_lr_sched_tf2 = _module
test_mead_tasks = _module
test_mead_utils = _module
test_model = _module
test_parallel_conv = _module
test_parse_extra_args = _module
test_pytorch_masks = _module
test_pytorch_transformer = _module
test_pytorch_variational_dropout = _module
test_pytorch_weight_sharing = _module
test_read_files = _module
test_readers = _module
test_reporting_hooks = _module
test_rnn_dropout = _module
test_sample = _module
test_tf_ema = _module
test_tf_transformer = _module
test_tf_weight_sharing = _module
test_tlm_serialization = _module
test_torchy = _module
test_transition_masks = _module
test_utils = _module
test_vectorizers = _module

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


from torch.utils.data.dataset import IterableDataset


from torch.utils.data.dataset import TensorDataset


import torch


import numpy as np


from typing import Tuple


from typing import Dict


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import logging


import time


from collections import Counter


from collections import namedtuple


from torch.nn.parallel import DistributedDataParallel


import random


import torch.backends.cudnn as cudnn


import torch.autograd


import math


from functools import partial


from torch.autograd import Variable


import copy


import torch.nn as nn


from typing import Optional


from typing import List


from collections import defaultdict


import re


from functools import wraps


from typing import Set


import collections


from collections import OrderedDict


from typing import Union


import torch.jit as jit


import tensorflow as tf


from typing import Any


import inspect


from itertools import chain


from typing import Pattern


from typing import TextIO


from functools import update_wrapper


from copy import deepcopy


import string


class TripletLoss(nn.Module):
    """Provide a Triplet Loss using the reversed batch for negatives"""

    def __init__(self, model):
        super().__init__()
        self.score = nn.CosineSimilarity(dim=1)
        self.model = model

    def forward(self, inputs, targets):
        neg = targets.flip(0)
        query = self.model.encode_query(inputs)
        response = self.model.encode_response(targets)
        neg_response = self.model.encode_response(neg)
        pos_score = self.score(query, response)
        neg_score = self.score(query, neg_response)
        score = neg_score - pos_score
        score = score.masked_fill(score < 0.0, 0.0).sum(0)
        return score


def vec_log_sum_exp(vec: torch.Tensor, dim: int) ->torch.Tensor:
    """Vectorized version of log-sum-exp

    :param vec: Vector
    :param dim: What dimension to operate on
    :return:
    """
    max_scores, idx = torch.max(vec, dim, keepdim=True)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), dim, keepdim=True))


class AllLoss(nn.Module):

    def __init__(self, model, warmup_steps=10000):
        """Loss from here https://arxiv.org/pdf/1705.00652.pdf see section 4

        We want to minimize the negative log prob of y given x

        -log P(y|x)

        P(y|x) P(x) = P(x, y)                             Chain Rule of Probability
        P(y|x) = P(x, y) / P(x)                           Algebra
        P(y|x) = P(x, y) / \\sum_\\hat(y) P(x, y = \\hat(y)) Marginalize over all possible ys to get the probability of x
        P_approx(y|x) = P(x, y) / \\sum_i^k P(x, y_k)      Approximate the Marginalization by just using the ys in the batch

        S(x, y) is the score (cosine similarity between x and y in this case) from our neural network
        P(x, y) = e^S(x, y)

        P(y|x) = e^S(x, y) / \\sum_i^k e^S(x, y_k)
        log P(y|x) = log( e^S(x, y) / \\sum_i^k e^S(x, y_k))
        log P(y|x) = S(x, y) - log \\sum_i^k e^S(x, y_k)
        -log P(y|x) = -(S(x, y) - log \\sum_i^k e^S(x, y_k))
        """
        super().__init__()
        self.score = nn.CosineSimilarity(dim=-1)
        self.model = model
        self.max_scale = math.sqrt(self.model.embedding_layers.get_dsz())
        self.steps = 0
        self.warmup_steps = warmup_steps

    def forward(self, inputs, targets):
        fract = min(self.steps / self.warmup_steps, 1)
        c = (self.max_scale - 1) * fract + 1
        self.steps += 1
        query = self.model.encode_query(inputs).unsqueeze(1)
        response = self.model.encode_response(targets).unsqueeze(0)
        all_score = c * self.score(query, response)
        pos_score = torch.diag(all_score)
        loss = pos_score - vec_log_sum_exp(all_score, -1).squeeze()
        loss = torch.sum(loss)
        return -loss


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    return x * tf.nn.sigmoid(x)


def get_activation(name: str='relu'):
    if name is None or name == 'ident':
        return tf.nn.identity
    if name == 'softmax':
        return tf.nn.softmax
    if name == 'tanh':
        return tf.nn.tanh
    if name == 'sigmoid':
        return tf.nn.sigmoid
    if name == 'gelu':
        return gelu
    if name == 'swish':
        return swish
        return tf.identity
    if name == 'leaky_relu':
        return tf.nn.leaky_relu
    return tf.nn.relu


def pytorch_linear(in_sz: int, out_sz: int, unif: float=0, initializer: str=None, bias: bool=True):
    """Utility function that wraps a linear (AKA dense) layer creation, with options for weight init and bias"""
    l = nn.Linear(in_sz, out_sz, bias=bias)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    elif initializer == 'ortho':
        nn.init.orthogonal(l.weight)
    elif initializer == 'he' or initializer == 'kaiming':
        nn.init.kaiming_uniform(l.weight)
    else:
        nn.init.xavier_uniform_(l.weight)
    if bias:
        l.bias.data.zero_()
    return l


class Dense(nn.Module):
    """Dense (Linear) layer with optional activation given

    This module is the equivalent of the tf.keras.layer.Dense, module with optional activations applied
    """

    def __init__(self, insz: int, outsz: int, activation: Optional[str]=None, unif: float=0, initializer: Optional[str]=None):
        """Constructor for "dense" or "linear" layer, with optional activation applied

        :param insz: The number of hidden units in the input
        :param outsz: The number of hidden units in the output
        :param activation: The activation function by name, defaults to `None`, meaning no activation is applied
        :param unif: An optional initialization value which can set the linear weights.  If given, biases will init to 0
        :param initializer: An initialization scheme by string name: `ortho`, `kaiming` or `he`, `xavier` or `glorot`
        """
        super().__init__()
        self.layer = pytorch_linear(insz, outsz, unif, initializer)
        self.activation = get_activation(activation)
        self.output_dim = outsz

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Run a linear projection over the input, followed by an optional activation given by constructor

        :param input: the input tensor
        :return: the transformed output
        """
        return self.activation(self.layer(input))


def TRAIN_FLAG():
    """Create a global training flag on first use"""
    global BASELINE_TF_TRAIN_FLAG
    if BASELINE_TF_TRAIN_FLAG is not None:
        return BASELINE_TF_TRAIN_FLAG
    BASELINE_TF_TRAIN_FLAG = tf.compat.v1.placeholder_with_default(False, shape=(), name='TRAIN_FLAG')
    return BASELINE_TF_TRAIN_FLAG


class SequenceSequenceAttention(tf.keras.layers.Layer):

    def __init__(self, hsz: Optional[int]=None, pdrop: float=0.1, name: str=None):
        super().__init__(name=name)
        self.hsz = hsz
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.attn = None

    def call(self, qkvm):
        query, key, value, mask = qkvm
        a = self._attention(query, key, mask)
        self.attn = a
        a = self.dropout(a, training=TRAIN_FLAG())
        return self._update(a, value)

    def _attention(self, queries, keys, mask=None):
        pass

    def _update(self, a, value):
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param values: The values [B, H, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        return tf.matmul(a, value)


def masked_fill(t, mask, value):
    return t * (1 - tf.cast(mask, t.dtype)) + value * tf.cast(mask, t.dtype)


class SeqDotProductAttention(SequenceSequenceAttention):

    def __init__(self, pdrop: float=0.1, name: str='dot_product_attention', **kwargs):
        super().__init__(pdrop, name=name, **kwargs)

    def _attention(self, query, key, mask=None):
        scores = tf.matmul(query, key, transpose_b=True)
        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1000000000.0)
        return tf.nn.softmax(scores, name='attention_weights')


class SeqScaledDotProductAttention(SequenceSequenceAttention):

    def __init__(self, pdrop: float=0.1, name: str='scaled_dot_product_attention', **kwargs):
        super().__init__(pdrop, name=name, **kwargs)

    def _attention(self, query, key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: A tensor that is (BxHxTxT)
        """
        d_k = tf.shape(query)[-1]
        scores = tf.matmul(query, key, transpose_b=True)
        scores *= tf.math.rsqrt(tf.cast(d_k, tf.float32))
        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1000000000.0)
        return tf.nn.softmax(scores, name='attention_weights')


class SingleHeadReduction(nn.Module):
    """
    Implementation of the "self_attention_head" layer from the conveRT paper (https://arxiv.org/pdf/1911.03688.pdf)
    """

    def __init__(self, d_model: int, dropout: float=0.0, scale: bool=True, d_k: Optional[int]=None):
        """
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        self.d_model = d_model
        if d_k is None:
            self.d_k = d_model
        else:
            self.d_k = d_k
        self.w_Q = Dense(d_model, self.d_k)
        self.w_K = Dense(d_model, self.d_k)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) ->torch.Tensor:
        """According to conveRT model's graph, they project token encodings to lower-dimensional query and key in single
        head, use them to calculate the attention score matrix that has dim [B, T, T], then sum over the query dim to
        get a tensor with [B, 1, T] (meaning the amount of attentions each token gets from all other tokens), scale it
        by sqrt of sequence lengths, then use it as the weight to weighted sum the token encoding to get the sentence
        encoding. we implement it in an equivalent way that can best make use of the eight_mile codes: do the matrix
        multiply with value first, then sum over the query dimension.
        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: sentence-level encoding with dim [B, d_model]
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_mask = mask.squeeze()
        seq_lengths = seq_mask.sum(dim=1)
        query = self.w_Q(query).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        value = value.view(batchsz, -1, 1, self.d_model).transpose(1, 2)
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn
        x = x.squeeze(1)
        x = x * seq_mask.unsqueeze(-1)
        x = x.sum(dim=1)
        x = x * seq_lengths.float().sqrt().unsqueeze(-1)
        return x


class TwoHeadConcat(nn.Module):
    """Use two parallel SingleHeadReduction, and concatenate the outputs. It is used in the conveRT
    paper (https://arxiv.org/pdf/1911.03688.pdf)"""

    def __init__(self, d_model, dropout, scale=False, d_k=None):
        """Two parallel 1-head self-attention, then concatenate the output
        :param d_model: dim of the self-attention
        :param dropout: dropout of the self-attention
        :param scale: scale fo the self-attention
        :param d_k: d_k of the self-attention
        :return: concatenation of the two 1-head attention
        """
        super().__init__()
        self.reduction1 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k)
        self.reduction2 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        encoding1 = self.reduction1(x)
        encoding2 = self.reduction2(x)
        x = torch.cat([encoding1, encoding2], dim=-1)
        return x


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, layer: Optional[tf.keras.layers.Layer]=None, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)
        self.layer = layer

    def call(self, inputs):
        return inputs + self.layer(inputs)

    @property
    def requires_length(self) ->bool:
        return False


class SkipConnection(ResidualBlock):

    def __init__(self, input_size: int, activation: str='relu'):
        super(SkipConnection, self).__init__(tf.keras.layers.Dense(input_size, activation=activation))


__all__ = []


def parameterize(func):
    """Allow as decorator to be called with arguments, returns a new decorator that should be called with the function to be wrapped."""

    @wraps(func)
    def decorator(*args, **kwargs):
        return lambda x: func(x, *args, **kwargs)
    return decorator


@parameterize
def exporter(obj, all_list: List[str]=None):
    """Add a function or class to the __all__.

    When exporting something with out using as a decorator do it like so:
        `func = exporter(func)`
    """
    all_list.append(obj.__name__)
    return obj


export = exporter(__all__)


@export
def is_sequence(x) ->bool:
    if isinstance(x, str):
        return False
    return isinstance(x, (collections.Sequence, collections.MappingView))


@export
def listify(x: Union[List[Any], Any]) ->List[Any]:
    """Take a scalar or list and make it a list iff not already a sequence or numpy array

    :param x: The input to convert
    :return: A list
    """
    if is_sequence(x) or isinstance(x, np.ndarray):
        return x
    return [x] if x is not None else []


class DenseStack(tf.keras.layers.Layer):

    def __init__(self, insz: Optional[int], hsz: Union[int, List[int]], activation: Union[str, List[str]]='relu', pdrop_value: float=0.5, init: Optional[Any]=None, name: Optional[str]=None, skip_connect=False, layer_norm=False, **kwargs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param hsz: The number of hidden units
        :param activation: The name of the activation function to use
        :param pdrop_value: The dropout probability
        :param init: The tensorflow initializer

        """
        super().__init__(name=name)
        hszs = listify(hsz)
        self.output_dim = hszs[-1]
        activations = listify(activation)
        if len(activations) == 1:
            activations = activations * len(hszs)
        if len(activations) != len(hszs):
            raise ValueError('Number of activations must match number of hidden sizes in a stack!')
        if layer_norm:
            layer_norm_eps = kwargs.get('layer_norm_eps', 1e-06)
        if skip_connect:
            if not insz:
                raise ValueError('In order to use skip connection, insz must be provided in DenseStack!')
            current = insz
        layer_stack = []
        for hsz, activation in zip(hszs, activations):
            if skip_connect and current == hsz:
                layer_stack.append(SkipConnection(hsz, activation))
                current = hsz
            else:
                layer_stack.append(tf.keras.layers.Dense(hsz, activation))
            if layer_norm:
                layer_stack.append(tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps))
        self.layer_stack = layer_stack
        self.dropout = tf.keras.layers.Dropout(pdrop_value)

    def call(self, inputs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param inputs: The fixed representation of the model
        :param training: (``bool``) A boolean specifying if we are training or not
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """
        x = inputs
        for layer in self.layer_stack:
            x = layer(x)
            x = self.dropout(x, TRAIN_FLAG())
        return x

    @property
    def requires_length(self) ->bool:
        return False


class ConveRTFFN(nn.Module):
    """Implementation of the FFN layer from the convert paper (https://arxiv.org/pdf/1911.03688.pdf)"""

    def __init__(self, insz, hszs, outsz, pdrop):
        """
        :param insz: input dim
        :param hszs: list of hidden sizes
        :param outsz: output dim
        :param pdrop: dropout of each hidden layer
        """
        super().__init__()
        self.dense_stack = DenseStack(insz, hszs, activation='gelu', pdrop_value=pdrop, skip_connect=True, layer_norm=True)
        self.final = Dense(hszs[-1], outsz)
        self.proj = Dense(insz, outsz) if insz != outsz else nn.Identity()
        self.ln1 = nn.LayerNorm(insz, eps=1e-06)
        self.ln2 = nn.LayerNorm(outsz, eps=1e-06)

    def forward(self, inputs):
        x = self.ln1(inputs)
        x = self.dense_stack(x)
        x = self.final(x)
        x = x + self.proj(inputs)
        return self.ln2(x)


@export
class Offsets:
    """Support pre 3.4"""
    PAD, GO, EOS, UNK, OFFSET = range(0, 5)
    VALUES = ['<PAD>', '<GO>', '<EOS>', '<UNK>']


class FFN(tf.keras.layers.Layer):
    """
    FFN from https://arxiv.org/abs/1706.03762

    The paper does not specify any dropout in this layer, but subsequent implementations (like XLM) do use dropout.
    """

    def __init__(self, d_model: int, activation: str='relu', d_ff: Optional[int]=None, pdrop: float=0.0, name: Optional[int]=None):
        """Constructor, takes in model size (which is the external currency of each block) and the feed-forward size

        :param d_model: The model size.  This is the size passed through each block
        :param d_ff: The feed-forward internal size, which is typical 4x larger, used internally
        :param pdrop: The probability of dropping output
        """
        super().__init__(name=name)
        if d_ff is None:
            d_ff = 4 * d_model
        self.expansion = tf.keras.layers.Dense(d_ff)
        self.squeeze = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.act = get_activation(activation)

    def call(self, inputs):
        return self.squeeze(self.dropout(self.act(self.expansion(inputs))))


def get_shape_as_list(x):
    """
    This function makes sure we get a number whenever possible, and otherwise, gives us back
    a graph operation, but in both cases, presents as a list.  This makes it suitable for a
    bunch of different operations within TF, and hides away some details that we really dont care about, but are
    a PITA to get right...

    Borrowed from Alec Radford:
    https://github.com/openai/finetune-transformer-lm/blob/master/utils.py#L38
    """
    try:
        ps = x.get_shape().as_list()
    except:
        ps = x.shape
    ts = tf.shape(x)
    return [(ts[i] if ps[i] is None else ps[i]) for i in range(len(ps))]


class MultiHeadedAttention(tf.keras.layers.Layer):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """

    def __init__(self, num_heads: int, d_model: int, dropout: float=0.1, scale: bool=False, d_k: Optional[int]=None, name: str=None):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super().__init__(name=name)
        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f'd_model ({d_model}) must be evenly divisible by num_heads ({num_heads})')
        else:
            self.d_k = d_k
        self.h = num_heads
        self.w_Q = tf.keras.layers.Dense(units=self.d_k * self.h, name='query_projection')
        self.w_K = tf.keras.layers.Dense(units=self.d_k * self.h, name='key_projection')
        self.w_V = tf.keras.layers.Dense(units=self.d_k * self.h, name='value_projection')
        self.w_O = tf.keras.layers.Dense(units=self.d_k * self.h, name='output_projection')
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def call(self, qkvm):
        query, key, value, mask = qkvm
        batchsz = get_shape_as_list(query)[0]
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.w_V(value), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batchsz, -1, self.h * self.d_k])
        return self.w_O(x)


class SequenceSequenceRelativeAttention(tf.keras.layers.Layer):
    """This form of attention is specified in Shaw et al 2018: https://www.aclweb.org/anthology/N18-2074.pdf
    """

    def __init__(self, hsz: int=None, pdrop: float=0.1, name=None, **kwargs):
        super().__init__(name=name)
        self.hsz = hsz
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.attn = None

    def call(self, q_k_v_ek_ev_m):
        query, key, value, edges_key, edges_value, mask = q_k_v_ek_ev_m
        a = self._attention(query, key, edges_key, mask)
        self.attn = a
        a = self.dropout(a, training=TRAIN_FLAG())
        return self._update(a, value, edges_value)

    def _attention(self, query, key, edges_key, mask=None):
        pass

    def _update(self, a, value, edges_value):
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param value: The values [B, H, T, D]
        :param edge_value: The edge values [T, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        B, H, T, D = get_shape_as_list(value)
        updated_values = tf.matmul(a, value)
        a = tf.transpose(tf.reshape(a, [B * H, T, T]), [1, 0, 2])
        t = tf.matmul(a, edges_value)
        t = tf.transpose(t, [1, 0, 2])
        update_edge_values = tf.reshape(t, [B, H, T, D])
        return updated_values + update_edge_values


class SeqDotProductRelativeAttention(SequenceSequenceRelativeAttention):

    def __init__(self, pdrop: float=0.1, name: str='dot_product_rel_attention', **kwargs):
        super().__init__(pdrop=pdrop, name=name, **kwargs)

    def _attention(self, query, key, edges_key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [TxTxD]
        :return: A tensor that is (BxHxTxT)
        """
        B, H, T, d_k = get_shape_as_list(query)
        scores_qk = tf.matmul(query, key, transpose_b=True)
        tbhd = tf.transpose(tf.reshape(query, [B * H, T, d_k]), [1, 0, 2])
        scores_qek = tf.matmul(tbhd, edges_key, transpose_b=True)
        scores_qek = tf.transpose(scores_qek, [1, 0, 2])
        scores_qek = tf.reshape(scores_qek, [B, H, T, T])
        scores = scores_qk + scores_qek
        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1000000000.0)
        return tf.nn.softmax(scores, name='rel_attention_weights')


class SeqScaledDotProductRelativeAttention(SequenceSequenceRelativeAttention):

    def __init__(self, pdrop: float=0.1, name: str='scaled_dot_product_rel_attention', **kwargs):
        super().__init__(pdrop=pdrop, name=name, **kwargs)

    def _attention(self, query, key, edges_key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [TxTxD]
        :return: A tensor that is (BxHxTxT)
        """
        B, H, T, d_k = get_shape_as_list(query)
        scores_qk = tf.matmul(query, key, transpose_b=True)
        tbhd = tf.transpose(tf.reshape(query, [B * H, T, d_k]), [1, 0, 2])
        scores_qek = tf.matmul(tbhd, edges_key, transpose_b=True)
        scores_qek = tf.transpose(scores_qek, [1, 0, 2])
        scores_qek = tf.reshape(scores_qek, [B, H, T, T])
        scores = (scores_qk + scores_qek) / math.sqrt(d_k)
        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1000000000.0)
        return tf.nn.softmax(scores, name='rel_attention_weights')


class MultiHeadedRelativeAttention(tf.keras.layers.Layer):
    """
    Multi-headed relative attention from Shaw et al 2018 (https://www.aclweb.org/anthology/N18-2074.pdf)

    This method follows the same approach of MultiHeadedAttention, but it computes Relative Position Representations (RPR)
    which are used as part of the attention computations.  To facilitate this, the model has its own internal
    embeddings lookup table, and it has an updated computation for both the attention weights and the application
    of those weights to follow them.

    """

    def __init__(self, num_heads: int, d_model: int, rpr_k: int, dropout: float=0.1, scale: bool=False, d_k: Optional[int]=None, name=None):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f'd_model ({d_model}) must be evenly divisible by num_heads ({num_heads})')
        else:
            self.d_k = d_k
        self.rpr_k = rpr_k
        self.rpr_key = tf.keras.layers.Embedding(2 * rpr_k + 1, self.d_k)
        self.rpr_value = tf.keras.layers.Embedding(2 * rpr_k + 1, self.d_k)
        self.h = num_heads
        self.w_Q = tf.keras.layers.Dense(units=self.d_k * self.h, name='query_projection')
        self.w_K = tf.keras.layers.Dense(units=self.d_k * self.h, name='key_projection')
        self.w_V = tf.keras.layers.Dense(units=self.d_k * self.h, name='value_projection')
        self.w_O = tf.keras.layers.Dense(units=self.d_k * self.h, name='output_projection')
        if scale:
            self.attn_fn = SeqScaledDotProductRelativeAttention(dropout)
        else:
            self.attn_fn = SeqDotProductRelativeAttention(dropout)
        self.attn = None

    def make_rpr(self, seq_len: int):
        """Create a matrix shifted by self.rpr_k and bounded between 0 and 2*self.rpr_k to provide 0-based indexing for embedding
        """
        seq = tf.range(seq_len)
        window_len = 2 * self.rpr_k
        edges = tf.reshape(seq, [1, -1]) - tf.reshape(seq, [-1, 1]) + self.rpr_k
        edges = tf.clip_by_value(edges, 0, window_len)
        return self.rpr_key(edges), self.rpr_value(edges)

    def call(self, qkvm):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        shp = get_shape_as_list(query)
        batchsz = shp[0]
        seq_len = shp[1]
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.w_V(value), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        rpr_key, rpr_value = self.make_rpr(seq_len)
        x = self.attn_fn((query, key, value, rpr_key, rpr_value, mask))
        self.attn = self.attn_fn.attn
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batchsz, -1, self.h * self.d_k])
        return self.w_O(x)


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_heads: int, d_model: int, pdrop: float, scale: bool=True, activation_type: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[int]=None, ffn_pdrop: Optional[float]=0.0, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name: Optional[str]=None):
        super().__init__(name=name)
        self.layer_norms_after = layer_norms_after
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k)
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale, d_k=d_k)
        self.ffn = FFN(d_model, activation_type, d_ff, ffn_pdrop, name='ffn')
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        """
        :param inputs: `(x, mask)`
        :return: The output tensor
        """
        x, mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        h = self.self_attn((x, x, x, mask))
        x = x + self.dropout(h, TRAIN_FLAG())
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x), TRAIN_FLAG())
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerEncoderStack(tf.keras.layers.Layer):

    def __init__(self, num_heads: int, d_model: int, pdrop: float, scale: bool=True, layers: int=1, activation: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[Union[int, List[int]]]=None, ffn_pdrop: Optional[float]=0.0, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name=None, **kwargs):
        super().__init__(name=name)
        self.encoders = []
        self.ln = tf.identity if layer_norms_after else tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers
        for i in range(layers):
            self.encoders.append(TransformerEncoder(num_heads, d_model, pdrop, scale, activation, d_ff, d_k, rpr_k=rpr_k[i], ffn_pdrop=ffn_pdrop, layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps, name=name))

    def call(self, inputs):
        x, mask = inputs
        for layer in self.encoders:
            x = layer((x, mask))
        return self.ln(x)


class PairedModel(nn.Module):

    def __init__(self, embeddings, d_model, d_ff, dropout, num_heads, num_layers, stacking_layers=None, d_out=512, d_k=None, weight_std=0.02, rpr_k=None, reduction_d_k=64, ff_pdrop=0.1):
        super().__init__()
        if stacking_layers is None:
            stacking_layers = [d_model] * 3
        self.weight_std = weight_std
        stacking_layers = listify(stacking_layers)
        transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model, pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff, d_k=d_k, rpr_k=rpr_k)
        self.attention_layer = TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k)
        self.transformer_layers = transformer
        self.embedding_layers = embeddings
        self.ff1 = ConveRTFFN(2 * d_model, stacking_layers, d_out, ff_pdrop)
        self.ff2 = ConveRTFFN(2 * d_model, stacking_layers, d_out, ff_pdrop)
        self.apply(self.init_layer_weights)

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def encode_query(self, query):
        query_mask = query != Offsets.PAD
        att_mask = query_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(query)
        encoded_query = self.transformer_layers((embedded, att_mask))
        encoded_query = self.attention_layer((encoded_query, encoded_query, encoded_query, att_mask))
        encoded_query = self.ff1(encoded_query)
        return encoded_query

    def encode_response(self, response):
        response_mask = response != Offsets.PAD
        att_mask = response_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(response)
        encoded_response = self.transformer_layers((embedded, att_mask))
        encoded_response = self.attention_layer((encoded_response, encoded_response, encoded_response, att_mask))
        encoded_response = self.ff2(encoded_response)
        return encoded_response

    def forward(self, query, response):
        encoded_query = self.encode_query(query)
        encoded_response = self.encode_response(response)
        return encoded_query, encoded_response

    def create_loss(self, loss_type='all'):
        if loss_type == 'all':
            return AllLoss(self)
        return TripletLoss(self)


class Reduction(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs: List[tf.Tensor]) ->tf.Tensor:
        pass


class ConcatReduction(Reduction):

    def __init__(self, output_dims: List[int], axis=-1):
        super().__init__()
        self.axis = axis
        self.output_dim = sum(output_dims)

    def call(self, inputs: List[tf.Tensor]) ->tf.Tensor:
        return tf.concat(values=inputs, axis=-1)


class SumLayerNormReduction(Reduction):

    def __init__(self, output_dims: List[int], layer_norm_eps: float=1e-12):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.output_dim = output_dims[0]

    def call(self, inputs: List[tf.Tensor]) ->tf.Tensor:
        outputs = tf.add_n(inputs)
        return self.ln(outputs)


class SumReduction(Reduction):

    def __init__(self, output_dims: List[int]):
        super().__init__()
        self.output_dim = output_dims[0]

    def call(self, inputs: List[tf.Tensor]) ->tf.Tensor:
        return tf.add_n(inputs)


class EmbeddingsStack(tf.keras.layers.Layer):

    def __init__(self, embeddings_dict: Dict[str, tf.keras.layers.Layer], dropout_rate: float=0.0, requires_length: bool=False, reduction: Optional[Union[str, tf.keras.layers.Layer]]='concat', name: Optional[str]=None, **kwargs):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """
        super().__init__(name=name)
        self.embeddings = embeddings_dict
        output_dims = []
        for embedding in embeddings_dict.values():
            output_dims += [embedding.get_dsz()]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self._requires_length = requires_length
        if isinstance(reduction, str):
            if reduction == 'sum':
                self.reduction = SumReduction(output_dims)
            elif reduction == 'sum-layer-norm':
                self.reduction = SumLayerNormReduction(output_dims, layer_norm_eps=kwargs.get('layer_norm_eps', 1e-12))
            else:
                self.reduction = ConcatReduction(output_dims)
        else:
            self.reduction = reduction
        self.dsz = self.reduction.output_dim

    def keys(self):
        return self.embeddings.keys()

    def items(self):
        return self.embeddings.items()

    def __getitem__(self, item: str):
        return self.embeddings[item]

    def call(self, inputs: Dict[str, tf.Tensor]) ->tf.Tensor:
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        for k, embedding in self.embeddings.items():
            x = inputs[k]
            embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
        word_embeddings = self.reduction(all_embeddings_out)
        return self.dropout(word_embeddings, TRAIN_FLAG())

    def keys(self):
        return self.embeddings.keys()

    @property
    def requires_length(self) ->bool:
        return self._requires_length

    @property
    def output_dim(self) ->bool:
        return self.dsz


class TransformerDiscriminator(nn.Module):

    def __init__(self, embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k, d_k, **kwargs):
        super().__init__()
        self.embeddings = EmbeddingsStack(embeddings, dropout)
        self.weight_std = kwargs.get('weight_std', 0.02)
        assert self.embeddings.dsz == d_model
        self.transformer = TransformerEncoderStack(num_heads, d_model=d_model, pdrop=dropout, scale=True, layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k)
        self.proj_to_output = pytorch_linear(d_model, 1)
        self.apply(self.init_layer_weights)
        self.lengths_feature = kwargs.get('lengths_feature', self.embeddings.keys()[0])

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, features):
        embedded = self.embeddings(features)
        x = features[self.lengths_feature]
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        transformer_out = self.transformer((embedded, input_mask))
        binary = self.proj_to_output(transformer_out)
        return torch.sigmoid(binary)

    def create_loss(self):


        class Loss(nn.Module):

            def __init__(self):
                super().__init__()
                self.loss = nn.BCELoss()

            def forward(self, input, target):
                fake_loss = self.loss(input[target == 0], target[target == 0])
                real_loss = self.loss(input[target != 0], target[target != 0])
                return real_loss + fake_loss
        return Loss()


logger = logging.getLogger('mead')


@export
class ClassifierModel:
    """Text classifier

    Provide an interface to DNN classifiers that use word lookup tables.
    """
    task_name = 'classify'

    def __init__(self, *args, **kwargs):
        super().__init__()

    def save(self, basename):
        """Save this model out

        :param basename: Name of the model, not including suffixes
        :return: None
        """
        pass

    @classmethod
    def load(cls, basename, **kwargs):
        """Load the model from a basename, including directory

        :param basename: Name of the model, not including suffixes
        :param kwargs: Anything that is useful to optimize experience for a specific framework
        :return: A newly created model
        """
        pass

    def predict(self, batch_dict):
        """Classify a batch of text with whatever features the model can use from the batch_dict.
        The indices correspond to get_vocab().get('word', 0)

        :param batch_dict: This normally contains `x`, a `BxT` tensor of indices.  Some classifiers and readers may
        provide other features

        :return: A list of lists of tuples (label, value)
        """
        pass

    def classify(self, batch_dict):
        logger.warning('`classify` is deprecated, use `predict` instead.')
        return self.predict(batch_dict)

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.

        :return: A list of the labels for the decision
        """
        pass


TensorDef = torch.Tensor


def unsort_batch(batch: torch.Tensor, perm_idx: torch.Tensor) ->torch.Tensor:
    """Undo the sort on a batch of tensors done for packing the data in the RNN.

    :param batch: The batch of data batch first `[B, ...]`
    :param perm_idx: The permutation index returned from the torch.sort.

    :returns: The batch in the original order.
    """
    perm_idx = perm_idx
    diff = len(batch.shape) - len(perm_idx.shape)
    extra_dims = [1] * diff
    perm_idx = perm_idx.view([-1] + extra_dims)
    return batch.scatter_(0, perm_idx.expand_as(batch), batch)


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)
    return wrapped


@export
@optional_params
def str_file(func, **kwargs):
    """A decorator to automatically open arguments that are files.

    If there are kwargs then they are name=mode. When the function is
    called if the argument name is a string then the file is opened with
    mode.

    If there are no kwargs then it is assumed the first argument is a
    file that should be opened as 'r'
    """
    possible_files = kwargs
    if inspect.isgeneratorfunction(func):

        @wraps(func)
        def open_files(*args, **kwargs):
            if not possible_files:
                if isinstance(args[0], str):
                    with io.open(args[0], mode='r', encoding='utf-8') as f:
                        for x in func(f, *args[1:], **kwargs):
                            yield x
                else:
                    for x in func(*args, **kwargs):
                        yield x
            else:
                to_close = []
                arg = inspect.getcallargs(func, *args, **kwargs)
                try:
                    for f, mode in possible_files.items():
                        if isinstance(arg[f], str):
                            arg[f] = io.open(arg[f], mode=mode, encoding=None if 'b' in mode else 'utf-8')
                            to_close.append(f)
                    for x in func(**arg):
                        yield x
                finally:
                    for f in to_close:
                        arg[f].close()
    else:

        @wraps(func)
        def open_files(*args, **kwargs):
            if not possible_files:
                if isinstance(args[0], str):
                    with io.open(args[0], mode='r', encoding='utf-8') as f:
                        return func(f, *args[1:], **kwargs)
                else:
                    return func(*args, **kwargs)
            else:
                to_close = []
                arg = inspect.getcallargs(func, *args, **kwargs)
                try:
                    for f, mode in possible_files.items():
                        if isinstance(arg[f], str):
                            arg[f] = io.open(arg[f], mode=mode, encoding=None if 'b' in mode else 'utf-8')
                            to_close.append(f)
                    return func(**arg)
                finally:
                    for f in to_close:
                        arg[f].close()
    return open_files


@export
@str_file(filepath='w')
def write_json(content, filepath):
    json.dump(content, filepath, indent=True)


class ClassifierModelBase(nn.Module, ClassifierModel):
    """Base for all baseline implementations of token-based classifiers

    This class provides a loose skeleton around which the baseline models
    are built.  It is built on the PyTorch `nn.Module` base, and fulfills the `ClassifierModel` interface.
    To override this class, the use would typically override the `create_layers` function which will
    create and attach all sub-layers of this model into the class, and the `forward` function which will
    give the model some implementation to call on forward.
    """

    def __init__(self):
        super().__init__()
        self.gpu = False

    @classmethod
    def load(cls, filename: str, **kwargs) ->'ClassifierModelBase':
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def save(self, outname: str):
        logger.info('saving %s' % outname)
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)
        write_json(self.labels, basename + '.labels')

    @classmethod
    def create(cls, embeddings, labels, **kwargs) ->'ClassifierModelBase':
        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.5)
        model.lengths_key = kwargs.get('lengths_key')
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels
        model.create_layers(embeddings, **kwargs)
        logger.info(model)
        return model

    def cuda(self, device=None):
        self.gpu = True
        return super()

    def create_loss(self):
        return nn.NLLLoss()

    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):
        """Transform a `batch_dict` into something usable in this model

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :return:
        """
        example_dict = dict({})
        perm_idx = None
        if self.lengths_key is not None:
            lengths = batch_dict[self.lengths_key]
            if numpy_to_tensor:
                lengths = torch.from_numpy(lengths)
            lengths, perm_idx = lengths.sort(0, descending=True)
            if self.gpu:
                lengths = lengths
            example_dict['lengths'] = lengths
        for key in self.embeddings.keys():
            tensor = batch_dict[key]
            if numpy_to_tensor:
                tensor = torch.from_numpy(tensor)
            if perm_idx is not None:
                tensor = tensor[perm_idx]
            if self.gpu:
                tensor = tensor
            example_dict[key] = tensor
        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if perm_idx is not None:
                y = y[perm_idx]
            if self.gpu:
                y = y
            example_dict['y'] = y
        if perm:
            return example_dict, perm_idx
        return example_dict

    def predict_batch(self, batch_dict: Dict[str, TensorDef], **kwargs) ->TensorDef:
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        examples, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        with torch.no_grad():
            probs = self(examples).exp()
            probs = unsort_batch(probs, perm_idx)
        return probs

    def predict(self, batch_dict: Dict[str, TensorDef], raw: bool=False, dense: bool=False, **kwargs):
        probs = self.predict_batch(batch_dict, **kwargs)
        if raw and not dense:
            logger.warning('Warning: `raw` parameter is deprecated pass `dense=True` to get back values as a single tensor')
            dense = True
        if dense:
            return probs
        results = []
        batchsz = probs.size(0)
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def get_labels(self) ->List[str]:
        return self.labels

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """


BaseLayer = nn.Module


class PassThru(tf.keras.layers.Layer):

    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def call(self, inputs: tf.Tensor) ->tf.Tensor:
        return inputs


class EmbedPoolStackClassifier(ClassifierModelBase):
    """Provides a simple but effective base for most `ClassifierModel`s

   This class provides a common base for classifiers by identifying and codifying
   and idiomatic pattern where a typical classifier may be though of as a composition
   between a stack of embeddings, followed by a pooling operation yielding a fixed length
   tensor, followed by one or more dense layers, and ultimately, a projection to the output space.

   To provide an useful interface to sub-classes, we override the `create_layers` to provide a hook
   for each layer identified in this idiom, and leave the implementations up to the sub-class.

   We also fully implement the `forward` method.

   """

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        self.stack_model = self.init_stacked(self.pool_model.output_dim, **kwargs)
        self.output_layer = self.init_output(self.stack_model.output_dim, **kwargs)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) ->BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def init_pool(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce a pooling operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A pooling operation
        """

    def init_stacked(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop)

    def init_output(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce the final output layer in the model

        :param input_dim: The input hidden size
        :param kwargs:
        :return:
        """
        return Dense(input_dim, len(self.labels), activation=kwargs.get('output_activation', 'log_softmax'))

    def forward(self, inputs: Dict[str, TensorDef]) ->TensorDef:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """
        lengths = inputs.get('lengths')
        embedded = self.embeddings(inputs)
        embedded = embedded, lengths
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


class ParallelConv(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(self, insz: Optional[int], outsz: Union[int, List[int]], filtsz: List[int], activation: str='relu', name: Optional[str]=None, **kwargs):
        """Do parallel convolutions with multiple filter widths and max-over-time pooling.

        :param insz: The input size (not required, can pass `None`)
        :param outsz: The output size(s).  Normally this is an int, but it can be a stack of them
        :param filtsz: The list of filter widths to use.
        :param activation: (``str``) The name of the activation function to use (`default='relu`)
        :param name: An optional name
        """
        super().__init__(name=name)
        self.Ws = []
        self.bs = []
        self.activation = get_activation(activation)
        self.insz = insz
        self.filtsz = filtsz
        motsz = outsz
        if not is_sequence(outsz):
            motsz = [outsz] * len(filtsz)
        self.motsz = motsz
        self.output_dim = sum(motsz)

    def build(self, input_shape):
        insz = self.insz if self.insz is not None else tf.shape(input_shape)[-1]
        for fsz, cmotsz in zip(self.filtsz, self.motsz):
            kernel_shape = [1, int(fsz), int(insz), int(cmotsz)]
            self.Ws.append(self.add_weight(f'cmot-{fsz}/W', shape=kernel_shape))
            self.bs.append(self.add_weight(f'cmot-{fsz}/b', shape=[cmotsz], initializer=tf.constant_initializer(0.0)))

    def call(self, inputs):
        """
        :param inputs: The inputs in the shape [B, T, H].
        :return: Combined result
        """
        mots = []
        expanded = tf.expand_dims(inputs, ParallelConv.DUMMY_AXIS)
        for W, b in zip(self.Ws, self.bs):
            conv = tf.nn.conv2d(expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='CONV')
            activation = self.activation(tf.nn.bias_add(conv, b), 'activation')
            mot = tf.reduce_max(activation, [ParallelConv.TIME_AXIS], keepdims=True)
            mots.append(mot)
        combine = tf.reshape(tf.concat(values=mots, axis=ParallelConv.FEATURE_AXIS), [-1, self.output_dim])
        return combine

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    @property
    def requires_length(self):
        return False


class WithDropout(tf.keras.layers.Layer):
    """This is a utility wrapper that applies dropout after executing the layer it wraps

    For variational dropout, we use `SpatialDropout1D` as described in:
    https://github.com/keras-team/keras/issues/7290
    """

    def __init__(self, layer: tf.keras.layers.Layer, pdrop: float=0.5, variational: bool=False):
        super().__init__()
        self.layer = layer
        self.dropout = tf.keras.layers.SpatialDropout1D(pdrop) if variational else tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        return self.dropout(self.layer(inputs), TRAIN_FLAG())

    @property
    def output_dim(self) ->int:
        return self.layer.output_dim


class WithoutLength(tf.keras.layers.Layer):
    """Wrapper layer to remove lengths from the input
    """

    def __init__(self, layer: tf.keras.layers.Layer, name=None):
        super().__init__(name=name)
        self.layer = layer
        self.output_dim = self.layer.output_dim if hasattr(self.layer, 'output_dim') else 0

    def call(self, inputs):
        output = self.layer(inputs[0])
        return output


BASELINE_LOADERS = {}


BASELINE_MODELS = {}


@export
@optional_params
def register_model(cls, task, name=None):
    """Register a function as a plug-in"""
    if name is None:
        name = cls.__name__
    names = listify(name)
    if task not in BASELINE_MODELS:
        BASELINE_MODELS[task] = {}
    if task not in BASELINE_LOADERS:
        BASELINE_LOADERS[task] = {}
    if hasattr(cls, 'create'):

        def create(*args, **kwargs):
            return cls.create(*args, **kwargs)
    else:

        def create(*args, **kwargs):
            return cls(*args, **kwargs)
    for alias in names:
        if alias in BASELINE_MODELS[task]:
            raise Exception('Error: attempt to re-define previously registered handler {} (old: {}, new: {}) for task {} in registry'.format(alias, BASELINE_MODELS[task], cls, task))
        BASELINE_MODELS[task][alias] = create
        if hasattr(cls, 'load'):
            BASELINE_LOADERS[task][alias] = cls.load
    return cls


class ConvModel(EmbedPoolStackClassifier):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width
    """

    def init_pool(self, input_dim: int, **kwargs) ->BaseLayer:
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling

        :param input_dim: Embedding output size
        :param kwargs: See below

        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use

        :return: A pooling layer
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        return WithoutLength(WithDropout(ParallelConv(input_dim, cmotsz, filtsz, 'relu', input_fmt='bth'), self.pdrop))


class BiLSTMEncoderBase(nn.Module):
    """BiLSTM encoder base for a set of encoders producing various outputs.

    All BiLSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiLSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiLSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0, requires_length: bool=True, batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz // 2, nlayers, dropout=pdrop, bidirectional=True, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state):
        return tuple(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0] for s in state)


def _cat_dir(h: torch.Tensor) ->torch.Tensor:
    """Concat forward and backword state vectors.

    The shape of the hidden is `[#layers * #dirs, B, H]`. The docs say you can
    separate directions with `h.view(#l, #dirs, B, H)` with the forward dir being
    index 0 and backwards dir being 1.

    This means that before separating with the view the forward dir are the even
    indices in the first dim while the backwards dirs are the odd ones. Here we select
    the even and odd values and concatenate them

    :param h: The hidden shape as it comes back from PyTorch modules
    """
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=-1)


def concat_state_dirs(state):
    """Convert the bidirectional out of an RNN so the forward and backward values are a single vector."""
    if isinstance(state, tuple):
        return tuple(_cat_dir(h) for h in state)
    return _cat_dir(state)


class BiLSTMEncoderHidden(BiLSTMEncoderBase):
    """BiLSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(concat_state_dirs(hidden))[0]


class LSTMEncoderBase(nn.Module):
    """The LSTM encoder is a base for a set of encoders producing various outputs.

    All LSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `LSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `LSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0, requires_length: bool=True, batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: Tuple[torch.Tensor, torch.Tensor]) ->List[torch.Tensor]:
        """Get a view of the top state of shape [B, H]`

        :param state:
        :return:
        """
        top = []
        for s in state:
            top.append(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0])
        return top


class LSTMEncoderHidden(LSTMEncoderBase):
    """LSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->torch.Tensor:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor of shape `[B, H]` representing the last RNNs hidden state
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(hidden)[0]


class LSTMModel(EmbedPoolStackClassifier):
    """A simple single-directional single-layer LSTM. No layer-stacking.
    """

    def init_pool(self, input_dim: int, **kwargs) ->BaseLayer:
        """LSTM with dropout yielding a final-state as output

        :param input_dim: The input word embedding depth
        :param kwargs: See below

        :Keyword Arguments:
        * *rnnsz* -- (``int``) The number of hidden units (defaults to `hsz`)
        * *rnntype/rnn_type* -- (``str``) The RNN type, defaults to `lstm`, other valid values: `blstm`
        * *hsz* -- (``int``) backoff for `rnnsz`, typically a result of stacking params.  This keeps things simple so
          its easy to do things like residual connections between LSTM and post-LSTM stacking layers

        :return: A pooling layer
        """
        unif = kwargs.get('unif')
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        weight_init = kwargs.get('weight_init', 'uniform')
        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        if rnntype == 'blstm':
            return BiLSTMEncoderHidden(input_dim, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)
        return LSTMEncoderHidden(input_dim, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)


class NBowModelBase(EmbedPoolStackClassifier):
    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """

    def init_stacked(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce a stacking operation that will be used in the model, defaulting to a single layer

        :param input_dim: The input dimension size
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``List[int]``) The number of hidden units (defaults to 100)
        """
        kwargs.setdefault('hsz', [100])
        return super().init_stacked(input_dim, **kwargs)


def tensor_and_lengths(inputs):
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None
    return in_tensor, lengths


class MeanPool1D(tf.keras.layers.Layer):

    def __init__(self, dsz: int, trainable: bool=False, name: Optional[str]=None, dtype: int=tf.float32, batch_first: bool=True, *args, **kwargs):
        """This is a layers the calculates the mean pooling in a length awareway.

           This was originally a wrapper around tf.keras.layers.GlobalAveragePooling1D()
           but that had problems because the mask didn't work then the dimension we
           are pooling over was variable length.

           looking here https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/pooling.py#L639

           We can see that the input shape is being gotten as a list where for the
           value of `input_shape[step_axis]` is `None` instead of getting the shape
           via `tf.shape`. This means that when they do the reshape the
           broadcast_shape is `[-1, None, 1]` which causes an error.
        """
        super().__init__(trainable, name, dtype)
        self.output_dim = dsz
        self.reduction_dim = 1 if batch_first else 0

    def call(self, inputs):
        tensor, lengths = tensor_and_lengths(inputs)
        return tf.reduce_sum(tensor, self.reduction_dim) / tf.cast(tf.expand_dims(lengths, -1), tf.float32)

    @property
    def requires_length(self):
        return True


class NBowModel(NBowModelBase):
    """Neural Bag-of-Words average pooling (standard) model"""

    def init_pool(self, input_dim: int, **kwargs) ->BaseLayer:
        """Do average pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: None
        :return: The average pooling representation
        """
        return MeanPool1D(input_dim)


MASK_FALSE = False


def bth2tbh(t):
    return t.transpose(t, [1, 0, 2])


@export
def sequence_mask(lengths, max_len: int=-1):
    if max_len < 0:
        max_len = np.max(lengths)
    row = np.arange(0, max_len).reshape(1, -1)
    col = np.reshape(lengths, (-1, 1))
    return (row < col).astype(np.uint8)


class MaxPool1D(nn.Module):
    """Do a max-pooling operation with or without a length given
    """

    def __init__(self, outsz, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) ->torch.Tensor:
        """If we are given a tuple as input, we will use the length, otherwise we will do an operation without masking

        :param inputs: either a tuple of `(input, lengths)` or a tensor `input`
        :return: A pooled tensor
        """
        tensor, lengths = tensor_and_lengths(inputs)
        if lengths is not None:
            mask = sequence_mask(lengths)
            mask = mask if self.batch_first else bth2tbh(mask)
            tensor = tensor.masked_fill(mask.unsqueeze(-1) == MASK_FALSE, -10000.0)
        dmax, _ = torch.max(tensor, self.reduction_dim, keepdim=False)
        return dmax

    def extra_repr(self) ->str:
        return f'batch_first={self.batch_first}'


class NBowMaxModel(NBowModelBase):
    """Max-pooling model for Neural Bag-of-Words.  Sometimes does better than avg pooling
    """

    def init_pool(self, input_dim: int, **kwargs) ->BaseLayer:
        """Do max pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: None
        :return: The max pooling representation
        """
        return MaxPool1D(input_dim)


class FineTuneModelClassifier(ClassifierModelBase):
    """Fine-tune based on pre-pooled representations"""

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) ->BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def init_stacked(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop)

    def init_output(self, input_dim: int, **kwargs) ->BaseLayer:
        """Produce the final output layer in the model

        :param input_dim: The input hidden size
        :param kwargs:
        :return:
        """
        return WithDropout(Dense(input_dim, len(self.labels), activation=kwargs.get('output_activation', 'log_softmax'), unif=kwargs.get('output_unif', 0.0)), pdrop=kwargs.get('output_dropout', 0.0))

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.stack_model = self.init_stacked(self.embeddings.output_dim, **kwargs)
        self.output_layer = self.init_output(self.stack_model.output_dim, **kwargs)

    def forward(self, inputs):
        base_layers = self.embeddings(inputs)
        stacked = self.stack_model(base_layers)
        return self.output_layer(stacked)


class CompositePooling(nn.Module):
    """Composite pooling allows for multiple sub-modules during pooling to be used in parallel
    """

    def __init__(self, models):
        """
        Note, this currently requires that each submodel is an eight_mile model with an `output_dim` attr
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.output_dim = sum(m.output_dim for m in self.models)
        self.requires_length = any(getattr(m, 'requires_length', False) for m in self.models)

    def forward(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        pooled = []
        for sub_model in self.models:
            if getattr(sub_model, 'requires_length', False):
                pooled.append(sub_model((inputs, lengths)))
            else:
                pooled.append(sub_model(inputs))
        return torch.cat(pooled, -1)


class CompositePoolingModel(ClassifierModelBase):
    """Fulfills pooling contract by aggregating pooling from a set of sub-models and concatenates each"""

    def init_pool(self, dsz, **kwargs):
        SubModels = [eval(model) for model in kwargs.get('sub')]
        sub_models = [SM.init_pool(self, dsz, **kwargs) for SM in SubModels]
        return CompositePooling(sub_models)

    def make_input(self, batch_dict):
        """Because the sub-model could contain an LSTM, make sure to sort lengths descending

        :param batch_dict:
        :return:
        """
        inputs = super().make_input(batch_dict)
        lengths = inputs['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        for k, value in inputs.items():
            inputs[k] = value[perm_idx]
        return inputs


@export
class LanguageModel(object):
    task_name = 'lm'

    def __init__(self):
        super().__init__()

    @staticmethod
    def load(basename, **kwargs):
        pass

    @classmethod
    def create(cls, embeddings, **kwargs):
        pass

    def predict(self, batch_dict, **kwargs):
        pass


class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        super(SequenceCriterion, self).__init__()
        if avg == 'token':
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=True)
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=False)
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)


class LanguageModelBase(nn.Module, LanguageModel):

    def __init__(self):
        super().__init__()

    def save(self, outname):
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)

    def create_loss(self):
        return SequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @classmethod
    def load(cls, filename, **kwargs):
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def zero_state(self, batchsz):
        return None

    @property
    def requires_state(self):
        pass

    def make_input(self, batch_dict, numpy_to_tensor=False):
        example_dict = dict({})
        for key in self.src_keys:
            tensor = batch_dict[key]
            if numpy_to_tensor:
                tensor = torch.from_numpy(tensor)
            if self.gpu:
                tensor = tensor
            example_dict[key] = tensor
        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if self.gpu:
                y = y
            example_dict['y'] = y
        return example_dict

    @classmethod
    def create(cls, embeddings, **kwargs):
        lm = cls()
        lm.gpu = kwargs.get('gpu', True)
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')
        lm.src_keys = kwargs.get('src_keys', embeddings.keys())
        lm.create_layers(embeddings, **kwargs)
        return lm

    def create_layers(self, embeddings, **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def predict(self, batch_dict, **kwargs):
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        batch_dict = self.make_input(batch_dict, numpy_to_tensor=numpy_to_tensor)
        hidden = batch_dict.get('h')
        step_softmax, _ = self(batch_dict, hidden)
        return F.softmax(step_softmax, dim=-1)


class WeightTieDense(tf.keras.layers.Layer):

    def __init__(self, tied, name='weight-tied', use_bias=False):
        super().__init__(name=name)
        self.tied = tied
        self.use_bias = use_bias

    def _add_bias(self, W):
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[tf.shape(W)[0]], initializer='zeros', regularizer=None, constraint=None, dtype=self.W.dtype, trainable=True)
        else:
            self.bias = None

    def build(self, input_shape):
        emb = getattr(self.tied, 'embedding_layer', None)
        if emb is not None:
            self.W = getattr(emb, 'W')
            self._add_bias(self.W)
            super().build(input_shape)
            return
        W = getattr(self.tied, 'W', None)
        if W is not None:
            self.W = W
            self._add_bias(self.W)
            super().build(input_shape)
            return
        self.W = getattr(self.tied, 'kernel')
        self._add_bias(self.W)
        super().build()

    def call(self, inputs):
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [-1, shape[-1]])
        outs = tf.matmul(inputs, self.W, transpose_b=True)
        if self.use_bias:
            outs = tf.nn.bias_add(outs, self.bias)
        new_shape = tf.concat([shape[:-1], tf.constant([-1])], axis=0)
        return tf.reshape(outs, new_shape)


class AbstractGeneratorLanguageModel(LanguageModelBase):

    def create_layers(self, embeddings, **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.embeddings_proj = self.init_embeddings_proj(**kwargs)
        self.generator = self.init_generate(**kwargs)
        self.output_layer = self.init_output(embeddings, **kwargs)

    def forward(self, input: Dict[str, TensorDef], hidden: TensorDef) ->Tuple[TensorDef, TensorDef]:
        emb = self.embed(input)
        output, hidden = self.generate(emb, hidden)
        return self.output_layer(output), hidden

    def embed(self, input):
        embedded_dropout = self.embeddings(input)
        return self.embeddings_proj(embedded_dropout)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) ->BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack({k: embeddings[k] for k in self.src_keys}, embeddings_dropout, reduction=reduction)

    def init_embeddings_proj(self, **kwargs):
        input_sz = self.embeddings.output_dim
        hsz = kwargs.get('hsz', kwargs.get('d_model'))
        if hsz != input_sz:
            proj = pytorch_linear(input_sz, hsz)
            None
        else:
            proj = nn.Identity()
        return proj

    def init_generate(self, **kwargs):
        pass

    def generate(self, emb, hidden):
        return self.generator((emb, hidden))

    def init_output(self, embeddings, **kwargs):
        self.vsz = embeddings[self.tgt_key].get_vsz()
        hsz = kwargs.get('hsz', kwargs.get('d_model'))
        unif = float(kwargs.get('unif', 0.0))
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        output_bias = kwargs.get('output_bias', False)
        if do_weight_tying:
            output = WeightTieDense(embeddings[self.tgt_key], output_bias)
        else:
            output = pytorch_linear(hsz, self.vsz, unif)
        return output


class LSTMEncoderWithState(nn.Module):
    """LSTM encoder producing the hidden state and the output, where the input doesnt require any padding

    PyTorch note: This type of encoder doesnt inherit the `LSTMEncoderWithState` base
    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0, batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs):
        """
        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param batch_first: PyTorch only! do batch first or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN

        """
        super().__init__()
        self.requires_length = False
        self.requires_state = True
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def forward(self, input_and_prev_h: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input_and_prev_h: The input at this timestep and the previous hidden unit or `None`
        :return: Raw `torch.nn.LSTM` output
        """
        inputs, hidden = input_and_prev_h
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden


class WithDropoutOnFirst(tf.keras.layers.Layer):
    """Wrapper for any layer that surrounds it with dropout

    This exists primarily for the LSTMEncoderWithState to allow dropout on the output while
    passing back the hidden state

    For variational dropout, we use `SpatialDropout1D` as described in:
    https://github.com/keras-team/keras/issues/7290
    """

    def __init__(self, layer: tf.keras.layers.Layer, pdrop: float=0.5, variational: bool=False):
        super().__init__()
        self.layer = layer
        self.dropout = tf.keras.layers.SpatialDropout1D(pdrop) if variational else tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        outputs = self.layer(inputs)
        return self.dropout(outputs[0], TRAIN_FLAG()), outputs[1]

    @property
    def output_dim(self) ->int:
        return self.layer.output_dim


class RNNLanguageModel(AbstractGeneratorLanguageModel):

    def __init__(self):
        super().__init__()

    def zero_state(self, batchsz):
        weight = next(self.parameters()).data
        return torch.autograd.Variable(weight.new(self.num_layers, batchsz, self.hsz).zero_()), torch.autograd.Variable(weight.new(self.num_layers, batchsz, self.hsz).zero_())

    @property
    def requires_state(self):
        True

    def init_generate(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        self.num_layers = kwargs.get('layers', kwargs.get('num_layers', 1))
        self.hsz = kwargs.get('hsz', kwargs.get('d_model'))
        return WithDropoutOnFirst(LSTMEncoderWithState(self.hsz, self.hsz, self.num_layers, pdrop, batch_first=True), pdrop, kwargs.get('variational', False))


def subsequent_mask(size: int):
    b = tf.compat.v1.matrix_band_part(tf.ones([size, size]), -1, 0)
    m = tf.reshape(b, [1, 1, size, size])
    return m


class TransformerLanguageModel(AbstractGeneratorLanguageModel):

    def __init__(self):
        super().__init__()

    @property
    def requires_state(self):
        False

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def init_generate(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.1))
        layers = kwargs.get('layers', kwargs.get('num_layers', 1))
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        num_heads = kwargs.get('num_heads', 4)
        d_ff = int(kwargs.get('d_ff', 4 * d_model))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        scale = bool(kwargs.get('scale', True))
        activation = kwargs.get('activation', 'gelu')
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        layer_norms_after = kwargs.get('layer_norms_after', False)
        return TransformerEncoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=scale, layers=layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k, activation=activation, layer_norm_eps=layer_norm_eps, layer_norms_after=layer_norms_after)

    def create_layers(self, embeddings, **kwargs):
        super().create_layers(embeddings, **kwargs)
        self.weight_std = kwargs.get('weight_std', 0.02)
        self.apply(self.init_layer_weights)

    def create_mask(self, bth):
        T = bth.shape[1]
        mask = subsequent_mask(T).type_as(bth)
        return mask

    def generate(self, bth, _):
        mask = self.create_mask(bth)
        return self.generator((bth, mask)), None


class TransformerMaskedLanguageModel(TransformerLanguageModel):

    def create_mask(self, bth):
        T = bth.shape[1]
        mask = torch.ones((1, 1, T, T)).type_as(bth)
        return mask


class ArcPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, hsz, beam_width=1):
        pass


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = tf.constant(np.arange(10).view(2, -1))
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = get_shape_as_list(t)
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = tf.tile(tf.expand_dims(t, dim + 1), tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim + 1:])
    return tf.reshape(tiled, new_shape)


class AbstractArcPolicy(ArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        pass

    def forward(self, encoder_output, hsz, beam_width=1):
        h_i = self.get_state(encoder_output)
        context = encoder_output.output
        if beam_width > 1:
            with torch.no_grad():
                context = repeat_batch(context, beam_width)
                if type(h_i) is tuple:
                    h_i = repeat_batch(h_i[0], beam_width, dim=1), repeat_batch(h_i[1], beam_width, dim=1)
                else:
                    h_i = repeat_batch(h_i, beam_width, dim=1)
        batch_size = context.shape[0]
        h_size = batch_size, hsz
        with torch.no_grad():
            init_zeros = context.data.new(*h_size).zero_()
        return h_i, init_zeros, context


BASELINE_SEQ2SEQ_ARC_POLICY = {}


@export
def register(cls, registry, name=None, error=''):
    if name is None:
        name = cls.__name__
    if name in registry:
        raise Exception('Error: attempt to re-define previously registered {} {} (old: {}, new: {})'.format(error, name, registry[name], cls))
    if hasattr(cls, 'create'):
        registry[name] = cls.create
    else:
        registry[name] = cls
    return cls


@export
@optional_params
def register_arc_policy(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ARC_POLICY, name, 'decoder')


class TransferLastHiddenPolicy(AbstractArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


class NoArcPolicy(AbstractArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        final_encoder_state = encoder_outputs.hidden
        if isinstance(final_encoder_state, tuple):
            s1, s2 = final_encoder_state
            return s1 * 0, s2 * 0
        return final_encoder_state * 0


def gather_k(a, b, best_idx, k):
    shape_a = get_shape_as_list(a)
    auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in tf.unstack(shape_a[:a.get_shape().ndims - 1]) + [k]], indexing='ij')
    sorted_b = tf.gather_nd(b, tf.stack(auxiliary_indices[:-1] + [best_idx], axis=-1))
    return sorted_b


def no_length_penalty(lengths):
    """A dummy function that returns a no penalty (1)."""
    return tf.expand_dims(np.ones_like(lengths), -1)


def update_lengths(lengths, eoses, idx):
    """Update the length of a generated tensor based on the first EOS found.

    This is useful for a decoding situation where tokens after an EOS
    can be something other than EOS. This also makes sure that a second
    generated EOS doesn't affect the lengths.

    :param lengths: `torch.LongTensor`: The lengths where zero means an
        unfinished sequence.
    :param eoses:  `torch.ByteTensor`: A mask that has 1 for sequences that
        generated an EOS.
    :param idx: `int`: What value to fill the finished lengths with (normally
        the current decoding timestep).

    :returns: `torch.Tensor`: The updated lengths tensor (same shape and type).
    """
    updatable_lengths = lengths == 0
    lengths_mask = updatable_lengths & eoses
    return masked_fill(lengths, lengths_mask, idx)


class BeamSearchBase:

    def __init__(self, beam: int=1, length_penalty=None, **kwargs):
        self.length_penalty = length_penalty if length_penalty else no_length_penalty
        self.K = beam

    def init(self, encoder_outputs):
        pass

    def step(self, paths, extra):
        pass

    def update(self, beams, extra):
        pass

    def __call__(self, encoder_outputs, **kwargs):
        """Perform batched Beam Search.

        Note:
            The paths and lengths generated do not include the <GO> token.

        :param encoder_outputs: `namedtuple` The outputs of the encoder class.
        :param init: `Callable(ecnoder_outputs: encoder_outputs, K: int)` -> Any: A
            callable that is called once at the start of the search to initialize
            things. This returns a blob that is passed to other callables.
        :param step: `Callable(paths: torch.LongTensor, extra) -> (probs: torch.FloatTensor, extra):
            A callable that is does a single decoding step. It returns the log
            probabilities over the vocabulary in the last dimension. It also returns
            any state the decoding process needs.
        :param update: `Callable(beams: torch.LongTensor, extra) -> extra:
            A callable that is called to edit the decoding state based on the selected
            best beams.
        :param length_penalty: `Callable(lengths: torch.LongTensor) -> torch.floatTensor
            A callable that generates a penalty based on the lengths. Lengths is
            [B, K] and the returned penalty should be [B, K, 1] (or [B, K, V] to
            have token based penalties?)

        :Keyword Arguments:
        * *beam* -- `int`: The number of beams to use.
        * *mxlen* -- `int`: The max number of steps to run the search for.

        :returns:
            tuple(preds: torch.LongTensor, lengths: torch.LongTensor, scores: torch.FloatTensor)
            preds: The predicted values: [B, K, max(lengths)]
            lengths: The length of each prediction [B, K]
            scores: The score of each path [B, K]
        """
        mxlen = kwargs.get('mxlen', 100)
        bsz = get_shape_as_list(encoder_outputs.output)[0]
        extra = self.init(encoder_outputs)
        paths = tf.fill((bsz, self.K, 1), Offsets.GO)
        log_probs = tf.zeros((bsz, self.K))
        lengths = tf.zeros((bsz, self.K), np.int32)
        for i in range(mxlen - 1):
            probs, extra = self.step(paths, extra)
            V = get_shape_as_list(probs)[-1]
            probs = tf.reshape(probs, (bsz, self.K, V))
            if i > 0:
                done_mask = lengths != 0
                done_mask = tf.expand_dims(done_mask, -1)
                eos_mask = tf.cast(tf.zeros((1, 1, V)) + tf.reshape(tf.cast(tf.range(V) == Offsets.EOS, tf.float32), (1, 1, V)), done_mask.dtype)
                mask = done_mask & eos_mask
                probs = masked_fill(probs, done_mask, -100000000.0)
                probs = masked_fill(probs, mask, 0)
                probs = tf.expand_dims(log_probs, -1) + probs
                valid_lengths = masked_fill(lengths, lengths == 0, i + 1)
                path_scores = probs / tf.cast(self.length_penalty(valid_lengths), tf.float32)
            else:
                path_scores = probs[:, (0), :]
            flat_scores = tf.reshape(path_scores, (bsz, -1))
            best_scores, best_idx = tf.math.top_k(flat_scores, self.K)
            probs = tf.reshape(probs, (bsz, -1))
            log_probs = gather_k(flat_scores, probs, best_idx, self.K)
            log_probs = tf.reshape(log_probs, (bsz, self.K))
            best_beams = best_idx // V
            best_idx = best_idx % V
            offsets = tf.range(bsz) * self.K
            offset_beams = best_beams + tf.expand_dims(offsets, -1)
            flat_beams = tf.reshape(offset_beams, [bsz * self.K])
            flat_paths = tf.reshape(paths, [bsz * self.K, -1])
            new_paths = tf.gather(flat_paths, flat_beams)
            new_paths = tf.reshape(new_paths, [bsz, self.K, -1])
            paths = tf.concat([new_paths, tf.expand_dims(best_idx, -1)], axis=2)
            flat_lengths = tf.reshape(lengths, [-1])
            lengths = tf.gather(flat_lengths, flat_beams)
            lengths = tf.reshape(lengths, (bsz, self.K))
            extra = self.update(flat_beams, extra)
            last = paths[:, :, (-1)]
            eoses = last == Offsets.EOS
            lengths = update_lengths(lengths, eoses, i + 1)
            if tf.reduce_sum(tf.cast(lengths != 0, np.int32)) == self.K:
                break
        else:
            probs, extra = self.step(paths, extra)
            V = get_shape_as_list(probs)[-1]
            probs = tf.reshape(probs, (bsz, self.K, V))
            probs = probs[:, :, (Offsets.EOS)]
            probs = masked_fill(probs, lengths != 0, 0)
            log_probs = log_probs + probs
            end_tokens = np.full((bsz, self.K, 1), Offsets.EOS)
            paths = tf.concat([paths, end_tokens], axis=2)
            lengths = update_lengths(lengths, np.ones_like(lengths) == 1, mxlen)
            best_scores = log_probs / tf.cast(tf.squeeze(self.length_penalty(lengths), -1), tf.float32)
        paths = paths[:, :, 1:]
        return paths, lengths, best_scores


@export
def create_seq2seq_arc_policy(**kwargs):
    arc_type = kwargs.get('arc_policy_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ARC_POLICY.get(arc_type)
    return Constructor()


def gnmt_length_penalty(lengths, alpha=0.8):
    """Calculate a length penalty from https://arxiv.org/pdf/1609.08144.pdf

    The paper states the penalty as (5 + |Y|)^a / (5 + 1)^a. This is implemented
    as ((5 + |Y|) / 6)^a for a (very) tiny performance boost

    :param lengths: `np.array`: [B, K] The lengths of the beams.
    :param alpha: `float`: A hyperparameter. See Table 2 for a search on this
        parameter.

    :returns:
        `torch.FloatTensor`: [B, K, 1] The penalties.
    """
    penalty = tf.constant(np.power((5.0 + lengths) / 6.0, alpha))
    return tf.expand_dims(penalty, -1)


BASELINE_SEQ2SEQ_DECODERS = {}


@export
@optional_params
def register_decoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_DECODERS, name, 'decoder')


class StackedGRUCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            self.layers.append(tf.keras.layers.GRUCell(rnn_size))

    def call(self, input, hidden):
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            input, h_i = layer(input, h_0)
            if i != self.num_layers:
                input = self.dropout(input)
            hs.append(h_i)
        hs = tf.stack(hs)
        return input, hs

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self.rnn_size


class StackedLSTMCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.rnn_size = rnn_size
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            self.layers.append(tf.keras.layers.LSTMCell(rnn_size, use_bias=False))

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self) ->int:
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self.rnn_size

    def call(self, input, hidden):
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            input, (h_i, c_i) = layer(input, (h_0[i], c_0[i]))
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs.append(h_i)
            cs.append(c_i)
        hs = tf.stack(hs)
        cs = tf.stack(cs)
        return input, (hs, cs)


def rnn_cell(insz: int, hsz: int, rnntype: str, nlayers: int, dropout: float):
    """This is a wrapper function around a stacked RNN cell

    :param insz: The input dimensions
    :param hsz: The hidden dimensions
    :param rnntype: An RNN type `gru` or `lstm`
    :param nlayers: The number of layers to stack
    :param dropout: The amount of dropout
    :return:
    """
    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


class RNNDecoder(torch.nn.Module):

    def __init__(self, tgt_embeddings, **kwargs):
        """Construct an RNN decoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides an RNN cell, followed by a linear projection, out to a softmax

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        super().__init__()
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs.get('rnntype', 'lstm')
        layers = kwargs.get('layers', 1)
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = rnn_cell(dsz, self.hsz, rnntype, layers, pdrop)
        self.dropout = torch.nn.Dropout(pdrop)
        self.init_attn(**kwargs)
        do_weight_tying = bool(kwargs.get('tie_weights', True))
        if do_weight_tying:
            if self.hsz != self.tgt_embeddings.get_dsz():
                raise ValueError('weight tying requires hsz == embedding dsz, got {} hsz and {} dsz'.format(self.hsz, self.tgt_embeddings.get_dsz()))
            self.preds = WeightTieDense(self.tgt_embeddings)
        else:
            self.preds = pytorch_linear(self.hsz, self.tgt_embeddings.get_vsz())

    @staticmethod
    def _basic_input(dst_embed_i, _):
        """
        In this function the destination embedding is passed directly to into the decoder.  The output of previous H
        is ignored.  This is implemented using a bound method to a field in the class for speed so that this decision
        is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param _: Ignored
        :return: basic input
        """
        return dst_embed_i.squeeze(0)

    @staticmethod
    def _feed_input(embed_i, attn_output_i):
        """
        In this function the destination embedding is concatenated with the previous attentional output and
        passed to the decoder. This is implemented using a bound method to a field in the class for speed
        so that this decision is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param output_i: This is the last H state
        :return: an input that is a concatenation of previous state and destination embedding
        """
        embed_i = embed_i.squeeze(0)
        return torch.cat([embed_i, attn_output_i], 1)

    def forward(self, encoder_outputs, dst):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context_bth = self.arc_policy(encoder_outputs, self.hsz)
        output_tbh, _ = self.decode_rnn(context_bth, h_i, output_i, dst.transpose(0, 1), src_mask)
        pred = self.output(output_tbh)
        return pred.transpose(0, 1).contiguous()

    def decode_rnn(self, context_bth, h_i, output_i, dst_bth, src_mask):
        embed_out_bth = self.tgt_embeddings(dst_bth)
        outputs = []
        for i, embed_i in enumerate(embed_out_bth.split(1)):
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            outputs.append(output_i)
        outputs_tbh = torch.stack(outputs)
        return outputs_tbh, h_i

    def attn(self, output_t, context, src_mask=None):
        return output_t

    def init_attn(self, **kwargs):
        pass

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0) * x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred


    class BeamSearch(BeamSearchBase):

        def __init__(self, parent, **kwargs):
            super().__init__(**kwargs)
            self.parent = parent

        def init(self, encoder_outputs):
            """Tile batches for encoder inputs and the likes."""
            src_mask = repeat_batch(encoder_outputs.src_mask, self.K)
            h_i, dec_out, context = self.parent.arc_policy(encoder_outputs, self.parent.hsz, self.K)
            return h_i, dec_out, context, src_mask

        def step(self, paths, extra):
            """Calculate the probs of the next output and update state."""
            h_i, dec_out, context, src_mask = extra
            last = paths[:, :, (-1)].view(1, -1)
            dec_out, h_i = self.parent.decode_rnn(context, h_i, dec_out, last, src_mask)
            probs = self.parent.output(dec_out)
            dec_out = dec_out.squeeze(0)
            return probs, (h_i, dec_out, context, src_mask)

        def update(self, beams, extra):
            """Select the correct hidden states and outputs to used based on the best performing beams."""
            h_i, dec_out, context, src_mask = extra
            h_i = tuple(hc[:, (beams), :] for hc in h_i)
            dec_out = dec_out[(beams), :]
            return h_i, dec_out, context, src_mask

    def beam_search(self, encoder_outputs, **kwargs):
        alpha = kwargs.get('alpha')
        if alpha is not None:
            kwargs['length_penalty'] = partial(gnmt_length_penalty, alpha=alpha)
        return RNNDecoder.BeamSearch(parent=self, **kwargs)(encoder_outputs)


class VectorSequenceAttention(tf.keras.layers.Layer):

    def __init__(self, hsz):
        super().__init__()
        self.hsz = hsz
        self.W_c = tf.keras.layers.Dense(hsz, use_bias=False)

    def call(self, qkvm):
        query_t, keys_bth, values_bth, keys_mask = qkvm
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)
        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        pass

    def _update(self, a, query_t, values_bth):
        B, H = get_shape_as_list(a)
        a = tf.reshape(a, [B, 1, H])
        c_t = tf.squeeze(a @ values_bth, 1)
        attended = tf.concat([c_t, query_t], -1)
        attended = tf.nn.tanh(self.W_c(attended))
        return attended


class BahdanauAttention(VectorSequenceAttention):

    def __init__(self, hsz: int):
        super().__init__(hsz)
        self.hsz = hsz
        self.W_a = tf.keras.layers.Dense(self.hsz, use_bias=False)
        self.E_a = tf.keras.layers.Dense(self.hsz, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        B, T, H = get_shape_as_list(keys_bth)
        q = tf.reshape(self.W_a(query_t), [B, 1, H])
        u = self.E_a(keys_bth)
        z = tf.nn.tanh(q + u)
        a = tf.squeeze(self.v(z), -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1000000000.0)
        a = tf.nn.softmax(a, axis=-1)
        return a

    def _update(self, a, query_t, values_bth):
        B, T_k = get_shape_as_list(a)
        a = tf.reshape(a, [B, 1, T_k])
        c_t = tf.squeeze(a @ values_bth, 1)
        attended = tf.concat([c_t, query_t], -1)
        attended = self.W_c(attended)
        return attended


class LuongDotProductAttention(VectorSequenceAttention):

    def __init__(self, hsz: int):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(query_t, 2)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1000000000.0)
        a = tf.nn.softmax(a, axis=-1)
        return a


class LuongGeneralAttention(VectorSequenceAttention):

    def __init__(self, hsz: int):
        super().__init__(hsz)
        self.W_a = tf.keras.layers.Dense(self.hsz, use_bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(self.W_a(query_t), 2)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1000000000.0)
        a = tf.nn.softmax(a, axis=-1)
        return a


class ScaledDotProductAttention(VectorSequenceAttention):

    def __init__(self, hsz: int):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(query_t, 2)
        a = a / math.sqrt(self.hsz)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1000000000.0)
        a = tf.nn.softmax(a, axis=-1)
        return a


class RNNDecoderWithAttn(RNNDecoder):

    def __init__(self, tgt_embeddings, **kwargs):
        super().__init__(tgt_embeddings, **kwargs)

    def init_attn(self, **kwargs):
        attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        if attn_type == 'dot':
            self.attn_module = LuongDotProductAttention(self.hsz)
        elif attn_type == 'concat' or attn_type == 'bahdanau':
            self.attn_module = BahdanauAttention(self.hsz)
        elif attn_type == 'sdp':
            self.attn_module = ScaledDotProductAttention(self.hsz)
        else:
            self.attn_module = LuongGeneralAttention(self.hsz)

    def attn(self, output_t, context, src_mask=None):
        return self.attn_module(output_t, context, context, src_mask)


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, pdrop: float, scale: bool=True, activation_type: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[int]=None, ffn_pdrop: Optional[float]=0.0, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name: str=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.layer_norms_after = layer_norms_after
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k, name='self_attention')
            self.src_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k, name='src_attention')
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k, name='self_attention')
            self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k, name='src_attention')
        self.ffn = FFN(d_model, activation_type, d_ff, pdrop=ffn_pdrop, name='ffn')
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x, tgt_mask)), TRAIN_FLAG())
        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory, src_mask)), TRAIN_FLAG())
        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x), TRAIN_FLAG())
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerDecoderStack(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, pdrop: float, scale: bool=True, layers: int=1, activation: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[Union[int, List[int]]]=None, ffn_pdrop: float=0.0, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)
        self.decoders = []
        self.ln = tf.identity if layer_norms_after else tf.keras.layers.LayerNormalization(epsilon=1e-06)
        for i in range(layers):
            self.decoders.append(TransformerDecoder(d_model, num_heads, pdrop, scale, activation, d_ff, ffn_pdrop=ffn_pdrop))

    def call(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        for layer in self.decoders:
            x = layer((x, memory, src_mask, tgt_mask))
        return self.ln(x)


TransformerEncoderOutput = namedtuple('TransformerEncoderOutput', ('output', 'src_mask'))


class TransformerDecoderWrapper(torch.nn.Module):

    def __init__(self, tgt_embeddings, dropout=0.5, layers=1, hsz=None, num_heads=4, scale=True, **kwargs):
        super().__init__()
        self.tgt_embeddings = tgt_embeddings
        dsz = self.tgt_embeddings.get_dsz()
        if hsz is None:
            hsz = dsz
        d_ff = int(kwargs.get('d_ff', 4 * hsz))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        activation = kwargs.get('activation', 'relu')
        scale = bool(kwargs.get('scale', True))
        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=hsz, d_ff=d_ff, pdrop=dropout, scale=scale, layers=layers, rpr_k=rpr_k, d_k=d_k, activation_type=activation)
        self.proj_to_dsz = self._identity
        self.proj_to_hsz = self._identity
        if hsz != dsz:
            self.proj_to_hsz = pytorch_linear(dsz, hsz)
            self.proj_to_dsz = pytorch_linear(hsz, dsz)
            del self.proj_to_dsz.weight
            self.proj_to_dsz.weight = torch.nn.Parameter(self.proj_to_hsz.weight.transpose(0, 1), requires_grad=True)
        do_weight_tying = bool(kwargs.get('tie_weights', True))
        if do_weight_tying:
            if hsz != self.tgt_embeddings.get_dsz():
                raise ValueError('weight tying requires hsz == embedding dsz, got {} hsz and {} dsz'.format(self.hsz, self.tgt_embeddings.get_dsz()))
            self.preds = WeightTieDense(self.tgt_embeddings)
        else:
            self.preds = pytorch_linear(dsz, self.tgt_embeddings.get_vsz())

    def _identity(self, x):
        return x

    def forward(self, encoder_output, dst):
        embed_out_bth = self.tgt_embeddings(dst)
        embed_out_bth = self.proj_to_hsz(embed_out_bth)
        context_bth = encoder_output.output
        T = embed_out_bth.shape[1]
        dst_mask = subsequent_mask(T).type_as(embed_out_bth)
        src_mask = encoder_output.src_mask.unsqueeze(1).unsqueeze(1)
        output = self.transformer_decoder((embed_out_bth, context_bth, src_mask, dst_mask))
        output = self.proj_to_dsz(output)
        prob = self.output(output)
        return prob

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0) * x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred


    class BeamSearch(BeamSearchBase):

        def __init__(self, parent, **kwargs):
            super().__init__(**kwargs)
            self.parent = parent

        def init(self, encoder_outputs):
            """Tile for the batch of the encoder inputs."""
            encoder_outputs = TransformerEncoderOutput(repeat_batch(encoder_outputs.output, self.K), repeat_batch(encoder_outputs.src_mask, self.K))
            return encoder_outputs

        def step(self, paths, extra):
            """Calculate the probs for the last item based on the full path."""
            B, K, T = paths.size()
            assert K == self.K
            return self.parent(extra, paths.view(B * K, T))[:, (-1)], extra

        def update(self, beams, extra):
            """There is no state for the transformer so just pass it."""
            return extra

    def beam_search(self, encoder_outputs, **kwargs):
        return TransformerDecoderWrapper.BeamSearch(parent=self, **kwargs)(encoder_outputs)


class BiLSTMEncoderAll(BiLSTMEncoderBase):
    """BiLSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, concat_state_dirs(hidden)


class LSTMEncoderAll(LSTMEncoderBase):
    """LSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H]` or `[B, H, S]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, hidden


RNNEncoderOutput = namedtuple('RNNEncoderOutput', ('output', 'hidden', 'src_mask'))


def _make_src_mask(output, lengths):
    T = output.shape[1]
    src_mask = sequence_mask(lengths, T).type_as(lengths.data)
    return src_mask


BASELINE_SEQ2SEQ_ENCODERS = {}


@export
@optional_params
def register_encoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ENCODERS, name, 'encoder')


class RNNEncoder(torch.nn.Module):

    def __init__(self, dsz=None, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, **kwargs):
        super().__init__()
        self.residual = residual
        hidden = hsz if hsz is not None else dsz
        Encoder = LSTMEncoderAll if rnntype == 'lstm' else BiLSTMEncoderAll
        self.rnn = Encoder(dsz, hidden, layers, pdrop, batch_first=True)
        self.src_mask_fn = _make_src_mask if create_src_mask is True else lambda x, y: None

    def forward(self, btc, lengths):
        output, hidden = self.rnn((btc, lengths))
        return RNNEncoderOutput(output=output + btc if self.residual else output, hidden=hidden, src_mask=self.src_mask_fn(output, lengths))


class TransformerEncoderWrapper(torch.nn.Module):

    def __init__(self, dsz, hsz=None, num_heads=4, layers=1, dropout=0.5, **kwargs):
        super().__init__()
        if hsz is None:
            hsz = dsz
        self.proj = pytorch_linear(dsz, hsz) if hsz != dsz else self._identity
        d_ff = int(kwargs.get('d_ff', 4 * hsz))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        activation = kwargs.get('activation', 'relu')
        scale = bool(kwargs.get('scale', True))
        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, d_ff=d_ff, pdrop=dropout, scale=scale, layers=layers, rpr_k=rpr_k, d_k=d_k, activation=activation)

    def _identity(self, x):
        return x

    def forward(self, bth, lengths):
        T = bth.shape[1]
        src_mask = sequence_mask(lengths, T).type_as(lengths.data)
        bth = self.proj(bth)
        output = self.transformer((bth, src_mask.unsqueeze(1).unsqueeze(1)))
        return TransformerEncoderOutput(output=output, src_mask=src_mask)


@export
class EncoderDecoderModel(object):
    task_name = 'seq2seq'

    def save(self, model_base):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def load(basename, **kwargs):
        pass

    @classmethod
    def create(cls, src_embeddings, dst_embedding, **kwargs):
        pass

    def create_loss(self):
        pass

    def predict(self, source_dict, **kwargs):
        pass

    def run(self, source_dict, **kwargs):
        logger.warning('`run` is deprecated, use `predict` instead.')
        return self.predict(source_dict, **kwargs)


@export
def create_seq2seq_decoder(tgt_embeddings, **kwargs):
    decoder_type = kwargs.get('decoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_DECODERS.get(decoder_type)
    return Constructor(tgt_embeddings, **kwargs)


@export
def create_seq2seq_encoder(**kwargs):
    encoder_type = kwargs.get('encoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ENCODERS.get(encoder_type)
    return Constructor(**kwargs)


class EncoderDecoderModelBase(nn.Module, EncoderDecoderModel):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super().__init__()
        self.beam_sz = kwargs.get('beam', 1)
        self.gpu = kwargs.get('gpu', True)
        src_dsz = self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.dropin_values = kwargs.get('dropin', {})
        self.encoder = self.init_encoder(src_dsz, **kwargs)
        self.decoder = self.init_decoder(tgt_embedding, **kwargs)

    def init_embed(self, src_embeddings, tgt_embedding, **kwargs):
        """This is the hook for providing embeddings.  It takes in a dictionary of `src_embeddings` and a single
        tgt_embedding` of type `PyTorchEmbedding`

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings, one per embedding
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        :return: Return the aggregate embedding input size
        """
        self.src_embeddings = EmbeddingsStack(src_embeddings, reduction=kwargs.get('embeddings_reduction', 'concat'))
        return self.src_embeddings.output_dim

    def init_encoder(self, input_sz, **kwargs):
        kwargs['dsz'] = input_sz
        return create_seq2seq_encoder(**kwargs)

    def init_decoder(self, tgt_embedding, **kwargs):
        return create_seq2seq_decoder(tgt_embedding, **kwargs)

    def encode(self, input, lengths):
        """

        :param input:
        :param lengths:
        :return:
        """
        embed_in_seq = self.embed(input)
        return self.encoder(embed_in_seq, lengths)

    def decode(self, encoder_outputs, dst):
        return self.decoder(encoder_outputs, dst)

    def save(self, model_file):
        """Save the model out

        :param model_file: (``str``) The filename
        :return:
        """
        torch.save(self, model_file)

    def create_loss(self):
        """Create a loss function.

        :return:
        """
        return SequenceCriterion()

    @classmethod
    def load(cls, filename, **kwargs):
        """Load a model from file

        :param filename: (``str``) The filename
        :param kwargs:
        :return:
        """
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        model = cls(src_embeddings, tgt_embedding, **kwargs)
        logger.info(model)
        return model

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)
        if not self.training or v == 0:
            return x
        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx, numpy_to_tensor=False):
        tensor = batch_dict[key]
        if numpy_to_tensor:
            tensor = torch.from_numpy(tensor)
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor
        return tensor

    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):
        """Prepare the input.

        :param batch_dict: `dict`: The data.
        :param perm: `bool`: If True return the permutation index
            so that you can undo the sort if you want.
        """
        example = dict({})
        lengths = batch_dict[self.src_lengths_key]
        if numpy_to_tensor:
            lengths = torch.from_numpy(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)
        if self.gpu:
            lengths = lengths
        example['src_len'] = lengths
        for key in self.src_embeddings.keys():
            example[key] = self.input_tensor(key, batch_dict, perm_idx, numpy_to_tensor=numpy_to_tensor)
        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt']
            if numpy_to_tensor:
                tgt = torch.from_numpy(tgt)
            example['dst'] = tgt[:, :-1]
            example['tgt'] = tgt[:, 1:]
            example['dst'] = example['dst'][perm_idx]
            example['tgt'] = example['tgt'][perm_idx]
            if self.gpu:
                example['dst'] = example['dst']
                example['tgt'] = example['tgt']
        if perm:
            return example, perm_idx
        return example

    def embed(self, input):
        return self.src_embeddings(input)

    def forward(self, input: Dict[str, torch.Tensor]):
        src_len = input['src_len']
        encoder_outputs = self.encode(input, src_len)
        output = self.decode(encoder_outputs, input['dst'])
        return output

    def predict(self, batch_dict, **kwargs):
        """Predict based on the batch.

        If `make_input` is True then run make_input on the batch_dict.
        This is false for being used during dev eval where the inputs
        are already transformed.
        """
        self.eval()
        make = kwargs.get('make_input', True)
        if make:
            numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
            inputs, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        else:
            inputs = batch_dict
        encoder_outputs = self.encode(inputs, inputs['src_len'])
        outs, lengths, scores = self.decoder.beam_search(encoder_outputs, **kwargs)
        if make:
            outs = unsort_batch(outs, perm_idx)
            lengths = unsort_batch(lengths, perm_idx)
            scores = unsort_batch(scores, perm_idx)
        return outs.cpu().numpy()


class Seq2SeqModel(EncoderDecoderModelBase):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        """This base model is extensible for attention and other uses.  It declares minimal fields allowing the
        subclass to take over most of the duties for drastically different implementations

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        """
        super().__init__(src_embeddings, tgt_embedding, **kwargs)


@export
class TaggerModel:
    """Structured prediction classifier, AKA a tagger

    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
    task_name = 'tagger'

    def __init__(self):
        super().__init__()

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

    def get_labels(self):
        pass


class TaggerModelBase(nn.Module, TaggerModel):
    """Base class for tagger models

    This class provides the model base for tagging.  To create a tagger, overload `create_layers()` and `forward()`.
    Most implementations should be able to subclass the `AbstractEncoderTaggerModel`, which inherits from this and imposes
    additional structure
    """

    def __init__(self):
        """Constructor"""
        super().__init__()
        self.gpu = False

    def save(self, outname: str):
        """Save out the model

        :param outname: The name of the checkpoint to write
        :return:
        """
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)
        write_json(self.labels, basename + '.labels')

    def cuda(self, device=None):
        self.gpu = True
        return super()

    @staticmethod
    def load(filename: str, **kwargs) ->'TaggerModelBase':
        """Create and load a tagger model from file

        """
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def drop_inputs(self, key, x):
        """Do dropout on inputs, using the dropout value (or none if not set)
        This works by applying a dropout mask with the probability given by a
        value within the `dropin_values: Dict[str, float]`, keyed off the text name
        of the feature

        :param key: The feature name
        :param x: The tensor to drop inputs for
        :return: The dropped out tensor
        """
        v = self.dropin_values.get(key, 0)
        if not self.training or v == 0:
            return x
        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v)
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx, numpy_to_tensor=False):
        """Given a batch of input, and a key, prepare and noise the input

        :param key: The key of the tensor
        :param batch_dict: The batch of data as a dictionary of tensors
        :param perm_idx: The proper permutation order to get lengths descending
        :param numpy_to_tensor: Should this tensor be converted to a `torch.Tensor`
        :return:
        """
        tensor = batch_dict[key]
        if numpy_to_tensor:
            tensor = torch.from_numpy(tensor)
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor
        return tensor

    def make_input(self, batch_dict: Dict[str, TensorDef], perm: bool=False, numpy_to_tensor: bool=False) ->Dict[str, TensorDef]:
        """Transform a `batch_dict` into format suitable for tagging

        :param batch_dict: A dictionary containing all inputs to the embeddings for this model
        :param perm: Should we sort data by length descending?
        :param numpy_to_tensor: Do we need to convert the input from numpy to a torch.Tensor?
        :return: A dictionary representation of this batch suitable for processing
        """
        example_dict = dict({})
        lengths = batch_dict[self.lengths_key]
        if numpy_to_tensor:
            lengths = torch.from_numpy(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)
        if self.gpu:
            lengths = lengths
        example_dict['lengths'] = lengths
        for key in self.embeddings.keys():
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx, numpy_to_tensor=numpy_to_tensor)
        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            y = y[perm_idx]
            if self.gpu:
                y = y
            example_dict['y'] = y
        ids = batch_dict.get('ids')
        if ids is not None:
            if numpy_to_tensor:
                ids = torch.from_numpy(ids)
            ids = ids[perm_idx]
            if self.gpu:
                ids = ids
            example_dict['ids'] = ids
        if perm:
            return example_dict, perm_idx
        return example_dict

    def get_labels(self) ->List[str]:
        """Get the labels (names of each class)

        :return: (`List[str]`) The labels
        """
        return self.labels

    def predict(self, batch_dict: Dict[str, TensorDef], **kwargs) ->TensorDef:
        """Take in a batch of data, and predict the tags

        :param batch_dict: A batch of features that is to be predicted
        :param kwargs: See Below

        :Keyword Arguments:

        * *numpy_to_tensor* (``bool``) Should we convert input from numpy to `torch.Tensor` Defaults to `True`
        :return: A batch-sized tensor of predictions
        """
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        inputs, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        outputs = self(inputs)
        return unsort_batch(outputs, perm_idx)

    @classmethod
    def create(cls, embeddings: Dict[str, TensorDef], labels: List[str], **kwargs) ->'TaggerModelBase':
        """Create a tagger from the inputs.  Most classes shouldnt extend this

        :param embeddings: A dictionary containing the input feature indices
        :param labels: A list of the labels (tags)
        :param kwargs: See below

        :Keyword Arguments:

        * *lengths_key* (`str`) Which feature identifies the length of the sequence
        * *activation* (`str`) What type of activation function to use (defaults to `tanh`)
        * *dropout* (`str`) What fraction dropout to apply
        * *dropin* (`str`) A dictionarwith feature keys telling what fraction of word masking to apply to each feature

        :return:
        """
        model = cls()
        model.lengths_key = kwargs.get('lengths_key')
        model.activation_type = kwargs.get('activation', 'tanh')
        model.pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.create_layers(embeddings, **kwargs)
        return model

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def compute_loss(self, inputs):
        """Define a loss function from the inputs, which includes the gold tag values as `inputs['y']`

        :param inputs:
        :return:
        """


class CRF(tf.keras.layers.Layer):

    def __init__(self, num_tags: int, constraint_mask: Optional[Tuple[Any, Any]]=None, name: Optional[str]=None):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint_mask: Tuple[np.ndarray, np.ndarray], Constraints on the transitions [1, N, N]
        :param name: str, Optional name, defaults to `None`
        """
        super().__init__(name=name)
        self.num_tags = num_tags
        self.A = None
        self.mask = None
        self.inv_mask = None
        if constraint_mask is not None:
            mask, inv_mask = constraint_mask
            self.mask = mask
            self.inv_mask = inv_mask * -10000.0

    def build(self, input_shape):
        self.A = self.add_weight('transitions', shape=(self.num_tags, self.num_tags), dtype=tf.float32, initializer=tf.zeros_initializer())
        if self.mask is not None:
            self.mask = self.add_weight('constraint_mask', shape=(self.num_tags, self.num_tags), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(self.mask))
        if self.inv_mask is not None:
            self.inv_mask = self.add_weight('inverse_constraint_mask', shape=(self.num_tags, self.num_tags), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(self.inv_mask))

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return self.A * self.mask + self.inv_mask
        return self.A

    def score_sentence(self, unary, tags, lengths):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [B, T, N]
        :param tags: torch.LongTensor: [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        return crf_sequence_score(unary, tf.cast(tags, tf.int32), tf.cast(lengths, tf.int32), self.transitions)

    def call(self, inputs, training=False):
        unary, lengths = inputs
        if training:
            return crf_log_norm(unary, lengths, self.transitions)
        else:
            return self.decode(unary, lengths)

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        bsz = tf.shape(unary)[0]
        lsz = self.num_tags
        np_gos = np.full((1, 1, lsz), -10000.0, dtype=np.float32)
        np_gos[:, :, (Offsets.GO)] = 0
        gos = tf.constant(np_gos)
        start = tf.tile(gos, [bsz, 1, 1])
        start = tf.nn.log_softmax(start, axis=-1)
        probv = tf.concat([start, unary], axis=1)
        viterbi, path_scores = crf_decode(probv, self.transitions, lengths + 1)
        return tf.identity(viterbi[:, 1:], name='best'), path_scores

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: unary outputs of length `[B, S, N]`
        :param tags: tag truth values `[B, T]`
        :param lengths: tensor of shape `[B]`

        :return: Tensor of shape `[B]`
        """
        lengths = tf.cast(lengths, tf.int32)
        max_length = tf.reduce_max(lengths)
        fwd_score = self((unary, lengths), training=True)
        tags = tags[:, :max_length]
        gold_score = self.score_sentence(unary, tags, lengths)
        log_likelihood = gold_score - fwd_score
        return -tf.reduce_mean(log_likelihood)


class TaggerGreedyDecoder(tf.keras.layers.Layer):

    def __init__(self, num_tags: int, constraint_mask: Optional[Tuple[Any, Any]]=None, name: Optional[str]=None):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint_mask: Tuple[np.ndarray, np.ndarray], Constraints on the transitions [1, N, N]
        :param name: str, Optional name, defaults to `None`
        """
        super().__init__(name=name)
        self.num_tags = num_tags
        self.A = None
        self.inv_mask = None
        if constraint_mask is not None:
            _, inv_mask = constraint_mask
            self.inv_mask = inv_mask * -10000.0

    def build(self, input_shape):
        self.A = self.add_weight('transitions', shape=(self.num_tags, self.num_tags), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        if self.inv_mask is not None:
            self.inv_mask = self.add_weight('inverse_constraint_mask', shape=(self.num_tags, self.num_tags), dtype=tf.float32, initializer=tf.constant_initializer(self.inv_mask), trainable=False)

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return tf.nn.log_softmax(self.A + self.inv_mask)
        return self.A

    def neg_log_loss(self, unary, tags, lengths):
        lengths = tf.cast(lengths, tf.int32)
        max_length = tf.reduce_max(lengths)
        tags = tags[:, :max_length]
        mask = tf.sequence_mask(lengths, max_length)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tags, logits=unary)
        cross_entropy *= tf.cast(mask, tf.float32)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        return tf.reduce_mean(cross_entropy, name='loss')

    def call(self, inputs, training=False, mask=None):
        unary, lengths = inputs
        if self.inv_mask is not None:
            bsz = tf.shape(unary)[0]
            lsz = self.num_tags
            np_gos = np.full((1, 1, lsz), -10000.0, dtype=np.float32)
            np_gos[:, :, (Offsets.GO)] = 0
            gos = tf.constant(np_gos)
            start = tf.tile(gos, [bsz, 1, 1])
            probv = tf.concat([start, unary], axis=1)
            viterbi, path_scores = crf_decode(probv, self.transitions, lengths + 1)
            return tf.identity(viterbi[:, 1:], name='best'), path_scores
        else:
            return tf.argmax(unary, 2, name='best'), None


class AbstractEncoderTaggerModel(TaggerModelBase):
    """Class defining a typical flow for taggers.  Most taggers should extend this class

    This class provides the model base for tagging by providing specific hooks for each phase.  There are
    4 basic steps identified in this class:

    1. embed
    2. encode (transduction)
    3. proj (projection to the final number of labels)
    4. decode

    There is an `init_* method for each of this phases, allowing you to
    define and return a custom layer.

    The actual forward method is defined as a combination of these 3 steps, which includes a
    projection from the encoder output to the number of labels.

    Decoding in taggers refers to the process of selecting the best path through the labels and is typically
    implemented either as a constrained greedy decoder or as a CRF layer
    """

    def __init__(self):
        """Constructor"""
        super().__init__()

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) ->BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        return EmbeddingsStack(embeddings, self.pdrop, reduction=kwargs.get('embeddings_reduction', 'concat'))

    def init_encode(self, input_dim, **kwargs) ->BaseLayer:
        """Provide a layer object that represents the `encode` phase of the model
        :param input_dim: The hidden input size
        :param kwargs:
        :return: The encoder
        """

    def init_proj(self, **kwargs) ->BaseLayer:
        """Provide a projection from the encoder output to the number of labels

        This projection typically will not include any activation, since its output is the logits that
        the decoder is built on

        :param kwargs:
        :return: A projection from the encoder output size to the final number of labels
        """
        return Dense(self.encoder.output_dim, len(self.labels))

    def init_decode(self, **kwargs) ->BaseLayer:
        """Define a decoder from the inputs

        :param kwargs: See below
        :keyword Arguments:
        * *crf* (``bool``) Should we create a CRF as the decoder
        * *constraint_mask* (``tensor``) A constraint mask to apply to the transitions
        * *reduction* (``str``) How to reduce the loss, defaults to `batch`
        :return: A decoder layer
        """
        use_crf = bool(kwargs.get('crf', False))
        constraint_mask = kwargs.get('constraint_mask')
        if constraint_mask is not None:
            constraint_mask = constraint_mask.unsqueeze(0)
        if use_crf:
            decoder = CRF(len(self.labels), constraint_mask=constraint_mask, batch_first=True)
        else:
            decoder = TaggerGreedyDecoder(len(self.labels), constraint_mask=constraint_mask, batch_first=True, reduction=kwargs.get('reduction', 'batch'))
        return decoder

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This class overrides this method to produce the outline of steps for a transduction tagger

        :param embeddings: The input embeddings dict
        :param kwargs:
        :return:
        """
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.encoder = self.init_encode(self.embeddings.output_dim, **kwargs)
        self.proj_layer = self.init_proj(**kwargs)
        self.decoder = self.init_decode(**kwargs)

    def transduce(self, inputs: Dict[str, TensorDef]) ->TensorDef:
        """This operation performs embedding of the input, followed by encoding and projection to logits

        :param inputs: The feature indices to embed
        :return: Transduced (post-encoding) output
        """
        lengths = inputs['lengths']
        embedded = self.embeddings(inputs)
        embedded = embedded, lengths
        transduced = self.proj_layer(self.encoder(embedded))
        return transduced

    def decode(self, tensor: TensorDef, lengths: TensorDef) ->TensorDef:
        """Take in the transduced (encoded) input and decode it

        :param tensor: Transduced input
        :param lengths: Valid lengths of the transduced input
        :return: A best path through the output
        """
        return self.decoder((tensor, lengths))

    def forward(self, inputs: Dict[str, TensorDef]) ->TensorDef:
        """Take the input and produce the best path of labels out

        :param inputs: The feature indices for the input
        :return: The most likely path through the output labels
        """
        transduced = self.transduce(inputs)
        path = self.decode(transduced, inputs.get('lengths'))
        return path

    def compute_loss(self, inputs):
        """Provide the loss by requesting it from the decoder

        :param inputs: A batch of inputs
        :return:
        """
        tags = inputs['y']
        lengths = inputs['lengths']
        unaries = self.transduce(inputs)
        return self.decoder.neg_log_loss(unaries, tags, lengths)


class BiLSTMEncoderSequence(BiLSTMEncoderBase):
    """BiLSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.


    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->torch.Tensor:
        """Take in a tuple of `(sequence, lengths)` and produce and output tensor of the last layer of LSTMs

        The value `S` here is defined as `max(lengths)`, `S <= T`

        :param inputs: sequence of shapes `[B, T, C]` or `[T, B, C]` and a lengths of shape `[B]`
        :return: A tensor of shape `[B, S, H]` or `[S, B, H]` depending on setting of `batch_first`
        """
        tensor, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tensor, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class LSTMEncoderSequence(LSTMEncoderBase):
    """LSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    *PyTorch Note:* The input shape of is either `[B, T, C]` or `[T, B, C]` depending on the value of `batch_first`,
    and defaults to `[T, B, C]` for consistency with other PyTorch modules. The output shape is of the same orientation.
    """

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->torch.Tensor:
        """Take in a tuple of `(sequence, lengths)` and produce and output tensor of the last layer of LSTMs

        The value `S` here is defined as `max(lengths)`, `S <= T`

        :param inputs: sequence of shapes `[B, T, C]` or `[T, B, C]` and a lengths of shape `[B]`
        :return: A tensor of shape `[B, S, H]` or `[S, B, H]` depending on setting of `batch_first`
        """
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output


class RNNTaggerModel(AbstractEncoderTaggerModel):
    """RNN-based tagger implementation: this is the default tagger for mead-baseline

    Overload the encoder, typically as a BiLSTM
    """

    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) ->BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *rnntype* (``str``) The type of RNN, defaults to `blstm`
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate
        * *weight_init* (``str``) The weight initializer, defaults to `uniform`
        * *unif* (``float``) A value for the weight initializer
        :return: An encoder
        """
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = int(kwargs.get('layers', 1))
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        pdrop = float(kwargs.get('dropout', 0.5))
        weight_init = kwargs.get('weight_init', 'uniform')
        Encoder = LSTMEncoderSequence if rnntype == 'lstm' else BiLSTMEncoderSequence
        return Encoder(input_dim, hsz, nlayers, pdrop, unif=unif, initializer=weight_init, batch_first=True)


class TransformerEncoderStackWithLengths(TransformerEncoderStack):

    def __init__(self, num_heads: int, d_model: int, pdrop: bool, scale: bool=True, layers: int=1, activation: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[Union[int, List[int]]]=None, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name: Optional[str]=None, **kwargs):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k, layer_norms_after, layer_norm_eps, name=name)
        self.proj = WithDropout(tf.keras.layers.Dense(d_model), pdrop)

    def call(self, inputs):
        x, lengths = inputs
        x = self.proj(x)
        max_seqlen = get_shape_as_list(x)[1]
        mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(lengths, max_seqlen, dtype=tf.float32), 1), 1)
        return super().call((x, mask))


class TransformerTaggerModel(AbstractEncoderTaggerModel):
    """Transformer-based tagger model

    Overload the encoder using a length-aware Transformer
    """

    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) ->BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *num_heads* (``int``) The number of heads for multi-headed attention
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *d_ff* (``int``) The feed-forward layer size
        * *rpr_k* (``list`` or ``int``) The relative attention sizes.  If its a list, one scalar per layer, if its
          a scalar, apply same size to each layer
        :return: An encoder
        """
        layers = int(kwargs.get('layers', 1))
        num_heads = int(kwargs.get('num_heads', 4))
        pdrop = float(kwargs.get('dropout', 0.5))
        scale = False
        hsz = int(kwargs['hsz'])
        rpr_k = kwargs.get('rpr_k', 100)
        d_ff = kwargs.get('d_ff')
        encoder = TransformerEncoderStackWithLengths(num_heads, hsz, pdrop, scale, layers, d_ff=d_ff, rpr_k=rpr_k, input_sz=input_dim)
        return encoder


class ConvEncoder(tf.keras.layers.Layer):

    def __init__(self, insz: Optional[int], outsz: int, filtsz: int, pdrop: float=0.0, activation: str='relu', name=None):
        super().__init__(name=name)
        self.output_dim = outsz
        self.conv = tf.keras.layers.Conv1D(filters=outsz, kernel_size=filtsz, padding='same')
        self.act = get_activation(activation)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        conv_out = self.act(self.conv(inputs))
        return self.dropout(conv_out, TRAIN_FLAG())


class ConvEncoderStack(tf.keras.layers.Layer):

    def __init__(self, insz: Optional[int], outsz: int, filtsz: int, nlayers: int=1, pdrop: float=0.0, activation: str='relu', name=None):
        super().__init__(name=name)
        self.layers = []
        first_layer = ConvEncoder(insz, outsz, filtsz, pdrop, activation)
        self.layers.append(first_layer)
        for i in range(nlayers - 1):
            subsequent_layer = ResidualBlock(ConvEncoder(insz, outsz, filtsz, pdrop, activation))
            self.layers.append(subsequent_layer)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNNTaggerModel(AbstractEncoderTaggerModel):
    """Convolutional (AKA TDNN) tagger

    Overload the encoder using a conv layer

    """

    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) ->BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *activation_type* (``str``) Defaults to `relu`
        * *wfiltsz* (``int``) The 1D filter size for the convolution
        :return: An encoder
        """
        layers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        filtsz = kwargs.get('wfiltsz', 5)
        activation_type = kwargs.get('activation_type', 'relu')
        hsz = int(kwargs['hsz'])
        return WithoutLength(ConvEncoderStack(input_dim, hsz, filtsz, layers, pdrop, activation_type))


class PassThruTaggerModel(AbstractEncoderTaggerModel):
    """A Pass-thru implementation of the encoder

    When we fine-tune our taggers from things like BERT embeddings, we might want to just pass through our
    embedding result directly to the output decoder.  This model provides a mechanism for this by providing
    a simple identity layer
    """

    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) ->BaseLayer:
        """Identity layer encoder

        :param input_dim: The input dims
        :param kwargs: None
        :return: An encoder
        """
        return WithoutLength(PassThru(input_dim))


class PyTorchEmbeddings(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    @property
    def output_dim(self):
        return self.get_dsz()

    def encode(self, x):
        return self(x)


def pytorch_embedding(weights: torch.Tensor, finetune: bool=True) ->nn.Embedding:
    """Creation function for making an nn.Embedding with the given weights

    :param weights: The weights to use
    :param finetune: Should we fine-tune the embeddings or freeze them
    """
    lut = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=Offsets.PAD)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=finetune)
    return lut


class LookupTableEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.dropin = kwargs.get('dropin', 0.0)
        weights = kwargs.get('weights')
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=Offsets.PAD)
        else:
            self.embeddings = pytorch_embedding(weights, self.finetune)
            self.vsz, self.dsz = weights.shape

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.dsz

    def forward(self, x):
        if not self.dropin:
            return self.embeddings(x)
        mask = self.embeddings.weight.data.new().resize_((self.embeddings.weight.size(0), 1)).bernoulli_(1 - self.dropin).expand_as(self.embeddings.weight) / (1 - self.dropin)
        masked_embed_weight = mask * self.embeddings.weight
        output = torch.nn.functional.embedding(x, masked_embed_weight, self.embeddings.padding_idx, self.embeddings.max_norm, self.embeddings.norm_type, self.embeddings.scale_grad_by_freq, self.embeddings.sparse)
        return output

    def extra_repr(self):
        return f'finetune=False' if not self.finetune else ''


class Highway(tf.keras.layers.Layer):

    def __init__(self, input_size: int, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)
        self.proj = tf.keras.layers.Dense(input_size, activation='relu')
        self.transform = tf.keras.layers.Dense(input_size, bias_initializer=tf.keras.initializers.Constant(value=-2.0), activation='sigmoid')
        self.output_dim = input_size

    def call(self, inputs):
        proj_result = self.proj(inputs)
        proj_gate = self.transform(inputs)
        gated = proj_gate * proj_result + (1 - proj_gate) * inputs
        return gated

    @property
    def requires_length(self):
        return False


@export
def calc_nfeats(filtsz: Union[List[Tuple[int, int]], List[int]], nfeat_factor: Optional[int]=None, max_feat: Optional[int]=None, nfeats: Optional[int]=None) ->Tuple[List[int], List[int]]:
    """Calculate the output sizes to use for multiple parallel convolutions.

    If filtsz is a List of Lists of ints then we assume that each element represents
        a filter size, feature size pair. This is the format used by ELMo
    If filtsz is a List of ints we assume each element represents a filter size
    If nfeat_factor and max_feat are set we calculate the nfeat size based on the
        nfeat_factor and the filter size capped by max_feat. This is the method used
        in Kim et. al. 2015 (https://arxiv.org/abs/1508.06615)
    Otherwise nfeats must be set and we assume this is output size to use for all of
        the parallel convs and return the feature size expanded to list the same length
        as filtsz

    :param filtsz: The filter sizes to use in parallel
    :param nfeat_factor: How to scale the feat size as you grow the filters
    :param max_feat: The cap on the feature size
    :param nfeats: A fall back constant feature size
    :returns: Associated arrays where the first one is the filter sizes and the second
        one has the corresponding number of feats as the output
    """
    if is_sequence(filtsz[0]):
        filtsz, nfeats = zip(*filtsz)
    elif nfeat_factor is not None:
        assert max_feat is not None, 'If using `nfeat_factor`, `max_feat` must not be None'
        nfeats = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
    else:
        assert nfeats is not None, 'When providing only `filtsz` and not `nfeat_factor` `nfeats` must be specified'
        assert isinstance(nfeats, int), 'If you want to use custom nfeat sizes do `filtsz = zip(filtsz, nfeats)` then call this function'
        nfeats = [nfeats] * len(filtsz)
    return filtsz, nfeats


class CharConvEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nfeat_factor = kwargs.get('nfeat_factor')
        self.cfiltsz = kwargs.get('cfiltsz', kwargs.get('filtsz', [3]))
        self.max_feat = kwargs.get('max_feat', 30)
        self.gating = kwargs.get('gating', 'skip')
        self.num_gates = kwargs.get('num_gates', 1)
        self.activation = kwargs.get('activation', 'tanh')
        self.wsz = kwargs.get('wsz', 30)
        self.projsz = kwargs.get('projsz', 0)
        self.pdrop = kwargs.get('pdrop', 0.5)
        self.filtsz, self.nfeats = calc_nfeats(self.cfiltsz, self.nfeat_factor, self.max_feat, self.wsz)
        self.conv_outsz = int(np.sum(self.nfeats))
        self.outsz = self.conv_outsz
        if self.projsz > 0:
            self.outsz = self.projsz
        self.proj = pytorch_linear(self.conv_outsz, self.outsz)
        self.embeddings = LookupTableEmbeddings(**kwargs)
        self.char_comp = WithDropout(ParallelConv(self.embeddings.output_dim, self.nfeats, self.filtsz, self.activation), self.pdrop)
        GatingConnection = SkipConnection if self.gating == 'skip' else Highway
        self.gating_seq = nn.Sequential(OrderedDict([('gate-{}'.format(i), GatingConnection(self.char_comp.output_dim)) for i in range(self.num_gates)]))

    def get_dsz(self):
        return self.outsz

    def get_vsz(self):
        return self.vsz

    def forward(self, xch):
        _0, _1, W = xch.shape
        char_vecs = self.embeddings(xch.view(-1, W))
        mots = self.char_comp(char_vecs)
        gated = self.gating_seq(mots)
        if self.projsz:
            gated = self.proj(gated)
        return gated.view(_0, _1, self.get_dsz())


class CharLSTMEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed = LookupTableEmbeddings(**kwargs)
        self.lstmsz = kwargs.get('lstmsz', 50)
        layers = kwargs.get('layers', 1)
        pdrop = kwargs.get('pdrop', 0.5)
        unif = kwargs.get('unif', 0)
        weight_init = kwargs.get('weight_init', 'uniform')
        self.char_comp = BiLSTMEncoderHidden(self.embed.output_dim, self.lstmsz, layers, pdrop, unif=unif, initializer=weight_init)

    def forward(self, xch):
        B, T, W = xch.shape
        flat_chars = xch.view(-1, W)
        char_embeds = self.embed(flat_chars)
        lengths = torch.sum(flat_chars != Offsets.PAD, dim=1)
        sorted_word_lengths, perm_idx = lengths.sort(0, descending=True)
        sorted_feats = char_embeds[perm_idx].transpose(0, 1).contiguous()
        patched_lengths = sorted_word_lengths.masked_fill(sorted_word_lengths == 0, 1)
        hidden = self.char_comp((sorted_feats, patched_lengths))
        hidden = hidden.masked_fill((sorted_word_lengths == 0).unsqueeze(-1), 0)
        results = unsort_batch(hidden, perm_idx)
        return results.reshape((B, T, -1))

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.embed.get_vsz()


class CharTransformerEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed = LookupTableEmbeddings(**kwargs)
        self.d_model = kwargs.get('wsz', 30)
        self.num_heads = kwargs.get('num_heads', 3)
        self.rpr_k = kwargs.get('rpr_k', 10)
        layers = kwargs.get('layers', 1)
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = TransformerEncoderStackWithLengths(self.num_heads, self.d_model, pdrop, False, layers, rpr_k=self.rpr_k, input_sz=self.embed.output_dim)

    def forward(self, xch):
        B, T, W = xch.shape
        flat_chars = xch.view(-1, W)
        char_embeds = self.embed(flat_chars)
        lengths = torch.sum(flat_chars != Offsets.PAD, dim=1)
        results = self.char_comp((char_embeds, lengths))
        pooled = torch.max(results, -2, keepdims=False)[0]
        return pooled.reshape((B, T, -1))

    def get_dsz(self):
        return self.d_model

    def get_vsz(self):
        return self.embed.get_vsz()


class PositionalMixin(nn.Module):
    """A Mixin that provides functionality to generate positional embeddings to be added to the normal embeddings.

    Note, mixins need to be before the base case when used, i.e.
        `Embedding(Mixin, BaseEmbed)` NOT `Embedding(BaseEmbed, Mixin)`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def positional(self, length):
        pass

    def extra_repr(self):
        return f'mxlen={self.mxlen}'


class SinusoidalPositionalMixin(PositionalMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mxlen = kwargs.get('mxlen', 1000)
        max_timescale = kwargs.get('max_timescale', 10000.0)
        word_dsz = self.get_dsz()
        log_timescale_increment = math.log(max_timescale) / word_dsz
        inv_timescales = torch.exp(torch.arange(0, word_dsz, 2).float() * -log_timescale_increment)
        pe = torch.zeros(self.mxlen, word_dsz)
        position = torch.arange(0, self.mxlen).float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * inv_timescales)
        pe[:, 1::2] = torch.cos(position * inv_timescales)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def positional(self, length):
        return self.pe[:, :length]


class LearnedPositionalMixin(PositionalMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())

    def positional(self, length):
        return self.pos_embeddings(torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)).unsqueeze(0)


class BERTLookupTableEmbeddings(LookupTableEmbeddings):
    """
    BERT style embeddings with a 0 token type

    TODO: Get rid of this, we dont need it anymore
    If you want to use BERT with token types, make a `LearnedPositionalLookupTableEmbeddings` feature
    and a `LookupTableEmbeddings` feature (for the token type)
    and put them in an `EmbeddingsStack` with an embeddings_reduction='sum-layer-norm' on the model

    Otherwise, if you do not plan on setting the token type, use the `LearnedPositionalLookupTableEmbeddingsWithBias`,
    which will add the BERT token_type=0 weights into the pos + word_embed and is more efficient
    than this class, since it doesnt do any memory allocation on the fly
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.tok_type_vsz = kwargs['tok_type_vsz']
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())
        self.tok_embeddings = nn.Embedding(self.tok_type_vsz, self.get_dsz())
        self.ln = nn.LayerNorm(self.get_dsz(), eps=1e-12)

    def forward(self, x):
        zeros = torch.zeros_like(x)
        x = super().forward(x)
        x = x + self.positional(x.size(1)) + self.tok_embeddings(zeros)
        x = self.ln(x)
        return self.dropout(x)

    def positional(self, length):
        return self.pos_embeddings(torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)).unsqueeze(0)


class LearnedPositionalLookupTableEmbeddingsWithBias(LookupTableEmbeddings):
    """Learned positional lookup table embeddings wih a bias and layer norm

    This is just a typical learned positional embedding but with a learnable
    bias and a layer norm.  This is equivalent to BERT embeddings when the
    token_type is not set.

    If you are using BERT but you have no interest in using token type embeddings
    (IOW if you are setting all the values of that feature zero anyhow), using this
    object is faster and simpler than having a separate vectorizer for token type.

    If you have a need for token type embeddings, you will want to create 2 sets of embeddings,
    one that acts on the tokens, of type `LearnedPositionalLookupTableEmbeddings` and one of the type
    `LookupTableEmbeddings` for the token type feature

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())
        self.bias = nn.Parameter(torch.zeros(self.get_dsz()))

    def forward(self, x):
        x = super().forward(x)
        x = x + self.positional(x.size(1)) + self.bias
        return x

    def positional(self, length):
        return self.pos_embeddings(torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)).unsqueeze(0)


class PositionalLookupTableEmbeddings(SinusoidalPositionalMixin, LookupTableEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, x):
        """Add a positional encoding to the embedding, followed by dropout

        :param x: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        x = super().forward(x) * self.scale
        x = x + self.positional(x.size(1))
        return self.dropout(x)


class LearnedPositionalLookupTableEmbeddings(LearnedPositionalMixin, LookupTableEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))

    def forward(self, x):
        T = x.size(1)
        x = super().forward(x)
        pos = self.positional(T)
        return self.dropout(x + pos)


class PositionalCharConvEmbeddings(SinusoidalPositionalMixin, CharConvEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, xch):
        """Add a positional encoding to the embedding, followed by dropout

        :param xch: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        xch = super().forward(xch) * self.scale
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class LearnedPositionalCharConvEmbeddings(LearnedPositionalMixin, CharConvEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))

    def forward(self, xch):
        """Add a positional encoding to the embedding, followed by dropout

        :param xch: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        xch = super().forward(xch)
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class PositionalCharLSTMEmbeddings(SinusoidalPositionalMixin, CharLSTMEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, xch):
        xch = super().forward(xch) * self.scale
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class LearnedPositionalCharLSTMEmbeddings(LearnedPositionalMixin, CharLSTMEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))

    def forward(self, xch):
        xch = super().forward(xch)
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class VariationalDropout(nn.Module):
    """Inverted dropout that applies the same mask at each time step."""

    def __init__(self, pdrop: float=0.5, batch_first: bool=False):
        """Variational Dropout

        :param pdrop: the percentage to drop
        """
        super().__init__()
        self.pdrop = pdrop
        self.batch_first = batch_first

    def extra_repr(self):
        return 'p=%.1f' % self.pdrop

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not self.training:
            return input
        if self.batch_first:
            dim0 = input.size(0)
            dim1 = 1
        else:
            dim0 = 1
            dim1 = input.size(1)
        mask = torch.zeros(dim0, dim1, input.size(2)).bernoulli_(1 - self.pdrop)
        mask = mask / self.pdrop
        return mask * input


class SequenceLoss(nn.Module):
    """Computes the loss over a sequence"""

    def __init__(self, LossFn: nn.Module=nn.NLLLoss, avg: str='token'):
        """A class that applies a Loss function to sequence via the folding trick.

        :param LossFn: A loss function to apply (defaults to `nn.NLLLoss`)
        :param avg: A divisor to apply, valid values are `token` and `batch`
        """
        super().__init__()
        self.avg = avg
        if avg == 'token':
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='mean')
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='sum')
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) ->torch.Tensor:
        """Evaluate some loss over a sequence.
        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.
        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)

    def extra_repr(self):
        return f'reduction={self.avg}'


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Conv1DSame(nn.Module):
    """Perform a 1D convolution with output size same as input size

    To make this operation work as expected, we cannot just use `padding=kernel_size//2` inside
    of the convolution operation.  Instead, we zeropad the input using the `ConstantPad1d` module

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool=True):
        """Create a 1D conv to produce the same output size as input

        :param in_channels: The number of input feature maps
        :param out_channels: The number of output feature maps
        :param kernel_size: The kernel size
        :param bias: Is bias on?
        """
        super().__init__()
        start_pad = kernel_size // 2
        end_pad = start_pad - 1 if kernel_size % 2 == 0 else start_pad
        self.conv = nn.Sequential(nn.ConstantPad1d((start_pad, end_pad), 0.0), nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Do convolution1d on an input tensor, `[B, C, T]`

        :param x: The input tensor of shape `[B, C, T]`
        :return: The output tensor of shape `[B, H, T]`
        """
        return self.conv(x)


def bth2bht(t):
    return tf.transpose(t, [0, 2, 1])


class BTH2BHT(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[B, H, T]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bth2bht(t)


def tbh2bht(t):
    return tf.tranpose(t, [0, 2, 1])


class TBH2BHT(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, H, T]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return tbh2bht(t)


def tbh2bth(t):
    return tf.transpose(t, [1, 0, 2])


class TBH2BTH(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, T, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return tbh2bth(t)


class BTH2TBH(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[T, B, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bth2tbh(t)


def bht2bth(t: torch.Tensor) ->torch.Tensor:
    return t.transpose(1, 2).contiguous()


class BHT2BTH(nn.Module):
    """Utility layer to convert from `[B, H, T]` to `[B, T, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bht2bth(t)


class LSTMEncoderSequenceHiddenContext(LSTMEncoderBase):

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(hidden)


class BiLSTMEncoderSequenceHiddenContext(BiLSTMEncoderBase):

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(concat_state_dirs(hidden))


class BiLSTMEncoderHiddenContext(BiLSTMEncoderBase):

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->torch.Tensor:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.extract_top_state(concat_state_dirs(hidden))


class GRUEncoderBase(nn.Module):
    """The GRU encoder is a base for a set of encoders producing various outputs.

    All GRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `GRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `GRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0, requires_length: bool=True, batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
            nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) ->torch.Tensor:
        return state[-1]


class GRUEncoder(tf.keras.layers.Layer):
    """GRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def __init__(self, insz: Optional[int], hsz: int, nlayers: int=1, pdrop: float=0.0, variational: bool=False, requires_length: bool=True, name: Optional[str]=None, dropout_in_single_layer: bool=False, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.
        :param insz: An optional input size for parity with other layer backends.  Can pass `None`
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: variational recurrence is on, defaults to `False`
        :param requires_length: Does the input require an input length (defaults to ``True``)
        :param name: TF only! Put a name in the graph for this layer. Optional, defaults to `None`
        :param dropout_in_single_layer: TF only! If there is a single layer, should we do dropout, defaults to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            self.rnns.append(tf.keras.layers.GRU(hsz, return_sequences=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        self.rnns.append(tf.keras.layers.GRU(hsz, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0))

    def output_fn(self, output, state):
        return output, state

    def call(self, inputs):
        """RNNs over input sequence of `[B, T, C]` and lengths `[B]`, output `[B, S, H]` where `S = max(lengths)`

        :param inputs: A tuple of `(sequence, lengths)`, `sequence` shape `[B, T, C]`, lengths shape = `[B]`
        :return: Output depends on the subclass handling
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs
        rnnout, h = outputs
        return self.output_fn(rnnout, h)

    @property
    def requires_length(self) ->bool:
        return self._requires_length


class GRUEncoderSequence(GRUEncoder):
    """GRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def output_fn(self, output, state):
        """Return sequence `(BxTxC)`

        :param output: The sequence
        :param state: The hidden state
        :return: The sequence `(BxTxC)`
        """
        return output


class GRUEncoderAll(tf.keras.layers.Layer):
    """GRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`
    """

    def __init__(self, insz: Optional[int], hsz: int, nlayers: int=1, pdrop: float=0.0, variational: bool=False, requires_length: bool=True, name: Optional[str]=None, dropout_in_single_layer=False, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(hsz, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
            self.rnns.append(rnn)
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(hsz, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
        self.rnns.append(rnn)

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and a hidden vector `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        hs = []
        for rnn in self.rnns:
            outputs, h = rnn(inputs, mask=mask)
            hs.append(h)
            inputs = outputs
        h = tf.stack(hs)
        return self.output_fn(outputs, h)

    @property
    def requires_length(self):
        return self._requires_length


class GRUEncoderHidden(GRUEncoder):
    """BiGRU encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, output, state):
        """Return last hidden state `h`

        :param output: The sequence
        :param state: The hidden state
        :return: The last hidden state `(h, c)`
        """
        return state


class BiGRUEncoderBase(nn.Module):
    """BiGRU encoder base for a set of encoders producing various outputs.

    All BiGRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiGRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiGRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0, requires_length: bool=True, batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiGRU (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiGRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz // 2, nlayers, dropout=pdrop, bidirectional=True, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) ->torch.Tensor:
        return state[-1]


class BiGRUEncoderSequenceHiddenContext(BiGRUEncoderBase):

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        tbc, lengths = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, self.extract_top_state(_cat_dir(hidden))


class BiGRUEncoderAll(tf.keras.layers.Layer):
    """BiGRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`
    """

    def __init__(self, insz: Optional[int], hsz: int, nlayers: int=1, pdrop: float=0.0, variational: bool=False, requires_length: bool=True, name: Optional[str]=None, dropout_in_single_layer: bool=False, **kwargs):
        """Produce a stack of BiGRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiGRU (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiGRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(hsz // 2, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(hsz // 2, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode='concat'))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and a hidden vector `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        hs = []
        for rnn in self.rnns:
            outputs, h1, h2 = rnn(inputs, mask=mask)
            h = tf.stack([h1, h2])
            hs.append(h)
            inputs = outputs
        _, B, H = get_shape_as_list(h)
        h = tf.reshape(tf.stack(hs), [-1, B, H * 2])
        return self.output_fn(outputs, h)

    @property
    def requires_length(self) ->bool:
        return self._requires_length


class BiGRUEncoder(tf.keras.layers.Layer):
    """BiGRU encoder base for a set of encoders producing various outputs.

    All BiGRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.
    """

    def __init__(self, insz: Optional[int], hsz: int, nlayers: int, pdrop: float=0.0, variational: bool=False, requires_length: bool=True, name: Optional[str]=None, dropout_in_single_layer: bool=False, **kwargs):
        """Produce a stack of BiGRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(hsz // 2, return_sequences=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(hsz // 2, return_sequences=True, return_state=True, recurrent_dropout=pdrop if variational else 0.0, dropout=pdrop if not variational else 0.0)
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode='concat'))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs
        rnnout, h_fwd, h_bwd = outputs
        return self.output_fn(rnnout, (h_fwd, h_bwd))

    @property
    def requires_length(self):
        return self._requires_length


class BiGRUEncoderSequence(BiGRUEncoder):
    """BiGRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def output_fn(self, rnnout, state):
        return rnnout


class BiGRUEncoderHidden(BiGRUEncoder):
    """BiGRU encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, _, state):
        return tf.concat([state[0], state[1]], axis=-1)


class FineTuneModel(tf.keras.Model):

    def __init__(self, nc: int, embeddings: tf.keras.layers.Layer, stack_model: Optional[tf.keras.layers.Layer]=None):
        super().__init__()
        self.finetuned = embeddings
        self.stack_model = stack_model
        self.output_layer = tf.keras.layers.Dense(nc)

    def call(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class EmbedPoolStackModel(tf.keras.Model):

    def __init__(self, nc: int, embeddings: tf.keras.layers.Layer, pool_model: tf.keras.layers.Layer, stack_model: Optional[tf.keras.layers.Layer]=None, output_model: Optional[tf.keras.layers.Layer]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.embed_model = embeddings
        self.pool_model = pool_model
        self.stack_model = stack_model
        self.output_layer = tf.keras.layers.Dense(nc) if output_model is None else output_model

    def call(self, inputs):
        lengths = inputs.get('lengths')
        embedded = self.embed_model(inputs)
        embedded = embedded, lengths
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled) if self.stack_model is not None else pooled
        return self.output_layer(stacked)


@torch.jit.script
def script_viterbi(unary: torch.Tensor, trans: torch.Tensor, start_idx: int, end_idx: int) ->Tuple[torch.Tensor, torch.Tensor]:
    seq_len: int = unary.size(0)
    num_tags: int = unary.size(1)
    fill_value: float = -10000.0
    alphas = torch.full((num_tags,), fill_value, dtype=unary.dtype, device=unary.device)
    broadcast_idx = torch.full((num_tags,), start_idx, dtype=torch.long)
    alphas.scatter_(0, broadcast_idx, torch.zeros((num_tags,)))
    alphas.unsqueeze_(0)
    backpointers: torch.Tensor = torch.zeros(num_tags, dtype=torch.long).unsqueeze(0)
    for i in range(seq_len):
        unary_t = unary[(i), :]
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 1)
        backpointers = torch.cat([backpointers, best_tag_ids.unsqueeze(0)], 0)
        alphas = (viterbi + unary_t).unsqueeze(0)
    terminal_vars = alphas.squeeze(0) + trans[(end_idx), :]
    path_score, best_tag_id = torch.max(terminal_vars, 0)
    best_path = best_tag_id.unsqueeze(0)
    for i in range(unary.size(0)):
        t = seq_len - i - 1
        best_tag_id = backpointers[t + 1, best_tag_id]
        best_path = torch.cat([best_path, best_tag_id.unsqueeze(0)], -1)
    new_path_vec = best_path.flip(0)
    return new_path_vec[1:], path_score


class ViterbiBatchSize1(nn.Module):

    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, _: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        unary = unary.squeeze(1)
        trans = trans.squeeze(0)
        path, score = script_viterbi(unary, trans, self.start_idx, self.end_idx)
        return path.unsqueeze(1), score


class Viterbi(nn.Module):

    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []
        alphas = torch.full((batch_size, 1, tag_size), -10000.0, device=unary.device)
        alphas[:, (0), (self.start_idx)] = 0
        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas
        terminal_var = alphas.squeeze(1) + trans[:, (self.end_idx), :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        rev_len = seq_len - lengths - 1
        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        seq_mask = sequence_mask(lengths, seq_len).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


class ViterbiLogSoftmaxNorm(Viterbi):

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []
        alphas = torch.full((batch_size, 1, tag_size), -10000.0, device=unary.device)
        alphas[:, (0), (self.start_idx)] = 0
        alphas = F.log_softmax(alphas, dim=-1)
        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas
        terminal_var = alphas.squeeze(1) + trans[:, (self.end_idx), :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        rev_len = seq_len - lengths - 1
        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        seq_mask = sequence_mask(lengths).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


class SequenceModel(nn.Module):

    def __init__(self, nc: int, embeddings: nn.Module, transducer: nn.Module, decoder: Optional[nn.Module]=None):
        super().__init__()
        self.embed_model = embeddings
        self.transducer_model = transducer
        if transducer.output_dim != nc:
            self.proj_layer = Dense(transducer.output_dim, nc)
        else:
            self.proj_layer = nn.Identity()
        self.decoder_model = decoder

    def transduce(self, inputs: Dict[str, torch.Tensor]) ->torch.Tensor:
        lengths = inputs['lengths']
        embedded = self.embed_model(inputs)
        embedded = embedded, lengths
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced: torch.Tensor, lengths: torch.Tensor) ->torch.Tensor:
        return self.decoder_model((transduced, lengths))

    def forward(self, inputs: Dict[str, torch.Tensor]) ->torch.Tensor:
        pass


class TimeDistributedProjection(tf.keras.layers.Layer):

    def __init__(self, num_outputs, name=None):
        """Set up a low-order projection (embedding) by flattening the batch and time dims and matmul

        TODO: Avoid where possible, Dense should work in most cases

        :param name: The name for this scope
        :param num_outputs: The number of feature maps out
        """
        super().__init__(True, name)
        self.output_dim = num_outputs
        self.W = None
        self.b = None

    def build(self, input_shape):
        nx = int(input_shape[-1])
        self.W = self.add_weight('W', [nx, self.output_dim])
        self.b = self.add_weight('b', [self.output_dim], initializer=tf.constant_initializer(0.0))
        super().build(input_shape)

    def call(self, inputs):
        """Low-order projection (embedding) by flattening the batch and time dims and matmul

        :param inputs: The input tensor
        :return: An output tensor having the same dims as the input, except the last which is `output_dim`
        """
        input_shape = get_shape_as_list(inputs)
        collapse = tf.reshape(inputs, [-1, input_shape[-1]])
        c = tf.matmul(collapse, self.W) + self.b
        c = tf.reshape(c, input_shape[:-1] + [self.output_dim])
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    @property
    def requires_length(self) ->bool:
        return False


class TagSequenceModel(tf.keras.Model):

    def __init__(self, nc: int, embeddings: tf.keras.layers.Layer, transducer: tf.keras.layers.Layer, decoder: Optional[tf.keras.layers.Layer]=None, name: str=None):
        super().__init__(name=name)
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings
        self.path_scores = None
        self.transducer_model = transducer
        self.proj_layer = TimeDistributedProjection(nc)
        decoder_model = CRF(nc) if decoder is None else decoder
        self.decoder_model = decoder_model

    def transduce(self, inputs):
        lengths = inputs.get('lengths')
        embedded = self.embed_model(inputs)
        embedded = embedded, lengths
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced, lengths):
        path, self.path_scores = self.decoder_model((transduced, lengths))
        return path

    def call(self, inputs, training=None):
        transduced = self.transduce(inputs)
        return self.decode(transduced, inputs.get('lengths'))

    def neg_log_loss(self, unary, tags, lengths):
        return self.decoder_model.neg_log_loss(unary, tags, lengths)


class LangSequenceModel(tf.keras.Model):

    def __init__(self, nc: int, embeddings: tf.keras.layers.Layer, transducer: tf.keras.layers.Layer, decoder: Optional[tf.keras.layers.Layer]=None, name: str=None):
        super().__init__(name=name)
        self.embed_model = embeddings
        self.transducer_model = transducer
        if hasattr(transducer, 'requires_state') and transducer.requires_state:
            self._call = self._call_with_state
            self.requires_state = True
        else:
            self._call = self._call_without_state
            self.requires_state = False
        self.output_layer = TimeDistributedProjection(nc)
        self.decoder_model = decoder

    def call(self, inputs):
        return self._call(inputs)

    def _call_with_state(self, inputs):
        h = inputs.get('h')
        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        transduced = self.output_layer(transduced)
        return transduced, hidden

    def _call_without_state(self, inputs):
        embedded = self.embed_model(inputs)
        transduced = self.transducer_model((embedded, None))
        transduced = self.output_layer(transduced)
        return transduced, None


class SeqBahdanauAttention(SequenceSequenceAttention):

    def __init__(self, hsz: int, pdrop: float=0.1, **kwargs):
        super().__init__(hsz, pdrop=pdrop, **kwargs)
        self.V = pytorch_linear(self.hsz, 1, bias=False)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        additive = query.unsqueeze(-2) + key.unsqueeze(-3)
        non_linear = torch.tanh(additive)
        scores = self.V(non_linear)
        scores = scores.squeeze(-1)
        return F.softmax(scores, dim=-1)


class TransformerEncoderStackWithTimeMask(TransformerEncoderStack):

    def __init__(self, num_heads: int, d_model: int, pdrop: bool, scale: bool=True, layers: int=1, activation: str='relu', d_ff: Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[Union[int, List[int]]]=None, layer_norms_after: bool=False, layer_norm_eps: float=1e-06, name: Optional[str]=None, **kwargs):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k, layer_norms_after, layer_norm_eps, name=name)
        self.proj = WithDropout(tf.keras.layers.Dense(d_model), pdrop)

    def call(self, inputs):
        x, _ = inputs
        x = self.proj(x)
        max_seqlen = get_shape_as_list(x)[1]
        mask = subsequent_mask(max_seqlen)
        return super().call((x, mask))


class TiedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.tgt_embeddings = nn.Embedding(100, 10)
        self.preds = pytorch_linear(10, 100)
        self.preds.weight = self.tgt_embeddings.weight

    def forward(self, input_vec):
        return self.preds(self.tgt_embeddings(input_vec))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ArcPolicy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BHT2BTH,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1DSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPool1D,
     lambda: ([], {'outsz': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TiedWeights,
     lambda: ([], {}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (VariationalDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dpressel_mead_baseline(_paritybench_base):
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

