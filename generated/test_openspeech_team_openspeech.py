import sys
_module = sys.modules[__name__]
del sys
conf = _module
openspeech = _module
callbacks = _module
criterion = _module
cross_entropy = _module
configuration = _module
cross_entropy = _module
ctc = _module
ctc = _module
joint_ctc_cross_entropy = _module
joint_ctc_cross_entropy = _module
label_smoothed_cross_entropy = _module
label_smoothed_cross_entropy = _module
perplexity = _module
perplexity = _module
transducer = _module
transducer = _module
data = _module
audio = _module
augment = _module
data_loader = _module
dataset = _module
filter_bank = _module
filter_bank = _module
load = _module
melspectrogram = _module
mfcc = _module
spectrogram = _module
spectrogram = _module
sampler = _module
data_loader = _module
dataset = _module
dataclass = _module
configurations = _module
initialize = _module
datasets = _module
aishell = _module
lit_data_module = _module
preprocess = _module
ksponspeech = _module
character = _module
grapheme = _module
subword = _module
language_model = _module
librispeech = _module
decoders = _module
lstm_attention_decoder = _module
openspeech_decoder = _module
rnn_transducer_decoder = _module
transformer_decoder = _module
transformer_transducer_decoder = _module
encoders = _module
conformer_encoder = _module
contextnet_encoder = _module
convolutional_lstm_encoder = _module
convolutional_transformer_encoder = _module
deepspeech2 = _module
jasper = _module
lstm_encoder = _module
openspeech_encoder = _module
quartznet = _module
rnn_transducer_encoder = _module
squeezeformer_encoder = _module
transformer_encoder = _module
transformer_transducer_encoder = _module
lm = _module
lstm_lm = _module
openspeech_lm = _module
transformer_lm = _module
metrics = _module
models = _module
conformer = _module
model = _module
contextnet = _module
model = _module
model = _module
model = _module
listen_attend_spell = _module
model = _module
model = _module
openspeech_ctc_model = _module
openspeech_encoder_decoder_model = _module
openspeech_language_model = _module
openspeech_model = _module
openspeech_transducer_model = _module
model = _module
rnn_transducer = _module
model = _module
squeezeformer = _module
model = _module
transformer = _module
model = _module
model = _module
transformer_transducer = _module
model = _module
modules = _module
add_normalization = _module
additive_attention = _module
batchnorm_relu_rnn = _module
conformer_attention_module = _module
conformer_block = _module
conformer_convolution_module = _module
conformer_feed_forward_module = _module
contextnet_block = _module
contextnet_module = _module
conv2d_extractor = _module
conv2d_subsampling = _module
conv_base = _module
conv_group_shuffle = _module
deepspeech2_extractor = _module
depthwise_conv1d = _module
depthwise_conv2d = _module
dot_product_attention = _module
glu = _module
jasper_block = _module
jasper_subblock = _module
location_aware_attention = _module
mask = _module
mask_conv1d = _module
mask_conv2d = _module
multi_head_attention = _module
pointwise_conv1d = _module
positional_encoding = _module
positionwise_feed_forward = _module
quartznet_block = _module
quartznet_subblock = _module
relative_multi_head_attention = _module
residual_connection_module = _module
squeezeformer_attention_module = _module
squeezeformer_block = _module
squeezeformer_module = _module
swish = _module
time_channel_separable_conv1d = _module
transformer_embedding = _module
vgg_extractor = _module
wrapper = _module
optim = _module
adamp = _module
novograd = _module
optimizer = _module
radam = _module
scheduler = _module
lr_scheduler = _module
reduce_lr_on_plateau_scheduler = _module
transformer_lr_scheduler = _module
tri_stage_lr_scheduler = _module
warmup_reduce_lr_on_plateau_scheduler = _module
warmup_scheduler = _module
search = _module
beam_search_base = _module
beam_search_ctc = _module
beam_search_lstm = _module
beam_search_rnn_transducer = _module
beam_search_transformer = _module
beam_search_transformer_transducer = _module
ensemble_search = _module
tokenizers = _module
tokenizer = _module
utils = _module
generate_openspeech_configs = _module
hydra_ensemble_eval = _module
hydra_eval = _module
hydra_lm_train = _module
hydra_train = _module
setup = _module
test_conformer = _module
test_conformer_lstm = _module
test_conformer_transducer = _module
test_contextnet = _module
test_contextnet_lstm = _module
test_contextnet_transducer = _module
test_conv2d_subsampling = _module
test_deep_cnn_with_joint_ctc_listen_attend_spell = _module
test_deepspeech2 = _module
test_jasper10x5 = _module
test_jasper5x3 = _module
test_joint_ctc_conformer_lstm = _module
test_joint_ctc_listen_attend_spell = _module
test_joint_ctc_transformer = _module
test_listen_attend_spell = _module
test_listen_attend_spell_with_location_aware = _module
test_listen_attend_spell_with_multi_head = _module
test_lstm_for_causal_lm = _module
test_quartznet10x5 = _module
test_quartznet15x5 = _module
test_quartznet5x5 = _module
test_rnn_transducer = _module
test_squeezeformer = _module
test_squeezeformer_lstm = _module
test_squeezeformer_transducer = _module
test_transformer = _module
test_transformer_transducer = _module
test_transformer_with_ctc = _module
test_vgg_transformer = _module
test_audio_augment = _module
test_lstm_lm = _module
test_transformer_for_causal_lm = _module
test_transformer_lm = _module
test_lr_scheduler = _module
test_warprnnt_loss = _module

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


import torch.nn as nn


from torch import Tensor


from typing import Tuple


import torch


import torch.nn.functional as F


import logging


import random


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torch.utils.data import Dataset


from typing import Any


from typing import Optional


from collections import OrderedDict


from typing import Dict


from torch.optim import ASGD


from torch.optim import SGD


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adam


from torch.optim import Adamax


from torch.optim import AdamW


import warnings


import math


from typing import Union


import torch.nn.init as init


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim import Optimizer


from typing import Iterable


import matplotlib.pyplot as plt


from torch import optim


class Tokenizer(object):
    """
    A tokenizer is in charge of preparing the inputs for a model.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def decode(self, labels):
        raise NotImplementedError

    def encode(self, labels):
        raise NotImplementedError

    def __call__(self, sentence):
        return self.encode(sentence)


CRITERION_DATACLASS_REGISTRY = dict()


CRITERION_REGISTRY = dict()


def register_criterion(name: str, dataclass=None):
    """
    New criterion types can be added to OpenSpeech with the :func:`register_criterion` function decorator.

    For example::
        @register_criterion('label_smoothed_cross_entropy')
        class LabelSmoothedCrossEntropyLoss(nn.Module):
            (...)

    .. note:: All criterion must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the criterion
        dataclass (Optional, str): the dataclass of the criterion (default: None)
    """

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f'Cannot register duplicate criterion ({name})')
        CRITERION_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in CRITERION_DATACLASS_REGISTRY:
                raise ValueError(f'Cannot register duplicate criterion ({name})')
            CRITERION_DATACLASS_REGISTRY[name] = dataclass
        return cls
    return register_criterion_cls


WARPRNNT_IMPORT_ERROR = """
Openspeech requires the warp-rnnt library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/1ytic/warp-rnnt and follow the ones that match your environment.
"""


class OpenspeechDecoder(nn.Module):
    """Interface of OpenSpeech decoder."""

    def __init__(self):
        super(OpenspeechDecoder, self).__init__()

    def count_parameters(self) ->int:
        """Count parameters of decoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) ->None:
        """Update dropout probability of decoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True) ->None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) ->Tensor:
        return self.linear(x)


class RNNTransducerDecoder(OpenspeechDecoder):
    """
    Decoder of RNN-Transducer

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoders (default: 512)
        output_dim (int, optional): output dimension of encoders and decoders (default: 512)
        num_layers (int, optional): number of decoders layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout_p (float, optional): dropout probability of decoders

    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoders. `FloatTensor` of size ``(batch, seq_length, dimension)``

    Returns:
        (Tensor, Tensor):

        * decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Reference:
        A Graves: Sequence Transduction with Recurrent Neural Networks
        https://arxiv.org/abs/1211.3711.pdf
    """
    supported_rnns = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, num_classes: int, hidden_state_dim: int, output_dim: int, num_layers: int, rnn_type: str='lstm', pad_id: int=0, sos_id: int=1, eos_id: int=2, dropout_p: float=0.2):
        super(RNNTransducerDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.pad_id = pad_id,
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(input_size=hidden_state_dim, hidden_size=hidden_state_dim, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout_p, bidirectional=False)
        self.out_proj = Linear(hidden_state_dim, output_dim)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor=None, hidden_states: torch.Tensor=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propage a `inputs` (targets) for training.

        Inputs:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded `LongTensor` of size ``(batch, target_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): Previous hidden states.

        Returns:
            (Tensor, Tensor):

            * outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.embedding(inputs)
        if hidden_states is not None:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
        else:
            outputs, hidden_states = self.rnn(embedded)
        outputs = self.out_proj(outputs)
        return outputs, hidden_states


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int, scale: bool=True) ->None:
        super(DotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        if len(query.size()) == 3:
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        else:
            score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask, -10000.0)
        attn = F.softmax(score, -1)
        if len(query.size()) == 3:
            context = torch.bmm(attn, value)
        else:
            context = torch.matmul(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        dim (int): The dimension of model (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int=512, num_heads: int=8) ->None:
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, 'hidden_dim % num_heads should be zero.'
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)
        return context, attn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Networks proposed in "Attention Is All You Need".
    Fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Another way of describing this is as two convolutions with kernel size 1.
    """

    def __init__(self, d_model: int=512, d_ff: int=2048, dropout_p: float=0.3) ->None:
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(Linear(d_model, d_ff), nn.Dropout(dropout_p), nn.ReLU(), Linear(d_ff, d_model), nn.Dropout(dropout_p))

    def forward(self, inputs: Tensor) ->Tensor:
        return self.feed_forward(inputs)


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoders layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer decoder layer
        encoder_outputs (torch.FloatTensor): outputs of encoder
        self_attn_mask (torch.BoolTensor): mask of self attention
        encoder_output_mask (torch.BoolTensor): mask of encoder outputs

    Returns:
        (Tensor, Tensor, Tensor)

        * outputs (torch.FloatTensor): output of transformer decoder layer
        * self_attn (torch.FloatTensor): output of self attention
        * encoder_attn (torch.FloatTensor): output of encoder attention

    Reference:
        Ashish Vaswani et al.: Attention Is All You Need
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model: int=512, num_heads: int=8, d_ff: int=2048, dropout_p: float=0.3) ->None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.decoder_attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, encoder_outputs: Tensor, self_attn_mask: Optional[Tensor]=None, encoder_attn_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate transformer decoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer decoder layer
            encoder_outputs (torch.FloatTensor): outputs of encoder
            self_attn_mask (torch.BoolTensor): mask of self attention
            encoder_output_mask (torch.BoolTensor): mask of encoder outputs

        Returns:
            outputs (torch.FloatTensor): output of transformer decoder layer
            self_attn (torch.FloatTensor): output of self attention
            encoder_attn (torch.FloatTensor): output of encoder attention
        """
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual
        residual = outputs
        outputs = self.decoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.decoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_attn_mask)
        outputs += residual
        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual
        return outputs, self_attn, encoder_attn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int=512, max_len: int=5000) ->None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) ->Tensor:
        return self.pe[:, :length]


class TransformerEmbedding(nn.Module):
    """
    Embedding layer. Similarly to other sequence transduction models, transformer use learned embeddings
    to convert the input tokens and output tokens to vectors of dimension d_model.
    In the embedding layers, transformer multiply those weights by sqrt(d_model)

    Args:
        num_embeddings (int): the number of embedding size
        pad_id (int): identification of pad token
        d_model (int): dimension of model

    Inputs:
        inputs (torch.FloatTensor): input of embedding layer

    Returns:
        outputs (torch.FloatTensor): output of embedding layer
    """

    def __init__(self, num_embeddings: int, pad_id: int, d_model: int=512) ->None:
        super(TransformerEmbedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor) ->Tensor:
        """
        Forward propagate of embedding layer.

        Inputs:
            inputs (torch.FloatTensor): input of embedding layer

        Returns:
            outputs (torch.FloatTensor): output of embedding layer
        """
        return self.embedding(inputs) * self.sqrt_dim


def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """mask position is set to 1"""

    def get_transformer_non_pad_mask(inputs: Tensor, input_lengths: Tensor) ->Tensor:
        """Padding position is set to 0, either use input_lengths or pad_id"""
        batch_size = inputs.size(0)
        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])
        else:
            raise ValueError(f'Unsupported input shape {inputs.size()}')
        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0
        return non_pad_mask
    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask
    return subsequent_mask


class TransformerDecoder(OpenspeechDecoder):
    """
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of layers
        num_heads: number of attention heads
        dropout_p: probability of dropout
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        max_length (int): max decoding length
    """

    def __init__(self, num_classes: int, d_model: int=512, d_ff: int=512, num_layers: int=6, num_heads: int=8, dropout_p: float=0.3, pad_id: int=0, sos_id: int=1, eos_id: int=2, max_length: int=128) ->None:
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = TransformerEmbedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout_p=dropout_p) for _ in range(num_layers)])
        self.fc = nn.Sequential(nn.LayerNorm(d_model), Linear(d_model, d_model, bias=False), nn.Tanh(), Linear(d_model, num_classes, bias=False))

    def forward_step(self, decoder_inputs: torch.Tensor, decoder_input_lengths: torch.Tensor, encoder_outputs: torch.Tensor, encoder_output_lengths: torch.Tensor, positional_encoding_length: int) ->torch.Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequent_mask, 0)
        encoder_attn_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, decoder_inputs.size(1))
        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)
        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(inputs=outputs, encoder_outputs=encoder_outputs, self_attn_mask=self_attn_mask, encoder_attn_mask=encoder_attn_mask)
        return outputs

    def forward(self, encoder_outputs: torch.Tensor, targets: Optional[torch.LongTensor]=None, encoder_output_lengths: torch.Tensor=None, target_lengths: torch.Tensor=None, teacher_forcing_ratio: float=1.0) ->torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size
                ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths (torch.LongTensor): The length of encoders outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if targets is not None and use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            target_length = targets.size(1)
            step_outputs = self.forward_step(decoder_inputs=targets, decoder_input_lengths=target_lengths, encoder_outputs=encoder_outputs, encoder_output_lengths=encoder_output_lengths, positional_encoding_length=target_length)
            step_outputs = self.fc(step_outputs).log_softmax(dim=-1)
            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                logits.append(step_output)
        else:
            input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            input_var = input_var.fill_(self.pad_id)
            input_var[:, 0] = self.sos_id
            for di in range(1, self.max_length):
                input_lengths = torch.IntTensor(batch_size).fill_(di)
                outputs = self.forward_step(decoder_inputs=input_var[:, :di], decoder_input_lengths=input_lengths, encoder_outputs=encoder_outputs, encoder_output_lengths=encoder_output_lengths, positional_encoding_length=di)
                step_output = self.fc(outputs).log_softmax(dim=-1)
                logits.append(step_output[:, -1, :])
                input_var[:, di] = logits[-1].topk(1)[1].squeeze()
        return torch.stack(logits, dim=1)


class TransformerTransducerEncoderLayer(nn.Module):
    """
    Repeated layers common to audio encoders and label encoders

    Args:
        model_dim (int): the number of features in the encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of encoder layer (default: 0.1)

    Inputs: inputs, self_attn_mask
        - **inputs**: Audio feature or label feature
        - **self_attn_mask**: Self attention mask to use in multi-head attention

    Returns: outputs, attn_distribution
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): Tensor containing higher (audio, label) feature values
        * attn_distribution (torch.FloatTensor): Attention distribution in multi-head attention
    """

    def __init__(self, model_dim: int=512, d_ff: int=2048, num_heads: int=8, dropout: float=0.1) ->None:
        super(TransformerTransducerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, d_ff, dropout)

    def forward(self, inputs: Tensor, self_attn_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs : A input sequence passed to encoder layer. ``(batch, seq_length, dimension)``
            self_attn_mask : Self attention mask to cover up padding ``(batch, seq_length, seq_length)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            **attn_distribution** (Tensor): ``(batch, seq_length, seq_length)``
        """
        inputs = self.layer_norm(inputs)
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs
        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)
        return output, attn_distribution


class TransformerTransducerDecoder(OpenspeechDecoder):
    """
    Converts the label to higher feature values

    Args:
        num_classes (int): the number of vocabulary
        model_dim (int): the number of features in the label encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of label encoder layers (default: 2)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of label encoder (default: 0.1)
        max_positional_length (int): Maximum length to use for positional encoding (default : 5000)
        pad_id (int): index of padding (default: 0)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)

    Inputs: inputs, inputs_lens
        - **inputs**: Ground truth of batch size number
        - **inputs_lens**: Tensor of target lengths

    Returns:
        (torch.FloatTensor, torch.FloatTensor)

        * outputs (torch.FloatTensor): ``(batch, seq_length, dimension)``
        * input_lengths (torch.FloatTensor):  ``(batch)``

    Reference:
        Qian Zhang et al.: Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss
        https://arxiv.org/abs/2002.02562
    """

    def __init__(self, num_classes: int, model_dim: int=512, d_ff: int=2048, num_layers: int=2, num_heads: int=8, dropout: float=0.1, max_positional_length: int=5000, pad_id: int=0, sos_id: int=1, eos_id: int=2) ->None:
        super(TransformerTransducerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.decoder_layers = nn.ModuleList([TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * outputs (Tensor): ``(batch, seq_length, dimension)``
            * output_lengths (Tensor):  ``(batch)``
        """
        batch = inputs.size(0)
        if len(inputs.size()) == 1:
            inputs = inputs.unsqueeze(1)
            target_lengths = inputs.size(1)
            outputs = self.forward_step(decoder_inputs=inputs, decoder_input_lengths=input_lengths, positional_encoding_length=target_lengths)
        else:
            target_lengths = inputs.size(1)
            outputs = self.forward_step(decoder_inputs=inputs, decoder_input_lengths=input_lengths, positional_encoding_length=target_lengths)
        return outputs, input_lengths

    def forward_step(self, decoder_inputs: torch.Tensor, decoder_input_lengths: torch.Tensor, positional_encoding_length: int=1) ->torch.Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequent_mask, 0)
        embedding_output = self.embedding(decoder_inputs) * self.scale
        positional_encoding_output = self.positional_encoding(positional_encoding_length)
        inputs = embedding_output + positional_encoding_output
        outputs = self.input_dropout(inputs)
        for decoder_layer in self.decoder_layers:
            outputs, _ = decoder_layer(outputs, self_attn_mask)
        return outputs


class BNReluRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoders (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    supported_rnns = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, input_size: int, hidden_state_dim: int=512, rnn_type: str='gru', bidirectional: bool=True, dropout_p: float=0.1):
        super(BNReluRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size=input_size, hidden_size=hidden_state_dim, num_layers=1, bias=True, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        total_length = inputs.size(0)
        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2)))
        inputs = inputs.transpose(1, 2)
        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu())
        outputs, hidden_states = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=total_length)
        return outputs


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) ->Tensor:
        return inputs * inputs.sigmoid()


def get_class_name(obj):
    return obj.__class__.__name__


class Conv2dExtractor(nn.Module):
    """
    Provides inteface of convolutional extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
        Define the 'self.conv' class variable.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    supported_activations = {'hardtanh': nn.Hardtanh(0, 20, inplace=True), 'relu': nn.ReLU(inplace=True), 'elu': nn.ELU(inplace=True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'gelu': nn.GELU(), 'swish': Swish()}

    def __init__(self, input_dim: int, activation: str='hardtanh') ->None:
        super(Conv2dExtractor, self).__init__()
        self.input_dim = input_dim
        self.activation = Conv2dExtractor.supported_activations[activation]
        self.conv = None

    def get_output_lengths(self, seq_lengths: torch.Tensor):
        assert self.conv is not None, 'self.conv should be defined'
        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1
            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1
        return seq_lengths.int()

    def get_output_dim(self):
        if get_class_name(self) == 'VGGExtractor':
            output_dim = self.input_dim - 1 << 5 if self.input_dim % 2 else self.input_dim << 5
        elif get_class_name(self) == 'DeepSpeech2Extractor':
            output_dim = int(math.floor(self.input_dim + 2 * 20 - 41) / 2 + 1)
            output_dim = int(math.floor(output_dim + 2 * 10 - 21) / 2 + 1)
            output_dim <<= 5
        elif get_class_name(self) == 'Conv2dSubsampling':
            factor = ((self.input_dim - 1) // 2 - 1) // 2
            output_dim = self.out_channels * factor
        else:
            raise ValueError(f'Unsupported Extractor : {self.extractor}')
        return output_dim

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        inputs: torch.FloatTensor (batch, time, dimension)
        input_lengths: torch.IntTensor (batch)
        """
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)
        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)
        return outputs, output_lengths


class DeepSpeech2Extractor(Conv2dExtractor):
    """
    DeepSpeech2 extractor for automatic speech recognition described in
    "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin" paper
    - https://arxiv.org/abs/1512.02595

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """

    def __init__(self, input_dim: int, in_channels: int=1, out_channels: int=32, activation: str='hardtanh') ->None:
        super(DeepSpeech2Extractor, self).__init__(input_dim=input_dim, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskConv2d(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False), nn.BatchNorm2d(out_channels), self.activation, nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False), nn.BatchNorm2d(out_channels), self.activation))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(inputs, input_lengths)


class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 is a set of speech recognition models based on Baidu DeepSpeech2. DeepSpeech2 is trained with CTC loss.

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoders (defulat: True)
        activation (str): type of activation function (default: hardtanh)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns:
        (Tensor, Tensor):

        * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
        * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``

    Reference:
        Dario Amodei et al.: Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
        https://arxiv.org/abs/1512.02595
    """

    def __init__(self, input_dim: int, num_classes: int, rnn_type='gru', num_rnn_layers: int=5, rnn_hidden_dim: int=512, dropout_p: float=0.1, bidirectional: bool=True, activation: str='hardtanh') ->None:
        super(DeepSpeech2, self).__init__()
        self.conv = DeepSpeech2Extractor(input_dim, activation=activation)
        self.rnn_layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim
        for idx in range(num_rnn_layers):
            self.rnn_layers.append(BNReluRNN(input_size=self.conv.get_output_dim() if idx == 0 else rnn_output_size, hidden_state_dim=rnn_hidden_dim, rnn_type=rnn_type, bidirectional=bidirectional, dropout_p=dropout_p))
        self.fc = nn.Sequential(nn.LayerNorm(rnn_output_size), Linear(rnn_output_size, num_classes, bias=False))

    def count_parameters(self) ->int:
        """Count parameters of encoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) ->None:
        """Update dropout probability of encoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder_only training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2).contiguous()
        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)
        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)
        return outputs, output_lengths


class MaskConv1d(nn.Conv1d):
    """
    1D convolution with masking

    Args:
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int): Stride of the convolution. Default: 1
        padding (int):  Zero-padding added to both sides of the input. Default: 0
        dilation (int): Spacing between kernel elements. Default: 1
        groups (int): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size (batch, dimension, time)
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the conv1d
        - **seq_lengths**: Sequence length of output from the conv1d
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False) ->None:
        super(MaskConv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def _get_sequence_lengths(self, seq_lengths):
        return (seq_lengths + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        inputs: (batch, dimension, time)
        input_lengths: (batch)
        """
        max_length = inputs.size(2)
        indices = torch.arange(max_length).to(input_lengths.dtype)
        indices = indices.expand(len(input_lengths), max_length)
        mask = indices >= input_lengths.unsqueeze(1)
        inputs = inputs.masked_fill(mask.unsqueeze(1), 0)
        output_lengths = self._get_sequence_lengths(input_lengths)
        output = super(MaskConv1d, self).forward(inputs)
        del mask, indices
        return output, output_lengths


class JasperSubBlock(nn.Module):
    """
    Jasper sub-block applies the following operations: a 1D-convolution, batch norm, ReLU, and dropout.

    Args:
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        padding (int): zero-padding added to both sides of the input. (default: 0)
        bias (bool): if True, adds a learnable bias to the output. (default: False)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """
    supported_activations = {'hardtanh': nn.Hardtanh(0, 20, inplace=True), 'relu': nn.ReLU(inplace=True), 'elu': nn.ELU(inplace=True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'gelu': nn.GELU()}

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, dilation: int=1, padding: int=0, bias: bool=False, dropout_p: float=0.2, activation: str='relu') ->None:
        super(JasperSubBlock, self).__init__()
        self.conv = MaskConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1)
        self.activation = self.supported_activations[activation]
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate of conformer's subblock.

        Inputs: inputs, input_lengths, residual
            - **inputs**: tensor contains input sequence vector
            - **input_lengths**: tensor contains sequence lengths
            - **residual**: tensor contains residual vector

        Returns: output, output_lengths
            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = self.batch_norm(outputs)
        if residual is not None:
            outputs += residual
        outputs = self.dropout(self.activation(outputs))
        return outputs, output_lengths


class JasperBlock(nn.Module):
    """
    Jasper Block: The Jasper Block consists of R Jasper sub-block.

    Args:
        num_sub_blocks (int): number of sub block
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        bias (bool): if True, adds a learnable bias to the output. (default: True)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        (torch.FloatTensor, torch.LongTensor)

        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """

    def __init__(self, num_sub_blocks: int, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, dilation: int=1, bias: bool=True, dropout_p: float=0.2, activation: str='relu') ->None:
        super(JasperBlock, self).__init__()
        padding = self._get_same_padding(kernel_size, stride, dilation)
        self.layers = nn.ModuleList([JasperSubBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, dropout_p=dropout_p, activation=activation) for i in range(num_sub_blocks)])

    def _get_same_padding(self, kernel_size: int, stride: int, dilation: int):
        if stride > 1 and dilation > 1:
            raise ValueError('Only stride OR dilation may be greater than 1')
        return kernel_size // 2 * dilation

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate of jasper block.

        Inputs: inputs, input_lengths, residual
            - **inputs**: tensor contains input sequence vector
            - **input_lengths**: tensor contains sequence lengths
            - **residual**: tensor contains residual vector

        Returns: output, output_lengths
            (torch.FloatTensor, torch.LongTensor)

            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)
        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)
        return outputs, output_lengths


class Conv2dSubsampling(Conv2dExtractor):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, input_dim: int, in_channels: int, out_channels: int, activation: str='relu') ->None:
        super(Conv2dSubsampling, self).__init__(input_dim, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskConv2d(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2), self.activation, nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2), self.activation))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        outputs, output_lengths = super().forward(inputs, input_lengths)
        return outputs, output_lengths


class VGGExtractor(Conv2dExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input image
        out_channels (int or tuple): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """

    def __init__(self, input_dim: int, in_channels: int=1, out_channels: (int or tuple)=(64, 128), activation: str='hardtanh'):
        super(VGGExtractor, self).__init__(input_dim=input_dim, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskConv2d(nn.Sequential(nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_features=out_channels[0]), self.activation, nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_features=out_channels[0]), self.activation, nn.MaxPool2d(2, stride=2), nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_features=out_channels[1]), self.activation, nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_features=out_channels[1]), self.activation, nn.MaxPool2d(2, stride=2)))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(inputs, input_lengths)


class OpenspeechEncoder(nn.Module):
    """
    Base Interface of Openspeech Encoder.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    """
    supported_activations = {'hardtanh': nn.Hardtanh(0, 20, inplace=True), 'relu': nn.ReLU(inplace=True), 'elu': nn.ELU(inplace=True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'gelu': nn.GELU(), 'swish': Swish()}
    supported_extractors = {'ds2': DeepSpeech2Extractor, 'vgg': VGGExtractor, 'conv2d_subsample': Conv2dSubsampling}

    def __init__(self):
        super(OpenspeechEncoder, self).__init__()

    def count_parameters(self) ->int:
        """Count parameters of encoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) ->None:
        """Update dropout probability of encoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        Forward propagate for encoders training.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        """
        raise NotImplementedError


class BaseConv1d(nn.Module):
    """Base convolution module."""

    def __init__(self):
        super(BaseConv1d, self).__init__()

    def _get_sequence_lengths(self, seq_lengths):
        return (seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class PointwiseConv1d(BaseConv1d):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, padding: int=0, bias: bool=True) ->None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, inputs: Tensor) ->Tensor:
        return self.conv(inputs)


class ConvGroupShuffle(nn.Module):
    """Convolution group shuffle module."""

    def __init__(self, groups, channels):
        super(ConvGroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x: Tensor):
        dim = x.size(-1)
        x = x.view(-1, self.groups, self.channels_per_group, dim)
        x = torch.transpose(x, 1, 2).contiguous()
        y = x.view(-1, self.groups * self.channels_per_group, dim)
        return y


class DepthwiseConv1d(BaseConv1d):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, bias: bool=False) ->None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, 'out_channels should be constant multiple of in_channels'
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=bias)

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor]=None) ->Tensor:
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)


class TimeChannelSeparableConv1d(BaseConv1d):
    """
    The total number of weights for a time-channel separable convolution block is K × cin + cin × cout weights. Since K is
    generally several times smaller than cout, most weights are
    concentrated in the pointwise convolution part.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=1, padding: int=0, groups: int=1, bias: bool=True):
        super(TimeChannelSeparableConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, dilation=1, padding=padding, groups=groups, bias=bias)

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor]=None) ->Tensor:
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)


class QuartzNetSubBlock(nn.Module):
    """
    QuartzNet sub-block applies the following operations: a 1D-convolution, batch norm, ReLU, and dropout.

    Args:
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        padding (int): zero-padding added to both sides of the input. (default: 0)
        bias (bool): if True, adds a learnable bias to the output. (default: False)

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool=False, padding: int=0, groups: int=1) ->None:
        super(QuartzNetSubBlock, self).__init__()
        self.depthwise_conf1d = DepthwiseConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.tcs_conv = TimeChannelSeparableConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, groups=groups, bias=bias)
        self.group_shuffle = ConvGroupShuffle(groups, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor, residual: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        outputs, output_lengths = self.depthwise_conf1d(inputs, input_lengths)
        outputs, output_lengths = self.tcs_conv(outputs, output_lengths)
        outputs = self.group_shuffle(outputs)
        outputs = self.batch_norm(outputs)
        if residual is not None:
            outputs += residual
        outputs = self.relu(outputs)
        return outputs, output_lengths


class QuartzNetBlock(nn.Module):
    """
    QuartzNet’s design is based on the Jasper architecture, which is a convolutional model trained with
    Connectionist Temporal Classification (CTC) loss. The main novelty in QuartzNet’s architecture is that QuartzNet
    replaced the 1D convolutions with 1D time-channel separable convolutions, an implementation of depthwise separable
    convolutions.

    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): tensor contains input sequence vector
        input_lengths (torch.LongTensor): tensor contains sequence lengths

    Returns: output, output_lengths
        (torch.FloatTensor, torch.LongTensor)

        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """
    supported_activations = {'hardtanh': nn.Hardtanh(0, 20, inplace=True), 'relu': nn.ReLU(inplace=True), 'elu': nn.ELU(inplace=True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'gelu': nn.GELU()}

    def __init__(self, num_sub_blocks: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool=True) ->None:
        super(QuartzNetBlock, self).__init__()
        padding = self._get_same_padding(kernel_size, stride=1, dilation=1)
        self.layers = nn.ModuleList([QuartzNetSubBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias) for i in range(num_sub_blocks)])
        self.conv1x1 = PointwiseConv1d(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1)

    def _get_same_padding(self, kernel_size: int, stride: int, dilation: int):
        if stride > 1 and dilation > 1:
            raise ValueError('Only stride OR dilation may be greater than 1')
        return kernel_size // 2 * dilation

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagate of QuartzNet block.

        Inputs: inputs, input_lengths
            inputs (torch.FloatTensor): tensor contains input sequence vector
            input_lengths (torch.LongTensor): tensor contains sequence lengths

        Returns: output, output_lengths
            (torch.FloatTensor, torch.LongTensor)

            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        residual = self.batch_norm(self.conv1x1(inputs))
        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)
        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)
        return outputs, output_lengths


class RNNTransducerEncoder(OpenspeechEncoder):
    """
    Encoder of RNN-Transducer.

    Args:
        input_dim (int): dimension of input vector
        hidden_state_dim (int, optional): hidden state dimension of encoders (default: 320)
        output_dim (int, optional): output dimension of encoders and decoders (default: 512)
        num_layers (int, optional): number of encoders layers (default: 4)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoders (default: True)

    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Reference:
        A Graves: Sequence Transduction with Recurrent Neural Networks
        https://arxiv.org/abs/1211.3711.pdf
    """
    supported_rnns = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, input_dim: int, hidden_state_dim: int, output_dim: int, num_layers: int, rnn_type: str='lstm', dropout_p: float=0.2, bidirectional: bool=True):
        super(RNNTransducerEncoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(input_size=input_dim, hidden_size=hidden_state_dim, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        self.fc = Linear(hidden_state_dim << 1 if bidirectional else hidden_state_dim, output_dim)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        inputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(0, 1), input_lengths.cpu())
        outputs, hidden_states = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = self.fc(outputs.transpose(0, 1))
        return outputs, input_lengths


class BaseConv2d(nn.Module):
    """Base convolution module."""

    def __init__(self):
        super(BaseConv2d, self).__init__()

    def _get_sequence_lengths(self, seq_lengths):
        return (seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DepthwiseConv2d(BaseConv2d):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

    Inputs: inputs
        - **inputs** (batch, in_channels, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time, dim): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple], stride: int=2, padding: int=0) ->None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, 'out_channels should be constant multiple of in_channels'
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor]=None) ->Tensor:
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)


class DepthwiseConv2dSubsampling(Conv2dExtractor):
    """
    Depthwise Convolutional 2D subsampling (to 1/4 length)

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, input_dim: int, in_channels: int, out_channels: int, activation: str='relu') ->None:
        super(DepthwiseConv2dSubsampling, self).__init__(input_dim, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskConv2d(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2), self.activation, DepthwiseConv2d(out_channels, out_channels, kernel_size=3, stride=2), self.activation))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        outputs, output_lengths = super().forward(inputs, input_lengths)
        return outputs, output_lengths


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """

    def __init__(self, module: nn.Module, module_factor: float=1.0, input_factor: float=1.0) ->None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor]=None) ->Tensor:
        if mask is None:
            return self.module(inputs) * self.module_factor + inputs * self.input_factor
        else:
            return self.module(inputs, mask) * self.module_factor + inputs * self.input_factor


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoders
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(self, encoder_dim: int=512, expansion_factor: int=4, dropout_p: float=0.1) ->None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(Linear(encoder_dim, encoder_dim * expansion_factor, bias=True), Swish(), nn.Dropout(p=dropout_p), Linear(encoder_dim * expansion_factor, encoder_dim, bias=True), nn.Dropout(p=dropout_p))

    def forward(self, inputs: Tensor) ->Tensor:
        """
        Forward propagate of squeezeformer's feed-forward module.

        Inputs: inputs
            - **inputs** (batch, time, dim): Tensor contains input sequences

        Outputs: outputs
            - **outputs** (batch, time, dim): Tensor produces by feed forward module.
        """
        return self.sequential(inputs)


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int=512, max_len: int=5000) ->None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1:self.pe.size(1) // 2 + x.size(1)]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        dim (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(self, dim: int=512, num_heads: int=16, dropout_p: float=0.1) ->None:
        super(RelativeMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, 'd_model % num_heads should be zero.'
        self.dim = dim
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)
        self.query_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.pos_proj = Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = Linear(dim, dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Optional[Tensor]=None) ->Tensor:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        score = (content_score + pos_score) / self.sqrt_dim
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -10000.0)
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim)
        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) ->Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, :seq_length2 // 2 + 1]
        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float=0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor]=None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) ->None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) ->Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)


class SqueezeformerConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(self, in_channels: int, kernel_size: int=31, expansion_factor: int=2, dropout_p: float=0.1) ->None:
        super(SqueezeformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, 'Currently, Only Supports expansion_factor 2'
        self.sequential = nn.Sequential(Transpose(shape=(1, 2)), PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True), GLU(dim=1), DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm1d(in_channels), Swish(), PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True), nn.Dropout(p=dropout_p))

    def forward(self, inputs: Tensor) ->Tensor:
        return self.sequential(inputs).transpose(1, 2)


class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(self, encoder_dim: int=512, num_attention_heads: int=8, feed_forward_expansion_factor: int=4, conv_expansion_factor: int=2, feed_forward_dropout_p: float=0.1, attention_dropout_p: float=0.1, conv_dropout_p: float=0.1, conv_kernel_size: int=31, half_step_residual: bool=False):
        super(SqueezeformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0
        self.sequential = nn.Sequential(ResidualConnectionModule(module=MultiHeadedSelfAttentionModule(d_model=encoder_dim, num_heads=num_attention_heads, dropout_p=attention_dropout_p)), nn.LayerNorm(encoder_dim), ResidualConnectionModule(module=FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p), module_factor=self.feed_forward_residual_factor), ResidualConnectionModule(module=SqueezeformerConvModule(in_channels=encoder_dim, kernel_size=conv_kernel_size, expansion_factor=conv_expansion_factor, dropout_p=conv_dropout_p)), nn.LayerNorm(encoder_dim), ResidualConnectionModule(module=FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p), module_factor=self.feed_forward_residual_factor), nn.LayerNorm(encoder_dim))

    def forward(self, inputs: Tensor) ->Tensor:
        return self.sequential(inputs)


class TimeReductionLayer(nn.Module):

    def __init__(self, in_channels: int=1, out_channels: int=1, kernel_size: int=3, stride: int=2) ->None:
        super(TimeReductionLayer, self).__init__()
        self.sequential = nn.Sequential(DepthwiseConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride), Swish())

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)
        output_lengths = input_lengths >> 1
        output_lengths -= 1
        return outputs, output_lengths


def recover_resolution(inputs: Tensor) ->Tensor:
    outputs = list()
    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
    return torch.stack(outputs, dim=1)


class SqueezeformerEncoder(OpenspeechEncoder):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        num_classes (int): Number of classification
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths

    Reference:
        Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
        https://arxiv.org/abs/2206.00888
    """

    def __init__(self, num_classes: int, input_dim: int=80, encoder_dim: int=512, num_layers: int=16, reduce_layer_index: int=7, recover_layer_index: int=15, num_attention_heads: int=8, feed_forward_expansion_factor: int=4, conv_expansion_factor: int=2, input_dropout_p: float=0.1, feed_forward_dropout_p: float=0.1, attention_dropout_p: float=0.1, conv_dropout_p: float=0.1, conv_kernel_size: int=31, half_step_residual: bool=False, joint_ctc_attention: bool=True):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = reduce_layer_index
        self.recover_layer_index = recover_layer_index
        self.joint_ctc_attention = joint_ctc_attention
        self.conv_subsample = DepthwiseConv2dSubsampling(input_dim, in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim), nn.Dropout(p=input_dropout_p))
        self.time_reduction_layer = TimeReductionLayer()
        self.time_reduction_proj = nn.Linear((encoder_dim - 1) // 2, encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.recover_tensor = None
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx < reduce_layer_index:
                self.layers.append(SqueezeformerBlock(encoder_dim=encoder_dim, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=feed_forward_expansion_factor, conv_expansion_factor=conv_expansion_factor, feed_forward_dropout_p=feed_forward_dropout_p, attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout_p, conv_kernel_size=conv_kernel_size, half_step_residual=half_step_residual))
            elif reduce_layer_index <= idx < recover_layer_index:
                self.layers.append(ResidualConnectionModule(module=SqueezeformerBlock(encoder_dim=encoder_dim, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=feed_forward_expansion_factor, conv_expansion_factor=conv_expansion_factor, feed_forward_dropout_p=feed_forward_dropout_p, attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout_p, conv_kernel_size=conv_kernel_size, half_step_residual=half_step_residual)))
            else:
                self.layers.append(SqueezeformerBlock(encoder_dim=encoder_dim, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=feed_forward_expansion_factor, conv_expansion_factor=conv_expansion_factor, feed_forward_dropout_p=feed_forward_dropout_p, attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout_p, conv_kernel_size=conv_kernel_size, half_step_residual=half_step_residual))
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(nn.Dropout(feed_forward_dropout_p), nn.Linear(encoder_dim, num_classes, bias=False))

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        encoder_logits = None
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_proj(outputs)
        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                self.recover_tensor = outputs
                outputs, output_lengths = self.time_reduction_layer(outputs, output_lengths)
                outputs = self.time_reduction_proj(outputs)
            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                length = outputs.size(1)
                outputs = self.time_recover_layer(outputs)
                outputs += self.recover_tensor[:, :length, :]
                output_lengths *= 2
            outputs = layer(outputs)
        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs).log_softmax(dim=2)
        return outputs, encoder_logits, output_lengths


class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoders layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer encoder layer
        self_attn_mask (torch.BoolTensor): mask of self attention

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): output of transformer encoder layer
        * attn (torch.FloatTensor): attention of transformer encoder layer
    """

    def __init__(self, d_model: int=512, num_heads: int=8, d_ff: int=2048, dropout_p: float=0.3) ->None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor=None) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate of transformer encoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer encoder layer
            self_attn_mask (torch.BoolTensor): mask of self attention

        Returns:
            outputs (torch.FloatTensor): output of transformer encoder layer
            attn (torch.FloatTensor): attention of transformer encoder layer
        """
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual
        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual
        return outputs, attn


class TransformerEncoder(OpenspeechEncoder):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        input_dim: dimension of feature vector
        d_model: dimension of model (default: 512)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoders layers (default: 6)
        num_heads: number of attention heads (default: 8)
        dropout_p:  probability of dropout (default: 0.3)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns:
        (Tensor, Tensor, Tensor):

        * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
            If joint_ctc_attention is False, return None.  ``(batch, seq_length, num_classes)``
        * output_lengths: The length of encoders outputs. ``(batch)``

    Reference:
        Ashish Vaswani et al.: Attention Is All You Need
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, num_classes: int, input_dim: int=80, d_model: int=512, d_ff: int=2048, num_layers: int=6, num_heads: int=8, dropout_p: float=0.3, joint_ctc_attention: bool=False) ->None:
        super(TransformerEncoder, self).__init__()
        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout_p=dropout_p) for _ in range(num_layers)])
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(Transpose(shape=(1, 2)), nn.Dropout(dropout_p), Linear(d_model, num_classes, bias=False))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor):

            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None. ``(batch, seq_length, num_classes)``
            * output_lengths: The length of encoders outputs. ``(batch)``
        """
        encoder_logits = None
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
        outputs = self.input_norm(self.input_proj(inputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)
        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)
        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=-1)
        return outputs, encoder_logits, input_lengths


class TransformerTransducerEncoder(OpenspeechEncoder):
    """
    Converts the audio signal to higher feature values

    Args:
        input_size (int): dimension of input vector (default : 80)
        model_dim (int): the number of features in the audio encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of audio encoder layers (default: 18)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of audio encoder (default: 0.1)
        max_positional_length (int): Maximum length to use for positional encoding (default : 5000)

    Inputs: inputs, inputs_lens
        - **inputs**: Parsed audio of batch size number
        - **inputs_lens**: Tensor of sequence lengths

    Returns:
        * outputs (torch.FloatTensor): ``(batch, seq_length, dimension)``
        * input_lengths (torch.LongTensor):  ``(batch)``

    Reference:
        Qian Zhang et al.: Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss
        https://arxiv.org/abs/2002.02562
    """

    def __init__(self, input_size: int=80, model_dim: int=512, d_ff: int=2048, num_layers: int=18, num_heads: int=8, dropout: float=0.1, max_positional_length: int=5000) ->None:
        super(TransformerTransducerEncoder, self).__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList([TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagate a `inputs` for audio encoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            ** input_lengths**(Tensor):  ``(batch)``
        """
        seq_len = inputs.size(1)
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, seq_len)
        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)
        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)
        return outputs, input_lengths


class OpenspeechLanguageModelBase(nn.Module):
    """Interface of OpenSpeech decoder."""

    def __init__(self):
        super(OpenspeechLanguageModelBase, self).__init__()

    def count_parameters(self) ->int:
        """Count parameters of decoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) ->None:
        """Update dropout probability of decoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward_step(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class TransformerForLanguageModelLayer(nn.Module):

    def __init__(self, d_model: int=768, num_attention_heads: int=8, d_ff: int=2048, dropout_p: float=0.3) ->None:
        super(TransformerForLanguageModelLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, _ = self.attention(inputs, inputs, inputs, mask)
        outputs += residual
        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual
        return outputs


class TransformerForLanguageModel(OpenspeechLanguageModelBase):
    """
    Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text,
    conversational system, and text summarization. A trained language model learns the likelihood of occurrence
    of a word based on the previous sequence of words used in the text.

    Args:
        num_classes (int): number of classification
        max_length (int): max decoding length (default: 128)
        d_model (int): dimension of model (default: 768)
        d_ff (int): dimension of feed forward network (default: 1536)
        num_attention_heads (int): number of attention heads (default: 8)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        num_layers (int, optional): number of transformer layers (default: 2)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs:, inputs, input_lengths
        inputs (torch.LongTensor): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    def __init__(self, num_classes: int, max_length: int=128, d_model: int=768, num_attention_heads: int=8, d_ff: int=1536, pad_id: int=0, sos_id: int=1, eos_id: int=2, num_layers: int=2, dropout_p: float=0.3):
        super(TransformerForLanguageModel, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = TransformerEmbedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([TransformerForLanguageModelLayer(d_model=d_model, num_attention_heads=num_attention_heads, d_ff=d_ff, dropout_p=dropout_p) for _ in range(num_layers)])
        self.fc = nn.Sequential(nn.LayerNorm(d_model), Linear(d_model, d_model, bias=False), nn.Tanh(), Linear(d_model, num_classes, bias=False))

    def forward_step(self, inputs, input_lengths):
        pad_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
        subsequent_mask = get_attn_subsequent_mask(inputs)
        mask = torch.gt(pad_mask + subsequent_mask, 0)
        outputs = self.embedding(inputs) + self.positional_encoding(inputs.size(1))
        outputs = self.input_dropout(outputs)
        for layer in self.layers:
            outputs = layer(inputs=outputs, mask=mask)
        step_outputs = self.fc(outputs).log_softmax(dim=-1)
        return step_outputs

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) ->torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            inputs (torch.LongTensor): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        step_outputs = self.forward_step(inputs, input_lengths)
        for di in range(step_outputs.size(1)):
            step_output = step_outputs[:, di, :]
            logits.append(step_output)
        return torch.stack(logits, dim=1)


class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need".
    Transformer employ a residual connection around each of the two sub-layers,
    (Multi-Head Attention & Feed-Forward) followed by layer normalization.
    """

    def __init__(self, sublayer: nn.Module, d_model: int=512) ->None:
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        outputs = self.sublayer(*args)
        if isinstance(outputs, tuple):
            return self.layer_norm(outputs[0] + residual), outputs[1]
        return self.layer_norm(outputs + residual)


class AdditiveAttention(nn.Module):
    """
    Applies a additive attention (bahdanau) mechanism on the output features from the decoders.
    Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

    Args:
        dim (int): dimension of model

    Inputs: query, key, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the encoders outputs.
    """

    def __init__(self, dim: int) ->None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = Linear(dim, dim, bias=False)
        self.key_proj = Linear(dim, dim, bias=False)
        self.score_proj = Linear(dim, 1)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, key: Tensor, value: Tensor) ->Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        context += query
        return context, attn


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(self, in_channels: int, kernel_size: int=31, expansion_factor: int=2, dropout_p: float=0.1) ->None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, 'Currently, Only Supports expansion_factor 2'
        self.sequential = nn.Sequential(nn.LayerNorm(in_channels), Transpose(shape=(1, 2)), PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True), GLU(dim=1), DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm1d(in_channels), Swish(), PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True), nn.Dropout(p=dropout_p))

    def forward(self, inputs: Tensor) ->Tensor:
        """
        Forward propagate of conformer's convolution module.

        Inputs: inputs
            inputs (batch, time, dim): Tensor contains input sequences

        Outputs: outputs
            outputs (batch, time, dim): Tensor produces by conformer convolution module.
        """
        return self.sequential(inputs).transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoders
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(self, encoder_dim: int=512, num_attention_heads: int=8, feed_forward_expansion_factor: int=4, conv_expansion_factor: int=2, feed_forward_dropout_p: float=0.1, attention_dropout_p: float=0.1, conv_dropout_p: float=0.1, conv_kernel_size: int=31, half_step_residual: bool=True) ->None:
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1
        self.sequential = nn.Sequential(ResidualConnectionModule(module=FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p), module_factor=self.feed_forward_residual_factor), ResidualConnectionModule(module=MultiHeadedSelfAttentionModule(d_model=encoder_dim, num_heads=num_attention_heads, dropout_p=attention_dropout_p)), ResidualConnectionModule(module=ConformerConvModule(in_channels=encoder_dim, kernel_size=conv_kernel_size, expansion_factor=conv_expansion_factor, dropout_p=conv_dropout_p)), ResidualConnectionModule(module=FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p), module_factor=self.feed_forward_residual_factor), nn.LayerNorm(encoder_dim))

    def forward(self, inputs: Tensor) ->Tensor:
        return self.sequential(inputs)


class ContextNetConvModule(nn.Module):
    """
    When the stride is 1, it pads the input so the output has the shape as the input.
    And when the stride is 2, it does not pad the input.

    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        activation (bool, optional): Flag indication use activation function or not (default : True)
        groups(int, optional): Value of groups (default : 1)
        bias (bool, optional): Flag indication use bias or not (default : True)

    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution layer `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output, output_lengths
        - **output**: Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=5, stride: int=1, padding: int=0, activation: bool=True, groups: int=1, bias: bool=True):
        super(ContextNetConvModule, self).__init__()
        assert kernel_size == 5, 'The convolution layer in the ContextNet model has 5 kernels.'
        if stride == 1:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=1, padding=(kernel_size - 1) // 2, groups=groups, bias=bias)
        elif stride == 2:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=1, padding=padding, groups=groups, bias=bias)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = activation
        if self.activation:
            self.swish = Swish()

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for convolution layer.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs), self._get_sequence_lengths(input_lengths)
        outputs = self.batch_norm(outputs)
        if self.activation:
            outputs = self.swish(outputs)
        return outputs, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1


class ContextNetSEModule(nn.Module):
    """
    Squeeze-and-excitation module.

    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers

    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """

    def __init__(self, dim: int) ->None:
        super(ContextNetSEModule, self).__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.sequential = nn.Sequential(nn.Linear(dim, dim // 8), Swish(), nn.Linear(dim // 8, dim))

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for SE Layer.

        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = inputs
        seq_lengths = inputs.size(2)
        inputs = inputs.sum(dim=2) / input_lengths.unsqueeze(1)
        output = self.sequential(inputs)
        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1, 1, seq_lengths)
        return output * residual


class ContextNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions, each followed by batch normalization and activation.
    Squeeze-and-excitation (SE) block operates on the output of the last convolution layer.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.

    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        residual (bool, optional): Flag indication residual or not (default : True)

    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int=5, kernel_size: int=5, stride: int=1, padding: int=0, residual: bool=True) ->None:
        super(ContextNetBlock, self).__init__()
        self.num_layers = num_layers
        self.swish = Swish()
        self.se_layer = ContextNetSEModule(out_channels)
        self.residual = None
        if residual:
            self.residual = ContextNetConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=False)
        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            stride_list = [(1) for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]
            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(ContextNetConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

    def forward(self, inputs: Tensor, input_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for convolution block.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths
        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)
        output = self.se_layer(output, output_lengths)
        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual
        return self.swish(output), output_lengths

    @staticmethod
    def make_conv_blocks(input_dim: int=80, num_layers: int=5, kernel_size: int=5, num_channels: int=256, output_dim: int=640) ->nn.ModuleList:
        """
        Create 23 convolution blocks.

        Args:
            input_dim (int, optional): Dimension of input vector (default : 80)
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)

        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()
        conv_blocks.append(ContextNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, False))
        for _ in range(1, 2 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))
        for _ in range(4, 6 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))
        for _ in range(8, 10 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))
        conv_blocks.append(ContextNetBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, True))
        for _ in range(12, 13 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True))
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, True))
        for i in range(15, 21 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True))
        conv_blocks.append(ContextNetBlock(num_channels << 1, output_dim, 1, kernel_size, 1, 0, False))
        return conv_blocks


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoders.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        dim (int): dimension of model
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoders outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.

    Reference:
        Jan Chorowski et al.: Attention-Based Models for Speech Recognition.
        https://arxiv.org/abs/1506.07503
    """

    def __init__(self, dim: int=1024, attn_dim: int=1024, smoothing: bool=False) ->None:
        super(LocationAwareAttention, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = Linear(dim, attn_dim, bias=False)
        self.value_proj = Linear(dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor) ->Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)
        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)
        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)
        alignmment_energy = self.fc(torch.tanh(self.query_proj(query) + self.value_proj(value) + last_alignment_energy + self.bias)).squeeze(dim=-1)
        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(dim=-1).unsqueeze(dim=-1))
        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)
        context = torch.bmm(alignmment_energy.unsqueeze(dim=1), value)
        return context, alignmment_energy


class MaskConv2d(nn.Module):
    """
    Masking Convolutional Neural Network

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)

    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License

    Args:
        sequential (torch.nn): sequential list of convolution layer

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """

    def __init__(self, sequential: nn.Sequential) ->None:
        super(MaskConv2d, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) ->Tuple[Tensor, Tensor]:
        output = None
        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)
            if output.is_cuda:
                mask = mask
            seq_lengths = self._get_sequence_lengths(module, seq_lengths)
            for idx, length in enumerate(seq_lengths):
                length = length.item()
                if mask[idx].size(2) - length > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)
            output = output.masked_fill(mask, 0)
            inputs = output
        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) ->Tensor:
        """
        Calculate convolutional neural network receptive formula

        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch

        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1
        elif isinstance(module, DepthwiseConv2d):
            numerator = seq_lengths + 2 * module.conv.padding[1] - module.conv.dilation[1] * (module.conv.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.conv.stride[1])
            seq_lengths = seq_lengths.int() + 1
        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1
        return seq_lengths.int()


class View(nn.Module):
    """Wrapper class of torch.view() for Sequential module."""

    def __init__(self, shape: tuple, contiguous: bool=False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()
        return inputs.view(*self.shape)


class OpenspeechBeamSearchBase(nn.Module):
    """
    Openspeech's beam-search base class. Implement the methods required for beamsearch.
    You have to implement `forward` method.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, decoder, beam_size: int):
        super(OpenspeechBeamSearchBase, self).__init__()
        self.decoder = decoder
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.pad_id = decoder.pad_id
        self.eos_id = decoder.eos_id
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.forward_step = decoder.forward_step

    def _inflate(self, tensor: torch.Tensor, n_repeat: int, dim: int) ->torch.Tensor:
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[dim] *= n_repeat
        return tensor.repeat(*repeat_dims)

    def _get_successor(self, current_ps: torch.Tensor, current_vs: torch.Tensor, finished_ids: tuple, num_successor: int, eos_count: int, k: int) ->int:
        finished_batch_idx, finished_idx = finished_ids
        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]
        successor_p = current_ps[finished_batch_idx, successor_idx]
        successor_v = current_vs[finished_batch_idx, successor_idx]
        prev_status_idx = successor_idx // k
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1]
        successor = torch.cat([prev_status, successor_v.view(1)])
        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor_p)
            eos_count = self._get_successor(current_ps=current_ps, current_vs=current_vs, finished_ids=finished_ids, num_successor=num_successor + eos_count, eos_count=eos_count + 1, k=k)
        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p
        return eos_count

    def _get_hypothesis(self):
        predictions = list()
        for batch_idx, batch in enumerate(self.finished):
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx]
                top_beam_idx = int(prob_batch.topk(1)[1])
                predictions.append(self.ongoing_beams[batch_idx, top_beam_idx])
            else:
                top_beam_idx = int(torch.FloatTensor(self.finished_ps[batch_idx]).topk(1)[1])
                predictions.append(self.finished[batch_idx][top_beam_idx])
        predictions = self._fill_sequence(predictions)
        return predictions

    def _is_all_finished(self, k: int) ->bool:
        for done in self.finished:
            if len(done) < k:
                return False
        return True

    def _fill_sequence(self, y_hats: list) ->torch.Tensor:
        batch_size = len(y_hats)
        max_length = -1
        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)
        matched = torch.zeros((batch_size, max_length), dtype=torch.long)
        for batch_idx, y_hat in enumerate(y_hats):
            matched[batch_idx, :len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat):] = int(self.pad_id)
        return matched

    def forward(self, *args, **kwargs):
        raise NotImplementedError


CTCDECODE_IMPORT_ERROR = """
Openspeech requires the ctcdecode library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/parlance/ctcdecode and follow the ones that match your environment.
"""


class LSTMAttentionDecoder(OpenspeechDecoder):
    """
    Converts higher level features (from encoders) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoders hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        attn_mechanism (str, optional): type of attention mechanism (default: multi-head)
        num_heads (int, optional): number of attention heads. (default: 4)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_state_dim): tensor with containing the outputs of the encoders.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: logits
        * logits (torch.FloatTensor) : log probabilities of model's prediction
    """
    supported_rnns = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, num_classes: int, max_length: int=150, hidden_state_dim: int=1024, pad_id: int=0, sos_id: int=1, eos_id: int=2, attn_mechanism: str='multi-head', num_heads: int=4, num_layers: int=2, rnn_type: str='lstm', dropout_p: float=0.3) ->None:
        super(LSTMAttentionDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(input_size=hidden_state_dim, hidden_size=hidden_state_dim, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout_p, bidirectional=False)
        if self.attn_mechanism == 'loc':
            self.attention = LocationAwareAttention(hidden_state_dim, attn_dim=hidden_state_dim, smoothing=False)
        elif self.attn_mechanism == 'multi-head':
            self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        elif self.attn_mechanism == 'additive':
            self.attention = AdditiveAttention(hidden_state_dim)
        elif self.attn_mechanism == 'dot':
            self.attention = DotProductAttention(dim=hidden_state_dim)
        elif self.attn_mechanism == 'scaled-dot':
            self.attention = DotProductAttention(dim=hidden_state_dim, scale=True)
        else:
            raise ValueError('Unsupported attention: %s'.format(attn_mechanism))
        self.fc = nn.Sequential(Linear(hidden_state_dim << 1, hidden_state_dim), nn.Tanh(), View(shape=(-1, self.hidden_state_dim), contiguous=True), Linear(hidden_state_dim, num_classes))

    def forward_step(self, input_var: torch.Tensor, hidden_states: Optional[torch.Tensor], encoder_outputs: torch.Tensor, attn: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.training:
            self.rnn.flatten_parameters()
        outputs, hidden_states = self.rnn(embedded, hidden_states)
        if self.attn_mechanism == 'loc':
            context, attn = self.attention(outputs, encoder_outputs, attn)
        else:
            context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)
        outputs = torch.cat((outputs, context), dim=2)
        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)
        return step_outputs, hidden_states, attn

    def forward(self, encoder_outputs: torch.Tensor, targets: Optional[torch.Tensor]=None, encoder_output_lengths: Optional[torch.Tensor]=None, teacher_forcing_ratio: float=1.0) ->torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoders outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        hidden_states, attn = None, None
        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            if self.attn_mechanism == 'loc' or self.attn_mechanism == 'additive':
                for di in range(targets.size(1)):
                    input_var = targets[:, di].unsqueeze(1)
                    step_outputs, hidden_states, attn = self.forward_step(input_var=input_var, hidden_states=hidden_states, encoder_outputs=encoder_outputs, attn=attn)
                    logits.append(step_outputs)
            else:
                step_outputs, hidden_states, attn = self.forward_step(input_var=targets, hidden_states=hidden_states, encoder_outputs=encoder_outputs, attn=attn)
                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    logits.append(step_output)
        else:
            input_var = targets[:, 0].unsqueeze(1)
            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(input_var=input_var, hidden_states=hidden_states, encoder_outputs=encoder_outputs, attn=attn)
                logits.append(step_outputs)
                input_var = logits[-1].topk(1)[1]
        logits = torch.stack(logits, dim=1)
        return logits

    def validate_args(self, targets: Optional[Any]=None, encoder_outputs: torch.Tensor=None, teacher_forcing_ratio: float=1.0) ->Tuple[torch.Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)
        if targets is None:
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length
            if torch.cuda.is_available():
                targets = targets
            if teacher_forcing_ratio > 0:
                raise ValueError('Teacher forcing has to be disabled (set 0) when no targets is provided.')
        else:
            max_length = targets.size(1) - 1
        return targets, batch_size, max_length


class BeamSearchLSTM(OpenspeechBeamSearchBase):
    """
    LSTM Beam Search Decoder

    Args: decoder, beam_size, batch_size
        decoder (DecoderLSTM): base decoder of lstm model.
        beam_size (int): size of beam.

    Inputs: encoder_outputs, targets, encoder_output_lengths, teacher_forcing_ratio
        encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        encoder_output_lengths (torch.LongTensor): A encoder output lengths sequence. `LongTensor` of size ``(batch)``
        teacher_forcing_ratio (float): Ratio of teacher forcing.

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    def __init__(self, decoder: LSTMAttentionDecoder, beam_size: int):
        super(BeamSearchLSTM, self).__init__(decoder, beam_size)
        self.hidden_state_dim = decoder.hidden_state_dim
        self.num_layers = decoder.num_layers
        self.validate_args = decoder.validate_args

    def forward(self, encoder_outputs: torch.Tensor, encoder_output_lengths: torch.Tensor) ->torch.Tensor:
        """
        Beam search decoding.

        Inputs: encoder_outputs
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size, hidden_states = encoder_outputs.size(0), None
        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]
        inputs, batch_size, max_length = self.validate_args(None, encoder_outputs, teacher_forcing_ratio=0.0)
        step_outputs, hidden_states, attn = self.forward_step(inputs, hidden_states, encoder_outputs)
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(self.beam_size)
        self.ongoing_beams = self.ongoing_beams.view(batch_size * self.beam_size, 1)
        self.cumulative_ps = self.cumulative_ps.view(batch_size * self.beam_size, 1)
        input_var = self.ongoing_beams
        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = self._inflate(encoder_outputs, self.beam_size, dim=0)
        encoder_outputs = encoder_outputs.view(self.beam_size, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * self.beam_size, -1, encoder_dim)
        if attn is not None:
            attn = self._inflate(attn, self.beam_size, dim=0)
        if isinstance(hidden_states, tuple):
            hidden_states = tuple([self._inflate(h, self.beam_size, 1) for h in hidden_states])
        else:
            hidden_states = self._inflate(hidden_states, self.beam_size, 1)
        for di in range(max_length - 1):
            if self._is_all_finished(self.beam_size):
                break
            if isinstance(hidden_states, tuple):
                tuple(h.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim) for h in hidden_states)
            else:
                hidden_states = hidden_states.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim)
            step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs, attn)
            step_outputs = step_outputs.view(batch_size, self.beam_size, -1)
            current_ps, current_vs = step_outputs.topk(self.beam_size)
            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)
            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1)
            current_ps = current_ps.view(batch_size, self.beam_size ** 2)
            current_vs = current_vs.view(batch_size, self.beam_size ** 2)
            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)
            topk_current_ps, topk_status_ids = current_ps.topk(self.beam_size)
            prev_status_ids = topk_status_ids // self.beam_size
            topk_current_vs = torch.zeros((batch_size, self.beam_size), dtype=torch.long)
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long)
            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]
            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2)
            self.cumulative_ps = topk_current_ps
            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size
                for batch_idx, idx in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])
                    if self.beam_size != 1:
                        eos_count = self._get_successor(current_ps=current_ps, current_vs=current_vs, finished_ids=(batch_idx, idx), num_successor=num_successors[batch_idx], eos_count=1, k=self.beam_size)
                        num_successors[batch_idx] += eos_count
            input_var = self.ongoing_beams[:, :, -1]
            input_var = input_var.view(batch_size * self.beam_size, -1)
        return self._get_hypothesis()


class BeamSearchRNNTransducer(OpenspeechBeamSearchBase):
    """
    RNN Transducer Beam Search
    Reference: RNN-T FOR LATENCY CONTROLLED ASR WITH IMPROVED BEAM SEARCH (https://arxiv.org/pdf/1911.01629.pdf)

    Args: joint, decoder, beam_size, expand_beam, state_beam, blank_id
        joint: joint `encoder_outputs` and `decoder_outputs`
        decoder (TransformerTransducerDecoder): base decoder of transformer transducer model.
        beam_size (int): size of beam.
        expand_beam (int): The threshold coefficient to limit the number of expanded hypotheses.
        state_beam (int): The threshold coefficient to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (ongoing_beams)
        blank_id (int): blank id

    Inputs: encoder_output, max_length
        encoder_output (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(seq_length, dimension)``
        max_length (int): max decoding time step

    Returns:
        * predictions (torch.LongTensor): model predictions.
    """

    def __init__(self, joint, decoder: RNNTransducerDecoder, beam_size: int=3, expand_beam: float=2.3, state_beam: float=4.6, blank_id: int=3) ->None:
        super(BeamSearchRNNTransducer, self).__init__(decoder, beam_size)
        self.joint = joint
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        """
        Beam search decoding.

        Inputs: encoder_output, max_length
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predictions (torch.LongTensor): model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()
        for batch_idx in range(encoder_outputs.size(0)):
            blank = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.blank_id
            step_input = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.sos_id
            hyp = {'prediction': [self.sos_id], 'logp_score': 0.0, 'hidden_states': None}
            ongoing_beams = [hyp]
            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()
                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break
                    a_best_hyp = max(process_hyps, key=lambda x: x['logp_score'] / len(x['prediction']))
                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(ongoing_beams, key=lambda x: x['logp_score'] / len(x['prediction']))
                        a_best_prob = a_best_hyp['logp_score']
                        b_best_prob = b_best_hyp['logp_score']
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break
                    process_hyps.remove(a_best_hyp)
                    step_input[0, 0] = a_best_hyp['prediction'][-1]
                    step_outputs, hidden_states = self.decoder(step_input, a_best_hyp['hidden_states'])
                    log_probs = self.joint(encoder_outputs[batch_idx, t_step, :], step_outputs.view(-1))
                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)
                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]
                    for j in range(topk_targets.size(0)):
                        topk_hyp = {'prediction': a_best_hyp['prediction'][:], 'logp_score': a_best_hyp['logp_score'] + topk_targets[j], 'hidden_states': a_best_hyp['hidden_states']}
                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue
                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp['prediction'].append(topk_idx[j].item())
                            topk_hyp['hidden_states'] = hidden_states
                            process_hyps.append(topk_hyp)
            ongoing_beams = sorted(ongoing_beams, key=lambda x: x['logp_score'] / len(x['prediction']), reverse=True)[0]
            hypothesis.append(torch.LongTensor(ongoing_beams['prediction'][1:]))
            hypothesis_score.append(ongoing_beams['logp_score'] / len(ongoing_beams['prediction']))
        return self._fill_sequence(hypothesis)


class BeamSearchTransformer(OpenspeechBeamSearchBase):
    """
    Transformer Beam Search Decoder

    Args: decoder, beam_size, batch_size
        decoder (DecoderLSTM): base decoder of lstm model.
        beam_size (int): size of beam.

    Inputs: encoder_outputs, targets, encoder_output_lengths, teacher_forcing_ratio
        encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        encoder_output_lengths (torch.LongTensor): A encoder output lengths sequence. `LongTensor` of size ``(batch)``
        teacher_forcing_ratio (float): Ratio of teacher forcing.

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    def __init__(self, decoder: TransformerDecoder, beam_size: int=3) ->None:
        super(BeamSearchTransformer, self).__init__(decoder, beam_size)
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, encoder_outputs: torch.FloatTensor, encoder_output_lengths: torch.FloatTensor):
        batch_size = encoder_outputs.size(0)
        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]
        decoder_inputs = torch.IntTensor(batch_size, self.decoder.max_length).fill_(self.sos_id).long()
        decoder_input_lengths = torch.IntTensor(batch_size).fill_(1)
        outputs = self.forward_step(decoder_inputs=decoder_inputs[:, :1], decoder_input_lengths=decoder_input_lengths, encoder_outputs=encoder_outputs, encoder_output_lengths=encoder_output_lengths, positional_encoding_length=1)
        step_outputs = self.decoder.fc(outputs).log_softmax(dim=-1)
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(self.beam_size)
        self.ongoing_beams = self.ongoing_beams.view(batch_size * self.beam_size, 1)
        self.cumulative_ps = self.cumulative_ps.view(batch_size * self.beam_size, 1)
        decoder_inputs = torch.IntTensor(batch_size * self.beam_size, 1).fill_(self.sos_id)
        decoder_inputs = torch.cat((decoder_inputs, self.ongoing_beams), dim=-1)
        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = self._inflate(encoder_outputs, self.beam_size, dim=0)
        encoder_outputs = encoder_outputs.view(self.beam_size, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * self.beam_size, -1, encoder_dim)
        encoder_output_lengths = encoder_output_lengths.unsqueeze(1).repeat(1, self.beam_size).view(-1)
        for di in range(2, self.decoder.max_length):
            if self._is_all_finished(self.beam_size):
                break
            decoder_input_lengths = torch.LongTensor(batch_size * self.beam_size).fill_(di)
            step_outputs = self.forward_step(decoder_inputs=decoder_inputs[:, :di], decoder_input_lengths=decoder_input_lengths, encoder_outputs=encoder_outputs, encoder_output_lengths=encoder_output_lengths, positional_encoding_length=di)
            step_outputs = self.decoder.fc(step_outputs).log_softmax(dim=-1)
            step_outputs = step_outputs.view(batch_size, self.beam_size, -1, 10)
            current_ps, current_vs = step_outputs.topk(self.beam_size)
            current_ps = current_ps[:, :, -1, :]
            current_vs = current_vs[:, :, -1, :]
            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)
            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1)
            current_ps = current_ps.view(batch_size, self.beam_size ** 2)
            current_vs = current_vs.contiguous().view(batch_size, self.beam_size ** 2)
            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)
            topk_current_ps, topk_status_ids = current_ps.topk(self.beam_size)
            prev_status_ids = topk_status_ids // self.beam_size
            topk_current_vs = torch.zeros((batch_size, self.beam_size), dtype=torch.long)
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long)
            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]
            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2)
            self.cumulative_ps = topk_current_ps
            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size
                for batch_idx, idx in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])
                    if self.beam_size != 1:
                        eos_count = self._get_successor(current_ps=current_ps, current_vs=current_vs, finished_ids=(batch_idx, idx), num_successor=num_successors[batch_idx], eos_count=1, k=self.beam_size)
                        num_successors[batch_idx] += eos_count
            ongoing_beams = self.ongoing_beams.clone().view(batch_size * self.beam_size, -1)
            decoder_inputs = torch.cat((decoder_inputs, ongoing_beams[:, :-1]), dim=-1)
        return self._get_hypothesis()


class BeamSearchTransformerTransducer(OpenspeechBeamSearchBase):
    """
    Transformer Transducer Beam Search
    Reference: RNN-T FOR LATENCY CONTROLLED ASR WITH IMPROVED BEAM SEARCH (https://arxiv.org/pdf/1911.01629.pdf)

    Args: joint, decoder, beam_size, expand_beam, state_beam, blank_id
        joint: joint `encoder_outputs` and `decoder_outputs`
        decoder (TransformerTransducerDecoder): base decoder of transformer transducer model.
        beam_size (int): size of beam.
        expand_beam (int): The threshold coefficient to limit the number
        of expanded hypotheses that are added in A (process_hyp).
        state_beam (int): The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (ongoing_beams)
        blank_id (int): blank id

    Inputs: encoder_outputs, max_length
        encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        max_length (int): max decoding time step

    Returns:
        * predictions (torch.LongTensor): model predictions.
    """

    def __init__(self, joint, decoder: TransformerTransducerDecoder, beam_size: int=3, expand_beam: float=2.3, state_beam: float=4.6, blank_id: int=3) ->None:
        super(BeamSearchTransformerTransducer, self).__init__(decoder, beam_size)
        self.joint = joint
        self.forward_step = self.decoder.forward_step
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        """
        Beam search decoding.

        Inputs: encoder_outputs, max_length
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predictions (torch.LongTensor): model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()
        for batch_idx in range(encoder_outputs.size(0)):
            blank = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.blank_id
            step_input = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.sos_id
            hyp = {'prediction': [self.sos_id], 'logp_score': 0.0}
            ongoing_beams = [hyp]
            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()
                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break
                    a_best_hyp = max(process_hyps, key=lambda x: x['logp_score'] / len(x['prediction']))
                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(ongoing_beams, key=lambda x: x['logp_score'] / len(x['prediction']))
                        a_best_prob = a_best_hyp['logp_score']
                        b_best_prob = b_best_hyp['logp_score']
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break
                    process_hyps.remove(a_best_hyp)
                    step_input[0, 0] = a_best_hyp['prediction'][-1]
                    step_lengths = encoder_outputs.new_tensor([0], dtype=torch.long)
                    step_outputs = self.forward_step(step_input, step_lengths).squeeze(0).squeeze(0)
                    log_probs = self.joint(encoder_outputs[batch_idx, t_step, :], step_outputs)
                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)
                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]
                    for j in range(topk_targets.size(0)):
                        topk_hyp = {'prediction': a_best_hyp['prediction'][:], 'logp_score': a_best_hyp['logp_score'] + topk_targets[j]}
                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue
                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp['prediction'].append(topk_idx[j].item())
                            process_hyps.append(topk_hyp)
            ongoing_beams = sorted(ongoing_beams, key=lambda x: x['logp_score'] / len(x['prediction']), reverse=True)[0]
            hypothesis.append(torch.LongTensor(ongoing_beams['prediction'][1:]))
            hypothesis_score.append(ongoing_beams['logp_score'] / len(ongoing_beams['prediction']))
        return self._fill_sequence(hypothesis)


class EnsembleSearch(nn.Module):
    """
    Class for ensemble search.

    Args:
        models (tuple): list of ensemble model

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * predictions (torch.LongTensor): prediction of ensemble models
    """

    def __init__(self, models: Union[list, tuple]):
        super(EnsembleSearch, self).__init__()
        assert len(models) > 1, 'Ensemble search should be multiple models.'
        self.models = models

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor):
        logits = list()
        for model in self.models:
            output = model(inputs, input_lengths)
            logits.append(output['logits'])
        output = logits[0]
        for logit in logits[1:]:
            output += logit
        return output.max(-1)[1]


class WeightedEnsembleSearch(nn.Module):
    """
    Args:
        models (tuple): list of ensemble model
        weights (tuple: list of ensemble's weight

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * predictions (torch.LongTensor): prediction of ensemble models
    """

    def __init__(self, models: Union[list, tuple], weights: Union[list, tuple]):
        super(WeightedEnsembleSearch, self).__init__()
        assert len(models) > 1, 'Ensemble search should be multiple models.'
        assert len(models) == len(weights), 'len(models), len(weight) should be same.'
        self.models = models
        self.weights = weights

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor):
        logits = list()
        for model in self.models:
            output = model(inputs, input_lengths)
            logits.append(output['logits'])
        output = logits[0] * self.weights[0]
        for idx, logit in enumerate(logits[1:]):
            output += logit * self.weights[1]
        return output.max(-1)[1]

