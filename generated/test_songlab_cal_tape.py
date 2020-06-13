import sys
_module = sys.modules[__name__]
del sys
adding_model = _module
adding_task = _module
fix_lmdb = _module
generate_plots = _module
lmdb_to_fasta = _module
tfrecord_to_json = _module
tfrecord_to_lmdb = _module
setup = _module
tape = _module
datasets = _module
errors = _module
main = _module
metrics = _module
models = _module
file_utils = _module
modeling_bert = _module
modeling_lstm = _module
modeling_onehot = _module
modeling_resnet = _module
modeling_trrosetta = _module
modeling_unirep = _module
modeling_utils = _module
optimization = _module
registry = _module
tokenizers = _module
training = _module
utils = _module
_sampler = _module
distributed_utils = _module
setup_utils = _module
visualization = _module
test_basic = _module

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


import torch


import torch.nn as nn


from typing import Union


from typing import List


from typing import Tuple


from typing import Sequence


from typing import Dict


from typing import Any


from typing import Optional


from typing import Collection


from copy import copy


import logging


import random


import numpy as np


import torch.nn.functional as F


from torch.utils.data import Dataset


from scipy.spatial.distance import pdist


from scipy.spatial.distance import squareform


import typing


import warnings


import inspect


import math


from torch import nn


from torch.utils.checkpoint import checkpoint


from torch.nn.utils import weight_norm


import copy


from torch.nn.utils.weight_norm import weight_norm


import torch.optim as optim


from torch.utils.data import DataLoader


from abc import ABC


from abc import abstractmethod


class ProteinBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinBertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs
            ) if self.output_attentions else (context_layer,)
        return outputs


class ProteinBertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None
        ).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class ProteinBertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ProteinBertSelfAttention(config)
        self.output = ProteinBertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.
            attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size * self.
            self.num_attention_heads)

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_activation_fn(name: str) ->typing.Callable:
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f'Unrecognized activation fn: {name}')


class ProteinBertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ProteinBertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = ProteinBertAttention(config)
        self.intermediate = ProteinBertIntermediate(config)
        self.output = ProteinBertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class ProteinBertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([ProteinBertLayer(config) for _ in range
            (config.num_hidden_layers)])

    def run_function(self, start, chunk_size):

        def custom_forward(hidden_states, attention_mask):
            all_hidden_states = ()
            all_attentions = ()
            chunk_slice = slice(start, start + chunk_size)
            for layer in self.layer[chunk_slice]:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]
                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = hidden_states,
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
        return custom_forward

    def forward(self, hidden_states, attention_mask, chunks=None):
        all_hidden_states = ()
        all_attentions = ()
        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer) + chunks - 1) // chunks
            for start in range(0, len(self.layer), chunk_size):
                outputs = checkpoint(self.run_function(start, chunk_size),
                    hidden_states, attention_mask)
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + outputs[1]
                if self.output_attentions:
                    all_attentions = all_attentions + outputs[-1]
                hidden_states = outputs[0]
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer_module(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]
                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = hidden_states,
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs


class ProteinBertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMLayer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class ProteinLSTMPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_hidden_layers, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MaskedConv1d(nn.Conv1d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


class ProteinResNetLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ProteinResNetBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = MaskedConv1d(config.hidden_size, config.hidden_size, 3,
            padding=1, bias=False)
        self.bn1 = ProteinResNetLayerNorm(config)
        self.conv2 = MaskedConv1d(config.hidden_size, config.hidden_size, 3,
            padding=1, bias=False)
        self.bn2 = ProteinResNetLayerNorm(config)
        self.activation_fn = get_activation_fn(config.hidden_act)

    def forward(self, x, input_mask=None):
        identity = x
        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.conv2(out, input_mask)
        out = self.bn2(out)
        out += identity
        out = self.activation_fn(out)
        return out


class ProteinResNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, embed_dim,
            padding_idx=0)
        inverse_frequency = 1 / 10000 ** (torch.arange(0.0, embed_dim, 2.0) /
            embed_dim)
        self.register_buffer('inverse_frequency', inverse_frequency)
        self.layer_norm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length - 1, -1, -1.0, dtype=
            words_embeddings.dtype, device=words_embeddings.device)
        sinusoidal_input = torch.ger(position_ids, self.inverse_frequency)
        position_embeddings = torch.cat([sinusoidal_input.sin(),
            sinusoidal_input.cos()], -1)
        position_embeddings = position_embeddings.unsqueeze(0)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinResNetPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_weights = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None):
        attention_scores = self.attention_weights(hidden_states)
        if mask is not None:
            attention_scores += -10000.0 * (1 - mask)
        attention_weights = torch.softmax(attention_scores, -1)
        weighted_mean_embedding = torch.matmul(hidden_states.transpose(1, 2
            ), attention_weights).squeeze(2)
        pooled_output = self.dense(weighted_mean_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([ProteinResNetBlock(config) for _ in
            range(config.num_hidden_layers)])

    def forward(self, hidden_states, input_mask=None):
        all_hidden_states = ()
        for layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states, input_mask)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


class DilatedResidualBlock(nn.Module):

    def __init__(self, num_features: int, kernel_size: int, dilation: int,
        dropout: float):
        super().__init__()
        padding = self._get_padding(kernel_size, dilation)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size,
            padding=padding, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-06)
        self.actv1 = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size,
            padding=padding, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-06)
        self.actv2 = nn.ELU(inplace=True)
        self.apply(self._init_weights)
        nn.init.constant_(self.norm2.weight, 0)

    def _get_padding(self, kernel_size: int, dilation: int) ->int:
        return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, features):
        shortcut = features
        features = self.conv1(features)
        features = self.norm1(features)
        features = self.actv1(features)
        features = self.dropout(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.actv2(features + shortcut)
        return features


class mLSTMCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        project_size = config.hidden_size * 4
        self.wmx = weight_norm(nn.Linear(config.input_size, config.
            hidden_size, bias=False))
        self.wmh = weight_norm(nn.Linear(config.hidden_size, config.
            hidden_size, bias=False))
        self.wx = weight_norm(nn.Linear(config.input_size, project_size,
            bias=False))
        self.wh = weight_norm(nn.Linear(config.hidden_size, project_size,
            bias=True))

    def forward(self, inputs, state):
        h_prev, c_prev = state
        m = self.wmx(inputs) * self.wmh(h_prev)
        z = self.wx(inputs) + self.wh(m)
        i, f, o, u = torch.chunk(z, 4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        c = f * c_prev + i * u
        h = o * torch.tanh(c)
        return h, c


class mLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlstm_cell = mLSTMCell(config)
        self.hidden_size = config.hidden_size

    def forward(self, inputs, state=None, mask=None):
        batch_size = inputs.size(0)
        seqlen = inputs.size(1)
        if mask is None:
            mask = torch.ones(batch_size, seqlen, 1, dtype=inputs.dtype,
                device=inputs.device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)
        if state is None:
            zeros = torch.zeros(batch_size, self.hidden_size, dtype=inputs.
                dtype, device=inputs.device)
            state = zeros, zeros
        steps = []
        for seq in range(seqlen):
            prev = state
            seq_input = inputs[:, (seq), :]
            hx, cx = self.mlstm_cell(seq_input, state)
            seqmask = mask[:, (seq)]
            hx = seqmask * hx + (1 - seqmask) * prev[0]
            cx = seqmask * cx + (1 - seqmask) * prev[1]
            state = hx, cx
            steps.append(hx)
        return torch.stack(steps, 1), (hx, cx)


CONFIG_NAME = 'config.json'


class SimpleMLP(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout:
        float=0.0):
        super().__init__()
        self.main = nn.Sequential(weight_norm(nn.Linear(in_dim, hid_dim),
            dim=None), nn.ReLU(), nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class SimpleConv(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout:
        float=0.0):
        super().__init__()
        self.main = nn.Sequential(nn.BatchNorm1d(in_dim), weight_norm(nn.
            Conv1d(in_dim, hid_dim, 5, padding=2), dim=None), nn.ReLU(), nn
            .Dropout(dropout, inplace=True), weight_norm(nn.Conv1d(hid_dim,
            out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


def accuracy(logits, labels, ignore_index: int=-100):
    with torch.no_grad():
        valid_mask = labels != ignore_index
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class Accuracy(nn.Module):

    def __init__(self, ignore_index: int=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class PredictionHeadTransform(nn.Module):

    def __init__(self, hidden_size: int, hidden_act: typing.Union[str,
        typing.Callable]='gelu', layer_norm_eps: float=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int, hidden_act:
        typing.Union[str, typing.Callable]='gelu', layer_norm_eps: float=
        1e-12, ignore_index: int=-100):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act,
            layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = hidden_states,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(hidden_states.view(-1, self.
                vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = masked_lm_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class ValuePredictionHead(nn.Module):

    def __init__(self, hidden_size: int, dropout: float=0.0):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, 512, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = value_pred,
        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        return outputs


class SequenceClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = logits,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = classification_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int, ignore_index: int
        =-100):
        super().__init__()
        self.classify = SimpleConv(hidden_size, 512, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = sequence_logits,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(sequence_logits.view(-1, self.
                num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy': acc_fct(sequence_logits.view(-1, self.
                num_labels), targets.view(-1))}
            loss_and_metrics = classification_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(nn.Dropout(), nn.Linear(2 *
            hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, sequence_lengths, targets=None):
        prod = inputs[:, :, (None), :] * inputs[:, (None), :, :]
        diff = inputs[:, :, (None), :] - inputs[:, (None), :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()
        outputs = prediction,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(prediction.view(-1, 2), targets.view(-1))
            metrics = {'precision_at_l5': self.compute_precision_at_l5(
                sequence_lengths, prediction, targets)}
            loss_and_metrics = contact_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=
                sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= (y_ind - x_ind >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, (1)]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs,
                labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_songlab_cal_tape(_paritybench_base):
    pass
    def test_000(self):
        self._check(ProteinBertPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ProteinLSTMLayer(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MaskedConv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ProteinResNetPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4])], {})

    def test_004(self):
        self._check(SimpleMLP(*[], **{'in_dim': 4, 'hid_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SimpleConv(*[], **{'in_dim': 4, 'hid_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(Accuracy(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(ValuePredictionHead(*[], **{'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SequenceClassificationHead(*[], **{'hidden_size': 4, 'num_labels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(SequenceToSequenceClassificationHead(*[], **{'hidden_size': 4, 'num_labels': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(PairwiseContactPredictionHead(*[], **{'hidden_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

