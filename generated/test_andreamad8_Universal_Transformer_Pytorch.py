import sys
_module = sys.modules[__name__]
del sys
main = _module
UTransformer = _module
common_layer = _module

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


from torchtext import datasets


from torchtext.datasets.babi import BABI20Field


import torch.nn as nn


import torch


import numpy as np


from copy import deepcopy


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn.init as I


import math


class ACT_basic(nn.Module):

    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1])
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1])
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1])
        previous_state = torch.zeros_like(inputs)
        step = 0
        while ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any():
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, (step), :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
            p = self.sigma(self.p(state)).squeeze(-1)
            still_running = (halting_probability < 1.0).float()
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = p * still_running + new_halted * remainders
            if encoder_output:
                state, _ = fn((state, encoder_output))
            else:
                state = fn(state)
            previous_state = state * update_weights.unsqueeze(-1) + previous_state * (1 - update_weights.unsqueeze(-1))
            step += 1
        return previous_state, (remainders, n_updates)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError('Key depth (%d) must be divisible by the number of attention heads (%d).' % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError('Value depth (%d) must be divisible by the number of attention heads (%d).' % (total_value_depth, num_heads))
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError('x must have rank 3')
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError('x must have rank 4')
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, src_mask=None):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if src_mask is not None:
            logits = logits.masked_fill(src_mask, -np.inf)
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)
        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)
        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = [(input_depth, filter_size)] + [(filter_size, filter_size)] * (len(layer_config) - 2) + [(filter_size, output_depth)]
        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError('Unknown layer type {}'.format(lc))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs
        x_norm = self.layer_norm_mha(x)
        y = self.multi_head_attention(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=10000.0):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    return torch.from_numpy(signal).type(torch.FloatTensor)


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        super(Encoder, self).__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)
        self.num_layers = num_layers
        self.act = act
        params = hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, _gen_bias_mask(max_length) if use_mask else None, layer_dropout, attention_dropout, relu_dropout
        self.proj_flag = False
        if embedding_size == hidden_size:
            self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
            self.proj_flag = True
        self.enc = EncoderLayer(*params)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if self.act:
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs):
        x = self.input_dropout(inputs)
        if self.proj_flag:
            x = self.embedding_proj(x)
        if self.act:
            x, (remainders, n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
            return x, (remainders, n_updates)
        else:
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, (l), :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                x = self.enc(x)
            return x, None


class BabiUTransformer(nn.Module):
    """
    A Transformer Module For BabI data. 
    Inputs should be in the shape story: [batch_size, memory_size, story_len ]
                                  query: [batch_size, 1, story_len]
    Outputs will have the shape [batch_size, ]
    """

    def __init__(self, num_vocab, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=71, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False):
        super(BabiUTransformer, self).__init__()
        self.embedding_dim = embedding_size
        self.emb = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
        self.transformer_enc = Encoder(embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=71, input_dropout=input_dropout, layer_dropout=layer_dropout, attention_dropout=attention_dropout, relu_dropout=relu_dropout, use_mask=False, act=act)
        self.W = nn.Linear(self.embedding_dim, num_vocab)
        self.W.weight = self.emb.weight
        self.softmax = nn.Softmax(dim=1)
        self.mask = nn.Parameter(I.constant_(torch.empty(11, self.embedding_dim), 1))

    def forward(self, story, query):
        story_size = story.size()
        embed = self.emb(story.view(story.size(0), -1))
        embed = embed.view(story_size + (embed.size(-1),))
        embed_story = torch.sum(embed * self.mask[:story.size(2), :].unsqueeze(0), 2)
        query_embed = self.emb(query)
        embed_query = torch.sum(query_embed.unsqueeze(1) * self.mask[:query.size(1), :], 2)
        embed = torch.cat([embed_story, embed_query], dim=1)
        logit, act = self.transformer_enc(embed)
        a_hat = self.W(torch.sum(logit, dim=1) / logit.size(1))
        return a_hat, self.softmax(a_hat), act


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout)
        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, None, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='left', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs = inputs
        x_norm = self.layer_norm_mha_dec(x)
        y = self.multi_head_attention_dec(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_mha_enc(x)
        y = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y, encoder_outputs


class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, act=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(Decoder, self).__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)
        self.num_layers = num_layers
        self.act = act
        params = hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, _gen_bias_mask(max_length), layer_dropout, attention_dropout, relu_dropout
        self.proj_flag = False
        if embedding_size == hidden_size:
            self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
            self.proj_flag = True
        self.dec = DecoderLayer(*params)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if self.act:
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs, encoder_output):
        x = self.input_dropout(inputs)
        if self.proj_flag:
            x = self.embedding_proj(x)
        if self.act:
            x, (remainders, n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output)
            return x, (remainders, n_updates)
        else:
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, (l), :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                x, _ = self.dec((x, encoder_output))
        return x


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, (self.padding_idx)] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'kernel_size': 4, 'pad_type': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'input_depth': 1, 'filter_size': 4, 'output_depth': 1}),
     lambda: ([torch.rand([1, 1])], {}),
     False),
]

class Test_andreamad8_Universal_Transformer_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

