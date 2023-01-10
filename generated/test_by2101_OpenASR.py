import sys
_module = sys.modules[__name__]
del sys
avg_last_ckpts = _module
data = _module
data_test = _module
decode = _module
decoder_layers = _module
encoder_layers = _module
encoder_layers_test = _module
lm_layers = _module
lm_train = _module
metric = _module
models = _module
modules = _module
prepare_data = _module
schedule = _module
sp_layers = _module
sp_layers_test = _module
stat_grapheme = _module
stat_length = _module
kaldi_io = _module
kaldi_signal = _module
transformer = _module
wavfile = _module
train = _module
trainer = _module
utils = _module
utils_test = _module

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


import logging


import torch


import numpy as np


import torch.utils.data as data


from torch.utils.data.sampler import Sampler


import torch.utils.data


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


from torch.nn.modules.normalization import LayerNorm


from collections import OrderedDict


from torch.nn.init import xavier_uniform_


import torch.nn.init as init


import random


import copy


from torch.nn.modules import Module


from torch.nn.modules.activation import MultiheadAttention


from torch.nn.modules.container import ModuleList


from torch.nn.modules.dropout import Dropout


from torch.nn.modules.linear import Linear


import time


from torch import nn


from torch.nn.utils import clip_grad_norm_


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_atten=False):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        atten_probs_list = []
        for i in range(self.num_layers):
            if return_atten:
                output, atten_probs_tuple = self.layers[i](output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, return_atten=True)
                atten_probs_list.append(atten_probs_tuple)
            else:
                output = self.layers[i](output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, return_atten=False)
        if self.norm:
            output = self.norm(output)
        if return_atten:
            return output, atten_probs_list
        return output


class Conv1dSubsample(torch.nn.Module):

    def __init__(self, input_dim, d_model, context_width, subsample):
        super(Conv1dSubsample, self).__init__()
        self.conv = nn.Conv1d(input_dim, d_model, context_width, stride=self.subsample)
        self.conv_norm = LayerNorm(self.d_model)
        self.subsample = subsample
        self.context_width = context_width

    def forward(self, feats, feat_lengths):
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = output.permute(0, 2, 1)
        outputs = self.conv_norm(outputs)
        output_lengths = ((feat_lengths - 1 * (self.context_width - 1) - 1) / self.subsample + 1).long()
        return outputs, output_lengths


class Conv2dSubsample(torch.nn.Module):

    def __init__(self, input_dim, d_model):
        super(Conv2dSubsample, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 3, 2), torch.nn.ReLU())
        self.affine = torch.nn.Linear(32 * (((input_dim - 1) // 2 - 1) // 2), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C * D)
        outputs = self.affine(outputs)
        output_lengths = (((feat_lengths - 1) / 2 - 1) / 2).long()
        return outputs, output_lengths


class Conv2dSubsampleV2(torch.nn.Module):

    def __init__(self, input_dim, d_model, layer_num=2):
        super(Conv2dSubsampleV2, self).__init__()
        assert layer_num >= 1
        self.layer_num = layer_num
        layers = [('subsample/conv0', torch.nn.Conv2d(1, 32, 3, (2, 1))), ('subsample/relu0', torch.nn.ReLU())]
        for i in range(layer_num - 1):
            layers += [('subsample/conv{}'.format(i + 1), torch.nn.Conv2d(32, 32, 3, (2, 1))), ('subsample/relu{}'.format(i + 1), torch.nn.ReLU())]
        layers = OrderedDict(layers)
        self.conv = torch.nn.Sequential(layers)
        self.affine = torch.nn.Linear(32 * (input_dim - 2 * layer_num), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C * D)
        outputs = self.affine(outputs)
        output_lengths = feat_lengths
        for _ in range(self.layer_num):
            output_lengths = ((output_lengths - 1) / 2).long()
        return outputs, output_lengths


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'glu':
        return F.glu
    else:
        raise RuntimeError('activation should be relu/gelu, not %s.' % activation)


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        if activation == 'glu':
            self.linear1 = Linear(d_model, 2 * dim_feedforward)
        else:
            self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_atten=False):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_atten_probs = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, enc_dec_atten_probs = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, 'activation'):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_atten:
            return tgt, (self_atten_probs, enc_dec_atten_probs)
        return tgt


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, return_atten=False):
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        atten_probs_list = []
        for i in range(self.num_layers):
            if return_atten:
                output, self_atten_probs = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, return_atten=True)
                atten_probs_list.append(self_atten_probs)
            else:
                output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, return_atten=False)
        if self.norm:
            output = self.norm(output)
        if return_atten:
            return output, atten_probs_list
        return output


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        if activation == 'glu':
            self.linear1 = Linear(d_model, 2 * dim_feedforward)
        else:
            self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_atten=False):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, self_atten_probs = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, 'activation'):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if return_atten:
            return src, self_atten_probs
        return src


class Transformer(Module):
    """A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class LSTM(nn.Module):

    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.emb = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.output_affine = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.emb.weight = self.output_affine.weight

    def forward(self, ids, lengths=None):
        outputs = self.emb(ids)
        outputs = self.dropout1(outputs)
        outputs, (h, c) = self.rnn(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.output_affine(outputs)
        return outputs

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        init.uniform_(self.emb.weight, a=-0.01, b=0.01)


class TransformerLM(nn.Module):

    def __init__(self, config):
        super(TransformerLM, self).__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.activation = config['activation']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(self.dropout_rate)
        self.scale = self.d_model ** 0.5
        self.pe = modules.PositionalEncoding(self.d_model)
        self.emb = nn.Embedding(self.vocab_size, self.d_model)
        encoder_layer = transformer.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout_rate, activation=self.activation)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layer, self.num_layers)
        self.output_affine = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.emb.weight = self.output_affine.weight

    def forward(self, ids, lengths, return_atten=False):
        B, T = ids.shape
        key_padding_mask = utils.get_transformer_padding_byte_masks(B, T, lengths)
        casual_masks = utils.get_transformer_casual_masks(T)
        outputs = self.emb(ids) * self.scale
        outputs = self.pe(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.permute(1, 0, 2)
        outputs, self_atten_list = self.transformer_encoder(outputs, mask=casual_masks, src_key_padding_mask=key_padding_mask, return_atten=True)
        outputs = self.output_affine(outputs)
        if return_atten:
            return outputs, self_atten_list
        return outputs


class Model(torch.nn.Module):

    def __init__(self, splayer, encoder, decoder, lm=None):
        super(Model, self).__init__()
        self.splayer = splayer
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()
        self.lm = lm

    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0.0, lst_w=0.0, lst_t=1.0, return_atten=False):
        target_lengths = torch.sum(1 - target_paddings, dim=-1).long()
        logits, atten_info = self.get_logits(batch_wave, lengths, target_ids, target_lengths, return_atten=True)
        losses = self._compute_cross_entropy_losses(logits, target_labels, target_paddings)
        loss = torch.sum(losses)
        if label_smooth > 0:
            loss = loss * (1 - label_smooth) + self._uniform_label_smooth(logits, target_paddings) * label_smooth
        if lst_w > 0.0:
            loss = loss * (1 - lst_w) + self._lst(logits, target_ids, target_paddings, T=lst_t) * lst_w
        if return_atten:
            return loss, atten_info
        return loss

    def _uniform_label_smooth(self, logits, paddings):
        log_probs = F.log_softmax(logits, dim=-1)
        nlabel = log_probs.shape[-1]
        ent_uniform = -torch.sum(log_probs, dim=-1) / nlabel
        return torch.sum(ent_uniform * (1 - paddings).float())

    def _lst(self, logits, target_ids, target_paddings, T=5.0):
        with torch.no_grad():
            self.lm.eval()
            lengths = torch.sum(1 - target_paddings, dim=-1).long()
            teacher_probs = self.lm.get_probs(target_ids, lengths, T=T)
        logprobs = torch.log_softmax(logits, dim=-1)
        losses = -torch.sum(teacher_probs * logprobs, dim=-1)
        return torch.sum(losses * (1 - target_paddings).float())

    def _compute_cross_entropy_losses(self, logits, labels, paddings):
        B, T, V = logits.shape
        losses = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction='none').view(B, T) * (1 - paddings).float()
        return losses

    def _compute_wers(self, hyps, labels):
        raise NotImplementedError()

    def _sample_nbest(self, encoder_output, encoder_output_lengths, nbest_keep=4):
        self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, maxlen)
        raise NotImplementedError()

    def _compute_mwer_loss(self):
        raise NotImplementedError()

    def get_logits(self, batch_wave, lengths, target_ids, target_lengths, return_atten=False):
        if return_atten:
            timer = utils.Timer()
            timer.tic()
            sp_outputs, sp_output_lengths = self.splayer(batch_wave, lengths)
            logging.debug('splayer time: {}s'.format(timer.toc()))
            timer.tic()
            encoder_outputs, encoder_output_lengths, enc_self_atten_list = self.encoder(sp_outputs, sp_output_lengths, return_atten=True)
            logging.debug('encoder time: {}s'.format(timer.toc()))
            timer.tic()
            outputs, decoder_atten_tuple_list = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths, return_atten=True)
            logging.debug('decoder time: {}s'.format(timer.toc()))
            timer.tic()
            return outputs, (encoder_outputs, encoder_output_lengths, enc_self_atten_list, target_lengths, decoder_atten_tuple_list, sp_outputs, sp_output_lengths)
        else:
            timer = utils.Timer()
            timer.tic()
            encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
            logging.debug('splayer time: {}s'.format(timer.toc()))
            timer.tic()
            encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths, return_atten=False)
            logging.debug('encoder time: {}s'.format(timer.toc()))
            timer.tic()
            outputs = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths, return_atten=False)
            logging.debug('decoder time: {}s'.format(timer.toc()))
            timer.tic()
            return outputs

    def decode(self, batch_wave, lengths, nbest_keep, sosid=1, eosid=2, maxlen=100):
        if type(nbest_keep) != int:
            raise ValueError('nbest_keep must be a int.')
        encoder_outputs, encoder_output_lengths = self._get_acoustic_representations(batch_wave, lengths)
        target_ids, scores = self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen)
        return target_ids, scores

    def _get_acoustic_representations(self, batch_wave, lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths, return_atten=False)
        return encoder_outputs, encoder_output_lengths

    def _beam_search(self, encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen):
        B = encoder_outputs.shape[0]
        init_target_ids = torch.ones(B, 1).long() * sosid
        init_target_lengths = torch.ones(B).long()
        outputs = self.decoder(encoder_outputs, encoder_output_lengths, init_target_ids, init_target_lengths)[:, -1, :]
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)
        topk_res = torch.topk(log_probs, k=nbest_keep, dim=-1)
        nbest_ids = topk_res[1].view(-1)
        nbest_logprobs = topk_res[0].view(-1)
        target_ids = torch.ones(B * nbest_keep, 1).long() * sosid
        target_lengths = torch.ones(B * nbest_keep).long()
        target_ids = torch.cat([target_ids, nbest_ids.view(B * nbest_keep, 1)], dim=-1)
        target_lengths += 1
        finished_sel = None
        ended = []
        ended_scores = []
        ended_batch_idx = []
        for step in range(1, maxlen):
            nbest_ids, nbest_logprobs, beam_from = self._decode_single_step(encoder_outputs, encoder_output_lengths, target_ids, target_lengths, nbest_logprobs, finished_sel)
            batch_idx = (torch.arange(B) * nbest_keep).view(B, -1).repeat(1, nbest_keep).contiguous()
            batch_beam_from = (batch_idx + beam_from.view(-1, nbest_keep)).view(-1)
            nbest_logprobs = nbest_logprobs.view(-1)
            finished_sel = nbest_ids.view(-1) == eosid
            target_ids = target_ids[batch_beam_from]
            target_ids = torch.cat([target_ids, nbest_ids.view(B * nbest_keep, 1)], dim=-1)
            target_lengths += 1
            for i in range(finished_sel.shape[0]):
                if finished_sel[i]:
                    ended.append(target_ids[i])
                    ended_scores.append(nbest_logprobs[i])
                    ended_batch_idx.append(i // nbest_keep)
            target_ids = target_ids * (1 - finished_sel[:, None].long())
        for i in range(target_ids.shape[0]):
            ended.append(target_ids[i])
            ended_scores.append(nbest_logprobs[i])
            ended_batch_idx.append(i // nbest_keep)
        formated = {}
        for i in range(B):
            formated[i] = []
        for i in range(len(ended)):
            if ended[i][0] == sosid:
                formated[ended_batch_idx[i]].append((ended[i], ended_scores[i]))
        for i in range(B):
            formated[i] = sorted(formated[i], key=lambda x: x[1], reverse=True)[:nbest_keep]
        target_ids = torch.zeros(B, nbest_keep, maxlen + 1).long()
        scores = torch.zeros(B, nbest_keep)
        for i in range(B):
            for j in range(nbest_keep):
                item = formated[i][j]
                l = min(item[0].shape[0], target_ids[i, j].shape[0])
                target_ids[i, j, :l] = item[0][:l]
                scores[i, j] = item[1]
        return target_ids, scores

    def _decode_single_step(self, encoder_outputs, encoder_output_lengths, target_ids, target_lengths, accumu_scores, finished_sel=None):
        """
        encoder_outputs: [B, T_e, D_e]
        encoder_output_lengths: [B]
        target_ids: [B*nbest_keep, T_d]
        target_lengths: [B*nbest_keep]
        accumu_scores: [B*nbest_keep]
        """
        B, T_e, D_e = encoder_outputs.shape
        B_d, T_d = target_ids.shape
        if B_d % B != 0:
            raise ValueError('The dim of target_ids does not match the encoder_outputs.')
        nbest_keep = B_d // B
        encoder_outputs = encoder_outputs.view(B, 1, T_e, D_e).repeat(1, nbest_keep, 1, 1).view(B * nbest_keep, T_e, D_e)
        encoder_output_lengths = encoder_output_lengths.view(B, 1).repeat(1, nbest_keep).view(-1)
        outputs = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)[:, -1, :]
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, nbest_keep, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)
        if finished_sel is not None:
            log_probs = log_probs.view(B * nbest_keep, -1) - finished_sel.view(B * nbest_keep, -1).float() * 9000000000.0
            log_probs = log_probs.view(B, nbest_keep, vocab_size)
        this_accumu_scores = accumu_scores.view(B, nbest_keep, 1) + log_probs
        topk_res = torch.topk(this_accumu_scores.view(B, nbest_keep * vocab_size), k=nbest_keep, dim=-1)
        nbest_logprobs = topk_res[0]
        nbest_ids = topk_res[1] % vocab_size
        beam_from = (topk_res[1] / vocab_size).long()
        return nbest_ids, nbest_logprobs, beam_from

    def package(self):
        pkg = {'splayer_config': self.splayer.config, 'splayer_state': self.splayer.state_dict(), 'encoder_config': self.encoder.config, 'encoder_state': self.encoder.state_dict(), 'decoder_config': self.decoder.config, 'decoder_state': self.decoder.state_dict()}
        return pkg

    def restore(self, pkg):
        logging.info('Restore model states...')
        for key in self.splayer.config.keys():
            if key == 'spec_aug':
                continue
            if self.splayer.config[key] != pkg['splayer_config'][key]:
                raise ValueError('splayer_config mismatch.')
        for key in self.encoder.config.keys():
            if key != 'dropout_rate' and self.encoder.config[key] != pkg['encoder_config'][key]:
                raise ValueError('encoder_config mismatch.')
        for key in self.decoder.config.keys():
            if key != 'dropout_rate' and self.decoder.config[key] != pkg['decoder_config'][key]:
                raise ValueError('decoder_config mismatch.')
        self.splayer.load_state_dict(pkg['splayer_state'])
        self.encoder.load_state_dict(pkg['encoder_state'])
        self.decoder.load_state_dict(pkg['decoder_state'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class LM(torch.nn.Module):

    def __init__(self, lmlayer):
        super(LM, self).__init__()
        self.lm_layer = lmlayer

    def forward(self, ids, labels, paddings, label_smooth=0.0):
        lengths = torch.sum(1 - paddings, dim=1).long()
        logits = self.get_logits(ids, lengths)
        ntoken = torch.sum(1 - paddings)
        tot_loss = torch.sum(self._compute_ce_loss(logits, labels, paddings))
        if label_smooth > 0:
            tot_loss = tot_loss * (1 - label_smooth) + self._uniform_label_smooth(logits, paddings) * label_smooth
        tot_ncorrect = self._compute_ncorrect(logits, labels, paddings)
        return tot_loss, tot_ncorrect

    def fetch_vis_info(self, ids, labels, paddings):
        lengths = torch.sum(1 - paddings, dim=1).long()
        atten = None
        if isinstance(self.lm_layer, lm_layers.TransformerLM):
            logits, atten = self.lm_layer(ids, lengths, return_atten=True)
        elif isinstance(self.lm_layer, clozer.ClozerV2) or isinstance(self.lm_layer, clozer.Clozer) or isinstance(self.lm_layer, clozer.UniClozer) or isinstance(self.lm_layer, clozer.BwdUniClozer):
            logits, atten = self.lm_layer(ids, lengths, return_atten=True)
        else:
            raise ValueError('Unknown lm layer')
        return atten

    def get_probs(self, ids, lengths, T=1.0):
        logits = self.get_logits(ids, lengths)
        probs = F.softmax(logits / T, dim=-1)
        return probs

    def get_logprobs(self, ids, lengths, T=1.0):
        logits = self.get_logits(ids, lengths)
        logprobs = F.log_softmax(logits / T, dim=-1)
        return logprobs

    def get_logits(self, ids, lengths=None):
        if len(ids.shape) == 1:
            B = ids.shape[0]
            ids = ids.view(B, 1).contiguous()
        logits = self.lm_layer(ids, lengths)
        return logits

    def _compute_ce_loss(self, logits, labels, paddings):
        D = logits.size(-1)
        losses = F.cross_entropy(logits.view(-1, D).contiguous(), labels.view(-1), reduction='none')
        return losses * (1 - paddings).view(-1).float()

    def _uniform_label_smooth(self, logits, paddings):
        log_probs = F.log_softmax(logits, dim=-1)
        nlabel = log_probs.shape[-1]
        ent_uniform = -torch.sum(log_probs, dim=-1) / nlabel
        return torch.sum(ent_uniform * (1 - paddings).float())

    def _compute_ncorrect(self, logits, labels, paddings):
        D = logits.size(-1)
        logprobs = F.log_softmax(logits, dim=-1)
        pred = torch.argmax(logprobs.view(-1, D), dim=-1)
        n_correct = torch.sum((pred == labels.view(-1)).float() * (1 - paddings).view(-1).float())
        return n_correct

    def package(self):
        pkg = {'lm_config': self.lm_layer.config, 'lm_state': self.lm_layer.state_dict()}
        return pkg

    def restore(self, pkg):
        logging.info('Restore model states...')
        for key in self.lm_layer.config.keys():
            if key != 'dropout_rate' and self.lm_layer.config[key] != pkg['lm_config'][key]:
                raise ValueError('lm_config mismatch.')
        self.lm_layer.load_state_dict(pkg['lm_state'])

    def _reset_parameters(self):
        self.lm_layer.reset_parameters()


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.scale = d_model ** 0.5
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return input * self.scale + self.pe[:, :length]


class SPLayer(nn.Module):

    def __init__(self, config):
        super(SPLayer, self).__init__()
        self.config = config
        self.feature_type = config['feature_type']
        self.sample_rate = float(config['sample_rate'])
        self.num_mel_bins = int(config['num_mel_bins'])
        self.use_energy = config['use_energy']
        self.spec_aug_conf = None
        if 'spec_aug' in config:
            self.spec_aug_conf = {'freq_mask_num': config['spec_aug']['freq_mask_num'], 'freq_mask_width': config['spec_aug']['freq_mask_width'], 'time_mask_num': config['spec_aug']['time_mask_num'], 'time_mask_width': config['spec_aug']['time_mask_width']}
        if self.feature_type == 'mfcc':
            self.num_ceps = config['num_ceps']
        else:
            self.num_ceps = None
        if self.feature_type == 'offline':
            feature_func = None
            logging.warn('Use offline features. It is your duty to keep features match.')
        elif self.feature_type == 'fbank':

            def feature_func(waveform):
                return ksp.fbank(waveform, sample_frequency=self.sample_rate, use_energy=self.use_energy, num_mel_bins=self.num_mel_bins)
        elif self.feature_type == 'mfcc':

            def feature_func(waveform):
                return ksp.mfcc(waveform, sample_frequency=self.sample_rate, use_energy=self.use_energy, num_mel_bins=self.num_mel_bins)
        else:
            raise ValueError('Unknown feature type.')
        self.func = feature_func

    def spec_aug(self, padded_features, feature_lengths):
        freq_means = torch.mean(padded_features, dim=-1)
        time_means = torch.sum(padded_features, dim=1) / feature_lengths[:, None].float()
        B, T, V = padded_features.shape
        for _ in range(self.spec_aug_conf['freq_mask_num']):
            fs = (self.spec_aug_conf['freq_mask_width'] * torch.rand(size=[B], device=padded_features.device, requires_grad=False)).long()
            f0s = ((V - fs).float() * torch.rand(size=[B], device=padded_features.device, requires_grad=False)).long()
            for b in range(B):
                padded_features[b, :, f0s[b]:f0s[b] + fs[b]] = freq_means[b][:, None]
        for _ in range(self.spec_aug_conf['time_mask_num']):
            ts = (self.spec_aug_conf['time_mask_width'] * torch.rand(size=[B], device=padded_features.device, requires_grad=False)).long()
            t0s = ((feature_lengths - ts).float() * torch.rand(size=[B], device=padded_features.device, requires_grad=False)).long()
            for b in range(B):
                padded_features[b, t0s[b]:t0s[b] + ts[b], :] = time_means[b][None, :]
        return padded_features, feature_lengths

    def forward(self, wav_batch, lengths):
        batch_size, batch_length = wav_batch.shape[0], wav_batch.shape[1]
        if self.func is not None:
            features = []
            feature_lengths = []
            for i in range(batch_size):
                feature = self.func(wav_batch[i, :lengths[i]].view(1, -1))
                features.append(feature)
                feature_lengths.append(feature.shape[0])
            max_length = max(feature_lengths)
            padded_features = torch.zeros(batch_size, max_length, feature.shape[-1])
            for i in range(batch_size):
                l = feature_lengths[i]
                padded_features[i, :l, :] += features[i]
        else:
            padded_features = torch.tensor(wav_batch)
            feature_lengths = lengths
        feature_lengths = torch.tensor(feature_lengths).long()
        if self.training and self.spec_aug_conf is not None:
            padded_features, feature_lengths = self.spec_aug(padded_features, feature_lengths)
        return padded_features, feature_lengths


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_by2101_OpenASR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

