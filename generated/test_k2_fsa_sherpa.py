import sys
_module = sys.modules[__name__]
del sys
generate_feats_scp = _module
cmake = _module
cmake_extension = _module
_ext = _module
rst_roles = _module
conf = _module
bpe_model_to_tokens = _module
generate_build_matrix = _module
setup = _module
beam_search = _module
stream = _module
streaming_server = _module
beam_search = _module
offline_asr = _module
stream = _module
streaming_server = _module
beam_search = _module
stream = _module
streaming_client = _module
streaming_server = _module
pruned_transducer_statelessX = _module
beam_search = _module
decode_manifest = _module
offline_asr = _module
offline_client = _module
offline_server = _module
beam_search = _module
stream = _module
streaming_server = _module
sherpa = _module
decode = _module
http_server = _module
lexicon = _module
nbest = _module
online_endpoint = _module
timestamp = _module
utils = _module
test_fast_beam_search_config = _module
test_feature_config = _module
test_hypothesis = _module
test_offline_ctc_decoder_config = _module
test_offline_recognizer = _module
test_offline_recognizer_config = _module
test_online_endpoint = _module
test_timestamp = _module
test_utils = _module
client = _module
decode_manifest_triton = _module
generate_perf_input = _module
speech_client = _module
model = _module
model = _module
search = _module
model = _module
model = _module
model = _module
model = _module
model = _module
model = _module
conformer_triton = _module
export_jit = _module
export_onnx = _module
onnx_triton_utils = _module

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


import torch


import re


from typing import List


import math


from typing import Tuple


import logging


import warnings


from typing import Optional


import numpy as np


from torch.nn.utils.rnn import pad_sequence


from typing import Union


import torchaudio


from torch.utils.dlpack import to_dlpack


from torch.utils.dlpack import from_dlpack


import copy


from torch import Tensor


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


class Fbank(torch.nn.Module):

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(self, channels: int, kernel_size: int, bias: bool=True) ->None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = ScaledConv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.deriv_balancer1 = ActivationBalancer(channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0)
        self.depthwise_conv = ScaledConv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias)
        self.deriv_balancer2 = ActivationBalancer(channel_dim=1, min_positive=0.05, max_positive=1.0)
        self.activation = DoubleSwish()
        self.pointwise_conv2 = ScaledConv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, initial_scale=0.25)

    def forward(self, x: Tensor) ->Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        x = x.permute(1, 2, 0)
        x = self.pointwise_conv1(x)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.deriv_balancer2(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.permute(2, 0, 1)


class RelPositionMultiheadAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0) ->None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.in_proj = ScaledLinear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = ScaledLinear(embed_dim, embed_dim, bias=True, initial_scale=0.25)
        self.linear_pos = ScaledLinear(embed_dim, embed_dim, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_u_scale = nn.Parameter(torch.zeros(()).detach())
        self.pos_bias_v_scale = nn.Parameter(torch.zeros(()).detach())
        self._reset_parameters()

    def _pos_bias_u(self):
        return self.pos_bias_u * self.pos_bias_u_scale.exp()

    def _pos_bias_v(self):
        return self.pos_bias_v * self.pos_bias_v_scale.exp()

    def _reset_parameters(self) ->None:
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_emb: Tensor, key_padding_mask: Optional[Tensor]=None, need_weights: bool=True, attn_mask: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(query, key, value, pos_emb, self.embed_dim, self.num_heads, self.in_proj.get_weight(), self.in_proj.get_bias(), self.dropout, self.out_proj.get_weight(), self.out_proj.get_bias(), training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)

    def rel_shift(self, x: Tensor) ->Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        batch_size, num_heads, time1, n = x.shape
        assert n == 2 * time1 - 1
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided((batch_size, num_heads, time1, time1), (batch_stride, head_stride, time1_stride - n_stride, n_stride), storage_offset=n_stride * (time1 - 1))

    def multi_head_attention_forward(self, query: Tensor, key: Tensor, value: Tensor, pos_emb: Tensor, embed_dim_to_check: int, num_heads: int, in_proj_weight: Tensor, in_proj_bias: Tensor, dropout_p: float, out_proj_weight: Tensor, out_proj_bias: Tensor, training: bool=True, key_padding_mask: Optional[Tensor]=None, need_weights: bool=True, attn_mask: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated. Use bool tensor instead.')
                attn_mask = attn_mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for key_padding_mask is deprecated. Use bool tensor instead.')
            key_padding_mask = key_padding_mask
        q = (q * scaling).contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, '{} == {}'.format(key_padding_mask.size(0), bsz)
            assert key_padding_mask.size(1) == src_len, '{} == {}'.format(key_padding_mask.size(1), src_len)
        q = q.transpose(0, 1)
        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self._pos_bias_u()).transpose(1, 2)
        q_with_bias_v = (q + self._pos_bias_v()).transpose(1, 2)
        k = k.permute(1, 2, 3, 0)
        matrix_ac = torch.matmul(q_with_bias_u, k)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        attn_output_weights = matrix_ac + matrix_bd
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, layer_dropout: float=0.075, cnn_module_kernel: int=31) ->None:
        super(ConformerEncoderLayer, self).__init__()
        self.layer_dropout = layer_dropout
        self.d_model = d_model
        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)
        self.feed_forward = nn.Sequential(ScaledLinear(d_model, dim_feedforward), ActivationBalancer(channel_dim=-1), DoubleSwish(), nn.Dropout(dropout), ScaledLinear(dim_feedforward, d_model, initial_scale=0.25))
        self.feed_forward_macaron = nn.Sequential(ScaledLinear(d_model, dim_feedforward), ActivationBalancer(channel_dim=-1), DoubleSwish(), nn.Dropout(dropout), ScaledLinear(dim_feedforward, d_model, initial_scale=0.25))
        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)
        self.norm_final = BasicNorm(d_model)
        self.balancer = ActivationBalancer(channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, pos_emb: Tensor, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, warmup: float=1.0) ->Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            warmup: controls selective bypass of of layers; if < 1.0, we will
              bypass layers more frequently.

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        src_orig = src
        warmup_scale = min(0.1 + warmup, 1.0)
        if self.training:
            alpha = warmup_scale if torch.rand(()).item() <= 1.0 - self.layer_dropout else 0.1
        else:
            alpha = 1.0
        src = src + self.dropout(self.feed_forward_macaron(src))
        src_att = self.self_attn(src, src, src, pos_emb=pos_emb, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src_att)
        src = src + self.dropout(self.conv_module(src))
        src = src + self.dropout(self.feed_forward(src))
        src = self.norm_final(self.balancer(src))
        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig
        return src


class ConformerEncoder(nn.Module):
    """ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) ->None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor, pos_emb: Tensor, mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, warmup: float=1.0) ->Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, pos_emb, src_mask=mask, src_key_padding_mask=src_key_padding_mask, warmup=warmup)
        return output


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int=5000) ->None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) ->None:
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
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

    def forward(self, x: torch.Tensor) ->Tuple[Tensor, Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1:self.pe.size(1) // 2 + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, in_channels: int, out_channels: int, layer1_channels: int=8, layer2_channels: int=32, layer3_channels: int=128) ->None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()
        self.conv = nn.Sequential(ScaledConv2d(in_channels=1, out_channels=layer1_channels, kernel_size=3, padding=1), ActivationBalancer(channel_dim=1), DoubleSwish(), ScaledConv2d(in_channels=layer1_channels, out_channels=layer2_channels, kernel_size=3, stride=2), ActivationBalancer(channel_dim=1), DoubleSwish(), ScaledConv2d(in_channels=layer2_channels, out_channels=layer3_channels, kernel_size=3, stride=2), ActivationBalancer(channel_dim=1), DoubleSwish())
        self.out = ScaledLinear(layer3_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels)
        self.out_norm = BasicNorm(out_channels, learn_eps=False)
        self.out_balancer = ActivationBalancer(channel_dim=-1, min_positive=0.45, max_positive=0.55)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x


class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(self, vocab_size: int, decoder_dim: int, blank_id: int, context_size: int):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()
        self.embedding = ScaledEmbedding(num_embeddings=vocab_size, embedding_dim=decoder_dim, padding_idx=blank_id)
        self.blank_id = blank_id
        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(in_channels=decoder_dim, out_channels=decoder_dim, kernel_size=context_size, padding=0, groups=decoder_dim, bias=False)

    def forward(self, y: torch.Tensor) ->torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y
        embedding_out = self.embedding(y)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out


class Joiner(nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int, joiner_dim: int, vocab_size: int):
        super().__init__()
        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) ->torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim
            assert encoder_out.ndim in (2, 4)
            assert encoder_out.shape == decoder_out.shape
        logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        logit = self.output_linear(torch.tanh(logit))
        return logit


def make_pad_mask(lengths: torch.Tensor) ->torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = lengths.max()
    n = lengths.size(0)
    expaned_lengths = torch.arange(max_len).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(1)


class StreamingEncoder(torch.nn.Module):
    """
    Args:
          left_context:
            How many previous frames the attention can see in current chunk.
            Note: It's not that each individual frame has `left_context` frames
            of left context, some have more.
          right_context:
            How many future frames the attention can see in current chunk.
            Note: It's not that each individual frame has `right_context` frames
            of right context, some have more.
          chunk_size:
            The chunk size for decoding, this will be used to simulate streaming
            decoding using masking.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
    """

    def __init__(self, model, left_context, right_context, chunk_size, warmup):
        super().__init__()
        self.encoder = model.encoder
        self.encoder_embed = model.encoder_embed
        self.encoder_layers = model.encoder_layers
        self.d_model = model.d_model
        self.cnn_module_kernel = model.cnn_module_kernel
        self.encoder_pos = model.encoder_pos
        self.left_context = left_context
        self.right_context = right_context
        self.chunk_size = chunk_size
        self.warmup = warmup

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor, attn_cache: torch.tensor, cnn_cache: torch.tensor, processed_lens: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            The decode states for previous frames which contains the cached data.
            It has two elements, the first element is the attn_cache which has
            a shape of (encoder_layers, left_context, batch, attention_dim),
            the second element is the conv_cache which has a shape of
            (encoder_layers, cnn_module_kernel-1, batch, conv_dim).
            Note: states will be modified in this function.
          processed_lens:
            How many frames (after subsampling) have been processed for each sequence.

        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
            - decode_states, the updated states including the information
              of current chunk.
        """
        lengths = (x_lens - 1 >> 1) - 1 >> 1
        attn_cache = attn_cache.transpose(0, 2)
        cnn_cache = cnn_cache.transpose(0, 2)
        states = [attn_cache, cnn_cache]
        assert states is not None
        assert processed_lens is not None
        assert len(states) == 2 and states[0].shape == (self.encoder_layers, self.left_context, x.size(0), self.d_model) and states[1].shape == (self.encoder_layers, self.cnn_module_kernel - 1, x.size(0), self.d_model), f"""The length of states MUST be equal to 2, and the shape of
            first element should be {self.encoder_layers, self.left_context, x.size(0), self.d_model},
            given {states[0].shape}. the shape of second element should be
            {self.encoder_layers, self.cnn_module_kernel - 1, x.size(0), self.d_model},
            given {states[1].shape}."""
        lengths -= 2
        embed = self.encoder_embed(x)
        embed = embed[:, 1:-1, :]
        embed, pos_enc = self.encoder_pos(embed, self.left_context)
        embed = embed.permute(1, 0, 2)
        src_key_padding_mask = make_pad_mask(lengths, embed.size(0))
        processed_mask = torch.arange(self.left_context, device=x.device).expand(x.size(0), self.left_context)
        processed_mask = (processed_lens <= processed_mask).flip(1)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)
        x, states = self.encoder.chunk_forward(embed, pos_enc, src_key_padding_mask=src_key_padding_mask, warmup=self.warmup, states=states, left_context=self.left_context, right_context=self.right_context)
        if self.right_context > 0:
            x = x[:-self.right_context, ...]
            lengths -= self.right_context
        x = x.permute(1, 0, 2)
        processed_lens = processed_lens + lengths.unsqueeze(-1)
        assert processed_lens.shape[1] == 1, processed_lens.shape
        return x, lengths, states[0].transpose(0, 2), states[1].transpose(0, 2), processed_lens


class OfflineEncoder(torch.nn.Module):
    """
    Args:
        model: Conformer Encoder
    """

    def __init__(self, model) ->None:
        super().__init__()
        self.num_features = model.num_features
        self.subsampling_factor = model.subsampling_factor
        if self.subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        self.encoder_embed = model.encoder_embed
        self.encoder_layers = model.encoder_layers
        self.d_model = model.d_model
        self.cnn_module_kernel = model.cnn_module_kernel
        self.causal = model.causal
        self.dynamic_chunk_training = model.dynamic_chunk_training
        self.short_chunk_threshold = model.short_chunk_threshold
        self.short_chunk_size = model.short_chunk_size
        self.num_left_chunks = model.num_left_chunks
        self.encoder_pos = model.encoder_pos
        self.encoder = model.encoder

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, d_model)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        warmup = 1.0
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)
        lengths = (x_lens - 1 >> 1) - 1 >> 1
        if not is_jit_tracing():
            assert x.size(0) == lengths.max().item()
        src_key_padding_mask = make_pad_mask(lengths, x.size(0))
        if self.dynamic_chunk_training:
            assert self.causal, 'Causal convolution is required for streaming conformer.'
            max_len = x.size(0)
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > max_len * self.short_chunk_threshold:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % self.short_chunk_size + 1
            mask = ~subsequent_chunk_mask(size=x.size(0), chunk_size=chunk_size, num_left_chunks=self.num_left_chunks, device=x.device)
            x = self.encoder(x, pos_emb, mask=mask, src_key_padding_mask=src_key_padding_mask, warmup=warmup)
        else:
            x = self.encoder(x, pos_emb, mask=None, src_key_padding_mask=src_key_padding_mask, warmup=warmup)
        x = x.permute(1, 0, 2)
        return x, lengths


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_k2_fsa_sherpa(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

