import sys
_module = sys.modules[__name__]
del sys
language_modelling = _module
setup = _module
perceiver_io = _module
attention = _module
decoders = _module
encoder = _module
perceiver = _module
positional_encoding = _module

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


from typing import Optional


import torch


from torch import nn


from abc import ABCMeta


from abc import abstractmethod


import math


from typing import Sequence


class BasePerceiverDecoder(nn.Module, metaclass=ABCMeta):
    """Abstract decoder class."""

    @abstractmethod
    def forward(self, *, query: torch.Tensor, latents: torch.Tensor, q_mask: Optional[torch.Tensor]=None):
        return NotImplementedError


class FeedForward(nn.Module):
    """Transformer Feed-Forward network."""

    def __init__(self, dim: int, widening_factor: int=4, dropout: float=0.0):
        """Constructor.

        Args:
            dim: Dimension of input tensor.
            widening_factor: Widening factor. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim * widening_factor), nn.GELU(), nn.Linear(dim * widening_factor, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(self, kv_dim: int, q_dim: int, *, qk_out_dim: Optional[int]=None, v_out_dim: Optional[int]=None, output_dim: Optional[int]=None, num_heads: int=1, dropout: float=0.0):
        """Constructor.

        Args:
            kv_dim: Size of input key and value vectors.
            q_dim: Size of input query vector.
            qk_out_dim: Size of Query and Key matrices last dimension.
                If None, it will be equal to q_dim. Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                If None, it will be equal to qk_out_dim. Defaults to None.
            output_dim: Size of output after the QKV attention.
                If none, it will be equal to v_out_dim. Defaults to None.
            num_heads: Number of heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim
        self.num_heads = num_heads
        self.qk_head_dim = qk_out_dim // num_heads
        self.v_head_dim = v_out_dim // num_heads
        self.k = nn.Linear(kv_dim, qk_out_dim)
        self.q = nn.Linear(q_dim, qk_out_dim)
        self.v = nn.Linear(kv_dim, v_out_dim)
        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.qk_head_dim ** -0.5

    def transform_for_scores(self, x: torch.Tensor, head_dim: int):
        *dims, seq, hid = x.size()
        x = x.view(*dims, seq, self.num_heads, head_dim)
        return x.transpose(-3, -2)

    def forward(self, inputs_kv: torch.Tensor, inputs_q: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, ..., M, C).
            inputs_q: Query embeddings of shape (B, ..., N, D)
            attention_mask: Tensor of shape (B, ..., N, M).

        Returns:
            Tensor of shape (B, ..., N, D)
        """
        keys, queries, values = self.k(inputs_kv), self.q(inputs_q), self.v(inputs_kv)
        keys = self.transform_for_scores(keys, self.qk_head_dim)
        queries = self.transform_for_scores(queries, self.qk_head_dim)
        values = self.transform_for_scores(values, self.v_head_dim)
        attention = queries @ keys.transpose(-2, -1) * self.scale
        if attention_mask is not None:
            min_value = torch.finfo(attention.dtype).min
            extended_mask = (1 - attention_mask) * min_value
            attention = attention + extended_mask
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        if attention_mask is not None:
            attention = attention.masked_fill(1 - attention_mask, value=0)
        weighted = attention @ values
        *dims, n_heads, seq, hid = weighted.size()
        weighted = weighted.transpose(-3, -2)
        weighted = weighted.reshape(*dims, seq, n_heads * hid)
        return self.projection(weighted)


class CrossAttention(nn.Module):
    """Cross-attention module."""

    def __init__(self, *, kv_dim: int, q_dim: int, qk_out_dim: Optional[int]=None, v_out_dim: Optional[int]=None, widening_factor: int=1, num_heads: int=1, use_query_residual: bool=True, dropout: float=0.0, attention_dropout: float=0.0):
        """Constructor.

        Args:
            kv_dim: Dimension of key/value input tensor.
            q_dim: Dimension of query input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.use_query_residual = use_query_residual
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(kv_dim=kv_dim, q_dim=q_dim, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, output_dim=q_dim, num_heads=num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(q_dim, widening_factor, dropout)

    def forward(self, inputs_kv: torch.Tensor, inputs_q: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, ..., M, C).
            inputs_q: Query embeddings of shape (B, ..., N, D)
            attention_mask: Tensor of shape (B, ..., N, M). Mask values selected
                in [0, 1]. Defaults to None.
        """
        attention = self.attention(inputs_kv=self.kv_layer_norm(inputs_kv), inputs_q=self.q_layer_norm(inputs_q), attention_mask=attention_mask)
        attention = self.dropout(attention)
        if self.use_query_residual:
            x = inputs_q + attention
        else:
            x = attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class PerceiverDecoder(BasePerceiverDecoder):
    """Basic cross-attention decoder."""

    def __init__(self, latent_dim: int, query_dim: int, widening_factor: int=1, num_heads: int=1, qk_out_dim: Optional[int]=None, v_out_dim: Optional[int]=None, projection_dim: Optional[int]=None, use_query_residual: bool=False):
        super().__init__()
        self.cross_attention = CrossAttention(kv_dim=latent_dim, q_dim=query_dim, widening_factor=widening_factor, num_heads=num_heads, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, use_query_residual=use_query_residual)
        if projection_dim is not None:
            self.projection = nn.Linear(query_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, *, query: torch.Tensor, latents: torch.Tensor, q_mask: Optional[torch.Tensor]=None):
        if q_mask is not None:
            q_mask = q_mask[:, None, None, :].transpose(-2, -1)
        outputs = self.cross_attention(inputs_kv=latents, inputs_q=query, attention_mask=q_mask)
        return self.projection(outputs)


class SelfAttention(nn.Module):
    """Self-attention module."""

    def __init__(self, *, hidden_dim: int, qk_out_dim: Optional[int]=None, v_out_dim: Optional[int]=None, widening_factor: int=4, num_heads: int=1, dropout: float=0.0, attention_dropout: float=0.0):
        """Constructor.

        Args:
            hidden_dim: Dimension of input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(kv_dim=hidden_dim, q_dim=hidden_dim, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, output_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(hidden_dim, widening_factor, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        """
        Args:
            x: Input tensor of shape (B, ..., M, C).
            attention_mask: Input mask tensor of shape (B, ..., M, M).
                Mask values selected in [0, 1]. Defaults to None.
        """
        x_norm = self.layer_norm(x)
        attention = self.attention(inputs_kv=x_norm, inputs_q=x_norm, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x + attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class PerceiverEncoder(nn.Module):
    """Perceiver encoder module. Consists of two components: cross-attention
    module that maps an input tensor and a trainable latent tensor to a latent
    tensor and a stacked Transformer blocks with shared weights.
    """

    def __init__(self, num_latents: int, latent_dim: int, input_dim: int, num_self_attn_per_block: int=2, num_blocks: int=4, qk_out_dim: Optional[int]=None, v_out_dim: Optional[int]=None, num_cross_attn_heads: int=1, num_self_attn_heads: int=8, cross_attn_widening_factor: int=1, self_attn_widening_factor: int=1, use_query_residual: bool=True, dropout: float=0.0, cross_attention_dropout: float=0.0, self_attention_dropout: float=0.0):
        """Constructor.

        Args:
            num_latents: Number of latent vectors.
            latent_dim: Dimension of latent vector.
            input_dim: Dimension of input tensor.
            num_self_attn_per_block: Number of self-attention modules per
                transformer block. Defaults to 2.
            num_blocks: Number of transformer blocks. Defaults to 4.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            num_cross_attn_heads: Number of cross-attention heads.
                Defaults to 1.
            num_self_attn_heads: Number of self-attention heads.
                Defaults to 8.
            cross_attn_widening_factor: Widening factor in cross-attention
                feed-forward layer. Defaults to 1.
            self_attn_widening_factor: Widening factor in self-attention
                feed-forward layer. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
            dropout: Feed-forward dropout probability. Defaults to 0.
            cross_attention_dropout: Cross-attention scores dropout probability.
                Defaults to 0.
            self_attention_dropout: Self-attention scores dropout probability.
                Defaults to 0.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttention(kv_dim=input_dim, q_dim=latent_dim, widening_factor=cross_attn_widening_factor, num_heads=num_cross_attn_heads, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, use_query_residual=use_query_residual, dropout=dropout, attention_dropout=cross_attention_dropout)
        self.self_attention_block = nn.ModuleList([SelfAttention(hidden_dim=latent_dim, widening_factor=self_attn_widening_factor, num_heads=num_self_attn_heads, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, dropout=dropout, attention_dropout=self_attention_dropout) for _ in range(num_self_attn_per_block)])

    def forward(self, x: torch.Tensor, kv_mask: Optional[torch.Tensor]=None):
        """
        Args:
            x: Input tensor of shape (B, M, C).
            kv_mask: Input mask tensor of shape (B, M). Mask values selected
                in [0, 1]. Defaults to None.

        Returns:
            Latent tensor.
        """
        batch_size = x.size(0)
        if kv_mask is not None:
            kv_mask = kv_mask[:, None, None, :]
        latents = self.cross_attn(inputs_kv=x, inputs_q=self.latents.repeat(batch_size, 1, 1), attention_mask=kv_mask)
        for _ in range(self.num_blocks):
            for self_attn_layer in self.self_attention_block:
                latents = self_attn_layer(latents)
        return latents


class PerceiverIO(nn.Module):
    """Perceiver IO encoder-decoder architecture."""

    def __init__(self, encoder: PerceiverEncoder, decoder: BasePerceiverDecoder):
        """Constructor.

        Args:
            encoder: Instance of Perceiver IO encoder.
            decoder: Instance of Perceiver IO decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor, query: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, query_mask: Optional[torch.Tensor]=None):
        """
        Args:
            inputs: Input tensor.
            query: Decoder query tensor. Can be a trainable or hand-made.
                Defaults to None.
            input_mask: Input mask tensor. Mask values selected in [0, 1].
                Defaults to None.
            query_mask: Decoder query mask tensor. Mask values selected in
                [0, 1]. Defaults to None.

        Returns:
            Output tensor.
        """
        latents = self.encoder(inputs, kv_mask=input_mask)
        outputs = self.decoder(query=query, latents=latents, q_mask=query_mask)
        return outputs


class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""

    def __init__(self, vocab_size: int, max_seq_len: int, embedding_dim: int, num_latents: int=256, latent_dim: int=512, num_self_attn_heads=8, self_attn_head_dim=None, cross_attn_head_dim=None, self_attn_widening_factor=1, cross_attn_widening_factor=1, num_blocks=1, num_self_attn_per_block=12, dropout: float=0.0):
        """Constructor.

        Args:
            vocab_size: Size of vocabulary.
            max_seq_len: Maximum length of token sequence.
            embedding_dim: Dimension of token embedding.
            num_latents: Number of latent vectors. Defaults to 256.
            latent_dim: Dimension of latent vector. Defaults to 512.
            num_self_attn_heads: Number of self-attention heads. Defaults to 8.
            self_attn_head_dim: Size of self-attention head. If None,this
                value will be calculated as latent_dim / num_self_attn_heads.
                Defaults to None.
            cross_attn_head_dim: Size of cross-attention head. If None,this
                value will be equal latent_dims. Defaults to None.
            self_attn_widening_factor: Widening factor in self-attention
                feed-forward layer. Defaults to 1.
            cross_attn_widening_factor: Widening factor in cross-attention
                feed-forward layer. Defaults to 1.
            num_blocks: Number of transformer blocks. Defaults to 1.
            num_self_attn_per_block: Number of self-attention modules per
                transformer block. Defaults to 12.
            dropout: Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder = PerceiverEncoder(num_latents=num_latents, latent_dim=latent_dim, input_dim=embedding_dim, num_self_attn_per_block=num_self_attn_per_block, num_blocks=num_blocks, cross_attn_head_dim=cross_attn_head_dim, self_attn_head_dim=self_attn_head_dim, num_self_attn_heads=num_self_attn_heads, cross_attn_widening_factor=cross_attn_widening_factor, self_attn_widening_factor=self_attn_widening_factor, dropout=dropout)
        decoder = PerceiverDecoder(latent_dim=latent_dim, query_dim=embedding_dim, widening_factor=cross_attn_widening_factor, projection_dim=vocab_size)
        self.perceiver = PerceiverIO(encoder, decoder)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None):
        """
        Args:
            inputs: Tensor of token ids.
            mask: Token mask. Mask values selected in [0, 1]. Defaults to None.

        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size).
        """
        seq_len = inputs.size(1)
        token_embeddings = self.token_embedding(inputs)
        positions_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)
        position_embeddings = self.position_embedding(positions_ids)
        embeddings = token_embeddings + position_embeddings
        outputs = self.perceiver(inputs=embeddings, query=position_embeddings, input_mask=mask, query_mask=mask)
        return outputs


class ProjectionDecoder(BasePerceiverDecoder):
    """Projection decoder without using a cross-attention layer."""

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.projection = nn.Linear(latent_dim, num_classes)

    def forward(self, *, query: torch.Tensor, latents: torch.Tensor, q_mask: Optional[torch.Tensor]=None):
        latents = latents.mean(dim=1)
        logits = self.projection(latents)
        return logits


class ClassificationDecoder(BasePerceiverDecoder):
    """Classification decoder. Based on PerceiverDecoder."""

    def __init__(self, num_classes: int, latent_dim: int, widening_factor: int=1, num_heads: int=1, head_dim: Optional[int]=None):
        super().__init__()
        self.task_ids = nn.Parameter(torch.randn(1, num_classes))
        self.decoder = PerceiverDecoder(latent_dim=latent_dim, query_dim=num_classes, widening_factor=widening_factor, num_heads=num_heads, head_dim=head_dim, projection_dim=None, use_query_residual=False)

    def forward(self, *, query: torch.Tensor, latents: torch.Tensor, q_mask: Optional[torch.Tensor]=None):
        batch_size = latents.size(0)
        logits = self.decoder.forward(query=self.task_ids.repeat(batch_size, 1, 1), latents=latents, q_mask=q_mask)
        return logits.squeeze(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CrossAttention,
     lambda: ([], {'kv_dim': 4, 'q_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'kv_dim': 4, 'q_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_esceptico_perceiver_io(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

