import sys
_module = sys.modules[__name__]
del sys
download = _module
feedly_data = _module
conf = _module
bert = _module
runner = _module
cbow = _module
mlp_parameter_tuning = _module
news = _module
surnames = _module
training = _module
dataset = _module
lm_tuner_runner = _module
model = _module
model = _module
utils = _module
setup = _module
test_vocabulary = _module
plugins = _module
test_config = _module
test_trainer = _module
trainer_utils = _module
test_config_loader = _module
test_experiment_runner = _module
transfer_nlp = _module
common = _module
tokenizers = _module
embeddings = _module
embeddings = _module
loaders = _module
vectorizers = _module
vocabulary = _module
config = _module
helpers = _module
metrics = _module
predictors = _module
regularizers = _module
reporters = _module
trainer_abc = _module
trainers = _module
experiment_runner = _module

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


import logging


from typing import Dict


from typing import List


from typing import Any


import numpy as np


import torch


import string


from collections import Counter


from typing import Tuple


import torch.nn as nn


import torch.optim as optim


import copy


from typing import Union


import inspect


from itertools import zip_longest


import re


from abc import abstractmethod


from collections import defaultdict


TQDM = True


logger = logging.getLogger(__name__)


REGISTRY = {}


def register_plugin(registrable: Any, alias: str=None):
    """
    Register a class, a function or a method to REGISTRY
    Args:
        registrable:
        alias:
    Returns:
    """
    alias = alias or registrable.__name__
    if alias in REGISTRY:
        raise ValueError(
            f'{alias} is already registered to registrable {REGISTRY[alias]}. Please select another name'
            )
    REGISTRY[alias] = registrable
    return registrable


class ElmanRNN(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool
        =False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell: torch.nn.RNNCell = torch.nn.RNNCell(input_size,
            hidden_size)
        self.batch_first: bool = batch_first
        self.hidden_size: int = hidden_size

    def _initial_hidden(self, batch_size: int) ->torch.tensor:
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in: torch.Tensor, initial_hidden: torch.Tensor=None
        ) ->torch.Tensor:
        """

        :param x_in: an input data tensor.
                If self.batch_first: x_in.shape = (batch, seq_size, feat_size)
                Else: x_in.shape = (seq_size, batch, feat_size)
        :param initial_hidden: the initial hidden state for the RNN
        :return: The outputs of the RNN at each time step.
                If self.batch_first: hiddens.shape = (batch, seq_size, hidden_size)
                Else: hiddens.shape = (seq_size, batch, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()
        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens


def column_gather(y_out: torch.FloatTensor, x_lengths: torch.LongTensor
    ) ->torch.FloatTensor:
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)


@register_plugin
class Transformer(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_embeddings: int,
        num_max_positions: int, num_heads: int, num_layers: int, dropout:
        float, causal: bool):
        super().__init__()
        self.causal: bool = causal
        self.tokens_embeddings: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings, embed_dim)
        self.position_embeddings: torch.nn.Embedding = torch.nn.Embedding(
            num_max_positions, embed_dim)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.attentions, self.feed_forwards = torch.nn.ModuleList(
            ), torch.nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = torch.nn.ModuleList(
            ), torch.nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(torch.nn.MultiheadAttention(embed_dim,
                num_heads, dropout=dropout))
            self.feed_forwards.append(torch.nn.Sequential(torch.nn.Linear(
                embed_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(
                hidden_dim, embed_dim)))
            self.layer_norms_1.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))
        self.attn_mask = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
            do_lower_case=False)

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        padding_mask = x == self.tokenizer.vocab['[PAD]']
        x = x.transpose(0, 1).contiguous()
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=
                h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self
            .layer_norms_1, self.attentions, self.layer_norms_2, self.
            feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=
                False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


@register_plugin
class TransformerWithLMHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions:
        int, num_heads: int, num_layers: int, dropout: float, causal: bool,
        initializer_range: float):
        """ Transformer with a language modeling head on top (tied weights) """
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
            do_lower_case=False)
        num_embeddings = len(tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim,
            num_embeddings, num_max_positions, num_heads, num_layers,
            dropout, causal=causal)
        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        """ initialize weights - nn.MultiheadAttention is already initalized by PyTorch (xavier) """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.
            nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)
            ) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)
        return logits


@register_plugin
class TransformerWithClfHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions:
        int, num_heads: int, num_layers: int, dropout: float, causal: bool,
        initializer_range: float, num_classes: int):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
            do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim,
            num_embeddings, num_max_positions, num_heads, num_layers,
            dropout, causal=causal)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.
            nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)
            ) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        clf_tokens_mask = x.transpose(0, 1).contiguous(
            ) == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        msk = clf_tokens_mask.unsqueeze(-1).float()
        clf_tokens_states = (hidden_states * msk).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return clf_logits


class TransformerWithAdapters(Transformer):

    def __init__(self, adapters_dim, embed_dim, hidden_dim, num_embeddings,
        num_max_positions, num_heads, num_layers, dropout, causal):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings,
            num_max_positions, num_heads, num_layers, dropout, causal)
        self.adapters_1 = torch.nn.ModuleList()
        self.adapters_2 = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(torch.nn.Sequential(torch.nn.Linear(
                embed_dim, adapters_dim), torch.nn.ReLU(), torch.nn.Linear(
                adapters_dim, embed_dim)))
            self.adapters_2.append(torch.nn.Sequential(torch.nn.Linear(
                embed_dim, adapters_dim), torch.nn.ReLU(), torch.nn.Linear(
                adapters_dim, embed_dim)))

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        padding_mask = x == self.tokenizer.vocab['[PAD]']
        x = x.transpose(0, 1).contiguous()
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=
                h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, adapter_1, layer_norm_2, feed_forward, adapter_2 in zip(
            self.layer_norms_1, self.attentions, self.adapters_1, self.
            layer_norms_2, self.feed_forwards, self.adapters_2):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=
                False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            x = adapter_1(x) + x
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            x = adapter_2(x) + x
            h = x + h
        return h


@register_plugin
class TransformerWithClfHeadAndAdapters(torch.nn.Module):

    def __init__(self, adapters_dim: int, embed_dim: int, hidden_dim: int,
        num_max_positions: int, num_heads: int, num_layers: int, dropout:
        float, causal: bool, initializer_range: float, num_classes: int):
        """ Transformer with a classification head and adapters. """
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
            do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer: TransformerWithAdapters = TransformerWithAdapters(
            adapters_dim, embed_dim, hidden_dim, num_embeddings,
            num_max_positions, num_heads, num_layers, dropout, causal=causal)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.
            nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)
            ) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        clf_tokens_mask = x.transpose(0, 1).contiguous(
            ) == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).
            float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return clf_logits


@register_plugin
class TransformerWithClfHeadAndLMHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions:
        int, num_heads: int, num_layers: int, dropout: float, causal: bool,
        initializer_range: float, num_classes: int):
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
            do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer = Transformer(embed_dim, hidden_dim,
            num_embeddings, num_max_positions, num_heads, num_layers,
            dropout, causal=causal)
        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.
            nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)
            ) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x and clf_tokens_mask have shape [seq length, batch] padding_mask has shape [batch, seq length] """
        clf_tokens_mask = x.transpose(0, 1).contiguous(
            ) == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        lm_logits = self.lm_head(hidden_states)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).
            float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return lm_logits, clf_logits


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_feedly_transfer_nlp(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ElmanRNN(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

