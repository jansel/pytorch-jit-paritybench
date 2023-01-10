import sys
_module = sys.modules[__name__]
del sys
classifiers = _module
tree2tabular = _module
explainer = _module
plotter = _module
predictor = _module
train_fasttext = _module
train_flair = _module
train_transformer = _module
loader = _module
model = _module

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


import pandas as pd


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


import numpy as np


import scipy


import sklearn.pipeline


from typing import List


from typing import Any


import torch


from torch.utils.data import TensorDataset


from torch.utils.data import random_split


from torch.utils.data import DataLoader


from typing import Tuple


import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions, self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithAdapters(Transformer):

    def __init__(self, adapters_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal)
        self.adapters_1 = nn.ModuleList()
        self.adapters_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(nn.Sequential(nn.Linear(embed_dim, adapters_dim), nn.ReLU(), nn.Linear(adapters_dim, embed_dim)))
            self.adapters_2.append(nn.Sequential(nn.Linear(embed_dim, adapters_dim), nn.ReLU(), nn.Linear(adapters_dim, embed_dim)))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, adapter_1, layer_norm_2, feed_forward, adapter_2 in zip(self.layer_norms_1, self.attentions, self.adapters_1, self.layer_norms_2, self.feed_forwards, self.adapters_2):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            x = adapter_1(x) + x
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            x = adapter_2(x) + x
            h = x + h
        return h


class TransformerWithClfHeadAndAdapters(nn.Module):

    def __init__(self, config, fine_tuning_config):
        """ Transformer with a classification head and adapters. """
        super().__init__()
        self.config = fine_tuning_config
        if fine_tuning_config.adapters_dim > 0:
            self.transformer = TransformerWithAdapters(fine_tuning_config.adapters_dim, config.embed_dim, config.hidden_dim, config.num_embeddings, config.num_max_positions, config.num_heads, config.num_layers, fine_tuning_config.dropout, causal=not config.mlm)
        else:
            self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings, config.num_max_positions, config.num_heads, config.num_layers, fine_tuning_config.dropout, causal=not config.mlm)
        self.classification_head = nn.Linear(config.embed_dim, fine_tuning_config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, clf_tokens_mask, lm_labels=None, clf_labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))
            return clf_logits, loss
        return clf_logits

