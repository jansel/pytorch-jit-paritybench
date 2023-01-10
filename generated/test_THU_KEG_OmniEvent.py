import sys
_module = sys.modules[__name__]
del sys
OmniEvent = _module
aggregation = _module
aggregation = _module
arguments = _module
backbone = _module
backbone = _module
evaluation = _module
convert_format = _module
dump_result = _module
metric = _module
utils = _module
head = _module
classification = _module
crf = _module
infer = _module
infer_module = _module
io_format = _module
seq2seq = _module
input_engineering = _module
base_processor = _module
input_utils = _module
mrc_converter = _module
mrc_processor = _module
seq2seq_processor = _module
sequence_labeling_processor = _module
token_classification_processor = _module
tokenizer = _module
whitespace_tokenizer = _module
model = _module
constraint_decoding = _module
label_smoother_sum = _module
model = _module
trainer = _module
trainer_seq2seq = _module
conf = _module
beam_search = _module
convert = _module
dump_ckpt = _module
train = _module
mrc = _module
sequence_labeling = _module
token_classification = _module
add_mention_id = _module
convert_to_openee = _module
parse_ace_event = _module
process_ace = _module
merge_en_eae = _module
merge_en_ed = _module
duee = _module
LDC2015E29 = _module
LDC2015E68 = _module
LDC2015E78 = _module
fewfc = _module
generate_mrc_prompt = _module
kbp2016 = _module
kbp2017 = _module
leven = _module
maven = _module
richere = _module
test_infer = _module
finetune_bert = _module
finetune_cpm1 = _module
pretrain_cpm1 = _module
finetune_cpm2 = _module
pretrain_cpm2 = _module
finetune_gpt2 = _module
finetune_gptj = _module
finetune_mt5 = _module
finetune_t5 = _module
model_center = _module
dataset = _module
bertdataset = _module
superglue = _module
cpm1 = _module
cpm1_dataset = _module
cpm1dataset = _module
down_data = _module
cpm2 = _module
dataset = _module
cpm2dataset = _module
down_data = _module
distributed_dataset = _module
distributed_indexed = _module
distributed_loader = _module
gpt2dataset = _module
superglue = _module
indexed = _module
t5dataset = _module
superglue = _module
layer = _module
attention = _module
blocks = _module
conv = _module
embedding = _module
feedforward = _module
layernorm = _module
linear = _module
position_embedding = _module
transformer = _module
basemodel = _module
bert = _module
config = _module
bert_config = _module
cpm1_config = _module
cpm2_config = _module
cpm3_config = _module
glm_config = _module
gpt2_config = _module
gptj_config = _module
longformer_config = _module
roberta_config = _module
t5_config = _module
vit_config = _module
cpm1 = _module
cpm2 = _module
cpm3 = _module
glm = _module
gpt2 = _module
gptj = _module
longformer = _module
roberta = _module
t5 = _module
vit = _module
base_tokenizer = _module
bert_tokenizer = _module
cpm1_tokenizer = _module
cpm2_tokenizer = _module
glm_tokenizer = _module
gpt2_tokenizer = _module
gptj_tokenizer = _module
roberta_tokenizer = _module
t5_tokenizer = _module
indexed_dataset = _module
preprocess_cpm1_lm = _module
net_utils = _module
print_utils = _module
setup = _module
test_bert = _module
test_bert_pkv = _module
test_glm = _module
test_gpt2 = _module
test_gpt_pkv = _module
test_gptj = _module
test_longformer = _module
test_mt5 = _module
test_roberta = _module
test_t5 = _module
test_t5v1_1 = _module
test_vit = _module
hugGPT2_bmtrainGPT2 = _module
hugGPTj_bmtrainGPTj = _module
hugLongformer_bmtrainLongformer = _module
hugMT5_bmtrainMT5 = _module
hugRoBERTa_bmtrainRoBERTa = _module
hugT5_bmtrainT5 = _module
hugT5v1_1_bmtrainT5v1_1 = _module

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


import torch.nn as nn


import torch.nn.functional as F


from typing import Optional


import numpy as np


from typing import List


from typing import Tuple


from typing import Union


import copy


from sklearn.metrics import f1_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from typing import Dict


import torch.cuda


import re


from collections import defaultdict


import logging


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import Any


from torch import nn


from abc import ABC


from copy import deepcopy


from collections import OrderedDict


import time


import random


from sklearn.metrics import accuracy_score


from torch.utils.tensorboard import SummaryWriter


from abc import abstractclassmethod


import torch.utils.data as data


import string


from itertools import accumulate


import math


from typing import *


import collections


from itertools import repeat


from functools import lru_cache


class DynamicPooling(nn.Module):
    """Dynamic multi-pooling layer for Convolutional Neural Network (CNN).

    Dynamic multi-pooling layer for Convolutional Neural Network (CNN), which is able to capture more valuable
    information within a sentence, particularly for some cases, such as multiple triggers are within a sentence and
    different argument candidate may play a different role with a different trigger.

    Attributes:
        dense (`nn.Linear`):
            TODO: The purpose of the linear layer should be configured.
        activation (`nn.Tanh`):
            An `nn.Tanh` layer representing the tanh activation function.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the default dropout rate (0.5).
    """

    def __init__(self, config) ->None:
        """Constructs a `DynamicPooling`."""
        super(DynamicPooling, self).__init__()
        self.dense = nn.Linear(config.hidden_size * config.head_scale, config.hidden_size * config.head_scale)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout()

    def get_mask(self, position: torch.Tensor, batch_size: int, seq_length: int, device: str) ->torch.Tensor:
        """Returns the mask indicating whether the token is padded or not."""
        all_masks = []
        for i in range(batch_size):
            mask = torch.zeros(seq_length, dtype=torch.int16, device=device)
            mask[:int(position[i])] = 1
            all_masks.append(mask)
        all_masks = torch.stack(all_masks, dim=0)
        return all_masks

    def get_lexical_level_features(self, embeddings, position, max_seq_length):
        llf_idx = torch.stack([position - 1, position, position + 1], dim=0)
        llf_idx[0] = llf_idx[0] * (llf_idx[0] != -1) + (position + 2) * (llf_idx[0] == -1)
        llf_idx[2] = llf_idx[2] * (llf_idx[2] != max_seq_length) + (position - 2) * (llf_idx[2] == max_seq_length)
        features = []
        for i in range(3):
            features.append(embeddings[torch.arange(embeddings.shape[0]), llf_idx[i]])
        features = torch.cat(features, dim=-1)
        return features

    def get_argument_lexical_features(self, embeddings, start, end, max_seq_length):
        mid_features = []
        for i in range(start.shape[0]):
            mid_features.append(torch.mean(embeddings[i, start[i]:end[i] + 1], dim=0))
        mid_features = torch.stack(mid_features, dim=0)
        llf_idx = torch.stack([start - 1, end + 1], dim=0)
        llf_idx[0] = llf_idx[0] * (llf_idx[0] != -1) + (end + 2) * (llf_idx[0] == -1)
        llf_idx[1] = llf_idx[1] * (llf_idx[1] != max_seq_length) + (start - 2) * (llf_idx[1] == max_seq_length)
        features = [mid_features]
        for i in range(2):
            features.append(embeddings[torch.arange(embeddings.shape[0]), llf_idx[i]])
        features = torch.cat(features, dim=-1)
        return features

    def max_pooling(self, hidden_states: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        """Conducts the max-pooling operation on the hidden states."""
        batch_size, seq_length, hidden_size = hidden_states.size()
        mask = mask.unsqueeze(2)
        states = hidden_states * mask + mask * 100
        pooled_states = torch.max(states, dim=1)[0]
        pooled_states -= 100
        return pooled_states

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, trigger_position: torch.Tensor, embeddings: Optional[torch.Tensor]=None, argument_left: Optional[torch.Tensor]=None, argument_right: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Conducts the dynamic multi-pooling process on the hidden states."""
        batch_size, seq_length = hidden_states.size()[:2]
        trigger_mask = self.get_mask(trigger_position, batch_size, seq_length, hidden_states.device)
        if embeddings is not None:
            lexical_features = self.get_lexical_level_features(embeddings, trigger_position, hidden_states.size(1))
        if argument_left is not None:
            if embeddings is not None:
                lexical_features = self.get_argument_lexical_features(embeddings, argument_left, argument_right, hidden_states.size(1))
            argument_mask = self.get_mask(argument_left, batch_size, seq_length, hidden_states.device)
            left_mask = torch.logical_and(trigger_mask, argument_mask) * attention_mask
            middle_mask = torch.logical_xor(trigger_mask, argument_mask) * attention_mask
            right_mask = (1 - torch.logical_or(trigger_mask, argument_mask)) * attention_mask
            left_states = self.max_pooling(hidden_states, left_mask)
            middle_states = self.max_pooling(hidden_states, middle_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, middle_states, right_states), dim=-1)
        else:
            left_mask = trigger_mask * attention_mask
            right_mask = (1 - left_mask) * attention_mask
            left_states = self.max_pooling(hidden_states, left_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, right_states), dim=-1)
        if embeddings is not None:
            final_output = torch.cat([pooled_output, lexical_features], dim=-1)
        else:
            final_output = pooled_output
        final_output = self.dropout(final_output)
        return final_output


class GraphAttentionLayer(nn.Module):
    """Simple graph attention layer.

    A simple graph attention layer for the aggregation process, which is the sole layer throughout all of the GAT
    architectures, performing self-attention on all nodes, and aggregating the information based on the importance of
    neighbors of each node.

    Attributes:
        dropout (`int`):
            An integer indicating the dropout rate.
        in_features (`int`):
            An integer indicating the dimension of the input features.
        out_features (`int`):
            An integer indicating the dimension of the output features.
        alpha (`float`):
            A float variable indicating the negative slope of the leaky relu activation.
        W (`nn.Parameter`):
            An `nn.Parameter` instance representing the weight of the fully connecting layer, transforming high-
            dimensional features into low dimensions.
        a (`nn.Parameter`):
            An `nn.Parameter` instance indicating the initial attention weight between nodes.
        leaky_relu (`nn.LeakyReLU`):
            An `nn.LeakyReLU` layer representing the leaky relu activation function.
    """

    def __init__(self, in_features: int, out_features: int, dropout: int, alpha: int, device, concat: Optional[bool]=False):
        """Constructs a `GraphAttentionLayer`."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, input):
        """The forward propagation of a simple graph attention layer."""
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


def matmuls(a: torch.Tensor, times: int) ->torch.Tensor:
    """Multiplies the input matrix with itself for `times` times.

    Multiplies the input matrix with itself multiple times, in which each time of multiplication follows matrix-matrix
    multiplication.

    Args:
        a (`torch.Tensor`):
            A tensor representing the input matrix for multiplication.
        times (`int`):
            An integer indicating the number of times the matrix would be multiplied with itself.

    Returns:
        res (`torch.Tensor`):
            A tensor representing the matrix after `times` times multiplication of the given matrix.
    """
    res = a
    for i in range(times):
        res = torch.matmul(res, a)
    return res


class MOGCN(nn.Module):
    """Multi-order Graph Convolutional Network (MOGAN).

    A Multi-order Graph Convolutional Network (MOGAN) class, which simply learns a list of representations over
    multi-order syntactic graphs by a few parallel Graph Attention Network (GAT) layers, which weights the importance of
    neighbors of each word in each syntactic graph during convolution.

    Attributes:
        in_dim (`int`):
            An integer indicating the dimension of GAT's input features.
        hidden_dim (`int`):
            An integer indicating the dimension of GAT's output features.
        device:
            The device of the operation, CPU or GPU.
            TODO: Configure the data type of the `device` variable.
        in_drop (`int`):
            An integer indicating the dropout rate.
        K (`int`):
            An integer indicating the number of times operating the graph attention convolution process.
        layers_a (`nn.ModuleList`):
            A GAT layer operating the first sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_along, containing the connection information of the first-order syntactic graph.
        layers_b (`nn.ModuleList`):
            A GAT layer operating the second sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_rev, which is a transpose matrix of A_along.
        layers_c (`nn.ModuleList`):
            A GAT layer operating the third sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_loop, which is an identity matrix.
        Wawa (`nn.Sequential`):
            An `nn.Sequential` container with a linear transformation and a tanh activation function, which is regarded
            as a graph attention convolutional function.
        Ctx (`nn.Linear`):
            An `nn.Linear` layer for computing the normalized weight of each neighbor when updating a node.
    """

    def __init__(self, in_dim: int, hidden_dim: int, K: int, dropout: int, device, alpha: Optional[int]=0.2) ->None:
        """Constructs a `MOGCN`."""
        super(MOGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.in_drop = dropout
        self.K = K
        self.layers_a = nn.ModuleList()
        self.layers_b = nn.ModuleList()
        self.layers_c = nn.ModuleList()
        for i in range(self.K):
            self.layers_a.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim, dropout=dropout, alpha=alpha, device=device, concat=False))
            self.layers_b.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim, dropout=dropout, alpha=alpha, device=device, concat=False))
            self.layers_c.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim, dropout=dropout, alpha=alpha, device=device, concat=False))
        self.Wawa = nn.Sequential(nn.Linear(self.hidden_dim, 100), nn.Tanh())
        self.Ctx = nn.Linear(100, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, adj: torch.Tensor) ->torch.Tensor:
        """The forward propagation of `NOGCN`."""
        adj_a, adj_b, adj_c = adj[:, 0, :, :], adj[:, 1, :, :], adj[:, 2, :, :]
        hs = []
        for layer in range(self.K):
            h_layer = self.layers_a[layer](matmuls(adj_a, layer), hidden_states) + self.layers_b[layer](matmuls(adj_b, layer), hidden_states) + self.layers_c[layer](matmuls(adj_c, layer), hidden_states)
            hs.append(h_layer)
        s_ctxs = []
        for layer in range(self.K):
            s_layer = self.Wawa(hs[layer])
            ctx_apply = self.Ctx(s_layer)
            s_ctxs.append(ctx_apply)
        vs = F.softmax(torch.cat(s_ctxs, dim=2), dim=2)
        h_concats = torch.cat([torch.unsqueeze(hs[layer], 2) for layer in range(self.K)], dim=2)
        final_h = torch.sum(torch.mul(torch.unsqueeze(vs, 3), h_concats), dim=2)
        return final_h


VOCAB_FILES_NAMES = {'vocab_file': 'vec.txt'}


def load_vocab(vocab_file: str, return_embeddings: bool=False) ->Union[Dict[str, int], np.ndarray]:
    """Loads a vocabulary file into a dictionary.

    Loads a vocabulary file, allocates a unique id for each word within the vocabulary and saves the correspondence
    between words and ids into a dictionary. Generates and returns word embeddings if it is required.

    Args:
        vocab_file (`str`):
            The path of the vocabulary file.
        return_embeddings (`bool`, `optional`, defaults to `False`):
            Whether or not to return the word embeddings.

    Returns:
        word_embeddings (`np.ndarray`):
            An numpy array represents each word's embedding within the vocabulary, with the size of (number of words) *
            (embedding dimension). Returns word embeddings if `return_embeddings` is set as True.
        vocab (`Dict[str, int]`):
            A dictionary indicates the unique id of each word within the vocabulary.
    """
    vocab = collections.OrderedDict()
    vocab['[PAD]'] = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
    num_embeddings = len(lines) + 1
    embedding_dim = len(lines[0].split()) - 1
    for index, line in enumerate(lines):
        token = ' '.join(line.split()[:-embedding_dim])
        if token in vocab:
            token = f'{token}_{index + 1}'
        vocab[token] = index + 1
    if return_embeddings:
        word_embeddings = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
        for index, line in enumerate(lines):
            embedding = [float(value) for value in line.strip().split()[-embedding_dim:]]
            word_embeddings[index + 1] = embedding
        return word_embeddings
    return vocab


class WordEmbedding(nn.Module):
    """Base class for word embedding.

    Base class for word embedding, in which the word embeddings are loaded from a pre-trained word embedding file and
    could be resized into a distinct size.

    Attributes:
        word_embeddings (`torch.Tensor`):
            A tensor representing the word embedding matrix, whose dimension is (number of tokens) * (embedding
            dimension).
        position_embeddings (`torch.Tensor`):
            A tensor representing the position embedding matrix, whose dimension is (number of positions) * (embedding
            dimension).
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
    """

    def __init__(self, config, vocab_size: int) ->None:
        """Constructs a `WordEmbedding`."""
        super(WordEmbedding, self).__init__()
        if not os.path.exists(os.path.join(config.vocab_file, VOCAB_FILES_NAMES['vocab_file'].replace('txt', 'npy'))):
            embeddings = load_vocab(os.path.join(config.vocab_file, VOCAB_FILES_NAMES['vocab_file']), return_embeddings=True)
            np.save(os.path.join(config.vocab_file, VOCAB_FILES_NAMES['vocab_file'].replace('txt', 'npy')), embeddings)
        else:
            embeddings = np.load(os.path.join(config.vocab_file, VOCAB_FILES_NAMES['vocab_file'].replace('txt', 'npy')))
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=False, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.num_position_embeddings, config.position_embedding_dim)
        self.register_buffer('position_ids', torch.arange(config.num_position_embeddings).expand((1, -1)))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resize_token_embeddings(vocab_size)
        self.config = config
        if config.has_type_embeddings:
            self.type_embeddings = nn.Embedding(config.num_types, config.type_embedding_dim)

    def resize_token_embeddings(self, vocab_size: int) ->None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        if len(self.word_embeddings.weight) > vocab_size:
            raise ValueError('Invalid vocab_size %d < original vocab size.' % vocab_size)
        elif len(self.word_embeddings.weight) == vocab_size:
            pass
        else:
            num_added_token = vocab_size - len(self.word_embeddings.weight)
            embedding_dim = self.word_embeddings.weight.shape[1]
            average_embedding = torch.mean(self.word_embeddings.weight, dim=0).expand(1, -1)
            self.word_embeddings.weight = nn.Parameter(torch.cat((self.word_embeddings.weight.data, average_embedding.expand(num_added_token, embedding_dim))))

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor]=None, position: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Generates word embeddings and position embeddings and concatenates them together."""
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape[0], input_shape[1]
        position_ids = self.position_ids[:, :seq_length].expand(batch_size, seq_length)
        if position is not None:
            position_ids = torch.abs(position_ids - position.unsqueeze(1))
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeds = torch.cat((inputs_embeds, position_embeds), dim=-1)
        if token_type_ids is not None and self.config.has_type_embeddings:
            embeds = torch.cat((embeds, self.type_embeddings(token_type_ids)), dim=-1)
        if self.config.dropout_after_wordvec:
            embeds = self.dropout(embeds)
        return embeds


class ModelOutput(OrderedDict):
    """
    This code follows the output implementation of HuggingFace Transformers, which
    can enable our toolkit to collaborate with HuggingFace Transformers.
    ---
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer 
    or slice (like a tuple) or strings (like a dictionary) that will ignore the `None` attributes. 
    Otherwise behaves like a regular python dictionary.
    """

    def __post_init__(self):
        class_fields = fields(self)
        if not len(class_fields):
            raise ValueError(f'{self.__class__.__name__} has no fields.')
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f'{self.__class__.__name__} should not have more than one required field.')
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
        if other_fields_are_none and not isinstance(first_field, torch.Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False
            if first_field_iterator:
                for element in iterator:
                    if not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        return tuple(self[k] for k in self.keys())


class Output(ModelOutput):
    """A class for the model's output, containing the hidden states of the sequence."""
    last_hidden_state: torch.Tensor = None


class CNN(nn.Module):
    """A Convolutional Neural Network (CNN) as backbone model.

    A Convolutional Neural Network (CNN) as the backbone model, which comprises a 1-d convolutional layer, a relu
    activation layer, and a dropout layer. The last hidden state of the model would be returned.

    Attributes:
        config:
            The configurations of the model.
        embedding (`WordEmbedding`):
            A `WordEmbedding` instance representing the embedding matrices of tokens and positions.
        conv (`nn.Conv1d`):
            A `nn.Conv1d` layer representing 1-dimensional convolution layer.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
    """

    def __init__(self, config, vocab_size: int, kernel_size: Optional[int]=3, padding_size: Optional[int]=1) ->None:
        """Constructs a `CNN`."""
        super(CNN, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        in_channels = config.word_embedding_dim + config.position_embedding_dim + config.type_embedding_dim if config.has_type_embeddings else config.word_embedding_dim + config.position_embedding_dim
        self.conv = nn.Conv1d(in_channels, config.hidden_size, kernel_size, padding=padding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self, vocab_size: int) ->None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        self.embedding.resize_token_embeddings(vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor]=None, position: Optional[torch.Tensor]=None, return_dict: Optional[bool]=True) ->Union[Output, Tuple[torch.Tensor]]:
        """Conducts the convolution operations on the input tokens."""
        x = self.embedding(input_ids, token_type_ids, position)
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x).transpose(1, 2))
        if return_dict:
            return Output(last_hidden_state=x)
        else:
            return x


class LSTM(nn.Module):
    """A Long Short-Term Memory (LSTM) network as backbone model.

    A bidirectional two-layered Long Short-Term Memory (LSTM) network as the backbone model, which utilizes recurrent
    computations for hidden states and addresses long-term information preservation and short-term input skipping
    using gated memory cells.

    Attributes:
        config:
            The configurations of the model.
        embedding (`WordEmbedding`):
            A `WordEmbedding` instance representing the embedding matrices of tokens and positions.
        rnn (`nn.LSTM`):
            A `nn.LSTM` layer representing a bi-directional two-layered LSTM network, which manipulates the word
            embedding and position embedding for recurrent computations.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
       """

    def __init__(self, config, vocab_size: int) ->None:
        """Constructs a `LSTM`."""
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        self.rnn = nn.LSTM(config.word_embedding_dim + config.position_embedding_dim, config.hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self, vocab_size: int) ->None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        self.embedding.resize_token_embeddings(vocab_size)

    def prepare_pack_padded_sequence(self, input_ids: torch.Tensor, input_lengths: torch.Tensor, descending: Optional[bool]=True):
        """Sorts the input sequences based on their length."""
        sorted_input_lengths, indices = torch.sort(input_lengths, descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_input_ids = input_ids[indices]
        return sorted_input_ids, sorted_input_lengths, desorted_indices

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position: Optional[torch.Tensor]=None, return_dict: Optional[bool]=True):
        """Forward propagation of a LSTM network."""
        add_pseudo = max(torch.sum(attention_mask, dim=-1).tolist()) != input_ids.shape[1]
        if add_pseudo:
            input_ids = torch.cat((torch.zeros_like(input_ids[0]).unsqueeze(0), input_ids), dim=0)
            attention_mask = torch.cat((torch.ones_like(attention_mask[0]).unsqueeze(0), attention_mask), dim=0)
        input_length = torch.sum(attention_mask, dim=-1)
        sorted_input_ids, sorted_seq_length, desorted_indices = self.prepare_pack_padded_sequence(input_ids, input_length)
        x = self.embedding(sorted_input_ids, position)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_length.cpu(), batch_first=True)
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        x = output[desorted_indices]
        if add_pseudo:
            x = self.dropout(x)[1:, :, :]
        else:
            x = self.dropout(x)
        if return_dict:
            return Output(last_hidden_state=x)
        else:
            return x


class LinearHead(nn.Module):
    """A token-wise classification head for classifying the hidden states to label distributions.

    A token-wise classification head for classifying hidden states to label distributions through a linear
    transformation, selecting the label with the highest probability corresponding to each logit.

    Attributes:
        classifier (`nn.Linear`):
            An `nn.Linear` layer classifying each logit into its corresponding label.
    """

    def __init__(self, config):
        super(LinearHead, self).__init__()
        if config.model_type == 'cnn':
            self.classifier = nn.Linear(config.hidden_size * config.head_scale + config.word_embedding_dim * 3, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size * config.head_scale, config.num_labels)

    def forward(self, hidden_state: torch.Tensor) ->torch.Tensor:
        """Classifies hidden states to label distribution."""
        logits = self.classifier(hidden_state)
        return logits


class MRCHead(nn.Module):
    """A token-wise classification head for the Machine Reading Comprehension (MRC) paradigm.

    A classification head for the Machine Reading Comprehension (MRC) paradigm, predicting the answer of each question
    corresponding to a mention type. The classifier returns two logits indicating the start and end position of each
    mention corresponding to the question.

    Attributes:
        qa_outputs (`nn.Linear`):
            An `nn.Linear` layer transforming the hidden states to two logits, indicating the start and end position
            of a given mention type.
    """

    def __init__(self, config) ->None:
        """Constructs a `MRCHead`."""
        super(MRCHead, self).__init__()
        self.qa_outputs = nn.Linear(config.hidden_size * config.head_scale, 2)

    def forward(self, hidden_state: torch.Tensor):
        """The forward propagation of `MRCHead`."""
        logits = self.qa_outputs(hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits


class CRF(nn.Module):
    """Conditional Random Field (CRF) module.

    This module implements a Conditional Random Field (CRF). The forward computation of this class computes the log
    likelihood of the given sequence of tags and emission score tensor. This class also has `CRF.decode()` method which
    finds the best tag sequence given an emission score tensor using Viterbi algorithm.

    Attributes:
        num_tags (`int`):
            An integer indicating the number of tags to be predicted.
        batch_first (`bool`):
            A boolean variable indicating whether or not splitting the data in batches.
        start_transitions (`nn.Parameter`):
            An `nn.Parameter` matrix containing the start transition score tensor of size `(num_tags,)`.
        end_transitions (`nn.Parameter`):
            An `nn.Parameter` matrix containing the end transition score tensor of size `(num_tags,)`.
        transitions (`nn.Parameter`):
            An `nn.Parameter` matrix indicating the score tensor of size `(num_tags, num_tags)`.
    """

    def __init__(self, num_tags: int, batch_first: bool=False) ->None:
        """Constructs a `CRF`."""
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) ->str:
        """Displays the class name and the number of tags."""
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: Optional[torch.ByteTensor]=None, reduction: str='sum') ->torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores."""
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor]=None) ->List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm."""
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        return self._viterbi_decode(emissions, mask)

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor]=None, mask: Optional[torch.ByteTensor]=None) ->None:
        """Validates the emission dimension and whether its slice satisfies tag number, tag shape and mask shape."""
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(f'the first two dimensions of emissions and tags must match, got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f'the first two dimensions of emissions and mask must match, got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) ->torch.Tensor:
        """Computes the score based on the emission and transition matrix."""
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()
        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) ->torch.Tensor:
        """Compute the log-sum-exp score."""
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) ->List[List[int]]:
        """Decodes the optimal path using Viterbi algorithm."""
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list


class Config(object):
    """The configurations of this project.

    The configurations of this project, configuring the annotation path, source text folder path, and saving folder
    path of the dataset.

    Attributes:
        ALL_DATA_FOLDER (`str`):
            A string indicating the folder containing all the datasets.
        DATA_FOLDER (`str`):
            A string indicating the folder containing the annotation folder and source text folder.
        GOLD_FOLDER (`str`):
            A string indicating the folder containing the annotations of event triggers, arguments, and entities of the
            documents.
        SOURCE_FOLDER (`str`):
            A string indicating the folder containing the source texts of the documents, corresponding to the documents
            under the `GOLD_FOLDER` folder.
        SAVE_DATA_FOLDER (`str`):
            A string indicating the folder of saving the manipulated dataset.
    """

    def __init__(self):
        self.ALL_DATA_FOLDER = '../../../data'
        self.DATA_FOLDER = os.path.join(self.ALL_DATA_FOLDER, 'LDC2015E78/data/eng')
        self.GOLD_FOLDER = os.path.join(self.DATA_FOLDER, 'ere')
        self.SOURCE_FOLDER = os.path.join(self.DATA_FOLDER, 'translation')
        self.SAVE_DATA_FOLDER = os.path.join(self.ALL_DATA_FOLDER, 'processed/ere')
        if not os.path.exists(self.SAVE_DATA_FOLDER):
            os.mkdir(self.SAVE_DATA_FOLDER)


MODEL_NAMES = {'s2s-mt5-ed': 'https://cloud.tsinghua.edu.cn/f/cdc4b333aff143ff870e/?dl=1', 's2s-mt5-eae': 'https://cloud.tsinghua.edu.cn/f/f4ac92ac8f2c4e769282/?dl=1'}


def download(path, base_path, url):
    req = requests.get(url, stream=True)
    file = open(path, 'wb')
    req.raise_for_status()
    None
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total, desc='Downloading')
    None
    None
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            file.write(chunk)
    progress.close()
    file.close()
    os.system(f'unzip {path} -d {base_path}')
    os.system(f'rm {path}')


def check_web_and_convert_path(path, load_type, base_path='~/.cache/OmniEvent_Model'):
    base_path = os.path.expanduser(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if os.path.isdir(path):
        None
        return path
    if os.path.isdir(os.path.join(base_path, path)):
        None
        return os.path.join(base_path, path)
    else:
        if path not in MODEL_NAMES:
            raise ValueError(f"'{path}' is not a valid model identifier")
        url = MODEL_NAMES[path]
        try:
            requests.get(url, stream=True).raise_for_status()
        except:
            raise ValueError(f"'{path}' is not a valid model identifier")
        cache_path = f'{base_path}/{path}'
        download(cache_path + '.zip', base_path, url)
        return cache_path


def aggregate(config, method, hidden_states: torch.Tensor, attention_mask: torch.Tensor, trigger_left: torch.Tensor, trigger_right: torch.Tensor, argument_left: torch.Tensor, argument_right: torch.Tensor, embeddings: Optional[torch.Tensor]=None):
    """Aggregates information to each position.

    Aggregates information to each position. The aggregation methods include selecting the "cls"s' representations,
    selecting the markers' representations, max-pooling, and dynamic multi-pooling.

    Args:
        config:
            The configurations of the model.
        method:
            The method proposed to be utilized in the aggregation process.
            TODO: The data type of the variable `method` should be configured.
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.
        trigger_left (`torch.Tensor`):
            A tensor indicating the left position of the triggers.
        trigger_right (`torch.Tensor`):
            A tensor indicating the right position of the triggers.
        argument_left (`torch.Tensor`):
            A tensor indicating the left position of the arguments.
        argument_right (`torch.Tensor`):
            A tensor indicating the right position of the arguments.
    """
    if config.aggregation == 'cls':
        return method(hidden_states)
    elif config.aggregation == 'marker':
        if argument_left is not None:
            return method(hidden_states, argument_left, argument_right)
        else:
            return method(hidden_states, trigger_left, trigger_right)
    elif config.aggregation == 'max_pooling':
        return method(hidden_states)
    elif config.aggregation == 'dynamic_pooling':
        return method(hidden_states, attention_mask, trigger_left, embeddings, argument_left, argument_right)
    else:
        raise ValueError('Invaild %s aggregation method' % config.aggregation)


def max_pooling(hidden_states: torch.Tensor) ->torch.Tensor:
    """Applies the max-pooling operation over the sentence representation.

    Applies the max-pooling operation over the representation of the entire input sequence to capture the most useful
    information. The operation processes on the hidden states, which are output by the backbone model.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.

    Returns:
        pooled_states (`torch.Tensor`):
            A tensor represents the max-pooled hidden states, containing the most useful information of the sequence.
    """
    batch_size, seq_length, hidden_size = hidden_states.size()
    pooled_states = F.max_pool1d(input=hidden_states.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
    return pooled_states


def select_cls(hidden_states: torch.Tensor) ->torch.Tensor:
    """Returns the representations of the `<cls>` tokens.

    Returns the representations of each sequence's `<cls>` token by slicing the hidden state tensor output by the
    backbone model. The representations of the `<cls>` tokens contain general information of the sequences.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor represents the hidden states output by the backbone model.

    Returns:
        `torch.Tensor`:
            A tensor containing the representations of each sequence's `<cls>` token.
    """
    return hidden_states[:, 0, :]


def select_marker(hidden_states: torch.Tensor, left: torch.Tensor, right: torch.Tensor) ->torch.Tensor:
    """Returns the representations of the marker tokens.

    Returns the representations of each sequence's marker tokens by slicing the hidden state tensor output by the
    backbone model.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.
        left (`torch.Tensor`):
            A tensor indicates the left position of the markers.
        right (`torch.Tensor`):
            A tensor indicates the right position of the markers.

    Returns:
        marker_output (`torch.Tensor`):
            A tensor containing the representations of each sequence's marker tokens by concatenating their left and
            right token's representations.
    """
    batch_size = hidden_states.size(0)
    batch_indice = torch.arange(batch_size)
    left_states = hidden_states[batch_indice, left, :]
    right_states = hidden_states[batch_indice, right, :]
    marker_output = torch.cat((left_states, right_states), dim=-1)
    return marker_output


def get_aggregation(config):
    """Obtains the aggregation method to be utilized.

    Obtains the aggregation method to be utilized based on the model's configurations. The aggregation methods include
    selecting the `<cls>`s' representations, selecting the markers' representations, max-pooling, and dynamic
    multi-pooling.

    Args:
        config:
            The configurations of the model.

    Returns:
        The proposed method/class for the aggregation process.
        TODO: The data type of the variable `method` should be configured.
    """
    if config.aggregation == 'cls':
        return select_cls
    elif config.aggregation == 'marker':
        return select_marker
    elif config.aggregation == 'dynamic_pooling':
        return DynamicPooling(config)
    elif config.aggregation == 'max_pooling':
        return max_pooling
    else:
        raise ValueError('Invaild %s aggregation method' % config.aggregation)


def get_head(config):
    if config.head_type == 'linear':
        return LinearHead(config)
    elif config.head_type == 'mrc':
        return MRCHead(config)
    elif config.head_type == 'crf':
        return CRF(config.num_labels, batch_first=True)
    elif config.head_type in ['none', 'None'] or config.head_type is None:
        return None
    else:
        raise ValueError('Invalid head_type %s in config' % config.head_type)


class BertConfig(Config):
    """
    This is a configuration class that stores the configuration of the BERT model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers and the pooler layer.
    You can choose to use the default value of 768 or customize their dimensions.  

    """

    def __init__(self, vocab_size=119547, type_size=2, dim_model=768, num_heads=12, dim_head=64, dim_ff=3072, num_layers=12, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='none', position_size=512, norm_init_var=1.0, norm_bias=True, norm_eps=1e-12, att_init_mean=0.0, att_init_std=0.02, att_bias=True, att_mask_value=float('-1e4'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=True, length_scale=False, attn_scale=True, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.position_size = position_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


@torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return hidden * weight


class BertPooler(nn.Module):

    def __init__(self, dim_model: int):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.FloatTensor):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % type(text))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode('utf-8', 'ignore')
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError('Unsupported string type: %s' % type(text))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


def is_contain_chinese(check_str):
    for ch in check_str:
        if u'一' <= ch <= u'\u9fff':
            return True
    return False


def is_contain_point(check_str):
    for ch in check_str:
        if u'0' <= ch <= u'9':
            return True
    return False


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token='<unk>', max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        token = convert_to_unicode(token)
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = ''.join(chars[start:end])
                if is_contain_chinese(substr) or is_contain_point(substr):
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                elif substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens


class Encoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = CPM1Tokenizer(os.path.join(self.args.tokenizer_path, 'vocab.txt'))

    def encode(self, line):
        if len(line) > 5000000:
            return None, None, 0
        data = line.strip()
        data = data.replace('<n>', '\n')
        doc_ids = Encoder.tokenizer.encode(data)
        if len(doc_ids) < 32:
            return None, None, 0
        doc_ids.append(Encoder.tokenizer.eod_id)
        doc_ids = [1] + doc_ids
        doc_ids = [j for j in doc_ids if j != Encoder.tokenizer.unk_id]
        contexts = []
        labels = []
        i = 0
        while i < len(doc_ids):
            piece = doc_ids[i:i + 512]
            if len(piece) < 32:
                break
            i += 512
            context = piece
            label = piece
            assert len(label) == len(context)
            assert len(label) <= 512
            contexts.append(context)
            labels.append(label)
        return contexts, labels, len(line)


class BertModel(torch.nn.Module):

    def __init__(self, args, num_types):
        super().__init__()
        self.bert: Bert = Bert.from_pretrained(args.model_config)
        dim_model = self.bert.input_embedding.dim_model
        self.dense = Linear(dim_model, num_types)
        bmt.init_parameters(self.dense)

    def forward(self, *args, **kwargs):
        pooler_output = self.bert(*args, **kwargs).pooler_output
        logits = self.dense(pooler_output)
        return logits


class SelfAttentionBlock(torch.nn.Module):
    """  The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_model: int, num_heads: int, dim_head: int, dtype=torch.half, int8=False, norm_init_var: float=1.0, norm_bias: bool=False, norm_eps: float=1e-05, att_init_mean: float=0.0, att_init_std: float=0.02, att_bias: bool=False, att_mask_value: float=float('-inf'), pos_bias_type: str='none', post_layer_norm: bool=False, length_scale: bool=False, attn_scale: bool=False, dropout_p: float=0, sparse_attention: bool=False, attention_window: int=512):
        super().__init__()
        self.layernorm_before_attention = LayerNorm(dim_norm=dim_model, bias=norm_bias, dtype=dtype, eps=norm_eps, init_var=norm_init_var)
        self.sparse_attention = sparse_attention
        if not sparse_attention:
            self.self_attention = Attention(dim_in=dim_model, num_heads=num_heads, dim_head=dim_head, dim_out=dim_model, dtype=dtype, int8=int8, init_mean=att_init_mean, init_std=att_init_std, bias=att_bias, mask_value=att_mask_value, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p)
        else:
            self.self_attention = SparseSelfAttention(dim_in=dim_model, num_heads=num_heads, dim_head=dim_head, dim_out=dim_model, dtype=dtype, int8=int8, init_mean=att_init_mean, init_std=att_init_std, bias=att_bias, mask_value=att_mask_value, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p, attention_window=attention_window)
        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.post_layer_norm = post_layer_norm

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_bias: Optional[torch.Tensor]=None, use_cache: bool=False, past_key_value=None):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.  
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """
        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        if not self.sparse_attention:
            x = self.self_attention(x, x, attention_mask, position_bias, use_cache, past_key_value)
        else:
            x = self.self_attention(x, attention_mask, position_bias)
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class CrossAttentionBlock(torch.nn.Module):
    """  The whole cross-attention block. A sequence of operation. Consists of layernorm, cross-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_model: int, num_heads: int, dim_head: int, dtype=torch.half, int8=False, norm_init_var: float=1.0, norm_bias: bool=False, norm_eps: float=1e-05, att_init_mean: float=0.0, att_init_std: float=0.02, att_bias: bool=False, att_mask_value: float=float('-inf'), pos_bias_type: str='none', post_layer_norm: bool=False, length_scale: bool=False, attn_scale: bool=False, dropout_p: float=0, sparse_attention: bool=False, attention_window: int=512):
        super().__init__()
        self.layernorm_before_attention = LayerNorm(dim_norm=dim_model, bias=norm_bias, dtype=dtype, eps=norm_eps, init_var=norm_init_var)
        self.sparse_attention = sparse_attention
        if not sparse_attention:
            self.self_attention = Attention(dim_in=dim_model, num_heads=num_heads, dim_head=dim_head, dim_out=dim_model, dtype=dtype, int8=int8, init_mean=att_init_mean, init_std=att_init_std, bias=att_bias, mask_value=att_mask_value, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p)
        else:
            self.self_attention = SparseSelfAttention(dim_in=dim_model, num_heads=num_heads, dim_head=dim_head, dim_out=dim_model, dtype=dtype, int8=int8, init_mean=att_init_mean, init_std=att_init_std, bias=att_bias, mask_value=att_mask_value, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p, attention_window=attention_window)
        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.post_layer_norm = post_layer_norm

    def forward(self, hidden_states: torch.Tensor, key_value_states: torch.Tensor, attention_mask: torch.Tensor, position_bias: Optional[torch.Tensor]=None, use_cache: bool=False, past_key_value=None):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of cross-attention block. It can be seen as query in the coming self-attention operation.
            key_value_states(:obj:`torch.Tensor` of shape ``(batch, seq_cross, dim_model)``): Used as key_value in coming self_attention operation. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_cross)``): Avoid invalid areas to participate in the calculation.  
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_cross)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of cross-attention block.

        """
        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        if not self.sparse_attention:
            x = self.self_attention(x, key_value_states, attention_mask, position_bias, use_cache, past_key_value)
        else:
            x = self.self_attention(x, attention_mask, position_bias)
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


@torch.jit.script
def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FFNBlock(torch.nn.Module):
    """ The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_model: int, dim_ff: int, dtype=torch.half, int8=False, norm_init_var: float=1.0, norm_bias: bool=False, norm_eps: float=1e-05, ffn_init_mean: float=0.0, ffn_init_std: float=0.02, ffn_bias: bool=False, ffn_activate_fn: str='gated_gelu', post_layer_norm: bool=False, length_scale: bool=False, dropout_p: float=0):
        super().__init__()
        self.layernorm_before_ffn = LayerNorm(dim_norm=dim_model, bias=norm_bias, dtype=dtype, eps=norm_eps, init_var=norm_init_var)
        self.ffn = FeedForward(dim_in=dim_model, dim_ff=dim_ff, dim_out=dim_model, dtype=dtype, int8=int8, init_mean=ffn_init_mean, init_std=ffn_init_std, bias=ffn_bias, activate_fn=ffn_activate_fn, length_scale=length_scale)
        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.post_layer_norm = post_layer_norm

    def forward(self, hidden_states: torch.Tensor):
        """ 
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """
        x = self.layernorm_before_ffn(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states


class TransformerBlock(torch.nn.Module):
    """ The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        is_decoder (bool, optional): whether to use cross-attention. Defaults to False.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_model: int, dim_ff: int, num_heads: int, dim_head: int, is_decoder: bool=False, dtype=torch.half, int8=False, norm_init_var: float=1.0, norm_bias: bool=False, norm_eps: float=1e-05, att_init_mean: float=0.0, att_init_std: float=0.02, att_bias: bool=False, att_mask_value: float=float('-inf'), ffn_init_mean: float=0.0, ffn_init_std: float=0.02, ffn_bias: bool=False, ffn_activate_fn: str='gated_gelu', pos_bias_type: str='none', post_layer_norm: bool=False, parallel_ffn: bool=False, length_scale: bool=False, attn_scale: bool=False, dropout_p: float=0, sparse_attention: bool=False, attention_window: int=512):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_att = SelfAttentionBlock(dim_model=dim_model, num_heads=num_heads, dim_head=dim_head, dtype=dtype, int8=int8, norm_eps=norm_eps, norm_init_var=norm_init_var, norm_bias=norm_bias, att_init_mean=att_init_mean, att_init_std=att_init_std, att_bias=att_bias, att_mask_value=att_mask_value, pos_bias_type=pos_bias_type, post_layer_norm=post_layer_norm, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p, sparse_attention=sparse_attention, attention_window=attention_window)
        if is_decoder:
            self.cross_att = CrossAttentionBlock(dim_model=dim_model, num_heads=num_heads, dim_head=dim_head, dtype=dtype, int8=int8, norm_eps=norm_eps, norm_init_var=norm_init_var, norm_bias=norm_bias, att_init_mean=att_init_mean, att_init_std=att_init_std, att_bias=att_bias, att_mask_value=att_mask_value, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p)
        else:
            self.cross_att = None
        self.ffn = FFNBlock(dim_model=dim_model, dim_ff=dim_ff, dtype=dtype, int8=int8, norm_eps=norm_eps, norm_init_var=norm_init_var, norm_bias=norm_bias, ffn_init_mean=ffn_init_mean, ffn_init_std=ffn_init_std, ffn_bias=ffn_bias, ffn_activate_fn=ffn_activate_fn, length_scale=length_scale, dropout_p=dropout_p, post_layer_norm=post_layer_norm)
        self.parallel_ffn = parallel_ffn

    def forward(self, self_hidden_states: torch.Tensor, self_attention_mask: torch.Tensor, self_position_bias: Optional[torch.Tensor]=None, cross_hidden_states=None, cross_attention_mask=None, cross_position_bias=None, use_cache: bool=False, past_key_value=None):
        """
        Args:
            self_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.  
            self_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.
            cross_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_cross, dim_model)``): Input of cross-attention block. 
            cross_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_cross)``): Avoid invalid areas to participate in the calculation of cross-attention.  
            cross_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_cross)``): Provide positional information to cross-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """
        current_key_value = None
        hidden_states = self.self_att(self_hidden_states, attention_mask=self_attention_mask, position_bias=self_position_bias, use_cache=use_cache, past_key_value=past_key_value)
        if use_cache:
            hidden_states, current_key_value = hidden_states
        if self.is_decoder and self.cross_att is not None:
            hidden_states = self.cross_att(hidden_states=hidden_states, key_value_states=cross_hidden_states, attention_mask=cross_attention_mask, position_bias=cross_position_bias)
        if self.parallel_ffn:
            hidden_states_2 = self.ffn(self_hidden_states)
            hidden_states = hidden_states - self_hidden_states + hidden_states_2
        else:
            hidden_states = self.ffn(hidden_states)
        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class RotaryEmbedding(nn.Module):
    """`Rotary Position Embedding <https://arxiv.org/abs/2104.09864v2>

    Args:
        rotary_dim (int): rotary dimension
    """

    def __init__(self, rotary_dim: int):
        super().__init__()
        self.rotary_dim = rotary_dim

    def fixed_pos_embedding(self, x, seq_len=None, dtype=torch.float):
        dim = x.shape[-1]
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2) / dim)
        sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(seq_len), inv_freq).to(x.device)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        if x.dim() == 4:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
        else:
            x1 = x[:, :, ::2]
            x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), axis=-1)
        return x.flatten(-2)

    def apply_rotary_pos_emb(self, x, sincos, offset=0):
        sin, cos = map(lambda t: t[None, offset:x.shape[-2] + offset, :].repeat_interleave(2, 2), sincos)
        return x * cos + self.rotate_every_two(x) * sin

    def forward(self, h_q, h_k):
        """
        Args:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)

        Return:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)
        """
        if h_q.dim() == 4:
            q_rot = h_q[:, :, :, :self.rotary_dim]
            q_pass = h_q[:, :, :, self.rotary_dim:]
            k_rot = h_k[:, :, :, :self.rotary_dim]
            k_pass = h_k[:, :, :, self.rotary_dim:]
        else:
            q_rot = h_q[:, :, :self.rotary_dim]
            q_pass = h_q[:, :, self.rotary_dim:]
            k_rot = h_k[:, :, :self.rotary_dim]
            k_pass = h_k[:, :, self.rotary_dim:]
        seq_len = h_k.shape[-2]
        sincos = self.fixed_pos_embedding(k_rot, seq_len=seq_len, dtype=h_k.dtype)
        k_rot = self.apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = self.apply_rotary_pos_emb(q_rot, sincos, offset=0)
        h_q = torch.cat([q_rot, q_pass], dim=-1)
        h_k = torch.cat([k_rot, k_pass], dim=-1)
        return h_q, h_k


class Decoder(torch.nn.Module):
    """ Layers of decoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, num_layers: int, dim_model: int, dim_ff: int, num_heads: int, dim_head: int, dtype: torch.dtype=torch.half, int8: bool=False, norm_init_var: float=1.0, norm_bias: bool=False, norm_eps: float=1e-05, att_init_mean: float=0.0, att_init_std: float=0.02, att_bias: bool=False, att_mask_value: float=float('-inf'), ffn_init_mean: float=0.0, ffn_init_std: float=0.02, ffn_bias: bool=False, ffn_activate_fn: str='gated_gelu', pos_bias_type: str='none', length_scale: bool=False, attn_scale: bool=False, dropout_p: float=0, parallel_ffn: bool=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = bmt.TransformerBlockList([bmt.CheckpointBlock(TransformerBlock(dim_model=dim_model, dim_ff=dim_ff, num_heads=num_heads, dim_head=dim_head, is_decoder=True, dtype=dtype, int8=int8, norm_init_var=norm_init_var, norm_bias=norm_bias, norm_eps=norm_eps, att_init_mean=att_init_mean, att_init_std=att_init_std, att_bias=att_bias, att_mask_value=att_mask_value, ffn_init_mean=ffn_init_mean, ffn_init_std=ffn_init_std, ffn_bias=ffn_bias, ffn_activate_fn=ffn_activate_fn, pos_bias_type=pos_bias_type, length_scale=length_scale, attn_scale=attn_scale, dropout_p=dropout_p, parallel_ffn=parallel_ffn)) for _ in range(num_layers)])
        self.output_layernorm = LayerNorm(dim_norm=dim_model, bias=norm_bias, dtype=dtype, eps=norm_eps, init_var=norm_init_var)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_bias: torch.Tensor, cross_hidden_states=None, cross_attention_mask=None, cross_position_bias=None, use_cache: bool=False, past_key_values=None):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``): Input of decoder, Can be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_dec, seq_dec)``): Avoid invalid areas to participate in the calculation. 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_dec, seq_dec)``) Provides position information to attention mechanism. 
            cross_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of decoder, Can be the output of encoder. 
            cross_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_dec, seq_enc)``): Avoid invalid areas to participate in the calculation when the output of encoder participates in the calculation. 
            cross_position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_dec, seq_enc)``) Provides position information to attention mechanism when the output of encoder participates in the calculation.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``: The decoder output. 

        """
        if not use_cache:
            hidden_states = self.layers(hidden_states, attention_mask, position_bias, cross_hidden_states, cross_attention_mask, cross_position_bias)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states
        else:
            with torch.no_grad():
                current_key_values = []
                for i, module in enumerate(self.layers):
                    hidden_states = module(hidden_states, attention_mask, position_bias, cross_hidden_states, cross_attention_mask, cross_position_bias, past_key_value=past_key_values[i] if past_key_values else None, use_cache=use_cache)
                    current_key_values.append(hidden_states[1])
                    hidden_states = hidden_states[0]
                hidden_states = self.output_layernorm(hidden_states)
                return hidden_states, current_key_values


class CPM1Config(Config):
    """
    This is a configuration class that stores the configuration of the CPM-1 model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """

    def __init__(self, vocab_size=30968, dim_model=768, num_heads=12, dim_head=64, dim_ff=1920, num_layers=12, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='relative', position_bias_num_buckets=32, position_bias_max_distance=128, pos_init_mean=0.0, pos_init_std=1, norm_init_var=1.0, norm_bias=False, norm_eps=1e-06, att_init_mean=0.0, att_init_std=0.02, att_bias=False, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=False, ffn_activate_fn='gated_gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=False, length_scale=False, attn_scale=False, half=True, int8=False, tied=False, cls_head=None, post_layer_norm=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


class CPM2Config(Config):
    """
    This is a configuration class that stores the configuration of the CPM-2 model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """

    def __init__(self, vocab_size=26240, dim_model=768, num_heads=12, dim_head=64, dim_ff=1920, num_encoder_layers=12, num_decoder_layers=12, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='relative', position_bias_num_buckets=32, position_bias_max_distance=128, pos_init_mean=0.0, pos_init_std=1, norm_init_var=1.0, norm_bias=False, norm_eps=1e-06, att_init_mean=0.0, att_init_std=0.02, att_bias=False, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=False, ffn_activate_fn='gated_gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=False, length_scale=False, attn_scale=False, half=True, int8=False, tied=False, cls_head=None, post_layer_norm=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


class CPM3Config(Config):

    def __init__(self, vocab_size=30720, dim_model=4096, num_heads=64, dim_head=64, dim_ff=10240, num_layers=32, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=1.0, pos_bias_type='relative', position_bias_num_buckets=512, position_bias_max_distance=2048, pos_init_mean=0.0, pos_init_std=1.0, norm_init_var=1.0, norm_bias=False, norm_eps=1e-06, att_init_mean=0.0, att_init_std=1.0, att_bias=False, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=1.0, ffn_bias=False, ffn_activate_fn='gated_gelu', proj_init_mean=0.0, proj_init_std=1.0, proj_bias=False, length_scale=True, attn_scale=True, half=True, int8=False, tied=True, prompt_types=32, prompt_length=64, segment_types=34, max_exact_rate=0.25, max_distance_rate=1.0, absolute_inner_segment=True, cls_head=None, post_layer_norm=False):
        super().__init__()
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.norm_eps = norm_eps
        self.norm_init_var = norm_init_var
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.length_scale = length_scale
        self.absolute_inner_segment = absolute_inner_segment
        self.max_distance_rate = max_distance_rate
        self.max_exact_rate = max_exact_rate
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.vocab_size = vocab_size
        self.pos_bias_type = pos_bias_type
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_bias = norm_bias
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.attn_scale = attn_scale
        self.post_layer_norm = post_layer_norm


class GLMConfig(Config):

    def __init__(self, vocab_size=50048, dim_model=1024, num_heads=16, dim_head=64, dim_ff=4096, num_layers=24, dropout_p=0.1, emb_init_mean=0, emb_init_std=0.02, pos_bias_type='none', position_size=1025, norm_init_var=1.0, norm_bias=True, norm_eps=1e-05, att_init_mean=0.0, att_init_std=0.02, att_bias=True, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=False, length_scale=False, attn_scale=True, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=False, sop_tok_id=50006, eop_tok_id=50007, mask_tok_id=50008):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_size = position_size
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm
        self.sop_tok_id = sop_tok_id
        self.eop_tok_id = eop_tok_id
        self.mask_tok_id = mask_tok_id


class GPT2Config(Config):
    """
    This is a configuration class that stores the configuration of the GPT-2 model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """

    def __init__(self, vocab_size=50258, dim_model=1024, num_heads=16, dim_head=64, dim_ff=4096, num_layers=24, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='none', position_size=1024, norm_init_var=1.0, norm_bias=True, norm_eps=1e-05, att_init_mean=0.0, att_init_std=0.02, att_bias=True, att_mask_value=float('-1e4'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=True, length_scale=False, attn_scale=True, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_size = position_size
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


class GPTjConfig(Config):
    """
    This is a configuration class that stores the configuration of the GPT-J model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 4096 or customize their dimensions.  
    
    """

    def __init__(self, vocab_size=50400, dim_model=4096, num_heads=16, dim_head=256, dim_ff=16384, num_layers=28, dropout_p=0, emb_init_mean=0.0, emb_init_std=1, pos_bias_type='rotary', pos_rotary_dim=64, norm_init_var=1.0, norm_bias=True, norm_eps=1e-05, att_init_mean=0.0, att_init_std=0.1, att_bias=False, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=0.1, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=1, proj_bias=True, length_scale=False, attn_scale=True, half=True, int8=False, tied=False, cls_head=None, post_layer_norm=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.pos_rotary_dim = pos_rotary_dim
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


class LongformerPooler(torch.nn.Module):

    def __init__(self, dim_model):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerLMHead(torch.nn.Module):

    def __init__(self, dim_model, vocab_size, norm_eps, dtype):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True, dtype=dtype)
        self.act_fn = torch.nn.functional.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps, dtype=dtype)
        self.decoder = Linear(dim_model, vocab_size, bias=True, dtype=dtype)

    def forward(self, hidden_states, input_embedding):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = input_embedding.projection(hidden_states) + self.decoder.bias
        return logits


class LongformerConfig(Config):
    """
    This is a configuration class that stores the configuration of the Longformer model, which inherits from the Config class.
    It is used to instantiate the Longformer model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers and the pooler layer.
    You can choose to use the default value of 768 or customize their dimensions.  

    """

    def __init__(self, vocab_size=119547, type_size=2, position_size=512, dim_model=768, num_heads=12, dim_head=64, dim_ff=3072, num_layers=12, dropout_p=0.1, emb_init_mean=0.0, emb_init_std=1, pos_bias_type='none', position_bias_max_distance=1024, norm_init_var=1.0, norm_bias=True, norm_eps=1e-12, att_init_mean=0.0, att_init_std=0.02, att_bias=True, att_mask_value=float('-1e4'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=1, proj_bias=True, length_scale=False, attn_scale=True, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=True, sep_tok_id=2, attention_window=512, pad_token_id=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.position_size = position_size
        self.position_size = position_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_max_distance = position_bias_max_distance
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm
        self.sep_tok_id = sep_tok_id
        self.attention_window = attention_window
        self.pad_token_id = pad_token_id


class RobertaPooler(nn.Module):

    def __init__(self, dim_model: int):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.FloatTensor):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaConfig(Config):
    """
    This is a configuration class that stores the configuration of the RoBERTa model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers and the pooler layer.
    You can choose to use the default value of 768 or customize their dimensions.

    """

    def __init__(self, vocab_size=50265, type_size=1, dim_model=1024, num_heads=16, dim_head=64, dim_ff=4096, num_layers=24, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='none', position_size=514, norm_init_var=1.0, norm_bias=True, norm_eps=1e-05, att_init_mean=0.0, att_init_std=0.02, att_bias=True, att_mask_value=float('-1e4'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=True, ffn_activate_fn='gelu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=True, length_scale=False, attn_scale=True, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=True, pad_token_id=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.position_size = position_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm
        self.pad_token_id = pad_token_id


class T5Config(Config):
    """
    This is a configuration class that stores the configuration of the T5 model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """

    def __init__(self, vocab_size=32128, dim_model=768, num_heads=12, dim_head=64, dim_ff=3072, num_encoder_layers=12, num_decoder_layers=12, dropout_p=0.0, emb_init_mean=0.0, emb_init_std=0.02, pos_bias_type='relative', position_bias_num_buckets=32, position_bias_max_distance=128, pos_init_mean=0.0, pos_init_std=1, norm_init_var=1.0, norm_bias=False, norm_eps=1e-06, att_init_mean=0.0, att_init_std=0.02, att_bias=False, att_mask_value=float('-inf'), ffn_init_mean=0.0, ffn_init_std=0.02, ffn_bias=False, ffn_activate_fn='relu', proj_init_mean=0.0, proj_init_std=0.02, proj_bias=False, length_scale=False, attn_scale=False, half=True, int8=False, tied=True, cls_head=None, post_layer_norm=False, scale=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        self.scale = scale
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class VitConfig(Config):
    """
    This is a configuration class that stores the configuration of the Vit model, which inherits from the Config class.
    It is used to instantiate the vit model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`hidden_size`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """

    def __init__(self, img_size=224, patch_size=16, channels_in=3, num_classes=1000, hidden_size=768, num_layers=12, num_heads=12, mlp_size=3072, attn_bias=True, attn_scale=None, norm_bias=True, ffn_bias=True, representation_size=None, drop=0.0, half=True, dtype=torch.float):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels_in = channels_in
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.attn_bias = attn_bias
        self.attn_scale = attn_scale
        self.norm_bias = norm_bias
        self.ffn_bias = ffn_bias
        self.representation_size = representation_size
        self.drop = drop
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GraphAttentionLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'dropout': 0.5, 'alpha': 4, 'device': 0}),
     lambda: ([torch.rand([4]), torch.rand([4, 4, 4])], {}),
     False),
    (MOGCN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'K': 4, 'dropout': 0.5, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RotaryEmbedding,
     lambda: ([], {'rotary_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_THU_KEG_OmniEvent(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

