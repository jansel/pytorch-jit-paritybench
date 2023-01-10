import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
documents = _module
pick_dataset = _module
DocBank = _module
utils = _module
logger = _module
visualization = _module
model = _module
crf = _module
decoder = _module
encoder = _module
graph = _module
pick = _module
resnet = _module
parse_config = _module
test = _module
tests = _module
test = _module
train = _module
trainer = _module
trainer = _module
class_utils = _module
entities_list = _module
metrics = _module
span_based_f1 = _module
util = _module

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


from typing import *


import warnings


import random


import torch


import torch.nn.functional as F


from torch.utils.data import Dataset


from torchvision import transforms


import pandas as pd


from typing import List


from typing import Tuple


from typing import Dict


import logging


import torch.nn as nn


from torch import Tensor


import math


from torchvision.ops import roi_align


from torchvision.ops import roi_pool


import numpy as np


import torch.utils.model_zoo as model_zoo


import collections


from functools import reduce


from functools import partial


import torch.distributed as dist


from torch.utils.data.dataloader import DataLoader


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.optim as optim


from numpy import inf


from collections import defaultdict


from torchtext.vocab import Vocab


from itertools import repeat


from collections import OrderedDict


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.
    """

    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]]=None, include_start_end_transitions: bool=True) ->None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))
        if constraints is None:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]
        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (1 - mask[i]).view(batch_size, 1)
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        return util.logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.LongTensor) ->torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.ByteTensor=None, input_batch_first=False, keepdim=False) ->torch.Tensor:
        """
        Computes the log likelihood. inputs, tags, mask are assumed to be batch first
        """
        if not input_batch_first:
            inputs = inputs.transpose(0, 1).contiguous()
            tags = tags.transpose(0, 1).contiguous()
            if mask is not None:
                mask = mask.transpose(0, 1).contiguous()
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        if keepdim:
            return log_numerator - log_denominator
        else:
            return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.Tensor, logits_batch_first=False) ->List[Tuple[List[int], float]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """
        if not logits_batch_first:
            logits = logits.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()
        _, max_seq_length, num_tags = logits.size()
        logits, mask = logits.data, mask.data
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)
        constrained_transitions = self.transitions * self._constraint_mask[:num_tags, :num_tags] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data + -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        best_paths = []
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)
        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:sequence_length + 1, :num_tags] = prediction[:sequence_length]
            tag_sequence[sequence_length + 1, end_tag] = 0.0
            viterbi_path, viterbi_score = util.viterbi_decode(tag_sequence[:sequence_length + 2], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))
        return best_paths


logger = logging.getLogger('PICK')


class MLPLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: Optional[int]=None, hidden_dims: Optional[List[int]]=None, layer_norm: bool=False, dropout: Optional[float]=0.0, activation: Optional[str]='relu'):
        """
        transform output of LSTM layer to logits, as input of crf layers
        :param in_dim:
        :param out_dim:
        :param hidden_dims:
        :param layer_norm:
        :param dropout:
        :param activation:
        """
        super().__init__()
        layers = []
        activation_layer = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU}
        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(activation_layer.get(activation, nn.Identity()))
                logger.warning('Activation function {} is not supported, and replace with Identity layer.'.format(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim
        if not out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim else hidden_dims[-1]

    def forward(self, *input: torch.Tensor) ->torch.Tensor:
        return self.mlp(torch.cat(input, 1))


class ClassVocab(Vocab):

    def __init__(self, classes, specials=['<pad>', '<unk>'], **kwargs):
        """
        convert key to index(stoi), and get key string by index(itos)
        :param classes: list or str, key string or entity list
        :param specials: list, special tokens except <unk> (default: {['<pad>', '<unk>']})
        :param kwargs:
        """
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        c = Counter(cls_list)
        self.special_count = len(specials)
        super().__init__(c, specials=specials, **kwargs)


class BiLSTMLayer(nn.Module):

    def __init__(self, lstm_kwargs, mlp_kwargs):
        super().__init__()
        self.lstm = nn.LSTM(**lstm_kwargs)
        self.mlp = MLPLayer(**mlp_kwargs)

    @staticmethod
    def sort_tensor(x: torch.Tensor, length: torch.Tensor, h_0: torch.Tensor=None, c_0: torch.Tensor=None):
        sorted_lenght, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        if h_0 is not None:
            h_0 = h_0[:, sorted_order, :]
        if c_0 is not None:
            c_0 = c_0[:, sorted_order, :]
        return x[sorted_order], sorted_lenght, invert_order, h_0, c_0

    def forward(self, x_seq: torch.Tensor, lenghts: torch.Tensor, initial: Tuple[torch.Tensor, torch.Tensor]):
        """

        :param x_seq: (B, N*T, D)
        :param lenghts: (B,)
        :param initial: (num_layers * directions, batch, D)
        :return: (B, N*T, out_dim)
        """
        x_seq, sorted_lengths, invert_order, h_0, c_0 = self.sort_tensor(x_seq, lenghts, initial[0], initial[0])
        packed_x = nn.utils.rnn.pack_padded_sequence(x_seq, lengths=sorted_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=keys_vocab_cls.stoi['<pad>'])
        output = output[invert_order]
        logits = self.mlp(output)
        return logits


def entities2iob_labels(entities: list):
    """
    get all iob string label by entities
    :param entities:
    :return:
    """
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


class UnionLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags):
        """
        For a document, we merge all non-paddding (valid) x and x_gcn value together in a document-level format,
        then feed it into crf layer.
        :param x: set of nodes, the output of encoder, (B, N, T, D)
        :param x_gcn: node embedding, the output of graph module, (B, N, D)
        :param mask: whether is non-padding (valid) value at i-th position of segments, (B, N, T)
        :param length: the length of every segments (boxes) of documents, (B, N)
        :param tags: IBO label for every segments of documents, (B, N, T)
        :return:
                new_x, (B, max_doc_seq_len, D)
                new_mask, (B, max_doc_seq_len)
                doc_seq_len, (B,)
                new_tag, (B, max_doc_seq_len)
        """
        B, N, T, D = x.shape
        x = x.reshape(B, N * T, -1)
        mask = mask.reshape(B, N * T)
        x_gcn = x_gcn.unsqueeze(2).expand(B, N, T, -1)
        x_gcn = x_gcn.reshape(B, N * T, -1)
        x = x_gcn + x
        doc_seq_len = length.sum(dim=-1)
        max_doc_seq_len = doc_seq_len.max()
        new_x = torch.zeros_like(x, device=x.device)
        new_mask = torch.zeros_like(mask, device=x.device)
        if self.training:
            tags = tags.reshape(B, N * T)
            new_tag = torch.full_like(tags, iob_labels_vocab_cls.stoi['<pad>'], device=x.device)
            new_tag = new_tag[:, :max_doc_seq_len]
        for i in range(B):
            doc_x = x[i]
            doc_mask = mask[i]
            valid_doc_x = doc_x[doc_mask == 1]
            num_valid = valid_doc_x.size(0)
            new_x[i, :num_valid] = valid_doc_x
            new_mask[i, :doc_seq_len[i]] = 1
            if self.training:
                valid_tag = tags[i][doc_mask == 1]
                new_tag[i, :num_valid] = valid_tag
        new_x = new_x[:, :max_doc_seq_len, :]
        new_mask = new_mask[:, :max_doc_seq_len]
        if self.training:
            return new_x, new_mask, doc_seq_len, new_tag
        else:
            return new_x, new_mask, doc_seq_len, None


class Decoder(nn.Module):

    def __init__(self, bilstm_kwargs, mlp_kwargs, crf_kwargs):
        super().__init__()
        self.union_layer = UnionLayer()
        self.bilstm_layer = BiLSTMLayer(bilstm_kwargs, mlp_kwargs)
        self.crf_layer = ConditionalRandomField(**crf_kwargs)

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags: Tensor):
        """

        :param x: set of nodes, the output of encoder, (B, N, T, D)
        :param x_gcn: node embedding, the output of graph module, (B, N, D)
        :param mask: whether is non-padding (valid) value at i-th position of segments, (B, N, T)
        :param length: the length of every segments (boxes) of documents, (B, N)
        :param tags: IBO label for every segments of documents, (B, N, T)
        :return:
        """
        new_x, new_mask, doc_seq_len, new_tag = self.union_layer(x, x_gcn, mask, length, tags)
        logits = self.bilstm_layer(new_x, doc_seq_len, (None, None))
        log_likelihood = None
        if self.training:
            log_likelihood = self.crf_layer(logits, new_tag, mask=new_mask, input_batch_first=True, keepdim=True)
        return logits, new_mask, log_likelihood


class Encoder(nn.Module):

    def __init__(self, char_embedding_dim: int, out_dim: int, image_feature_dim: int=512, nheaders: int=8, nlayers: int=6, feedforward_dim: int=2048, dropout: float=0.1, max_len: int=100, image_encoder: str='resnet50', roi_pooling_mode: str='roi_align', roi_pooling_size: Tuple[int, int]=(7, 7)):
        """
        convert image segments and text segments to node embedding.
        :param char_embedding_dim:
        :param out_dim:
        :param image_feature_dim:
        :param nheaders:
        :param nlayers:
        :param feedforward_dim:
        :param dropout:
        :param max_len:
        :param image_encoder:
        :param roi_pooling_mode:
        :param roi_pooling_size:
        """
        super().__init__()
        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], 'roi pooling model: {} not support.'.format(roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim, nhead=nheaders, dim_feedforward=feedforward_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)
        if image_encoder == 'resnet18':
            self.cnn = resnet.resnet18(output_channels=image_feature_dim)
        elif image_encoder == 'resnet34':
            self.cnn = resnet.resnet34(output_channels=image_feature_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet.resnet50(output_channels=image_feature_dim)
        elif image_encoder == 'resnet101':
            self.cnn = resnet.resnet101(output_channels=image_feature_dim)
        elif image_encoder == 'resnet152':
            self.cnn = resnet.resnet152(output_channels=image_feature_dim)
        else:
            raise NotImplementedError()
        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)
        self.projection = nn.Linear(2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        position_embedding = torch.zeros(max_len, char_embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2).float() * -(math.log(10000.0) / char_embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)
        self.register_buffer('position_embedding', position_embedding)
        self.pe_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """

        :param images: whole_images, shape is (B, N, H, W, C), where B is batch size, N is the number of segments of
                the documents, H is height of image, W is width of image, C is channel of images (default is 3).
        :param boxes_coordinate: boxes coordinate, shape is (B, N, 8),
                where 8 is coordinates (x1, y1, x2, y2, x3, y3, x4, y4).
        :param transcripts: text segments, shape is (B, N, T, D), where T is the max length of transcripts,
                                D is dimension of model.
        :param src_key_padding_mask: text padding mask, shape is (B*N, T), True for padding value.
            if provided, specified padding elements in the key will be ignored by the attention.
            This is an binary mask. When the value is True, the corresponding value on the attention layer of Transformer
            will be filled with -inf.
        need_weights: output attn_output_weights.
        :return: set of nodes X, shape is (B*N, T, D)
        """
        B, N, T, D = transcripts.shape
        _, _, origin_H, origin_W = images.shape
        images = self.cnn(images)
        _, C, H, W = images.shape
        rois_batch = torch.zeros(B, N, 5, device=images.device)
        for i in range(B):
            doc_boxes = boxes_coordinate[i]
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i
        spatial_scale = float(H / origin_H)
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            image_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        image_segments = F.relu(self.bn(self.conv(image_segments)))
        image_segments = image_segments.squeeze()
        image_segments = image_segments.unsqueeze(dim=1)
        transcripts_segments = self.pe_dropout(transcripts + self.position_embedding[:, :, :transcripts.size(2), :])
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)
        image_segments = image_segments.expand_as(transcripts_segments)
        out = image_segments + transcripts_segments
        out = out.transpose(0, 1).contiguous()
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1).contiguous()
        out = self.norm(out)
        out = self.output_dropout(out)
        return out


class GraphLearningLayer(nn.Module):

    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float):
        super().__init__()
        self.projection = nn.Linear(in_dim, learning_dim, bias=False)
        self.learn_w = nn.Parameter(torch.empty(learning_dim))
        self.gamma = gamma
        self.eta = eta
        self.inint_parameters()

    def inint_parameters(self):
        nn.init.uniform_(self.learn_w, a=0, b=1)

    def forward(self, x: Tensor, adj: Tensor, box_num: Tensor=None):
        """

        :param x: nodes set, (B*N, D)
        :param adj: init adj, (B, N, N, default is 1)
        :param box_num: (B, 1)
        :return:
                out, soft adj matrix
                gl loss
        """
        B, N, D = x.shape
        x_hat = self.projection(x)
        _, _, learning_dim = x_hat.shape
        x_i = x_hat.unsqueeze(2).expand(B, N, N, learning_dim)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, learning_dim)
        distance = torch.abs(x_i - x_j)
        if box_num is not None:
            mask = self.compute_dynamic_mask(box_num)
            distance = distance + mask
        distance = torch.einsum('bijd, d->bij', distance, self.learn_w)
        out = F.leaky_relu(distance)
        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v
        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj
        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10
        gl_loss = None
        if self.training:
            gl_loss = self._graph_learning_loss(x_hat, soft_adj, box_num)
        return soft_adj, gl_loss

    @staticmethod
    def compute_static_mask(box_num: Tensor):
        """
        compute -1 mask, if node(box) is not exist, the length of mask is documents.MAX_BOXES_NUM,
        this will help with one nodes multi gpus training mechanism, and ensure batch shape is same. but this operation
        lead to waste memory.
        :param box_num: (B, 1)
        :return: (B, N, N, 1)
        """
        max_len = documents.MAX_BOXES_NUM
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))
        box_num = box_num.expand_as(mask)
        mask = mask < box_num
        row_mask = mask.unsqueeze(1)
        column_mask = mask.unsqueeze(2)
        mask = row_mask & column_mask
        mask = ~mask * -1
        return mask.unsqueeze(-1)

    @staticmethod
    def compute_dynamic_mask(box_num: Tensor):
        """
        compute -1 mask, if node(box) is not exist, the length of mask is calculate by max(box_num),
        this will help with multi nodes multi gpus training mechanism, ensure batch of different gpus have same shape.
        :param box_num: (B, 1)
        :return: (B, N, N, 1)
        """
        max_len = torch.max(box_num)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))
        box_num = box_num.expand_as(mask)
        mask = mask < box_num
        row_mask = mask.unsqueeze(1)
        column_mask = mask.unsqueeze(2)
        mask = row_mask & column_mask
        mask = ~mask * -1
        return mask.unsqueeze(-1)

    def _graph_learning_loss(self, x_hat: Tensor, adj: Tensor, box_num: Tensor):
        """
        calculate graph learning loss
        :param x_hat: (B, N, D)
        :param adj: (B, N, N)
        :param box_num: (B, 1)
        :return:
            gl_loss
        """
        B, N, D = x_hat.shape
        x_i = x_hat.unsqueeze(2).expand(B, N, N, D)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, D)
        box_num_div = 1 / torch.pow(box_num.float(), 2)
        dist_loss = adj + self.eta * torch.norm(x_i - x_j, dim=3)
        dist_loss = torch.exp(dist_loss)
        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)
        f_norm = torch.norm(adj, dim=(1, 2))
        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss


class GCNLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        """
        perform graph convolution operation
        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.w_alpha = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_vi = nn.Parameter(torch.empty(in_dim, in_dim))
        self.w_vj = nn.Parameter(torch.empty(in_dim, in_dim))
        self.bias_h = nn.Parameter(torch.empty(in_dim))
        self.w_node = nn.Parameter(torch.empty(in_dim, out_dim))
        self.inint_parameters()

    def inint_parameters(self):
        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))
        nn.init.uniform_(self.bias_h, a=0, b=1)
        nn.init.kaiming_uniform_(self.w_node, a=math.sqrt(5))

    def forward(self, x: Tensor, alpha: Tensor, adj: Tensor, box_num: Tensor):
        """

        :param x: nodes set (node embedding), (B, N, in_dim)
        :param alpha: relation embedding, (B, N, N, in_dim)
        :param adj: learned soft adj matrix, (B, N, N)
        :param box_num: (B, 1)
        :return:
                x_out: updated node embedding, (B, N, out_dim)
                alpha: updated relation embedding, (B, N, N, out_dim)
        """
        B, N, in_dim = x.shape
        x_i = x.unsqueeze(2).expand(B, N, N, in_dim)
        x_j = x.unsqueeze(1).expand(B, N, N, in_dim)
        x_i = torch.einsum('bijd, dk->bijk', x_i, self.w_vi)
        x_j = torch.einsum('bijd, dk->bijk', x_j, self.w_vj)
        H = F.relu(x_i + x_j + alpha + self.bias_h)
        AH = torch.einsum('bij, bijd-> bid', adj, H)
        new_x = torch.einsum('bid,dk->bik', AH, self.w_node)
        new_x = F.relu(new_x)
        new_alpha = torch.einsum('bijd,dk->bijk', H, self.w_alpha)
        new_alpha = F.relu(new_alpha)
        return new_x, new_alpha


class GLCN(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, gamma: float=0.0001, eta: float=1, learning_dim: int=128, num_layers=2):
        """
        perform graph learning and multi-time graph convolution operation
        :param in_dim:
        :param out_dim:
        :param gamma:
        :param eta:
        :param learning_dim:
        :param num_layers:
        """
        super().__init__()
        self.gl_layer = GraphLearningLayer(in_dim=in_dim, gamma=gamma, eta=eta, learning_dim=learning_dim)
        modules = []
        in_dim_cur = in_dim
        for i in range(num_layers):
            m = GCNLayer(in_dim_cur, out_dim)
            in_dim_cur = out_dim
            out_dim = in_dim_cur
            modules.append(m)
        self.gcn = nn.ModuleList(modules)
        self.alpha_transform = nn.Linear(6, in_dim, bias=False)

    def forward(self, x: Tensor, rel_features: Tensor, adj: Tensor, box_num: Tensor, **kwargs):
        """

        :param x: nodes embedding, (B*N, D)
        :param rel_features: relation embedding, (B, N, N, 6)
        :param adj: default adjacent matrix, (B, N, N)
        :param box_num: (B, 1)
        :param kwargs:
        :return:
        """
        alpha = self.alpha_transform(rel_features)
        soft_adj, gl_loss = self.gl_layer(x, adj, box_num)
        adj = adj * soft_adj
        for i, gcn_layer in enumerate(self.gcn):
            x, alpha = gcn_layer(x, alpha, adj, box_num)
        return x, soft_adj, gl_loss


class PICKModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_kwargs = kwargs['embedding_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        graph_kwargs = kwargs['graph_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.make_model(embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs)

    def make_model(self, embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs):
        embedding_kwargs['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = nn.Embedding(**embedding_kwargs)
        encoder_kwargs['char_embedding_dim'] = embedding_kwargs['embedding_dim']
        self.encoder = Encoder(**encoder_kwargs)
        graph_kwargs['in_dim'] = encoder_kwargs['out_dim']
        graph_kwargs['out_dim'] = encoder_kwargs['out_dim']
        self.graph = GLCN(**graph_kwargs)
        decoder_kwargs['bilstm_kwargs']['input_size'] = encoder_kwargs['out_dim']
        if decoder_kwargs['bilstm_kwargs']['bidirectional']:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size'] * 2
        else:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size']
        decoder_kwargs['mlp_kwargs']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_kwargs['crf_kwargs']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder_kwargs)

    def _aggregate_avg_pooling(self, input, text_mask):
        """
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        """
        input = input * text_mask.detach().unsqueeze(2).float()
        sum_out = torch.sum(input, dim=1)
        text_len = text_mask.float().sum(dim=1)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()
        mean_out = sum_out.div(text_len)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        """
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        """
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)
        graph_node_mask = mask_sum != 0
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        whole_image = kwargs['whole_image']
        relation_features = kwargs['relation_features']
        text_segments = kwargs['text_segments']
        text_length = kwargs['text_length']
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None
        mask = kwargs['mask']
        boxes_coordinate = kwargs['boxes_coordinate']
        text_emb = self.word_emb(text_segments)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)
        x = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb, src_key_padding_mask=src_key_padding_mask)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        x_gcn = x_gcn * graph_node_mask.byte()
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)
        x_gcn = x_gcn.reshape(B, N, -1)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length, iob_tags_label)
        output = {'logits': logits, 'new_mask': new_mask, 'adj': adj}
        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_channels=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv2 = nn.Conv2d(512 * block.expansion, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCNLayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GraphLearningLayer,
     lambda: ([], {'in_dim': 4, 'learning_dim': 4, 'gamma': 4, 'eta': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_wenwenyu_PICK_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

