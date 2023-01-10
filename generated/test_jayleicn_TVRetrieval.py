import sys
_module = sys.modules[__name__]
del sys
master = _module
baselines = _module
clip_alignment_with_language = _module
config = _module
inference = _module
local_utils = _module
compute_proposal_upper_bound = _module
proposal = _module
mix_model_prediction = _module
model = _module
proposal_retrieval_dataset = _module
train = _module
crossmodal_moment_localization = _module
config = _module
inference = _module
model_components = _module
model_xml = _module
optimization = _module
start_end_dataset = _module
train = _module
excl = _module
config = _module
inference = _module
inference_with_vcmr = _module
model = _module
model_components = _module
optimization = _module
start_end_dataset = _module
train = _module
mixture_embedding_experts = _module
config = _module
inference = _module
model = _module
model_components = _module
retrieval_dataset = _module
train = _module
profile_main = _module
search_time_performance = _module
standalone_eval = _module
eval = _module
utils = _module
basic_utils = _module
mk_video_split_with_duration = _module
model_utils = _module
temporal_nms = _module
tensor_utils = _module
convert_sub_feature_word_to_clip = _module
lm_finetuning_on_single_sentences = _module
preprocess_subtitles = _module
convert_feature_frm_to_clip = _module
extract_i3d_features = _module
extract_image_features = _module
i3d = _module
merge_align_i3d = _module
normalize_and_concat = _module

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


import time


import torch


import math


import numpy as np


from collections import defaultdict


from collections import OrderedDict


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import logging


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


import random


from torch.utils.tensorboard import SummaryWriter


import copy


from torch.optim import Optimizer


from torch.optim.optimizer import required


from torch.nn.utils import clip_grad_norm_


import abc


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import re


from torch.utils.data import SequentialSampler


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from torchvision import models


from torchvision import transforms


class RNNEncoder(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """

    def __init__(self, word_embedding_size, hidden_size, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm', return_hidden=True, return_outputs=True, allow_zero=False):
        super(RNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.allow_zero = allow_zero
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        if self.allow_zero:
            sorted_lengths[sorted_lengths == 0] = 1
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed_inputs)
        if self.return_outputs:
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
        else:
            outputs = None
        if self.return_hidden:
            if self.rnn_type.lower() == 'lstm':
                hidden = hidden[0]
            hidden = hidden[-self.n_dirs:, :, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)
            hidden = hidden[reverse_indices]
        else:
            hidden = None
        return outputs, hidden


class CAL(nn.Module):

    def __init__(self, config):
        super(CAL, self).__init__()
        self.config = config
        self.moment_mlp = nn.Sequential(nn.Linear(config.visual_input_size, config.visual_hidden_size), nn.ReLU(True), nn.Linear(config.visual_hidden_size, config.output_size))
        self.query_lstm = RNNEncoder(word_embedding_size=config.embedding_size, hidden_size=config.lstm_hidden_size, bidirectional=False, rnn_type='lstm', dropout_p=0, n_layers=1, return_outputs=False)
        self.query_linear = nn.Linear(config.lstm_hidden_size, config.output_size)

    def moment_encoder(self, moment_feat):
        """moment_feat: (N, L_clip, D_v)"""
        return F.normalize(self.moment_mlp(moment_feat), p=2, dim=-1)

    def query_encoder(self, query_feat, query_mask):
        """
        Args:
            query_feat: (N, L_q, D_q), torch.float32
            query_mask: (N, L_q), torch.float32, with 1 indicates valid query, 0 indicates mask
        """
        _, hidden = self.query_lstm(query_feat, torch.sum(query_mask, dim=1).long())
        return F.normalize(self.query_linear(hidden), p=2, dim=-1)

    def compute_pdist(self, query_embedding, moment_feat, moment_mask):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_feat: (N, L_clip, D_v)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        moment_embedding = self.moment_encoder(moment_feat)
        moment_clip_dist = torch.sum((moment_embedding - query_embedding.unsqueeze(1)) ** 2, dim=2)
        moment_dist = torch.sum(moment_clip_dist * moment_mask, dim=1) / moment_mask.sum(1)
        return moment_dist

    @classmethod
    def compute_cdist_inference(cls, query_embeddings, moment_embeddings, moment_mask):
        """ Compute L2 distance for every possible pair of queries and proposals. This is different from
        compute_pdist as the latter computes only pairs at each row.
        Args:
            query_embeddings: (N_q, D_o)
            moment_embeddings: (N_prop, N_clips, D_o)
            moment_mask: (N_prop, N_clips)
        return:
            query_moment_scores: (N_q, N_prop)
        """
        query_device = query_embeddings.device
        if moment_embeddings.device != query_device:
            moment_embeddings = moment_embeddings
            moment_mask = moment_mask
        n_query = query_embeddings.shape[0]
        n_prop, n_clips, d = moment_embeddings.shape
        query_clip_dist = torch.cdist(query_embeddings, moment_embeddings.reshape(-1, d), p=2) ** 2
        query_clip_dist = query_clip_dist.reshape(n_query, n_prop, n_clips)
        query_moment_dist = torch.sum(query_clip_dist * moment_mask.unsqueeze(0), dim=2) / moment_mask.sum(1).unsqueeze(0)
        return query_moment_dist

    def forward(self, query_feat, query_mask, pos_moment_feat, pos_moment_mask, intra_neg_moment_feat, intra_neg_moment_mask, inter_neg_moment_feat, inter_neg_moment_mask):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            pos_moment_feat: (N, L_clip_1, D_v)
            pos_moment_mask: (N, L_clip_1)
            intra_neg_moment_feat: (N, L_clip_2, D_v)
            intra_neg_moment_mask: (N, L_clip_2)
            inter_neg_moment_feat: (N, L_clip_3, D_v)
            inter_neg_moment_mask: (N, L_clip_2)
        """
        query_embed = self.query_encoder(query_feat, query_mask)
        pos_dist = self.compute_pdist(query_embed, pos_moment_feat, pos_moment_mask)
        intra_neg_dist = self.compute_pdist(query_embed, intra_neg_moment_feat, intra_neg_moment_mask)
        if self.config.inter_loss_weight == 0:
            loss_inter = 0.0
        else:
            inter_neg_dist = self.compute_pdist(query_embed, inter_neg_moment_feat, inter_neg_moment_mask)
            loss_inter = self.calc_loss(pos_dist, inter_neg_dist)
        loss = self.calc_loss(pos_dist, intra_neg_dist) + self.config.inter_loss_weight * loss_inter
        return loss

    def calc_loss(self, pos_dist, neg_dist):
        """ Note here we encourage positive distance to be smaller than negative distance.
        Args:
            pos_dist: (N, ), torch.float32
            neg_dist: (N, ), torch.float32
        """
        if self.config.loss_type == 'hinge':
            return torch.clamp(self.config.margin + pos_dist - neg_dist, min=0).sum() / len(pos_dist)
        elif self.config.loss_type == 'lse':
            return torch.log1p(torch.exp(pos_dist - neg_dist)).sum() / len(pos_dist)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


class CALWithSub(nn.Module):

    def __init__(self, config):
        super(CALWithSub, self).__init__()
        self.config = config
        self.use_video = 'video' in config.ctx_mode
        self.use_sub = 'sub' in config.ctx_mode
        self.use_tef = 'tef' in config.ctx_mode
        self.tef_only = self.use_tef and not self.use_video and not self.use_sub
        if self.use_video or self.tef_only:
            self.video_moment_mlp = nn.Sequential(nn.Linear(config.visual_input_size, config.visual_hidden_size), nn.ReLU(True), nn.Linear(config.visual_hidden_size, config.output_size))
        if self.use_sub:
            self.sub_moment_mlp = nn.Sequential(nn.Linear(config.textual_input_size, config.visual_hidden_size), nn.ReLU(True), nn.Linear(config.visual_hidden_size, config.output_size))
        self.query_lstm = RNNEncoder(word_embedding_size=config.query_feat_size, hidden_size=config.lstm_hidden_size, bidirectional=False, rnn_type='lstm', dropout_p=0, n_layers=1, return_outputs=False)
        self.query_linear = nn.Linear(config.lstm_hidden_size, config.output_size)

    def moment_encoder(self, moment_feat, module_name='video'):
        """moment_feat: (N, L_clip, D_v)"""
        if moment_feat is not None:
            encoder = getattr(self, module_name + '_moment_mlp')
            return F.normalize(encoder(moment_feat), p=2, dim=-1)
        else:
            return None

    def query_encoder(self, query_feat, query_mask):
        """
        Args:
            query_feat: (N, L_q, D_q), torch.float32
            query_mask: (N, L_q), torch.float32, with 1 indicates valid query, 0 indicates mask
        """
        _, hidden = self.query_lstm(query_feat, torch.sum(query_mask, dim=1).long())
        return F.normalize(self.query_linear(hidden), p=2, dim=-1)

    def _compute_pdist(self, query_embedding, moment_feat, moment_mask, module_name='video'):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_feat: (N, L_clip, D_v)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        moment_embedding = self.moment_encoder(moment_feat, module_name=module_name)
        moment_clip_dist = torch.sum((moment_embedding - query_embedding.unsqueeze(1)) ** 2, dim=2)
        moment_dist = torch.sum(moment_clip_dist * moment_mask, dim=1) / moment_mask.sum(1)
        return moment_dist

    def compute_pdist(self, query_embedding, moment_video_feat, moment_sub_feat, moment_mask):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_video_feat: (N, L_clip, D_v)
            moment_sub_feat: (N, L_clip, D_t)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        divisor = (self.use_video or self.tef_only) + self.use_sub
        video_moment_dist = self._compute_pdist(query_embedding, moment_video_feat, moment_mask, module_name='video') if self.use_video or self.tef_only else 0
        sub_moment_dist = self._compute_pdist(query_embedding, moment_sub_feat, moment_mask, module_name='sub') if self.use_sub else 0
        return (video_moment_dist + sub_moment_dist) / divisor

    def _compute_cdist_inference(self, query_embeddings, moment_embeddings, moment_mask):
        """ Compute L2 distance for every possible pair of queries and proposals. This is different from
        compute_pdist as the latter computes only pairs at each row.
        Args:
            query_embeddings: (N_q, D_o)
            moment_embeddings: (N_prop, N_clips, D_o)
            moment_mask: (N_prop, N_clips)
        return:
            query_moment_scores: (N_q, N_prop)
        """
        query_device = query_embeddings.device
        if moment_embeddings.device != query_device:
            moment_embeddings = moment_embeddings
            moment_mask = moment_mask
        n_query = query_embeddings.shape[0]
        n_prop, n_clips, d = moment_embeddings.shape
        query_clip_dist = torch.cdist(query_embeddings, moment_embeddings.reshape(-1, d), p=2) ** 2
        query_clip_dist = query_clip_dist.reshape(n_query, n_prop, n_clips)
        query_moment_dist = torch.sum(query_clip_dist * moment_mask.unsqueeze(0), dim=2) / moment_mask.sum(1).unsqueeze(0)
        return query_moment_dist

    def compute_cdist_inference(self, query_embeddings, video_moment_embeddings, sub_moment_embeddings, moment_mask):
        divisor = (self.use_video or self.tef_only) + self.use_sub
        video_moment_dist = self._compute_cdist_inference(query_embeddings, video_moment_embeddings, moment_mask) if self.use_video or self.tef_only else 0
        sub_moment_dist = self._compute_cdist_inference(query_embeddings, sub_moment_embeddings, moment_mask) if self.use_sub else 0
        return (video_moment_dist + sub_moment_dist) / divisor

    def forward(self, query_feat, query_mask, pos_moment_video_feat, pos_moment_video_mask, intra_neg_moment_video_feat, intra_neg_moment_video_mask, inter_neg_moment_video_feat, inter_neg_moment_video_mask, pos_moment_sub_feat, pos_moment_sub_mask, intra_neg_moment_sub_feat, intra_neg_moment_sub_mask, inter_neg_moment_sub_feat, inter_neg_moment_sub_mask):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            pos_moment_video_feat: (N, L_clip_1, D_v)
            pos_moment_video_mask: (N, L_clip_1)
            intra_neg_moment_video_feat: (N, L_clip_2, D_v)
            intra_neg_moment_video_mask: (N, L_clip_2)
            inter_neg_moment_video_feat: (N, L_clip_3, D_v)
            inter_neg_moment_video_mask: (N, L_clip_2)
            pos_moment_sub_feat:
            pos_moment_sub_mask:
            intra_neg_moment_sub_feat:
            intra_neg_moment_sub_mask:
            inter_neg_moment_sub_feat:
            inter_neg_moment_sub_mask:
        """
        query_embed = self.query_encoder(query_feat, query_mask)
        pos_dist = self.compute_pdist(query_embed, pos_moment_video_feat, pos_moment_sub_feat, moment_mask=pos_moment_sub_mask if self.use_sub else pos_moment_video_mask)
        intra_neg_dist = self.compute_pdist(query_embed, intra_neg_moment_video_feat, intra_neg_moment_sub_feat, moment_mask=intra_neg_moment_sub_mask if self.use_sub else intra_neg_moment_video_mask)
        if self.config.inter_loss_weight == 0:
            loss_inter = 0.0
        else:
            inter_neg_dist = self.compute_pdist(query_embed, inter_neg_moment_video_feat, inter_neg_moment_sub_feat, moment_mask=inter_neg_moment_sub_mask if self.use_sub else inter_neg_moment_video_mask)
            loss_inter = self.calc_loss(pos_dist, inter_neg_dist)
        loss = self.calc_loss(pos_dist, intra_neg_dist) + self.config.inter_loss_weight * loss_inter
        return loss

    def calc_loss(self, pos_dist, neg_dist):
        """ Note here we encourage positive distance to be smaller than negative distance.
        Args:
            pos_dist: (N, ), torch.float32
            neg_dist: (N, ), torch.float32
        """
        if self.config.loss_type == 'hinge':
            return torch.clamp(self.config.margin + pos_dist - neg_dist, min=0).sum() / len(pos_dist)
        elif self.config.loss_type == 'lse':
            return torch.log1p(torch.exp(pos_dist - neg_dist)).sum() / len(pos_dist)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input_tensor = torch.randn(32, 300, 20)
        >>> output = m(input_tensor)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, relu=True):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.relu = relu
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)
        else:
            raise Exception('Incorrect dimension!')

    def forward(self, x):
        """
        :Input: (N, L_in, D)
        :Output: (N, L_out, D)
        """
        x = x.transpose(1, 2)
        if self.relu:
            out = F.relu(self.pointwise_conv(self.depthwise_conv(x)), inplace=True)
        else:
            out = self.pointwise_conv(self.depthwise_conv(x))
        return out.transpose(1, 2)


class ConvEncoder(nn.Module):

    def __init__(self, kernel_size=7, n_filters=128, dropout=0.1):
        super(ConvEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_filters)
        self.conv = DepthwiseSeparableConv(in_ch=n_filters, out_ch=n_filters, k=kernel_size, relu=True)

    def forward(self, x, mask):
        """
        :param x: (N, L, D)
        :param mask: (N, L), is not used.
        :return: (N, L, D)
        """
        return self.layer_norm(self.dropout(self.conv(x)) + x)


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(n_filters=6, max_len=10)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500, pe_type='cosine'):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        :param pe_type: cosine or linear or None
        """
        super(PositionEncoding, self).__init__()
        self.pe_type = pe_type
        if pe_type != 'none':
            position = torch.arange(0, max_len).float().unsqueeze(1)
            if pe_type == 'cosine':
                pe = torch.zeros(max_len, n_filters)
                div_term = torch.exp(torch.arange(0, n_filters, 2).float() * -(math.log(10000.0) / n_filters))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
            elif pe_type == 'linear':
                pe = position / max_len
            else:
                raise ValueError
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        if self.pe_type != 'none':
            pe = self.pe.data[:x.size(-2), :]
            extra_dim = len(x.size()) - 2
            for _ in range(extra_dim):
                pe = pe.unsqueeze(0)
            x = x + pe
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        Returns:
        """
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.0
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.ReLU(True))

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config, use_self_attention=True):
        super(BertLayer, self).__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states:  (N, L, D)
            attention_mask:  (N, L) with 1 indicate valid, 0 indicates invalid
        Returns:

        """
        if self.use_self_attention:
            attention_output = self.attention(hidden_states, attention_mask)
        else:
            attention_output = hidden_states
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def mask_logits(target, mask):
    return target * mask + (1 - mask) * -10000000000.0


class XML(nn.Module):

    def __init__(self, config):
        super(XML, self).__init__()
        self.config = config
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l, hidden_size=config.hidden_size, dropout=config.input_drop)
        self.ctx_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l, hidden_size=config.hidden_size, dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True, dropout=config.input_drop, relu=True)
        if config.encoder_type == 'transformer':
            self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size, hidden_dropout_prob=config.drop, attention_probs_dropout_prob=config.drop, num_attention_heads=config.n_heads))
        elif config.encoder_type == 'cnn':
            self.query_encoder = ConvEncoder(kernel_size=5, n_filters=config.hidden_size, dropout=config.drop)
        elif config.encoder_type in ['gru', 'lstm']:
            self.query_encoder = RNNEncoder(word_embedding_size=config.hidden_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type=config.encoder_type, return_outputs=True, return_hidden=False)
        conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=config.conv_kernel_size, stride=config.conv_stride, padding=config.conv_kernel_size // 2, bias=False)
        cross_att_cfg = edict(hidden_size=config.hidden_size, num_attention_heads=config.n_heads, attention_probs_dropout_prob=config.drop)
        self.use_video = 'video' in config.ctx_mode
        if self.use_video:
            self.video_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True, dropout=config.input_drop, relu=True)
            self.video_encoder1 = copy.deepcopy(self.query_encoder)
            self.video_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.video_cross_att = BertSelfAttention(cross_att_cfg)
                self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
            elif self.config.encoder_type == 'transformer':
                self.video_encoder3 = copy.deepcopy(self.query_encoder)
            self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)
            if config.span_predictor_type == 'conv':
                if not config.merge_two_stream:
                    self.video_st_predictor = nn.Conv1d(**conv_cfg)
                    self.video_ed_predictor = nn.Conv1d(**conv_cfg)
            elif config.span_predictor_type == 'cat_linear':
                self.video_st_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
                self.video_ed_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
        self.use_sub = 'sub' in config.ctx_mode
        if self.use_sub:
            self.sub_input_proj = LinearLayer(config.sub_input_size, config.hidden_size, layer_norm=True, dropout=config.input_drop, relu=True)
            self.sub_encoder1 = copy.deepcopy(self.query_encoder)
            self.sub_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.sub_cross_att = BertSelfAttention(cross_att_cfg)
                self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
            elif self.config.encoder_type == 'transformer':
                self.sub_encoder3 = copy.deepcopy(self.query_encoder)
            self.sub_query_linear = nn.Linear(config.hidden_size, config.hidden_size)
            if config.span_predictor_type == 'conv':
                if not config.merge_two_stream:
                    self.sub_st_predictor = nn.Conv1d(**conv_cfg)
                    self.sub_ed_predictor = nn.Conv1d(**conv_cfg)
            elif config.span_predictor_type == 'cat_linear':
                self.sub_st_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
                self.sub_ed_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size, out_features=self.use_sub + self.use_video, bias=False)
        self.temporal_criterion = nn.CrossEntropyLoss(reduction='mean')
        if config.merge_two_stream and config.span_predictor_type == 'conv':
            if self.config.stack_conv_predictor_conv_kernel_sizes == -1:
                self.merged_st_predictor = nn.Conv1d(**conv_cfg)
                self.merged_ed_predictor = nn.Conv1d(**conv_cfg)
            else:
                None
                self.merged_st_predictors = nn.ModuleList()
                self.merged_ed_predictors = nn.ModuleList()
                num_convs = len(self.config.stack_conv_predictor_conv_kernel_sizes)
                for k in self.config.stack_conv_predictor_conv_kernel_sizes:
                    conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=k, stride=config.conv_stride, padding=k // 2, bias=False)
                    self.merged_st_predictors.append(nn.Conv1d(**conv_cfg))
                    self.merged_ed_predictors.append(nn.Conv1d(**conv_cfg))
                self.combine_st_conv = nn.Linear(num_convs, 1, bias=False)
                self.combine_ed_conv = nn.Linear(num_convs, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, tef_feat, tef_mask, st_ed_indices):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
        """
        video_feat1, video_feat2, sub_feat1, sub_feat2 = self.encode_context(video_feat, video_mask, sub_feat, sub_mask)
        query_context_scores, st_prob, ed_prob = self.get_pred_from_raw_query(query_feat, query_mask, video_feat1, video_feat2, video_mask, sub_feat1, sub_feat2, sub_mask, cross=False)
        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)
        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {'loss_st_ed': float(loss_st_ed), 'loss_neg_ctx': float(loss_neg_ctx), 'loss_neg_q': float(loss_neg_q), 'loss_overall': float(loss)}

    def get_visualization_data(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, tef_feat, tef_mask, st_ed_indices):
        assert self.config.merge_two_stream and self.use_video and self.use_sub and not self.config.no_modular
        video_feat1, video_feat2, sub_feat1, sub_feat2 = self.encode_context(video_feat, video_mask, sub_feat, sub_mask)
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder, self.query_pos_embed)
        video_query, sub_query, modular_att_scores = self.get_modularized_queries(encoded_query, query_mask, return_modular_att=True)
        st_prob, ed_prob, similarity_scores, video_similarity, sub_similarity = self.get_merged_st_ed_prob(video_query, video_feat2, sub_query, sub_feat2, video_mask, cross=False, return_similaity=True)
        data = dict(modular_att_scores=modular_att_scores.cpu().numpy(), st_prob=st_prob.cpu().numpy(), ed_prob=ed_prob.cpu().numpy(), similarity_scores=similarity_scores.cpu().numpy(), video_similarity=video_similarity.cpu().numpy(), sub_similarity=sub_similarity.cpu().numpy(), st_ed_indices=st_ed_indices.cpu().numpy())
        query_lengths = query_mask.sum(1).cpu().tolist()
        ctx_lengths = video_mask.sum(1).cpu().tolist()
        for k, v in data.items():
            if k == 'modular_att_scores':
                data[k] = [e[:l] for l, e in zip(query_lengths, v)]
            else:
                data[k] = [e[:l] for l, e in zip(ctx_lengths, v)]
        datalist = []
        for idx in range(len(data['modular_att_scores'])):
            datalist.append({k: v[idx] for k, v in data.items()})
        return datalist

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder, self.query_pos_embed)
        video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)
        return video_query, sub_query

    def non_cross_encode_context(self, context_feat, context_mask, module_name='video'):
        encoder_layer3 = getattr(self, module_name + '_encoder3') if self.config.encoder_type == 'transformer' else None
        return self._non_cross_encode_context(context_feat, context_mask, input_proj_layer=getattr(self, module_name + '_input_proj'), encoder_layer1=getattr(self, module_name + '_encoder1'), encoder_layer2=getattr(self, module_name + '_encoder2'), encoder_layer3=encoder_layer3)

    def _non_cross_encode_context(self, context_feat, context_mask, input_proj_layer, encoder_layer1, encoder_layer2, encoder_layer3=None):
        """
        Args:
            context_feat: (N, L, D)
            context_mask: (N, L)
            input_proj_layer:
            encoder_layer1:
            encoder_layer2:
            encoder_layer3
        """
        context_feat1 = self.encode_input(context_feat, context_mask, input_proj_layer, encoder_layer1, self.ctx_pos_embed)
        if self.config.encoder_type in ['transformer', 'cnn']:
            context_mask = context_mask.unsqueeze(1)
            context_feat2 = encoder_layer2(context_feat1, context_mask)
            if self.config.encoder_type == 'transformer':
                context_feat2 = encoder_layer3(context_feat2, context_mask)
        elif self.config.encoder_type in ['gru', 'lstm']:
            context_mask = context_mask.sum(1).long()
            context_feat2 = encoder_layer2(context_feat1, context_mask)[0]
        else:
            raise NotImplementedError
        return context_feat1, context_feat2

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        if self.config.cross_att:
            assert self.use_video and self.use_sub
            return self.cross_encode_context(video_feat, video_mask, sub_feat, sub_mask)
        else:
            video_feat1, video_feat2 = (None,) * 2
            if self.use_video:
                video_feat1, video_feat2 = self.non_cross_encode_context(video_feat, video_mask, module_name='video')
            sub_feat1, sub_feat2 = (None,) * 2
            if self.use_sub:
                sub_feat1, sub_feat2 = self.non_cross_encode_context(sub_feat, sub_mask, module_name='sub')
            return video_feat1, video_feat2, sub_feat1, sub_feat2

    def cross_encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder1, self.ctx_pos_embed)
        encoded_sub_feat = self.encode_input(sub_feat, sub_mask, self.sub_input_proj, self.sub_encoder1, self.ctx_pos_embed)
        x_encoded_video_feat = self.cross_context_encoder(encoded_video_feat, video_mask, encoded_sub_feat, sub_mask, self.video_cross_att, self.video_cross_layernorm, self.video_encoder2)
        x_encoded_sub_feat = self.cross_context_encoder(encoded_sub_feat, sub_mask, encoded_video_feat, video_mask, self.sub_cross_att, self.sub_cross_layernorm, self.sub_encoder2)
        return encoded_video_feat, x_encoded_video_feat, encoded_sub_feat, x_encoded_sub_feat

    def cross_context_encoder(self, main_context_feat, main_context_mask, side_context_feat, side_context_mask, cross_att_layer, norm_layer, self_att_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer:
            norm_layer:
            self_att_layer:
        """
        cross_mask = torch.einsum('bm,bn->bmn', main_context_mask, side_context_mask)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)
        residual_out = norm_layer(cross_out + main_context_feat)
        if self.config.encoder_type in ['cnn', 'transformer']:
            return self_att_layer(residual_out, main_context_mask.unsqueeze(1))
        elif self.config.encoder_type in ['gru', 'lstm']:
            return self_att_layer(residual_out, main_context_mask.sum(1).long())[0]

    def encode_input(self, feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            # add_pe: bool, whether to add positional encoding
            pos_embed_layer
        """
        feat = input_proj_layer(feat)
        if self.config.encoder_type in ['cnn', 'transformer']:
            feat = pos_embed_layer(feat)
            mask = mask.unsqueeze(1)
            return encoder_layer(feat, mask)
        elif self.config.encoder_type in ['gru', 'lstm']:
            if self.config.add_pe_rnn:
                feat = pos_embed_layer(feat)
            mask = mask.sum(1).long()
            return encoder_layer(feat, mask)[0]

    def get_modularized_queries(self, encoded_query, query_mask, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if self.config.no_modular:
            modular_query = torch.max(mask_logits(encoded_query, query_mask.unsqueeze(2)), dim=1)[0]
            return modular_query, modular_query
        else:
            modular_attention_scores = self.modular_vector_mapping(encoded_query)
            modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
            modular_queries = torch.einsum('blm,bld->bmd', modular_attention_scores, encoded_query)
            if return_modular_att:
                assert modular_queries.shape[1] == 2
                return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
            elif modular_queries.shape[1] == 2:
                return modular_queries[:, 0], modular_queries[:, 1]
            else:
                return modular_queries[:, 0], modular_queries[:, 0]

    def get_modular_weights(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
        """
        max_encoded_query, _ = torch.max(mask_logits(encoded_query, query_mask.unsqueeze(2)), dim=1)
        modular_weights = self.modular_weights_calculator(max_encoded_query)
        modular_weights = F.softmax(modular_weights, dim=-1)
        return modular_weights[:, 0:1], modular_weights[:, 1:2]

    def get_video_level_scores(self, modularied_query, context_feat1, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat1: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum('md,nld->mln', modularied_query, context_feat1)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)
        query_context_scores = mask_logits(query_context_scores, context_mask)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return query_context_scores

    def get_merged_st_ed_prob(self, video_query, video_feat, sub_query, sub_feat, context_mask, cross=False, return_similaity=False):
        """context_mask could be either video_mask or sub_mask, since they are the same"""
        assert self.use_video and self.use_sub and self.config.span_predictor_type == 'conv'
        video_query = self.video_query_linear(video_query)
        sub_query = self.sub_query_linear(sub_query)
        stack_conv = self.config.stack_conv_predictor_conv_kernel_sizes != -1
        num_convs = len(self.config.stack_conv_predictor_conv_kernel_sizes) if stack_conv else None
        if cross:
            video_similarity = torch.einsum('md,nld->mnl', video_query, video_feat)
            sub_similarity = torch.einsum('md,nld->mnl', sub_query, sub_feat)
            similarity = (video_similarity + sub_similarity) / 2
            n_q, n_c, l = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, l)
            if not stack_conv:
                st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, l)
                ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, l)
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](similarity).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](similarity).squeeze().unsqueeze(2))
                st_prob = self.combine_st_conv(torch.cat(st_prob_list, dim=2)).view(n_q, n_c, l)
                ed_prob = self.combine_ed_conv(torch.cat(ed_prob_list, dim=2)).view(n_q, n_c, l)
        else:
            video_similarity = torch.einsum('bd,bld->bl', video_query, video_feat)
            sub_similarity = torch.einsum('bd,bld->bl', sub_query, sub_feat)
            similarity = (video_similarity + sub_similarity) / 2
            if not stack_conv:
                st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()
                ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                st_prob = self.combine_st_conv(torch.cat(st_prob_list, dim=2)).squeeze()
                ed_prob = self.combine_ed_conv(torch.cat(ed_prob_list, dim=2)).squeeze()
        st_prob = mask_logits(st_prob, context_mask)
        ed_prob = mask_logits(ed_prob, context_mask)
        if return_similaity:
            assert not cross
            return st_prob, ed_prob, similarity, video_similarity, sub_similarity
        else:
            return st_prob, ed_prob

    def get_st_ed_prob(self, modularied_query, context_feat2, context_mask, module_name='video', cross=False):
        return self._get_st_ed_prob(modularied_query, context_feat2, context_mask, module_query_linear=getattr(self, module_name + '_query_linear'), st_predictor=getattr(self, module_name + '_st_predictor'), ed_predictor=getattr(self, module_name + '_ed_predictor'), cross=cross)

    def _get_st_ed_prob(self, modularied_query, context_feat2, context_mask, module_query_linear, st_predictor, ed_predictor, cross=False):
        """
        Args:
            modularied_query: (N, D)
            context_feat2: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
            module_query_linear:
            st_predictor:
            ed_predictor:
            cross: at inference, calculate prob for each possible pairs of query and context.
        """
        query = module_query_linear(modularied_query)
        if cross:
            if self.config.span_predictor_type == 'conv':
                similarity = torch.einsum('md,nld->mnl', query, context_feat2)
                n_q, n_c, l = similarity.shape
                similarity = similarity.view(n_q * n_c, 1, l)
                st_prob = st_predictor(similarity).view(n_q, n_c, l)
                ed_prob = ed_predictor(similarity).view(n_q, n_c, l)
            elif self.config.span_predictor_type == 'cat_linear':
                st_prob_q = st_predictor[0](query).unsqueeze(1)
                st_prob_ctx = st_predictor[1](context_feat2).squeeze().unsqueeze(0)
                st_prob = st_prob_q + st_prob_ctx
                ed_prob_q = ed_predictor[0](query).unsqueeze(1)
                ed_prob_ctx = ed_predictor[1](context_feat2).squeeze().unsqueeze(0)
                ed_prob = ed_prob_q + ed_prob_ctx
            context_mask = context_mask.unsqueeze(0)
        elif self.config.span_predictor_type == 'conv':
            similarity = torch.einsum('bd,bld->bl', query, context_feat2)
            st_prob = st_predictor(similarity.unsqueeze(1)).squeeze()
            ed_prob = ed_predictor(similarity.unsqueeze(1)).squeeze()
        elif self.config.span_predictor_type == 'cat_linear':
            st_prob = st_predictor[0](query) + st_predictor[1](context_feat2).squeeze()
            ed_prob = ed_predictor[0](query) + ed_predictor[1](context_feat2).squeeze()
        st_prob = mask_logits(st_prob, context_mask)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask, video_feat1, video_feat2, video_mask, sub_feat1, sub_feat2, sub_mask, cross=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat1: (N, Lv, D) or None
            video_feat2:
            video_mask: (N, Lv)
            sub_feat1: (N, Lv, D) or None
            sub_feat2:
            sub_mask: (N, Lv)
            cross:
        """
        video_query, sub_query = self.encode_query(query_feat, query_mask)
        divisor = self.use_sub + self.use_video
        video_q2ctx_scores = self.get_video_level_scores(video_query, video_feat1, video_mask) if self.use_video else 0
        sub_q2ctx_scores = self.get_video_level_scores(sub_query, sub_feat1, sub_mask) if self.use_sub else 0
        q2ctx_scores = (video_q2ctx_scores + sub_q2ctx_scores) / divisor
        if self.config.merge_two_stream and self.use_video and self.use_sub:
            st_prob, ed_prob = self.get_merged_st_ed_prob(video_query, video_feat2, sub_query, sub_feat2, video_mask, cross=cross)
        else:
            video_st_prob, video_ed_prob = self.get_st_ed_prob(video_query, video_feat2, video_mask, module_name='video', cross=cross) if self.use_video else (0, 0)
            sub_st_prob, sub_ed_prob = self.get_st_ed_prob(sub_query, sub_feat2, sub_mask, module_name='sub', cross=cross) if self.use_sub else (0, 0)
            st_prob = (video_st_prob + sub_st_prob) / divisor
            ed_prob = (video_ed_prob + sub_ed_prob) / divisor
        return q2ctx_scores, st_prob, ed_prob

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1), query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx, size=(bsz,))]
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == 'hinge':
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == 'lse':
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


class EXCL(nn.Module):

    def __init__(self, config):
        super(EXCL, self).__init__()
        self.config = config
        self.use_video = 'video' in config.ctx_mode
        self.use_sub = 'sub' in config.ctx_mode
        self.query_encoder = RNNEncoder(word_embedding_size=config.query_input_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type='lstm', return_outputs=False, return_hidden=True)
        if self.use_video:
            self.video_encoder = RNNEncoder(word_embedding_size=config.visual_input_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type='lstm', return_outputs=True, return_hidden=False)
            self.video_encoder2 = RNNEncoder(word_embedding_size=2 * config.hidden_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type='lstm', return_outputs=True, return_hidden=False)
            self.video_st_predictor = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size), nn.Tanh(), nn.Linear(config.hidden_size, 1))
            self.video_ed_predictor = copy.deepcopy(self.video_st_predictor)
        if self.use_sub:
            self.sub_encoder = RNNEncoder(word_embedding_size=config.sub_input_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type='lstm', return_outputs=True, return_hidden=False)
            self.sub_encoder2 = RNNEncoder(word_embedding_size=2 * config.hidden_size, hidden_size=config.hidden_size // 2, bidirectional=True, n_layers=1, rnn_type='lstm', return_outputs=True, return_hidden=False)
            self.sub_st_predictor = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size), nn.Tanh(), nn.Linear(config.hidden_size, 1))
            self.sub_ed_predictor = copy.deepcopy(self.video_st_predictor)
        self.temporal_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)

    def get_prob_single_stream(self, encoded_query, ctx_feat, ctx_mask, module_name=None):
        ctx_mask_rnn = ctx_mask.sum(1).long()
        ctx_feat1 = getattr(self, module_name + '_encoder')(F.dropout(ctx_feat, p=self.config.drop, training=self.training), ctx_mask_rnn)[0]
        ctx_feat2 = getattr(self, module_name + '_encoder2')(F.dropout(torch.cat([ctx_feat1, encoded_query], dim=-1), p=self.config.drop, training=self.training), ctx_mask_rnn)[0]
        ctx_feat3 = torch.cat([ctx_feat2, ctx_feat1, encoded_query], dim=2)
        st_probs = getattr(self, module_name + '_st_predictor')(ctx_feat3).squeeze()
        ed_probs = getattr(self, module_name + '_ed_predictor')(ctx_feat3).squeeze()
        st_probs = mask_logits(st_probs, ctx_mask)
        ed_probs = mask_logits(ed_probs, ctx_mask)
        return st_probs, ed_probs

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, tef_feat, tef_mask, st_ed_indices, is_training=True):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            is_training:
        """
        query_mask = query_mask.sum(1).long()
        encoded_query = self.query_encoder(query_feat, query_mask)[1]
        encoded_query = encoded_query.unsqueeze(1).repeat(1, video_feat.shape[1], 1)
        video_st_prob, video_ed_prob = self.get_prob_single_stream(encoded_query, video_feat, video_mask, module_name='video') if self.use_video else (0, 0)
        sub_st_prob, sub_ed_prob = self.get_prob_single_stream(encoded_query, sub_feat, sub_mask, module_name='sub') if self.use_sub else (0, 0)
        st_prob = (video_st_prob + sub_st_prob) / (self.use_video + self.use_sub)
        ed_prob = (video_ed_prob + sub_ed_prob) / (self.use_video + self.use_sub)
        if is_training:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
            return loss_st_ed, {'loss_st_ed': float(loss_st_ed)}, st_prob, ed_prob
        else:
            prob_product = torch.einsum('bm,bn->bmn', st_prob, ed_prob)
            prob_product = torch.triu(prob_product)
            prob_product = prob_product.view(prob_product.shape[0], -1)
            prob_product = torch.topk(prob_product, k=100, dim=1, largest=True)
            return None


class ContextGating(nn.Module):

    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbeddingUnit(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(GatedEmbeddingUnit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]
        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)
        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)
        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))
        return max_margin.mean()


class NetVLAD(nn.Module):

    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter(1 / math.sqrt(feature_size) * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter(1 / math.sqrt(feature_size) * torch.randn(1, feature_size, cluster_size))
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)
        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)
        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2
        assignment = assignment.transpose(1, 2)
        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a
        vlad = F.normalize(vlad)
        vlad = vlad.view(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)
        return vlad


class MEE(nn.Module):

    def __init__(self, config):
        super(MEE, self).__init__()
        self.config = config
        self.use_video = 'video' in config.ctx_mode
        self.use_sub = 'sub' in config.ctx_mode
        self.query_pooling = NetVLAD(feature_size=config.text_input_size, cluster_size=2)
        if self.use_sub:
            self.sub_query_gu = GatedEmbeddingUnit(input_dimension=self.query_pooling.out_dim, output_dimension=config.output_size)
            self.sub_gu = GatedEmbeddingUnit(input_dimension=config.text_input_size, output_dimension=config.output_size)
        if self.use_video:
            self.video_query_gu = GatedEmbeddingUnit(input_dimension=self.query_pooling.out_dim, output_dimension=config.output_size)
            self.video_gu = GatedEmbeddingUnit(input_dimension=config.vid_input_size, output_dimension=config.output_size)
        if self.use_video and self.use_sub:
            self.moe_fc = nn.Linear(self.query_pooling.out_dim, 2)
        self.max_margin_loss = MaxMarginRankingLoss(margin=config.margin)

    def forward(self, query_feat, query_mask, video_feat, sub_feat):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            video_feat: (N, Dv)
            sub_feat: (N, Dt)
        """
        pooled_query = self.query_pooling(query_feat)
        encoded_video, encoded_sub = self.encode_context(video_feat, sub_feat)
        confusion_matrix = self.get_score_from_pooled_query_with_encoded_ctx(pooled_query, encoded_video, encoded_sub)
        return self.max_margin_loss(confusion_matrix)

    def encode_context(self, video_feat, sub_feat):
        """(N, D)"""
        encoded_video = self.video_gu(video_feat) if self.use_video else None
        encoded_sub = self.sub_gu(sub_feat) if self.use_sub else None
        return encoded_video, encoded_sub

    def compute_single_stream_scores_with_encoded_ctx(self, pooled_query, encoded_ctx, module_name='video'):
        encoded_query = getattr(self, module_name + '_query_gu')(pooled_query)
        return torch.einsum('md,nd->mn', encoded_query, encoded_ctx)

    def get_score_from_pooled_query_with_encoded_ctx(self, pooled_query, encoded_video, encoded_sub):
        """Nq may not equal to Nc
        Args:
            pooled_query: (Nq, Dt)
            encoded_video: (Nc, Dc)
            encoded_sub: (Nc, Dc)
        """
        video_confusion_matrix = self.compute_single_stream_scores_with_encoded_ctx(pooled_query, encoded_video, module_name='video') if self.use_video else 0
        sub_confusion_matrix = self.compute_single_stream_scores_with_encoded_ctx(pooled_query, encoded_sub, module_name='sub') if self.use_sub else 0
        if self.use_video and self.use_sub:
            stream_weights = self.moe_fc(pooled_query)
            confusion_matrix = stream_weights[:, 0:1] * video_confusion_matrix + stream_weights[:, 1:2] * sub_confusion_matrix
        else:
            confusion_matrix = video_confusion_matrix + sub_confusion_matrix
        return confusion_matrix


class ImageNetResNetFeature(nn.Module):

    def __init__(self, output_dim='2048'):
        super(ImageNetResNetFeature, self).__init__()
        resnet = models.resnet152(pretrained=True)
        if output_dim == '2048':
            n_layers_to_rm = 1
        elif output_dim == '2048x7x7':
            n_layers_to_rm = 2
        else:
            raise ValueError('Wrong value for argument output_dim')
        self.feature = nn.Sequential(*list(resnet.children())[:-n_layers_to_rm])

    def forward(self, x):
        """return: B x 2048 or B x 2048x7x7"""
        return self.feature(x).squeeze()


class ResNetC3FeatureExtractor(nn.Module):

    def __init__(self):
        super(ResNetC3FeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=True)
        component_list = list(resnet.children())[:-3]
        component_list.extend(list(resnet.layer4.children())[:2])
        self.resnet_base = nn.Sequential(*component_list)
        layer4_children = list(resnet.layer4.children())[2]
        self.layer4_head = nn.Sequential(layer4_children.conv1, layer4_children.bn1, layer4_children.relu, layer4_children.conv2, layer4_children.bn2, layer4_children.relu)

    def forward(self, x):
        base_out = self.resnet_base(x)
        c3_feature = self.layer4_head(base_out)
        return c3_feature


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertLayer,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextGating,
     lambda: ([], {'dimension': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 128]), torch.rand([4, 4])], {}),
     True),
    (DepthwiseSeparableConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GatedEmbeddingUnit,
     lambda: ([], {'input_dimension': 4, 'output_dimension': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ImageNetResNetFeature,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (LinearLayer,
     lambda: ([], {'in_hsz': 4, 'out_hsz': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxMarginRankingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PositionEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 128])], {}),
     True),
    (ResNetC3FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TrainablePositionalEncoding,
     lambda: ([], {'max_position_embeddings': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jayleicn_TVRetrieval(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

