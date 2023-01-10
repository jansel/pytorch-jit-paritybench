import sys
_module = sys.modules[__name__]
del sys
config = _module
eval_tvqa_plus = _module
maskrcnn_voc = _module
bounding_box = _module
boxlist_ops = _module
voc_eval = _module
utils = _module
inference = _module
main = _module
model = _module
cnn = _module
context_query_attention = _module
encoder = _module
model_utils = _module
position_encoding = _module
self_attention = _module
stage = _module
tvqa_dataset = _module
utils = _module

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


import numpy as np


from collections import defaultdict


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


import math


import copy


from torch.utils.data.dataset import Dataset


import re


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


class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dim=1, stride=1, padding=0, relu=True, dropout=0.1):
        """
        :param in_channels: input hidden dimension size
        :param out_channels: output hidden dimension size
        :param kernel_size: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(ConvRelu, self).__init__()
        self.relu = relu
        self.dropout = dropout
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise Exception('Incorrect dimension!')

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        x = F.dropout(x, training=self.training, p=self.dropout)
        if self.relu:
            return F.relu(self.conv(x), inplace=True)
        else:
            return self.conv(x)


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, n_filters=128, kernel_size=7, padding=3):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, groups=n_filters)
        self.separable = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.separable(x)
        return x


class StructuredAttention(nn.Module):
    """Use each word in context to attend to words in query.
    In my case, context is question-answer, query is the object-level
    features in an image.

    Note the values in S are cosine similarity scores, and are in [-1, 1]
    They are scaled before softmax to make sure the maximum value could
    get very high probability.
    S_ = F.softmax(S * self.scale, dim=-1)

    Consider softmax function f(m) = exp(m) / [24 * exp(-m) + exp(m)]
    If not scaled, S * scale \\in [-100, 100], the weight the maximum value could only get is
    exp(1) / [24 * exp(-1) + exp(1)] = 0.04 .
    When set the scale = 100, S * scale \\in [-100, 100]
    exp(100) / [24 * exp(-100) + exp(100)] = 0.9976
    """

    def __init__(self, dropout=0.1, scale=100, add_void=False):
        """
        Args:
            dropout:
            scale:
            add_void:
        """
        super(StructuredAttention, self).__init__()
        self.dropout = dropout
        self.scale = scale
        self.add_void = add_void

    def forward(self, C, Q, c_mask, q_mask, noun_mask=None, void_vector=None):
        """
        match the dim of '*', singlton is allowed
        Args:
            C: (N, 5, Li, Lqa, D)
            Q: (N, 1, Li, Lr, D)
            c_mask: (N, 5, Li, Lqa)
            q_mask: (N, 1, Li, Lr)
            noun_mask: (N, 5, Lqa) , where 1 indicate the current position is a noun
               or (N, 5, Li, Lqa), where each entry is the probability of the current
               image being a positive bag for the word
            void_vector: (D, )
        Returns:
            (N, *, Lc, D)
        """
        bsz, _, num_img, num_region, hsz = Q.shape
        if void_vector is not None:
            num_void = len(void_vector)
            Q_void = void_vector.view(1, 1, 1, num_void, hsz).repeat(bsz, 1, num_img, 1, 1)
            Q = torch.cat([Q, Q_void], dim=-2)
            q_mask_void = q_mask.new_ones(bsz, 1, num_img, num_void)
            q_mask = torch.cat([q_mask, q_mask_void], dim=-1)
        S, S_mask = self.similarity(C, Q, c_mask, q_mask)
        S_ = F.softmax(S * self.scale, dim=-1)
        S_ = S_ * S_mask
        if noun_mask is not None:
            if len(noun_mask.shape) == 3:
                bsz, num_qa, lqa = noun_mask.shape
                S_ = S_ * noun_mask.view(bsz, num_qa, 1, lqa, 1)
            elif len(noun_mask.shape) == 4:
                S_ = S_ * noun_mask.unsqueeze(-1)
            else:
                raise NotImplementedError
        if void_vector is not None:
            if self.add_void:
                A = torch.matmul(S_, Q)
                S, S_mask, S_ = S[:, :, :, :, :-num_void], S_mask[:, :, :, :, :-num_void], S_[:, :, :, :, :-num_void]
            else:
                S, S_mask, S_ = S[:, :, :, :, :-num_void], S_mask[:, :, :, :, :-num_void], S_[:, :, :, :, :-num_void]
                Q = Q[:, :, :, :-num_void, :]
                A = torch.matmul(S_, Q)
        else:
            A = torch.matmul(S_, Q)
        return A, S, S_mask, S_

    def similarity(self, C, Q, c_mask, q_mask):
        """
        word2word dot-product similarity
        Args:
            C: (N, 5, Li, Lqa, D)
            Q: (N, 1, Li, Lr, D)
            c_mask: (N, 5, Li, Lqa)
            q_mask: (N, 1, Li, Lr)
        Returns:
            (N, *, Lc, Lq)
        """
        C = F.dropout(F.normalize(C, p=2, dim=-1), p=self.dropout, training=self.training)
        Q = F.dropout(F.normalize(Q, p=2, dim=-1), p=self.dropout, training=self.training)
        S_mask = torch.matmul(c_mask.unsqueeze(-1), q_mask.unsqueeze(-2))
        S = torch.matmul(C, Q.transpose(-2, -1))
        masked_S = S - 10000000000.0 * (1 - S_mask)
        return masked_S, S_mask


class ContextQueryAttention(nn.Module):
    """
    sub-a attention
    """

    def __init__(self):
        super(ContextQueryAttention, self).__init__()

    def forward(self, C, Q, c_mask, q_mask):
        """
        match the dim of '*', singlton is allowed
        :param C: (N, *, Lc, D)
        :param Q: (N, *, Lq, D)
        :param c_mask: (N, *, Lc)
        :param q_mask: (N, *, Lq)
        :return: (N, Lc, D) and (N, Lq, D)
        """
        S = self.similarity(C, Q, c_mask, q_mask)
        S_ = F.softmax(S, dim=-1)
        A = torch.matmul(S_, Q)
        return A

    def similarity(self, C, Q, c_mask, q_mask):
        """
        word2word dot-product similarity
        :param C: (N, *, Lc, D)
        :param Q: (N, *, Lq, D)
        :param c_mask: (N, *, Lc)
        :param q_mask: (N, *, Lq)
        :return: (N, *, Lc, Lq)
        """
        C = F.dropout(C, p=0.1, training=self.training)
        Q = F.dropout(Q, p=0.1, training=self.training)
        hsz_root = math.sqrt(C.shape[-1])
        S_mask = torch.matmul(c_mask.unsqueeze(-1), q_mask.unsqueeze(-2))
        S = torch.matmul(C, Q.transpose(-2, -1)) / hsz_root
        masked_S = S - 10000000000.0 * (1 - S_mask)
        return masked_S


def clones(module, n):
    """Produce n identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadedAttention(nn.Module):

    def __init__(self, nh, d_model, dropout=0.1):
        """
        Args:
            nh (int): number of heads
            d_model (int): input hidden size
            dropout:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nh == 0
        self.d_k = d_model // nh
        self.nh = nh
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L, D)
            mask: (N, L)
        """
        bsz = x.size(0)
        if mask is not None:
            mask = mask.view(bsz, 1, -1, 1)
        query, key, value = [l(x).view(bsz, -1, self.nh, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (x, x, x))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.nh * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        """ Compute 'Scaled Dot Product Attention'
        Args:
            query: (N, nh, L, d_k)
            key: (N, nh, L, d_k)
            value: (N, nh, L, d_k)
            mask: (N, 1, L, 1)
            dropout:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, n_filters)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * -(math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class EncoderBlock(nn.Module):

    def __init__(self, n_conv, kernel_size=7, n_filters=128, dropout=0.1, num_heads=4):
        super(EncoderBlock, self).__init__()
        self.dropout = dropout
        self.n_conv = n_conv
        self.num_heads = num_heads
        self.position_encoding = PositionEncoding(n_filters=n_filters)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(n_filters) for _ in range(n_conv)])
        self.final_layer_norm = nn.LayerNorm(n_filters)
        self.conv = nn.ModuleList([DepthwiseSeparableConv(in_ch=n_filters, out_ch=n_filters, k=kernel_size, relu=True) for _ in range(n_conv)])
        if self.num_heads != 0:
            self.multi_head_attn = MultiHeadedAttention(nh=num_heads, d_model=n_filters)
            self.attn_layer_norm = nn.LayerNorm(n_filters)

    def forward(self, x, mask):
        """
        :param x: (N, L, D)
        :param mask: (N, L)
        :return: (N, L, D)
        """
        outputs = self.position_encoding(x)
        for i in range(self.n_conv):
            residual = outputs
            outputs = self.layer_norm[i](outputs)
            if i % 2 == 0:
                outputs = F.dropout(outputs, p=self.dropout, training=self.training)
            outputs = self.conv[i](outputs)
            outputs = outputs + residual
        if self.num_heads != 0:
            residual = outputs
            outputs = self.attn_layer_norm(outputs)
            outputs = self.multi_head_attn(outputs, mask=mask)
            outputs = outputs + residual
        return self.final_layer_norm(outputs)


class StackedEncoder(nn.Module):

    def __init__(self, n_blocks=7, n_conv=2, kernel_size=7, hidden_size=128, dropout=0.1, num_heads=4):
        super(StackedEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv, kernel_size=kernel_size, n_filters=hidden_size, dropout=dropout, num_heads=num_heads) for _ in range(n_blocks)])

    def forward(self, x, mask):
        """
        :param x: # (N, L, D)
        :param mask:  # (N, L)
        :return:  (N, L, D)
        """
        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask)
        return x


class NormalizeScale(nn.Module):

    def __init__(self, dim, num_additional_dims=1, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        dims = [1] * num_additional_dims + [dim]
        self.weight = nn.Parameter(torch.ones(dims) * init_norm)

    def forward(self, bottom):
        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


class LinearWrapper(nn.Module):
    """1D conv layer"""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearWrapper, self).__init__()
        self.relu = relu
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.relu:
            return F.relu(self.conv(x), inplace=True)
        else:
            return self.conv(x)


class ConvLinear(nn.Module):
    """1D conv layer"""

    def __init__(self, in_hsz, out_hsz, kernel_size=3, layer_norm=True, dropout=0.1, relu=True):
        super(ConvLinear, self).__init__()
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [nn.Dropout(dropout), DepthwiseSeparableConv(in_ch=in_hsz, out_ch=out_hsz, k=kernel_size, dim=1, relu=relu)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        return self.conv(x)


def expand_span(span, expand_length=2):
    """
    Args:
        span (list): [st, ed]
        expand_length (int): length to add on the two sides

    Returns:
        expanded_span (list): [max(0, st-expand_length), ed + expand_length]
            Only use the span for indexing, no need to worry the case where
            (ed + expand_length) >= max_length.
    """
    return [max(0, span[0] - expand_length), span[1] + expand_length]


def topN_array_2d(array_2d, topN=None):
    """ Get topN indices and values of a 2d array, return a tuple of indices and their values,
    ranked by the value
    """
    row_indices, column_indices = np.unravel_index(np.argsort(array_2d, axis=None), array_2d.shape)
    row_indices = row_indices[::-1][:topN]
    column_indices = column_indices[::-1][:topN]
    sorted_values = array_2d[row_indices, column_indices]
    sorted_triples = zip(row_indices, column_indices, sorted_values)
    return sorted_triples


def find_max_triples(p1, p2, topN=5, prob_thd=None):
    """ Find a list of (k1, k2) where k1 >= k2 with the maximum values of p1[k1] * p2[k2]
    Args:
        p1 (torch.CudaTensor): (N, L) batched start_idx probabilities
        p2 (torch.CudaTensor): (N, L) batched end_idx probabilities
        topN (int): return topN pairs with highest values
        prob_thd (float):
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    product = torch.bmm(p1.unsqueeze(2), p2.unsqueeze(1))
    upper_product = torch.stack([torch.triu(p) for p in product]).data.cpu().numpy()
    batched_sorted_triple = []
    for idx, e in enumerate(upper_product):
        sorted_triple = topN_array_2d(e, topN=topN)
        if prob_thd is not None:
            sorted_triple = [t for t in sorted_triple if t[2] >= prob_thd]
        batched_sorted_triple.append(sorted_triple)
    return batched_sorted_triple


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def compute_temporal_iou(pred, gt):
    """ compute intersection-over-union along temporal axis
    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def get_high_iou_sapns(gt_ts_list, pred_ts_list, iou_thd=0.5, add_gt=True):
    """ Note
    Args:
        gt_ts_list: N * (st, ed)
        pred_ts_list: N * [(st_idx, ed_idx, confidence), ...]
        iou_thd (float):
        add_gt (bool):
    Returns:

    """
    spans = []
    for idx, (gt_ts, pred_ts_sublist) in enumerate(zip(gt_ts_list, pred_ts_list)):
        if add_gt:
            cur_spans = [gt_ts]
        else:
            cur_spans = []
        for pred_ts in pred_ts_sublist:
            pred_ts = pred_ts[:2]
            if compute_temporal_iou(pred_ts, gt_ts) >= iou_thd:
                cur_spans.append(pred_ts)
        spans.append(cur_spans)
    return spans


def mask_logits(target, mask):
    return target * mask + (1 - mask) * -10000000000.0


def save_pickle(data, data_path, highest=False):
    protocol = 2 if highest else 0
    with open(data_path, 'w') as f:
        pickle.dump(data, f, protocol=protocol)


class STAGE(nn.Module):

    def __init__(self, opt):
        super(STAGE, self).__init__()
        self.opt = opt
        self.inference_mode = False
        self.sub_flag = opt.sub_flag
        self.vfeat_flag = opt.vfeat_flag
        self.vfeat_size = opt.vfeat_size
        self.t_iter = opt.t_iter
        self.extra_span_length = opt.extra_span_length
        self.add_local = opt.add_local
        self.use_sup_att = opt.use_sup_att
        self.num_negatives = opt.num_negatives
        self.negative_pool_size = opt.negative_pool_size
        self.num_hard = opt.num_hard
        self.drop_topk = opt.drop_topk
        self.margin = opt.margin
        self.att_loss_type = opt.att_loss_type
        self.scale = opt.scale
        self.alpha = opt.alpha
        self.dropout = opt.dropout
        self.hsz = opt.hsz
        self.bsz = None
        self.num_seg = None
        self.num_a = 5
        self.flag_cnt = self.sub_flag + self.vfeat_flag
        self.wd_size = opt.embedding_size
        self.bridge_hsz = 300
        self.bert_word_encoding_fc = nn.Sequential(nn.LayerNorm(self.wd_size), nn.Dropout(self.dropout), nn.Linear(self.wd_size, self.bridge_hsz), nn.ReLU(True), nn.LayerNorm(self.bridge_hsz))
        if self.sub_flag:
            None
        if self.vfeat_flag:
            None
            self.vid_fc = nn.Sequential(nn.LayerNorm(self.vfeat_size), nn.Dropout(self.dropout), nn.Linear(self.vfeat_size, self.bridge_hsz), nn.ReLU(True), nn.LayerNorm(self.bridge_hsz))
        if self.flag_cnt == 2:
            self.concat_fc = nn.Sequential(nn.LayerNorm(3 * self.hsz), nn.Dropout(self.dropout), nn.Linear(3 * self.hsz, self.hsz), nn.ReLU(True), nn.LayerNorm(self.hsz))
        self.input_embedding = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.bridge_hsz, self.hsz), nn.ReLU(True), nn.LayerNorm(self.hsz))
        self.input_encoder = StackedEncoder(n_blocks=opt.input_encoder_n_blocks, n_conv=opt.input_encoder_n_conv, kernel_size=opt.input_encoder_kernel_size, num_heads=opt.input_encoder_n_heads, hidden_size=self.hsz, dropout=self.dropout)
        self.str_attn = StructuredAttention(dropout=self.dropout, scale=opt.scale, add_void=opt.add_non_visual)
        self.c2q_down_projection = nn.Sequential(nn.LayerNorm(3 * self.hsz), nn.Dropout(self.dropout), nn.Linear(3 * self.hsz, self.hsz), nn.ReLU(True))
        self.cls_encoder = StackedEncoder(n_blocks=opt.cls_encoder_n_blocks, n_conv=opt.cls_encoder_n_conv, kernel_size=opt.cls_encoder_kernel_size, num_heads=opt.cls_encoder_n_heads, hidden_size=self.hsz, dropout=self.dropout)
        self.cls_projection_layers = nn.ModuleList([LinearWrapper(in_hsz=self.hsz, out_hsz=self.hsz, layer_norm=True, dropout=self.dropout, relu=True)] + [ConvLinear(in_hsz=self.hsz, out_hsz=self.hsz, kernel_size=3, layer_norm=True, dropout=self.dropout, relu=True) for _ in range(self.t_iter)])
        self.temporal_scoring_st_layers = nn.ModuleList([LinearWrapper(in_hsz=self.hsz, out_hsz=1, layer_norm=True, dropout=self.dropout, relu=False) for _ in range(self.t_iter + 1)])
        self.temporal_scoring_ed_layers = nn.ModuleList([LinearWrapper(in_hsz=self.hsz, out_hsz=1, layer_norm=True, dropout=self.dropout, relu=False) for _ in range(self.t_iter + 1)])
        self.temporal_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.classifier = LinearWrapper(in_hsz=self.hsz * 2 if self.add_local else self.hsz, out_hsz=1, layer_norm=True, dropout=self.dropout, relu=False)

    def load_word_embedding(self, pretrained_embedding, requires_grad=False):
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embedding.weight.requires_grad = requires_grad

    def forward(self, batch):
        if self.inference_mode:
            return self.forward_main(batch)
        else:
            out, att_loss, att_predictions, temporal_loss, temporal_predictions, other_outputs = self.forward_main(batch)
            return out, att_loss, att_predictions, temporal_loss, temporal_predictions

    def forward_main(self, batch):
        """
        Args:
            batch: edict, keys = qas, qas_mask, qa_noun_masks, sub, sub_mask, vcpt, vcpt_mask, vid, vid_mask,
                                 att_labels, att_labels_mask, qid, target, vid_name, ts_label
                qas, qas_mask, qa_noun_masks: (N, 5, Lqa)
                sub, sub_mask: (N, #imgs, Ls)
                vcpt, vcpt_mask: (N, #imgs, #regions)
                vid, vid_mask: (N, #imgs, #regions, D), (N, #imgs, #regions)
                att_labels, att_labels_mask: A list of N (#imgs, #qa-words, #regions)
                qid: list(int)
                vid_name: list(str)
                target: torch.LongTensor
                use_hard_negatives: bool, true to sample hard negatives
                q_l: int, length of the tokenized question
                anno_st_idx (list of int): each element is an index (at 0.5fps) of the first image
                    with spatial annotation.
                ts_label: {"st": (N, ), "ed": (N, )} for 'st_ed'. (N, L) for 'frm'
                ts_label_mask: (N, L) for both 'st_ed' and 'frm'
        Returns:
        """
        self.bsz = len(batch.qid)
        bsz = self.bsz
        num_a = self.num_a
        hsz = self.hsz
        a_embed = self.base_encoder(batch.qas_bert.view(bsz * num_a, -1, self.wd_size), batch.qas_mask.view(bsz * num_a, -1), self.bert_word_encoding_fc, self.input_embedding, self.input_encoder)
        a_embed = a_embed.view(bsz, num_a, 1, -1, hsz)
        a_mask = batch.qas_mask.view(bsz, num_a, 1, -1)
        attended_sub, attended_vid, attended_vid_mask, attended_sub_mask = (None,) * 4
        other_outputs = {}
        if self.sub_flag:
            num_imgs, num_words = batch.sub_bert.shape[1:3]
            sub_embed = self.base_encoder(batch.sub_bert.view(bsz * num_imgs, num_words, -1), batch.sub_mask.view(bsz * num_imgs, num_words), self.bert_word_encoding_fc, self.input_embedding, self.input_encoder)
            sub_embed = sub_embed.contiguous().view(bsz, 1, num_imgs, num_words, -1)
            sub_mask = batch.sub_mask.view(bsz, 1, num_imgs, num_words)
            attended_sub, attended_sub_mask, sub_raw_s, sub_normalized_s = self.qa_ctx_attention(a_embed, sub_embed, a_mask, sub_mask, noun_mask=None, non_visual_vectors=None)
            other_outputs['sub_normalized_s'] = sub_normalized_s
            other_outputs['sub_raw_s'] = sub_raw_s
        if self.vfeat_flag:
            num_imgs, num_regions = batch.vid.shape[1:3]
            vid_embed = F.normalize(batch.vid, p=2, dim=-1)
            vid_embed = self.base_encoder(vid_embed.view(bsz * num_imgs, num_regions, -1), batch.vid_mask.view(bsz * num_imgs, num_regions), self.vid_fc, self.input_embedding, self.input_encoder)
            vid_embed = vid_embed.contiguous().view(bsz, 1, num_imgs, num_regions, -1)
            vid_mask = batch.vid_mask.view(bsz, 1, num_imgs, num_regions)
            attended_vid, attended_vid_mask, vid_raw_s, vid_normalized_s = self.qa_ctx_attention(a_embed, vid_embed, a_mask, vid_mask, noun_mask=None, non_visual_vectors=None)
            other_outputs['vid_normalized_s'] = vid_normalized_s
            other_outputs['vid_raw_s'] = vid_raw_s
        if self.flag_cnt == 2:
            visual_text_embedding = torch.cat([attended_sub, attended_vid, attended_sub * attended_vid], dim=-1)
            visual_text_embedding = self.concat_fc(visual_text_embedding)
            out, target, t_scores = self.classfier_head_multi_proposal(visual_text_embedding, attended_vid_mask, batch.target, batch.ts_label, batch.ts_label_mask, extra_span_length=self.extra_span_length)
        elif self.sub_flag:
            out, target, t_scores = self.classfier_head_multi_proposal(attended_sub, attended_sub_mask, batch.target, batch.ts_label, batch.ts_label_mask, extra_span_length=self.extra_span_length)
        elif self.vfeat_flag:
            out, target, t_scores = self.classfier_head_multi_proposal(attended_vid, attended_vid_mask, batch.target, batch.ts_label, batch.ts_label_mask, extra_span_length=self.extra_span_length)
        else:
            raise NotImplementedError
        assert len(out) == len(target)
        other_outputs['temporal_scores'] = t_scores
        if self.inference_mode:
            inference_outputs = {'answer': out, 't_scores': F.softmax(t_scores, dim=2), 'att_predictions': self.get_att_prediction(scores=other_outputs['vid_raw_s'], object_vocab=batch.eval_object_word_ids, words=batch.qas, vid_names=batch.vid_name, qids=batch.qid, img_indices=batch.image_indices, boxes=batch.boxes, start_indices=batch.anno_st_idx) if self.vfeat_flag else None}
            return inference_outputs
        att_loss = 0
        att_predictions = None
        if self.use_sup_att and self.training and self.vfeat_flag:
            start_indices = batch.anno_st_idx
            try:
                cur_att_loss, cur_att_predictions = self.get_att_loss(other_outputs['vid_raw_s'], batch.att_labels, batch.target, batch.qas, qids=batch.qid, q_lens=batch.q_l, vid_names=batch.vid_name, img_indices=batch.image_indices, boxes=batch.boxes, start_indices=start_indices, num_negatives=self.num_negatives, use_hard_negatives=batch.use_hard_negatives, drop_topk=self.drop_topk)
            except AssertionError as e:
                save_pickle({'batch': batch, 'start_indices': start_indices, 'vid_raw_s': other_outputs['vid_raw_s']}, 'err_dict.pickle')
                sys.exit(1)
            att_loss += cur_att_loss
            att_predictions = cur_att_predictions
        temporal_loss = self.get_ts_loss(temporal_scores=t_scores, ts_labels=batch.ts_label, answer_indices=batch.target)
        if self.training:
            return [out, target], att_loss, att_predictions, temporal_loss, t_scores, other_outputs
        else:
            return out, att_loss, att_predictions, temporal_loss, F.softmax(t_scores, dim=2), other_outputs

    @classmethod
    def base_encoder(cls, data, data_mask, init_encoder, downsize_encoder, input_encoder):
        """ Raw data --> higher-level embedding
        Args:
            data: (N, L) for text, (N, L, D) for video
            data_mask: (N, L)
            init_encoder: word_embedding layer for text, MLP (downsize) for video
            downsize_encoder: MLP, down project to hsz
            input_encoder: multiple layer of encoder block, with residual connection, CNN, layernorm, etc
        Returns:
            encoded_data: (N, L, D)
        """
        data = downsize_encoder(init_encoder(data))
        return input_encoder(data, data_mask)

    def qa_ctx_attention(self, qa_embed, ctx_embed, qa_mask, ctx_mask, noun_mask, non_visual_vectors):
        """ Align image regions with QA words
        Args:
            qa_embed: (N, 5, 1, Lqa, D)
            qa_mask:  (N, 5, 1, Lqa)
            ctx_embed: (N, 1, Li, Lr, D)
            ctx_mask: (N, 1, Li, Lr)
            noun_mask: (N, 5, Lqa)
            non_visual_vectors: (m, D), m is a tunable parameter
        Returns:
        """
        num_img, num_region = ctx_mask.shape[2:]
        u_a, raw_s, s_mask, s_normalized = self.str_attn(qa_embed, ctx_embed, qa_mask, ctx_mask, noun_mask=noun_mask, void_vector=non_visual_vectors)
        qa_embed = qa_embed.repeat(1, 1, num_img, 1, 1)
        mixed = torch.cat([qa_embed, u_a, qa_embed * u_a], dim=-1)
        mixed = self.c2q_down_projection(mixed)
        mixed_mask = (s_mask.sum(-1) != 0).float()
        return mixed, mixed_mask, raw_s, s_normalized

    def get_proposals(self, max_statement, max_statement_mask, temporal_scores, targets, ts_labels, max_num_proposal=1, iou_thd=0.5, ce_prob_thd=0.01, extra_span_length=3):
        """
        Args:
            max_statement: (N, 5, Li, D)
            max_statement_mask: (N, 5, Li, 1)
            temporal_scores: (N, 5, Li, 2)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            max_num_proposal:
            iou_thd:
            ce_prob_thd:
            extra_span_length:
        Returns:

        """
        bsz, num_a, num_img, _ = max_statement_mask.shape
        if self.training:
            ca_temporal_scores_st_ed = temporal_scores[torch.arange(bsz, dtype=torch.long), targets].data
            ca_temporal_scores_st_ed = F.softmax(ca_temporal_scores_st_ed, dim=1)
            ca_pred_spans = find_max_triples(ca_temporal_scores_st_ed[:, :, 0], ca_temporal_scores_st_ed[:, :, 1], topN=max_num_proposal, prob_thd=ce_prob_thd)
            ca_pred_spans = [[[sub_e[0], sub_e[1] + 1, sub_e[2]] for sub_e in e] for e in ca_pred_spans]
            spans = get_high_iou_sapns(zip(ts_labels['st'].tolist(), (ts_labels['ed'] + 1).tolist()), ca_pred_spans, iou_thd=iou_thd, add_gt=True)
            local_max_max_statement_list = []
            global_max_max_statement_list = []
            span_targets = []
            for idx, (t, span_sublist) in enumerate(zip(targets, spans)):
                span_targets.extend([t] * len(span_sublist))
                cur_global_max_max_statement = torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 1)[0]
                global_max_max_statement_list.extend([cur_global_max_max_statement] * len(span_sublist))
                for span in span_sublist:
                    span = expand_span(span, expand_length=extra_span_length)
                    cur_span_max_statement = mask_logits(max_statement[idx, :, span[0]:span[1]], max_statement_mask[idx, :, span[0]:span[1]])
                    local_max_max_statement_list.append(torch.max(cur_span_max_statement, 1)[0])
            local_max_max_statement = torch.stack(local_max_max_statement_list)
            global_max_max_statement = torch.stack(global_max_max_statement_list)
            max_max_statement = torch.cat([local_max_max_statement, global_max_max_statement], dim=-1)
            return max_max_statement, targets.new_tensor(span_targets)
        else:
            temporal_scores_st_ed = F.softmax(temporal_scores, dim=2)
            temporal_scores_st_ed_reshaped = temporal_scores_st_ed.view(bsz * num_a, -1, 2)
            pred_spans = find_max_triples(temporal_scores_st_ed_reshaped[:, :, 0], temporal_scores_st_ed_reshaped[:, :, 1], topN=1, prob_thd=None)
            pred_spans = flat_list_of_lists(pred_spans)
            pred_spans = torch.FloatTensor(pred_spans)
            pred_spans, pred_scores = pred_spans[:, :2].long(), pred_spans[:, 2]
            pred_spans = [[e[0], e[1] + 1] for e in pred_spans]
            max_statement = max_statement.view(bsz * num_a, num_img, -1)
            max_statement_mask = max_statement_mask.view(bsz * num_a, num_img, -1)
            local_max_max_statement_list = []
            global_max_max_statement_list = []
            for idx, span in enumerate(pred_spans):
                span = expand_span(span, expand_length=extra_span_length)
                cur_global_max_max_statement = torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 0)[0]
                global_max_max_statement_list.append(cur_global_max_max_statement)
                cur_span_max_statement = mask_logits(max_statement[idx, span[0]:span[1]], max_statement_mask[idx, span[0]:span[1]])
                local_max_max_statement_list.append(torch.max(cur_span_max_statement, 0)[0])
            local_max_max_statement = torch.stack(local_max_max_statement_list)
            global_max_max_statement = torch.stack(global_max_max_statement_list)
            max_max_statement = torch.cat([local_max_max_statement, global_max_max_statement], dim=-1)
            return max_max_statement.view(bsz, num_a, -1), targets

    def residual_temporal_predictor(self, layer_idx, input_tensor):
        """
        Args:
            layer_idx (int):
            input_tensor: (N, L, D)

        Returns:
            temporal_score
        """
        input_tensor = input_tensor + self.cls_projection_layers[layer_idx](input_tensor)
        t_score_st = self.temporal_scoring_st_layers[layer_idx](input_tensor)
        t_score_ed = self.temporal_scoring_ed_layers[layer_idx](input_tensor)
        t_score = torch.cat([t_score_st, t_score_ed], dim=2)
        return input_tensor, t_score

    def classfier_head_multi_proposal(self, statement, statement_mask, targets, ts_labels, ts_labels_mask, max_num_proposal=1, ce_prob_thd=0.01, iou_thd=0.5, extra_span_length=3):
        """Predict the probabilities of each statements being true. Statements = QA + Context.
        Args:
            statement: (N, 5, Li, Lqa, D)
            statement_mask: (N, 5, Li, Lqa)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            ts_labels_mask: (N, Li)
            max_num_proposal (int):
            ce_prob_thd (float): threshold for p1*p2 (st, ed)
            iou_thd (float): threshold for temporal iou
            extra_span_length (int): expand the localized span to give a little bit extra context
        Returns:
        """
        bsz, num_a, num_img, num_words = statement_mask.shape
        statement = statement.view(bsz * num_a * num_img, num_words, -1)
        statement_mask = statement_mask.view(bsz * num_a * num_img, num_words)
        statement = self.cls_encoder(statement, statement_mask)
        max_statement = torch.max(mask_logits(statement, statement_mask.unsqueeze(2)), 1)[0]
        max_statement_mask = (statement_mask.sum(1) != 0).float().view(bsz, num_a, num_img, 1)
        max_statement = max_statement.view(bsz * num_a, num_img, -1)
        t_score_container = []
        encoded_max_statement_container = []
        encoded_max_statement = max_statement
        for layer_idx in range(self.t_iter + 1):
            encoded_max_statement, prev_t_score = self.residual_temporal_predictor(layer_idx, encoded_max_statement)
            t_score_container.append(prev_t_score.view(bsz, num_a, num_img, 2))
            encoded_max_statement_container.append(encoded_max_statement)
        if self.t_iter > 0:
            temporal_scores_st_ed = 0.5 * (t_score_container[0] + torch.stack(t_score_container[:1]).mean(0))
        else:
            temporal_scores_st_ed = t_score_container[0]
        temporal_scores_st_ed = mask_logits(temporal_scores_st_ed, ts_labels_mask.view(bsz, 1, num_img, 1))
        stacked_max_statement = encoded_max_statement_container[0].view(bsz, num_a, num_img, -1)
        if self.add_local:
            max_max_statement, targets = self.get_proposals(stacked_max_statement, max_statement_mask, temporal_scores_st_ed, targets, ts_labels, max_num_proposal=max_num_proposal, iou_thd=iou_thd, ce_prob_thd=ce_prob_thd, extra_span_length=extra_span_length)
        else:
            max_max_statement = torch.max(mask_logits(stacked_max_statement, max_statement_mask), 2)[0]
        answer_scores = self.classifier(max_max_statement).squeeze(2)
        return answer_scores, targets, temporal_scores_st_ed

    def get_ts_loss(self, temporal_scores, ts_labels, answer_indices):
        """
        Args:
            temporal_scores: (N, 5, Li, 2)
            ts_labels: dict(st=(N, ), ed=(N, ))
            answer_indices: (N, )

        Returns:

        """
        bsz = len(answer_indices)
        ca_temporal_scores_st_ed = temporal_scores[torch.arange(bsz, dtype=torch.long), answer_indices]
        loss_st = self.temporal_criterion(ca_temporal_scores_st_ed[:, :, 0], ts_labels['st'])
        loss_ed = self.temporal_criterion(ca_temporal_scores_st_ed[:, :, 1], ts_labels['ed'])
        return (loss_st + loss_ed) / 2.0

    @classmethod
    def sample_negatives(cls, pred_score, pos_indices, neg_indices, num_negatives=2, use_hard_negatives=False, negative_pool_size=0, num_hard=2, drop_topk=0):
        """ Sample negatives from a set of indices. Several sampling strategies are supported:
        1, random; 2, hard negatives; 3, drop_topk hard negatives; 4, mix easy and hard negatives
        5, sampling within a pool of hard negatives; 6, sample across images of the same video.
        Args:
            pred_score: (num_img, num_words, num_region)
            pos_indices: (N_pos, 3) all positive region indices for the same word, not necessaryily the same image.
            neg_indices: (N_neg, 3) ...
            num_negatives (int):
            use_hard_negatives (bool):
            negative_pool_size (int):
            num_hard (int):
            drop_topk (int):
        Returns:

        """
        num_unique_pos = len(pos_indices)
        sampled_pos_indices = torch.cat([pos_indices] * num_negatives, dim=0)
        if use_hard_negatives:
            neg_scores = pred_score[neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]
            max_indices = torch.sort(neg_scores, descending=True)[1].tolist()
            if negative_pool_size > num_negatives:
                hard_pool = max_indices[drop_topk:drop_topk + negative_pool_size]
                hard_pool_indices = neg_indices[hard_pool]
                num_hard_negs = num_negatives
                sampled_easy_neg_indices = []
                if num_hard < num_negatives:
                    easy_pool = max_indices[drop_topk + negative_pool_size:]
                    easy_pool_indices = neg_indices[easy_pool]
                    num_hard_negs = num_hard
                    num_easy_negs = num_negatives - num_hard_negs
                    sampled_easy_neg_indices = easy_pool_indices[torch.randint(low=0, high=len(easy_pool_indices), size=(num_easy_negs * num_unique_pos,), dtype=torch.long)]
                sampled_hard_neg_indices = hard_pool_indices[torch.randint(low=0, high=len(hard_pool_indices), size=(num_hard_negs * num_unique_pos,), dtype=torch.long)]
                if len(sampled_easy_neg_indices) != 0:
                    sampled_neg_indices = torch.cat([sampled_hard_neg_indices, sampled_easy_neg_indices], dim=0)
                else:
                    sampled_neg_indices = sampled_hard_neg_indices
            else:
                sampled_neg_indices = neg_indices[max_indices[drop_topk:drop_topk + len(sampled_pos_indices)]]
        else:
            sampled_neg_indices = neg_indices[torch.randint(low=0, high=len(neg_indices), size=(len(sampled_pos_indices),), dtype=torch.long)]
        return sampled_pos_indices, sampled_neg_indices

    def get_att_loss(self, scores, att_labels, target, words, vid_names, qids, q_lens, img_indices, boxes, start_indices, num_negatives=2, use_hard_negatives=False, drop_topk=0):
        """ compute ranking loss, use for loop to find the indices,
        use advanced indexing to perform the real calculation
        Build a list contains a quaduple

        Args:
            scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
            att_labels: list(tensor), each has dimension (#num_imgs, #num_words, #regions), not batched
            target: 1D tensor (N, )
            words: LongTensor (N, 5, Lqa)
            vid_names: list(str) (N,)
            qids: list(int), (N, )
            q_lens: list(int), (N, )
            img_indices: list(list(int)), (N, Li), or None
            boxes: list(list(box)) of length N, each sublist represent an image,
                each box contains the coordinates of xyxy, or None
            num_negatives: number of negatives for each positive region
            use_hard_negatives: use hard negatives, uselect negatives with high scores
            drop_topk: drop topk highest negatives (since the top negatives might be correct, they are just not labeled)
            start_indices (list of int): each element is an index (at 0.5fps) of the first image
                with spatial annotation. If with_ts, set to zero
        Returns:
            att_loss: loss value for the batch
            att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
        """
        pos_container = []
        neg_container = []
        for batch_idx in range(len(target)):
            ca_idx = target[batch_idx].cpu().item()
            gt_score = att_labels[batch_idx]
            start_idx = start_indices[batch_idx]
            num_img = len(gt_score)
            sen_l, _ = gt_score[0].shape
            pred_score = scores[batch_idx, ca_idx, :num_img, :sen_l]
            batch_pos_indices = []
            batch_neg_indices = []
            for img_idx, img_gt_score in enumerate(gt_score):
                img_idx = start_idx + img_idx
                img_pos_indices = torch.nonzero(img_gt_score)
                if len(img_pos_indices) == 0:
                    continue
                img_pos_indices = torch.cat([img_pos_indices.new_full([len(img_pos_indices), 1], img_idx), img_pos_indices], dim=1)
                img_neg_indices = torch.nonzero(img_gt_score == 0)
                img_neg_indices = torch.cat([img_neg_indices.new_full([len(img_neg_indices), 1], img_idx), img_neg_indices], dim=1)
                batch_pos_indices.append(img_pos_indices)
                batch_neg_indices.append(img_neg_indices)
            if len(batch_pos_indices) == 0:
                continue
            batch_pos_indices = torch.cat(batch_pos_indices, dim=0)
            batch_neg_indices = torch.cat(batch_neg_indices, dim=0)
            available_img_indices = batch_pos_indices[:, 0].unique().tolist()
            for img_idx in available_img_indices:
                img_idx_pos_indices = batch_pos_indices[batch_pos_indices[:, 0] == img_idx]
                img_idx_neg_indices = batch_neg_indices[batch_neg_indices[:, 0] == img_idx]
                available_word_indices = img_idx_pos_indices[:, 1].unique().tolist()
                for word_idx in available_word_indices:
                    img_idx_word_idx_pos_indices = img_idx_pos_indices[img_idx_pos_indices[:, 1] == word_idx]
                    img_idx_word_idx_neg_indices = img_idx_neg_indices[img_idx_neg_indices[:, 1] == word_idx]
                    sampled_pos_indices, sampled_neg_indices = self.sample_negatives(pred_score, img_idx_word_idx_pos_indices, img_idx_word_idx_neg_indices, num_negatives=num_negatives, use_hard_negatives=use_hard_negatives, negative_pool_size=self.negative_pool_size, num_hard=self.num_hard, drop_topk=drop_topk)
                    base_indices = torch.LongTensor([[batch_idx, ca_idx]] * len(sampled_pos_indices))
                    pos_container.append(torch.cat([base_indices, sampled_pos_indices], dim=1))
                    neg_container.append(torch.cat([base_indices, sampled_neg_indices], dim=1))
        pos_container = torch.cat(pos_container, dim=0)
        neg_container = torch.cat(neg_container, dim=0)
        att_predictions = None
        if not self.training and self.vfeat_flag:
            att_predictions = dict(det_q=[], det_ca=[])
            unique_pos_container = np.unique(pos_container.cpu().numpy(), axis=0)
            for row in unique_pos_container:
                batch_idx, ca_idx, img_idx, word_idx, region_idx = row
                start_idx = start_indices[batch_idx]
                cur_q_len = q_lens[batch_idx]
                num_region = att_labels[batch_idx][img_idx - start_idx].shape[1]
                if len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()) != len(boxes[batch_idx][img_idx - start_idx]):
                    None
                    None
                    None
                    None
                    raise AssertionError
                cur_det_data = {'pred': scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu(), 'word': words[batch_idx, ca_idx, word_idx], 'qid': qids[batch_idx], 'vid_name': vid_names[batch_idx], 'img_idx': img_indices[batch_idx][img_idx], 'boxes': boxes[batch_idx][img_idx - start_idx]}
                if word_idx < cur_q_len:
                    att_predictions['det_q'].append(cur_det_data)
                else:
                    att_predictions['det_ca'].append(cur_det_data)
        pos_scores = scores[pos_container[:, 0], pos_container[:, 1], pos_container[:, 2], pos_container[:, 3], pos_container[:, 4]]
        neg_scores = scores[neg_container[:, 0], neg_container[:, 1], neg_container[:, 2], neg_container[:, 3], neg_container[:, 4]]
        if self.att_loss_type == 'hinge':
            att_loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0).sum()
        elif self.att_loss_type == 'lse':
            att_loss = torch.log1p(torch.exp(self.alpha * (neg_scores - pos_scores))).sum()
        else:
            raise NotImplementedError('Only support hinge and lse')
        return att_loss, att_predictions

    def get_att_prediction(self, scores, object_vocab, words, vid_names, qids, img_indices, boxes, start_indices, score_thd=0.2):
        """ compute ranking loss, use for loop to find the indices,
        use advanced indexing to perform the real calculation
        Build a list contains a quaduple

        Args:
            scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
            object_vocab: list, object word ids in the vocabulary
            words: LongTensor (N, 5, Lqa)
            vid_names: list(str) (N,)
            qids: list(int), (N, )
            img_indices: list(list(int)), (N, Li), or None
            boxes: list(list(box)) of length N, each sublist represent an image,
                each box contains the coordinates of xyxy, or None
            start_indices (list of int): each element is an index (at 0.5fps) of the first image
                with spatial annotation. If with_ts, set to zero
            score_thd: only keep boxes with score higher than this value
        Returns:
            att_loss: loss value for the batch
            att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
        """
        att_predictions = None
        if self.vfeat_flag:
            att_predictions = []
            for batch_idx in range(len(scores)):
                start_idx = start_indices[batch_idx]
                q_att_predictions = dict()
                for ans_idx in range(5):
                    q_att_predictions[ans_idx] = []
                    for img_idx_local in range(len(boxes[batch_idx])):
                        img_idx_global = img_idx_local + start_idx
                        cur_img_scores = scores[batch_idx, ans_idx, img_idx_global]
                        cur_words = words[batch_idx, ans_idx].tolist()
                        cur_img_boxes = boxes[batch_idx][img_idx_local]
                        for word_idx, w in enumerate(cur_words):
                            if w in object_vocab:
                                cur_word_region_scores = cur_img_scores[word_idx].data.cpu().numpy()
                                accepted_region_ids = np.nonzero(cur_word_region_scores >= score_thd)[0].tolist()
                                accepted_region_scores = [float(cur_word_region_scores[i]) for i in accepted_region_ids]
                                accepted_region_boxes = [cur_img_boxes[i] for i in accepted_region_ids]
                                sorted_indices = np.argsort(accepted_region_scores)
                                accepted_region_scores = [accepted_region_scores[i] for i in sorted_indices]
                                accepted_region_boxes = [accepted_region_boxes[i] for i in sorted_indices]
                                cur_det_data = {'pred': accepted_region_scores, 'bbox': accepted_region_boxes, 'word': int(words[batch_idx, ans_idx, word_idx]), 'qid': int(qids[batch_idx]), 'vid_name': vid_names[batch_idx], 'img_idx': img_indices[batch_idx][img_idx_global]}
                                q_att_predictions[ans_idx].append(cur_det_data)
                att_predictions.append(q_att_predictions)
        return att_predictions


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContextQueryAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLinear,
     lambda: ([], {'in_hsz': 4, 'out_hsz': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DepthwiseSeparableConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearWrapper,
     lambda: ([], {'in_hsz': 4, 'out_hsz': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'nh': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormalizeScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 128])], {}),
     True),
    (StructuredAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_jayleicn_TVQAplus(_paritybench_base):
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

