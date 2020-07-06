import sys
_module = sys.modules[__name__]
del sys
ban_model = _module
baseline_model = _module
config = _module
counting = _module
counting_model = _module
data = _module
graph_model = _module
inter_intra_model = _module
reuse_modules = _module
train = _module
utils = _module
word_embedding = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


from torch.autograd import Variable


from torch.nn.utils import weight_norm


from torch.nn.utils.rnn import pack_padded_sequence


import re


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


from collections import defaultdict


from torch.nn.parameter import Parameter


import itertools


import torch.utils.data


import math


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.nn.utils import clip_grad_norm_


import torch.backends.cudnn as cudnn


APPLY_MASK = True


class FCNet(nn.Module):

    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        self.activate = activate.lower() if activate is not None else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class Classifier(nn.Sequential):

    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop / 2.5)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        num_obj = v_mask.shape[1]
        max_len = q_mask.shape[1]
        if APPLY_MASK:
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        else:
            v_mean = v.sum(1) / num_obj
            q_mean = q.sum(1) / max_len
        out = self.lin1(v_mean * q_mean)
        out = self.lin2(out)
        return out


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v4q_gate_lin = FCNet(v_size, output_size, drop=drop)
        self.q4v_gate_lin = FCNet(q_size, output_size, drop=drop)
        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)
        self.v_output = FCNet(output_size, output_size, drop=drop)
        self.q_output = FCNet(output_size, output_size, drop=drop)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _, max_len = q_mask.shape
        if APPLY_MASK:
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        else:
            v_mean = v.sum(1) / num_obj
            q_mean = q.sum(1) / max_len
        v4q_gate = self.sigmoid(self.v4q_gate_lin(v_mean)).unsqueeze(1)
        q4v_gate = self.sigmoid(self.q4v_gate_lin(q_mean)).unsqueeze(1)
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        if APPLY_MASK:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        gated_v_qry = (1 + q4v_gate) * v_qry
        gated_v_key = (1 + q4v_gate) * v_key
        gated_v_val = (1 + q4v_gate) * v_val
        gated_q_qry = (1 + v4q_gate) * q_qry
        gated_q_key = (1 + v4q_gate) * q_key
        gated_q_val = (1 + v4q_gate) * q_val
        v_key_set = torch.split(gated_v_key, gated_v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(gated_v_qry, gated_v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(gated_v_val, gated_v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(gated_q_key, gated_q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(gated_q_qry, gated_q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(gated_q_val, gated_q_val.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]
            v2v = v_qry_slice @ v_key_slice.transpose(1, 2) / (self.output_size // self.num_head) ** 0.5
            q2q = q_qry_slice @ q_key_slice.transpose(1, 2) / (self.output_size // self.num_head) ** 0.5
            if APPLY_MASK:
                v2v.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -float('inf'))
                q2q.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -float('inf'))
            dyIntraMAF_v2v = F.softmax(v2v, dim=2).unsqueeze(3)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2).unsqueeze(3)
            v_update = (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2) if i == 0 else torch.cat((v_update, (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2) if i == 0 else torch.cat((q_update, (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
        updated_v = self.v_output(v + v_update)
        updated_q = self.q_output(q + q_update)
        return updated_v, updated_q


class OneSideInterModalityUpdate(nn.Module):
    """
    One-Side Inter-modality Attention Flow
    
    According to original paper, instead of parallel V->Q & Q->V, we first to V->Q and then Q->V
    """

    def __init__(self, src_size, tgt_size, output_size, num_head, drop=0.0):
        super(OneSideInterModalityUpdate, self).__init__()
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.output_size = output_size
        self.num_head = num_head
        self.src_lin = FCNet(src_size, output_size * 2, drop=drop)
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop)
        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=drop)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: src feature      [batch, num_src, feat_size]
        tgt: tgt feautre      [batch, num_tgt, feat_size]
        src_mask              [batch, num_src]
        tgt_mask              [batch, num_tgt]
        """
        batch_size, num_src = src_mask.shape
        _, num_tgt = tgt_mask.shape
        src_trans = self.src_lin(src)
        tgt_trans = self.tgt_lin(tgt)
        if APPLY_MASK:
            src_trans = src_trans * src_mask.unsqueeze(2)
            tgt_trans = tgt_trans * tgt_mask.unsqueeze(2)
        src_key, src_val = torch.split(src_trans, src_trans.size(2) // 2, dim=2)
        tgt_qry = tgt_trans
        src_key_set = torch.split(src_key, src_key.size(2) // self.num_head, dim=2)
        src_val_set = torch.split(src_val, src_val.size(2) // self.num_head, dim=2)
        tgt_qry_set = torch.split(tgt_qry, tgt_qry.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            src_key_slice, tgt_qry_slice, src_val_slice = src_key_set[i], tgt_qry_set[i], src_val_set[i]
            src2tgt = tgt_qry_slice @ src_key_slice.transpose(1, 2) / (self.output_size // self.num_head) ** 0.5
            if APPLY_MASK:
                src2tgt.masked_fill_(src_mask.unsqueeze(1).expand([batch_size, num_tgt, num_src]) == 0, -float('inf'))
            interMAF_src2tgt = F.softmax(src2tgt, dim=2).unsqueeze(3)
            tgt_update = (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2) if i == 0 else torch.cat((tgt_update, (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_tgt = torch.cat((tgt, tgt_update), dim=2)
        update_tgt = self.tgt_output(cat_tgt)
        return update_tgt


class MultiBlock(nn.Module):
    """
    Multi Block (different parameters) Inter-/Intra-modality
    """

    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block
        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)
        blocks = []
        for i in range(num_block):
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.multi_blocks[i * 3 + 0](v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.multi_blocks[i * 3 + 1](q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = self.multi_blocks[i * 3 + 2](v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)


class Net(nn.Module):
    """
    Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
    Based on code from https://github.com/Cyanogenoid/vqa-counting
    """

    def __init__(self, words_list):
        super(Net, self).__init__()
        self.question_features = 1280
        self.vision_features = config.output_features
        self.hidden_features = 512
        self.num_inter_head = 8
        self.num_intra_head = 8
        self.num_block = 1
        assert self.hidden_features % self.num_inter_head == 0
        assert self.hidden_features % self.num_intra_head == 0
        self.text = word_embedding.TextProcessor(classes=words_list, embedding_features=300, lstm_features=self.question_features, use_hidden=False, drop=0.0)
        self.interIntraBlocks = MultiBlock(num_block=self.num_block, v_size=self.vision_features, q_size=self.question_features, output_size=self.hidden_features, num_inter_head=self.num_inter_head, num_intra_head=self.num_intra_head, drop=0.1)
        self.classifier = Classifier(in_features=self.hidden_features, mid_features=2048, out_features=config.max_answers, drop=0.5)

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        """
        v: visual feature      [batch, num_obj, 2048]
        b: bounding box        [batch, num_obj, 4]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        """
        q = self.text(q, list(q_len.data))
        if config.v_feat_norm:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v)
        v, q = self.interIntraBlocks(v, q, v_mask, q_mask)
        answer = self.classifier(v, q, v_mask, q_mask)
        return answer


class BiAttention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(BiAttention, self).__init__()
        self.hidden_aug = 3
        self.glimpses = glimpses
        self.lin_v = FCNet(v_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop / 2.5)
        self.lin_q = FCNet(q_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop / 2.5)
        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_features * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        """
        v_num = v.size(1)
        q_num = q.size(1)
        v_ = self.lin_v(v).unsqueeze(1)
        q_ = self.lin_q(q).unsqueeze(1)
        v_ = self.drop(v_)
        h_ = v_ * self.h_weight
        logits = torch.matmul(h_, q_.transpose(2, 3))
        logits = logits + self.h_bias
        logits.data.masked_fill_(v_mask.unsqueeze(1).unsqueeze(3).expand(logits.shape) == 0, -float('inf'))
        atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
        return atten.view(-1, self.glimpses, v_num, q_num), logits


class ApplySingleAttention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, num_obj, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)
        self.lin_count = FCNet(num_obj + 1, mid_features, activate='relu', drop=0)

    def forward(self, v, q, b, v_mask, q_mask, atten, logits, count_layer):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x v_num x q_num
        logits:  batch x v_num x q_num
        """
        v_ = self.lin_v(v).transpose(1, 2).unsqueeze(2)
        q_ = self.lin_q(q).transpose(1, 2).unsqueeze(3)
        v_ = torch.matmul(v_, atten.unsqueeze(1))
        h_ = torch.matmul(v_, q_)
        h_ = h_.squeeze(3).squeeze(2)
        atten_h = self.lin_atten(h_.unsqueeze(1))
        count_embed = count_layer(b.transpose(1, 2), logits.max(2)[0])
        count_h = self.lin_count(count_embed).unsqueeze(1)
        return atten_h, count_h


class ApplyAttention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, glimpses, num_obj, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, num_obj, drop))
        self.glimpse_layers = nn.ModuleList(layers)

    def forward(self, v, q, b, v_mask, q_mask, atten, logits, count_layer):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x glimpses x v_num x q_num
        logits:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h, count_h = self.glimpse_layers[g](v, q, b, v_mask, q_mask, atten[:, (g), :, :], logits[:, (g), :, :], count_layer)
            q = q + atten_h + count_h
        return q.sum(1)


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return -(x - y) ** 2 + F.relu(x + y)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled


class Attention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = Fusion()

    def forward(self, v, q):
        q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x))
        return x


class PiecewiseLin(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        self.weight.data[0] = 0

    def forward(self, x):
        w = self.weight.abs()
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()
        x = csum.gather(0, idx.clamp(max=self.n))
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)


class Counter(nn.Module):
    """ Counting module as proposed in [1].
    Count the number of objects from a set of bounding boxes and a set of scores for each bounding box.
    This produces (self.objects + 1) number of count features.

    [1]: Yan Zhang, Jonathon Hare, Adam Prügel-Bennett: Learning to Count Objects in Natural Images for Visual Question Answering.
    https://openreview.net/forum?id=B12Js_yRb
    """

    def __init__(self, objects, already_sigmoided=False):
        super().__init__()
        self.objects = objects
        self.already_sigmoided = already_sigmoided
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(16)])

    def forward(self, boxes, attention):
        """ Forward propagation of attention weights and bounding boxes to produce count features.
        `boxes` has to be a tensor of shape (n, 4, m) with the 4 channels containing the x and y coordinates of the top left corner and the x and y coordinates of the bottom right corner in this order.
        `attention` has to be a tensor of shape (n, m). Each value should be in [0, 1] if already_sigmoided is set to True, but there are no restrictions if already_sigmoided is set to False. This value should be close to 1 if the corresponding boundign box is relevant and close to 0 if it is not.
        n is the batch size, m is the number of bounding boxes per image.
        """
        boxes, attention = self.filter_most_important(self.objects, boxes, attention)
        if not self.already_sigmoided:
            attention = self.sigmoid(attention)
        relevancy = self.outer_product(attention)
        distance = 1 - self.iou(boxes, boxes)
        score = self.f[0](relevancy) * self.f[1](distance)
        dedup_score = self.f[3](relevancy) * self.f[4](distance)
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attention)
        score = score / dedup_per_entry
        correction = self.f[0](attention * attention) / dedup_per_row
        score = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)
        score = (score + 1e-20).sqrt()
        one_hot = self.to_one_hot(score)
        att_conf = (self.f[5](attention) - 0.5).abs()
        dist_conf = (self.f[6](distance) - 0.5).abs()
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + dist_conf.mean(dim=2).mean(dim=1, keepdim=True))
        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        row_sims = sim.sum(dim=2)
        all_sims = self.outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        """ Turn a bunch of non-negative scalar values into a one-hot encoding.
        E.g. with self.objects = 3, 0 -> [1 0 0 0], 2.75 -> [0 0 0.25 0.75].
        """
        scores = scores.clamp(min=0, max=self.objects)
        i = scores.long().data
        f = scores.frac()
        target_l = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_l.scatter_(dim=1, index=i.clamp(max=self.objects), value=1)
        target_r.scatter_(dim=1, index=(i + 1).clamp(max=self.objects), value=1)
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

    def filter_most_important(self, n, boxes, attention):
        """ Only keep top-n object proposals, scored by attention weight """
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx)
        return boxes, attention

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        x = (box[:, (2), :] - box[:, (0), :]).clamp(min=0)
        y = (box[:, (3), :] - box[:, (1), :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = a.size(0), 2, a.size(2), b.size(2)
        min_point = torch.max(a[:, :2, :].unsqueeze(dim=3).expand(*size), b[:, :2, :].unsqueeze(dim=2).expand(*size))
        max_point = torch.min(a[:, 2:, :].unsqueeze(dim=3).expand(*size), b[:, 2:, :].unsqueeze(dim=2).expand(*size))
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, (0), :, :] * inter[:, (1), :, :]
        return area


qa_path = 'data'


class TextProcessor(nn.Module):

    def __init__(self, classes, embedding_features, lstm_features, drop=0.0, use_hidden=True, use_tanh=False, only_embed=False):
        super(TextProcessor, self).__init__()
        self.use_hidden = use_hidden
        self.use_tanh = use_tanh
        self.only_embed = only_embed
        classes = list(classes)
        self.embed = nn.Embedding(len(classes) + 1, embedding_features, padding_idx=len(classes))
        weight_init = torch.from_numpy(np.load(qa_path + '/glove6b_init_300d.npy'))
        assert weight_init.shape == (len(classes), embedding_features)
        None
        self.embed.weight.data[:len(classes)] = weight_init
        None
        self.drop = nn.Dropout(drop)
        if self.use_tanh:
            self.tanh = nn.Tanh()
        if not self.only_embed:
            self.lstm = nn.GRU(input_size=embedding_features, hidden_size=lstm_features, num_layers=1, batch_first=not use_hidden)

    def forward(self, q, q_len):
        embedded = self.embed(q)
        embedded = self.drop(embedded)
        if self.use_tanh:
            embedded = self.tanh(embedded)
        if self.only_embed:
            return embedded
        self.lstm.flatten_parameters()
        if self.use_hidden:
            packed = pack_padded_sequence(embedded, q_len, batch_first=True)
            _, hid = self.lstm(packed)
            return hid.squeeze(0)
        else:
            out, _ = self.lstm(embedded)
            return out


class PseudoCoord(nn.Module):

    def __init__(self):
        super(PseudoCoord, self).__init__()

    def forward(self, b):
        """
        Input: 
        b: bounding box        [batch, num_obj, 4]  (x1,y1,x2,y2)
        Output:
        pseudo_coord           [batch, num_obj, num_obj, 2] (rho, theta)
        """
        batch_size, num_obj, _ = b.shape
        centers = (b[:, :, 2:] + b[:, :, :2]) * 0.5
        relative_coord = centers.view(batch_size, num_obj, 1, 2) - centers.view(batch_size, 1, num_obj, 2)
        rho = torch.sqrt(relative_coord[:, :, :, (0)] ** 2 + relative_coord[:, :, :, (1)] ** 2)
        theta = torch.atan2(relative_coord[:, :, :, (0)], relative_coord[:, :, :, (1)])
        new_coord = torch.cat((rho.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)
        return new_coord


class GraphLearner(nn.Module):

    def __init__(self, v_features, q_features, mid_features, dropout=0.0, sparse_graph=True):
        super(GraphLearner, self).__init__()
        self.sparse_graph = sparse_graph
        self.lin1 = FCNet(v_features + q_features, mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')

    def forward(self, v, q, v_mask, top_K):
        """
        Input:
        v: visual feature      [batch, num_obj, 2048]
        q: bounding box        [batch, 1024]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none

        Return:
        adjacent_logits        [batch, num_obj, K(sum=1)]
        adjacent_matrix        [batch, num_obj, K(sum=1)]
        """
        batch_size, num_obj, _ = v.shape
        q_repeated = q.unsqueeze(1).repeat(1, num_obj, 1)
        v_cat_q = torch.cat((v, q_repeated), dim=2)
        h = self.lin1(v_cat_q)
        h = self.lin2(h)
        h = h.view(batch_size, num_obj, -1)
        adjacent_logits = torch.matmul(h, h.transpose(1, 2))
        if self.sparse_graph:
            top_value, top_ind = torch.topk(adjacent_logits, k=top_K, dim=-1, sorted=False)
        adjacent_matrix = F.softmax(top_value, dim=-1)
        return adjacent_matrix, top_ind


class GraphConv(nn.Module):

    def __init__(self, v_features, mid_features, num_kernels, bias=False):
        super(GraphConv, self).__init__()
        self.num_kernels = num_kernels
        self.conv_weights = nn.ModuleList([nn.Linear(v_features, mid_features // num_kernels, bias=bias) for i in range(num_kernels)])
        self.mean_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.mean_theta = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_theta = Parameter(torch.FloatTensor(num_kernels, 1))
        self.init_param()

    def init_param(self):
        self.mean_rho.data.uniform_(0, 1.0)
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.precision_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0, 1.0)

    def forward(self, v, v_mask, coord, adj_matrix, top_ind, weight_adj=True):
        """
        Input:
        v: visual feature      [batch, num_obj, 2048]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord
        adj_matrix: sparse     [batch, num_obj, K(sum=1)]
        top_ind:               [batch, num_obj, K]
        Output:
        v: visual feature      [batch, num_obj, dim]
        """
        batch_size, num_obj, feat_dim = v.shape
        K = adj_matrix.shape[-1]
        conv_v = v.unsqueeze(1).expand(batch_size, num_obj, num_obj, feat_dim)
        coord_weight = self.get_gaussian_weights(coord)
        slice_idx1 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K, feat_dim)
        slice_idx2 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K, self.num_kernels)
        sparse_v = torch.gather(conv_v, dim=2, index=slice_idx1)
        sparse_weight = torch.gather(coord_weight, dim=2, index=slice_idx2)
        if weight_adj:
            adj_mat = adj_matrix.unsqueeze(-1)
            attentive_v = sparse_v * adj_mat
        else:
            attentive_v = sparse_v
        weighted_neighbourhood = torch.matmul(sparse_weight.transpose(2, 3), attentive_v)
        weighted_neighbourhood = [self.conv_weights[i](weighted_neighbourhood[:, :, (i), :]) for i in range(self.num_kernels)]
        output = torch.cat(weighted_neighbourhood, dim=2)
        return output

    def get_gaussian_weights(self, coord):
        """
        Input:
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord

        Output:
        weights                [batch, num_obj, num_obj, n_kernels)
        """
        batch_size, num_obj, _, _ = coord.shape
        diff = (coord[:, :, :, (0)].contiguous().view(-1, 1) - self.mean_rho.view(1, -1)) ** 2
        weights_rho = torch.exp(-0.5 * diff / (1e-14 + self.precision_rho.view(1, -1) ** 2))
        first_angle = torch.abs(coord[:, :, :, (1)].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * torch.min(first_angle, second_angle) ** 2 / (1e-14 + self.precision_theta.view(1, -1) ** 2))
        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-14)
        return weights.view(batch_size, num_obj, num_obj, self.num_kernels)


class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """

    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block
        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)
        self.v2q_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.q2v_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.v2q_interBlock(v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.q2v_interBlock(q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = intraBlock(v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)


class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)
        self.v_output = FCNet(output_size + v_size, output_size, drop=drop)
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _, max_len = q_mask.shape
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        if APPLY_MASK:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        v_key_set = torch.split(v_key, v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(v_qry, v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(v_val, v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(q_key, q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(q_qry, q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(q_val, q_val.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]
            q2v = v_qry_slice @ q_key_slice.transpose(1, 2) / (self.output_size // self.num_head) ** 0.5
            v2q = q_qry_slice @ v_key_slice.transpose(1, 2) / (self.output_size // self.num_head) ** 0.5
            if APPLY_MASK:
                q2v.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -float('inf'))
                v2q.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -float('inf'))
            interMAF_q2v = F.softmax(q2v, dim=2).unsqueeze(3)
            interMAF_v2q = F.softmax(v2q, dim=2).unsqueeze(3)
            v_update = (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2) if i == 0 else torch.cat((v_update, (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2) if i == 0 else torch.cat((q_update, (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(cat_v)
        updated_q = self.q_output(cat_q)
        return updated_v, updated_q


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'v_features': 4, 'q_features': 4, 'mid_features': 4, 'glimpses': 4}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 4])], {}),
     False),
    (BiAttention,
     lambda: ([], {'v_features': 4, 'q_features': 4, 'mid_features': 4, 'glimpses': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Classifier,
     lambda: ([], {'in_features': 4, 'mid_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Counter,
     lambda: ([], {'objects': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (DyIntraModalityUpdate,
     lambda: ([], {'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_head': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FCNet,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (InterModalityUpdate,
     lambda: ([], {'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_head': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (MultiBlock,
     lambda: ([], {'num_block': 4, 'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_inter_head': 4, 'num_intra_head': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (OneSideInterModalityUpdate,
     lambda: ([], {'src_size': 4, 'tgt_size': 4, 'output_size': 4, 'num_head': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (PiecewiseLin,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PseudoCoord,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_KaihuaTang_VQA2_0_Recent_Approachs_2018_pytorch(_paritybench_base):
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

