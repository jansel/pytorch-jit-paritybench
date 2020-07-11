import sys
_module = sys.modules[__name__]
del sys
clevr_extract_feat = _module
gqa_feat_preproc = _module
conf = _module
base_cfgs = _module
base_dataset = _module
path_cfgs = _module
clevr_loader = _module
result_eval = _module
dataset_loader = _module
gqa_eval = _module
gqa_loader = _module
vqa = _module
vqaEval = _module
vqa_loader = _module
adapter = _module
ban = _module
model_cfgs = _module
net = _module
adapter = _module
net = _module
tda = _module
adapter = _module
mca = _module
net = _module
adapter = _module
mfb = _module
net = _module
model_loader = _module
fc = _module
layer_norm = _module
ans_punct = _module
feat_filter = _module
make_mask = _module
optim = _module
run = _module
exec = _module
proc_dict_gqa = _module
proc_dict_vqa = _module
test_engine = _module
train_engine = _module

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


import numpy as np


import torch


import torchvision


import random


from types import MethodType


import torch.utils.data as Data


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.weight_norm import weight_norm


import math


import torch.optim as Optim


import time


def feat_filter(dataset, frcn_feat, grid_feat, bbox_feat):
    feat_dict = {}
    if dataset in ['vqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['BBOX_FEAT'] = bbox_feat
    elif dataset in ['gqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['GRID_FEAT'] = grid_feat
        feat_dict['BBOX_FEAT'] = bbox_feat
    elif dataset in ['clevr']:
        feat_dict['GRID_FEAT'] = grid_feat
    else:
        exit(-1)
    return feat_dict


class BaseAdapter(nn.Module):

    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        if self.__C.DATASET in ['vqa']:
            self.vqa_init(__C)
        elif self.__C.DATASET in ['gqa']:
            self.gqa_init(__C)
        elif self.__C.DATASET in ['clevr']:
            self.clevr_init(__C)
        else:
            exit(-1)

    def vqa_init(self, __C):
        raise NotImplementedError()

    def gqa_init(self, __C):
        raise NotImplementedError()

    def clevr_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)
        if self.__C.DATASET in ['vqa']:
            return self.vqa_forward(feat_dict)
        elif self.__C.DATASET in ['gqa']:
            return self.gqa_forward(feat_dict)
        elif self.__C.DATASET in ['clevr']:
            return self.clevr_forward(feat_dict)
        else:
            exit(-1)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()

    def gqa_forward(self, feat_dict):
        raise NotImplementedError()

    def clevr_forward(self, feat_dict):
        raise NotImplementedError()


def make_mask(feature):
    return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class Adapter(BaseAdapter):

    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
        self.frcn_linear = nn.Linear(__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def gqa_init(self, __C):
        self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
        self.frcn_linear = nn.Linear(__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1] + __C.BBOXFEAT_EMB_SIZE, __C.HIDDEN_SIZE)
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        img_feat_mask = make_mask(frcn_feat)
        img_feat = frcn_feat
        return img_feat, img_feat_mask

    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']
        img_feat_mask = torch.cat((make_mask(frcn_feat), make_mask(grid_feat)), dim=-1)
        bbox_feat = self.bbox_linear(bbox_feat)
        frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        frcn_feat = self.frcn_linear(frcn_feat)
        grid_feat = self.grid_linear(grid_feat)
        img_feat = torch.cat((frcn_feat, grid_feat), dim=1)
        return img_feat, img_feat_mask

    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']
        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)
        return img_feat, img_feat_mask


class FC(nn.Module):

    def __init__(self, in_size, out_size, dropout_r=0.0, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x


class MLP(nn.Module):

    def __init__(self, in_size, mid_size, out_size, dropout_r=0.0, use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class BC(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, __C, atten=False):
        super(BC, self).__init__()
        self.__C = __C
        self.v_net = MLP([__C.IMG_FEAT_SIZE, __C.BA_HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        self.q_net = MLP([__C.HIDDEN_SIZE, __C.BA_HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        if not atten:
            self.p_net = nn.AvgPool1d(__C.K_TIMES, stride=__C.K_TIMES)
        else:
            self.dropout = nn.Dropout(__C.CLASSIFER_DROPOUT_R)
            self.h_mat = nn.Parameter(torch.Tensor(1, __C.GLIMPSE, 1, __C.BA_HIDDEN_SIZE).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, __C.GLIMPSE, 1, 1).normal_())

    def forward(self, v, q):
        v_ = self.dropout(self.v_net(v))
        q_ = self.q_net(q)
        logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        return logits

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        logits = logits.unsqueeze(1)
        logits = self.p_net(logits).squeeze(1) * self.__C.K_TIMES
        return logits


class BiAttention(nn.Module):

    def __init__(self, __C):
        super(BiAttention, self).__init__()
        self.__C = __C
        self.logits = weight_norm(BC(__C, True), name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)
        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.__C.GLIMPSE, v_num * q_num), 2)
            return p.view(-1, self.__C.GLIMPSE, v_num, q_num), logits
        return logits


class BAN(nn.Module):

    def __init__(self, __C):
        super(BAN, self).__init__()
        self.__C = __C
        self.BiAtt = BiAttention(__C)
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(__C.GLIMPSE):
            b_net.append(BC(__C))
            q_prj.append(MLP([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE], '', __C.DROPOUT_R))
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)

    def forward(self, q, v):
        att, logits = self.BiAtt(v, q)
        for g in range(self.__C.GLIMPSE):
            bi_emb = self.b_net[g].forward_with_weights(v, q, att[:, (g), :, :])
            q = self.q_prj[g](bi_emb.unsqueeze(1)) + q
        return q


class MFB(nn.Module):

    def __init__(self, __C, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_K * __C.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, __C.MFB_K * __C.MFB_O)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        """
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        """
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)
        ques_feat = self.proj_q(ques_feat)
        exp_out = img_feat * ques_feat
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
        z = self.pool(exp_out) * self.__C.MFB_K
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))
        z = z.view(batch_size, -1, self.__C.MFB_O)
        return z, exp_out


class IAtt(nn.Module):

    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(in_size=__C.MFB_O, mid_size=__C.HIDDEN_SIZE, out_size=__C.I_GLIMPSES, dropout_r=__C.DROPOUT_R, use_relu=True)

    def forward(self, img_feat, ques_att_feat):
        """
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        """
        ques_att_feat = ques_att_feat.unsqueeze(1)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)
        iatt_maps = self.mlp(z)
        iatt_maps = F.softmax(iatt_maps, dim=1)
        iatt_feat_list = []
        for i in range(self.__C.I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]
            mask = mask * img_feat
            mask = torch.sum(mask, dim=1)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)
        return iatt_feat


class QAtt(nn.Module):

    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(in_size=__C.LSTM_OUT_SIZE, mid_size=__C.HIDDEN_SIZE, out_size=__C.Q_GLIMPSES, dropout_r=__C.DROPOUT_R, use_relu=True)

    def forward(self, ques_feat):
        """
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        """
        qatt_maps = self.mlp(ques_feat)
        qatt_maps = F.softmax(qatt_maps, dim=1)
        qatt_feat_list = []
        for i in range(self.__C.Q_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]
            mask = mask * ques_feat
            mask = torch.sum(mask, dim=1)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)
        return qatt_feat


class CoAtt(nn.Module):

    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C
        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE'][1]
        img_att_feat_size = img_feat_size * __C.I_GLIMPSES
        ques_att_feat_size = __C.LSTM_OUT_SIZE * __C.Q_GLIMPSES
        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)
        if self.__C.HIGH_ORDER:
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        """
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        """
        ques_feat = self.q_att(ques_feat)
        fuse_feat = self.i_att(img_feat, ques_feat)
        if self.__C.HIGH_ORDER:
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)
        else:
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))
            z = z.squeeze(1)
        return z


class Net(nn.Module):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.adapter = Adapter(__C)
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE)
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.lstm = nn.LSTM(input_size=__C.WORD_EMBED_SIZE, hidden_size=__C.LSTM_OUT_SIZE, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.dropout_lstm = nn.Dropout(__C.DROPOUT_R)
        self.backbone = CoAtt(__C)
        if __C.HIGH_ORDER:
            self.proj = nn.Linear(2 * __C.MFB_O, answer_size)
        else:
            self.proj = nn.Linear(__C.MFB_O, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)
        ques_feat = self.embedding(ques_ix)
        ques_feat = self.dropout(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)
        ques_feat = self.dropout_lstm(ques_feat)
        z = self.backbone(img_feat, ques_feat)
        proj_feat = self.proj(z)
        return proj_feat


class AttnMap(nn.Module):
    """
    implementation of top down attention
    """

    def __init__(self, __C):
        super(AttnMap, self).__init__()
        self.__C = __C
        self.linear_q = weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE), dim=None)
        self.linear_v = weight_norm(nn.Linear(__C.IMG_FEAT_SIZE, __C.IMG_FEAT_SIZE), dim=None)
        self.nonlinear = MLP([__C.IMG_FEAT_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        self.linear = weight_norm(nn.Linear(__C.HIDDEN_SIZE, 1), dim=None)

    def forward(self, q, v):
        v = self.linear_v(v)
        q = self.linear_q(q)
        logits = self.logits(q, v)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, q, v):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class TDA(nn.Module):

    def __init__(self, __C):
        super(TDA, self).__init__()
        self.__C = __C
        self.v_att = AttnMap(__C)
        self.q_net = MLP([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE])
        self.v_net = MLP([__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE])

    def forward(self, q, v):
        att = self.v_att(q, v)
        atted_v = (att * v).sum(1)
        q_repr = self.q_net(q)
        v_repr = self.v_net(atted_v)
        joint_repr = q_repr * v_repr
        return joint_repr


class MHAtt(nn.Module):

    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)).transpose(1, 2)
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.__C.HIDDEN_SIZE)
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1000000000.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class FFN(nn.Module):

    def __init__(self, __C):
        super(FFN, self).__init__()
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.FF_SIZE, out_size=__C.HIDDEN_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)

    def forward(self, x):
        return self.mlp(x)


class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SA(nn.Module):

    def __init__(self, __C):
        super(SA, self).__init__()
        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(self.mhatt(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.ffn(y)))
        return y


class SGA(nn.Module):

    def __init__(self, __C):
        super(SGA, self).__init__()
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(v=x, k=x, q=x, mask=x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(v=y, k=y, q=x, mask=y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class MCA_ED(nn.Module):

    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        for enc in self.enc_list:
            y = enc(y, y_mask)
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)
        return y, x


class AttFlat(nn.Module):

    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.FLAT_MLP_SIZE, out_size=__C.FLAT_GLIMPSES, dropout_r=__C.DROPOUT_R, use_relu=True)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES, __C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1000000000.0)
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i:i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_size': 4, 'mid_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_MILVLG_openvqa(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

