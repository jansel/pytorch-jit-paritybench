import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
data_review = _module
framework = _module
fusion = _module
models = _module
prediction = _module
main = _module
d_attn = _module
daml = _module
deepconn = _module
mpcn = _module
narre = _module
data_pro = _module

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


from torch.utils.data import Dataset


import torch


import torch.nn as nn


import time


import torch.nn.functional as F


import random


import math


import torch.optim as optim


from torch.utils.data import DataLoader


class SelfAtt(nn.Module):
    """
    self attention for interaction
    """

    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)
        out = self.encoder(fea)
        return out.permute(1, 0, 2)


class FusionLayer(nn.Module):
    """
    Fusion Layer for user feature and item feature
    """

    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)
        self.drop_out = nn.Dropout(0.5)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, u_out, i_out):
        if self.opt.self_att:
            out = self.attn(u_out, i_out)
            s_u_out, s_i_out = torch.split(out, out.size(1) // 2, 1)
            u_out = u_out + s_u_out
            i_out = i_out + s_i_out
        if self.opt.r_id_merge == 'cat':
            u_out = u_out.reshape(u_out.size(0), -1)
            i_out = i_out.reshape(i_out.size(0), -1)
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)
        if self.opt.ui_merge == 'cat':
            out = torch.cat([u_out, i_out], 1)
        elif self.opt.ui_merge == 'add':
            out = u_out + i_out
        else:
            out = u_out * i_out
        return out


class FM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, 1)
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):
        """
        y = w_0 + \\sum {w_ix_i} + \\sum_{i=1}\\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        """
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)
        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]


class LFM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM, self).__init__()
        self.fc = nn.Linear(dim, 1)
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, 0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)

    def forward(self, feature, *args, **kwargs):
        return F.relu(self.fc(feature))


class NFM(nn.Module):
    """
    Neural FM
    """

    def __init__(self, dim):
        super(NFM, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, 1)
        self.fm_V = nn.Parameter(torch.randn(16, dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, input_vec, *args):
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)
        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)
        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out


class PredictionLayer(nn.Module):
    """
        Rating Prediciton Methods
        - LFM: Latent Factor Model
        - (N)FM: (Neural) Factorization Machine
        - MLP
        - SUM
    """

    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == 'fm':
            self.model = FM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'lfm':
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'mlp':
            self.model = MLP(opt.feature_dim)
        elif opt.output == 'nfm':
            self.model = NFM(opt.feature_dim)
        else:
            self.model = torch.sum

    def forward(self, feature, uid, iid):
        if self.output == 'lfm' or 'fm' or 'nfm':
            return self.model(feature, uid, iid)
        else:
            return self.model(feature, 1, keepdim=True)


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.net = Net(opt)
        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                feature_dim = self.opt.id_emb_size * 2
        elif self.opt.r_id_merge == 'cat':
            feature_dim = self.opt.id_emb_size * self.opt.num_fea
        else:
            feature_dim = self.opt.id_emb_size
        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        user_feature, item_feature = self.net(datas)
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)
        return output

    def load(self, path):
        """
        加载指定模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, epoch=None, name=None, opt=None):
        """
        保存模型
        """
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name


class Net(nn.Module):

    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt
        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num
        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)
        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list):
        reviews = self.word_embs(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)
        id_emb = self.id_embedding(ids)
        u_i_id_emb = self.u_i_id_embedding(ids_list)
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))
        rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(u_i_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = fea * att_weight
        r_fea = r_fea.sum(1)
        r_fea = self.dropout(r_fea)
        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)

    def reset_para(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)
        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)
        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)
        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v)
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)


class D_ATTN(nn.Module):
    """
    Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction
    Rescys 2017
    """

    def __init__(self, opt):
        super(D_ATTN, self).__init__()
        self.opt = opt
        self.num_fea = 1
        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        u_fea = self.user_net(user_doc)
        i_fea = self.item_net(item_doc)
        return u_fea, i_fea


class LocalAttention(nn.Module):

    def __init__(self, seq_len, win_size, emb_size, filters_num):
        super(LocalAttention, self).__init__()
        self.att_conv = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(win_size, emb_size), padding=((win_size - 1) // 2, 0)), nn.Sigmoid())
        self.cnn = nn.Conv2d(1, filters_num, kernel_size=(1, emb_size))

    def forward(self, x):
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        out = x.mul(score)
        out = out.unsqueeze(1)
        out = torch.tanh(self.cnn(out)).squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


class GlobalAttention(nn.Module):

    def __init__(self, seq_len, emb_size, filters_size=[2, 3, 4], filters_num=100):
        super(GlobalAttention, self).__init__()
        self.att_conv = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(seq_len, emb_size)), nn.Sigmoid())
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, emb_size)) for k in filters_size])

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.att_conv(x)
        x = x.mul(score)
        conv_outs = [torch.tanh(cnn(x).squeeze(3)) for cnn in self.convs]
        conv_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return conv_outs


class DAML(nn.Module):
    """
    KDD 2019 DAML
    """

    def __init__(self, opt):
        super(DAML, self).__init__()
        self.opt = opt
        self.num_fea = 2
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.word_cnn = nn.Conv2d(1, 1, (5, opt.word_dim), padding=(2, 0))
        self.user_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.user_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
        self.unfold = nn.Unfold((3, opt.filters_num), padding=(1, 0))
        self.user_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.item_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
        self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)
        self.reset_para()

    def forward(self, datas):
        """
        user_reviews, item_reviews, uids, iids,         user_item2id, item_user2id, user_doc, item_doc = datas
        """
        _, _, uids, iids, _, _, user_doc, item_doc = datas
        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)
        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)
        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)
        user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)
        uid_emb = self.uid_embedding(uids)
        iid_emb = self.iid_embedding(iids)
        use_fea = torch.stack([user_doc_fea, uid_emb], 1)
        item_fea = torch.stack([item_doc_fea, iid_emb], 1)
        return use_fea, item_fea

    def local_attention_cnn(self, word_embs, doc_cnn):
        """
        :Eq1 - Eq7
        """
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        d_fea = doc_cnn(word_embs.unsqueeze(1))
        return d_fea

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        """
        :Eq11 - Eq13
        feature: (?, 100, DOC_LEN ,1)
        attention: (?, DOC_LEN)
        """
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)
        attention = attention.reshape(bs, 1, doc_len, 1)
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)
        pools = pools.transpose(2, 3)
        abs_fea = cnn(pools).squeeze(3)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))
        return abs_fea

    def reset_para(self):
        cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]
        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        fcs = [self.user_fc, self.item_fc]
        for fc in fcs:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)
        w2v = torch.from_numpy(np.load(self.opt.w2v_path))
        self.user_word_embs.weight.data.copy_(w2v)
        self.item_word_embs.weight.data.copy_(w2v)


class DeepCoNN(nn.Module):
    """
    deep conn 2017
    """

    def __init__(self, opt, uori='user'):
        super(DeepCoNN, self).__init__()
        self.opt = opt
        self.num_fea = 1
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas
        user_doc = self.user_word_embs(user_doc)
        item_doc = self.item_word_embs(item_doc)
        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        u_fea = self.dropout(self.user_fc_linear(u_fea))
        i_fea = self.dropout(self.item_fc_linear(i_fea))
        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def reset_para(self):
        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)
        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


class Co_Attention(nn.Module):
    """
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    """

    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)
        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
        """
        u_fea: B * L1 * d
        i_fea: B * L2 * d
        return:
        B * L1 * 1
        B * L2 * 1
        """
        u = self.fc_u(u_fea)
        i = self.fc_i(i_fea)
        S = u.matmul(self.M).bmm(i.permute(0, 2, 1))
        if self.pooling == 'max':
            u_score = S.max(2)[0]
            i_score = S.max(1)[0]
        else:
            u_score = S.mean(2)
            i_score = S.mean(1)
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)


class MPCN(nn.Module):
    """
    Multi-Pointer Co-Attention Network for Recommendation
    WWW 2018
    """

    def __init__(self, opt, head=3):
        """
        head: the number of pointers
        """
        super(MPCN, self).__init__()
        self.opt = opt
        self.num_fea = 1
        self.head = head
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)
        self.review_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=True, pooling='max') for _ in range(head)])
        self.word_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=False, pooling='avg') for _ in range(head)])
        self.u_fc = self.fc_layer()
        self.i_fc = self.fc_layer()
        self.drop_out = nn.Dropout(opt.drop_out)
        self.reset_para()

    def fc_layer(self):
        return nn.Sequential(nn.Linear(self.opt.word_dim * self.head, self.opt.word_dim), nn.ReLU(), nn.Linear(self.opt.word_dim, self.opt.id_emb_size))

    def forward(self, datas):
        """
        user_reviews, item_reviews, uids, iids,         user_item2id, item_user2id, user_doc, item_doc = datas
        :user_reviews: B * L1 * N
        :item_reviews: B * L2 * N
        """
        user_reviews, item_reviews, _, _, _, _, _, _ = datas
        u_word_embs = self.user_word_embs(user_reviews)
        i_word_embs = self.item_word_embs(item_reviews)
        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)
        u_fea = []
        i_fea = []
        for i in range(self.head):
            r_coatt = self.review_coatt[i]
            w_coatt = self.word_coatt[i]
            p_u, p_i = r_coatt(u_reviews, i_reviews)
            u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)
            i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)
            u_words = self.user_word_embs(u_r_words.squeeze(2).long())
            i_words = self.item_word_embs(i_r_words.squeeze(2).long())
            p_u, p_i = w_coatt(u_words, i_words)
            u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
            i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)
            u_fea.append(u_w_fea)
            i_fea.append(i_w_fea)
        u_fea = torch.cat(u_fea, 1)
        i_fea = torch.cat(i_fea, 1)
        u_fea = self.drop_out(self.u_fc(u_fea))
        i_fea = self.drop_out(self.i_fc(i_fea))
        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def review_gate(self, reviews):
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.u_fc[0], self.u_fc[-1], self.i_fc[0], self.i_fc[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


class NARRE(nn.Module):
    """
    NARRE: WWW 2018
    """

    def __init__(self, opt):
        super(NARRE, self).__init__()
        self.opt = opt
        self.num_fea = 2
        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        u_fea = self.user_net(user_reviews, uids, user_item2id)
        i_fea = self.item_net(item_reviews, iids, item_user2id)
        return u_fea, i_fea


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Co_Attention,
     lambda: ([], {'dim': 4, 'gumbel': 4, 'pooling': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FM,
     lambda: ([], {'dim': 4, 'user_num': 4, 'item_num': 4}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (FusionLayer,
     lambda: ([], {'opt': _mock_config(self_att=4, id_emb_size=4, num_heads=4, feature_dim=4, r_id_merge=4, ui_merge=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (GlobalAttention,
     lambda: ([], {'seq_len': 4, 'emb_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LocalAttention,
     lambda: ([], {'seq_len': 4, 'win_size': 4, 'emb_size': 4, 'filters_num': 4}),
     lambda: ([torch.rand([4, 2, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NFM,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SelfAtt,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_ShomyLiu_Neu_Review_Rec(_paritybench_base):
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

