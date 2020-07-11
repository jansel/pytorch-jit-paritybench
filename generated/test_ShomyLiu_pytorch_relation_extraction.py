import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
filternyt = _module
nyt = _module
main_att = _module
main_mil = _module
BasicModule = _module
PCNN_ATT = _module
PCNN_ONE = _module
models = _module
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


from torch.utils.data import Dataset


import numpy as np


import torch


from torch.utils.data import DataLoader


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


import time


from torch.autograd import Variable


class BasicModule(torch.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '.pth'
        torch.save(self.state_dict(), name)
        return name


class PCNN_ATT(BasicModule):
    """
    Lin 2016 Att PCNN
    """

    def __init__(self, opt):
        super(PCNN_ATT, self).__init__()
        self.opt = opt
        self.model_name = 'PCNN_ATT'
        self.test_scale_p = 0.5
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        all_filter_num = self.opt.filters_num * len(self.opt.filters)
        rel_dim = all_filter_num
        if self.opt.use_pcnn:
            rel_dim = all_filter_num * 3
            masks = torch.LongTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            if self.opt.use_gpu:
                masks = masks
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False
        self.rel_embs = nn.Parameter(torch.randn(self.opt.rel_num, rel_dim))
        self.rel_bias = nn.Parameter(torch.randn(self.opt.rel_num))
        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.opt.filters])
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        """
        use xavier to init
        """
        nn.init.xavier_uniform(self.rel_embs)
        nn.init.uniform(self.rel_bias)
        for conv in self.convs:
            nn.init.xavier_uniform(conv.weight)
            nn.init.uniform(conv.bias)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v
        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)
        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v)
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def init_int_constant(self, num):
        """
        a util function for generating a LongTensor Variable
        """
        if self.opt.use_gpu:
            return Variable(torch.LongTensor([num]))
        else:
            return Variable(torch.LongTensor([num]))

    def mask_piece_pooling(self, x, mask):
        """
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        """
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks + x
        x = torch.max(x, 1)[0] - 100
        return x.view(-1, x.size(1) * x.size(2))

    def piece_max_pooling(self, x, insPool):
        """
        piecewise pool into 3 segements
        x: the batch data
        insPool: the batch Pool
        """
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()
            pool = split_pool[i].squeeze().data
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)
            seg_2 = ins[:, pool[0]:pool[1]].max(1)[0].unsqueeze(1)
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)
            batch_res.append(piece_max_pool)
        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, label=None):
        self.bags_feature = self.get_bags_feature(x)
        if label is None:
            assert self.training is False
            return self.test(x)
        else:
            assert self.training is True
            return self.fit(x, label)

    def fit(self, x, label):
        """
        train process
        """
        x = self.get_batch_feature(label)
        x = self.dropout(x)
        out = x.mm(self.rel_embs.t()) + self.rel_bias
        if self.opt.use_gpu:
            v_label = torch.LongTensor(label)
        else:
            v_label = torch.LongTensor(label)
        ce_loss = F.cross_entropy(out, Variable(v_label))
        return ce_loss

    def test(self, x):
        """
        test process
        """
        pre_y = []
        for label in range(0, self.opt.rel_num):
            labels = [label for _ in range(len(x))]
            bags_feature = self.get_batch_feature(labels)
            out = self.test_scale_p * bags_feature.mm(self.rel_embs.t()) + self.rel_bias
            pre_y.append(out.unsqueeze(1))
        res = torch.cat(pre_y, 1).max(1)[0]
        return F.softmax(res, 1).t()

    def get_batch_feature(self, labels):
        """
        Using Attention to get all bags embedding in a batch
        """
        batch_feature = []
        for bag_embs, label in zip(self.bags_feature, labels):
            alpha = bag_embs.mm(self.rel_embs[label].view(-1, 1))
            bag_embs = bag_embs * F.softmax(alpha, 0)
            bag_vec = torch.sum(bag_embs, 0)
            batch_feature.append(bag_vec.unsqueeze(0))
        return torch.cat(batch_feature, 0)

    def get_bags_feature(self, bags):
        """
        get all bags embedding in one batch before Attention
        """
        bags_feature = []
        for bag in bags:
            if self.opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)
            bag_embs = self.get_ins_emb(data)
            bags_feature.append(bag_embs)
        return bags_feature

    def get_ins_emb(self, x):
        """
        x: all instance in a Bag
        """
        insEnt, _, insX, insPFs, insPool, mask = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]
        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)
        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [self.mask_piece_pooling(i, mask) for i in x]
        x = torch.cat(x, 1).tanh()
        return x


class PCNN_ONE(BasicModule):
    """
    Zeng 2015 DS PCNN
    """

    def __init__(self, opt):
        super(PCNN_ONE, self).__init__()
        self.opt = opt
        self.model_name = 'PCNN_ONE'
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.opt.filters])
        all_filter_num = self.opt.filters_num * len(self.opt.filters)
        if self.opt.use_pcnn:
            all_filter_num = all_filter_num * 3
            masks = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            if self.opt.use_gpu:
                masks = masks
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False
        self.linear = nn.Linear(all_filter_num, self.opt.rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        """
        use xavier to init
        """
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v
        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)
        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v)
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def mask_piece_pooling(self, x, mask):
        """
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        """
        x = x.unsqueeze(-1).permute(0, 2, 1, -1)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks.float() + x
        x = torch.max(x, 1)[0] - torch.FloatTensor([100])
        x = x.view(-1, x.size(1) * x.size(2))
        return x

    def piece_max_pooling(self, x, insPool):
        """
        old version piecewise
        """
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()
            pool = split_pool[i].squeeze().data
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)
            seg_2 = ins[:, pool[0]:pool[1]].max(1)[0].unsqueeze(1)
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)
            batch_res.append(piece_max_pool)
        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, train=False):
        insEnt, _, insX, insPFs, insPool, insMasks = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]
        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)
        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)
        x = [conv(x).squeeze(3) for conv in self.convs]
        if self.opt.use_pcnn:
            x = [self.mask_piece_pooling(i, insMasks) for i in x]
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1).tanh()
        x = self.dropout(x)
        x = self.linear(x)
        return x

