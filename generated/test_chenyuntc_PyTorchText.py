import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
dataset = _module
fold_dataset = _module
main4model = _module
main_boost = _module
main_fold = _module
search_all = _module
search_aug_noMultimodel_weight1 = _module
search_multimodel = _module
search_paris = _module
search_test = _module
search_weight5 = _module
searchstack = _module
searchstack_new = _module
test = _module
test_aug_multimodel = _module
test_stack = _module
main = _module
BasicModule = _module
CNNText_inception = _module
FastText3 = _module
LSTMText = _module
MultiCNNTextBNDeep = _module
MultiModelAll = _module
MultiModelAll2 = _module
MultiModelAll4zhihu = _module
RCNN = _module
models = _module
alias_multinomial = _module
loss = _module
nce = _module
rep = _module
embedding2matrix = _module
label2id = _module
question2array = _module
ensamble = _module
graph2vec = _module
mer_csv = _module
search = _module
utils = _module
calculate_score = _module
optimizer = _module
visualize = _module

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


from torch.utils import data


import torch as t


import numpy as np


import random


import time


from torch.autograd import Variable


import torch


from torch import nn


from collections import OrderedDict


import torch.nn as nn


from itertools import chain


import torchvision as tv


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, change_opt=True):
        None
        data = t.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
                self.opt.embedding_path = None
                self.__init__(self.opt)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self

    def save(self, name=None, new=False):
        prefix = 'checkpoints/' + self.model_name + '_' + self.opt.type_ + '_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name
        if new:
            data = {'opt': self.opt.state_dict(), 'd': self.state_dict()}
        else:
            data = self.state_dict()
        t.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = [p for p in self.parameters() if id(p) not in ignored_params]
        if lr2 is None:
            lr2 = lr1 * 0.5
        optimizer = t.optim.Adam([dict(params=base_params, weight_decay=weight_decay, lr=lr1), {'params': self.encoder.parameters(), 'lr': lr2}])
        return optimizer


class Inception(nn.Module):

    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert co % 4 == 0
        cos = [co / 4] * 4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[0], 1, stride=1))]))
        self.branch2 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[1], 1)), ('norm1', nn.BatchNorm1d(cos[1])), ('relu1', nn.ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1))]))
        self.branch3 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)), ('norm1', nn.BatchNorm1d(cos[2])), ('relu1', nn.ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2))]))
        self.branch4 = nn.Sequential(OrderedDict([('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1))]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class CNNText_inception(BasicModule):

    def __init__(self, opt):
        super(CNNText_inception, self).__init__()
        incept_dim = opt.inception_dim
        self.model_name = 'CNNText_inception'
        self.opt = opt
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.title_conv = nn.Sequential(Inception(opt.embedding_dim, incept_dim), Inception(incept_dim, incept_dim), nn.MaxPool1d(opt.title_seq_len))
        self.content_conv = nn.Sequential(Inception(opt.embedding_dim, incept_dim), Inception(incept_dim, incept_dim), nn.MaxPool1d(opt.content_seq_len))
        self.fc = nn.Sequential(nn.Linear(incept_dim * 2, opt.linear_hidden_size), nn.BatchNorm1d(opt.linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            None
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        if self.opt.static:
            title = title.detach()
            content = content.detach(0)
        title_out = self.title_conv(title.permute(0, 2, 1))
        content_out = self.content_conv(content.permute(0, 2, 1))
        out = torch.cat((title_out, content_out), 1).view(content_out.size(0), -1)
        out = self.fc(out)
        return out


class FastText3(BasicModule):

    def __init__(self, opt):
        super(FastText3, self).__init__()
        self.model_name = 'FastText3'
        self.opt = opt
        self.pre1 = nn.Sequential(nn.Linear(opt.embedding_dim, opt.embedding_dim * 2), nn.BatchNorm1d(opt.embedding_dim * 2), nn.ReLU(True))
        self.pre2 = nn.Sequential(nn.Linear(opt.embedding_dim, opt.embedding_dim * 2), nn.BatchNorm1d(opt.embedding_dim * 2), nn.ReLU(True))
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.fc = nn.Sequential(nn.Linear(opt.embedding_dim * 4, opt.linear_hidden_size), nn.BatchNorm1d(opt.linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            None
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title_em = self.encoder(title)
        content_em = self.encoder(content)
        title_size = title_em.size()
        content_size = content_em.size()
        title_2 = self.pre1(title_em.contiguous().view(-1, 256)).view(title_size[0], title_size[1], -1)
        content_2 = self.pre2(content_em.contiguous().view(-1, 256)).view(content_size[0], content_size[1], -1)
        title_ = t.mean(title_2, dim=1)
        content_ = t.mean(content_2, dim=1)
        inputs = t.cat((title_.squeeze(), content_.squeeze()), 1)
        out = self.fc(inputs)
        return out


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class LSTMText(BasicModule):

    def __init__(self, opt):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMText'
        self.opt = opt
        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.title_lstm = nn.LSTM(input_size=opt.embedding_dim, hidden_size=opt.hidden_size, num_layers=opt.num_layers, bias=True, batch_first=False, bidirectional=True)
        self.content_lstm = nn.LSTM(input_size=opt.embedding_dim, hidden_size=opt.hidden_size, num_layers=opt.num_layers, bias=True, batch_first=False, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(opt.kmax_pooling * (opt.hidden_size * 2 * 2), opt.linear_hidden_size), nn.BatchNorm1d(opt.linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        if self.opt.static:
            title = title.detach()
            content = content.detach()
        title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
        title_conv_out = kmax_pooling(title_out, 2, self.opt.kmax_pooling)
        content_conv_out = kmax_pooling(content_out, 2, self.opt.kmax_pooling)
        conv_out = t.cat((title_conv_out, content_conv_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


kernel_sizes = [1, 2, 3, 4]


class MultiCNNTextBNDeep(BasicModule):

    def __init__(self, opt):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.opt = opt
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        title_convs = [nn.Sequential(nn.Conv1d(in_channels=opt.embedding_dim, out_channels=opt.title_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.title_dim), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.title_dim, out_channels=opt.title_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.title_dim), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.title_seq_len - kernel_size * 2 + 2)) for kernel_size in kernel_sizes]
        content_convs = [nn.Sequential(nn.Conv1d(in_channels=opt.embedding_dim, out_channels=opt.content_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.content_dim), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.content_dim, out_channels=opt.content_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.content_dim), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.content_seq_len - kernel_size * 2 + 2)) for kernel_size in kernel_sizes]
        self.title_convs = nn.ModuleList(title_convs)
        self.content_convs = nn.ModuleList(content_convs)
        self.fc = nn.Sequential(nn.Linear(len(kernel_sizes) * (opt.title_dim + opt.content_dim), opt.linear_hidden_size), nn.BatchNorm1d(opt.linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        if self.opt.static:
            title.detach()
            content.detach()
        title_out = [title_conv(title.permute(0, 2, 1)) for title_conv in self.title_convs]
        content_out = [content_conv(content.permute(0, 2, 1)) for content_conv in self.content_convs]
        conv_out = t.cat(title_out + content_out, dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


tfmt = '%m%d_%H%M%S'


class Config(object):
    """
    并不是所有的配置都生效,实际运行中只根据需求获取自己需要的参数
    """
    loss = 'multilabelloss'
    model = 'CNNText'
    title_dim = 100
    content_dim = 200
    num_classes = 1999
    embedding_dim = 256
    linear_hidden_size = 2000
    kmax_pooling = 2
    hidden_size = 256
    num_layers = 2
    inception_dim = 512
    vocab_size = 411720
    kernel_size = 3
    kernel_sizes = [2, 3, 4]
    title_seq_len = 50
    content_seq_len = 250
    type_ = 'word'
    all = False
    embedding_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/char_embedding.npz'
    train_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/train.npz'
    labels_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/labels.json'
    test_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/test.npz'
    result_path = 'csv/' + time.strftime(tfmt) + '.csv'
    shuffle = True
    num_workers = 4
    pin_memory = True
    batch_size = 128
    env = time.strftime(tfmt)
    plot_every = 10
    max_epoch = 100
    lr = 0.005
    lr2 = 0.001
    min_lr = 1e-05
    lr_decay = 0.99
    weight_decay = 0
    weight = 1
    decay_every = 3000
    model_path = None
    optimizer_path = 'optimizer.pth'
    debug_file = '/tmp/debug2'
    debug = False
    gpu1 = False
    floyd = False
    zhuge = False
    model_names = ['MultiCNNTextBNDeep', 'CNNText_inception', 'RCNN', 'LSTMText', 'CNNText_inception']
    model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788', 'checkpoints/CNNText_tmp_0.380390420742', 'checkpoints/RCNN_word_0.373609030286', 'checkpoints/LSTMText_word_0.381833388089', 'checkpoints/CNNText_tmp_0.376364647145']
    static = False
    val = False
    fold = 1
    augument = True
    model_num = 7
    data_root = '/data/text/zhihu/result/'
    labels_file = '/home/a/code/pytorch/zhihu/ddd/labels.json'
    val = '/home/a/code/pytorch/zhihu/ddd/val.npz'


class MultiModelAll(BasicModule):

    def __init__(self, opt):
        super(MultiModelAll, self).__init__()
        self.model_name = 'MultiModelAll'
        self.opt = opt
        self.models = []
        self.word_embedding = nn.Embedding(411720, 256)
        self.char_embedding = nn.Embedding(11973, 256)
        if opt.embedding_path:
            self.word_embedding.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path.replace('char', 'word'))['vector']))
            self.char_embedding.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path.replace('word', 'char'))['vector']))
        for _name, _path in zip(opt.model_names, opt.model_paths):
            tmp_config = Config().parse(opt.state_dict(), print_=False)
            tmp_config.embedding_path = None
            _model = getattr(models, _name)(tmp_config)
            if _path is not None:
                _model.load(_path)
            _model.encoder = self.char_embedding if _model.opt.type_ == 'char' else self.word_embedding
            self.models.append(_model)
        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes, self.model_num))
        assert self.opt.loss == 'bceloss'

    def reinit(self):
        pass

    def load(self, path, **kwargs):
        self.load_state_dict(t.load(path)['d'])

    def forward(self, char, word):
        weights = t.nn.functional.softmax(self.weights)
        outs = []
        for ii, model in enumerate(self.models):
            if model.opt.type_ == 'char':
                out = t.sigmoid(model(*char))
            else:
                out = t.sigmoid(model(*word))
            out = out * weights[:, (ii)].contiguous().view(1, -1).expand_as(out)
            outs.append(out)
        return sum(outs)

    def get_optimizer(self, lr1=0.001, lr2=0.0003, lr3=0.0003, weight_decay=0):
        encoders = list(self.char_embedding.parameters()) + list(self.word_embedding.parameters())
        other_params = [param_ for model_ in self.models for name_, param_ in model_.named_parameters() if name_.find('encoder') == -1]
        new_params = [self.weights]
        optimizer = t.optim.Adam([dict(params=other_params, weight_decay=weight_decay, lr=lr1), dict(params=encoders, weight_decay=weight_decay, lr=lr2), dict(params=new_params, weight_decay=weight_decay, lr=lr3)])
        return optimizer


class MultiModelAll2(BasicModule):

    def __init__(self, opt):
        super(MultiModelAll2, self).__init__()
        self.model_name = 'MultiModelAll2'
        self.opt = opt
        self.models = []
        for _name, _path in zip(opt.model_names, opt.model_paths):
            tmp_config = Config().parse(opt.state_dict(), print_=False)
            tmp_config.embedding_path = None
            _model = getattr(models, _name)(tmp_config)
            if _path is not None:
                _model.load(_path)
            self.models.append(_model)
        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes, self.model_num))
        assert self.opt.loss == 'bceloss'
        self.eval()

    def reinit(self):
        pass

    def forward(self, char, word):
        weights = t.nn.functional.softmax(self.weights)
        outs = []
        for ii, model in enumerate(self.models):
            if model.opt.type_ == 'char':
                out = t.sigmoid(model(*char))
            else:
                out = t.sigmoid(model(*word))
            if self.opt.static:
                out = out.detach()
            out = out * weights[:, (ii)].contiguous().view(1, -1).expand_as(out)
            outs.append(out)
        return sum(outs)

    def get_optimizer(self, lr1=0.0001, lr2=0.0001, lr3=0, weight_decay=0):
        other_params = [param_ for model_ in self.models for name_, param_ in model_.named_parameters() if name_.find('encoder') == -1]
        encoders = [param_ for model_ in self.models for name_, param_ in model_.named_parameters() if name_.find('encoder') != -1]
        new_params = [self.weights]
        optimizer = t.optim.Adam([dict(params=other_params, weight_decay=weight_decay, lr=lr1), dict(params=encoders, weight_decay=weight_decay, lr=lr2), dict(params=new_params, weight_decay=weight_decay, lr=lr3)])
        return optimizer


class MultiModelAll4zhihu(BasicModule):

    def __init__(self, opt):
        super(MultiModelAll4zhihu, self).__init__()
        self.model_name = 'MultiModelAll4zhihu'
        self.opt = opt
        self.models = []
        self.word_embedding = nn.Embedding(411720, 256)
        self.char_embedding = nn.Embedding(11973, 256)
        model_opts = t.load(opt.model_path + '.json')
        for _name, _path, model_opt_ in zip(opt.model_names, opt.model_paths, model_opts):
            tmp_config = Config().parse(model_opt_, print_=False)
            tmp_config.embedding_path = None
            _model = getattr(models, _name)(tmp_config)
            _model.encoder = self.char_embedding if _model.opt.type_ == 'char' else self.word_embedding
            self.models.append(_model)
        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes, self.model_num))
        self.load(opt.model_path)

    def load(self, path, **kwargs):
        self.load_state_dict(t.load(path)['d'])

    def forward(self, char, word):
        weights = t.nn.functional.softmax(self.weights)
        outs = []
        for ii, model in enumerate(self.models):
            if model.opt.type_ == 'char':
                out = t.sigmoid(model(*char))
            else:
                out = t.sigmoid(model(*word))
            out = out * weights[:, (ii)].contiguous().view(1, -1).expand_as(out)
            outs.append(out)
        return sum(outs)

    def get_optimizer(self, lr1=0.001, lr2=0.0003, lr3=0.0003, weight_decay=0):
        encoders = list(self.char_embedding.parameters()) + list(self.word_embedding.parameters())
        other_params = [param_ for model_ in self.models for name_, param_ in model_.named_parameters() if name_.find('encoder') == -1]
        new_params = [self.weights]
        optimizer = t.optim.Adam([dict(params=other_params, weight_decay=weight_decay, lr=lr1), dict(params=encoders, weight_decay=weight_decay, lr=lr2), dict(params=new_params, weight_decay=weight_decay, lr=lr3)])
        return optimizer


class RCNN(BasicModule):

    def __init__(self, opt):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'
        self.opt = opt
        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.title_lstm = nn.LSTM(input_size=opt.embedding_dim, hidden_size=opt.hidden_size, num_layers=opt.num_layers, bias=True, batch_first=False, bidirectional=True)
        self.title_conv = nn.Sequential(nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim, out_channels=opt.title_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.title_dim), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.title_dim, out_channels=opt.title_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.title_dim), nn.ReLU(inplace=True))
        self.content_lstm = nn.LSTM(input_size=opt.embedding_dim, hidden_size=opt.hidden_size, num_layers=opt.num_layers, bias=True, batch_first=False, bidirectional=True)
        self.content_conv = nn.Sequential(nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim, out_channels=opt.content_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.content_dim), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.content_dim, out_channels=opt.content_dim, kernel_size=kernel_size), nn.BatchNorm1d(opt.content_dim), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(opt.kmax_pooling * (opt.title_dim + opt.content_dim), opt.linear_hidden_size), nn.BatchNorm1d(opt.linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        if self.opt.static:
            title.detach()
            content.detach()
        title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        title_em = title.permute(0, 2, 1)
        title_out = t.cat((title_out, title_em), dim=1)
        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_em = content.permute(0, 2, 1)
        content_out = t.cat((content_out, content_em), dim=1)
        title_conv_out = kmax_pooling(self.title_conv(title_out), 2, self.opt.kmax_pooling)
        content_conv_out = kmax_pooling(self.content_conv(content_out), 2, self.opt.kmax_pooling)
        conv_out = t.cat((title_conv_out, content_conv_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


class AliasMethod(object):
    """
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def draw(self, N):
        """
            Draw N samples from multinomial
        """
        K = self.alias.size(0)
        kk = torch.LongTensor(np.random.randint(0, K, size=N))
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj


class IndexLinear(nn.Linear):
    """A linear layer that only decodes the results of provided indices
    Args:
        input: the list of embedding
        indices: the indices of interests.
    Shape:
        - Input :math:`(N, in\\_features)`
        - Indices :math:`(N, 1+N_r)` where `max(M) <= N`
    Return:
        - out :math:`(N, 1+N_r)`
    """

    def forward(self, input, indices=None):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """
        if indices is None:
            return super(IndexLinear, self).forward(input)
        input = input.unsqueeze(1)
        target_batch = self.weight.index_select(0, indices.view(-1)).view(indices.size(0), indices.size(1), -1).transpose(1, 2)
        bias = self.bias.index_select(0, indices.view(-1)).view(indices.size(0), 1, indices.size(1))
        out = torch.baddbmm(1, bias, 1, input, target_batch)
        return out.squeeze()

    def reset_parameters(self):
        init_range = 0.1
        self.bias.data.fill_(0)
        self.weight.data.uniform_(-init_range, init_range)


class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        nhidden: hidden size of LSTM(a.k.a the output size)
        ntokens: vocabulary size
        noise: the distribution of noise
        noise_ratio: $rac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        decoder: the decoder matrix
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - decoder: :math:`(E, V)` where `E = embedding size`
    """

    def __init__(self, ntokens, nhidden, noise, noise_ratio=10, norm_term=9, size_average=True, decoder_weight=None):
        super(NCELoss, self).__init__()
        self.noise = noise
        self.alias = AliasMethod(noise)
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.ntokens = ntokens
        self.size_average = size_average
        self.decoder = IndexLinear(nhidden, ntokens)
        if decoder_weight:
            self.decoder.weight = decoder_weight

    def forward(self, input, target=None):
        """compute the loss with output and the desired target
        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.
        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`
        Return:
            the scalar NCELoss Variable ready for backward
        """
        length = target.size(0)
        if self.training:
            assert input.size(0) == target.size(0)
            noise_samples = self.alias.draw(self.noise_ratio).unsqueeze(0).repeat(length, 1)
            data_prob, noise_in_data_probs = self._get_prob(input, target.data, noise_samples)
            noise_probs = Variable(self.noise[noise_samples.view(-1)].view_as(noise_in_data_probs))
            rnn_loss = torch.log(data_prob / (data_prob + self.noise_ratio * Variable(self.noise[target.data])))
            noise_loss = torch.sum(torch.log(self.noise_ratio * noise_probs / (noise_in_data_probs + self.noise_ratio * noise_probs)), 1)
            loss = -1 * torch.sum(rnn_loss + noise_loss)
        else:
            out = self.decoder(input, indices=target.unsqueeze(1))
            nll = out.sub(self.norm_term)
            loss = -1 * nll.sum()
        if self.size_average:
            loss = loss / length
        return loss

    def _get_prob(self, embedding, target_idx, noise_idx):
        """Get the NCE estimated probability for target and noise
        Shape:
            - Embedding: :math:`(N, E)`
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """
        embedding = embedding
        indices = Variable(torch.cat([target_idx.unsqueeze(1), noise_idx], dim=1))
        probs = self.decoder(embedding, indices)
        probs = probs.sub(self.norm_term).exp()
        return probs[:, (0)], probs[:, 1:]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IndexLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_chenyuntc_PyTorchText(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

