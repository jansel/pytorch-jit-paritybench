import sys
_module = sys.modules[__name__]
del sys
earlystopping = _module
lrscheduler = _module
modelcheckpoint = _module
progressbar = _module
trainingmonitor = _module
writetensorboard = _module
basic_config = _module
processed = _module
data_loader = _module
data_transformer = _module
distance = _module
evaluate = _module
word_analogy = _module
glove_word = _module
word2vec = _module
bilstm = _module
crf = _module
embed_layer = _module
model_utils = _module
spatial_dropout = _module
understand_crf = _module
bilstm_crf = _module
checkpoints = _module
embedding = _module
feature = _module
figure = _module
log = _module
result = _module
augmentation = _module
eda = _module
preprocessor = _module
predict_utils = _module
predicter = _module
losses = _module
metrics = _module
train_utils = _module
trainer = _module
logginger = _module
utils = _module
test_predict = _module
train_bilstm_crf = _module
train_word2vec = _module

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


import warnings


from torch.optim.optimizer import Optimizer


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import torch.autograd as autograd


from itertools import repeat


from torch.autograd import Variable


from sklearn.metrics import f1_score


from sklearn.metrics import classification_report


from collections import Counter


import time


import random


from torch import optim


def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
    """
    :param use_cuda:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices


class BILSTM(nn.Module):

    def __init__(self, hidden_size, num_layer, input_size, dropout_p, num_classes, bi_tag):
        super(BILSTM, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True, dropout=dropout_p, bidirectional=bi_tag)
        bi_num = 2 if bi_tag else 1
        self.linear = nn.Linear(in_features=hidden_size * bi_num, out_features=num_classes)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, inputs, length):
        inputs, length, desorted_indice = prepare_pack_padded_sequence(inputs, length)
        embeddings_packed = pack_padded_sequence(inputs, length, batch_first=True)
        output, _ = self.lstm(embeddings_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[desorted_indice]
        output = F.dropout(output, p=self.dropout_p, training=self.training)
        output = F.tanh(output)
        logit = self.linear(output)
        return logit


class CRF(nn.Module):
    """线性条件随机场"""

    def __init__(self, num_classes, use_cuda=True):
        super(CRF, self).__init__()
        self.num_classes = num_classes
        self.start_tag = num_classes
        self.end_tag = num_classes + 1
        self.use_cuda = use_cuda
        self._reset()

    def _reset(self):
        self.transitions = torch.zeros(self.num_classes + 2, self.num_classes + 2)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.end_tag, :] = -10000.0
        self.transitions.data[:, self.start_tag] = -10000.0
        if self.use_cuda:
            self.transitions = self.transitions
        self.transitions = nn.Parameter(self.transitions)

    def real_path_score(self, features, tags):
        """
        features: (time_steps, num_tag)
        real_path_score表示真实路径分数
        它由Emission score和Transition score两部分相加组成
        Emission score由LSTM输出结合真实的tag决定，表示我们希望由输出得到真实的标签
        Transition score则是crf层需要进行训练的参数，它是随机初始化的，表示标签序列前后间的约束关系（转移概率）
        Transition矩阵存储的是标签序列相互间的约束关系
        在训练的过程中，希望real_path_score最高，因为这是所有路径中最可能的路径
        """
        r = torch.LongTensor(range(features.size(0)))
        pad_start_tags = torch.cat([torch.LongTensor([self.start_tag]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.end_tag])])
        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]) + torch.sum(features[r, tags])
        return score

    def all_possible_path_score(self, features):
        """
        计算所有可能的路径分数的log和：前向算法
        step1: 将forward列expand成3*3
        step2: 将下个单词的emission行expand成3*3
        step3: 将1和2和对应位置的转移矩阵相加
        step4: 更新forward，合并行
        step5: 取forward指数的对数计算total
        """
        time_steps = features.size(0)
        forward = Variable(torch.zeros(self.num_classes))
        for i in range(0, time_steps):
            emission_start = forward.expand(self.num_classes, self.num_classes).t()
            emission_end = features[i, :].expand(self.num_classes, self.num_classes)
            if i == 0:
                trans_score = self.transitions[self.start_tag, :self.start_tag]
            else:
                trans_score = self.transitions[:self.start_tag, :self.start_tag]
            score_sum = emission_start + emission_end + trans_score
            forward = self.log_sum(score_sum, dim=0)
        forward = forward + self.transitions[:self.start_tag, self.end_tag]
        total_score = self.log_sum(forward, dim=0)
        return total_score

    def negative_log_loss(self, inputs, length, tags):
        """
        features:(batch_size, time_step, num_tag)
        target_function = P_real_path_score/P_all_possible_path_score
                        = exp(S_real_path_score)/ sum(exp(certain_path_score))
        我们希望P_real_path_score的概率越高越好，即target_function的值越大越好
        因此，loss_function取其相反数，越小越好
        loss_function = -log(target_function)
                      = -S_real_path_score + log(exp(S_1 + exp(S_2) + exp(S_3) + ...))
                      = -S_real_path_score + log(all_possible_path_score)
        """
        inputs = inputs.cpu()
        length = length.cpu()
        tags = tags.cpu()
        loss = Variable(torch.tensor(0.0), requires_grad=True)
        num_chars = torch.sum(length.data).float()
        for ix, (features, tag) in enumerate(zip(inputs, tags)):
            features = features[:length[ix]]
            tag = tag[:length[ix]]
            real_score = self.real_path_score(features, tag)
            total_score = self.all_possible_path_score(features)
            cost = total_score - real_score
            loss = loss + cost
        return loss / num_chars

    def viterbi(self, features):
        time_steps = features.size(0)
        forward = Variable(torch.zeros(self.num_classes))
        back_points, index_points = [self.transitions[self.start_tag, :self.start_tag]], [torch.LongTensor([-1]).expand_as(forward)]
        for i in range(1, time_steps):
            emission_start = forward.expand(self.num_classes, self.num_classes).t()
            emission_end = features[i, :].expand(self.num_classes, self.num_classes)
            trans_score = self.transitions[:self.start_tag, :self.start_tag]
            score_sum = emission_start + emission_end + trans_score
            forward, index = torch.max(score_sum.detach(), dim=0)
            back_points.append(forward)
            index_points.append(index)
        back_points.append(forward + self.transitions[:self.start_tag, self.end_tag])
        return back_points, index_points

    def get_best_path(self, features):
        back_points, index_points = self.viterbi(features)
        best_last_point = self.argmax(back_points[-1])
        index_points = torch.stack(index_points)
        m = index_points.size(0)
        best_path = [best_last_point]
        for i in range(m - 1, 0, -1):
            best_index_point = index_points[i][best_last_point]
            best_path.append(best_index_point)
            best_last_point = best_index_point
        best_path.reverse()
        return best_path

    def get_batch_best_path(self, inputs, length):
        inputs = inputs.cpu()
        length = length.cpu()
        max_len = inputs.size(1)
        batch_best_path = []
        for ix, features in enumerate(inputs):
            features = features[:length[ix]]
            best_path = self.get_best_path(features)
            best_path = torch.Tensor(best_path).long()
            best_path = self.padding(best_path, max_len)
            batch_best_path.append(best_path)
        batch_best_path = torch.stack(batch_best_path, dim=0)
        return batch_best_path

    def compute_loss(self, output, labels, length):
        loss = self.negative_log_loss(inputs=output, length=length, tags=labels)
        return loss

    def predict(self, output, length):
        best_path = self.get_batch_best_path(inputs=output, length=length)
        return best_path

    def log_sum(self, matrix, dim):
        """
        前向算法是不断累积之前的结果，这样就会有个缺点
        指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
        为了避免这种情况，我们做了改动：
        1. 用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
        SUM = log(exp(s1)+exp(s2)+...+exp(s100))
            = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
            = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
        where clip=max
        """
        clip_value = torch.max(matrix)
        clip_value = int(clip_value.data.tolist())
        log_sum_value = clip_value + torch.log(torch.sum(torch.exp(matrix - clip_value), dim=dim))
        return log_sum_value

    def argmax(self, matrix, dim=0):
        """(0.5, 0.4, 0.3)"""
        _, index = torch.max(matrix, dim=dim)
        return index

    def padding(self, vec, max_len, pad_token=-1):
        new_vec = torch.zeros(max_len).long()
        new_vec[:vec.size(0)] = vec
        new_vec[vec.size(0):] = pad_token
        return new_vec


class Embed_Layer(nn.Module):

    def __init__(self, embedding_weight=None, vocab_size=None, embedding_dim=None, training=False, dropout_emb=0.25):
        super(Embed_Layer, self).__init__()
        self.training = training
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_emb)
        if not self.training:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if embedding_weight is not None:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_weight))
        else:
            self.encoder.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.dropout(x)
        return x


class Spatial_Dropout(nn.Module):

    def __init__(self, drop_prob):
        super(Spatial_Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self, input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))


class Model(nn.Module):

    def __init__(self, model_config, embedding_dim, num_classes, vocab_size, embedding_weight, device):
        super(Model, self).__init__()
        self.embedding = Embed_Layer(vocab_size=vocab_size, embedding_weight=embedding_weight, embedding_dim=embedding_dim, dropout_emb=model_config['dropout_emb'], training=True)
        self.lstm = BILSTM(input_size=embedding_dim, hidden_size=model_config['hidden_size'], num_layer=model_config['num_layer'], bi_tag=model_config['bi_tag'], dropout_p=model_config['dropout_p'], num_classes=num_classes + 2)
        self.crf = CRF(device=device, tagset_size=num_classes)

    def forward(self, inputs, length):
        x = self.embedding(inputs)
        x = self.lstm(x, length)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Spatial_Dropout,
     lambda: ([], {'drop_prob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lonePatient_BiLSTM_CRF_NER_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

