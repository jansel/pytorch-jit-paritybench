import sys
_module = sys.modules[__name__]
del sys
main = _module
sltk = _module
data = _module
dataset = _module
infer = _module
inference = _module
metrics = _module
nn = _module
functional = _module
initialize = _module
modules = _module
crf = _module
feature = _module
rnn = _module
sequence_labeling_model = _module
preprocessing = _module
normalize = _module
train = _module
sequence_labeling_trainer = _module
utils = _module
conllu = _module
embedding = _module
general = _module
io = _module
test_dataset = _module
test_yml = _module
bio2bieo = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from string import ascii_letters


from string import digits


from collections import Counter


import numpy as np


import torch


import torch.optim as optim


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            target_size: int, target size
            use_cuda: bool, 是否使用gpu, default is True
            average_batch: bool, loss是否作平均, default is True
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'average_batch'):
            self.__setattr__('average_batch', True)
        if not hasattr(self, 'use_cuda'):
            self.__setattr__('use_cuda', True)
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2)
        init_transitions[:, (self.START_TAG_IDX)] = -1000.0
        init_transitions[(self.END_TAG_IDX), :] = -1000.0
        if self.use_cuda:
            init_transitions = init_transitions
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, (self.START_TAG_IDX), :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            mask_idx = mask[(idx), :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, (self.END_TAG_IDX)]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, (self.START_TAG_IDX), :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
        if self.use_cuda:
            pad_zero = pad_zero
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        pointer = last_bp[:, (self.END_TAG_IDX)]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            decode_idx = decode_idx
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)
        new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        if self.use_cuda:
            new_tags = new_tags
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, (0)] = (tag_size - 2) * tag_size + tags[:, (0)]
            else:
                new_tags[:, (idx)] = tags[:, (idx - 1)] * tag_size + tags[:, (idx)]
        end_transition = self.transitions[:, (self.END_TAG_IDX)].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score


def init_embedding(input_embedding, seed=1337):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


class CharFeature(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            feature_size: int, 字符表的大小
            feature_dim: int, 字符embedding 维度
            require_grad: bool，char的embedding表是否需要更新

            filter_sizes: list(int), 卷积核尺寸
            filter_nums: list(int), 卷积核数量
        """
        super(CharFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.char_embedding = nn.Embedding(self.feature_size, self.feature_dim)
        init_embedding(self.char_embedding.weight)
        self.char_encoders = nn.ModuleList()
        for i, filter_size in enumerate(self.filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.filter_nums[i], kernel_size=(1, filter_size, self.feature_dim))
            self.char_encoders.append(f)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)
        input_embed = self.char_embedding(inputs)
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.feature_dim)
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0])
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()
        return char_conv_outputs


class WordFeature(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
             feature_names: list(str), 特征名称
             feature_size_dict: dict({str: int}), 特征名称到特征alphabet大小映射字典
             feature_dim_dict: dict({str: int}), 特征名称到特征维度映射字典

             require_grad_dict: dict({str: bool})，特征embedding表是否需要更新，特征名: bool
             pretrained_embed_dict: dict({str: np.array}): 特征的预训练embedding table
        """
        super(WordFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'require_grad_dict'):
            self.require_grad_dict = dict()
            for feature_name in self.feature_names:
                self.require_grad_dict[feature_name] = True
        for feature_name in self.feature_names:
            if feature_name not in self.require_grad_dict:
                self.require_grad_dict[feature_name] = True
        if not hasattr(self, 'pretrained_embed_dict'):
            self.pretrained_embed_dict = dict()
            for feature_name in self.feature_names:
                self.pretrained_embed_dict[feature_name] = None
        for feature_name in self.feature_names:
            if feature_name not in self.pretrained_embed_dict:
                self.pretrained_embed_dict[feature_name] = None
        self.feature_embedding_list = nn.ModuleList()
        for feature_name in self.feature_names:
            embed = nn.Embedding(self.feature_size_dict[feature_name], self.feature_dim_dict[feature_name])
            if self.pretrained_embed_dict[feature_name] is not None:
                embed.weight.data.copy_(torch.from_numpy(self.pretrained_embed_dict[feature_name]))
            else:
                init_embedding(embed.weight)
            embed.weight.requires_grad = self.require_grad_dict[feature_name]
            self.feature_embedding_list.append(embed)

    def forward(self, **input_dict):
        """
        Args:
            input_dict: dict({str: LongTensor})

        Returns:
            embed_outputs: 3D tensor, [bs, max_len, input_size]
        """
        embed_outputs = []
        for i, feature_name in enumerate(self.feature_names):
            embed_outputs.append(self.feature_embedding_list[i](input_dict[feature_name]))
        embed_outputs = torch.cat(embed_outputs, dim=2)
        return embed_outputs


class RNN(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            rnn_unit_type: str, options: ['rnn', 'lstm', 'gru']
            input_dim: int, rnn输入维度
            num_rnn_units: int, rnn单元数
            num_layers: int, 层数
            bi_flag: bool, 是否双向, default is True
        """
        super(RNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'bi_flag'):
            self.__setattr__('bi_flag', True)
        if self.rnn_unit_type == 'rnn':
            self.rnn = nn.RNN(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'gru':
            self.rnn = nn.GRU(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)

    def forward(self, feats):
        """
        Args:
             feats: 3D tensor, shape=[bs, max_len, input_dim]

        Returns:
            rnn_outputs: 3D tensor, shape=[bs, max_len, self.rnn_unit_num]
        """
        rnn_outputs, _ = self.rnn(feats)
        return rnn_outputs


class SLModel(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            feature_names: list(str), 特征名称, 不包括`label`和`char`

            feature_size_dict: dict({str: int}), 特征表大小字典
            feature_dim_dict: dict({str: int}), 输入特征dim字典
            pretrained_embed_dict: dict({str: np.array})
            require_grad_dict: bool, 是否更新feature embedding的权重

            # char parameters
            use_char: bool, 是否使用字符特征, default is False
            filter_sizes: list(int), 卷积核尺寸, default is [3]
            filter_nums: list(int), 卷积核数量, default is [32]

            # rnn parameters
            rnn_unit_type: str, options: ['rnn', 'lstm', 'gru']
            num_rnn_units: int, rnn单元数
            num_layers: int, 层数
            bi_flag: bool, 是否双向, default is True

            use_crf: bool, 是否使用crf层

            dropout_rate: float, dropout rate

            average_batch: bool, 是否对batch的loss做平均
            use_cuda: bool
        """
        super(SLModel, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.word_feature_layer = WordFeature(feature_names=self.feature_names, feature_size_dict=self.feature_size_dict, feature_dim_dict=self.feature_dim_dict, require_grad_dict=self.require_grad_dict, pretrained_embed_dict=self.pretrained_embed_dict)
        rnn_input_dim = 0
        for name in self.feature_names:
            rnn_input_dim += self.feature_dim_dict[name]
        if self.use_char:
            self.char_feature_layer = CharFeature(feature_size=self.feature_size_dict['char'], feature_dim=self.feature_dim_dict['char'], require_grad=self.require_grad_dict['char'], filter_sizes=self.filter_sizes, filter_nums=self.filter_nums)
            rnn_input_dim += sum(self.filter_nums)
        self.dropout_feature = nn.Dropout(self.dropout_rate)
        self.rnn_layer = RNN(rnn_unit_type=self.rnn_unit_type, input_dim=rnn_input_dim, num_rnn_units=self.num_rnn_units, num_layers=self.num_layers, bi_flag=self.bi_flag)
        self.dropout_rnn = nn.Dropout(self.dropout_rate)
        self.target_size = self.feature_size_dict['label']
        args_crf = dict({'target_size': self.target_size, 'use_cuda': self.use_cuda})
        args_crf['average_batch'] = self.average_batch
        if self.use_crf:
            self.crf_layer = CRF(**args_crf)
        hidden_input_dim = self.num_rnn_units * 2 if self.bi_flag else self.num_rnn_units
        target_size = self.target_size + 2 if self.use_crf else self.target_size
        self.hidden2tag = nn.Linear(hidden_input_dim, target_size)
        if not self.use_crf:
            self.loss_function = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
        else:
            self.loss_function = self.crf_layer.neg_log_likelihood_loss

    def loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        if not self.use_crf:
            batch_size, max_len = feats.size(0), feats.size(1)
            lstm_feats = feats.view(batch_size * max_len, -1)
            tags = tags.view(-1)
            return self.loss_function(lstm_feats, tags)
        else:
            loss_value = self.loss_function(feats, mask, tags)
        if self.average_batch:
            batch_size = feats.size(0)
            loss_value /= float(batch_size)
        return loss_value

    def forward(self, **feed_dict):
        """
        Args:
             inputs: list
        """
        batch_size = feed_dict[self.feature_names[0]].size(0)
        max_len = feed_dict[self.feature_names[0]].size(1)
        word_feed_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            word_feed_dict[feature_name] = feed_dict[feature_name]
        word_feature = self.word_feature_layer(**word_feed_dict)
        if self.use_char:
            char_feature = self.char_feature_layer(feed_dict['char'])
            word_feature = torch.cat([word_feature, char_feature], 2)
        word_feature = self.dropout_feature(word_feature)
        word_feature = torch.transpose(word_feature, 1, 0)
        rnn_outputs = self.rnn_layer(word_feature)
        rnn_outputs = rnn_outputs.transpose(1, 0).contiguous()
        rnn_outputs = self.dropout_rnn(rnn_outputs.view(-1, rnn_outputs.size(-1)))
        rnn_feats = self.hidden2tag(rnn_outputs)
        return rnn_feats.view(batch_size, max_len, -1)

    def predict(self, rnn_outputs, actual_lens, mask=None):
        batch_size = rnn_outputs.size(0)
        tags_list = []
        if not self.use_crf:
            _, arg_max = torch.max(rnn_outputs, dim=2)
            for i in range(batch_size):
                tags_list.append(arg_max[i].cpu().data.numpy()[:actual_lens.data[i]])
        else:
            path_score, best_paths = self.crf_layer(rnn_outputs, mask)
            for i in range(batch_size):
                tags_list.append(best_paths[i].cpu().data.numpy()[:actual_lens.data[i]])
        return tags_list

