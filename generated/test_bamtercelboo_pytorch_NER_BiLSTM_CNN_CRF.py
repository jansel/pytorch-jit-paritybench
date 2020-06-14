import sys
_module = sys.modules[__name__]
del sys
config = _module
Alphabet = _module
Batch_Iterator = _module
Batch_Iterator_torch = _module
Common = _module
Embed = _module
Embed_From_Pretrained = _module
Load_Pretrained_Embed = _module
Optim = _module
Pickle = _module
DataUtils = _module
eval = _module
eval_bio = _module
mainHelp = _module
pickle_test = _module
tagSchemeConverter = _module
utils = _module
DataLoader_NER = _module
Instance = _module
master = _module
main = _module
BiLSTM = _module
BiLSTM_CNN = _module
CRF = _module
Sequence_Label = _module
models = _module
initialize = _module
modelHelp = _module
test = _module
trainer = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import numpy as np


import torch


import torch.nn as nn


import torch.nn.init as init


from collections import OrderedDict


import random


import torch.optim


from torch.nn.utils.clip_grad import clip_grad_norm_


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn import init


from torch.autograd.variable import Variable


import torch.optim as optim


import torch.nn.utils as utils


def init_embedding(input_embedding, seed=666):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear(input_linear, seed=1337):
    """初始化全连接层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.
        weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


cpu_device = 'cpu'


def prepare_pack_padded_sequence(inputs_words, seq_lengths, device='cpu',
    descending=True):
    """
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long
        (), descending=descending)
    if device != cpu_device:
        sorted_seq_lengths, indices = sorted_seq_lengths.cuda(), indices.cuda()
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths.cpu().numpy(
        ), desorted_indices


class BiLSTM(nn.Module):
    """
        BiLSTM
    """

    def __init__(self, **kwargs):
        super(BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)
        self.bilstm = nn.LSTM(input_size=D, hidden_size=self.lstm_hiddens,
            num_layers=self.lstm_layers, bidirectional=True, batch_first=
            True, bias=True)
        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2,
            out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :param desorted_indices:
        :return:
        """
        word, sentence_length, desorted_indices = prepare_pack_padded_sequence(
            word, sentence_length, device=self.device)
        x = self.embed(word)
        x = self.dropout_embed(x)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first
            =True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[desorted_indices]
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit


def init_embed(input_embedding, seed=656):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(input_embedding)


def init_linear_weight_bias(input_linear, seed=1337):
    """
    :param input_linear:
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(input_linear.weight)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + 1))
    if input_linear.bias is not None:
        input_linear.bias.data.uniform_(-scope, scope)


class BiLSTM_CNN(nn.Module):
    """
        BiLSTM_CNN
    """

    def __init__(self, **kwargs):
        super(BiLSTM_CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        char_paddingId = self.char_paddingId
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        self.char_embedding = nn.Embedding(self.char_embed_num, self.
            char_dim, padding_idx=char_paddingId)
        init_embed(self.char_embedding.weight)
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)
        self.char_encoders = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.conv_filter_nums
                [i], kernel_size=(1, filter_size, self.char_dim))
            self.char_encoders.append(f)
        for conv in self.char_encoders:
            if self.device != cpu_device:
                conv
        lstm_input_dim = D + sum(self.conv_filter_nums)
        self.bilstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.
            lstm_hiddens, num_layers=self.lstm_layers, bidirectional=True,
            batch_first=True, bias=True)
        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2,
            out_features=C, bias=True)
        init_linear_weight_bias(self.linear)

    def _char_forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)
        input_embed = self.char_embedding(inputs)
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.
            char_dim)
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)
        return char_conv_outputs

    def forward(self, word, char, sentence_length):
        """
        :param char:
        :param word:
        :param sentence_length:
        :return:
        """
        char_conv = self._char_forward(char)
        char_conv = self.dropout(char_conv)
        word = self.embed(word)
        x = torch.cat((word, char_conv), -1)
        x = self.dropout_embed(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1,
        m_size)
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec -
        max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):
    """
        CRF
    """

    def __init__(self, **kwargs):
        """
        kwargs:
            target_size: int, target size
            device: str, device
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        device = self.device
        self.START_TAG, self.STOP_TAG = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.
            target_size + 2, device=device)
        init_transitions[:, (self.START_TAG)] = -10000.0
        init_transitions[(self.STOP_TAG), :] = -10000.0
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
        tag_size = feats.size(2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        """ be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1) """
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size
            ).expand(ins_num, tag_size, tag_size)
        """ need to consider start """
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        """ only need start from start_tag """
        partition = inivalues[:, (self.START_TAG), :].clone().view(batch_size,
            tag_size, 1)
        """
        add start score (from start to all tag, duplicate to batch_size)
        partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        iter over last scores
        """
        for idx, cur_values in seq_iter:
            """
            previous to_target is current from_target
            partition: previous results log(exp(from_target)), #(batch_size * from_target)
            cur_values: bat_size * from_target * to_target
            """
            cur_values = cur_values + partition.contiguous().view(batch_size,
                tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            mask_idx = mask[(idx), :].view(batch_size, 1).expand(batch_size,
                tag_size)
            """ effective updated partition part, only keep the partition value of mask value = 1 """
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            """ let mask_idx broadcastable, to disable warning """
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            """ replace the partition where the maskvalue=1, other partition value keeps the same """
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        """ 
        until the last state, add transition score for all partition (and do log_sum_exp) 
        then select the value in STOP_TAG 
        """
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, (self.STOP_TAG)]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        """ calculate sentence length for each sentence """
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        """ mask to (seq_len, batch_size) """
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        """ be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1) """
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size
            ).expand(ins_num, tag_size, tag_size)
        """ need to consider start """
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        """ only need start from start_tag """
        partition = inivalues[:, (self.START_TAG), :].clone().view(batch_size,
            tag_size)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            """
            previous to_target is current from_target
            partition: previous results log(exp(from_target)), #(batch_size * from_target)
            cur_values: batch_size * from_target * to_target
            """
            cur_values = cur_values + partition.contiguous().view(batch_size,
                tag_size, 1).expand(batch_size, tag_size, tag_size)
            """ forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG """
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            """
            cur_bp: (batch_size, tag_size) max source score position in current tag
            set padded label as 0, which will be filtered in post processing
            """
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(
                batch_size, tag_size), 0)
            back_points.append(cur_bp)
        """ add score to final STOP_TAG """
        partition_history = torch.cat(partition_history, 0).view(seq_len,
            batch_size, -1).transpose(1, 0).contiguous()
        """ get the last position for each setences, and select the last partitions using gather() """
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size,
            1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position
            ).view(batch_size, tag_size, 1)
        """ calculate the score from last partition to end state (and then select the STOP_TAG from it) """
        last_values = last_partition.expand(batch_size, tag_size, tag_size
            ) + self.transitions.view(1, tag_size, tag_size).expand(batch_size,
            tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size, device=self.device,
            requires_grad=True).long()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size
            )
        """ elect end ids in STOP_TAG """
        pointer = last_bp[:, (self.STOP_TAG)]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(
            batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        """move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values """
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        """ decode from the end, padded position ids are 0, which will be filtered if following evaluation """
        decode_idx = torch.empty(seq_len, batch_size, device=self.device,
            requires_grad=True).long()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous(
                ).view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        """
        :param feats:
        :param mask:
        :return:
        """
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
        tags = tags.view(batch_size, seq_len)
        """ convert tag value into a new format, recorded label bigram information to index """
        new_tags = torch.empty(batch_size, seq_len, device=self.device,
            requires_grad=True).long()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, (0)] = (tag_size - 2) * tag_size + tags[:, (0)]
            else:
                new_tags[:, (idx)] = tags[:, (idx - 1)] * tag_size + tags[:,
                    (idx)]
        """ transition for label to STOP_TAG """
        end_transition = self.transitions[:, (self.STOP_TAG)].contiguous(
            ).view(1, tag_size).expand(batch_size, tag_size)
        """ length for batch,  last word position = length - 1 """
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        """ index the label id of last word """
        end_ids = torch.gather(tags, 1, length_mask - 1)
        """ index the transition score for end_id to STOP_TAG """
        end_energy = torch.gather(end_transition, 1, end_ids)
        """ convert tag as (seq_len, batch_size, 1) """
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len,
            batch_size, 1)
        """ need convert tags id to search from 400 positions of scores """
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2,
            new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        """
        add all score together
        gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        """
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
        return forward_score - gold_score


class Sequence_Label(nn.Module):
    """
        Sequence_Label
    """

    def __init__(self, config):
        super(Sequence_Label, self).__init__()
        self.config = config
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.class_num
        self.paddingId = config.paddingId
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight
        self.use_char = config.use_char
        self.char_embed_num = config.char_embed_num
        self.char_paddingId = config.char_paddingId
        self.char_dim = config.char_dim
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        self.conv_filter_nums = self._conv_filter(config.conv_filter_nums)
        assert len(self.conv_filter_sizes) == len(self.conv_filter_nums)
        self.use_crf = config.use_crf
        self.device = config.device
        self.target_size = (self.label_num if self.use_crf is False else 
            self.label_num + 2)
        if self.use_char is True:
            self.encoder_model = BiLSTM_CNN(embed_num=self.embed_num,
                embed_dim=self.embed_dim, label_num=self.target_size,
                paddingId=self.paddingId, dropout_emb=self.dropout_emb,
                dropout=self.dropout, lstm_hiddens=self.lstm_hiddens,
                lstm_layers=self.lstm_layers, pretrained_embed=self.
                pretrained_embed, pretrained_weight=self.pretrained_weight,
                char_embed_num=self.char_embed_num, char_dim=self.char_dim,
                char_paddingId=self.char_paddingId, conv_filter_sizes=self.
                conv_filter_sizes, conv_filter_nums=self.conv_filter_nums,
                device=self.device)
        else:
            self.encoder_model = BiLSTM(embed_num=self.embed_num, embed_dim
                =self.embed_dim, label_num=self.target_size, paddingId=self
                .paddingId, dropout_emb=self.dropout_emb, dropout=self.
                dropout, lstm_hiddens=self.lstm_hiddens, lstm_layers=self.
                lstm_layers, pretrained_embed=self.pretrained_embed,
                pretrained_weight=self.pretrained_weight, device=self.device)
        if self.use_crf is True:
            args_crf = dict({'target_size': self.label_num, 'device': self.
                device})
            self.crf_layer = CRF(**args_crf)

    @staticmethod
    def _conv_filter(str_list):
        """
        :param str_list:
        :return:
        """
        int_list = []
        str_list = str_list.split(',')
        for str in str_list:
            int_list.append(int(str))
        return int_list

    def forward(self, word, char, sentence_length, train=False):
        """
        :param char:
        :param word:
        :param sentence_length:
        :param train:
        :return:
        """
        if self.use_char is True:
            encoder_output = self.encoder_model(word, char, sentence_length)
            return encoder_output
        else:
            encoder_output = self.encoder_model(word, sentence_length)
            return encoder_output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_bamtercelboo_pytorch_NER_BiLSTM_CNN_CRF(_paritybench_base):
    pass
