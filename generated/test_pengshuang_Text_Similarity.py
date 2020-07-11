import sys
_module = sys.modules[__name__]
del sys
ABCNN = _module
ABCNN2 = _module
BiMPM = _module
ESIM = _module
SiaGRU = _module
sequence_module = _module
attention = _module
decoder = _module
encoder = _module
evaluation = _module

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


import torch


from torch import nn


from torch.nn import functional as F


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


def attention_matrix(x1, x2, eps=1e-06):
    """compute attention matrix using match score

    1 / (1 + |x · y|)
    |·| is euclidean distance
    Parameters
    ----------
    x1, x2 : 4-D torch Tensor
        size (batch_size, 1, sentence_length, width)

    Returns
    -------
    output : 3-D torch Tensor
        match score result of size (batch_size, sentence_length(for x2), sentence_length(for x1))
    """
    eps = torch.tensor(eps)
    one = torch.tensor(1.0)
    euclidean = (torch.pow(x1 - x2.permute(0, 2, 1, 3), 2).sum(dim=3) + eps).sqrt()
    return (euclidean + one).reciprocal()


class Abcnn1Portion(nn.Module):
    """Part of Abcnn1
    """

    def __init__(self, in_dim, out_dim):
        super(Abcnn1Portion, self).__init__()
        self.batchNorm = nn.BatchNorm2d(2)
        self.attention_feature_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x1, x2):
        """
        1. compute attention matrix
            attention_m : size (batch_size, sentence_length, sentence_length)
        2. generate attention feature map(weight matrix are parameters of the model to be learned)
            x_attention : size (batch_size, 1, sentence_length, emb_dim)
        3. stack the representation feature map and attention feature map
            x : size (batch_size, 2, sentence_length, emb_dim)
        4. batch norm(not in paper)
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length, emb_dim)
        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 2, sentence_length, emb_dim)
        """
        attention_m = attention_matrix(x1, x2)
        x1_attention = self.attention_feature_layer(attention_m.permute(0, 2, 1))
        x1_attention = x1_attention.unsqueeze(1)
        x1 = torch.cat([x1, x1_attention], 1)
        x2_attention = self.attention_feature_layer(attention_m)
        x2_attention = x2_attention.unsqueeze(1)
        x2 = torch.cat([x2, x2_attention], 1)
        x1 = self.batchNorm(x1)
        x2 = self.batchNorm(x2)
        return x1, x2


class WpLayer(nn.Module):
    """column-wise averaging over windows of w consecutive columns
    Attributes
    ----------
    attention : bool
        compute layer with attention matrix
    """

    def __init__(self, sentence_length, filter_width, attention):
        super(WpLayer, self).__init__()
        self.attention = attention
        if attention:
            self.sentence_length = sentence_length
            self.filter_width = filter_width
        else:
            self.wp = nn.AvgPool2d((filter_width, 1), stride=1)

    def forward(self, x, attention_matrix=None):
        """
        if attention
            reweight the convolution output with attention matrix
        else
            average pooling
        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length + filter_width - 1, height)
        attention_matrix: 2-D torch Tensor
            attention matrix between (convolution output x1 and convolution output x2) of size (batch_size, sentence_length + filter_width - 1)

        Returns
        -------
        output : 4-D torch Tensor
            size (batch_size, 1, sentence_length, height)
        """
        if self.attention:
            pools = []
            attention_matrix = attention_matrix.unsqueeze(1).unsqueeze(3)
            for i in range(self.sentence_length):
                pools.append((x[:, :, i:i + self.filter_width, :] * attention_matrix[:, :, i:i + self.filter_width, :]).sum(dim=2, keepdim=True))
            return torch.cat(pools, dim=2)
        else:
            return self.wp(x)


class Abcnn2Portion(nn.Module):
    """Part of Abcnn2
    """

    def __init__(self, sentence_length, filter_width):
        super(Abcnn2Portion, self).__init__()
        self.wp = WpLayer(sentence_length, filter_width, True)

    def forward(self, x1, x2):
        """
        1. compute attention matrix
            attention_m : size (batch_size, sentence_length + filter_width - 1, sentence_length + filter_width - 1)
        2. sum all attention values for a unit to derive a single attention weight for that unit
            x_a_conv : size (batch_size, sentence_length + filter_width - 1)
        3. average pooling(w-ap)
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length + filter_width - 1, height)
        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 1, sentence_length, height)
        """
        attention_m = attention_matrix(x1, x2)
        x1_a_conv = attention_m.sum(dim=1)
        x2_a_conv = attention_m.sum(dim=2)
        x1 = self.wp(x1, x1_a_conv)
        x2 = self.wp(x2, x2_a_conv)
        return x1, x2


class ApLayer(nn.Module):
    """column-wise averaging over all columns
    """

    def __init__(self, width):
        super(ApLayer, self).__init__()
        self.ap = nn.AvgPool2d((1, width), stride=1)

    def forward(self, x):
        """
        1. average pooling
            x size (batch_size, 1, sentence_length, 1)
        2. representation vector for the sentence
            output size (batch_size, sentence_length)
        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length, width)

        Returns
        -------
        output : 2-D torch Tensor
            representation vector of size (batch_size, width)
        """
        return self.ap(x).squeeze(1).squeeze(2)


def convolution(in_channel, filter_width, filter_height, filter_channel, padding):
    """convolution layer
    """
    model = nn.Sequential(nn.Conv2d(in_channel, filter_channel, (filter_width, filter_height), stride=1, padding=(padding, 0)), nn.BatchNorm2d(filter_channel), nn.Tanh())
    return model


class InceptionModule(nn.Module):
    """
    inception module(not in paper)
    first layer width is filter_width(given)
    second layer width is filter_width + 4
    third layer width is sentence_length
    this helps model to be learned(when the number of layers > 8)
    """

    def __init__(self, in_channel, sentence_length, filter_width, filter_height, filter_channel):
        super(InceptionModule, self).__init__()
        self.conv_1 = convolution(in_channel, filter_width, filter_height, int(filter_channel / 3) + filter_channel - 3 * int(filter_channel / 3), filter_width - 1)
        self.conv_2 = convolution(in_channel, filter_width + 4, filter_height, int(filter_channel / 3), filter_width + 1)
        self.conv_3 = convolution(in_channel, sentence_length, filter_height, int(filter_channel / 3), int((sentence_length + filter_width - 2) / 2))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(x)
        out_3 = self.conv_3(x)
        output = torch.cat([out_1, out_2, out_3], dim=1)
        return output


class ConvLayer(nn.Module):
    """
    convolution layer for abcnn
    Attributes
    ----------
    inception : bool
        whether use inception module
    """

    def __init__(self, isAbcnn2, sentence_length, filter_width, filter_height, filter_channel, inception):
        super(ConvLayer, self).__init__()
        if inception:
            self.model = InceptionModule(1 if isAbcnn2 else 2, sentence_length, filter_width, filter_height, filter_channel)
        else:
            self.model = convolution(1 if isAbcnn2 else 2, filter_width, filter_height, filter_channel, filter_width - 1)

    def forward(self, x):
        """
        1. convlayer
            size (batch_size, filter_channel, height, 1)
        2. transpose
            size (batch_size, 1, height, filter_channel)
        Parameters
        ----------
        x : 4-D torch Tensor
            size (batch_size, 1, height, width)

        Returns
        -------
        output : 4-D torch Tensor
            size (batch_size, 1, height, filter_channel)
        """
        output = self.model(x)
        output = output.permute(0, 3, 2, 1)
        return output


def cosine_similarity(x1, x2):
    """compute cosine similarity between x1 and x2
    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)
    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    """
    return F.cosine_similarity(x1, x2).unsqueeze(1)


def manhattan_distance(x1, x2):
    """compute manhattan distance between x1 and x2 (not in paper)
    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)
    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    """
    return torch.div(torch.norm(x1 - x2, 1, 1, keepdim=True), x1.size()[1])


class Abcnn3(nn.Module):
    """
    ABCNN3
    1. ABCNN1
    2. wide convolution
    3. ABCNN2
    Attributes
    ----------
    layer_size : int
        the number of (abcnn1 + abcnn2)
    distance : function
        cosine similarity or manhattan
    abcnn1 : list of abcnn1
    abcnn2 : list of abcnn2
    conv : list of convolution layer
    ap : list of pooling layer
    fc : last linear layer(in paper use logistic regression)
    """

    def __init__(self, emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine', inception=True):
        super(Abcnn3, self).__init__()
        self.layer_size = layer_size
        if match == 'cosine':
            self.distance = cosine_similarity
        else:
            self.distance = manhattan_distance
        self.abcnn1 = nn.ModuleList()
        self.abcnn2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([ApLayer(emb_dim)])
        self.fc = nn.Linear(layer_size + 1, 1)
        for i in range(layer_size):
            self.abcnn1.append(Abcnn1Portion(sentence_length, emb_dim if i == 0 else filter_channel))
            self.abcnn2.append(Abcnn2Portion(sentence_length, filter_width))
            self.conv.append(ConvLayer(False, sentence_length, filter_width, emb_dim if i == 0 else filter_channel, filter_channel, inception))
            self.ap.append(ApLayer(filter_channel))

    def forward(self, x1, x2):
        """
        1. stack sentence vector similarity
        2. for layer_size
            abcnn1
            convolution
            stack sentence representation vector similarity
            abcnn2

        3. concatenate similarity list
            size (batch_size, layer_size + 1)
        4. Linear layer
            size (batch_size, 1)
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length, emb_dim)
        Returns
        -------
        output : 2-D torch Tensor
            size (batch_size, 1)
        """
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))
        for i in range(self.layer_size):
            x1, x2 = self.abcnn1[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1, x2 = self.abcnn2[i](x1, x2)
        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output


class Abcnn1(nn.Module):
    """
    ABCNN1
    1. ABCNN1
    2. wide convolution
    3. W-ap
    Attributes
    ----------
    layer_size : int
        the number of (abcnn1)
    distance : function
        cosine similarity or manhattan
    abcnn : list of abcnn1
    conv : list of convolution layer
    wp : list of w-ap pooling layer
    ap : list of pooling layer
    fc : last linear layer(in paper use logistic regression)
    """

    def __init__(self, emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine', inception=True):
        super(Abcnn1, self).__init__()
        self.layer_size = layer_size
        if match == 'cosine':
            self.distance = cosine_similarity
        else:
            self.distance = manhattan_distance
        self.abcnn = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([ApLayer(emb_dim)])
        self.wp = nn.ModuleList()
        self.fc = nn.Linear(layer_size + 1, 1)
        for i in range(layer_size):
            self.abcnn.append(Abcnn1Portion(sentence_length, emb_dim if i == 0 else filter_channel))
            self.conv.append(ConvLayer(False, sentence_length, filter_width, emb_dim if i == 0 else filter_channel, filter_channel, inception))
            self.ap.append(ApLayer(filter_channel))
            self.wp.append(WpLayer(sentence_length, filter_width, False))

    def forward(self, x1, x2):
        """
        1. stack sentence vector similarity
        2. for layer_size
            abcnn1
            convolution
            stack sentence vector similarity
            W-ap for next loop x1, x2

        3. concatenate similarity list
            size (batch_size, layer_size + 1)
        4. Linear layer
            size (batch_size, 1)
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length, emb_dim)
        Returns
        -------
        output : 2-D torch Tensor
            size (batch_size, 1)
        """
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))
        for i in range(self.layer_size):
            x1, x2 = self.abcnn[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1 = self.wp[i](x1)
            x2 = self.wp[i](x2)
        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output


class Abcnn2(nn.Module):
    """
    ABCNN2
    1. wide convolution
    2. ABCNN2
    Attributes
    ----------
    layer_size : int
        the number of (abcnn2)
    distance : function
        cosine similarity or manhattan
    abcnn : list of abcnn2
    conv : list of convolution layer
    ap : list of pooling layer
    fc : last linear layer(in paper use logistic regression)
    """

    def __init__(self, emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine', inception=True):
        super(Abcnn2, self).__init__()
        self.layer_size = layer_size
        if match == 'cosine':
            self.distance = cosine_similarity
        else:
            self.distance = manhattan_distance
        self.abcnn = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([ApLayer(sentence_length, emb_dim)])
        self.fc = nn.Linear(layer_size + 1, 1)
        for i in range(layer_size):
            self.abcnn.append(Abcnn2Portion(sentence_length, filter_width))
            self.conv.append(ConvLayer(True, sentence_length, filter_width, emb_dim if i == 0 else filter_channel, filter_channel, inception))
            self.ap.append(ApLayer(sentence_length + filter_width - 1, filter_channel))

    def forward(self, x1, x2):
        """
        1. stack sentence vector similarity
        2. for layer_size
            convolution
            stack sentence vector similarity
            abcnn2

        3. concatenate similarity list
            size (batch_size, layer_size + 1)
        4. Linear layer
            size (batch_size, 1)
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length, emb_dim)
        Returns
        -------
        output : 2-D torch Tensor
            size (batch_size, 1)
        """
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))
        for i in range(self.layer_size):
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1, x2 = self.abcnn[i](x1, x2)
        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output


def match_score(s1, s2, mask1, mask2):
    """
    s1, s2:  batch_size * seq_len  * dim
    """
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


class Wide_Conv(nn.Module):

    def __init__(self, seq_len, embeds_size):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn([seq_len, embeds_size]))
        nn.init.xavier_normal_(self.W)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        """
        sent1, sent2: batch_size * seq_len * dim
        """
        A = match_score(sent1, sent2, mask1, mask2)
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2


def attention_avg_pooling(sent1, sent2, mask1, mask2):
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


class ABCNN3(nn.Module):

    def __init__(self, args):
        super(ABCNN3, self).__init__()
        self.args = args
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.linear_size = args.linear_size
        self.num_layer = args.num_layer
        self.conv = nn.ModuleList([Wide_Conv(args.max_length, self.embeds_dim) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(nn.Linear(self.embeds_dim * (1 + self.num_layer) * 2, self.linear_size), nn.LayerNorm(self.linear_size), nn.ReLU(inplace=True), nn.Linear(self.linear_size, 2), nn.Softmax())

    def forward(self, *input):
        s1, s2 = input[0], input[1]
        mask1, mask2 = s1.eq(0), s2.eq(0)
        res = [[], []]
        s1, s2 = self.embeds(s1), self.embeds(s2)
        res[0].append(F.avg_pool1d(s1.transpose(1, 2), kernel_size=s1.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(s2.transpose(1, 2), kernel_size=s2.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(s1, s2, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            s1, s2 = o1 + s1, o2 + s2
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        sim = self.fc(x)
        return sim


class BiMPM(nn.Module):

    def __init__(self, args, data):
        super(BiMPM, self).__init__()
        self.args = args
        self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_size
        self.l = self.args.num_perspective
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        self.word_emb.weight.requires_grad = False
        self.char_LSTM = nn.LSTM(input_size=self.args.char_dim, hidden_size=self.args.char_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.context_LSTM = nn.LSTM(input_size=self.d, hidden_size=self.args.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(self.l, self.args.hidden_size)))
        self.aggregation_LSTM = nn.LSTM(input_size=self.l * 8, hidden_size=self.args.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.pred_fc1 = nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2)
        self.pred_fc2 = nn.Linear(self.args.hidden_size * 2, self.args.class_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.char_emb.weight, -0.005, 0.005)
        self.char_emb.weight.data[0].fill_(0)
        nn.init.uniform(self.word_emb.weight.data[0], -0.1, 0.1)
        nn.init.kaiming_normal(self.char_LSTM.weight_ih_l0)
        nn.init.constant(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.char_LSTM.weight_hh_l0)
        nn.init.constant(self.char_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0)
        nn.init.constant(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0)
        nn.init.constant(self.context_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_hh_l0_reverse, val=0)
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal(w)
        nn.init.kaiming_normal(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)
        nn.init.uniform(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc1.bias, val=0)
        nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.training)

    def forward(self, **kwargs):

        def mp_matching_func(v1, v2, w):
            """
            :param v1: (batch, seq_len, hidden_size)
            :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l)
            """
            seq_len = v1.size(1)
            """
            if len(v2.size()) == 2:
                v2 = torch.stack([v2] * seq_len, dim=1)
            m = []
            for i in range(self.l):
                # v1: (batch, seq_len, hidden_size)
                # v2: (batch, seq_len, hidden_size)
                # w: (1, 1, hidden_size)
                # -> (batch, seq_len)
                m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))
            # list of (batch, seq_len) -> (batch, seq_len, l)
            m = torch.stack(m, dim=2)
            """
            w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
            v1 = w * torch.stack([v1] * self.l, dim=3)
            if len(v2.size()) == 3:
                v2 = w * torch.stack([v2] * self.l, dim=3)
            else:
                v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)
            m = F.cosine_similarity(v1, v2, dim=2)
            return m

        def mp_matching_func_pairwise(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l, seq_len1, seq_len2)
            """
            """
            m = []
            for i in range(self.l):
                # (1, 1, hidden_size)
                w_i = w[i].view(1, 1, -1)
                # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
                v1, v2 = w_i * v1, w_i * v2
                # (batch, seq_len, hidden_size->1)
                v1_norm = v1.norm(p=2, dim=2, keepdim=True)
                v2_norm = v2.norm(p=2, dim=2, keepdim=True)
                # (batch, seq_len1, seq_len2)
                n = torch.matmul(v1, v2.permute(0, 2, 1))
                d = v1_norm * v2_norm.permute(0, 2, 1)
                m.append(div_with_small_value(n, d))
            # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
            m = torch.stack(m, dim=3)
            """
            w = w.unsqueeze(0).unsqueeze(2)
            v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True)
            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)
            m = div_with_small_value(n, d).permute(0, 2, 3, 1)
            return m

        def attention(v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm
            return div_with_small_value(a, d)

        def div_with_small_value(n, d, eps=1e-08):
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d
        p = self.word_emb(kwargs['p'])
        h = self.word_emb(kwargs['h'])
        if self.args.use_char_emb:
            seq_len_p = kwargs['char_p'].size(1)
            seq_len_h = kwargs['char_h'].size(1)
            char_p = kwargs['char_p'].view(-1, self.args.max_word_len)
            char_h = kwargs['char_h'].view(-1, self.args.max_word_len)
            _, (char_p, _) = self.char_LSTM(self.char_emb(char_p))
            _, (char_h, _) = self.char_LSTM(self.char_emb(char_h))
            char_p = char_p.view(-1, seq_len_p, self.args.char_hidden_size)
            char_h = char_h.view(-1, seq_len_h, self.args.char_hidden_size)
            p = torch.cat([p, char_p], dim=-1)
            h = torch.cat([h, char_h], dim=-1)
        p = self.dropout(p)
        h = self.dropout(h)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)
        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)
        con_p_fw, con_p_bw = torch.split(con_p, self.args.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.args.hidden_size, dim=-1)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, (-1), :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, (0), :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, (-1), :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, (0), :], self.mp_w2)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)
        mv_p = torch.cat([mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw, mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat([mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw, mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)
        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)
        x = torch.cat([agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2), agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)], dim=1)
        x = self.dropout(x)
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)
        return x


class ESIM(nn.Module):

    def __init__(self, args):
        super(ESIM, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.BatchNorm1d(self.hidden_size * 8), nn.Linear(self.hidden_size * 8, args.linear_size), nn.ELU(inplace=True), nn.BatchNorm1d(args.linear_size), nn.Dropout(self.dropout), nn.Linear(args.linear_size, args.linear_size), nn.ELU(inplace=True), nn.BatchNorm1d(args.linear_size), nn.Dropout(self.dropout), nn.Linear(args.linear_size, 2), nn.Softmax(dim=-1))

    def soft_attention_align(self, x1, x2, mask1, mask2):
        """
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        """
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity


class SiaGRU(nn.Module):

    def __init__(self, args):
        super(SiaGRU, self).__init__()
        self.args = args
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.ln_embeds = nn.LayerNorm(self.embeds_dim)
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden, cell = self.gru(x)
        return hidden.squeeze()

    def forward(self, *input):
        sent1 = input[0]
        sent2 = input[1]
        x1 = self.ln_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.ln_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        encoding1 = self.forward_once(x1)
        encoding2 = self.forward_once(x2)
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        return self.fc(sim)


class Attention(nn.Module):

    def __init__(self, hidden_size, method='general'):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs, encoder_lengths=None, return_weight=False):
        """
        hidden : query (previous hidden) B,1,D <FloatTensor>
        encoder_outputs : context (encoder outputs) B,T,D <FloatTensor>
        encoder_lengths : list[int]
        """
        q, c = hidden, encoder_outputs
        batch_size_q, n_q, dim_q = q.size()
        batch_size_c, n_c, dim_c = c.size()
        if not batch_size_q == batch_size_c:
            msg = 'batch size mismatch (query: {}, context: {}, value: {})'
            raise ValueError(msg.format(q.size(), c.size()))
        batch_size = batch_size_q
        s = self.score(q, c)
        if encoder_lengths is not None:
            mask = s.data.new(batch_size, n_q, n_c)
            mask = self.fill_context_mask(mask, sizes=encoder_lengths, v_mask=float('-inf'), v_unmask=0)
            s = Variable(mask) + s
        s_flat = s.view(batch_size * n_q, n_c)
        w_flat = F.softmax(s_flat, 1)
        w = w_flat.view(batch_size, n_q, n_c)
        z = w.bmm(c)
        if return_weight:
            return w, z
        return z

    def score(self, q, c):
        """
        q: B,1,D
        c: B,T,D
        """
        if self.method == 'dot':
            return q.bmm(c.transpose(1, 2))
        elif self.method == 'general':
            energy = self.attn(c)
            return q.bmm(energy.transpose(1, 2))
        elif self.method == 'concat':
            q = q.repeat(1, c.size(1), 1)
            energy = self.attn(torch.cat([q, c], 2))
            v = self.v.repeat(c.size(1), 1).unsqueeze(1)
            return v.bmm(energy.transpose(1, 2))

    def fill_context_mask(self, mask, sizes, v_mask, v_unmask):
        """Fill attention mask inplace for a variable length context.
        Args
        ----
        mask: Tensor of size (B, T, D)
            Tensor to fill with mask values. 
        sizes: list[int]
            List giving the size of the context for each item in
            the batch. Positions beyond each size will be masked.
        v_mask: float
            Value to use for masked positions.
        v_unmask: float
            Value to use for unmasked positions.
        Returns
        -------
        mask:
            Filled with values in {v_mask, v_unmask}
        """
        mask.fill_(v_unmask)
        n_context = mask.size(2)
        for i, size in enumerate(sizes):
            if size < n_context:
                mask[(i), :, size:] = v_mask
        return mask


class Beam:

    def __init__(self, root, num_beam):
        """
        root : (score, hidden)
        """
        self.num_beam = num_beam
        score = F.log_softmax(root[0], 1)
        s, i = score.topk(num_beam)
        s = s.data.tolist()[0]
        i = i.data.tolist()[0]
        i = [[ii] for ii in i]
        hiddens = [root[1] for _ in range(num_beam)]
        self.beams = list(zip(s, i, hiddens))
        self.beams = sorted(self.beams, key=lambda x: x[0], reverse=True)

    def select_k(self, siblings):
        """
        siblings : [score,hidden]
        """
        candits = []
        for p_index, sibling in enumerate(siblings):
            parents = self.beams[p_index]
            score = F.log_softmax(sibling[0], 1)
            s, i = score.topk(self.num_beam)
            scores = s.data.tolist()[0]
            indices = i.data.tolist()[0]
            candits.extend([(parents[0] + scores[i], parents[1] + [indices[i]], sibling[1]) for i in range(len(scores))])
        candits = sorted(candits, key=lambda x: x[0], reverse=True)
        self.beams = candits[:self.num_beam]
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1, -1), b[2]] for b in self.beams]

    def get_best_seq(self):
        return self.beams[0][1]

    def get_next_nodes(self):
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1, -1), b[2]] for b in self.beams]


class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.3, use_cuda=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attention = Attention(hidden_size)
        self.use_cuda = use_cuda

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        return hidden if self.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_normal(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_normal(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)

    def greedy_decode(self, inputs, context, max_length, encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        decode = []
        i = 0
        decoded = inputs
        while decoded.data.tolist()[0] != 3:
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score, 1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            context = self.attention(hidden, encoder_outputs, None)
            i += 1
        scores = torch.cat(decode)
        return scores.max(1)[1]

    def beam_search_decode(self, inputs, context, max_length, encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
        concated = torch.cat((hidden, context.transpose(0, 1)), 2)
        score = self.linear(concated.squeeze(0))
        beam = Beam([score, hidden], 3)
        nodes = beam.get_next_nodes()
        i = 1
        while i < max_length:
            siblings = []
            for node in nodes:
                embedded = self.embedding(node[0])
                hidden = node[1]
                context = self.attention(hidden, encoder_outputs, None)
                _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
                concated = torch.cat((hidden, context.transpose(0, 1)), 2)
                score = self.linear(concated.squeeze(0))
                siblings.append([score, hidden])
            i += 1
            nodes = beam.select_k(siblings)
        return beam.get_best_seq()

    def forward(self, inputs, context, max_length, encoder_outputs, encoder_lengths=None):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        encoder_outputs : B,T,D
        encoder_lengths : B,T # list
        max_length : int, max length to decode
        """
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        embedded = self.dropout(embedded)
        decode = []
        for i in range(max_length):
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score, 1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            embedded = self.dropout(embedded)
            context = self.attention(hidden.transpose(0, 1), encoder_outputs, encoder_lengths)
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)


class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, bidirec=False, dropout_p=0.5, use_cuda=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.use_cuda = use_cuda
        if bidirec:
            self.n_direction = 2
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size))
        return hidden if self.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_normal(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_normal(self.gru.weight_ih_l0)

    def forward(self, inputs, input_lengths):
        """
        inputs : B,T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Abcnn2Portion,
     lambda: ([], {'sentence_length': 4, 'filter_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ApLayer,
     lambda: ([], {'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Attention,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ConvLayer,
     lambda: ([], {'isAbcnn2': 4, 'sentence_length': 4, 'filter_width': 4, 'filter_height': 4, 'filter_channel': 4, 'inception': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Encoder,
     lambda: ([], {'input_size': 4, 'embedding_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (InceptionModule,
     lambda: ([], {'in_channel': 4, 'sentence_length': 4, 'filter_width': 4, 'filter_height': 4, 'filter_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Wide_Conv,
     lambda: ([], {'seq_len': 4, 'embeds_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_pengshuang_Text_Similarity(_paritybench_base):
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

