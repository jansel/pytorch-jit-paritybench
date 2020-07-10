import sys
_module = sys.modules[__name__]
del sys
master = _module
config = _module
main = _module
model = _module
bidaf = _module
mlp = _module
rnn = _module
tvqa_abc = _module
preprocessing = _module
test = _module
tvqa_dataset = _module
utils = _module

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
xrange = range
wraps = functools.wraps


import time


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch import nn


import numpy as np


from torch.utils.data.dataset import Dataset


class BidafAttn(nn.Module):
    """from the BiDAF paper https://arxiv.org/abs/1611.01603.
    Implemented by @easonnie and @jayleicn
    """

    def __init__(self, channel_size, method='original', get_h=False):
        super(BidafAttn, self).__init__()
        """
        This method do biDaf from s2 to s1:
            The return value will have the same size as s1.
        :param channel_size: Hidden size of the input
        """
        self.method = method
        self.get_h = get_h
        if method == 'original':
            self.mlp = nn.Linear(channel_size * 3, 1, bias=False)

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        if self.method == 'original':
            t1 = s1.size(1)
            t2 = s2.size(1)
            repeat_s1 = s1.unsqueeze(2).repeat(1, 1, t2, 1)
            repeat_s2 = s2.unsqueeze(1).repeat(1, t1, 1, 1)
            packed_s1_s2 = torch.cat([repeat_s1, repeat_s2, repeat_s1 * repeat_s2], dim=3)
            s = self.mlp(packed_s1_s2).squeeze()
        elif self.method == 'dot':
            s = torch.bmm(s1, s2.transpose(1, 2))
        s_mask = s.data.new(*s.size()).fill_(1).byte()
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data.byte(), -float('inf'))
        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = F.softmax(s, dim=2)
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
        u_tile = torch.bmm(a_weight, s2)
        return u_tile

    @classmethod
    def get_h_tile(cls, s, s1):
        """
        attended vectors of s1
        which words in s1 is most similar to each words in s2
        """
        t1 = s1.size(1)
        b_weight = F.softmax(torch.max(s, dim=2)[0], dim=-1).unsqueeze(1)
        h_tile = torch.bmm(b_weight, s1).repeat(1, t1, 1)
        return h_tile

    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile = self.get_u_tile(s, s2)
        h_tile = self.get_h_tile(s, s1) if self.get_h else None
        return u_tile, h_tile


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hsz, n_layers):
        super(MLP, self).__init__()
        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([nn.Linear(prev_dim, hsz), nn.ReLU(True), nn.Dropout(0.5)])
                prev_dim = hsz
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RNNEncoder(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """

    def __init__(self, word_embedding_size, hidden_size, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm', return_hidden=True, return_outputs=True):
        super(RNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed_inputs)
        if self.return_outputs:
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
        else:
            outputs = None
        if self.return_hidden:
            if self.rnn_type.lower() == 'lstm':
                hidden = hidden[0]
            hidden = hidden[-self.n_dirs:, :, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)
            hidden = hidden[reverse_indices]
        else:
            hidden = None
        return outputs, hidden


def max_along_time(outputs, lengths):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :return: (B, D)
    """
    outputs = [outputs[(i), :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
    return torch.stack(outputs, dim=0)


class ABC(nn.Module):

    def __init__(self, opt):
        super(ABC, self).__init__()
        self.vid_flag = 'imagenet' in opt.input_streams
        self.sub_flag = 'sub' in opt.input_streams
        self.vcpt_flag = 'vcpt' in opt.input_streams
        hidden_size_1 = opt.hsz1
        hidden_size_2 = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size_1 * 3, method='dot')
        self.lstm_raw = RNNEncoder(300, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
        if self.vid_flag:
            None
            self.video_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(vid_feat_size, embedding_size), nn.Tanh())
            self.lstm_mature_vid = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
            self.classifier_vid = MLP(hidden_size_2 * 2, 1, 500, n_layers_cls)
        if self.sub_flag:
            None
            self.lstm_mature_sub = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
            self.classifier_sub = MLP(hidden_size_2 * 2, 1, 500, n_layers_cls)
        if self.vcpt_flag:
            None
            self.lstm_mature_vcpt = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
            self.classifier_vcpt = MLP(hidden_size_2 * 2, 1, 500, n_layers_cls)

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l):
        e_q = self.embedding(q)
        e_a0 = self.embedding(a0)
        e_a1 = self.embedding(a1)
        e_a2 = self.embedding(a2)
        e_a3 = self.embedding(a3)
        e_a4 = self.embedding(a4)
        raw_out_q, _ = self.lstm_raw(e_q, q_l)
        raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
        raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
        raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
        raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
        raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)
        if self.sub_flag:
            e_sub = self.embedding(sub)
            raw_out_sub, _ = self.lstm_raw(e_sub, sub_l)
            sub_out = self.stream_processor(self.lstm_mature_sub, self.classifier_sub, raw_out_sub, sub_l, raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l, raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            sub_out = 0
        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt, vcpt_l, raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l, raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vcpt_out = 0
        if self.vid_flag:
            e_vid = self.video_fc(vid)
            raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, raw_out_vid, vid_l, raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l, raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vid_out = 0
        out = sub_out + vcpt_out + vid_out
        return out.squeeze()

    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l, q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)
        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)
        mature_maxout_a0, _ = lstm_mature(concat_a0, ctx_l)
        mature_maxout_a1, _ = lstm_mature(concat_a1, ctx_l)
        mature_maxout_a2, _ = lstm_mature(concat_a2, ctx_l)
        mature_maxout_a3, _ = lstm_mature(concat_a3, ctx_l)
        mature_maxout_a4, _ = lstm_mature(concat_a4, ctx_l)
        mature_maxout_a0 = max_along_time(mature_maxout_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(mature_maxout_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(mature_maxout_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(mature_maxout_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(mature_maxout_a4, ctx_l).unsqueeze(1)
        mature_answers = torch.cat([mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4], dim=1)
        out = classifier(mature_answers)
        return out

    @staticmethod
    def get_fake_inputs(device='cuda:0'):
        bsz = 16
        q = torch.ones(bsz, 25).long()
        q_l = torch.ones(bsz).fill_(25).long()
        a = torch.ones(bsz, 5, 20).long()
        a_l = torch.ones(bsz, 5).fill_(20).long()
        a0, a1, a2, a3, a4 = [a[:, (i), :] for i in range(5)]
        a0_l, a1_l, a2_l, a3_l, a4_l = [a_l[:, (i)] for i in range(5)]
        sub = torch.ones(bsz, 300).long()
        sub_l = torch.ones(bsz).fill_(300).long()
        vcpt = torch.ones(bsz, 300).long()
        vcpt_l = torch.ones(bsz).fill_(300).long()
        vid = torch.ones(bsz, 100, 2048)
        vid_l = torch.ones(bsz).fill_(100).long()
        return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'hsz': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jayleicn_TVQA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

