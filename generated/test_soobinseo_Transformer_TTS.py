import sys
_module = sys.modules[__name__]
del sys
hyperparams = _module
module = _module
network = _module
prepare_data = _module
preprocess = _module
synthesis = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
train_postnet = _module
train_transformer = _module
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


import torch.nn as nn


import torch as t


import torch.nn.functional as F


import math


import numpy as np


import copy


from collections import OrderedDict


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import collections


from scipy import signal


from scipy.io.wavfile import write


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "


_eos = '~'


_pad = '_'


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """

    def __init__(self, embedding_size, num_hidden):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)
        self.conv1 = Conv(in_channels=embedding_size, out_channels=num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.conv2 = Conv(in_channels=num_hidden, out_channels=num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.conv3 = Conv(in_channels=num_hidden, out_channels=num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = self.embed(input_)
        input_ = input_.transpose(1, 2)
        input_ = self.dropout1(t.relu(self.batch_norm1(self.conv1(input_))))
        input_ = self.dropout2(t.relu(self.batch_norm2(self.conv2(input_))))
        input_ = self.dropout3(t.relu(self.batch_norm3(self.conv3(input_))))
        input_ = input_.transpose(1, 2)
        input_ = self.projection(input_)
        return input_


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        x = input_.transpose(1, 2)
        x = self.w_2(t.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = x + input_
        x = self.layer_norm(x)
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """

    def __init__(self, num_hidden):
        """
        
        :param num_hidden: dimension of hidden 
        """
        super(PostConvNet, self).__init__()
        self.conv1 = Conv(in_channels=hp.num_mels * hp.outputs_per_step, out_channels=num_hidden, kernel_size=5, padding=4, w_init='tanh')
        self.conv_list = clones(Conv(in_channels=num_hidden, out_channels=num_hidden, kernel_size=5, padding=4, w_init='tanh'), 3)
        self.conv2 = Conv(in_channels=num_hidden, out_channels=hp.num_mels * hp.outputs_per_step, kernel_size=5, padding=4)
        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, input_, mask=None):
        input_ = self.dropout1(t.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(t.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        return input_


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)
        if query_mask is not None:
            attn = attn * query_mask
        result = t.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)
        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        result = t.cat([decoder_input, result], dim=-1)
        result = self.final_linear(result)
        result = result + decoder_input
        result = self.layer_norm_1(result)
        return result, attns


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([('fc1', Linear(self.input_size, self.hidden_size)), ('relu1', nn.ReLU()), ('dropout1', nn.Dropout(p)), ('fc2', Linear(self.hidden_size, self.output_size)), ('relu2', nn.ReLU()), ('dropout2', nn.Dropout(p))]))

    def forward(self, input_):
        out = self.layer(input_)
        return out


class Highwaynet(nn.Module):
    """
    Highway network
    """

    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))

    def forward(self, input_):
        out = input_
        for fc1, fc2 in zip(self.linears, self.gates):
            h = t.relu(fc1.forward(out))
            t_ = t.sigmoid(fc2.forward(out))
            c = 1.0 - t_
            out = h * t_ + out * c
        return out


class CBHG(nn.Module):
    """
    CBHG Module
    """

    def __init__(self, hidden_size, K=16, projection_size=256, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size, out_channels=hidden_size, kernel_size=1, padding=int(np.floor(1 / 2))))
        for i in range(2, K + 1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=i, padding=int(np.floor(i / 2))))
        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))
        convbank_outdim = hidden_size * K
        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim, out_channels=hidden_size, kernel_size=3, padding=int(np.floor(3 / 2)))
        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size, out_channels=projection_size, kernel_size=3, padding=int(np.floor(3 / 2)))
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers, batch_first=True, bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        input_ = input_.contiguous()
        batch_size = input_.size(0)
        total_length = input_.size(-1)
        convbank_list = list()
        convbank_input = input_
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = t.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k + 1).contiguous()))
            convbank_list.append(convbank_input)
        conv_cat = t.cat(convbank_list, dim=1)
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]
        conv_projection = t.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_
        highway = self.highway.forward(conv_projection.transpose(1, 2))
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)
        return out


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return t.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """
    Encoder Network
    """

    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(t.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0), freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)
        self.layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)

    def forward(self, x, pos):
        if self.training:
            c_mask = pos.ne(0).type(t.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None
        x = self.encoder_prenet(x)
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x
        x = self.pos_dropout(x)
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)
        return x, c_mask, attns


class MelDecoder(nn.Module):
    """
    Decoder Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden
        """
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0), freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.norm = Linear(num_hidden, num_hidden)
        self.selfattn_layers = clones(Attention(num_hidden), 3)
        self.dotattn_layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')
        self.postconvnet = PostConvNet(num_hidden)

    def forward(self, memory, decoder_input, c_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)
        if self.training:
            m_mask = pos.ne(0).type(t.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask = None, None
        decoder_input = self.decoder_prenet(decoder_input)
        decoder_input = self.norm(decoder_input)
        pos = self.pos_emb(pos)
        decoder_input = pos * self.alpha + decoder_input
        decoder_input = self.pos_dropout(decoder_input)
        attn_dot_list = list()
        attn_dec_list = list()
        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)
        mel_out = self.mel_linear(decoder_input)
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)
        stop_tokens = self.stop_linear(decoder_input)
        return mel_out, out, attn_dot_list, stop_tokens, attn_dec_list


class Model(nn.Module):
    """
    Transformer Network
    """

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = MelDecoder(hp.hidden_size)

    def forward(self, characters, mel_input, pos_text, pos_mel):
        memory, c_mask, attns_enc = self.encoder.forward(characters, pos=pos_text)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder.forward(memory, mel_input, c_mask, pos=pos_mel)
        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """

    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, hp.n_fft // 2 + 1)

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)
        return mag_pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'num_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (CBHG,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 256, 64])], {}),
     False),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (FFN,
     lambda: ([], {'num_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Highwaynet,
     lambda: ([], {'num_units': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'num_hidden_k': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Prenet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_soobinseo_Transformer_TTS(_paritybench_base):
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

