import sys
_module = sys.modules[__name__]
del sys
data = _module
hyperparams = _module
module = _module
network = _module
synthesis = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import numpy as np


import random


from torch import optim


import time


class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """

    def __init__(self, input_size, output_size, time_dim=2):
        """
        :param input_size: dimension of input
        :param output_size: dimension of output
        :param time_dim: index of time dimension
        """
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        """

        :param input_: sequences
        :return: outputs
        """
        batch_size = input_.size()[0]
        if self.time_dim == 2:
            input_ = input_.transpose(1, 2).contiguous()
        input_ = input_.view(-1, self.input_size)
        out = self.linear(input_).view(batch_size, -1, self.output_size)
        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)
        return out


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size):
        """

        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([('fc1', SeqLinear(self.
            input_size, self.hidden_size)), ('relu1', nn.ReLU()), (
            'dropout1', nn.Dropout(0.5)), ('fc2', SeqLinear(self.
            hidden_size, self.output_size)), ('relu2', nn.ReLU()), (
            'dropout2', nn.Dropout(0.5))]))

    def forward(self, input_):
        out = self.layer(input_)
        return out


use_cuda = torch.cuda.is_available()


class CBHG(nn.Module):
    """
    CBHG Module
    """

    def __init__(self, hidden_size, K=16, projection_size=128,
        num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
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
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
            out_channels=hidden_size, kernel_size=1, padding=int(np.floor(1 /
            2))))
        for i in range(2, K + 1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                out_channels=hidden_size, kernel_size=i, padding=int(np.
                floor(i / 2))))
        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))
        convbank_outdim = hidden_size * K
        if is_post:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                out_channels=hidden_size * 2, kernel_size=3, padding=int(np
                .floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                out_channels=projection_size, kernel_size=3, padding=int(np
                .floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)
        else:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                out_channels=hidden_size, kernel_size=3, padding=int(np.
                floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                out_channels=projection_size, kernel_size=3, padding=int(np
                .floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size,
            num_layers=2, batch_first=True, bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        input_ = input_.contiguous()
        batch_size = input_.size()[0]
        convbank_list = list()
        convbank_input = input_
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.
            batchnorm_list)):
            convbank_input = F.relu(batchnorm(self._conv_fit_dim(conv(
                convbank_input), k + 1).contiguous()))
            convbank_list.append(convbank_input)
        conv_cat = torch.cat(convbank_list, dim=1)
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]
        conv_projection = F.relu(self.batchnorm_proj_1(self._conv_fit_dim(
            self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.
            conv_projection_2(conv_projection))) + input_
        highway = self.highway.forward(conv_projection)
        highway = torch.transpose(highway, 1, 2)
        if use_cuda:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers,
                batch_size, self.hidden_size))
        else:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers,
                batch_size, self.hidden_size))
        self.gru.flatten_parameters()
        out, _ = self.gru(highway, init_gru)
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
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):
        out = input_
        for fc1, fc2 in zip(self.linears, self.gates):
            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))
            c = 1.0 - t
            out = h * t + out * c
        return out


class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism (Vinyals et al.)
    """

    def __init__(self, num_units):
        """

        :param num_units: dimension of hidden units
        """
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units
        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias=False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)
        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)
        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, hp.num_mels * hp.outputs_per_step)

    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden,
        gru2_hidden):
        memory_len = memory.size()[1]
        batch_size = memory.size()[0]
        keys = self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)
        d_t = self.attn_grucell(decoder_input, attn_hidden)
        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)
        attn_weights = self.v(F.tanh(keys + d_t_duplicate).view(-1, self.
            num_units)).view(-1, memory_len, 1)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)
        d_t_prime = torch.bmm(attn_weights.view([batch_size, 1, -1]), memory
            ).squeeze(1)
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1))
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        gru2_input = gru1_input + gru1_hidden
        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input + gru2_hidden
        output = self.out(bf_out).view(-1, hp.num_mels, hp.outputs_per_step)
        return output, d_t, gru1_hidden, gru2_hidden

    def inithidden(self, batch_size):
        if use_cuda:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
        else:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units),
                requires_grad=False)
        return attn_hidden, gru1_hidden, gru2_hidden


_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "


_eos = '~'


_pad = '_'


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, embedding_size):
        """

        :param embedding_size: dimension of embedding
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size)
        self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size
            )
        self.cbhg = CBHG(hp.hidden_size)

    def forward(self, input_):
        input_ = torch.transpose(self.embed(input_), 1, 2)
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)
        return memory


class MelDecoder(nn.Module):
    """
    Decoder
    """

    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(
            decoder_input.size()[0])
        outputs = list()
        if self.training:
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step
            prev_output = dec_input[:, :, (0)]
            for i in range(timesteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = (self.
                    attn_decoder.forward(prev_output, memory, attn_hidden=
                    attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=
                    gru2_hidden))
                outputs.append(prev_output)
                if random.random() < hp.teacher_forcing_ratio:
                    prev_output = dec_input[:, :, (i * hp.outputs_per_step)]
                else:
                    prev_output = prev_output[:, :, (-1)]
            outputs = torch.cat(outputs, 2)
        else:
            prev_output = decoder_input
            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:, :, (0)]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = (self.
                    attn_decoder.forward(prev_output, memory, attn_hidden=
                    attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=
                    gru2_hidden))
                outputs.append(prev_output)
                prev_output = prev_output[:, :, (-1)].unsqueeze(2)
            outputs = torch.cat(outputs, 2)
        return outputs


class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """

    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size, K=8, projection_size=hp.
            num_mels, is_post=True)
        self.linear = SeqLinear(hp.hidden_size * 2, hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out, 1, 2))
        return out


class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        memory = self.encoder.forward(characters)
        mel_output = self.decoder1.forward(mel_input, memory)
        linear_output = self.decoder2.forward(mel_output)
        return mel_output, linear_output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_soobinseo_Tacotron_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(CBHG(*[], **{'hidden_size': 4}), [torch.rand([4, 128, 64])], {})

    def test_001(self):
        self._check(Highwaynet(*[], **{'num_units': 4}), [torch.rand([4, 4, 4])], {})

    def test_002(self):
        self._check(Prenet(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(SeqLinear(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

