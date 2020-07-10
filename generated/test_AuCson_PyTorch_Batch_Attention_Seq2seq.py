import sys
_module = sys.modules[__name__]
del sys
attentionRNN = _module

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


import torch


from torch import nn


from torch.autograd import Variable


import torch.nn.functional as F


class EncoderRNN(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        """
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        """
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DynamicEncoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        embedded = embedded[sort_idx].transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        """
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask).unsqueeze(1)
            attn_energies = attn_energies.masked_fill(mask, -1e+18)
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class BahdanauAttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        """
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)
        word_embedded = self.dropout(word_embedded)
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)
        output = F.log_softmax(self.out(output))
        return output, hidden


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attn,
     lambda: ([], {'method': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (DynamicEncoder,
     lambda: ([], {'input_size': 4, 'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (EncoderRNN,
     lambda: ([], {'input_size': 4, 'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_AuCson_PyTorch_Batch_Attention_Seq2seq(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

