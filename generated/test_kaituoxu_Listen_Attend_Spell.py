import sys
_module = sys.modules[__name__]
del sys
src = _module
recognize = _module
train = _module
data = _module
models = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
solver = _module
solver = _module
utils = _module
filt = _module
json2trn = _module
mergejson = _module
scp2json = _module
text2token = _module
learn_pytorch = _module
learn_visdom = _module
test_attention = _module
test_data = _module
test_decoder = _module
test_encoder = _module
test_seq2seq = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import time


class DotProductAttention(nn.Module):
    """Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.

    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H

        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(attention_scores.view(-1,
            input_lengths), dim=1).view(batch_size, -1, input_lengths)
        attention_output = torch.bmm(attention_distribution, values)
        return attention_output, attention_distribution


IGNORE_ID = -1


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[(i), :xs[i].size(0)] = xs[i]
    return pad


class Decoder(nn.Module):
    """
    """

    def __init__(self, vocab_size, embedding_dim, sos_id, eos_id,
        hidden_size, num_layers, bidirectional_encoder=True):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder_hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim + self.
            encoder_hidden_size, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.hidden_size)]
        self.attention = DotProductAttention()
        self.mlp = nn.Sequential(nn.Linear(self.encoder_hidden_size + self.
            hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.
            hidden_size, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, padded_input, encoder_padded_outputs):
        """
        Args:
            padded_input: N x To
            # encoder_hidden: (num_layers * num_directions) x N x H
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        ys = [y[y != IGNORE_ID] for y in padded_input]
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)
        h_list = [self.zero_state(encoder_padded_outputs)]
        c_list = [self.zero_state(encoder_padded_outputs)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs, H=
            encoder_padded_outputs.size(2))
        y_all = []
        embedded = self.embedding(ys_in_pad)
        for t in range(output_length):
            rnn_input = torch.cat((embedded[:, (t), :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](rnn_input, (h_list[0],
                c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](h_list[l - 1], (h_list[l
                    ], c_list[l]))
            rnn_output = h_list[-1]
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            y_all.append(predicted_y_t)
        y_all = torch.stack(y_all, dim=1)
        y_all = y_all.view(batch_size * output_length, self.vocab_size)
        ce_loss = F.cross_entropy(y_all, ys_out_pad.view(-1), ignore_index=
            IGNORE_ID, reduction='elementwise_mean')
        return ce_loss

    def recognize_beam(self, encoder_outputs, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen = args.decode_max_len
        h_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        c_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
            c_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
        att_c = self.zero_state(encoder_outputs.unsqueeze(0), H=
            encoder_outputs.unsqueeze(0).size(2))
        y = self.sos_id
        vy = encoder_outputs.new_zeros(1).long()
        hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'h_prev':
            h_list, 'a_prev': att_c}
        hyps = [hyp]
        ended_hyps = []
        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]
                embedded = self.embedding(vy)
                rnn_input = torch.cat((embedded, hyp['a_prev']), dim=1)
                h_list[0], c_list[0] = self.rnn[0](rnn_input, (hyp['h_prev'
                    ][0], hyp['c_prev'][0]))
                for l in range(1, self.num_layers):
                    h_list[l], c_list[l] = self.rnn[l](h_list[l - 1], (hyp[
                        'h_prev'][l], hyp['c_prev'][l]))
                rnn_output = h_list[-1]
                att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                    encoder_outputs.unsqueeze(0))
                att_c = att_c.squeeze(dim=1)
                mlp_input = torch.cat((rnn_output, att_c), dim=1)
                predicted_y_t = self.mlp(mlp_input)
                local_scores = F.log_softmax(predicted_y_t, dim=1)
                local_best_scores, local_best_ids = torch.topk(local_scores,
                    beam, dim=1)
                for j in range(beam):
                    new_hyp = {}
                    new_hyp['h_prev'] = h_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_c[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[
                        0, j])
                    hyps_best_kept.append(new_hyp)
                hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x[
                    'score'], reverse=True)[:beam]
            hyps = hyps_best_kept
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'].append(self.eos_id)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)
            hyps = remained_hyps
            if len(hyps) > 0:
                None
            else:
                None
                break
            for hyp in hyps:
                None
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True
            )[:min(len(ended_hyps), nbest)]
        return nbest_hyps


class Encoder(nn.Module):
    """Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0,
        bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H 
        """
        total_length = padded_input.size(1)
        packed_input = pack_padded_sequence(padded_input, input_lengths,
            batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
            total_length=total_length)
        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        loss = self.decoder(padded_target, encoder_padded_outputs)
        return loss

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam

        Returns:
            nbest_hyps:
        """
        encoder_outputs, _ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
            char_list, args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['einput'], package['ehidden'], package[
            'elayer'], dropout=package['edropout'], bidirectional=package[
            'ebidirectional'], rnn_type=package['etype'])
        decoder = Decoder(package['dvocab_size'], package['dembed'],
            package['dsos_id'], package['deos_id'], package['dhidden'],
            package['dlayer'], bidirectional_encoder=package['ebidirectional'])
        encoder.flatten_parameters()
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {'einput': model.encoder.input_size, 'ehidden': model.
            encoder.hidden_size, 'elayer': model.encoder.num_layers,
            'edropout': model.encoder.dropout, 'ebidirectional': model.
            encoder.bidirectional, 'etype': model.encoder.rnn_type,
            'dvocab_size': model.decoder.vocab_size, 'dembed': model.
            decoder.embedding_dim, 'dsos_id': model.decoder.sos_id,
            'deos_id': model.decoder.eos_id, 'dhidden': model.decoder.
            hidden_size, 'dlayer': model.decoder.num_layers, 'state_dict':
            model.state_dict(), 'optim_dict': optimizer.state_dict(),
            'epoch': epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kaituoxu_Listen_Attend_Spell(_paritybench_base):
    pass
    def test_000(self):
        self._check(DotProductAttention(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(Encoder(*[], **{'input_size': 4, 'hidden_size': 4, 'num_layers': 1}), [torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

