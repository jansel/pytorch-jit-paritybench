import sys
_module = sys.modules[__name__]
del sys
zsdg = _module
criterions = _module
dataset = _module
corpora = _module
data_loaders = _module
dataloader_bases = _module
enc2dec = _module
base_modules = _module
decoders = _module
encoders = _module
evaluators = _module
hred_utils = _module
main = _module
models = _module
model_bases = _module
models = _module
nn_lib = _module
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


import torch


import torch.nn.functional as F


from torch.nn.modules.loss import _Loss


import numpy as np


import logging


import torch.nn as nn


from torch.autograd import Variable


from collections import defaultdict


from torch.nn.modules.module import _addindent


from torch.autograd import Function


class L2Loss(_Loss):
    logger = logging.getLogger()

    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a - s_b, 2)
        else:
            losses = torch.pow(state_a - state_b, 2)
        return torch.mean(losses)


FLOAT = 2


INT = 0


LONG = 1


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    elif dtype == INT:
        var = var.type(torch.IntTensor)
    elif dtype == LONG:
        var = var.type(torch.LongTensor)
    elif dtype == FLOAT:
        var = var.type(torch.FloatTensor)
    else:
        raise ValueError('Unknown dtype')
    return var


class NLLEntropy(_Loss):
    logger = logging.getLogger()

    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = config.avg_type
        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info('Use extra cost for key words')
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT, config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(input, target, size_average=False, ignore_index=self.padding_idx, weight=self.weight)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(input, target, size_average=False, ignore_index=self.padding_idx, weight=self.weight)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(input, target, size_average=True, ignore_index=self.padding_idx, weight=self.weight, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(input, target, size_average=True, ignore_index=self.padding_idx, weight=self.weight)
        else:
            raise ValueError('Unknown avg type')
        return loss


EOS = '</s>'


PAD = '<pad>'


class BaseRNN(nn.Module):
    SYM_MASK = PAD
    SYM_EOS = EOS
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_LATENT = 'latent'
    KEY_RECOG_LATENT = 'recog_latent'
    KEY_POLICY = 'policy'
    KEY_G = 'g'
    KEY_PTR_SOFTMAX = 'ptr_softmax'
    KEY_PTR_CTX = 'ptr_context'

    def __init__(self, vocab_size, input_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, bidirectional):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: {0}'.format(rnn_cell))
        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

    def gumbel_max(self, log_probs):
        """
        Obtain a sample from the Gumbel max. Not this is not differentibale.
        :param log_probs: [batch_size x vocab_size]
        :return: [batch_size x 1] selected token IDs
        """
        sample = torch.Tensor(log_probs.size()).uniform_(0, 1)
        sample = cast_type(Variable(sample), FLOAT, self.use_gpu)
        matrix_u = -1.0 * torch.log(-1.0 * torch.log(sample))
        gumbel_log_probs = log_probs + matrix_u
        max_val, max_ids = torch.max(gumbel_log_probs, dim=-1, keepdim=True)
        return max_ids

    def repeat_state(self, state, batch_size, times):
        new_s = state.repeat(1, 1, times)
        return new_s.view(-1, batch_size * times, self.hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \\begin{array}{ll}
            x = context*output \\\\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\\\
            output = \\tanh(w * (attn * context) + b * output)
            \\end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dec_size, attn_size, mode, project=False):
        super(Attention, self).__init__()
        self.mask = None
        self.mode = mode
        self.attn_size = attn_size
        self.dec_size = dec_size
        if project:
            self.linear_out = nn.Linear(dec_size + attn_size, dec_size)
        else:
            self.linear_out = None
        if mode == 'general':
            self.attn_w = nn.Linear(dec_size, attn_size)
        elif mode == 'cat':
            self.dec_w = nn.Linear(dec_size, dec_size)
            self.attn_w = nn.Linear(attn_size, dec_size)
            self.query_w = nn.Linear(dec_size, 1)

    def forward(self, output, context):
        """
        :param output: [batch, out_len, dec_size]
        :param context: [batch, in_len, attn_size]
        :return: output, attn
        """
        batch_size = output.size(0)
        input_size = context.size(1)
        if self.mode == 'dot':
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == 'general':
            mapped_output = self.attn_w(output)
            attn = torch.bmm(mapped_output, context.transpose(1, 2))
        elif self.mode == 'cat':
            mapped_attn = self.attn_w(context)
            mapped_out = self.dec_w(output)
            tiled_out = mapped_out.unsqueeze(2).repeat(1, 1, input_size, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_attn + tiled_out)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError('Unknown attention')
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        if self.linear_out is None:
            return combined, attn
        else:
            output = F.tanh(self.linear_out(combined.view(-1, self.dec_size + self.attn_size))).view(batch_size, -1, self.dec_size)
            return output, attn


GEN = 'gen'


TEACH_FORCE = 'teacher_forcing'


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id, eos_id, n_layers=1, rnn_cell='lstm', input_dropout_p=0, dropout_p=0, use_attention=False, attn_mode='cat', attn_size=None, use_gpu=True, embedding=None, output_size=None, tie_output_embed=False):
        super(DecoderRNN, self).__init__(vocab_size, input_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, False)
        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.use_gpu = use_gpu
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)
        else:
            self.embedding = embedding
        if use_attention:
            self.attention = Attention(self.hidden_size, attn_size, attn_mode, project=True)
        if tie_output_embed:
            self.project = lambda x: x * self.embedding.weight.transpose(0, 1)
        else:
            self.project = nn.Linear(self.hidden_size, self.output_size)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)
        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))
        predicted_softmax = self.function(logits, dim=logits.dim() - 1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, batch_size, inputs=None, init_state=None, attn_context=None, mode=TEACH_FORCE, gen_type='greedy', beam_size=4):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        if mode == GEN:
            inputs = None
        if gen_type != 'beam':
            beam_size = 1
        if inputs is not None:
            decoder_input = inputs
        else:
            bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size * beam_size, 1)
        if mode == GEN and gen_type == 'beam':
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = self.repeat_state(h, batch_size, beam_size), self.repeat_state(c, batch_size, beam_size)
            else:
                decoder_hidden = self.repeat_state(init_state, batch_size, beam_size)
        else:
            decoder_hidden = init_state
        decoder_outputs = []
        sequence_symbols = []
        back_pointers = []
        lengths = np.array([self.max_length] * batch_size * beam_size)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == 'beam':
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)
                    seq_score = seq_score[:, 0:self.output_size]
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)
                top_v, top_id = seq_score.topk(beam_size)
                back_ptr = top_id.div(self.output_size).view(-1, 1)
                symbols = top_id.fmod(self.output_size).view(-1, 1)
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError('Unsupported decoding mode')
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = (lengths > di) & eos_batches != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, attn_context)
            decoder_outputs = decoder_output
        else:
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, attn_context)
                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                decoder_input = symbols
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            if gen_type == 'beam':
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]
                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)
                    selected_symbols = []
                    selected_parents = []
                    for b_id in range(batch_size):
                        selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_symbols.append(symbol2ds[b_id, max_seq_id[b_id]])
                    final_seq_symbols.append(torch.cat(selected_symbols).unsqueeze(1))
                    max_seq_id = torch.cat(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        return decoder_outputs, decoder_hidden, ret_dict


class DecoderPointerGen(BaseRNN):

    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id, eos_id, n_layers=1, rnn_cell='lstm', input_dropout_p=0, dropout_p=0, attn_mode='cat', attn_size=None, use_gpu=True, embedding=None):
        super(DecoderPointerGen, self).__init__(vocab_size, input_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, False)
        self.output_size = vocab_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.use_gpu = use_gpu
        self.attn_size = attn_size
        if embedding is None:
            self.embedding = nn.Embedding(self.output_size, self.input_size)
        else:
            self.embedding = embedding
        self.attention = Attention(self.hidden_size, attn_size, attn_mode, project=True)
        self.project = nn.Linear(self.hidden_size, self.output_size)
        self.sentinel = nn.Parameter(torch.randn((1, 1, attn_size)), requires_grad=True)
        self.register_parameter('sentinel', self.sentinel)

    def forward_step(self, input_var, hidden, attn_ctxs, attn_words, ctx_embed=None):
        """
        attn_size: number of context to attend
        :param input_var: 
        :param hidden: 
        :param attn_ctxs: batch_size x attn_size+1 x ctx_size. If None, then leave it empty
        :param attn_words: batch_size x attn_size
        :return: 
        """
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)
        embedded = self.embedding(input_var)
        if ctx_embed is not None:
            embedded += ctx_embed
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        if attn_ctxs is None:
            logits = self.project(output.contiguous().view(-1, self.hidden_size))
            predicted_softmax = F.log_softmax(logits, dim=1)
            return predicted_softmax, None, hidden, None, None
        else:
            attn_size = attn_words.size(1)
            combined_output, attn = self.attention(output, attn_ctxs)
            rnn_softmax = F.softmax(self.project(output.view(-1, self.hidden_size)), dim=1)
            g = attn[:, :, (0)].contiguous()
            ptr_attn = attn[:, :, 1:].contiguous()
            ptr_softmax = Variable(torch.zeros((batch_size * seq_len * attn_size, self.vocab_size)))
            ptr_softmax = cast_type(ptr_softmax, FLOAT, self.use_gpu)
            flat_attn_words = attn_words.unsqueeze(1).repeat(1, seq_len, 1).view(-1, 1)
            flat_attn = ptr_attn.view(-1, 1)
            ptr_softmax = ptr_softmax.scatter_(1, flat_attn_words, flat_attn)
            ptr_softmax = ptr_softmax.view(batch_size * seq_len, attn_size, self.vocab_size)
            ptr_softmax = torch.sum(ptr_softmax, dim=1)
            mixture_softmax = rnn_softmax * g.view(-1, 1) + ptr_softmax
            logits = torch.log(mixture_softmax.clamp(min=1e-08))
            predicted_softmax = logits.view(batch_size, seq_len, -1)
            ptr_softmax = ptr_softmax.view(batch_size, seq_len, -1)
            return predicted_softmax, ptr_softmax, hidden, ptr_attn, g

    def forward(self, batch_size, attn_context, attn_words, inputs=None, init_state=None, mode=TEACH_FORCE, gen_type='greedy', ctx_embed=None):
        ret_dict = dict()
        if mode == GEN:
            inputs = None
        if inputs is not None:
            decoder_input = inputs
        else:
            bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size, 1)
        if attn_context is not None:
            attn_context = torch.cat([self.sentinel.expand(batch_size, 1, self.attn_size), attn_context], dim=1)
        decoder_hidden = init_state
        decoder_outputs = []
        sequence_symbols = []
        attentions = []
        pointer_gs = []
        pointer_outputs = []
        lengths = np.array([self.max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)
            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            else:
                raise ValueError('Unsupported decoding mode')
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = (lengths > di) & eos_batches != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        if mode == TEACH_FORCE:
            pred_softmax, ptr_softmax, decoder_hidden, attn, step_g = self.forward_step(decoder_input, decoder_hidden, attn_context, attn_words, ctx_embed)
            attentions = attn
            decoder_outputs = pred_softmax
            pointer_gs = step_g
            pointer_outputs = ptr_softmax
        else:
            for di in range(self.max_length):
                pred_softmax, ptr_softmax, decoder_hidden, step_attn, step_g = self.forward_step(decoder_input, decoder_hidden, attn_context, attn_words, ctx_embed)
                symbols = decode(di, pred_softmax)
                attentions.append(step_attn)
                pointer_gs.append(step_g)
                pointer_outputs.append(ptr_softmax)
                decoder_input = symbols
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            pointer_outputs = torch.cat(pointer_outputs, dim=1)
            pointer_gs = torch.cat(pointer_gs, dim=1)
        ret_dict[self.KEY_ATTN_SCORE] = attentions
        ret_dict[self.KEY_SEQUENCE] = sequence_symbols
        ret_dict[self.KEY_LENGTH] = lengths
        ret_dict[self.KEY_G] = pointer_gs
        ret_dict[self.KEY_PTR_SOFTMAX] = pointer_outputs
        ret_dict[self.KEY_PTR_CTX] = attn_words
        return decoder_outputs, decoder_hidden, ret_dict


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, input_size, hidden_size, input_dropout_p=0, dropout_p=0, n_layers=1, rnn_cell='gru', variable_lengths=False, bidirection=False):
        super(EncoderRNN, self).__init__(-1, input_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, bidirection)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size * 2 if bidirection else hidden_size

    def forward(self, input_var, input_lengths=None, init_state=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len, embedding size): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class RnnUttEncoder(nn.Module):

    def __init__(self, utt_cell_size, dropout, rnn_cell='gru', bidirection=True, use_attn=False, embedding=None, vocab_size=None, embed_dim=None, feat_size=0):
        super(RnnUttEncoder, self).__init__()
        self.bidirection = bidirection
        self.utt_cell_size = utt_cell_size
        if embedding is None:
            self.embed_size = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = embedding
            self.embed_size = embedding.embedding_dim
        self.rnn = EncoderRNN(self.embed_size + feat_size, utt_cell_size, 0.0, dropout, rnn_cell=rnn_cell, variable_lengths=False, bidirection=bidirection)
        self.multipler = 2 if bidirection else 1
        self.output_size = self.utt_cell_size * self.multipler
        self.use_attn = use_attn
        self.feat_size = feat_size
        if use_attn:
            self.key_w = nn.Linear(self.utt_cell_size * self.multipler, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, return_all=False):
        batch_size = int(utterances.size()[0])
        max_ctx_lens = int(utterances.size()[1])
        max_utt_len = int(utterances.size()[2])
        if init_state is not None:
            init_state = init_state.repeat(1, max_ctx_lens, 1)
        flat_words = utterances.view(-1, max_utt_len)
        words_embeded = self.embedding(flat_words)
        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            words_embeded = torch.cat([words_embeded, flat_feats], dim=2)
        enc_outs, enc_last = self.rnn(words_embeded, init_state=init_state)
        if self.use_attn:
            fc1 = F.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim() - 1).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)
        utt_embedded = utt_embedded.view(batch_size, max_ctx_lens, self.output_size)
        if return_all:
            return utt_embedded, enc_outs, enc_last, attn
        else:
            return utt_embedded


class BaseModel(nn.Module):

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)), dtype, self.use_gpu)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for key, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def _gather_last_out(self, rnn_outs, lens):
        """
        :param rnn_outs: batch_size x T_len x dimension
        :param lens: [a list of lens]
        :return: batch_size x dimension
        """
        time_dimension = 1
        len_vars = self.np2var(np.array(lens), LONG)
        len_vars = len_vars.view(-1, 1).expand(len(lens), rnn_outs.size(2)).unsqueeze(1)
        slices = rnn_outs.gather(time_dimension, len_vars - 1)
        return slices.squeeze(time_dimension)

    def _remove_padding(self, feats, words):
        """"
        :param feats: batch_size x num_words x feats
        :param words: batch_size x num_words
        :return: the same input without padding
        """
        if feats is None:
            return None, None
        batch_size = words.size(0)
        valid_mask = torch.sign(words).float()
        batch_lens = torch.sum(valid_mask, dim=1)
        max_word_num = torch.max(batch_lens)
        padded_lens = (max_word_num - batch_lens).cpu().data.numpy()
        valid_words = []
        valid_feats = []
        for b_id in range(batch_size):
            valid_idxs = valid_mask[b_id].nonzero().view(-1)
            valid_row_words = torch.index_select(words[b_id], 0, valid_idxs)
            valid_row_feat = torch.index_select(feats[b_id], 0, valid_idxs)
            padded_len = int(padded_lens[b_id])
            valid_row_words = F.pad(valid_row_words, (0, padded_len))
            valid_row_feat = F.pad(valid_row_feat, (0, 0, 0, padded_len))
            valid_words.append(valid_row_words.unsqueeze(0))
            valid_feats.append(valid_row_feat.unsqueeze(0))
        feats = torch.cat(valid_feats, dim=0)
        words = torch.cat(valid_words, dim=0)
        return feats, words

    def get_optimizer(self, config):
        if config.op == 'adam':
            None
            return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            None
            return torch.optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            None
            return torch.optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

    def ptr_loss(self, dec_ctx, labels):
        g = dec_ctx[DecoderPointerGen.KEY_G]
        ptr_softmax = dec_ctx[DecoderPointerGen.KEY_PTR_SOFTMAX]
        flat_ptr = ptr_softmax.view(-1, self.vocab_size)
        label_mask = labels.view(-1, 1) == self.rev_vocab[PAD]
        label_ptr = flat_ptr.gather(1, labels.view(-1, 1))
        not_in_ctx = label_ptr == 0
        mix_ptr = torch.cat([label_ptr, g.view(-1, 1)], dim=1).gather(1, not_in_ctx.long())
        attention_loss = -1.0 * torch.log(mix_ptr.clamp(min=1e-08))
        attention_loss.masked_fill_(label_mask, 0)
        valid_cnt = label_mask.size(0) - torch.sum(label_mask).float() + 1e-08
        avg_attn_loss = torch.sum(attention_loss) / valid_cnt
        return avg_attn_loss


class Pack(dict):

    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    @staticmethod
    def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
        pack = Pack()
        for k, v in dictionary.items():
            pack[k] = v
        pack['speaker'] = speaker2id[pack.speaker]
        pack['conf'] = dictionary.get('conf', 1.0)
        utt = pack['utt']
        if 'QUERY' in utt or 'RET' in utt:
            utt = str(utt)
            utt = utt.translate(None, ''.join([':', '"', '{', '}', ']', '[']))
            utt = unicode(utt)
        if include_domain:
            pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
        else:
            pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
        return pack


class PtrBase(BaseModel):

    def compute_loss(self, dec_outs, dec_ctx, labels):
        rnn_loss = self.nll_loss(dec_outs, labels)
        g = dec_ctx.get(DecoderPointerGen.KEY_G)
        if g is not None:
            ptr_softmax = dec_ctx[DecoderPointerGen.KEY_PTR_SOFTMAX]
            flat_ptr = ptr_softmax.view(-1, self.vocab_size)
            label_mask = labels.view(-1, 1) == self.rev_vocab[PAD]
            label_ptr = flat_ptr.gather(1, labels.view(-1, 1))
            not_in_ctx = label_ptr == 0
            mix_ptr = torch.cat([label_ptr, g.view(-1, 1)], dim=1).gather(1, not_in_ctx.long())
            attention_loss = -1.0 * torch.log(mix_ptr.clamp(min=1e-10))
            attention_loss.masked_fill_(label_mask, 0)
            valid_cnt = (label_mask.size(0) - torch.sum(label_mask).float()).clamp(min=1e-10)
            avg_attn_loss = torch.sum(attention_loss) / valid_cnt
        else:
            avg_attn_loss = None
        return Pack(nll=rnn_loss, attn_loss=avg_attn_loss)


BOS = '<s>'


class Bi2UniConnector(nn.Module):

    def __init__(self, rnn_cell, num_layer, hidden_size, output_size):
        super(Bi2UniConnector, self).__init__()
        if rnn_cell == 'lstm':
            self.fch = nn.Linear(hidden_size * 2 * num_layer, output_size)
            self.fcc = nn.Linear(hidden_size * 2 * num_layer, output_size)
        else:
            self.fc = nn.Linear(hidden_size * 2 * num_layer, output_size)
        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, hidden_state):
        """
        :param hidden_state: [num_layer, batch_size, feat_size]
        :param inputs: [batch_size, feat_size]
        :return: 
        """
        if self.rnn_cell == 'lstm':
            h, c = hidden_state
            num_layer = h.size()[0]
            flat_h = h.transpose(0, 1).contiguous()
            flat_c = c.transpose(0, 1).contiguous()
            new_h = self.fch(flat_h.view(-1, self.hidden_size * num_layer))
            new_c = self.fch(flat_c.view(-1, self.hidden_size * num_layer))
            return new_h.view(1, -1, self.output_size), new_c.view(1, -1, self.output_size)
        else:
            num_layer = hidden_state.size()[0]
            new_s = self.fc(hidden_state.view(-1, self.hidden_size * num_layer))
            new_s = new_s.view(1, -1, self.output_size)
            return new_s


class IdentityConnector(nn.Module):

    def __init__(self):
        super(IdentityConnector, self).__init__()

    def forward(self, hidden_state):
        return hidden_state


class HRED(BaseModel):

    def valid_loss(self, loss, batch_cnt=None):
        return loss.nll

    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout, use_attn=config.utt_type == 'attn_rnn', vocab_size=self.vocab_size, embed_dim=config.embed_size, feat_size=1)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size, config.ctx_cell_size, 0.0, config.dropout, config.num_layer, config.rnn_cell, variable_lengths=False, bidirection=config.bi_ctx_cell)
        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer, config.ctx_cell_size, config.dec_cell_size)
        else:
            self.connector = IdentityConnector()
        self.decoder = DecoderRNN(self.vocab_size, config.max_dec_len, config.embed_size, config.dec_cell_size, self.go_id, self.eos_id, n_layers=1, rnn_cell=config.rnn_cell, input_dropout_p=config.dropout, dropout_p=config.dropout, use_attention=config.use_attn, attn_size=self.ctx_encoder.output_size, attn_mode=config.attn_type, use_gpu=config.use_gpu)
        self.nll = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """         
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X
        
        :param data_feed: 
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(ctx_lens)
        enc_inputs = self.utt_encoder(ctx_utts, ctx_confs)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, ctx_lens)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.use_attn:
            attn_inputs = enc_outs
        else:
            attn_inputs = None
        dec_init_state = self.connector(enc_last)
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, dec_inputs, dec_init_state, attn_context=attn_inputs, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size)
        if mode == GEN:
            return dec_ctx, labels
        elif return_latent:
            return Pack(nll=self.nll(dec_outs, labels), latent_actions=dec_init_state)
        else:
            return Pack(nll=self.nll(dec_outs, labels))


class PtrHRED(PtrBase):

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def __init__(self, corpus, config):
        super(PtrHRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout, use_attn=True, vocab_size=self.vocab_size, embedding=self.embedding, feat_size=1)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size, config.ctx_cell_size, 0.0, config.dropout, config.num_layer, config.rnn_cell, variable_lengths=False, bidirection=config.bi_ctx_cell)
        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer, config.ctx_cell_size, config.dec_cell_size)
        else:
            self.connector = IdentityConnector()
        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderPointerGen(self.vocab_size, config.max_dec_len, config.embed_size, config.dec_cell_size, self.go_id, self.eos_id, n_layers=1, rnn_cell=config.rnn_cell, input_dropout_p=config.dropout, dropout_p=config.dropout, attn_size=self.attn_size, attn_mode=config.attn_type, use_gpu=config.use_gpu, embedding=self.embedding)
        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(ctx_lens)
        utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
        ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        dec_init_state = self.connector(ctx_last)
        ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
        utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
        attn_inputs = ctx_outs + utt_outs
        flat_ctx_words = ctx_utts.view(batch_size, -1)
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, attn_inputs, flat_ctx_words, inputs=dec_inputs, init_state=dec_init_state, mode=mode, gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            results = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                results['latent_actions'] = dec_init_state
            return results


class ZeroShotHRED(PtrBase):

    def __init__(self, corpus, config):
        super(ZeroShotHRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout, use_attn=config.utt_type == 'rnn_attn', vocab_size=self.vocab_size, embedding=self.embedding, feat_size=1)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size, config.ctx_cell_size, 0.0, config.dropout, config.num_layer, config.rnn_cell, variable_lengths=False, bidirection=config.bi_ctx_cell)
        self.policy = nn_lib.Hidden2Feat(self.ctx_encoder.output_size, config.dec_cell_size, is_lstm=config.rnn_cell == 'lstm')
        self.utt_policy = lambda x: x
        self.connector = nn_lib.LinearConnector(config.dec_cell_size, config.dec_cell_size, is_lstm=config.rnn_cell == 'lstm')
        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderRNN(self.vocab_size, config.max_dec_len, config.embed_size, config.dec_cell_size, self.go_id, self.eos_id, n_layers=1, rnn_cell=config.rnn_cell, input_dropout_p=config.dropout, dropout_p=config.dropout, use_attention=config.use_attn, attn_size=self.ctx_encoder.output_size, attn_mode=config.attn_type, use_gpu=config.use_gpu)
        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)
        domain_metas = self.np2var(data_feed.get('domain_metas'), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)
        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts.unsqueeze(1), out_confs, return_all=True)
        out_embedded = self.utt_policy(out_embedded.squeeze(1))
        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1), out_confs, return_all=True)
            act_embedded = act_embedded.squeeze(1)
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
            latent_action = self.policy(ctx_last)
            ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs
            attn_words = ctx_utts.view(batch_size, -1)
        dec_init_state = self.connector(latent_action)
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, dec_inputs, dec_init_state, attn_context=attn_inputs, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size)
        if mode == GEN:
            return dec_ctx, labels
        else:
            rnn_loss = self.nll_loss(dec_outs, labels)
            loss_pack = Pack(nll=rnn_loss)
            if return_latent:
                loss_pack['latent_actions'] = latent_action
            loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
            return loss_pack


class ZeroShotPtrHRED(PtrBase):

    def __init__(self, corpus, config):
        super(ZeroShotPtrHRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout, use_attn=config.utt_type == 'rnn_attn', vocab_size=self.vocab_size, embedding=self.embedding, feat_size=1)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size, config.ctx_cell_size, 0.0, config.dropout, config.num_layer, config.rnn_cell, variable_lengths=False, bidirection=config.bi_ctx_cell)
        self.policy = nn.Linear(self.ctx_encoder.output_size, config.dec_cell_size)
        self.utt_policy = lambda x: x
        self.connector = nn_lib.LinearConnector(config.dec_cell_size, config.dec_cell_size, is_lstm=config.rnn_cell == 'lstm')
        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderPointerGen(self.vocab_size, config.max_dec_len, config.embed_size, config.dec_cell_size, self.go_id, self.eos_id, n_layers=1, rnn_cell=config.rnn_cell, input_dropout_p=config.dropout, dropout_p=config.dropout, attn_size=self.attn_size, attn_mode=config.attn_type, use_gpu=config.use_gpu, embedding=self.embedding)
        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)
        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts.unsqueeze(1), out_confs, return_all=True)
        out_embedded = self.utt_policy(out_embedded.squeeze(1))
        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1), out_confs, return_all=True)
            act_embedded = act_embedded.squeeze(1)
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
            pi_inputs = self._gather_last_out(ctx_outs, ctx_lens)
            latent_action = self.policy(pi_inputs)
            ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs
            attn_words = ctx_utts.view(batch_size, -1)
        dec_init_state = self.connector(latent_action)
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, attn_inputs, attn_words, inputs=dec_inputs, init_state=dec_init_state, mode=mode, gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            loss_pack = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                loss_pack['latent_actions'] = latent_action
            loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
            return loss_pack


class AttnConnector(nn.Module):

    def __init__(self, rnn_cell, query_size, key_size, content_size, output_size, attn_size):
        super(AttnConnector, self).__init__()
        self.query_embed = nn.Linear(query_size, attn_size)
        self.key_embed = nn.Linear(key_size, attn_size)
        self.attn_w = nn.Linear(attn_size, 1)
        if rnn_cell == 'lstm':
            self.project_h = nn.Linear(content_size + query_size, output_size)
            self.project_c = nn.Linear(content_size + query_size, output_size)
        else:
            self.project = nn.Linear(content_size + query_size, output_size)
        self.rnn_cell = rnn_cell
        self.query_size = query_size
        self.key_size = key_size
        self.content_size = content_size
        self.output_size = output_size

    def forward(self, queries, keys, contents):
        batch_size = keys.size(0)
        num_key = keys.size(1)
        query_embeded = self.query_embed(queries)
        key_embeded = self.key_embed(keys)
        tiled_query = query_embeded.unsqueeze(1).repeat(1, num_key, 1)
        fc1 = F.tanh(tiled_query + key_embeded)
        attn = self.attn_w(fc1).squeeze(-1)
        attn = F.sigmoid(attn.view(-1, num_key)).view(batch_size, -1, num_key)
        mix = torch.bmm(attn, contents).squeeze(1)
        out = torch.cat([mix, queries], dim=1)
        if self.rnn_cell == 'lstm':
            h = self.project_h(out).unsqueeze(0)
            c = self.project_c(out).unsqueeze(0)
            new_s = h, c
        else:
            new_s = self.project(out).unsqueeze(0)
        return new_s


class LinearConnector(nn.Module):

    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(LinearConnector, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size 
        :return: 
        """
        if self.is_lstm:
            h = self.linear_h(inputs).unsqueeze(0)
            c = self.linear_c(inputs).unsqueeze(0)
            return h, c
        else:
            return self.linear(inputs).unsqueeze(0)

    def get_w(self):
        if self.is_lstm:
            return self.linear_h.weight
        else:
            return self.linear.weight


class Hidden2Feat(nn.Module):

    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(Hidden2Feat, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h = self.linear_h(inputs[0].squeeze(0))
            c = self.linear_c(inputs[1].squeeze(0))
            return h + c
        else:
            return self.linear(inputs.squeeze(0))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttnConnector,
     lambda: ([], {'rnn_cell': 4, 'query_size': 4, 'key_size': 4, 'content_size': 4, 'output_size': 4, 'attn_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Hidden2Feat,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'is_lstm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IdentityConnector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearConnector,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'is_lstm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_snakeztc_NeuralDialog_ZSDG(_paritybench_base):
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

