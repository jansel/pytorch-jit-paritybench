import sys
_module = sys.modules[__name__]
del sys
beam_search = _module
config = _module
evaluate = _module
logger_test = _module
predict = _module
preprocess = _module
preprocess_testset = _module
pykp = _module
data = _module
export_unique_keyphrase = _module
mag = _module
export_doctag2vec = _module
extract = _module
post_clean = _module
remove_duplicates = _module
remove_duplicates_multiprocess = _module
stanford = _module
corenlp = _module
test_dataset_producer = _module
dataloader = _module
eric_layers = _module
example = _module
producer_consumer = _module
io = _module
metric = _module
bleu = _module
model = _module
post_evaluate = _module
stat_print = _module
train = _module
train_rl = _module
utils = _module

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


import torch


import numpy as np


import torch.nn.functional as F


import logging


import torch.nn as nn


import torch.nn.functional as func


from torch.autograd import Variable


import random


import time


from torch.optim import Adam


import copy


from torch.utils.data import DataLoader


from torch import cuda


class GetMask(torch.nn.Module):
    """
    inputs: x:          any size
    outputs:mask:       same size as input x
    """

    def __init__(self, pad_idx=0):
        super(GetMask, self).__init__()
        self.pad_idx = pad_idx

    def forward(self, x):
        mask = torch.ne(x, self.pad_idx).float()
        return mask


class StandardNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        log_prob:   batch x time x class
        y_true:     batch x time
        mask:       batch x time
        output:     batch
    """

    def forward(self, log_prob, y_true, mask):
        mask = mask.float()
        log_P = torch.gather(log_prob.view(-1, log_prob.size(2)), 1, y_true
            .contiguous().view(-1, 1))
        log_P = log_P.view(y_true.size(0), y_true.size(1))
        log_P = log_P * mask
        sum_log_P = torch.sum(log_P, dim=1) / torch.sum(mask, dim=1)
        return -sum_log_P


class TimeDistributedDense(torch.nn.Module):
    """
    input:  x:          batch x time x a
            mask:       batch x time
    output: y:          batch x time x b
    """

    def __init__(self, mlp):
        super(TimeDistributedDense, self).__init__()
        self.mlp = mlp

    def forward(self, x, mask=None):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        y = self.mlp.forward(x)
        y = y.view(x_size[:-1] + (y.size(-1),))
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return y


class AttentionExample(nn.Module):

    def __init__(self, hidden_size, method='concat'):
        super(AttentionExample, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len))
        if torch.cuda.is_available():
            attn_energies = attn_energies
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return torch.nn.functional.softmax(attn_energies).unsqueeze(0
            ).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


def masked_softmax(x, m=None, axis=-1):
    """
    Softmax with mask (optional)
    """
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-06)
    return softmax


class Attention(nn.Module):

    def __init__(self, enc_dim, trg_dim, method='general'):
        super(Attention, self).__init__()
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(enc_dim, trg_dim)
        elif self.method == 'concat':
            attn = nn.Linear(enc_dim + trg_dim, trg_dim)
            v = nn.Linear(trg_dim, 1)
            self.attn = TimeDistributedDense(mlp=attn)
            self.v = TimeDistributedDense(mlp=v)
        self.softmax = nn.Softmax()
        if self.method == 'dot':
            self.linear_out = nn.Linear(2 * trg_dim, trg_dim, bias=False)
        else:
            self.linear_out = nn.Linear(enc_dim + trg_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def score(self, hiddens, encoder_outputs, encoder_mask=None):
        """
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        """
        if self.method == 'dot':
            energies = torch.bmm(hiddens, encoder_outputs.transpose(1, 2))
        elif self.method == 'general':
            energies = self.attn(encoder_outputs)
            if encoder_mask is not None:
                energies = energies * encoder_mask.view(encoder_mask.size(0
                    ), encoder_mask.size(1), 1)
            energies = torch.bmm(hiddens, energies.transpose(1, 2))
        elif self.method == 'concat':
            energies = []
            batch_size = encoder_outputs.size(0)
            src_len = encoder_outputs.size(1)
            for i in range(hiddens.size(1)):
                hidden_i = hiddens[:, i:i + 1, :].expand(-1, src_len, -1)
                concated = torch.cat((hidden_i, encoder_outputs), 2)
                if encoder_mask is not None:
                    concated = concated * encoder_mask.view(encoder_mask.
                        size(0), encoder_mask.size(1), 1)
                energy = self.tanh(self.attn(concated, encoder_mask))
                if encoder_mask is not None:
                    energy = energy * encoder_mask.view(encoder_mask.size(0
                        ), encoder_mask.size(1), 1)
                energy = self.v(energy, encoder_mask).squeeze(-1)
                energies.append(energy)
            energies = torch.stack(energies, dim=1)
            if encoder_mask is not None:
                energies = energies * encoder_mask.view(encoder_mask.size(0
                    ), 1, encoder_mask.size(1))
        return energies.contiguous()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        """
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
        :return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        """
        """
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1))) # src_seq_len * batch_size
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(encoder_outputs.size(0)):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, transpose to (batch_size * src_seq_len)
        attn = torch.nn.functional.softmax(attn_energies.t())
        # get the weighted context, (batch_size, src_layer_number * src_encoder_dim)
        weighted_context = torch.bmm(encoder_outputs.permute(1, 2, 0), attn.unsqueeze(2)).squeeze(2)  # (batch_size, src_hidden_dim * num_directions)
        """
        batch_size = hidden.size(0)
        src_len = encoder_outputs.size(1)
        trg_len = hidden.size(1)
        context_dim = encoder_outputs.size(2)
        trg_hidden_dim = hidden.size(2)
        attn_energies = self.score(hidden, encoder_outputs)
        if encoder_mask is None:
            attn_weights = torch.nn.functional.softmax(attn_energies.view(-
                1, src_len), dim=1).view(batch_size, trg_len, src_len)
        else:
            attn_energies = attn_energies * encoder_mask.view(encoder_mask.
                size(0), 1, encoder_mask.size(1))
            attn_weights = masked_softmax(attn_energies, encoder_mask.view(
                encoder_mask.size(0), 1, encoder_mask.size(1)), -1)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)
        h_tilde = torch.cat((weighted_context, hidden), 2)
        h_tilde = self.tanh(self.linear_out(h_tilde.view(-1, context_dim +
            trg_hidden_dim)))
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim
            ), attn_weights, attn_energies

    def forward_(self, hidden, context):
        """
        Original forward for DotAttention, it doesn't work if the dim of encoder and decoder are not same
        input and context must be in same dim: return Softmax(hidden.dot([c for c in context]))
        input: batch x hidden_dim
        context: batch x source_len x hidden_dim
        """
        target = self.linear_in(hidden).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, hidden), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()
        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.src_hidden_dim = opt.rnn_size
        self.trg_hidden_dim = opt.rnn_size
        self.ctx_hidden_dim = opt.rnn_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.nlayers_src = opt.enc_layers
        self.nlayers_trg = opt.dec_layers
        self.dropout = opt.dropout
        self.pad_token_src = opt.word2id[pykp.io.PAD_WORD]
        self.pad_token_trg = opt.word2id[pykp.io.PAD_WORD]
        self.unk_word = opt.word2id[pykp.io.UNK_WORD]
        self.attention_mode = opt.attention_mode
        self.input_feeding = opt.input_feeding
        self.copy_attention = opt.copy_attention
        self.copy_mode = opt.copy_mode
        self.copy_input_feeding = opt.copy_input_feeding
        self.reuse_copy_attn = opt.reuse_copy_attn
        self.copy_gate = opt.copy_gate
        self.must_teacher_forcing = opt.must_teacher_forcing
        self.teacher_forcing_ratio = opt.teacher_forcing_ratio
        self.scheduled_sampling = opt.scheduled_sampling
        self.scheduled_sampling_batches = opt.scheduled_sampling_batches
        self.scheduled_sampling_type = 'inverse_sigmoid'
        self.current_batch = 0
        if self.scheduled_sampling:
            logging.info(
                'Applying scheduled sampling with %s decay for the first %d batches'
                 % (self.scheduled_sampling_type, self.
                scheduled_sampling_batches))
        if self.must_teacher_forcing or self.teacher_forcing_ratio >= 1:
            logging.info('Training with All Teacher Forcing')
        elif self.teacher_forcing_ratio <= 0:
            logging.info('Training with All Sampling')
        else:
            logging.info(
                'Training with Teacher Forcing with static rate=%f' % self.
                teacher_forcing_ratio)
        self.get_mask = GetMask(self.pad_token_src)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.
            pad_token_src)
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=self.
            src_hidden_dim, num_layers=self.nlayers_src, bidirectional=self
            .bidirectional, batch_first=True, dropout=self.dropout)
        self.decoder = nn.LSTM(input_size=self.emb_dim, hidden_size=self.
            trg_hidden_dim, num_layers=self.nlayers_trg, bidirectional=
            False, batch_first=False, dropout=self.dropout)
        self.attention_layer = Attention(self.src_hidden_dim * self.
            num_directions, self.trg_hidden_dim, method=self.attention_mode)
        self.encoder2decoder_hidden = nn.Linear(self.src_hidden_dim * self.
            num_directions, self.trg_hidden_dim)
        self.encoder2decoder_cell = nn.Linear(self.src_hidden_dim * self.
            num_directions, self.trg_hidden_dim)
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)
        if self.copy_attention:
            if self.copy_mode == None and self.attention_mode:
                self.copy_mode = self.attention_mode
            assert self.copy_mode != None
            assert self.unk_word != None
            logging.info('Applying Copy Mechanism, type=%s' % self.copy_mode)
            self.copy_attention_layer = Attention(self.src_hidden_dim *
                self.num_directions, self.trg_hidden_dim, method=self.copy_mode
                )
        else:
            self.copy_mode = None
            self.copy_input_feeding = False
            self.copy_attention_layer = None
        self.dec_input_dim = self.emb_dim
        if self.input_feeding:
            logging.info('Applying input feeding')
            self.dec_input_dim += self.trg_hidden_dim
        if self.copy_input_feeding:
            logging.info('Applying copy input feeding')
            self.dec_input_dim += self.trg_hidden_dim
        if self.dec_input_dim == self.emb_dim:
            self.dec_input_bridge = None
        else:
            self.dec_input_bridge = nn.Linear(self.dec_input_dim, self.emb_dim)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder_hidden.bias.data.fill_(0)
        self.encoder2decoder_cell.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def init_encoder_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) if self.encoder.batch_first else input.size(
            1)
        h0_encoder = Variable(torch.zeros(self.encoder.num_layers * self.
            num_directions, batch_size, self.src_hidden_dim), requires_grad
            =False)
        c0_encoder = Variable(torch.zeros(self.encoder.num_layers * self.
            num_directions, batch_size, self.src_hidden_dim), requires_grad
            =False)
        if torch.cuda.is_available():
            return h0_encoder, c0_encoder
        return h0_encoder, c0_encoder

    def init_decoder_state(self, enc_h, enc_c):
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder_hidden(enc_h)
            ).unsqueeze(0)
        decoder_init_cell = nn.Tanh()(self.encoder2decoder_cell(enc_c)
            ).unsqueeze(0)
        return decoder_init_hidden, decoder_init_cell

    def forward(self, input_src, input_src_len, input_trg, input_src_ext,
        oov_lists, trg_mask=None, ctx_mask=None):
        """
        The differences of copy model from normal seq2seq here are:
         1. The size of decoder_logits is (batch_size, trg_seq_len, vocab_size + max_oov_number).Usually vocab_size=50000 and max_oov_number=1000. And only very few of (it's very rare to have many unk words, in most cases it's because the text is not in English)
         2. Return the copy_attn_weights as well. If it's See's model, the weights are same to attn_weights as it reuse the original attention
         3. Very important: as we need to merge probs of copying and generative part, thus we have to operate with probs instead of logits. Thus here we return the probs not logits. Respectively, the loss criterion outside is NLLLoss but not CrossEntropyLoss any more.
        :param
            input_src : numericalized source text, oov words have been replaced with <unk>
            input_trg : numericalized target text, oov words have been replaced with temporary oov index
            input_src_ext : numericalized source text in extended vocab, oov words have been replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words
        :returns
            decoder_logits      : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        """
        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_probs, decoder_hiddens, attn_weights, copy_attn_weights = (self
            .decode(trg_inputs=input_trg, src_map=input_src_ext, oov_list=
            oov_lists, enc_context=src_h, enc_hidden=(src_h_t, src_c_t),
            trg_mask=trg_mask, ctx_mask=ctx_mask))
        return decoder_probs, decoder_hiddens, (attn_weights, copy_attn_weights
            )

    def encode(self, input_src, input_src_len):
        """
        Propogate input through the network.
        """
        self.h0_encoder, self.c0_encoder = self.init_encoder_state(input_src)
        src_emb = self.embedding(input_src)
        src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len,
            batch_first=True)
        src_h, (src_h_t, src_c_t) = self.encoder(src_emb, (self.h0_encoder,
            self.c0_encoder))
        src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True)
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        return src_h, (h_t, c_t)

    def merge_decode_inputs(self, trg_emb, h_tilde, copy_h_tilde):
        """
        Input-feeding: merge the information of current word and attentional hidden vectors
        :param trg_emb: (batch_size, 1, embed_dim)
        :param h_tilde: (batch_size, 1, trg_hidden)
        :param copy_h_tilde: (batch_size, 1, trg_hidden)
        :return:
        """
        trg_emb = trg_emb.permute(1, 0, 2)
        inputs = trg_emb
        if self.input_feeding:
            h_tilde = h_tilde.permute(1, 0, 2)
            inputs = torch.cat((inputs, h_tilde), 2)
        if self.copy_input_feeding:
            copy_h_tilde = copy_h_tilde.permute(1, 0, 2)
            inputs = torch.cat((inputs, copy_h_tilde), 2)
        if self.dec_input_bridge:
            dec_input = nn.Tanh()(self.dec_input_bridge(inputs))
        else:
            dec_input = trg_emb
        return dec_input

    def decode(self, trg_inputs, src_map, oov_list, enc_context, enc_hidden,
        trg_mask, ctx_mask):
        """
        :param
                trg_input:         (batch_size, trg_len)
                src_map  :         (batch_size, src_len), almost the same with src but oov words are replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words. The word index can be beyond vocab_size, e.g. 50000, 50001, 50002 etc, depends on how many oov words appear in the source text
                context vector:    (batch_size, src_len, hidden_size * num_direction) the outputs (hidden vectors) of encoder
                context mask:      (batch_size, src_len)
        :returns
            decoder_probs       : (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        """
        batch_size = trg_inputs.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_inputs.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context
                .contiguous().view(-1, context_dim))).view(batch_size,
                src_len, trg_hidden_dim)
            enc_context = enc_context * ctx_mask.view(ctx_mask.size() + (1,))
        max_length = trg_inputs.size(1) - 1
        self.current_batch += 1
        do_word_wisely_training = False
        if not do_word_wisely_training:
            """
            Teacher Forcing
            (1) Feedforwarding RNN
            """
            trg_inputs = trg_inputs[:, :-1]
            trg_emb = self.embedding(trg_inputs)
            trg_emb = trg_emb.permute(1, 0, 2)
            decoder_outputs, dec_hidden = self.decoder(trg_emb, init_hidden)
            """
            (2) Standard Attention
            """
            h_tildes, attn_weights, attn_logits = self.attention_layer(
                decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask
                =ctx_mask)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1,
                trg_hidden_dim)).view(batch_size, max_length, -1)
            """
            (3) Copy Attention
            """
            if self.copy_attention:
                if not self.reuse_copy_attn:
                    _, copy_weights, copy_logits = self.copy_attention_layer(
                        decoder_outputs.permute(1, 0, 2), enc_context,
                        encoder_mask=ctx_mask)
                else:
                    copy_logits = attn_logits
                decoder_log_probs = self.merge_copy_probs(decoder_logits,
                    copy_logits, src_map, oov_list)
                decoder_outputs = decoder_outputs.permute(1, 0, 2)
            else:
                decoder_log_probs = torch.nn.functional.log_softmax(
                    decoder_logits, dim=-1).view(batch_size, -1, self.
                    vocab_size)
                copy_weights = []
        else:
            """
            Word Sampling
            (1) Feedforwarding RNN
            """
            trg_input = trg_inputs[:, (0)].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs = []
            attn_weights = []
            copy_weights = []
            dec_hidden = init_hidden
            h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)
                ) if torch.cuda.is_available() else Variable(torch.zeros(
                batch_size, 1, trg_hidden_dim))
            copy_h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)
                ) if torch.cuda.is_available() else Variable(torch.zeros(
                batch_size, 1, trg_hidden_dim))
            for di in range(max_length):
                trg_emb = self.embedding(trg_input)
                dec_input = self.merge_decode_inputs(trg_emb, h_tilde,
                    copy_h_tilde)
                decoder_output, dec_hidden = self.decoder(dec_input, dec_hidden
                    )
                """
                (2) Standard Attention
                """
                h_tilde, attn_weight, attn_logit = self.attention_layer(
                    decoder_output.permute(1, 0, 2), enc_context,
                    encoder_mask=ctx_mask)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1,
                    trg_hidden_dim)).view(batch_size, 1, -1)
                """
                (3) Copy Attention
                """
                if self.copy_attention:
                    if not self.reuse_copy_attn:
                        copy_h_tilde, copy_weight, copy_logit = (self.
                            copy_attention_layer(decoder_output.permute(1, 
                            0, 2), enc_context, encoder_mask=ctx_mask))
                    else:
                        copy_h_tilde, copy_weight, copy_logit = (h_tilde,
                            attn_weight, attn_logit)
                    decoder_log_prob = self.merge_copy_probs(decoder_logit,
                        copy_logit, src_map, oov_list)
                else:
                    decoder_log_prob = torch.nn.functional.log_softmax(
                        decoder_logit, dim=-1).view(batch_size, -1, self.
                        vocab_size)
                    copy_weight = None
                """
                Prepare for the next iteration
                """
                if self.do_teacher_forcing():
                    trg_input = trg_inputs[:, (di + 1)].unsqueeze(1)
                else:
                    top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                    top_idx[top_idx >= self.vocab_size] = self.unk_word
                    top_idx = Variable(top_idx.squeeze(2))
                    trg_input = top_idx if torch.cuda.is_available(
                        ) else top_idx
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                if self.copy_attention:
                    copy_weights.append(copy_weight.permute(1, 0, 2))
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2
                )
            decoder_outputs = torch.cat(decoder_outputs, 0).permute(1, 0, 2)
            attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)
            if self.copy_attention:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2)
        return decoder_log_probs, decoder_outputs, attn_weights, copy_weights

    def merge_oov2unk(self, decoder_log_prob, max_oov_number):
        """
        Merge the probs of oov words to the probs of <unk>, in order to generate the next word
        :param decoder_log_prob: log_probs after merging generative and copying (batch_size, trg_seq_len, vocab_size + max_oov_number)
        :return:
        """
        batch_size, seq_len, _ = decoder_log_prob.size()
        vocab_index = Variable(torch.arange(start=0, end=self.vocab_size).
            type(torch.LongTensor))
        oov_index = Variable(torch.arange(start=self.vocab_size, end=self.
            vocab_size + max_oov_number).type(torch.LongTensor))
        oov2unk_index = Variable(torch.zeros(batch_size * seq_len,
            max_oov_number).type(torch.LongTensor) + self.unk_word)
        if torch.cuda.is_available():
            vocab_index = vocab_index
            oov_index = oov_index
            oov2unk_index = oov2unk_index
        merged_log_prob = torch.index_select(decoder_log_prob, dim=2, index
            =vocab_index).view(batch_size * seq_len, self.vocab_size)
        oov_log_prob = torch.index_select(decoder_log_prob, dim=2, index=
            oov_index).view(batch_size * seq_len, max_oov_number)
        merged_log_prob = merged_log_prob.scatter_add_(1, oov2unk_index,
            oov_log_prob)
        merged_log_prob = merged_log_prob.view(batch_size, seq_len, self.
            vocab_size)
        return merged_log_prob

    def merge_copy_probs(self, decoder_logits, copy_logits, src_map, oov_list):
        """
        The function takes logits as inputs here because Gu's model applies softmax in the end, to normalize generative/copying together
        The tricky part is, Gu's model merges the logits of generative and copying part instead of probabilities,
            then simply initialize the entended part to zeros would be erroneous because many logits are large negative floats.
        To the sentences that have oovs it's fine. But if some sentences in a batch don't have oovs but mixed with sentences have oovs, the extended oov part would be ranked highly after softmax (zero is larger than other negative values in logits).
        Thus we have to carefully initialize the oov-extended part of no-oov sentences to negative infinite floats.
        Note that it may cause exception on early versions like on '0.3.1.post2', but it works well on 0.4 ({RuntimeError}in-place operations can be only used on variables that don't share storage with any other variables, but detected that there are 2 objects sharing it)
        :param decoder_logits: (batch_size, trg_seq_len, vocab_size)
        :param copy_logits:    (batch_size, trg_len, src_len) the pointing/copying logits of each target words
        :param src_map:        (batch_size, src_len)
        :return:
            decoder_copy_probs: return the log_probs (batch_size, trg_seq_len, vocab_size + max_oov_number)
        """
        batch_size, max_length, _ = decoder_logits.size()
        src_len = src_map.size(1)
        max_oov_number = max([len(oovs) for oovs in oov_list])
        flattened_decoder_logits = decoder_logits.view(batch_size *
            max_length, self.vocab_size)
        if max_oov_number > 0:
            """
            extended_zeros           = Variable(torch.zeros(batch_size * max_length, max_oov_number))
            extended_zeros           = extended_zeros.cuda() if torch.cuda.is_available() else extended_zeros
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_zeros), dim=1)
            """
            extended_logits = Variable(torch.FloatTensor([([0.0] * len(oov) +
                [float('-inf')] * (max_oov_number - len(oov))) for oov in
                oov_list]))
            extended_logits = extended_logits.unsqueeze(1).expand(batch_size,
                max_length, max_oov_number).contiguous().view(batch_size *
                max_length, -1)
            extended_logits = extended_logits if torch.cuda.is_available(
                ) else extended_logits
            flattened_decoder_logits = torch.cat((flattened_decoder_logits,
                extended_logits), dim=1)
        expanded_src_map = src_map.unsqueeze(1).expand(batch_size,
            max_length, src_len).contiguous().view(batch_size * max_length, -1)
        flattened_decoder_logits = flattened_decoder_logits.scatter_add_(1,
            expanded_src_map, copy_logits.view(batch_size * max_length, -1))
        flattened_decoder_logits = torch.nn.functional.log_softmax(
            flattened_decoder_logits, dim=1)
        decoder_log_probs = flattened_decoder_logits.view(batch_size,
            max_length, self.vocab_size + max_oov_number)
        return decoder_log_probs

    def do_teacher_forcing(self):
        if self.scheduled_sampling:
            if self.scheduled_sampling_type == 'linear':
                teacher_forcing_ratio = 1 - float(self.current_batch
                    ) / self.scheduled_sampling_batches
            elif self.scheduled_sampling_type == 'inverse_sigmoid':
                x = (float(self.current_batch) / self.
                    scheduled_sampling_batches * 10 if self.
                    scheduled_sampling_batches > 0 else 0.0)
                teacher_forcing_ratio = 1.0 / (1.0 + np.exp(x - 5))
        elif self.must_teacher_forcing:
            teacher_forcing_ratio = 1.0
        else:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        coin = random.random()
        do_tf = coin < teacher_forcing_ratio
        return do_tf

    def generate(self, trg_input, dec_hidden, enc_context, ctx_mask=None,
        src_map=None, oov_list=None, max_len=1, return_attention=False):
        """
        Given the initial input, state and the source contexts, return the top K restuls for each time step
        :param trg_input: just word indexes of target texts (usually zeros indicating BOS <s>)
        :param dec_hidden: hidden states for decoder RNN to start with
        :param enc_context: context encoding vectors
        :param src_map: required if it's copy model
        :param oov_list: required if it's copy model
        :param k (deprecated): Top K to return
        :param feed_all_timesteps: it's one-step predicting or feed all inputs to run through all the time steps
        :param get_attention: return attention vectors?
        :return:
        """
        batch_size = trg_input.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_input.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim
        h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)
            ) if torch.cuda.is_available() else Variable(torch.zeros(
            batch_size, 1, trg_hidden_dim))
        copy_h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)
            ) if torch.cuda.is_available() else Variable(torch.zeros(
            batch_size, 1, trg_hidden_dim))
        attn_weights = []
        copy_weights = []
        log_probs = []
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context
                .contiguous().view(-1, context_dim))).view(batch_size,
                src_len, trg_hidden_dim)
        for i in range(max_len):
            trg_emb = self.embedding(trg_input)
            dec_input = self.merge_decode_inputs(trg_emb, h_tilde, copy_h_tilde
                )
            decoder_output, dec_hidden = self.decoder(dec_input, dec_hidden)
            h_tilde, attn_weight, attn_logit = self.attention_layer(
                decoder_output.permute(1, 0, 2), enc_context, encoder_mask=
                ctx_mask)
            decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim)
                )
            if not self.copy_attention:
                decoder_log_prob = torch.nn.functional.log_softmax(
                    decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
            else:
                decoder_logit = decoder_logit.view(batch_size, 1, self.
                    vocab_size)
                if not self.reuse_copy_attn:
                    copy_h_tilde, copy_weight, copy_logit = (self.
                        copy_attention_layer(decoder_output.permute(1, 0, 2
                        ), enc_context, encoder_mask=ctx_mask))
                else:
                    copy_h_tilde, copy_weight, copy_logit = (h_tilde,
                        attn_weight, attn_logit)
                copy_weights.append(copy_weight.permute(1, 0, 2))
                decoder_log_prob = self.merge_copy_probs(decoder_logit,
                    copy_logit, src_map, oov_list)
            top_1_v, top_1_idx = decoder_log_prob.data.topk(1, dim=-1)
            trg_input = Variable(top_1_idx.squeeze(2))
            log_probs.append(decoder_log_prob.permute(1, 0, 2))
            attn_weights.append(attn_weight.permute(1, 0, 2))
        log_probs = torch.cat(log_probs, 0).permute(1, 0, 2)
        attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)
        if return_attention:
            if not self.copy_attention:
                return log_probs, dec_hidden, attn_weights
            else:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2)
                return log_probs, dec_hidden, (attn_weights, copy_weights)
        else:
            return log_probs, dec_hidden

    def greedy_predict(self, input_src, input_trg, trg_mask=None, ctx_mask=None
        ):
        src_h, (src_h_t, src_c_t) = self.encode(input_src)
        if torch.cuda.is_available():
            input_trg = input_trg
        decoder_logits, hiddens, attn_weights = self.decode_old(trg_input=
            input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t),
            trg_mask=trg_mask, ctx_mask=ctx_mask, is_train=False)
        if torch.cuda.is_available():
            max_words_pred = decoder_logits.data.cpu().numpy().argmax(axis=-1
                ).flatten()
        else:
            max_words_pred = decoder_logits.data.numpy().argmax(axis=-1
                ).flatten()
        return max_words_pred

    def forward_without_copy(self, input_src, input_src_len, input_trg,
        trg_mask=None, ctx_mask=None):
        """
        [Obsolete] To be compatible with the Copy Model, we change the output of logits to log_probs
        :param input_src: padded numeric source sequences
        :param input_src_len: (list of int) length of each sequence before padding (required for pack_padded_sequence)
        :param input_trg: padded numeric target sequences
        :param trg_mask:
        :param ctx_mask:

        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        """
        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_log_probs, decoder_hiddens, attn_weights = self.decode(
            trg_inputs=input_trg, enc_context=src_h, enc_hidden=(src_h_t,
            src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_log_probs, decoder_hiddens, attn_weights

    def decode_without_copy(self, trg_inputs, enc_context, enc_hidden,
        trg_mask, ctx_mask):
        """
        [Obsolete] Initial decoder state h0 (batch_size, trg_hidden_size), converted from h_t of encoder (batch_size, src_hidden_size * num_directions) through a linear layer
            No transformation for cell state c_t. Pass directly to decoder.
            Nov. 11st: update: change to pass c_t as well
            People also do that directly feed the end hidden state of encoder and initialize cell state as zeros
        :param
                trg_input:         (batch_size, trg_len)
                context vector:    (batch_size, src_len, hidden_size * num_direction) is outputs of encoder
        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        """
        batch_size = trg_inputs.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_inputs.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context
                .contiguous().view(-1, context_dim))).view(batch_size,
                src_len, trg_hidden_dim)
        max_length = trg_inputs.size(1) - 1
        self.current_batch += 1
        if self.do_teacher_forcing():
            trg_inputs = trg_inputs[:, :-1]
            trg_emb = self.embedding(trg_inputs)
            trg_emb = trg_emb.permute(1, 0, 2)
            decoder_outputs, dec_hidden = self.decoder(trg_emb, init_hidden)
            h_tildes, attn_weights, _ = self.attention_layer(decoder_outputs
                .permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1,
                trg_hidden_dim))
            decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits,
                dim=-1).view(batch_size, max_length, self.vocab_size)
            decoder_outputs = decoder_outputs.permute(1, 0, 2)
        else:
            trg_input = trg_inputs[:, (0)].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs = []
            attn_weights = []
            dec_hidden = init_hidden
            for di in range(max_length):
                trg_emb = self.embedding(trg_input)
                trg_emb = trg_emb.permute(1, 0, 2)
                decoder_output, dec_hidden = self.decoder(trg_emb, dec_hidden)
                h_tilde, attn_weight, _ = self.attention_layer(decoder_output
                    .permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1,
                    trg_hidden_dim))
                decoder_log_prob = torch.nn.functional.log_softmax(
                    decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
                """
                Prepare for the next iteration
                """
                top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                top_idx = Variable(top_idx.squeeze(2))
                trg_input = top_idx if torch.cuda.is_available() else top_idx
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2
                )
            decoder_outputs = torch.cat(decoder_outputs, 0).permute(1, 0, 2)
            attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)
        return decoder_log_probs, decoder_outputs, attn_weights


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_memray_seq2seq_keyphrase_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Attention(*[], **{'enc_dim': 4, 'trg_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(GetMask(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(StandardNLL(*[], **{}), [torch.zeros([4, 4, 4], dtype=torch.int64), torch.zeros([4, 4], dtype=torch.int64), torch.rand([4, 4, 4, 4])], {})

