import sys
_module = sys.modules[__name__]
del sys
make_cnndm_data = _module
make_google_data = _module
model = _module
params = _module
test = _module
train = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import random


from typing import Union


from typing import List


import math


from torch import optim


from torch.nn.utils import clip_grad_norm_


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0
        ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidi else 1
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi,
            dropout=rnn_drop)

    def forward(self, embedded, hidden, input_lengths=None):
        """
    :param embedded: (src seq len, batch size, embed size)
    :param hidden: (num directions, batch size, encoder hidden size)
    :param input_lengths: list containing the non-padded length of each sequence in this batch;
                          if set, we use `PackedSequence` to skip the PAD inputs and leave the
                          corresponding encoder states as zeros
    :return: (src seq len, batch size, hidden size * num directions = decoder hidden size)

    Perform multi-step encoding.
    """
        if input_lengths is not None:
            embedded = pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(embedded, hidden)
        if input_lengths is not None:
            output, _ = pad_packed_sequence(output)
        if self.num_directions > 1:
            batch_size = hidden.size(1)
            hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                self.hidden_size * self.num_directions)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.
            hidden_size, device=DEVICE)


eps = 1e-31


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, *, enc_attn=
        True, dec_attn=True, enc_attn_cover=True, pointer=True,
        tied_embedding=None, out_embed_size=None, in_drop: float=0,
        rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.out_embed_size = out_embed_size
        if (tied_embedding is not None and self.out_embed_size and 
            embed_size != self.out_embed_size):
            None
            self.out_embed_size = embed_size
        self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
        self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)
        if enc_attn:
            if not enc_hidden_size:
                enc_hidden_size = self.hidden_size
            self.enc_bilinear = nn.Bilinear(self.hidden_size,
                enc_hidden_size, 1)
            self.combined_size += enc_hidden_size
            if enc_attn_cover:
                self.cover_weight = nn.Parameter(torch.rand(1))
        if dec_attn:
            self.dec_bilinear = nn.Bilinear(self.hidden_size, self.
                hidden_size, 1)
            self.combined_size += self.hidden_size
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
        if pointer:
            self.ptr = nn.Linear(self.combined_size, 1)
        if tied_embedding is not None and embed_size != self.combined_size:
            self.out_embed_size = embed_size
        if self.out_embed_size:
            self.pre_out = nn.Linear(self.combined_size, self.out_embed_size)
            size_before_output = self.out_embed_size
        else:
            size_before_output = self.combined_size
        self.out = nn.Linear(size_before_output, vocab_size)
        if tied_embedding is not None:
            self.out.weight = tied_embedding.weight

    def forward(self, embedded, hidden, encoder_states=None, decoder_states
        =None, coverage_vector=None, *, encoder_word_idx=None,
        ext_vocab_size: int=None, log_prob: bool=True):
        """
    :param embedded: (batch size, embed size)
    :param hidden: (1, batch size, decoder hidden size)
    :param encoder_states: (src seq len, batch size, hidden size), for attention mechanism
    :param decoder_states: (past dec steps, batch size, hidden size), for attention mechanism
    :param encoder_word_idx: (src seq len, batch size), for pointer network
    :param ext_vocab_size: the dynamic vocab size, determined by the max num of OOV words contained
                           in any src seq in this batch, for pointer network
    :param log_prob: return log probability instead of probability
    :return: tuple of four things:
             1. word prob or log word prob, (batch size, dynamic vocab size);
             2. RNN hidden state after this step, (1, batch size, decoder hidden size);
             3. attention weights over encoder states, (batch size, src seq len);
             4. prob of copying by pointing as opposed to generating, (batch size, 1)

    Perform single-step decoding.
    """
        batch_size = embedded.size(0)
        combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)
        if self.in_drop:
            embedded = self.in_drop(embedded)
        output, hidden = self.gru(embedded.unsqueeze(0), hidden)
        combined[:, :self.hidden_size] = output.squeeze(0)
        offset = self.hidden_size
        enc_attn, prob_ptr = None, None
        if self.enc_attn or self.pointer:
            num_enc_steps = encoder_states.size(0)
            enc_total_size = encoder_states.size(2)
            enc_energy = self.enc_bilinear(hidden.expand(num_enc_steps,
                batch_size, -1).contiguous(), encoder_states)
            if self.enc_attn_cover and coverage_vector is not None:
                enc_energy += self.cover_weight * torch.log(coverage_vector
                    .transpose(0, 1).unsqueeze(2) + eps)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
            if self.enc_attn:
                enc_context = torch.bmm(encoder_states.permute(1, 2, 0),
                    enc_attn)
                combined[:, offset:offset + enc_total_size
                    ] = enc_context.squeeze(2)
                offset += enc_total_size
            enc_attn = enc_attn.squeeze(2)
        if self.dec_attn:
            if decoder_states is not None and len(decoder_states) > 0:
                dec_energy = self.dec_bilinear(hidden.expand_as(
                    decoder_states).contiguous(), decoder_states)
                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_states.permute(1, 2, 0),
                    dec_attn)
                combined[:, offset:offset + self.hidden_size
                    ] = dec_context.squeeze(2)
            offset += self.hidden_size
        if self.out_drop:
            combined = self.out_drop(combined)
        if self.out_embed_size:
            out_embed = self.pre_out(combined)
        else:
            out_embed = combined
        logits = self.out(out_embed)
        if self.pointer:
            output = torch.zeros(batch_size, ext_vocab_size, device=DEVICE)
            prob_ptr = F.sigmoid(self.ptr(combined))
            prob_gen = 1 - prob_ptr
            gen_output = F.softmax(logits, dim=1)
            output[:, :self.vocab_size] = prob_gen * gen_output
            ptr_output = enc_attn
            output.scatter_add_(1, encoder_word_idx.transpose(0, 1), 
                prob_ptr * ptr_output)
            if log_prob:
                output = torch.log(output + eps)
        elif log_prob:
            output = F.log_softmax(logits, dim=1)
        else:
            output = F.softmax(logits, dim=1)
        return output, hidden, enc_attn, prob_ptr


class Seq2SeqOutput(object):

    def __init__(self, encoder_outputs: torch.Tensor, encoder_hidden: torch
        .Tensor, decoded_tokens: torch.Tensor, loss: Union[torch.Tensor,
        float]=0, loss_value: float=0, enc_attn_weights: torch.Tensor=None,
        ptr_probs: torch.Tensor=None):
        self.encoder_outputs = encoder_outputs
        self.encoder_hidden = encoder_hidden
        self.decoded_tokens = decoded_tokens
        self.loss = loss
        self.loss_value = loss_value
        self.enc_attn_weights = enc_attn_weights
        self.ptr_probs = ptr_probs


class Hypothesis(object):

    def __init__(self, tokens, log_probs, dec_hidden, dec_states,
        enc_attn_weights, num_non_words):
        self.tokens = tokens
        self.log_probs = log_probs
        self.dec_hidden = dec_hidden
        self.dec_states = dec_states
        self.enc_attn_weights = enc_attn_weights
        self.num_non_words = num_non_words

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(self, token, log_prob, dec_hidden, add_dec_states,
        enc_attn, non_word):
        return Hypothesis(tokens=self.tokens + [token], log_probs=self.
            log_probs + [log_prob], dec_hidden=dec_hidden, dec_states=self.
            dec_states + [dec_hidden] if add_dec_states else self.
            dec_states, enc_attn_weights=self.enc_attn_weights + [enc_attn] if
            enc_attn is not None else self.enc_attn_weights, num_non_words=
            self.num_non_words + 1 if non_word else self.num_non_words)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ymfa_seq2seq_summarizer(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(EncoderRNN(*[], **{'embed_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4]), torch.rand([2, 4, 4])], {})
