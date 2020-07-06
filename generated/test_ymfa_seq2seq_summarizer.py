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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import random


from typing import Union


from typing import List


from typing import Dict


from typing import Tuple


from typing import Optional


import math


from torch import optim


from torch.nn.utils import clip_grad_norm_


import re


import numpy as np


from typing import NamedTuple


from typing import Callable


from collections import Counter


from random import shuffle


from functools import lru_cache


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidi else 1
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

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
            hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size, self.hidden_size * self.num_directions)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)


eps = 1e-31


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, *, enc_attn=True, dec_attn=True, enc_attn_cover=True, pointer=True, tied_embedding=None, out_embed_size=None, in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.out_embed_size = out_embed_size
        if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
            None
            self.out_embed_size = embed_size
        self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
        self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)
        if enc_attn:
            if not enc_hidden_size:
                enc_hidden_size = self.hidden_size
            self.enc_bilinear = nn.Bilinear(self.hidden_size, enc_hidden_size, 1)
            self.combined_size += enc_hidden_size
            if enc_attn_cover:
                self.cover_weight = nn.Parameter(torch.rand(1))
        if dec_attn:
            self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
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

    def forward(self, embedded, hidden, encoder_states=None, decoder_states=None, coverage_vector=None, *, encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True):
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
            enc_energy = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(), encoder_states)
            if self.enc_attn_cover and coverage_vector is not None:
                enc_energy += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
            if self.enc_attn:
                enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn)
                combined[:, offset:offset + enc_total_size] = enc_context.squeeze(2)
                offset += enc_total_size
            enc_attn = enc_attn.squeeze(2)
        if self.dec_attn:
            if decoder_states is not None and len(decoder_states) > 0:
                dec_energy = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(), decoder_states)
                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
                combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
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
            output.scatter_add_(1, encoder_word_idx.transpose(0, 1), prob_ptr * ptr_output)
            if log_prob:
                output = torch.log(output + eps)
        elif log_prob:
            output = F.log_softmax(logits, dim=1)
        else:
            output = F.softmax(logits, dim=1)
        return output, hidden, enc_attn, prob_ptr


class Hypothesis(object):

    def __init__(self, tokens, log_probs, dec_hidden, dec_states, enc_attn_weights, num_non_words):
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

    def create_next(self, token, log_prob, dec_hidden, add_dec_states, enc_attn, non_word):
        return Hypothesis(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob], dec_hidden=dec_hidden, dec_states=self.dec_states + [dec_hidden] if add_dec_states else self.dec_states, enc_attn_weights=self.enc_attn_weights + [enc_attn] if enc_attn is not None else self.enc_attn_weights, num_non_words=self.num_non_words + 1 if non_word else self.num_non_words)


class Params:
    vocab_size: int = 30000
    hidden_size: int = 150
    dec_hidden_size: Optional[int] = 200
    embed_size: int = 100
    enc_bidi: bool = True
    enc_attn: bool = True
    dec_attn: bool = False
    pointer: bool = True
    out_embed_size: Optional[int] = None
    tie_embed: bool = True
    enc_attn_cover: bool = True
    cover_func: str = 'max'
    cover_loss: float = 1
    show_cover_loss: bool = False
    enc_rnn_dropout: float = 0
    dec_in_dropout: float = 0
    dec_rnn_dropout: float = 0
    dec_out_dropout: float = 0
    optimizer: str = 'adam'
    lr: float = 0.001
    adagrad_accumulator: float = 0.1
    lr_decay_step: int = 5
    lr_decay: Optional[float] = None
    batch_size: int = 32
    n_batches: int = 1000
    val_batch_size: int = 32
    n_val_batches: int = 100
    n_epochs: int = 75
    pack_seq: bool = True
    forcing_ratio: float = 0.75
    partial_forcing: bool = True
    forcing_decay_type: Optional[str] = 'exp'
    forcing_decay: float = 0.9999
    sample: bool = True
    grad_norm: float = 1
    rl_ratio: float = 0
    rl_ratio_power: float = 1
    rl_start_epoch: int = 1
    embed_file: Optional[str] = 'data/.vector_cache/glove.6B.100d.txt'
    data_path: str = 'data/cnndm.gz'
    val_data_path: Optional[str] = 'data/cnndm.val.gz'
    max_src_len: int = 400
    max_tgt_len: int = 100
    truncate_src: bool = True
    truncate_tgt: bool = True
    model_path_prefix: Optional[str] = 'checkpoints/cnndm05'
    keep_every_epoch: bool = False
    beam_size: int = 4
    min_out_len: int = 60
    max_out_len: Optional[int] = 100
    out_len_in_words: bool = False
    test_data_path: str = 'data/cnndm.test.gz'
    test_sample_ratio: float = 1
    test_save_results: bool = False

    def update(self, cmd_args: List[str]):
        """Update configuration by a list of command line arguments"""
        arg_name = None
        for arg_text in cmd_args:
            if arg_name is None:
                assert arg_text.startswith('--')
                arg_name = arg_text[2:]
            else:
                arg_curr_value = getattr(self, arg_name)
                if arg_text.lower() == 'none':
                    arg_new_value = None
                elif arg_text.lower() == 'true':
                    arg_new_value = True
                elif arg_text.lower() == 'false':
                    arg_new_value = False
                else:
                    arg_type = self.__annotations__[arg_name]
                    if type(arg_type) is not type:
                        assert arg_type.__origin__ is Union
                        arg_types = [t for t in arg_type.__args__ if t is not type(None)]
                        assert len(arg_types) == 1
                        arg_type = arg_types[0]
                        assert type(arg_type) is type
                    arg_new_value = arg_type(arg_text)
                setattr(self, arg_name, arg_new_value)
                None
                arg_name = None
        if arg_name is not None:
            None


class Seq2SeqOutput(object):

    def __init__(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, decoded_tokens: torch.Tensor, loss: Union[torch.Tensor, float]=0, loss_value: float=0, enc_attn_weights: torch.Tensor=None, ptr_probs: torch.Tensor=None):
        self.encoder_outputs = encoder_outputs
        self.encoder_hidden = encoder_hidden
        self.decoded_tokens = decoded_tokens
        self.loss = loss
        self.loss_value = loss_value
        self.enc_attn_weights = enc_attn_weights
        self.ptr_probs = ptr_probs


word_detector = re.compile('\\w')


class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words: List[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def trim(self, *, vocab_size: int=None, min_freq: int=1):
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.word2index)):
            return
        ordered_words = sorted(((c, w) for w, c in self.word2count.items()), reverse=True)
        if vocab_size:
            ordered_words = ordered_words[:vocab_size]
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = self.reserved[:]
        for count, word in ordered_words:
            if count < min_freq:
                break
            self.word2index[word] = len(self.index2word)
            self.word2count[word] = count
            self.index2word.append(word)

    def load_embeddings(self, file_path: str, dtype=np.float32) ->int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) ->bool:
        """Return whether the token at `token_id` is a word; False for punctuations."""
        if token_id < 4:
            return False
        if token_id >= len(self):
            return True
        token_str = self.index2word[token_id]
        if not word_detector.search(token_str) or token_str == '<P>':
            return False
        return True


class Seq2Seq(nn.Module):

    def __init__(self, vocab: Vocab, params: Params, max_dec_steps=None):
        """
    :param vocab: mainly for info about special tokens and vocab size
    :param params: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the seq2seq model; its encoder and decoder will be created automatically.
    """
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        if vocab.embeddings is not None:
            self.embed_size = vocab.embeddings.shape[1]
            if params.embed_size is not None and self.embed_size != params.embed_size:
                None
            embedding_weights = torch.from_numpy(vocab.embeddings)
        else:
            self.embed_size = params.embed_size
            embedding_weights = None
        self.max_dec_steps = params.max_tgt_len + 1 if max_dec_steps is None else max_dec_steps
        self.enc_attn = params.enc_attn
        self.enc_attn_cover = params.enc_attn_cover
        self.dec_attn = params.dec_attn
        self.pointer = params.pointer
        self.cover_loss = params.cover_loss
        self.cover_func = params.cover_func
        enc_total_size = params.hidden_size * 2 if params.enc_bidi else params.hidden_size
        if params.dec_hidden_size:
            dec_hidden_size = params.dec_hidden_size
            self.enc_dec_adapter = nn.Linear(enc_total_size, dec_hidden_size)
        else:
            dec_hidden_size = enc_total_size
            self.enc_dec_adapter = None
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD, _weight=embedding_weights)
        self.encoder = EncoderRNN(self.embed_size, params.hidden_size, params.enc_bidi, rnn_drop=params.enc_rnn_dropout)
        self.decoder = DecoderRNN(self.vocab_size, self.embed_size, dec_hidden_size, enc_attn=params.enc_attn, dec_attn=params.dec_attn, pointer=params.pointer, out_embed_size=params.out_embed_size, tied_embedding=self.embedding if params.tie_embed else None, in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout, out_drop=params.dec_out_dropout, enc_hidden_size=enc_total_size)

    def filter_oov(self, tensor, ext_vocab_size):
        """Replace any OOV index in `tensor` with UNK"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = self.vocab.UNK
            return result
        return tensor

    def get_coverage_vector(self, enc_attn_weights):
        """Combine the past attention weights into one vector"""
        if self.cover_func == 'max':
            coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
        elif self.cover_func == 'sum':
            coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
        else:
            raise ValueError('Unrecognized cover_func: ' + self.cover_func)
        return coverage_vector

    def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *, forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False, saved_out: Seq2SeqOutput=None, visualize: bool=None, include_cover_loss: bool=False) ->Seq2SeqOutput:
        """
    :param input_tensor: tensor of word indices, (src seq len, batch size)
    :param target_tensor: tensor of word indices, (tgt seq len, batch size)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the seq2seq model for training or testing.
    """
        input_length = input_tensor.size(0)
        batch_size = input_tensor.size(1)
        log_prob = not (sample or self.decoder.pointer)
        if visualize is None:
            visualize = criterion is None
        if visualize and not (self.enc_attn or self.pointer):
            visualize = False
        if target_tensor is None:
            target_length = self.max_dec_steps
        else:
            target_length = target_tensor.size(0)
        if forcing_ratio == 1:
            use_teacher_forcing = True
        elif forcing_ratio > 0:
            if partial_forcing:
                use_teacher_forcing = None
            else:
                use_teacher_forcing = random.random() < forcing_ratio
        else:
            use_teacher_forcing = False
        if saved_out:
            encoder_outputs = saved_out.encoder_outputs
            encoder_hidden = saved_out.encoder_hidden
            assert input_length == encoder_outputs.size(0)
            assert batch_size == encoder_outputs.size(1)
        else:
            encoder_hidden = self.encoder.init_hidden(batch_size)
            encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
            encoder_outputs, encoder_hidden = self.encoder(encoder_embedded, encoder_hidden, input_lengths)
        r = Seq2SeqOutput(encoder_outputs, encoder_hidden, torch.zeros(target_length, batch_size, dtype=torch.long))
        if visualize:
            r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
            if self.pointer:
                r.ptr_probs = torch.zeros(target_length, batch_size)
        decoder_input = torch.tensor([self.vocab.SOS] * batch_size, device=DEVICE)
        if self.enc_dec_adapter is None:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = self.enc_dec_adapter(encoder_hidden)
        decoder_states = []
        enc_attn_weights = []
        for di in range(target_length):
            decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = self.decoder(decoder_embedded, decoder_hidden, encoder_outputs, torch.cat(decoder_states) if decoder_states else None, coverage_vector, encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size, log_prob=log_prob)
            if self.dec_attn:
                decoder_states.append(decoder_hidden)
            if not sample:
                _, top_idx = decoder_output.data.topk(1)
            else:
                prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
                top_idx = torch.multinomial(prob_distribution, 1)
            top_idx = top_idx.squeeze(1).detach()
            r.decoded_tokens[di] = top_idx
            if criterion:
                if target_tensor is None:
                    gold_standard = top_idx
                else:
                    gold_standard = target_tensor[di]
                if not log_prob:
                    decoder_output = torch.log(decoder_output + eps)
                nll_loss = criterion(decoder_output, gold_standard)
                r.loss += nll_loss
                r.loss_value += nll_loss.item()
            if self.enc_attn_cover or criterion and self.cover_loss > 0:
                if coverage_vector is not None and criterion and self.cover_loss > 0:
                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
                    r.loss += coverage_loss
                    if include_cover_loss:
                        r.loss_value += coverage_loss.item()
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
            if visualize:
                r.enc_attn_weights[di] = dec_enc_attn.data
                if self.pointer:
                    r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data
            if use_teacher_forcing or use_teacher_forcing is None and random.random() < forcing_ratio:
                decoder_input = target_tensor[di]
            else:
                decoder_input = top_idx
        return r

    def beam_search(self, input_tensor, input_lengths=None, ext_vocab_size=None, beam_size=4, *, min_out_len=1, max_out_len=None, len_in_words=True) ->List[Hypothesis]:
        """
    :param input_tensor: tensor of word indices, (src seq len, batch size); for now, batch size has
                         to be 1
    :param input_lengths: see explanation in `EncoderRNN`
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param beam_size: the beam size
    :param min_out_len: required minimum output length
    :param max_out_len: required maximum output length (if None, use the model's own value)
    :param len_in_words: if True, count output length in words instead of tokens (i.e. do not count
                         punctuations)
    :return: list of the best decoded sequences, in descending order of probability

    Use beam search to generate summaries.
    """
        batch_size = input_tensor.size(1)
        assert batch_size == 1
        if max_out_len is None:
            max_out_len = self.max_dec_steps - 1
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
        encoder_outputs, encoder_hidden = self.encoder(encoder_embedded, encoder_hidden, input_lengths)
        if self.enc_dec_adapter is None:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = self.enc_dec_adapter(encoder_hidden)
        encoder_outputs = encoder_outputs.expand(-1, beam_size, -1).contiguous()
        input_tensor = input_tensor.expand(-1, beam_size).contiguous()
        hypos = [Hypothesis([self.vocab.SOS], [], decoder_hidden, [], [], 1)]
        results, backup_results = [], []
        step = 0
        while hypos and step < 2 * max_out_len:
            n_hypos = len(hypos)
            if n_hypos < beam_size:
                hypos.extend(hypos[-1] for _ in range(beam_size - n_hypos))
            decoder_input = torch.tensor([h.tokens[-1] for h in hypos], device=DEVICE)
            decoder_hidden = torch.cat([h.dec_hidden for h in hypos], 1)
            if self.dec_attn and step > 0:
                decoder_states = torch.cat([torch.cat(h.dec_states, 0) for h in hypos], 1)
            else:
                decoder_states = None
            if self.enc_attn_cover:
                enc_attn_weights = [torch.cat([h.enc_attn_weights[i] for h in hypos], 1) for i in range(step)]
            else:
                enc_attn_weights = []
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
            decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = self.decoder(decoder_embedded, decoder_hidden, encoder_outputs, decoder_states, coverage_vector, encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size)
            top_v, top_i = decoder_output.data.topk(beam_size)
            new_hypos = []
            for in_idx in range(n_hypos):
                for out_idx in range(beam_size):
                    new_tok = top_i[in_idx][out_idx].item()
                    new_prob = top_v[in_idx][out_idx].item()
                    if len_in_words:
                        non_word = not self.vocab.is_word(new_tok)
                    else:
                        non_word = new_tok == self.vocab.EOS
                    new_hypo = hypos[in_idx].create_next(new_tok, new_prob, decoder_hidden[0][in_idx].unsqueeze(0).unsqueeze(0), self.dec_attn, dec_enc_attn[in_idx].unsqueeze(0).unsqueeze(0) if dec_enc_attn is not None else None, non_word)
                    new_hypos.append(new_hypo)
            new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)
            hypos = []
            new_complete_results, new_incomplete_results = [], []
            for nh in new_hypos:
                length = len(nh)
                if nh.tokens[-1] == self.vocab.EOS:
                    if len(new_complete_results) < beam_size and min_out_len <= length <= max_out_len:
                        new_complete_results.append(nh)
                elif len(hypos) < beam_size and length < max_out_len:
                    hypos.append(nh)
                elif length == max_out_len and len(new_incomplete_results) < beam_size:
                    new_incomplete_results.append(nh)
            if new_complete_results:
                results.extend(new_complete_results)
            elif new_incomplete_results:
                backup_results.extend(new_incomplete_results)
            step += 1
        if not results:
            results = backup_results
        return sorted(results, key=lambda h: -h.avg_log_prob)[:beam_size]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderRNN,
     lambda: ([], {'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([2, 4, 4])], {}),
     False),
]

class Test_ymfa_seq2seq_summarizer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

