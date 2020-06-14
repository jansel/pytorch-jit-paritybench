import sys
_module = sys.modules[__name__]
del sys
data = _module
batcher = _module
decode_baselines = _module
decode_full_model = _module
decoding = _module
eval_acl = _module
eval_baselines = _module
eval_full_model = _module
evaluate = _module
make_eval_references = _module
make_extraction_labels = _module
metric = _module
model = _module
attention = _module
beam_search = _module
copy_summ = _module
extract = _module
rl = _module
rnn = _module
summ = _module
util = _module
rl = _module
train_abstractor = _module
train_extractor_ml = _module
train_full_rl = _module
train_word2vec = _module
training = _module
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


from torch.nn import functional as F


import torch


from torch import nn


from torch.nn import init


import math


import numpy as np


from torch import autograd


from torch.nn.utils import clip_grad_norm_


from torch import optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import DataLoader


from itertools import starmap


INIT = 0.01


class _CopyLinear(nn.Module):

    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter(None, '_b')

    def forward(self, context, state, input_):
        output = torch.matmul(context, self._v_c.unsqueeze(1)) + torch.matmul(
            state, self._v_s.unsqueeze(1)) + torch.matmul(input_, self._v_i
            .unsqueeze(1))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """

    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i) for i in
            range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2), self._dropout,
            training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0] for conv in
            self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


INI = 0.01


def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers * (2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size
    states = torch.zeros(n_layer, batch_size, n_hidden).to(device
        ), torch.zeros(n_layer, batch_size, n_hidden).to(device)
    return states


def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]
    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = lstm_states[0].index_select(index=order, dim=1
        ), lstm_states[1].index_select(index=order, dim=1)
    return sorted_states


def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)
    return sorted_


def lstm_encoder(sequence, lstm, seq_lens=None, init_states=None, embedding
    =None):
    """ functional LSTM encoder (sequence is [b, t]/[b, t, d],
    lstm should be rolled lstm)"""
    batch_size = sequence.size(0)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)
        emb_sequence = embedding(sequence
            ) if embedding is not None else sequence
    if seq_lens:
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i],
            reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        emb_sequence = reorder_sequence(emb_sequence, sort_ind, lstm.
            batch_first)
    if init_states is None:
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = init_states[0].contiguous(), init_states[1].contiguous()
    if seq_lens:
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, seq_lens)
        packed_out, final_states = lstm(packed_seq, init_states)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(emb_sequence, init_states)
    return lstm_out, final_states


class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer * (2 if
            bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer * (2 if
            bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer, dropout=dropout,
            bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = self._init_h.size(0), input_.size(0), self._init_h.size(1)
        init_states = self._init_h.unsqueeze(1).expand(*size
            ), self._init_c.unsqueeze(1).expand(*size)
        lstm_out, _ = lstm_encoder(input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([(s / l) for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean


class ExtractSumm(nn.Module):
    """ ff-ext """

    def __init__(self, vocab_size, emb_dim, conv_hidden, lstm_hidden,
        lstm_layer, bidirectional, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(vocab_size, emb_dim, conv_hidden,
            dropout)
        self._art_enc = LSTMEncoder(3 * conv_hidden, lstm_hidden,
            lstm_layer, dropout=dropout, bidirectional=bidirectional)
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat([s[:n] for s, n in zip(saliency, sent_nums)],
            dim=0)
        content = self._sent_linear(torch.cat([s[:n] for s, n in zip(
            enc_sent, sent_nums)], dim=0))
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if sent_nums is None:
            assert len(article_sents) == 1
            n_sent = logit.size(1)
            extracted = logit[0].topk(k if k < n_sent else n_sent, sorted=False
                )[1].tolist()
        else:
            extracted = [l[:n].topk(k if k < n else n)[1].tolist() for n, l in
                zip(sent_nums, logit)]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent) for art_sent in article_sents
                ]

            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack([(torch.cat([s, zero(max_n - n, s.device
                )], dim=0) if n != max_n else s) for s, n in zip(enc_sents,
                sent_nums)], dim=0)
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = F.tanh(self._art_linear(sequence_mean(lstm_out, sent_nums,
            dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""

    def __init__(self, emb_dim, vocab_size, conv_hidden, lstm_hidden,
        lstm_layer, bidirectional, n_hop=1, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(vocab_size, emb_dim, conv_hidden,
            dropout)
        self._art_enc = LSTMEncoder(3 * conv_hidden, lstm_hidden,
            lstm_layer, dropout=dropout, bidirectional=bidirectional)
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(enc_out_dim, lstm_hidden,
            lstm_layer, dropout, n_hop)

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(enc_out, dim=1, index=target.unsqueeze(2).
            expand(bs, nt, d))
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent) for art_sent in article_sents
                ]

            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack([(torch.cat([s, zero(max_n - n, s.device
                )], dim=0) if n != max_n else s) for s, n in zip(enc_sents,
                sent_nums)], dim=0)
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""

    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = self._init_h.unsqueeze(1), self._init_c.unsqueeze(1)
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, (-1), :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query, self.
                    _hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(attn_feat, query, self.
                _attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e+18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = h, c
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(PtrExtractorRL.attention_score(attention, query,
            v, w), dim=-1)
        output = torch.mm(score, attention)
        return output


class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""

    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = self._init_h.unsqueeze(1), self._init_c.unsqueeze(1)
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, (-1), :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query, self
                    ._hop_v, self._hop_wq)
            output = PtrScorer.attention(attn_mem, attn_feat, query, self.
                _attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(F.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output


class PtrExtractorRLStop(PtrExtractorRL):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
        self._stop = nn.Parameter(torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, n_ext=None):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = self._init_h.unsqueeze(1), self._init_c.unsqueeze(1)
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, (-1), :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query, self.
                    _hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(attn_feat, query, self.
                _attn_v, self._attn_wq)
            for o in outputs:
                score[0, o.item()] = -1e+18
            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == max_step:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = h, c
        if dists:
            return outputs, dists
        else:
            return outputs


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""

    def __init__(self, sent_encoder, art_encoder, extractor, art_batcher):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher

    def forward(self, raw_article_sents, n_abs=None):
        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._art_enc(enc_sent).squeeze(0)
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_art)
        else:
            outputs = self._ext(enc_art, n_abs)
        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_art, n_abs)
            return outputs, scores
        else:
            return outputs


class StackedLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""

    def __init__(self, cells, dropout=0.0):
        super().__init__()
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout

    def forward(self, input_, state):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = state[0][(i), :, :], state[1][(i), :, :]
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)
        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)
        return new_h, new_c

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return self._cells[0].bidirectional


def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))


def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e+18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score


def step_attention(query, key, value, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)


class AttentionalLSTMDecoder(object):

    def __init__(self, embedding, lstm, attn_w, projection):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm
        self._attn_w = attn_w
        self._projection = projection

    def __call__(self, attention, init_states, target):
        max_len = target.size(1)
        states = init_states
        logits = []
        for i in range(max_len):
            tok = target[:, i:i + 1]
            logit, states, _ = self._step(tok, states, attention)
            logits.append(logit)
        logit = torch.stack(logits, dim=1)
        return logit

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), prev_out], dim=1)
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask = attention
        context, score = step_attention(query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        states = states, dec_out
        logit = torch.mm(dec_out, self._embedding.weight.t())
        return logit, states, score

    def decode_step(self, tok, states, attention):
        logit, states, score = self._step(tok, states, attention)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score


def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[(i), :l].fill_(1)
    return mask


class Seq2SeqSumm(nn.Module):

    def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional,
        n_layer, dropout=0.0):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(emb_dim, n_hidden, n_layer, bidirectional=
            bidirectional, dropout=dropout)
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(torch.Tensor(state_layer, n_hidden))
        self._init_enc_c = nn.Parameter(torch.Tensor(state_layer, n_hidden))
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)
        self._dec_lstm = MultiLayerLSTMCells(2 * emb_dim, n_hidden, n_layer,
            dropout=dropout)
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        self._projection = nn.Sequential(nn.Linear(2 * n_hidden, n_hidden),
            nn.Tanh(), nn.Linear(n_hidden, emb_dim, bias=False))
        self._decoder = AttentionalLSTMDecoder(self._embedding, self.
            _dec_lstm, self._attn_wq, self._projection)

    def forward(self, article, art_lens, abstract):
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder((attention, mask), init_dec_states, abstract)
        return logit

    def encode(self, article, art_lens=None):
        size = self._init_enc_h.size(0), len(art_lens
            ) if art_lens else 1, self._init_enc_h.size(1)
        init_enc_states = self._init_enc_h.unsqueeze(1).expand(*size
            ), self._init_enc_c.unsqueeze(1).expand(*size)
        enc_art, final_states = lstm_encoder(article, self._enc_lstm,
            art_lens, init_enc_states, self._embedding)
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = torch.cat(h.chunk(2, dim=0), dim=2), torch.cat(c
                .chunk(2, dim=0), dim=2)
        init_h = torch.stack([self._dec_h(s) for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s) for s in final_states[1]], dim=0)
        init_dec_states = init_h, init_c
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat([init_h[-1],
            sequence_mean(attention, art_lens, dim=1)], dim=1))
        return attention, (init_dec_states, init_attn_out)

    def batch_decode(self, article, art_lens, go, eos, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = attention, mask
        tok = torch.LongTensor([go] * batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(tok, states,
                attention)
            outputs.append(tok[:, (0)])
            attns.append(attn_score)
        return outputs, attns

    def decode(self, article, go, eos, max_len):
        attention, init_dec_states = self.encode(article)
        attention = attention, None
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(tok, states,
                attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ChenRocks_fast_abs_rl(_paritybench_base):
    pass
    def test_000(self):
        self._check(_CopyLinear(*[], **{'context_dim': 4, 'state_dim': 4, 'input_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

