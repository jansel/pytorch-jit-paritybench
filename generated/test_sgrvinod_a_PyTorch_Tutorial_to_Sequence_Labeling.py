import sys
_module = sys.modules[__name__]
del sys
datasets = _module
dynamic_rnn = _module
inference = _module
models = _module
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


from torch import nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn as nn


import torch.optim as optim


from collections import Counter


import itertools


from functools import reduce


import numpy as np


import torch.nn.init


class Highway(nn.Module):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        """
        :param size: size of linear layer (matches input size)
        :param num_layers: number of transform and gate layers
        :param dropout: dropout
        """
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_layers):
            transform = nn.Linear(size, size)
            gate = nn.Linear(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def forward(self, x):
        """
        Forward propagation.

        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = nn.functional.relu(self.transform[0](x))
        g = nn.functional.sigmoid(self.gate[0](x))
        out = g * transformed + (1 - g) * x
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = nn.functional.relu(self.transform[i](out))
            g = nn.functional.sigmoid(self.gate[i](out))
            out = g * transformed + (1 - g) * out
        return out


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.
            tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)
        emission_scores = self.emission(feats)
        emission_scores = emission_scores.unsqueeze(2).expand(self.
            batch_size, self.timesteps, self.tagset_size, self.tagset_size)
        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(0
            )
        return crf_scores


class LM_LSTM_CRF(nn.Module):
    """
    The encompassing LM-LSTM-CRF model.
    """

    def __init__(self, tagset_size, charset_size, char_emb_dim,
        char_rnn_dim, char_rnn_layers, vocab_size, lm_vocab_size,
        word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1
        ):
        """
        :param tagset_size: number of tags
        :param charset_size: size of character vocabulary
        :param char_emb_dim: size of character embeddings
        :param char_rnn_dim: size of character RNNs/LSTMs
        :param char_rnn_layers: number of layers in character RNNs/LSTMs
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        :param highway_layers: number of transform and gate layers
        """
        super(LM_LSTM_CRF, self).__init__()
        self.tagset_size = tagset_size
        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_rnn_dim = char_rnn_dim
        self.char_rnn_layers = char_rnn_layers
        self.wordset_size = vocab_size
        self.lm_vocab_size = lm_vocab_size
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers
        self.highway_layers = highway_layers
        self.dropout = nn.Dropout(p=dropout)
        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim,
            num_layers=self.char_rnn_layers, bidirectional=False, dropout=
            dropout)
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim,
            num_layers=self.char_rnn_layers, bidirectional=False, dropout=
            dropout)
        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)
        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_rnn_dim * 2,
            self.word_rnn_dim // 2, num_layers=self.word_rnn_layers,
            bidirectional=True, dropout=dropout)
        self.crf = CRF(self.word_rnn_dim // 2 * 2, self.tagset_size)
        self.forw_lm_hw = Highway(self.char_rnn_dim, num_layers=self.
            highway_layers, dropout=dropout)
        self.back_lm_hw = Highway(self.char_rnn_dim, num_layers=self.
            highway_layers, dropout=dropout)
        self.subword_hw = Highway(2 * self.char_rnn_dim, num_layers=self.
            highway_layers, dropout=dropout)
        self.forw_lm_out = nn.Linear(self.char_rnn_dim, self.lm_vocab_size)
        self.back_lm_out = nn.Linear(self.char_rnn_dim, self.lm_vocab_size)

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps,
        tmaps, wmap_lengths, cmap_lengths):
        """
        Forward propagation.

        :param cmaps_f: padded encoded forward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmaps_b: padded encoded backward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmarkers_f: padded forward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param cmarkers_b: padded backward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = cmaps_f.size(0)
        self.word_pad_len = wmaps.size(1)
        cmap_lengths, char_sort_ind = cmap_lengths.sort(dim=0, descending=True)
        cmaps_f = cmaps_f[char_sort_ind]
        cmaps_b = cmaps_b[char_sort_ind]
        cmarkers_f = cmarkers_f[char_sort_ind]
        cmarkers_b = cmarkers_b[char_sort_ind]
        wmaps = wmaps[char_sort_ind]
        tmaps = tmaps[char_sort_ind]
        wmap_lengths = wmap_lengths[char_sort_ind]
        cf = self.char_embeds(cmaps_f)
        cb = self.char_embeds(cmaps_b)
        cf = self.dropout(cf)
        cb = self.dropout(cb)
        cf = pack_padded_sequence(cf, cmap_lengths.tolist(), batch_first=True)
        cb = pack_padded_sequence(cb, cmap_lengths.tolist(), batch_first=True)
        cf, _ = self.forw_char_lstm(cf)
        cb, _ = self.back_char_lstm(cb)
        cf, _ = pad_packed_sequence(cf, batch_first=True)
        cb, _ = pad_packed_sequence(cb, batch_first=True)
        assert cf.size(1) == max(cmap_lengths.tolist()) == list(cmap_lengths)[0
            ]
        cmarkers_f = cmarkers_f.unsqueeze(2).expand(self.batch_size, self.
            word_pad_len, self.char_rnn_dim)
        cmarkers_b = cmarkers_b.unsqueeze(2).expand(self.batch_size, self.
            word_pad_len, self.char_rnn_dim)
        cf_selected = torch.gather(cf, 1, cmarkers_f)
        cb_selected = torch.gather(cb, 1, cmarkers_b)
        if self.training:
            lm_f = self.forw_lm_hw(self.dropout(cf_selected))
            lm_b = self.back_lm_hw(self.dropout(cb_selected))
            lm_f_scores = self.forw_lm_out(self.dropout(lm_f))
            lm_b_scores = self.back_lm_out(self.dropout(lm_b))
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        tmaps = tmaps[word_sort_ind]
        cf_selected = cf_selected[word_sort_ind]
        cb_selected = cb_selected[word_sort_ind]
        if self.training:
            lm_f_scores = lm_f_scores[word_sort_ind]
            lm_b_scores = lm_b_scores[word_sort_ind]
        w = self.word_embeds(wmaps)
        w = self.dropout(w)
        subword = self.subword_hw(self.dropout(torch.cat((cf_selected,
            cb_selected), dim=2)))
        subword = self.dropout(subword)
        w = torch.cat((w, subword), dim=2)
        w = pack_padded_sequence(w, list(wmap_lengths), batch_first=True)
        w, _ = self.word_blstm(w)
        w, _ = pad_packed_sequence(w, batch_first=True)
        w = self.dropout(w)
        crf_scores = self.crf(w)
        if self.training:
            return (crf_scores, lm_f_scores, lm_b_scores, wmaps, tmaps,
                wmap_lengths, word_sort_ind, char_sort_ind)
        else:
            return (crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind,
                char_sort_ind)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def forward(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)
        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size,
            word_pad_len, -1), 2, targets).squeeze(2)
        scores_at_targets, _ = pack_padded_sequence(scores_at_targets,
            lengths, batch_first=True)
        gold_score = scores_at_targets.sum()
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)
        for t in range(max(lengths)):
            batch_size_t = sum([(l > t) for l in lengths])
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, (t), (
                    self.start_tag), :]
            else:
                scores_upto_t[:batch_size_t] = log_sum_exp(scores[:
                    batch_size_t, (t), :, :] + scores_upto_t[:batch_size_t]
                    .unsqueeze(2), dim=1)
        all_paths_scores = scores_upto_t[:, (self.end_tag)].sum()
        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size
        return viterbi_loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sgrvinod_a_PyTorch_Tutorial_to_Sequence_Labeling(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Highway(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CRF(*[], **{'hidden_dim': 4, 'tagset_size': 4}), [torch.rand([4, 4])], {})

