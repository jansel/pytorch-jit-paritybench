import sys
_module = sys.modules[__name__]
del sys
extract_features_detectron = _module
evaluate = _module
setup = _module
train = _module
visdialch = _module
data = _module
dataset = _module
readers = _module
vocabulary = _module
decoders = _module
disc = _module
gen = _module
encoders = _module
lf = _module
metrics = _module
model = _module
utils = _module
checkpointing = _module
dynamic_rnn = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch import nn


from torch.utils.data import DataLoader


import itertools


from torch import optim


from torch.optim import lr_scheduler


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from torch.nn.functional import normalize


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


import copy


from typing import Set


from typing import Union


from torch.nn import functional as F


import warnings


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


class DynamicRNN(nn.Module):

    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            Single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(sorted_seq_input, lengths=sorted_len, batch_first=True)
        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None
        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)
        outputs = pad_packed_sequence(outputs, batch_first=True, total_length=max_sequence_length)
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order


class DiscriminativeDecoder(nn.Module):

    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.word_embed = nn.Embedding(len(vocabulary), config['word_embedding_size'], padding_idx=vocabulary.PAD_INDEX)
        self.option_rnn = nn.LSTM(config['word_embedding_size'], config['lstm_hidden_size'], config['lstm_num_layers'], batch_first=True, dropout=config['dropout'])
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, batch):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """
        options = batch['opt']
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)
        options_length = batch['opt_len']
        options_length = options_length.view(batch_size * num_rounds * num_options)
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]
        nonzero_options_embed = self.word_embed(nonzero_options)
        _, (nonzero_options_embed, _) = self.option_rnn(nonzero_options_embed, nonzero_options_length)
        options_embed = torch.zeros(batch_size * num_rounds * num_options, nonzero_options_embed.size(-1), device=nonzero_options_embed.device)
        options_embed[nonzero_options_length_indices] = nonzero_options_embed
        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1)
        encoder_output = encoder_output.view(batch_size * num_rounds * num_options, self.config['lstm_hidden_size'])
        scores = torch.sum(options_embed * encoder_output, 1)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores


class GenerativeDecoder(nn.Module):

    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.word_embed = nn.Embedding(len(vocabulary), config['word_embedding_size'], padding_idx=vocabulary.PAD_INDEX)
        self.answer_rnn = nn.LSTM(config['word_embedding_size'], config['lstm_hidden_size'], config['lstm_num_layers'], batch_first=True, dropout=config['dropout'])
        self.lstm_to_words = nn.Linear(self.config['lstm_hidden_size'], len(vocabulary))
        self.dropout = nn.Dropout(p=config['dropout'])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, batch):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.

        During evaluation, assign log-likelihood scores to all answer options.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """
        if self.training:
            ans_in = batch['ans_in']
            batch_size, num_rounds, max_sequence_length = ans_in.size()
            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)
            ans_in_embed = self.word_embed(ans_in)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(self.config['lstm_num_layers'], 1, 1)
            init_cell = torch.zeros_like(init_hidden)
            ans_out, (hidden, cell) = self.answer_rnn(ans_in_embed, (init_hidden, init_cell))
            ans_out = self.dropout(ans_out)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores
        else:
            ans_in = batch['opt_in']
            batch_size, num_rounds, num_options, max_sequence_length = ans_in.size()
            ans_in = ans_in.view(batch_size * num_rounds * num_options, max_sequence_length)
            ans_in_embed = self.word_embed(ans_in)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(1, batch_size * num_rounds * num_options, -1)
            init_hidden = init_hidden.repeat(self.config['lstm_num_layers'], 1, 1)
            init_cell = torch.zeros_like(init_hidden)
            ans_out, (hidden, cell) = self.answer_rnn(ans_in_embed, (init_hidden, init_cell))
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))
            target_ans_out = batch['opt_out'].view(batch_size * num_rounds * num_options, -1)
            ans_word_scores = torch.gather(ans_word_scores, -1, target_ans_out.unsqueeze(-1)).squeeze()
            ans_word_scores = ans_word_scores * (target_ans_out > 0).float()
            ans_scores = torch.sum(ans_word_scores, -1)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)
            return ans_scores


class LateFusionEncoder(nn.Module):

    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.word_embed = nn.Embedding(len(vocabulary), config['word_embedding_size'], padding_idx=vocabulary.PAD_INDEX)
        self.hist_rnn = nn.LSTM(config['word_embedding_size'], config['lstm_hidden_size'], config['lstm_num_layers'], batch_first=True, dropout=config['dropout'])
        self.ques_rnn = nn.LSTM(config['word_embedding_size'], config['lstm_hidden_size'], config['lstm_num_layers'], batch_first=True, dropout=config['dropout'])
        self.dropout = nn.Dropout(p=config['dropout'])
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        self.image_features_projection = nn.Linear(config['img_feature_size'], config['lstm_hidden_size'])
        self.attention_proj = nn.Linear(config['lstm_hidden_size'], 1)
        fusion_size = config['img_feature_size'] + config['lstm_hidden_size'] * 2
        self.fusion = nn.Linear(fusion_size, config['lstm_hidden_size'])
        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        img = batch['img_feat']
        ques = batch['ques']
        hist = batch['hist']
        batch_size, num_rounds, max_sequence_length = ques.size()
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)
        _, (ques_embed, _) = self.ques_rnn(ques_embed, batch['ques_len'])
        projected_image_features = self.image_features_projection(img)
        projected_image_features = projected_image_features.view(batch_size, 1, -1, self.config['lstm_hidden_size']).repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.config['lstm_hidden_size'])
        projected_ques_features = ques_embed.unsqueeze(1).repeat(1, img.shape[1], 1)
        projected_ques_image = projected_ques_features * projected_image_features
        projected_ques_image = self.dropout(projected_ques_image)
        image_attention_weights = self.attention_proj(projected_ques_image).squeeze()
        image_attention_weights = F.softmax(image_attention_weights, dim=-1)
        img = img.view(batch_size, 1, -1, self.config['img_feature_size']).repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.config['img_feature_size'])
        image_attention_weights = image_attention_weights.unsqueeze(-1).repeat(1, 1, self.config['img_feature_size'])
        attended_image_features = (image_attention_weights * img).sum(1)
        img = attended_image_features
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed = self.word_embed(hist)
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch['hist_len'])
        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output

