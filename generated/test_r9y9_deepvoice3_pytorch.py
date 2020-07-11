import sys
_module = sys.modules[__name__]
del sys
audio = _module
compute_timestamp_ratio = _module
deepvoice3_pytorch = _module
builder = _module
conv = _module
deepvoice3 = _module
frontend = _module
en = _module
es = _module
jp = _module
ko = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
modules = _module
nyanko = _module
tfcompat = _module
hparam = _module
dump_hparams_to_json = _module
gentle_web_align = _module
hparams = _module
json_meta = _module
jsut = _module
ljspeech = _module
lrschedule = _module
nikl_m = _module
prepare_metafile = _module
nikl_s = _module
preprocess = _module
setup = _module
synthesis = _module
test_audio = _module
test_conv = _module
test_deepvoice3 = _module
test_embedding = _module
test_frontend = _module
test_nyanko = _module
train = _module
vctk = _module
extract_feats = _module
prepare_htk_alignments_vctk = _module
prepare_vctk_labels = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.nn import functional as F


import math


import numpy as np


from torch import optim


import torch.backends.cudnn as cudnn


from torch.utils import data as data_utils


from torch.utils.data.sampler import Sampler


import random


from matplotlib import pyplot as plt


from matplotlib import cm


from warnings import warn


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


class MultiSpeakerTTSModel(nn.Module):
    """Attention seq2seq model + post processing network
    """

    def __init__(self, seq2seq, postnet, mel_dim=80, linear_dim=513, n_speakers=1, speaker_embed_dim=16, padding_idx=None, trainable_positional_encodings=False, use_decoder_state_for_postnet_input=False, speaker_embedding_weight_std=0.01, freeze_embedding=False):
        super(MultiSpeakerTTSModel, self).__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.trainable_positional_encodings = trainable_positional_encodings
        self.use_decoder_state_for_postnet_input = use_decoder_state_for_postnet_input
        self.freeze_embedding = freeze_embedding
        if n_speakers > 1:
            self.embed_speakers = Embedding(n_speakers, speaker_embed_dim, padding_idx=None, std=speaker_embedding_weight_std)
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)

    def get_trainable_parameters(self):
        freezed_param_ids = set()
        encoder, decoder = self.seq2seq.encoder, self.seq2seq.decoder
        if not self.trainable_positional_encodings:
            pe_query_param_ids = set(map(id, decoder.embed_query_positions.parameters()))
            pe_keys_param_ids = set(map(id, decoder.embed_keys_positions.parameters()))
            freezed_param_ids |= pe_query_param_ids | pe_keys_param_ids
        if self.freeze_embedding:
            embed_param_ids = set(map(id, encoder.embed_tokens.parameters()))
            freezed_param_ids |= embed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, text_sequences, mel_targets=None, speaker_ids=None, text_positions=None, frame_positions=None, input_lengths=None):
        B = text_sequences.size(0)
        if speaker_ids is not None:
            assert self.n_speakers > 1
            speaker_embed = self.embed_speakers(speaker_ids)
        else:
            speaker_embed = None
        mel_outputs, alignments, done, decoder_states = self.seq2seq(text_sequences, mel_targets, speaker_embed, text_positions, frame_positions, input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        if self.use_decoder_state_for_postnet_input:
            postnet_inputs = decoder_states.view(B, mel_outputs.size(1), -1)
        else:
            postnet_inputs = mel_outputs
        linear_outputs = self.postnet(postnet_inputs, speaker_embed)
        assert linear_outputs.size(-1) == self.linear_dim
        return mel_outputs, linear_outputs, alignments, done


class AttentionSeq2Seq(nn.Module):
    """Encoder + Decoder with attention
    """

    def __init__(self, encoder, decoder):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if isinstance(self.decoder.attention, nn.ModuleList):
            self.encoder.num_attention_layers = sum([(layer is not None) for layer in decoder.attention])

    def forward(self, text_sequences, mel_targets=None, speaker_embed=None, text_positions=None, frame_positions=None, input_lengths=None):
        encoder_outputs = self.encoder(text_sequences, lengths=input_lengths, speaker_embed=speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder(encoder_outputs, mel_targets, text_positions=text_positions, frame_positions=frame_positions, speaker_embed=speaker_embed, lengths=input_lengths)
        return mel_outputs, alignments, done, decoder_states


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    m = Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(std_mul * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class HighwayConv1d(nn.Module):
    """Weight normzlized Conv1d + Highway network (support incremental forward)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None, dilation=1, causal=False, dropout=0, std_mul=None, glu=False):
        super(HighwayConv1d, self).__init__()
        if std_mul is None:
            std_mul = 4.0 if glu else 1.0
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.dropout = dropout
        self.glu = glu
        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, dropout=dropout, std_mul=std_mul)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        """Forward

        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :residual.size(-1)] if self.causal else x
        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = torch.sigmoid(b)
            return T * a + (1 - T) * residual

    def clear_buffer(self):
        self.conv.clear_buffer()


class Encoder(nn.Module):

    def __init__(self, n_vocab, embed_dim, channels, kernel_size=3, n_speakers=1, speaker_embed_dim=16, embedding_weight_std=0.01, padding_idx=None, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.embed_tokens = Embedding(n_vocab, embed_dim, padding_idx, embedding_weight_std)
        E = embed_dim
        D = channels
        self.convnet = nn.Sequential(Conv1d(E, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0), nn.ReLU(inplace=True), Conv1d(2 * D, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=2.0), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=3, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=9, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=27, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=3, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=9, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=27, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * D, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0, dropout=dropout))

    def forward(self, text_sequences, text_positions=None, lengths=None, speaker_embed=None):
        x = self.embed_tokens(text_sequences)
        x = self.convnet(x.transpose(1, 2)).transpose(1, 2)
        keys, values = x.split(x.size(-1) // 2, dim=-1)
        return keys, values


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class AttentionLayer(nn.Module):

    def __init__(self, conv_channels, embed_dim, dropout=0.1, window_ahead=3, window_backward=1, key_projection=True, value_projection=True):
        super(AttentionLayer, self).__init__()
        self.query_projection = Linear(conv_channels, embed_dim)
        if key_projection:
            self.key_projection = Linear(embed_dim, embed_dim)
            if conv_channels == embed_dim:
                self.key_projection.weight.data = self.query_projection.weight.data.clone()
        else:
            self.key_projection = None
        if value_projection:
            self.value_projection = Linear(embed_dim, embed_dim)
        else:
            self.value_projection = None
        self.out_projection = Linear(embed_dim, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query
        if self.value_projection is not None:
            values = self.value_projection(values)
        if self.key_projection is not None:
            keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)
        x = self.query_projection(query)
        x = torch.bmm(x, keys)
        mask_value = -float('inf')
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)
        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.bmm(x, values)
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    max_len = max(memory_lengths)
    mask = torch.arange(max_len).expand(memory.size(0), max_len) < torch.tensor(memory_lengths).unsqueeze(-1)
    mask = mask
    return ~mask


def position_encoding_init(n_position, d_pos_vec, position_rate=1.0, sinusoidal=True):
    """ Init the sinusoid position encoding table """
    position_enc = np.array([([(position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec)) for i in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec)) for pos in range(n_position)])
    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])
    return position_enc


class Decoder(nn.Module):

    def __init__(self, embed_dim, in_dim=80, r=5, channels=256, kernel_size=3, n_speakers=1, speaker_embed_dim=16, max_positions=512, padding_idx=None, dropout=0.1, use_memory_mask=False, force_monotonic_attention=False, query_position_rate=1.0, key_position_rate=1.29, window_ahead=3, window_backward=1, key_projection=False, value_projection=False):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        D = channels
        F = in_dim * r
        self.audio_encoder_modules = nn.ModuleList([Conv1d(F, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0), nn.ReLU(inplace=True), Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0), nn.ReLU(inplace=True), Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=1, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=3, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=9, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=27, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=1, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=3, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=9, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=27, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=3, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=3, causal=True, std_mul=1.0, dropout=dropout)])
        self.attention = AttentionLayer(D, D, dropout=dropout, window_ahead=window_ahead, window_backward=window_backward, key_projection=key_projection, value_projection=value_projection)
        self.audio_decoder_modules = nn.ModuleList([Conv1d(2 * D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=1, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=3, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=9, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=27, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=1, causal=True, std_mul=1.0, dropout=dropout), HighwayConv1d(D, D, kernel_size=kernel_size, padding=None, dilation=1, causal=True, std_mul=1.0, dropout=dropout), Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0), nn.ReLU(inplace=True), Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0), nn.ReLU(inplace=True), Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0), nn.ReLU(inplace=True)])
        self.last_conv = Conv1d(D, F, kernel_size=1, padding=0, dilation=1, std_mul=2.0)
        self.fc = Linear(F, 1)
        self.embed_query_positions = Embedding(max_positions, D, padding_idx)
        self.embed_query_positions.weight.data = position_encoding_init(max_positions, D, position_rate=query_position_rate, sinusoidal=True)
        self.embed_keys_positions = Embedding(max_positions, D, padding_idx)
        self.embed_keys_positions.weight.data = position_encoding_init(max_positions, D, position_rate=key_position_rate, sinusoidal=True)
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None, text_positions=None, frame_positions=None, speaker_embed=None, lengths=None):
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions)
            return outputs
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r
        keys, values = encoder_out
        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys = keys + text_pos_embed
        if frame_positions is not None:
            frame_pos_embed = self.embed_query_positions(frame_positions)
        keys = keys.transpose(1, 2).contiguous()
        x = inputs
        x = x.transpose(1, 2)
        for f in self.audio_encoder_modules:
            x = f(x)
        Q = x
        x = x.transpose(1, 2)
        x = x if frame_positions is None else x + frame_pos_embed
        R, alignments = self.attention(x, (keys, values), mask=mask)
        R = R.transpose(1, 2)
        Rd = torch.cat((R, Q), dim=1)
        x = Rd
        for f in self.audio_decoder_modules:
            x = f(x)
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)
        x = x.transpose(1, 2)
        outputs = torch.sigmoid(x)
        done = torch.sigmoid(self.fc(x))
        alignments = alignments.unsqueeze(0)
        return outputs, alignments, done, decoder_states

    def incremental_forward(self, encoder_out, text_positions, initial_input=None, test_inputs=None):
        keys, values = encoder_out
        B = keys.size(0)
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys = keys + text_pos_embed
        keys = keys.transpose(1, 2).contiguous()
        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        last_attended = 0 if self.force_monotonic_attention else None
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim * self.r).zero_()
        current_input = initial_input
        while True:
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            frame_pos_embed = self.embed_query_positions(frame_pos)
            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, (t), :].unsqueeze(1)
            elif t > 0:
                current_input = outputs[-1]
            x = current_input
            for f in self.audio_encoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)
            Q = x
            R, alignment = self.attention(x + frame_pos_embed, (keys, values), last_attended=last_attended)
            if self.force_monotonic_attention:
                last_attended = alignment.max(-1)[1].view(-1).data[0]
            Rd = torch.cat((R, Q), dim=-1)
            x = Rd
            for f in self.audio_decoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)
            decoder_state = x
            x = self.last_conv.incremental_forward(x)
            output = torch.sigmoid(x)
            done = torch.sigmoid(self.fc(x))
            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [alignment]
            dones += [done]
            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))
        alignments = torch.stack(alignments).transpose(0, 1)
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        return outputs, alignments, dones, decoder_states

    def start_fresh_sequence(self):
        _clear_modules(self.audio_encoder_modules)
        _clear_modules(self.audio_decoder_modules)
        self.last_conv.clear_buffer()


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=1.0, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(std_mul * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Converter(nn.Module):

    def __init__(self, in_dim, out_dim, channels=512, kernel_size=3, dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        F = in_dim
        Fd = out_dim
        C = channels
        self.convnet = nn.Sequential(Conv1d(F, C, kernel_size=1, padding=0, dilation=1, std_mul=1.0), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=3, std_mul=1.0, dropout=dropout), ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=1.0), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=3, std_mul=1.0, dropout=dropout), ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=1.0), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(C, C, kernel_size=kernel_size, padding=None, dilation=3, std_mul=1.0, dropout=dropout), Conv1d(C, 2 * C, kernel_size=1, padding=0, dilation=1, std_mul=1.0), HighwayConv1d(2 * C, 2 * C, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), HighwayConv1d(2 * C, 2 * C, kernel_size=kernel_size, padding=None, dilation=1, std_mul=1.0, dropout=dropout), Conv1d(2 * C, Fd, kernel_size=1, padding=0, dilation=1, std_mul=1.0), Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=1.0), nn.ReLU(inplace=True), Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0), nn.ReLU(inplace=True), Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0), nn.Sigmoid())

    def forward(self, x, speaker_embed=None):
        return self.convnet(x.transpose(1, 2)).transpose(1, 2)


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())
    return y


class SinusoidalEncoding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim, *args, padding_idx=0, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim, position_rate=1.0, sinusoidal=False)

    def forward(self, x, w=1.0):
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None
        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(F.embedding(x[batch_idx], weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, n_speakers, speaker_embed_dim, in_channels, out_channels, kernel_size, dropout, padding=None, dilation=1, causal=False, residual=False, *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size, *args, dropout=dropout, padding=padding, dilation=dilation, **kwargs)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None

    def forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, False)

    def incremental_forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :residual.size(-1)] if self.causal else x
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionLayer,
     lambda: ([], {'conv_channels': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), (torch.rand([4, 4, 4]), torch.rand([4, 4, 4]))], {}),
     False),
    (SinusoidalEncoding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_r9y9_deepvoice3_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

