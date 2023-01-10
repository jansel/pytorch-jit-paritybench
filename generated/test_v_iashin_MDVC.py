import sys
_module = sys.modules[__name__]
del sys
dataset = _module
run_epoch = _module
evaluate = _module
loss = _module
main = _module
transformer = _module
lr_scheduler = _module
parse_subs = _module
utils = _module

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


import pandas as pd


import torch


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data.dataset import Dataset


from torchtext import data


import numpy as np


from time import time


from time import strftime


from time import localtime


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils import tensorboard as tensorboard


from copy import deepcopy


class LabelSmoothing(nn.Module):

    def __init__(self, smoothing, pad_idx):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx

    def forward(self, pred, target):
        B, S, V = pred.shape
        pred = pred.contiguous().view(-1, V)
        target = target.contiguous().view(-1)
        dist = self.smoothing * torch.ones_like(pred) / (V - 2)
        dist.scatter_(1, target.unsqueeze(-1).long(), 1 - self.smoothing)
        dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target == self.pad_idx)
        if mask.sum() > 0 and len(mask) > 0:
            dist.index_fill_(0, mask.squeeze(), 0)
        return F.kl_div(pred, dist, reduction='sum')


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VocabularyEmbedder(nn.Module):

    def __init__(self, voc_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model)

    def forward(self, x):
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        return x


class FeatureEmbedder(nn.Module):

    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        self.embedder = nn.Linear(d_feat, d_model)

    def forward(self, x):
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        return x


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)
        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / 10000 ** (odds / d_model))
            pos_enc_mat[pos, evens] = np.cos(pos / 10000 ** (evens / d_model))
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        return x


def attention(Q, K, V, mask):
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    return out


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class MultiheadedAttention(nn.Module):

    def __init__(self, d_model, H):
        super(MultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        self.linears = clone(nn.Linear(d_model, d_model), 4)

    def forward(self, Q, K, V, mask):
        B, seq_len, d_model = Q.shape
        Q = self.linears[0](Q)
        K = self.linears[1](K)
        V = self.linears[2](V)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        att = attention(Q, K, V, mask)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
        att = self.linears[3](att)
        return att


class ResidualConnection(nn.Module):

    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        return x + res


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, src_mask):
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)

    def forward(self, x, src_mask):
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.self_att = MultiheadedAttention(d_model, H)
        self.enc_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, memory, src_mask, trg_mask):
        sublayer0 = lambda x: self.self_att(x, x, x, trg_mask)
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        return x


class Decoder(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(d_model, dout_p, H, d_ff), N)

    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)
        return x


class SubsAudioVideoGeneratorConcatLinearDoutLinear(nn.Module):

    def __init__(self, d_model_subs, d_model_audio, d_model_video, voc_size, dout_p):
        super(SubsAudioVideoGeneratorConcatLinearDoutLinear, self).__init__()
        self.linear = nn.Linear(d_model_subs + d_model_audio + d_model_video, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        None

    def forward(self, subs_x, audio_x, video_x):
        x = torch.cat([subs_x, audio_x, video_x], dim=-1)
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        return F.log_softmax(x, dim=-1)


class SubsAudioVideoTransformer(nn.Module):

    def __init__(self, trg_voc_size, src_subs_voc_size, d_feat_audio, d_feat_video, d_model_audio, d_model_video, d_model_subs, d_ff_audio, d_ff_video, d_ff_subs, N_audio, N_video, N_subs, dout_p, H, use_linear_embedder):
        super(SubsAudioVideoTransformer, self).__init__()
        self.src_emb_subs = VocabularyEmbedder(src_subs_voc_size, d_model_subs)
        if use_linear_embedder:
            self.src_emb_audio = FeatureEmbedder(d_feat_audio, d_model_audio)
            self.src_emb_video = FeatureEmbedder(d_feat_video, d_model_video)
        else:
            assert d_feat_video == d_model_video and d_feat_audio == d_model_audio
            self.src_emb_audio = Identity()
            self.src_emb_video = Identity()
        self.trg_emb_subs = VocabularyEmbedder(trg_voc_size, d_model_subs)
        self.trg_emb_audio = VocabularyEmbedder(trg_voc_size, d_model_audio)
        self.trg_emb_video = VocabularyEmbedder(trg_voc_size, d_model_video)
        self.pos_emb_subs = PositionalEncoder(d_model_subs, dout_p)
        self.pos_emb_audio = PositionalEncoder(d_model_audio, dout_p)
        self.pos_emb_video = PositionalEncoder(d_model_video, dout_p)
        self.encoder_subs = Encoder(d_model_subs, dout_p, H, d_ff_subs, N_subs)
        self.encoder_audio = Encoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.encoder_video = Encoder(d_model_video, dout_p, H, d_ff_video, N_video)
        self.decoder_subs = Decoder(d_model_subs, dout_p, H, d_ff_subs, N_subs)
        self.decoder_audio = Decoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.decoder_video = Decoder(d_model_video, dout_p, H, d_ff_video, N_video)
        self.generator = SubsAudioVideoGeneratorConcatLinearDoutLinear(d_model_subs, d_model_audio, d_model_video, trg_voc_size, dout_p)
        None
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg, mask):
        src_video, src_audio, src_subs = src
        src_mask, trg_mask, src_subs_mask = mask
        src_subs = self.src_emb_subs(src_subs)
        src_audio = self.src_emb_audio(src_audio)
        src_video = self.src_emb_video(src_video)
        trg_subs = self.trg_emb_subs(trg)
        trg_audio = self.trg_emb_audio(trg)
        trg_video = self.trg_emb_video(trg)
        src_subs = self.pos_emb_subs(src_subs)
        src_audio = self.pos_emb_audio(src_audio)
        src_video = self.pos_emb_video(src_video)
        trg_subs = self.pos_emb_subs(trg_subs)
        trg_audio = self.pos_emb_audio(trg_audio)
        trg_video = self.pos_emb_video(trg_video)
        memory_subs = self.encoder_subs(src_subs, src_subs_mask)
        memory_audio = self.encoder_audio(src_audio, src_mask)
        memory_video = self.encoder_video(src_video, src_mask)
        out_subs = self.decoder_subs(trg_subs, memory_subs, src_subs_mask, trg_mask)
        out_audio = self.decoder_audio(trg_audio, memory_audio, src_mask, trg_mask)
        out_video = self.decoder_video(trg_video, memory_video, src_mask, trg_mask)
        out = self.generator(out_subs, out_audio, out_video)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'d_model': 4, 'dout_p': 0.5, 'H': 4, 'd_ff': 4, 'N': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (DecoderLayer,
     lambda: ([], {'d_model': 4, 'dout_p': 0.5, 'H': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'d_model': 4, 'dout_p': 0.5, 'H': 4, 'd_ff': 4, 'N': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'dout_p': 0.5, 'H': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FeatureEmbedder,
     lambda: ([], {'d_feat': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadedAttention,
     lambda: ([], {'d_model': 4, 'H': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PositionalEncoder,
     lambda: ([], {'d_model': 4, 'dout_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualConnection,
     lambda: ([], {'size': 4, 'dout_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (SubsAudioVideoGeneratorConcatLinearDoutLinear,
     lambda: ([], {'d_model_subs': 4, 'd_model_audio': 4, 'd_model_video': 4, 'voc_size': 4, 'dout_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_v_iashin_MDVC(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

