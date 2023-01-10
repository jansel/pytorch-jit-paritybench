import sys
_module = sys.modules[__name__]
del sys
build_vocab = _module
data = _module
evaluate = _module
model = _module
beam_search = _module
decoding = _module
model = _module
position_embedding = _module
score = _module
training = _module
preprocess = _module
train = _module
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


from torch.utils.data import Dataset


import torch


from functools import partial


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


from torch.distributions.uniform import Uniform


import math


from torch.nn.utils import clip_grad_norm_


from torchvision import transforms


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.distributions.bernoulli import Bernoulli


INIT = 0.01


def get_range_vector(size: int, device) ->torch.Tensor:
    return torch.arange(0, size, dtype=torch.long, device=device)


def add_positional_features(tensor: torch.Tensor, min_timescale: float=1.0, max_timescale: float=10000.0):
    """
    Implements the frequency-based positional encoding described
    in `Attention is all you Need

    Parameters
    ----------
    tensor : ``torch.Tensor``
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : ``float``, optional (default = 1.0)
        The largest timescale to use.
    Returns
    -------
    The input tensor augmented with the sinusoidal frequencies.
    """
    _, timesteps, hidden_dim = tensor.size()
    timestep_range = get_range_vector(timesteps, tensor.device).data.float()
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, tensor.device).data.float()
    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    sinusoids = torch.randn(scaled_time.size(0), 2 * scaled_time.size(1), device=tensor.device)
    sinusoids[:, ::2] = torch.sin(scaled_time)
    sinusoids[:, 1::2] = torch.sin(scaled_time)
    if hidden_dim % 2 != 0:
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)


class Im2LatexModel(nn.Module):

    def __init__(self, out_size, emb_size, dec_rnn_h, enc_out_dim=512, n_layer=1, add_pos_feat=False, dropout=0.0):
        super(Im2LatexModel, self).__init__()
        self.cnn_encoder = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2, 1), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2, 1), nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1), 0), nn.Conv2d(256, enc_out_dim, 3, 1, 0), nn.ReLU())
        self.rnn_decoder = nn.LSTMCell(dec_rnn_h + emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)
        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)
        self.W_3 = nn.Linear(dec_rnn_h + enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)
        self.add_pos_feat = add_pos_feat
        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0, 1)

    def forward(self, imgs, formulas, epsilon=1.0):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        encoded_imgs = self.encode(imgs)
        dec_states, o_t = self.init_decoder(encoded_imgs)
        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t + 1]
            if logits and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)
            dec_states, O_t, logit = self.step_decoding(dec_states, o_t, encoded_imgs, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H * W, -1)
        if self.add_pos_feat:
            encoded_imgs = add_positional_features(encoded_imgs)
        return encoded_imgs

    def step_decoding(self, dec_states, o_t, enc_out, tgt):
        """Runing one step decoding"""
        prev_y = self.embedding(tgt).squeeze(1)
        inp = torch.cat([prev_y, o_t], dim=1)
        h_t, c_t = self.rnn_decoder(inp, dec_states)
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)
        context_t, attn_scores = self._get_attn(enc_out, h_t)
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)
        logit = F.softmax(self.W_out(o_t), dim=1)
        return (h_t, c_t), o_t, logit

    def _get_attn(self, enc_out, h_t):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        alpha = torch.tanh(self.W_1(enc_out) + self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta * alpha, dim=-1)
        alpha = F.softmax(alpha, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)
        return context, alpha

    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))

