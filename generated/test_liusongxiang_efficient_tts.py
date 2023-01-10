import sys
_module = sys.modules[__name__]
del sys
prepare_expanded_phn_key = _module
prepare_features = _module
prepare_features_libri = _module
prepare_scps = _module
prepare_scps_libri = _module
prepare_scps_tmp = _module
prepare_scps_vctk_libritts_oneshot = _module
preprocess_scripts = _module
audio = _module
f0 = _module
feature_extract = _module
melodia_pitch = _module
preprocess = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
parse_pronounce = _module
symbols = _module
util = _module
convert_state_to_full_lab = _module
extract_alignment = _module
gen_fids = _module
infolog = _module
ops = _module
plot = _module
register = _module
txtd2 = _module
nntts = _module
bin = _module
inference = _module
train = _module
datasets = _module
collate_fn = _module
meldataset = _module
taco2_data = _module
distributed = _module
launch = _module
attention = _module
duration_predictor = _module
duration_predictor_bk = _module
efts_modules = _module
embedding = _module
encoder_layer = _module
initializer = _module
layer_norm = _module
length_regulator = _module
multi_layer_conv = _module
positionwise_feed_forward = _module
repeat = _module
taco2_postnet = _module
transformer_block = _module
losses = _module
duration_loss = _module
fastspeech_loss = _module
log_mse_loss = _module
stft_loss = _module
models = _module
duration_model = _module
efficient_tts = _module
optimizers = _module
radam = _module
schedulers = _module
warmup_lr = _module
trainers = _module
efficient_tts_trainer = _module
utils = _module
load_yaml = _module
nets_utils = _module
plotting = _module
utils = _module
version = _module
vocoders = _module
env = _module
hifigan_model = _module
utils = _module
setup = _module
test = _module

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


import logging


import time


import numpy as np


import torch


import matplotlib


import matplotlib.pylab as plt


from torch.utils.data import DataLoader


import math


import random


import torch.utils.data


from scipy.io.wavfile import read


import numpy


from torch import nn


import torch.nn.functional as F


from typing import Dict


from typing import Tuple


from torch.optim import *


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import *


from typing import Union


from torch.optim.lr_scheduler import _LRScheduler


from collections import defaultdict


import torch.nn as nn


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    
    Args:
        nout: output dim size
        dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        
        Args:
            x (torch.Tensor): input tensor
        Returns: 
            layer normalized tensor
        """
        if self.dim == -1:
            return super().forward(x)
        else:
            return super().forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.

    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super().__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)
        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, True)


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ResConv1d(torch.nn.Module):
    """Residual Conv1d layer"""

    def __init__(self, n_channels=512, k_size=5, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, dropout_rate=0.1):
        super().__init__()
        if dropout_rate < 1e-05:
            self.conv = torch.nn.Sequential(torch.nn.Conv1d(n_channels, n_channels, kernel_size=k_size, padding=(k_size - 1) // 2), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))
        else:
            self.conv = torch.nn.Sequential(torch.nn.Conv1d(n_channels, n_channels, kernel_size=k_size, padding=(k_size - 1) // 2), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Dropout(dropout_rate))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class ResConvBlock(torch.nn.Module):
    """Block containing several ResConv1d layers."""

    def __init__(self, num_layers, n_channels=512, k_size=5, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, dropout_rate=0.1, use_weight_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.Sequential(*[ResConv1d(n_channels, k_size, nonlinear_activation, nonlinear_activation_params, dropout_rate) for _ in range(num_layers)])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        return self.layers(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)


def rename_state_dict(old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]):
    """Replace keys of old prefix with new prefix in state dict."""
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f'Rename: {old_prefix} -> {new_prefix}')
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


def _pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    rename_state_dict(prefix + 'input_layer.', prefix + 'embed.', state_dict)
    rename_state_dict(prefix + 'norm.', prefix + 'after_norm.', state_dict)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Encoder layer module.
    
    Args:
    size: input dim
    self_attn: self attention module
    feed_forward: feed forward module
    dropout_rate: dropout rate
    normalize_before: whether to use layer_norm before the first block
    concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, mask, cache=None):
        """Compute encoded features.
        
        Args:
        x: encoded source features (batch, max_time_in, size)
        mask: mask for x (batch, max_time_in)
        cache: cache for x (batch, max_time_in - 1, size)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, mask


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, ilens, alpha=1.0):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        xs = [x[:ilen] for x, ilen in zip(xs, ilens)]
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]
        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration.

        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])

        """
        if d.sum() == 0:
            logging.warn('all of the predicted durations are 0. fill 0 with 1.')
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network, which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_. The Postnet predicts refines the predicted
    Mel-filterbank of the decoder, which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, odim, n_layers=5, n_chans=512, n_filts=5, dropout_rate=0.5, use_batch_norm=True):
        """Initialize postnet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..

        """
        super().__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.BatchNorm1d(ochans), torch.nn.Tanh(), torch.nn.Dropout(dropout_rate))]
            else:
                self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.Tanh(), torch.nn.Dropout(dropout_rate))]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, odim, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.BatchNorm1d(odim), torch.nn.Dropout(dropout_rate))]
        else:
            self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, odim, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.Dropout(dropout_rate))]

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).

        """
        for l in range(len(self.postnet)):
            xs = self.postnet[l](xs)
        return xs


def repeat(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])


class TransformerBlock(torch.nn.Module):
    """Transformer module.

    Args:
        attention_dim: dimention of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(self, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0, input_layer=None, pos_enc_class=PositionalEncoding, normalize_before=True, concat_after=False, positionwise_layer_type='linear', positionwise_conv_kernel_size=1, padding_idx=-1):
        """Construct an Encoder object."""
        super().__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)
        if isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(input_layer, pos_enc_class(attention_dim, positional_dropout_rate))
        elif input_layer is None:
            self.embed = torch.nn.Sequential(pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise ValueError('unknown input_layer: ' + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == 'linear':
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = attention_dim, linear_units, dropout_rate
        elif positionwise_layer_type == 'conv1d':
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate
        elif positionwise_layer_type == 'conv1d-linear':
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate
        else:
            raise NotImplementedError('Support only linear or conv1d.')
        self.encoders = repeat(num_blocks, lambda : EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate), positionwise_layer(*positionwise_layer_args), dropout_rate, normalize_before, concat_after))
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class DurationMSELoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction='mean'):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)
        return loss


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


class FastSpeechLoss(torch.nn.Module):
    """Loss function for Fastspeech model."""

    def __init__(self, use_masking: bool=True, use_weighted_masking: bool=False, use_mse: bool=True):
        """Initialize Fastspeech loss module.
        
        Args:
            use_masking (bool): Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss calculation.

        """
        super().__init__()
        assert use_masking != use_weighted_masking or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        reduction = 'none' if self.use_weighted_masking else 'mean'
        if use_mse:
            self.mel_criterion = torch.nn.MSELoss(reduction=reduction)
        else:
            self.mel_criterion = torch.nn.L1Loss(reduction=reduction)
        self.duration_criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, after_outs, before_outs, d_outs, ys, ds, ilens, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = after_outs.masked_select(out_masks) if after_outs is not None else None
            ys = ys.masked_select(out_masks)
        mel_loss = self.mel_criterion(before_outs, ys)
        if after_outs is not None:
            mel_loss += self.mel_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        return mel_loss, duration_loss


class LogMSELoss(torch.nn.Module):
    """Loss function module for durationi/log-f0 predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction='mean'):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)
        return loss


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-07)).transpose(2, 1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window'):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


class DurationModel(torch.nn.Module):
    """Duration model."""

    def __init__(self, idim: int, odim: int=1, duration_predictor_layers: int=2, duration_predictor_chans: int=256, duration_predictor_kernel_size: int=3, num_spks: int=None, spk_embed_dim: int=None, spk_embed_integration_type: str='add', duration_predictor_dropout_rate: float=0.1, use_masking: bool=True, use_weighted_masking: bool=False):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        padding_idx = 0
        self.duration_predictor = DurationPredictor(idim=idim, n_layers=duration_predictor_layers, n_chans=duration_predictor_chans, kernel_size=duration_predictor_kernel_size, dropout_rate=duration_predictor_dropout_rate, num_spks=num_spks, spk_embed_dim=self.spk_embed_dim, spk_embed_integration_type=spk_embed_integration_type)
        reduction = 'none' if self.use_weighted_masking else 'mean'
        self.criterion = DurationMSELoss(reduction=reduction)

    def _forward(self, xs, ilens, spkids=None, is_inference=False):
        d_masks = make_pad_mask(ilens)
        if is_inference:
            d_outs = self.duration_predictor(xs, d_masks, spkids)
            d_outs = torch.clamp(torch.round(d_outs.exp() - 1.0), min=0).long()
        else:
            d_outs = self.duration_predictor(xs, d_masks, spkids)
        return d_outs

    def forward(self, xs: torch.Tensor, ilens: torch.Tensor, durations: torch.Tensor, spkids: torch.Tensor=None) ->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = xs.size(0)
        d_outs = self._forward(xs, ilens, spkids, is_inference=False)
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens)
            d_outs = d_outs.masked_select(duration_masks)
            d_trgs = durations.masked_select(duration_masks)
        duration_loss = self.criterion(d_outs, d_trgs)
        stats = dict(loss=duration_loss.item())
        return duration_loss, stats

    def inference(self, xs: torch.Tensor, spkids: torch.Tensor=None) ->torch.Tensor:
        ilens = torch.tensor([xs.shape[1]], dtype=torch.long, device=xs.device)
        d_outs = self._forward(xs, ilens, spkids=spkids, is_inference=True)
        return d_outs[0]


class EfficientTTSCNN(torch.nn.Module):
    """EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture
    """

    def __init__(self, num_symbols: int, odim: int=80, symbol_embedding_dim: int=512, n_channels: int=512, n_text_encoder_layer: int=5, n_mel_encoder_layer: int=3, n_decoder_layer: int=6, n_duration_layer: int=2, k_size: int=5, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, use_weight_norm=True, dropout_rate=0.1, use_masking: bool=False, use_weighted_masking: bool=False, duration_offset=1.0, sigma=0.01, sigma_e=0.5, delta_e_method_1=True, share_text_encoder_key_value=False, use_mel_query_fc=False):
        super().__init__()
        self.duration_offset = duration_offset
        self.sigma = sigma
        self.sigma_e = sigma_e
        self.delta_e_method_1 = delta_e_method_1
        self.share_text_encoder_key_value = share_text_encoder_key_value
        self.text_embedding_table = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.text_encoder = ResConvBlock(num_layers=n_text_encoder_layer, n_channels=n_channels, k_size=k_size, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, dropout_rate=dropout_rate, use_weight_norm=use_weight_norm)
        self.text_encoder_key = torch.nn.Linear(n_channels, n_channels)
        if not share_text_encoder_key_value:
            self.text_encoder_value = torch.nn.Linear(n_channels, n_channels)
        self.mel_prenet = torch.nn.Sequential(torch.nn.Linear(odim, n_channels), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Dropout(dropout_rate))
        self.mel_encoder = ResConvBlock(num_layers=n_mel_encoder_layer, n_channels=n_channels, k_size=k_size, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, dropout_rate=dropout_rate, use_weight_norm=use_weight_norm)
        if use_mel_query_fc:
            self.mel_query_fc = torch.nn.Linear(n_channels, n_channels)
        else:
            self.mel_query_fc = None
        self.decoder = ResConvBlock(num_layers=n_decoder_layer, n_channels=n_channels, k_size=k_size, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, dropout_rate=dropout_rate, use_weight_norm=use_weight_norm)
        self.mel_output_layer = torch.nn.Linear(n_channels, odim)
        self.duration_predictor = DurationPredictor(idim=n_channels, n_layers=n_duration_layer, n_chans=n_channels, offset=duration_offset)
        self.criterion = FastSpeechLoss(use_masking=use_masking, use_weighted_masking=use_weighted_masking)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, speech: torch.Tensor, speech_lengths: torch.Tensor) ->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward propagations.
        Args:
            text: Batch of padded text ids (B, T1).
            text_lengths: Batch of lengths of each input batch (B,).
            speech: Batch of mel-spectrograms (B, T2, num_mels)
            speech_lengths: Batch of mel-spectrogram lengths (B,)
        """
        device = text.device
        text_mask = make_non_pad_mask(text_lengths)
        mel_mask = make_non_pad_mask(speech_lengths)
        text_mel_mask = text_mask.unsqueeze(-1) & mel_mask.unsqueeze(1)
        text_embedding = self.text_embedding_table(text).transpose(1, 2)
        text_h = self.text_encoder(text_embedding).transpose(1, 2)
        text_key = self.text_encoder_key(text_h)
        if self.share_text_encoder_key_value:
            text_value = text_key
        else:
            text_value = self.text_encoder_value(text_h)
        _tmp_mask = ~text_mask.unsqueeze(-1).repeat(1, 1, text_h.size(2))
        text_key = text_key.masked_fill(_tmp_mask, 0.0)
        text_value = text_value.masked_fill(_tmp_mask, 0.0)
        mel_h = self.mel_prenet(speech).transpose(1, 2)
        mel_h = self.mel_encoder(mel_h).transpose(1, 2)
        if self.mel_query_fc is not None:
            mel_h = self.mel_query_fc(mel_h)
        alpha = self.scaled_dot_product_attention(mel_h, text_key, text_mask)
        alpha = alpha.masked_fill(~text_mel_mask, 0.0)
        text_index_vector = self.generate_index_vector(text_mask)
        imv = self.imv_generator(alpha, text_index_vector, mel_mask, text_lengths)
        e = self.get_aligned_positions(imv, text_index_vector, mel_mask, text_mask, sigma=self.sigma_e)
        e = e.squeeze(-1)
        reconst_alpha = self.reconstruct_align_from_aligned_position(e, delta=self.sigma, mel_mask=mel_mask, text_mask=text_mask).masked_fill(~text_mel_mask, 0.0)
        text_value_expanded = torch.bmm(text_value.transpose(1, 2), reconst_alpha)
        _tmp_mask_2 = ~mel_mask.unsqueeze(1).repeat(1, text_value.size(2), 1)
        text_value_expanded = text_value_expanded.masked_fill(_tmp_mask_2, 0.0)
        mel_pred = self.decoder(text_value_expanded)
        mel_pred = self.mel_output_layer(mel_pred.transpose(1, 2))
        _tmp_mask_3 = ~mel_mask.unsqueeze(-1).repeat(1, 1, 80)
        mel_pred = mel_pred.masked_fill(_tmp_mask_3, 0.0)
        if self.delta_e_method_1:
            delta_e = torch.cat([e[:, :1], e[:, 1:] - e[:, :-1]], dim=1).detach()
        else:
            B = speech_lengths.size(0)
            e = e.detach()
            e = torch.cat([e, torch.zeros(B, 1)], dim=1)
            for i in range(B):
                max_len = speech_lengths[i].cpu().item()
                max_index = text_lengths[i].cpu().item()
                e[i, max_index] = max_len
            delta_e = e[:, 1:] - e[:, :-1]
        log_delta_e = torch.log(delta_e + self.duration_offset)
        log_delta_e = log_delta_e.masked_fill(~text_mask, 0.0)
        dur_pred = self.duration_predictor(text_value, ~text_mask)
        mel_loss, dur_loss = self.criterion(None, mel_pred, dur_pred, speech, log_delta_e, text_lengths, speech_lengths)
        loss = mel_loss + dur_loss
        stats = dict(loss=loss.item(), mel_loss=mel_loss.item(), duration_loss=dur_loss.item())
        return loss, stats, imv, reconst_alpha, mel_pred, speech

    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor=None):
        """Inference.
        Args:
            text: Batch of padded text ids (B, T1).
            text_lengths: Batch of lengths of each input batch (B,).
        """
        device = text.device
        text_embedding = self.text_embedding_table(text).transpose(1, 2)
        text_h = self.text_encoder(text_embedding).transpose(1, 2)
        text_key = self.text_encoder_key(text_h)
        if self.share_text_encoder_key_value:
            text_value = text_key
        else:
            text_value = self.text_encoder_value(text_h)
        delta_e = self.duration_predictor.inference(text_value, to_round=False)
        if self.delta_e_method_1:
            e = torch.cumsum(delta_e, dim=1)
        else:
            e = torch.cumsum(torch.cat([torch.zeros(1, 1), delta_e], dim=1), dim=1)
        reconst_alpha = self.reconstruct_align_from_aligned_position(e, delta=self.sigma, mel_mask=None, text_mask=None, trim_e=not self.delta_e_method_1)
        text_value_expanded = torch.bmm(text_value.transpose(1, 2), reconst_alpha)
        mel_pred = self.decoder(text_value_expanded)
        mel_pred = self.mel_output_layer(mel_pred.transpose(1, 2))
        return mel_pred, reconst_alpha

    def generate_index_vector(self, text_mask):
        """Create index vector of text sequence.
        Args:
            text_mask: mask of text-sequence.[B, T1]
        Returns:
            index vector of text sequence. [B, T1]
        """
        device = text_mask.device
        B, T1 = text_mask.size()
        p = torch.arange(0, T1).repeat(B, 1).float()
        return p * text_mask

    def imv_generator(self, alpha, p, mel_mask, text_length):
        """Compute index mapping index (IMV) from alignment matrix alpha.
        Implementation of HMA.
        Args:
            alpha: scaled dot attention [B, T1, T2]
            p: index vector, output of generate_index_vector. [B, T1]
            mel_mask: mask of mel-spectrogram [B, T2]
            text_length: lengths of input text-sequence [B]
        Returns:
            Index mapping vector (IMV) [B, T2]
        """
        B, T1, T2 = alpha.size()
        imv_dummy = torch.bmm(alpha.transpose(1, 2), p.unsqueeze(-1)).squeeze(-1)
        delta_imv = torch.relu(imv_dummy[:, 1:] - imv_dummy[:, :-1])
        delta_imv = torch.cat([torch.zeros(B, 1).type_as(alpha), delta_imv], -1)
        imv = torch.cumsum(delta_imv, -1) * mel_mask.float()
        last_imv, _ = torch.max(imv, dim=-1)
        last_imv = torch.clamp(last_imv, min=1e-08)
        imv = imv / last_imv.unsqueeze(1) * (text_length.float().unsqueeze(-1) - 1)
        return imv

    def get_aligned_positions(self, imv, p, mel_mask, text_mask, sigma=0.5):
        """Compute aligned positions from imv
        Args:
            imv: index mapping vector [B, T2].
            p: index vector, output of generate_index_vector [B, T1].
            mel_mask: mask of mel-sepctrogram [B, T2].
            text_mask: mask of text sequences [B, T1].
            sigma: a scalar, default 0.5
        Returns:
            Aligned positions [B, T1].
        """
        energies = -1 * (imv.unsqueeze(1) - p.unsqueeze(-1)) ** 2 * sigma
        energies = energies.masked_fill(~mel_mask.unsqueeze(1).repeat(1, energies.size(1), 1), -float('inf'))
        beta = torch.softmax(energies, dim=2)
        q = torch.arange(0, mel_mask.size(-1)).unsqueeze(0).repeat(imv.size(0), 1).float()
        q = q * mel_mask.float()
        return torch.bmm(beta, q.unsqueeze(-1)) * text_mask.unsqueeze(-1)

    def reconstruct_align_from_aligned_position(self, e, delta=0.1, mel_mask=None, text_mask=None, trim_e=False):
        """Reconstruct alignment matrix from aligned positions.
        Args:
            e: aligned positions [B, T1].
            delta: a scalar, default 0.1
            mel_mask: mask of mel-spectrogram [B, T2], None if inference and B==1.
            text_mask: mask of text-sequence, None if B==1.
        Returns:
            alignment matrix [B, T1, T2].
        """
        if mel_mask is None:
            max_length = torch.round(e[:, -1]).squeeze().item()
            if trim_e:
                e = e[:, :-1]
        else:
            max_length = mel_mask.size(-1)
        q = torch.arange(0, max_length).unsqueeze(0).repeat(e.size(0), 1).float()
        if mel_mask is not None:
            q = q * mel_mask.float()
        energies = -1 * delta * (q.unsqueeze(1) - e.unsqueeze(-1)) ** 2
        if text_mask is not None:
            energies = energies.masked_fill(~text_mask.unsqueeze(-1).repeat(1, 1, max_length), -float('inf'))
        return torch.softmax(energies, dim=1)

    def scaled_dot_product_attention(self, query, key, key_mask):
        """
        Args:
            query: mel-encoder output [B, T2, D]
            key: text-encoder output [B, T1, D]
            key_mask: mask of text-encoder output [B, T1]
        Returns:
            attention alignment matrix alpha [B, T1, T2].
        """
        D = key.size(-1)
        T1 = key.size(1)
        T2 = query.size(1)
        scores = torch.bmm(query, key.transpose(-2, -1)) / np.sqrt(float(D))
        key_mask = ~key_mask.unsqueeze(1).repeat(1, T2, 1)
        scores = scores.masked_fill(key_mask, -float('inf'))
        alpha = torch.softmax(scores, dim=-1).masked_fill(key_mask, 0.0)
        return alpha.transpose(-2, -1)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv1dLinear,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {}),
     False),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (DurationMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DurationPredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogSTFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiLayeredConv1d,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {}),
     False),
    (MultiSequential,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Postnet,
     lambda: ([], {'idim': 4, 'odim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ScaledPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpectralConvergenceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_liusongxiang_efficient_tts(_paritybench_base):
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

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

