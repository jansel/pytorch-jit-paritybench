import sys
_module = sys.modules[__name__]
del sys
audio = _module
wavallin = _module
evaluate = _module
hparams = _module
lrschedule = _module
mksubset = _module
preprocess = _module
preprocess_normalize = _module
setup = _module
synthesis = _module
test_audio = _module
test_misc = _module
test_mixture = _module
test_model = _module
tojson = _module
train = _module
wavenet_vocoder = _module
conv = _module
mixture = _module
modules = _module
tfcompat = _module
hparam = _module
upsample = _module
util = _module
version = _module
wavenet = _module

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


from scipy.io import wavfile


from torch.utils import data as data_utils


from torch.nn import functional as F


from torch import nn


from functools import partial


import random


from torch import optim


import torch.backends.cudnn as cudnn


from torch.utils.data.sampler import Sampler


from warnings import warn


import math


from torch.distributions import Normal


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand
        )
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return (losses * mask_).sum() / mask_.sum()


def _parse_fail(name, var_type, value, values):
    """Helper function for raising a value error for bad assignment."""
    raise ValueError(
        "Could not parse hparam '%s' of type '%s' with value '%s' in %s" %
        (name, var_type.__name__, value, values))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def mix_gaussian_loss(y_hat, y, log_scale_min=-7.0, reduce=True):
    """Mixture of continuous gaussian distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    C = y_hat.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y_hat.size(1) % 3 == 0
        nr_mix = y_hat.size(1) // 3
    y_hat = y_hat.transpose(1, 2)
    if C == 2:
        logit_probs = None
        means = y_hat[:, :, 0:1]
        log_scales = torch.clamp(y_hat[:, :, 1:2], min=log_scale_min)
    else:
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=
            log_scale_min)
    y = y.expand_as(means)
    centered_y = y - means
    dist = Normal(loc=0.0, scale=torch.exp(log_scales))
    log_probs = dist.log_prob(centered_y)
    if nr_mix > 1:
        log_probs = log_probs + F.log_softmax(logit_probs, -1)
    if reduce:
        if nr_mix == 1:
            return -torch.sum(log_probs)
        else:
            return -torch.sum(log_sum_exp(log_probs))
    elif nr_mix == 1:
        return -log_probs
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


class MixtureGaussianLoss(nn.Module):

    def __init__(self):
        super(MixtureGaussianLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(target)
        losses = mix_gaussian_loss(input, target, log_scale_min=hparams.
            log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return (losses * mask_).sum() / mask_.sum()


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]
        bsz = input.size(0)
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (
                    dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :
                    ].clone()
            self.input_buffer[:, (-1), :] = input[:, (-1), :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                weight = self.weight.transpose(2, 1).transpose(1, 0
                    ).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1-by-1 convolution layer
    """
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
        dilation=1, bias=bias)


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
        skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 -
        0.95, padding=None, dilation=1, causal=True, bias=True, *args, **kwargs
        ):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, *
            args, padding=padding, dilation=dilation, bias=bias, **kwargs)
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)
        else:
            self.conv1x1c = None
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=False)
        else:
            self.conv1x1g = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels,
            bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels,
            bias=bias)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Tensor: output
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
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb
        x = torch.tanh(a) * torch.sigmoid(b)
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)
        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip, self.
            conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()


class Stretch2d(nn.Module):

    def __init__(self, x_scale, y_scale, mode='nearest'):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale),
            mode=self.mode)


def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


class UpsampleNetwork(nn.Module):

    def __init__(self, upsample_scales, upsample_activation='none',
        upsample_activation_params={}, mode='nearest',
        freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = freq_axis_kernel_size, scale * 2 + 1
            padding = freq_axis_padding, scale
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding,
                bias=False)
            conv.weight.data.fill_(1.0 / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != 'none':
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
        """
        Args:
            c : B x C x T
        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        c = c.squeeze(1)
        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        return c


class ConvInUpsampleNetwork(nn.Module):

    def __init__(self, upsample_scales, upsample_activation='none',
        upsample_activation_params={}, mode='nearest',
        freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(ConvInUpsampleNetwork, self).__init__()
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(cin_channels, cin_channels, kernel_size=ks,
            bias=False)
        self.upsample = UpsampleNetwork(upsample_scales,
            upsample_activation, upsample_activation_params, mode,
            freq_axis_kernel_size, cin_pad=0, cin_channels=cin_channels)

    def forward(self, c):
        c_up = self.upsample(self.conv_in(c))
        return c_up


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]
    if nr_mix > 1:
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 -
            1e-05)
        temp = logit_probs.data - torch.log(-torch.log(temp))
        _, argmax = temp.max(dim=-1)
        one_hot = to_one_hot(argmax, nr_mix)
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1
            )
    elif C == 2:
        means, log_scales = y[:, :, (0)], y[:, :, (1)]
    elif C == 3:
        means, log_scales = y[:, :, (1)], y[:, :, (2)]
    else:
        assert False, "shouldn't happen"
    scales = torch.exp(log_scales)
    dist = Normal(loc=means, scale=scales)
    x = dist.sample()
    x = torch.clamp(x, min=-1.0, max=1.0)
    return x


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda
    x: 2 ** x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Tensor): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Tensor: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0,
    clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05
        )
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """

    def __init__(self, out_channels=256, layers=20, stacks=2,
        residual_channels=512, gate_channels=512, skip_out_channels=512,
        kernel_size=3, dropout=1 - 0.95, cin_channels=-1, gin_channels=-1,
        n_speakers=None, upsample_conditional_features=False, upsample_net=
        'ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4,
        4, 4]}, scalar_input=False, use_speaker_embedding=False,
        output_distribution='Logistic', cin_pad=0):
        super(WaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.output_distribution = output_distribution
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)
        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(residual_channels, gate_channels,
                kernel_size=kernel_size, skip_out_channels=
                skip_out_channels, bias=True, dilation=dilation, dropout=
                dropout, cin_channels=cin_channels, gin_channels=gin_channels)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([nn.ReLU(inplace=True),
            Conv1d1x1(skip_out_channels, skip_out_channels), nn.ReLU(
            inplace=True), Conv1d1x1(skip_out_channels, out_channels)])
        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(n_speakers, gin_channels,
                padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None
        if upsample_conditional_features:
            self.upsample_net = getattr(upsample, upsample_net)(**
                upsample_params)
        else:
            self.upsample_net = None
        self.receptive_field = receptive_field_size(layers, stacks, kernel_size
            )

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_bct = _expand_global_features(B, T, g, bct=True)
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = F.softmax(x, dim=1) if softmax else x
        return x

    def incremental_forward(self, initial_input=None, c=None, g=None, T=100,
        test_inputs=None, tqdm=lambda x: x, softmax=True, quantize=True,
        log_scale_min=-50.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Tensor): Initial decoder input, (B x C x 1)
            c (Tensor): Local conditioning features, shape (B x C' x T)
            g (Tensor): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Tensor): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Tensor: Generated one-hot encoded samples. B x C x T　
              or scaler vector B x 1 x T
        """
        self.clear_buffer()
        B = 1
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            elif test_inputs.size(1) == self.out_channels:
                test_inputs = test_inputs.transpose(1, 2).contiguous()
            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(1)
            else:
                T = max(T, test_inputs.size(1))
        T = int(T)
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g, bct=False)
        if c is not None:
            B = c.shape[0]
            if self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.size(-1) == T
            if c.size(-1) == T:
                c = c.transpose(1, 2).contiguous()
        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = torch.zeros(B, 1, 1)
            else:
                initial_input = torch.zeros(B, 1, self.out_channels)
                initial_input[:, :, (127)] = 1
            if next(self.parameters()).is_cuda:
                initial_input = initial_input
        elif initial_input.size(1) == self.out_channels:
            initial_input = initial_input.transpose(1, 2).contiguous()
        current_input = initial_input
        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, (t), :].unsqueeze(1)
            elif t > 0:
                current_input = outputs[-1]
            ct = None if c is None else c[:, (t), :].unsqueeze(1)
            gt = None if g is None else g_btc[:, (t), :].unsqueeze(1)
            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                skips += h
            skips *= math.sqrt(1.0 / len(self.conv_layers))
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)
            if self.scalar_input:
                if self.output_distribution == 'Logistic':
                    x = sample_from_discretized_mix_logistic(x.view(B, -1, 
                        1), log_scale_min=log_scale_min)
                elif self.output_distribution == 'Normal':
                    x = sample_from_mix_gaussian(x.view(B, -1, 1),
                        log_scale_min=log_scale_min)
                else:
                    assert False
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B,
                    -1)
                if quantize:
                    dist = torch.distributions.OneHotCategorical(x)
                    x = dist.sample()
            outputs += [x.data]
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()
        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_r9y9_wavenet_vocoder(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ResidualConv1dGLU(*[], **{'residual_channels': 4, 'gate_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    def test_002(self):
        self._check(Stretch2d(*[], **{'x_scale': 1.0, 'y_scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(WaveNet(*[], **{}), [torch.rand([4, 256, 64])], {})

