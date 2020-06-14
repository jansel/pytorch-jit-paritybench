import sys
_module = sys.modules[__name__]
del sys
config_omniglot = _module
config_synthetic = _module
config_yahoo = _module
config_yelp = _module
data = _module
text_data = _module
image = _module
logger = _module
modules = _module
decoders = _module
dec_lstm = _module
dec_pixelcnn = _module
dec_pixelcnn_v2 = _module
decoder = _module
decoder_helper = _module
encoders = _module
enc_lstm = _module
enc_mix = _module
enc_resnet = _module
enc_resnet_v2 = _module
encoder = _module
lm = _module
lm_lstm = _module
plotter = _module
utils = _module
vae = _module
plot_multiple = _module
plot_single = _module
prepare_data = _module
text = _module
toy = _module

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


import time


import numpy as np


import torch


import torch.utils.data


from torch import nn


from torch import optim


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import math


from torch.autograd import Variable


from itertools import chain


import torch.distributions as dist


class GatedMaskedConv2d(nn.Module):

    def __init__(self, in_dim, out_dim=None, kernel_size=3, mask='B'):
        super(GatedMaskedConv2d, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.dim = out_dim
        self.size = kernel_size
        self.mask = mask
        pad = self.size // 2
        self.v_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(pad + 1,
            self.size))
        self.v_pad1 = nn.ConstantPad2d((pad, pad, pad, 0), 0)
        self.v_pad2 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.vh_conv = nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=1)
        self.h_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(1, pad + 1))
        self.h_pad1 = nn.ConstantPad2d((self.size // 2, 0, 0, 0), 0)
        self.h_pad2 = nn.ConstantPad2d((1, 0, 0, 0), 0)
        self.h_conv_res = nn.Conv2d(self.dim, self.dim, 1)

    def forward(self, v_map, h_map):
        v_out = self.v_pad2(self.v_conv(self.v_pad1(v_map)))[:, :, :-1, :]
        v_map_out = F.tanh(v_out[:, :self.dim]) * F.sigmoid(v_out[:, self.dim:]
            )
        vh = self.vh_conv(v_out)
        h_out = self.h_conv(self.h_pad1(h_map))
        if self.mask == 'A':
            h_out = self.h_pad2(h_out)[:, :, :, :-1]
        h_out = h_out + vh
        h_out = F.tanh(h_out[:, :self.dim]) * F.sigmoid(h_out[:, self.dim:])
        h_map_out = self.h_conv_res(h_out)
        if self.mask == 'B':
            h_map_out = h_map_out + h_map
        return v_map_out, h_map_out


class StackedGatedMaskedConv2d(nn.Module):

    def __init__(self, img_size=[1, 28, 28], layers=[64, 64, 64],
        kernel_size=[7, 7, 7], latent_dim=64, latent_feature_map=1):
        super(StackedGatedMaskedConv2d, self).__init__()
        input_dim = img_size[0]
        self.conv_layers = []
        if latent_feature_map > 0:
            self.latent_feature_map = latent_feature_map
            self.z_linear = nn.Linear(latent_dim, latent_feature_map * 28 * 28)
        for i in range(len(kernel_size)):
            if i == 0:
                self.conv_layers.append(GatedMaskedConv2d(input_dim +
                    latent_feature_map, layers[i], kernel_size[i], 'A'))
            else:
                self.conv_layers.append(GatedMaskedConv2d(layers[i - 1],
                    layers[i], kernel_size[i]))
        self.modules = nn.ModuleList(self.conv_layers)

    def forward(self, img, q_z=None):
        """
        Args:
            img: (batch, nc, H, W)
            q_z: (batch, nsamples, nz)
        """
        batch_size, nsamples, _ = q_z.size()
        if q_z is not None:
            z_img = self.z_linear(q_z)
            z_img = z_img.view(img.size(0), nsamples, self.
                latent_feature_map, img.size(2), img.size(3))
            img = img.unsqueeze(1).expand(batch_size, nsamples, *img.size()[1:]
                )
        for i in range(len(self.conv_layers)):
            if i == 0:
                if q_z is not None:
                    v_map = torch.cat([img, z_img], 2)
                    v_map = v_map.view(-1, *v_map.size()[2:])
                else:
                    v_map = img
                h_map = v_map
            v_map, h_map = self.conv_layers[i](v_map, h_map)
        return h_map


class MaskedConv2d(nn.Conv2d):

    def __init__(self, mask_type, masked_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :masked_channels, (kH // 2), kW // 2 + (mask_type == 'B'):
            ] = 0
        self.mask[:, :masked_channels, kH // 2 + 1:] = 0

    def reset_parameters(self):
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2.0 / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super(MaskedConv2d, self).forward(x)


class PixelCNNBlock(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(PixelCNNBlock, self).__init__()
        self.mask_type = 'B'
        padding = kernel_size // 2
        out_channels = in_channels // 2
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), nn.BatchNorm2d(out_channels), nn.ELU(),
            MaskedConv2d(self.mask_type, out_channels, out_channels,
            out_channels, kernel_size, padding=padding, bias=False), nn.
            BatchNorm2d(out_channels), nn.ELU(), nn.Conv2d(out_channels,
            in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        return self.activation(self.main(input) + input)


class MaskABlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, masked_channels
        ):
        super(MaskABlock, self).__init__()
        self.mask_type = 'A'
        padding = kernel_size // 2
        self.main = nn.Sequential(MaskedConv2d(self.mask_type,
            masked_channels, in_channels, out_channels, kernel_size,
            padding=padding, bias=False), nn.BatchNorm2d(out_channels), nn.
            ELU())
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    def forward(self, input):
        return self.main(input)


class PixelCNN(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks, kernel_sizes,
        masked_channels):
        super(PixelCNN, self).__init__()
        assert num_blocks == len(kernel_sizes)
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = MaskABlock(in_channels, out_channels, kernel_sizes[
                    i], masked_channels)
            else:
                block = PixelCNNBlock(out_channels, kernel_sizes[i])
            self.blocks.append(block)
        self.main = nn.ModuleList(self.blocks)
        self.direct_connects = []
        for i in range(1, num_blocks - 1):
            self.direct_connects.append(PixelCNNBlock(out_channels,
                kernel_sizes[i]))
        self.direct_connects = nn.ModuleList(self.direct_connects)

    def forward(self, input):
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                input = input + direct_conncet(direct_input)
            input = layer(input)
            direct_inputs.append(input)
        assert len(direct_inputs) == 3, 'architecture error: %d' % len(
            direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return input + direct_conncet(direct_inputs.pop(0))


class DecoderBase(nn.Module):
    """docstring for Decoder"""

    def __init__(self):
        super(DecoderBase, self).__init__()

    def decode(self, x, z):
        raise NotImplementedError

    def reconstruct_error(self, x, z):
        """reconstruction loss
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """
        raise NotImplementedError

    def beam_search_decode(self, z, K):
        """beam search decoding
        Args:
            z: (batch_size, nz)
            K: the beam size

        Returns: List1
            List1: the decoded word sentence list
        """
        raise NotImplementedError

    def sample_decode(self, z):
        """sampling from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """
        raise NotImplementedError

    def greedy_decode(self, z):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """
        raise NotImplementedError

    def log_probability(self, x, z):
        """
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        raise NotImplementedError


class CNNClassifier(nn.Module):
    """CNNClassifier from Yoon Kim's paper"""

    def __init__(self, args):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.
            ni)) for K in args.kernel_sizes])
        self.dropout = nn.Dropout(args.cnn_dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args
            .mix_num)

    def forward(self, x):
        """
        Args:
            x: Tensor
                the embedding of input, with shape (batch_size, seq_length, ni)


        Returns: Tensor1
            Tensor1: the logits for the mixture prob, shape (batch_size, mix_num)
        """
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=
            keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class MixLSTMEncoder(nn.Module):
    """Mixture of Gaussian LSTM Encoder with constant-length input"""

    def __init__(self, args, vocab_size, model_init, emb_init):
        super(MixLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.embed = nn.Embedding(vocab_size, args.ni)
        self.classifier = CNNClassifier(args)
        self.lstm_lists = nn.ModuleList([nn.LSTM(input_size=args.ni,
            hidden_size=self.nh, num_layers=1, batch_first=True, dropout=0) for
            _ in range(args.mix_num)])
        self.linear_lists = nn.ModuleList([nn.Linear(self.nh, 2 * args.nz,
            bias=False) for _ in range(args.mix_num)])
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def sample(self, mu, logvar, mix_prob, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean tensors of mixed gaussian distribution, 
                with shape (batch_size, mix_num, nz)

            logvar: Tensor
                logvar tensors of mixed gaussian distibution,
                 with shape (batch_size, mix_num, nz)

            mix_prob: Tensor
                the mixture probability weights, 
                with shape (batch_size, mix_num)

        Returns: Tensor
            Sampled z with shape (batch_size, nsamples, nz)
        """
        batch_size = mix_prob.size(0)
        classes = torch.multinomial(mix_prob, nsamples, replacement=True
            ).unsqueeze(2).expand(batch_size, nsamples, self.nz)
        mu_ = torch.gather(mu, dim=1, index=classes)
        logvar_ = torch.gather(logvar, dim=1, index=classes)
        std = (0.5 * logvar_).exp()
        return torch.normal(mu_, std)

    def forward(self, input):
        """
        Args:
            input: (batch_size, seq_len, ni)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor of different inference nets, 
                shape (batch, mix_num, nz)
            Tensor2: the logvar tensor of different inference nets, 
                shape (batch, mix_num, nz)
        """
        mean_list = []
        logvar_list = []
        for lstm, linear in zip(self.lstm_lists, self.linear_lists):
            _, (last_state, last_cell) = lstm(input)
            mean, logvar = linear(last_state).unsqueeze(2).chunk(2, -1)
            mean_list.append(mean)
            logvar_list.append(logvar)
        return torch.cat(mean_list, dim=2).squeeze(0), torch.cat(logvar_list,
            dim=2).squeeze(0)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term
        Args:
            input: (batch_size, seq_len)

        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: containes two tensors:
                1. the tensor of KL for each x with shape [batch, nsamples]
                2. the tensor of log q(z | x) with shape [batch, nsamples]

        """
        embed = self.embed(input)
        log_mix_weights = self.classifier(embed)
        mix_prob = (log_mix_weights - log_sum_exp(log_mix_weights, dim=1,
            keepdim=True)).exp()
        mu, logvar = self.forward(embed)
        z = self.sample(mu, logvar, mix_prob, nsamples)
        log_posterior = self.log_posterior(z, mu, logvar, mix_prob)
        KL = log_posterior - self.log_prior(z)
        return z, (KL, log_posterior, mix_prob)

    def log_prior(self, z):
        """evaluate the log density of prior at z
        Args:
            z: Tensor
                the points to be evaluated, with shape 
                (batch_size, nsamples, nz)

        Returns: Tensor1
            Tensor1: the log density of shape (batch_size, nsamples)     
        """
        return -0.5 * (z ** 2).sum(-1) - 0.5 * self.nz * math.log(2 * math.pi)

    def log_posterior(self, z, mu, logvar, mix_prob):
        """evaluate the log density of approximate 
        posterior at z

        Args:
            z: Tensor
                the points to be evaluated, with shape 
                (batch_size, nsamples, nz)

            mu: Tensor
                Mean tensors of mixed gaussian distribution, 
                with shape (batch_size, mix_num, nz)

            logvar: Tensor
                logvar tensors of mixed gaussian distibution,
                 with shape (batch_size, mix_num, nz)

            mix_prob: Tensor
                the mixture probability weights, 
                with shape (batch_size, mix_num)

        Returns: Tensor1
            Tensor1: the log density of shape (batch_size, nsamples)
        """
        z = z.unsqueeze(1)
        mu, logvar = mu.unsqueeze(2), logvar.unsqueeze(2)
        var = logvar.exp()
        dev = z - mu
        log_density = -0.5 * (dev ** 2 / var).sum(dim=-1) - 0.5 * (self.nz *
            math.log(2 * math.pi) + logvar.sum(-1))
        log_density = log_density + mix_prob.log().unsqueeze(2)
        return log_sum_exp(log_density, dim=1)


class MaskedConv2d(nn.Conv2d):

    def __init__(self, include_center=False, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, (kH // 2), kW // 2 + (include_center == True):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim=None, with_residual=True,
        with_batchnorm=True, mask=None, kernel_size=3, padding=1):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        if mask is None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                padding=padding)
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=
                kernel_size, padding=padding)
        else:
            self.conv1 = MaskedConv2d(mask, in_dim, out_dim, kernel_size=
                kernel_size, padding=padding)
            self.conv2 = MaskedConv2d(mask, out_dim, out_dim, kernel_size=
                kernel_size, padding=padding)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class ResNetBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.ELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                planes))
        self.downsample = downsample
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out + residual)
        return out


class ResNet(nn.Module):

    def __init__(self, inplanes, planes, strides):
        super(ResNet, self).__init__()
        assert len(planes) == len(strides)
        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = ResNetBlock(inplanes, plane, stride=stride)
            blocks.append(block)
            inplanes = plane
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)


class GaussianEncoderBase(nn.Module):
    """docstring for EncoderBase"""

    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """
        raise NotImplementedError

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """
        mu, logvar = self.forward(input)
        z = self.reparameterize(mu, logvar, nsamples)
        return z, (mu, logvar)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """
        mu, logvar = self.forward(input)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()
        dev = z - mu
        log_density = -0.5 * (dev ** 2 / var).sum(dim=-1) - 0.5 * (nz *
            math.log(2 * math.pi) + logvar.sum(-1))
        return log_density

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """
        mu, logvar = self.forward(x)
        x_batch, nz = mu.size()
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 +
            logvar).sum(-1)).mean()
        z_samples = self.reparameterize(mu, logvar, 1)
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z_samples - mu
        log_density = -0.5 * (dev ** 2 / var).sum(dim=-1) - 0.5 * (nz *
            math.log(2 * math.pi) + logvar.sum(-1))
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()


class LSTM_LM(nn.Module):
    """LSTM decoder with constant-length data"""

    def __init__(self, args, vocab, model_init, emb_init):
        super(LSTM_LM, self).__init__()
        self.ni = args.ni
        self.nh = args.nh
        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)
        self.dropout_in = nn.Dropout(args.dropout_in)
        self.dropout_out = nn.Dropout(args.dropout_out)
        self.lstm = nn.LSTM(input_size=args.ni, hidden_size=args.nh,
            num_layers=1, batch_first=True)
        self.pred_linear = nn.Linear(args.nh, len(vocab), bias=False)
        vocab_mask = torch.ones(len(vocab))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def decode(self, input):
        """
        Args:
            input: (batch_size, seq_len)
        """
        batch_size, seq_len = input.size()
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)
        c_init = word_embed.new_zeros((1, batch_size, self.nh))
        h_init = word_embed.new_zeros((1, batch_size, self.nh))
        output, _ = self.lstm(word_embed, (h_init, c_init))
        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)
        return output_logits

    def reconstruct_error(self, x):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size). Loss across different sentences
        """
        src = x[:, :-1]
        tgt = x[:, 1:]
        batch_size, seq_len = src.size()
        output_logits = self.decode(src)
        tgt = tgt.contiguous().view(-1)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)), tgt)
        return loss.view(batch_size, -1).sum(-1)

    def log_probability(self, x):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
        Returns:
            log_p: (batch_size).
        """
        return -self.reconstruct_error(x)


class VAE(nn.Module):
    """VAE with normal prior"""

    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.nz = args.nz
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def encode(self, x, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """
        return self.encoder(x)

    def decode(self, z, strategy, K=5):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """
        if strategy == 'beam':
            return self.decoder.beam_search_decode(z, K)
        elif strategy == 'greedy':
            return self.decoder.greedy_decode(z)
        elif strategy == 'sample':
            return self.decoder.sample_decode(z)
        else:
            raise ValueError('the decoding strategy is not supported')

    def reconstruct(self, x, decoding_strategy='greedy', K=5):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter (if applicable)

        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)
        return self.decode(z, decoding_strategy, K)

    def loss(self, x, kl_weight, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        z, KL = self.encode(x, nsamples)
        reconstruct_err = self.decoder.reconstruct_error(x, z).mean(dim=1)
        return reconstruct_err + kl_weight * KL, reconstruct_err, KL

    def nll_iw(self, x, nsamples, ns=100):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """
        tmp = []
        for _ in range(int(nsamples / ns)):
            z, param = self.encoder.sample(x, ns)
            log_comp_ll = self.eval_complete_ll(x, z)
            log_infer_ll = self.eval_inference_dist(x, z, param)
            tmp.append(log_comp_ll - log_infer_ll)
        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples
            )
        return -ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)
        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)
        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """
        return self.decoder.log_probability(x, z)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace

        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()
            ).contiguous()
        log_comp = self.eval_complete_ll(x, grid_z)
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)
        return log_posterior

    def sample_from_prior(self, nsamples):
        """sampling from prior distribution

        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.prior.sample((nsamples,))

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)
        return z

    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur, std=cur.new_full(size=cur.size(),
                fill_value=self.args.mh_std))
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll
            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))
            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()
            mask = (uniform_t < accept_prob).float()
            mask_ = mask.unsqueeze(2)
            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll
            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in
                ) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))
        return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]

        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]

        """
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """
        mean, logvar = self.encoder.forward(x)
        return mean

    def eval_inference_dist(self, x, z, param=None):
        """
        Returns: Tensor
            Tensor: the posterior density tensor with
                shape (batch_size, nsamples)
        """
        return self.encoder.eval_inference_dist(x, z, param)

    def calc_mi_q(self, x):
        """Approximate the mutual information between x and z
        under distribution q(z|x)

        Args:
            x: [batch_size, *]. The sampled data to estimate mutual info
        """
        return self.encoder.calc_mi(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jxhe_vae_lagging_encoder(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(GatedMaskedConv2d(*[], **{'in_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(MaskABlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'masked_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MaskedConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(PixelCNNBlock(*[], **{'in_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 2, 2])], {})

    def test_004(self):
        self._check(ResNet(*[], **{'inplanes': 4, 'planes': [4, 4], 'strides': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(ResNetBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ResidualBlock(*[], **{'in_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

