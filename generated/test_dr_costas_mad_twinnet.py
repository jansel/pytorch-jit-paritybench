import sys
_module = sys.modules[__name__]
del sys
helpers = _module
arg_parsing = _module
audio_io = _module
data_feeder = _module
printing = _module
settings = _module
signal_transforms = _module
modules = _module
_affine_transform = _module
_fnn = _module
_fnn_denoiser = _module
_masker = _module
_rnn_dec = _module
_rnn_enc = _module
_twin_net = _module
_twin_rnn_dec = _module
mad = _module
madtwinnet = _module
objectives = _module
objectives_functions = _module
testing = _module
training = _module
use_me = _module

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


from torch.nn import Module


from torch.nn import Linear


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


from torch.nn.functional import relu


from collections import namedtuple


import torch


from torch.nn import GRU


from torch.nn.init import orthogonal_


import time


from functools import partial


import numpy as np


from torch import cuda


from torch import load


from torch import from_numpy


from torch import no_grad


from torch import optim


from torch import nn


from torch import save


class AffineTransform(Module):

    def __init__(self, input_dim):
        """The affine transform for the TwinNet regularization.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(AffineTransform, self).__init__()
        self._input_dim = input_dim
        self.linear_layer = Linear(self._input_dim, self._input_dim)
        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.linear_layer.weight)
        constant_(self.linear_layer.bias, 0)

    def forward(self, h_j_dec):
        """Forward pass.

        :param h_j_dec: The output from the RNN decoder.
        :type h_j_dec: torch.Tensor
        :return: The output of the affine transform.
        :rtype: torch.Tensor
        """
        return self.linear_layer(h_j_dec)


class FNNMasker(Module):

    def __init__(self, input_dim, output_dim, context_length):
        """The FNN of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param output_dim: The output dimensionality.
        :type output_dim: int
        :param context_length: The context length.
        :type context_length: int
        """
        super(FNNMasker, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._context_length = context_length
        self.linear_layer = Linear(in_features=self._input_dim, out_features=self._output_dim, bias=True)
        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.linear_layer.weight)
        constant_(self.linear_layer.bias, 0)

    def forward(self, h_j_dec, v_in):
        """Forward pass.

        :param h_j_dec: The output from the RNN decoder.
        :type h_j_dec: torch.Tensor
        :param v_in: The original magnitude spectrogram input.
        :type v_in: torch.Tensor
        :return: The output of the AffineTransform of the masker.
        :rtype: torch.Tensor
        """
        v_in_prime = v_in[:, self._context_length:-self._context_length, :]
        m_j = relu(self.linear_layer(h_j_dec))
        v_j_filt_prime = m_j.mul(v_in_prime)
        return v_j_filt_prime


class FNNDenoiser(Module):

    def __init__(self, input_dim):
        """The FNN enc and FNN dec of the Denoiser.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(FNNDenoiser, self).__init__()
        self._input_dim = input_dim
        self.fnn_enc = Linear(in_features=self._input_dim, out_features=int(self._input_dim / 2), bias=True)
        self.fnn_dec = Linear(in_features=int(self._input_dim / 2), out_features=self._input_dim, bias=True)
        self.initialize_module()

    def initialize_module(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.fnn_enc.weight)
        constant_(self.fnn_enc.bias, 0)
        xavier_normal_(self.fnn_dec.weight)
        constant_(self.fnn_dec.bias, 0)

    def forward(self, v_j_filt_prime):
        """The forward pass.

        :param v_j_filt_prime: The output of the Masker.
        :type v_j_filt_prime: torch.Tensor
        :return: The output of the Denoiser.
        :rtype: torch.Tensor
        """
        fnn_enc_output = relu(self.fnn_enc(v_j_filt_prime))
        fnn_dec_output = relu(self.fnn_dec(fnn_enc_output))
        v_j_filt = fnn_dec_output.mul(v_j_filt_prime)
        return v_j_filt


class Masker(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim, context_length, original_input_dim):
        """The Masker module of the MaD TwinNet.

        :param rnn_enc_input_dim: The input dimensionality for                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality for                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param context_length: The amount of time steps used for                               context length.
        :type context_length: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        """
        super(Masker, self).__init__()
        self.rnn_enc = _rnn_enc.RNNEnc(input_dim=rnn_enc_input_dim, context_length=context_length)
        self.rnn_dec = _rnn_dec.RNNDec(input_dim=rnn_dec_input_dim)
        self.fnn = _fnn.FNNMasker(input_dim=rnn_dec_input_dim, output_dim=original_input_dim, context_length=context_length)
        self.output = namedtuple(typename='masker_output', field_names=['h_enc', 'h_dec', 'v_j_filt_prime'])

    def forward(self, x):
        """Forward pass of the Masker.

        :param x: The input to the Masker.
        :type x: torch.Tensor
        :return: The outputs of the RNN encoder,                 RNN decoder, and the FNN.
        :rtype: collections.namedtuple
        """
        h_enc = self.rnn_enc(x)
        h_dec = self.rnn_dec(h_enc)
        return self.output(h_enc, h_dec, self.fnn(h_dec, x))


class RNNDec(Module):

    def __init__(self, input_dim):
        """The RNN dec of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(RNNDec, self).__init__()
        self._input_dim = input_dim
        self.gru_dec = GRU(input_size=self._input_dim, hidden_size=self._input_dim, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.gru_dec.weight_ih_l0)
        orthogonal_(self.gru_dec.weight_hh_l0)
        constant_(self.gru_dec.bias_ih_l0, 0)
        constant_(self.gru_dec.bias_hh_l0, 0)

    def forward(self, h_enc):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.Tensor
        :return: The output of the RNN dec (h_j_dec).
        :rtype: torch.Tensor
        """
        return self.gru_dec(h_enc)[0]


class RNNEnc(Module):

    def __init__(self, input_dim, context_length):
        """The RNN encoder of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param context_length: The context length.
        :type context_length: int
        """
        super(RNNEnc, self).__init__()
        self._input_dim = input_dim
        self._con_len = context_length
        self.gru_enc = GRU(input_size=self._input_dim, hidden_size=self._input_dim, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.initialize_encoder()

    def initialize_encoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.gru_enc.weight_ih_l0)
        orthogonal_(self.gru_enc.weight_hh_l0)
        constant_(self.gru_enc.bias_ih_l0, 0)
        constant_(self.gru_enc.bias_hh_l0, 0)
        xavier_normal_(self.gru_enc.weight_ih_l0_reverse)
        orthogonal_(self.gru_enc.weight_hh_l0_reverse)
        constant_(self.gru_enc.bias_ih_l0_reverse, 0)
        constant_(self.gru_enc.bias_hh_l0_reverse, 0)

    def forward(self, v_in):
        """Forward pass.

        :param v_in: The input to the RNN encoder of the Masker.
        :type v_in: torch.Torch
        :return: The output of the Masker.
        :rtype: torch.Torch
        """
        v_tr = v_in[:, :, :self._input_dim]
        rnn_output = self.gru_enc(v_tr)[0]
        rnn_output = rnn_output[:, self._con_len:-self._con_len, :]
        return rnn_output + torch.cat([v_tr[:, self._con_len:-self._con_len, :], v_tr[:, self._con_len:-self._con_len, :].flip([1, 2])], dim=-1)


class TwinNet(Module):

    def __init__(self, rnn_dec_input_dim, original_input_dim, context_length):
        super(TwinNet, self).__init__()
        self.rnn_dec = _twin_rnn_dec.TwinRNNDec(input_dim=rnn_dec_input_dim)
        self.fnn = _fnn.FNNMasker(input_dim=rnn_dec_input_dim, output_dim=original_input_dim, context_length=context_length)
        self.output = namedtuple('twin_net_output', ['h_dec_twin', 'v_j_filt_prime_twin'])

    def forward(self, h_enc, x):
        """The forward pass of the TwinNet.

        :param h_enc: The input to the TwinNet.
        :type h_enc: torch.Tensor
        :param x: The original input to the non-twin                  counterpart.
        :type x: torch.Tensor
        :return: The output of the TwinNet.
        :rtype: torch.Tensor
        """
        h_dec_twin = self.rnn_dec(h_enc)
        return self.output(h_dec_twin, self.fnn(h_dec_twin, x))


class TwinRNNDec(Module):

    def __init__(self, input_dim):
        """The RNN decoder of the TwinNet.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(TwinRNNDec, self).__init__()
        self._input_dim = input_dim
        self.gru_dec = GRU(input_size=self._input_dim, hidden_size=self._input_dim, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.gru_dec.weight_ih_l0)
        orthogonal_(self.gru_dec.weight_hh_l0)
        constant_(self.gru_dec.bias_ih_l0, 0.0)
        constant_(self.gru_dec.bias_hh_l0, 0.0)

    def forward(self, h_enc):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.Tensor
        :return: The output of the TwinNet RNN decoder.
        :rtype: torch.Tensor
        """
        return self.gru_dec(h_enc.flip([1, 2]))[0].flip([1, 2])


class MaD(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim, context_length, original_input_dim):
        super(MaD, self).__init__()
        self.masker = Masker(rnn_enc_input_dim=rnn_enc_input_dim, rnn_dec_input_dim=rnn_dec_input_dim, context_length=context_length, original_input_dim=original_input_dim)
        self.denoiser = FNNDenoiser(input_dim=original_input_dim)
        self.output = namedtuple('mad_output', ['v_j_filt_prime', 'v_j_filt', 'h_enc', 'h_dec'])

    def forward(self, x):
        """The forward pass of the MaD.

        :param x: The input to the MaD.
        :type x: torch.Tensor
        :return: The output of the MaD. The                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
                   - `h_enc`, the output of the RNN encoder
                   - `h_dec`, the output of the RNN decoder
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor                torch.Tensor, torch.Tensor]
        """
        m_out = self.masker(x)
        v_j_filt = self.denoiser(m_out.v_j_filt_prime)
        return self.output(m_out.v_j_filt_prime, v_j_filt, m_out.h_enc, m_out.h_dec)


class MaDTwinNet(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim, original_input_dim, context_length):
        """The MaD TwinNet as a module.

        This class implements the MaD TwinNet as a module        and it is based on the separate modules of MaD and        TwinNet.

        :param rnn_enc_input_dim: The input dimensionality of                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality of                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        :param context_length: The amount of time frames used as                               context.
        :type context_length: int
        """
        super(MaDTwinNet, self).__init__()
        self.mad = MaD(rnn_enc_input_dim=rnn_enc_input_dim, rnn_dec_input_dim=rnn_dec_input_dim, context_length=context_length, original_input_dim=original_input_dim)
        self.twin_net = TwinNet(rnn_dec_input_dim=rnn_dec_input_dim, original_input_dim=original_input_dim, context_length=context_length)
        self.affine = AffineTransform(input_dim=rnn_dec_input_dim)
        self.output = namedtuple('mad_twin_net_output', ['v_j_filt_prime', 'v_j_filt', 'v_j_filt_prime_twin', 'affine_output', 'h_dec_twin'])

    def forward(self, x):
        """The forward pass of the MaD TwinNet.

        :param x: The input to the MaD TwinNet.
        :type x: torch.Tensor
        :return: The output of the MaD TwinNet. The                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
                   - `v_j_filt_prime_twin`, the output of the                     TwinNet FNN
                   - `affine_output`, the output of the affine                     transform for the TwinNet regularization
                   - `h_dec_twin`, the output of the RNN of the                     TwinNet
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor                torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mad_out = self.mad(x)
        twin_net_out = self.twin_net(mad_out.h_enc, x)
        affine = self.affine(mad_out.h_dec)
        return self.output(mad_out.v_j_filt_prime, mad_out.v_j_filt, twin_net_out.v_j_filt_prime_twin, affine, twin_net_out.h_dec_twin)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineTransform,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FNNDenoiser,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FNNMasker,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'context_length': 4}),
     lambda: ([torch.rand([4, 0, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNNDec,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RNNEnc,
     lambda: ([], {'input_dim': 4, 'context_length': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (TwinRNNDec,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_dr_costas_mad_twinnet(_paritybench_base):
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

