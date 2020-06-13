import sys
_module = sys.modules[__name__]
del sys
main_experiment = _module
VAE = _module
models = _module
flows = _module
layers = _module
optimization = _module
loss = _module
training = _module
utils = _module
distributions = _module
load_data = _module
log_likelihood = _module
plotting = _module
visual_evaluation = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import numpy as np


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type
        if self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
            self.last_kernel_size = 7
        elif self.input_size == [1, 28, 20]:
            self.last_kernel_size = 7, 5
        else:
            raise ValueError('invalid input size!!')
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.q_z_nn_output_dim = 256
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """
        if self.input_type == 'binary':
            q_z_nn = nn.Sequential(GatedConv2d(self.input_size[0], 32, 5, 1,
                2), GatedConv2d(32, 32, 5, 2, 2), GatedConv2d(32, 64, 5, 1,
                2), GatedConv2d(64, 64, 5, 2, 2), GatedConv2d(64, 64, 5, 1,
                2), GatedConv2d(64, 256, self.last_kernel_size, 1, 0))
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(nn.Linear(256, self.z_size), nn.Softplus())
            return q_z_nn, q_z_mean, q_z_var
        elif self.input_type == 'multinomial':
            act = None
            q_z_nn = nn.Sequential(GatedConv2d(self.input_size[0], 32, 5, 1,
                2, activation=act), GatedConv2d(32, 32, 5, 2, 2, activation
                =act), GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, 2, activation=act), GatedConv2d(
                64, 64, 5, 1, 2, activation=act), GatedConv2d(64, 256, self
                .last_kernel_size, 1, 0, activation=act))
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(nn.Linear(256, self.z_size), nn.
                Softplus(), nn.Hardtanh(min_val=0.01, max_val=7.0))
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """
        num_classes = 256
        if self.input_type == 'binary':
            p_x_nn = nn.Sequential(GatedConvTranspose2d(self.z_size, 64,
                self.last_kernel_size, 1, 0), GatedConvTranspose2d(64, 64, 
                5, 1, 2), GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2), GatedConvTranspose2d
                (32, 32, 5, 2, 2, 1), GatedConvTranspose2d(32, 32, 5, 1, 2))
            p_x_mean = nn.Sequential(nn.Conv2d(32, self.input_size[0], 1, 1,
                0), nn.Sigmoid())
            return p_x_nn, p_x_mean
        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(GatedConvTranspose2d(self.z_size, 64,
                self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act))
            p_x_mean = nn.Sequential(nn.Conv2d(32, 256, 5, 1, 2), nn.Conv2d
                (256, self.input_size[0] * num_classes, 1, 1, 0))
            return p_x_nn, p_x_mean
        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """
        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """
        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, self.log_det_j, z, z


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):
        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """
        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        zk = zk.unsqueeze(2)
        uw = torch.bmm(w, u)
        m_uw = -1.0 + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + (m_uw - uw) * w.transpose(2, 1) / w_norm_sq
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return z, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):
        super(Sylvester, self).__init__()
        self.num_ortho_vecs = num_ortho_vecs
        self.h = nn.Tanh()
        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs),
            diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        zk = zk.unsqueeze(1)
        diag_r1 = r1[:, (self.diag_idx), (self.diag_idx)]
        diag_r2 = r2[:, (self.diag_idx), (self.diag_idx)]
        r1_hat = r1
        r2_hat = r2
        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)
        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j
        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):
        super(TriangularSylvester, self).__init__()
        self.z_size = z_size
        self.h = nn.Tanh()
        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        zk = zk.unsqueeze(1)
        diag_r1 = r1[:, (self.diag_idx), (self.diag_idx)]
        diag_r2 = r2[:, (self.diag_idx), (self.diag_idx)]
        if permute_z is not None:
            z_per = zk[:, :, (permute_z)]
        else:
            z_per = zk
        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))
        if permute_z is not None:
            z = z[:, :, (permute_z)]
        z += zk
        z = z.squeeze(1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j
        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.

     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50,
        forget_bias=1.0, conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)
        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())
            if torch.cuda.is_available():
                z_feats = z_feats
                zh_feats = zh_feats
                linear_mean = linear_mean
                linear_std = linear_std
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))
        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):
        logdets = 0.0
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                z = z[:, (self.flip_idx)]
            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GatedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class GatedConvTranspose2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding, output_padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels, output_channels,
            kernel_size, stride, padding, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels,
            kernel_size, stride, padding, output_padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class MaskedLinear(nn.Module):
    """
    Creates masked linear layer for MLP MADE.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, diagonal_zeros=False,
        bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0
        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)
        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ', diagonal_zeros=' + str(
            self.diagonal_zeros) + ', bias=' + str(bias) + ')'


class MaskedConv2d(nn.Module):
    """
    Creates masked convolutional autoregressive layer for pixelCNN.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, size_kernel=(3, 3),
        diagonal_zeros=False, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(out_features, in_features,
            *self.size_kernel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_out % n_in == 0 or n_in % n_out == 0, '%d - %d' % (n_in, n_out
            )
        l = (self.size_kernel[0] - 1) // 2
        m = (self.size_kernel[1] - 1) // 2
        mask = np.ones((n_out, n_in, self.size_kernel[0], self.size_kernel[
            1]), dtype=np.float32)
        mask[:, :, :l, :] = 0
        mask[:, :, (l), :m] = 0
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i * k:(i + 1) * k, i + 1:, (l), (m)] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1, (l), (m)] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[i:i + 1, (i + 1) * k:, (l), (m)] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k, (l), (m)] = 0
        return mask

    def forward(self, x):
        output = F.conv2d(x, self.mask * self.weight, bias=self.bias,
            padding=(1, 1))
        return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ', diagonal_zeros=' + str(
            self.diagonal_zeros) + ', bias=' + str(bias
            ) + ', size_kernel=' + str(self.size_kernel) + ')'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_riannevdberg_sylvester_flows(_paritybench_base):
    pass
    def test_000(self):
        self._check(GatedConv2d(*[], **{'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(GatedConvTranspose2d(*[], **{'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(MaskedConv2d(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MaskedLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4])], {})

    def test_005(self):
        self._check(Planar(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

