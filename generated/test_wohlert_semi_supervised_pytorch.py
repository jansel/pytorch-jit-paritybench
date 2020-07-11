import sys
_module = sys.modules[__name__]
del sys
betavae = _module
mnist_sslvae = _module
datautils = _module
inference = _module
distributions = _module
variational = _module
layers = _module
flow = _module
stochastic = _module
models = _module
dgm = _module
vae = _module
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


import torch


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


import math


import torch.nn.functional as F


from itertools import repeat


from torch import nn


import torch.nn as nn


from torch.autograd import Variable


from torch.nn import init


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-08) + max


class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """

    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)
    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])
    if x.is_cuda:
        generated = generated
    return Variable(generated.float())


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-08), dim=1)
    return cross_entropy


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True
        xs, ys = x, y
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)
        reconstruction = self.model(xs, ys)
        likelihood = -self.likelihood(reconstruction, xs)
        prior = -log_standard_categorical(ys)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)
        if is_labelled:
            return torch.mean(L)
        logits = self.model.classify(x)
        L = L.view_as(logits.t()).t()
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-08)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)
        U = L + H
        return torch.mean(U)


class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """

    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2)
        zwb = torch.mv(z, self.w) + self.b
        f_z = z + uhat.view(1, -1) * F.tanh(zwb).view(-1, 1)
        psi = (1 - F.tanh(zwb) ** 2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-08)
        return f_z, logdet_jacobian


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """

    def __init__(self, in_features, flow_type=PlanarNormalizingFlow, n_flows=1):
        super(NormalizingFlows, self).__init__()
        self.flows = nn.ModuleList([flow_type(in_features) for _ in range(n_flows)])

    def forward(self, z):
        log_det_jacobian = []
        for flow in self.flows:
            z, j = flow(z)
            log_det_jacobian.append(j)
        return z, sum(log_det_jacobian)


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        if mu.is_cuda:
            epsilon = epsilon
        std = log_var.mul(0.5).exp_()
        z = mu.addcmul(std, epsilon)
        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))
        return self.reparametrize(mu, log_var), mu, log_var


class GaussianMerge(GaussianSample):
    """
    Precision weighted merging of two Gaussian
    distributions.
    Merges information from z into the given
    mean and log variance and produces
    a sample from this new distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianMerge, self).__init__(in_features, out_features)

    def forward(self, z, mu1, log_var1):
        mu2 = self.mu(z)
        log_var2 = F.softplus(self.log_var(z))
        precision1, precision2 = 1 / torch.exp(log_var1), 1 / torch.exp(log_var2)
        mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)
        var = 1 / (precision1 + precision2)
        log_var = torch.log(var + 1e-08)
        return self.reparametrize(mu, log_var), mu, log_var


class GumbelSoftmax(Stochastic):
    """
    Layer that represents a sample from a categorical
    distribution. Enables sampling and stochastic
    backpropagation using the Gumbel-Softmax trick.
    """

    def __init__(self, in_features, out_features, n_distributions):
        super(GumbelSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_distributions = n_distributions
        self.logits = nn.Linear(in_features, n_distributions * out_features)

    def forward(self, x, tau=1.0):
        logits = self.logits(x).view(-1, self.n_distributions)
        softmax = F.softmax(logits, dim=-1)
        sample = self.reparametrize(logits, tau).view(-1, self.n_distributions, self.out_features)
        sample = torch.mean(sample, dim=1)
        return sample, softmax

    def reparametrize(self, logits, tau=1.0):
        epsilon = Variable(torch.rand(logits.size()), requires_grad=False)
        if logits.is_cuda:
            epsilon = epsilon
        gumbel = -torch.log(-torch.log(epsilon + 1e-08) + 1e-08)
        y = F.softmax((logits + gumbel) / tau, dim=1)
        return y


class Classifier(nn.Module):

    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class Perceptron(nn.Module):

    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation
        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)
        return x


class Encoder(nn.Module):

    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):

    def __init__(self, dims):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = -0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu) ** 2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


class VariationalAutoencoder(nn.Module):

    def __init__(self, dims):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()
        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None
        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        mu, log_var = q_param
        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            mu, log_var = p_param
            pz = log_gaussian(z, mu, log_var)
        kl = qz - pz
        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))
        x_mu = self.decoder(z)
        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class GumbelAutoencoder(nn.Module):

    def __init__(self, dims, n_samples=100):
        super(GumbelAutoencoder, self).__init__()
        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.encoder = Perceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), x_dim], output_activation=F.sigmoid)
        self.kl_divergence = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        k = Variable(torch.FloatTensor([self.z_dim]), requires_grad=False)
        kl = qz * (torch.log(qz + 1e-08) - torch.log(1.0 / k))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1):
        x = self.encoder(x)
        sample, qz = self.sampler(x, tau)
        self.kl_divergence = self._kld(qz)
        x_mu = self.decoder(sample)
        return x_mu

    def sample(self, z):
        return self.decoder(z)


class LadderEncoder(nn.Module):

    def __init__(self, dims):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim
        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
        return x, self.sample(x)


class LadderDecoder(nn.Module):

    def __init__(self, dims):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()
        [self.z_dim, h_dim, x_dim] = dims
        self.linear1 = nn.Linear(x_dim, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, self.z_dim)
        self.linear2 = nn.Linear(x_dim, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            z = self.linear1(x)
            z = F.leaky_relu(self.batchnorm1(z), 0.1)
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)
        z = self.linear2(x)
        z = F.leaky_relu(self.batchnorm2(z), 0.1)
        z, p_mu, p_log_var = self.sample(z)
        if l_mu is None:
            return z
        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVariationalAutoencoder(VariationalAutoencoder):

    def __init__(self, dims):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        [x_dim, z_dim, h_dim] = dims
        super(LadderVariationalAutoencoder, self).__init__([x_dim, z_dim[0], h_dim])
        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]
        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0], h_dim, x_dim])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        latents = []
        for encoder in self.encoder:
            x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))
        latents = list(reversed(latents))
        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += self._kld(z, (l_mu, l_log_var))
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += self._kld(*kl)
        x_mu = self.reconstruction(z)
        return x_mu

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier,
     lambda: ([], {'dims': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianMerge,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianSample,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GumbelSoftmax,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'n_distributions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LadderDecoder,
     lambda: ([], {'dims': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LadderEncoder,
     lambda: ([], {'dims': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (NormalizingFlows,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Perceptron,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PlanarNormalizingFlow,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_wohlert_semi_supervised_pytorch(_paritybench_base):
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

