import sys
_module = sys.modules[__name__]
del sys
ase_calculator = _module
fit_scaling = _module
gemnet = _module
initializers = _module
atom_update_block = _module
base_layers = _module
basis_layers = _module
basis_utils = _module
efficient = _module
embedding_block = _module
envelope = _module
interaction_block = _module
scaling = _module
utils = _module
data_container = _module
data_provider = _module
ema_decay = _module
metrics = _module
schedules = _module
trainer = _module
setup = _module
train_seml = _module

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


import numpy as np


import logging


import torch


import scipy.sparse as sp


import functools


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data.sampler import SequentialSampler


from typing import Iterable


from typing import Optional


import copy


from torch.optim.lr_scheduler import LambdaLR


import string


import random


import time


from torch.utils.tensorboard import SummaryWriter


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(93, emb_size)
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)
        return h


def read_json(path):
    """ """
    if not path.endswith('.json'):
        raise UserWarning(f'Path {path} is not a json-path.')
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """
    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable
        self.scale_file = scale_file
        self._name = name
        self._fitted = False
        self.load_maybe()
        if AutomaticFit.fitting_mode and not self._fitted:
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []
            else:
                self._add2queue()

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logging.debug(f'Add {self._name} to queue.')
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(f'Variable with the same name ({self._name}) was already added to queue!')
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logging.debug('Processed all variables.')
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logging.info(f"Initialize variable {self._name}' to {self.variable.numpy():.3f}")
        else:
            self._fitted = True
            logging.debug(f'Set scale factor {self._name} : {value}')
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p, name='envelope'):
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = 1 + self.a * d_scaled ** self.p + self.b * d_scaled ** (self.p + 1) + self.c * d_scaled ** (self.p + 2)
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class BesselBasisLayer(torch.nn.Module):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int=5, name='bessel_basis'):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = torch.nn.Parameter(data=torch.Tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)), requires_grad=True)

    def forward(self, d):
        d = d[:, None]
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d


class ScaledSiLU(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-06
    if len(kernel.shape) == 3:
        axis = [0, 1]
    else:
        axis = 1
    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)
    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]
    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5
    return tensor


class Dense(torch.nn.Module):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None, name=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ['swish', 'silu']:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError('Activation function not implemented for GemNet (yet).')

    def reset_parameters(self):
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(self, atom_features, edge_features, out_features, activation=None, name=None):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idnb_a, idnb_c):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_a = h[idnb_a]
        h_c = h[idnb_c]
        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)
        m_ca = self.dense(m_ca)
        return m_ca


class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        num_spherical: int
            Same as the setting in the basis layers.
        num_radial: int
            Same as the setting in the basis layers.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
    """

    def __init__(self, num_spherical: int, num_radial: int, emb_size_interm: int, name='EfficientDownProj'):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)), requires_grad=True)
        he_orthogonal_init(self.weight)

    def forward(self, tbf):
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        rbf_env, sph = tbf
        rbf_W1 = torch.matmul(rbf_env, self.weight)
        rbf_W1 = rbf_W1.permute(1, 2, 0)
        sph = torch.transpose(sph, 1, 2)
        return rbf_W1, sph


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
    """

    def __init__(self, units: int, nLayers: int=2, activation=None, name=None):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(*[Dense(units, units, activation=activation, bias=False) for i in range(nLayers)])
        self.inv_sqrt_2 = 1 / 2.0 ** 0.5

    def forward(self, inputs):
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x


def write_json(path, data):
    """ """
    if not path.endswith('.json'):
        raise UserWarning(f'Path {path} is not a json-path.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_json(path, data):
    """ """
    if not path.endswith('.json'):
        raise UserWarning(f'Path {path} is not a json-path.')
    content = read_json(path)
    content.update(data)
    write_json(path, content)


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: tf.Variable
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)
        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    def observe(self, x, y):
        """
        Observe variances for inut x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += torch.mean(torch.var(x, dim=0)) * nSamples
            self.variance_out += torch.mean(torch.var(y, dim=0)) * nSamples
            self.nSamples += nSamples

    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(f'Did not track the variable {self._name}. Add observe calls to track the variance before and after.')
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples
            ratio = self.variance_out / self.variance_in
            value = np.sqrt(1 / ratio, dtype='float32')
            logging.info(f'Variable: {self._name}, Var_in: {self.variance_in.numpy():.3f}, Var_out: {self.variance_out.numpy():.3f}, ' + f'Ratio: {ratio:.3f} => Scaling factor: {value:.3f}')
            with torch.no_grad():
                self.variable.copy_(self.variable * value)
            update_json(self.scale_file, {self._name: float(self.variable.numpy())})
            self.set_next_active()


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)
        return y


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, activation=None, scale_file=None, name: str='atom_update'):
        super().__init__()
        self.name = name
        self.emb_size_edge = emb_size_edge
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + '_sum')
        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(self, units, nHidden, activation):
        dense1 = Dense(self.emb_size_edge, units, activation=activation, bias=False)
        res = [ResidualLayer(units, nLayers=2, activation=activation) for i in range(nHidden)]
        mlp = [dense1] + res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]
        mlp_rbf = self.dense_rbf(rbf)
        x = m * mlp_rbf
        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce='add')
        x = self.scale_sum(m, x2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        emb_size: int
            Edge embedding size.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(self, emb_size: int, emb_size_interm: int, units_out: int, name='EfficientBilinear'):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.empty((self.emb_size, self.emb_size_interm, self.units_out), requires_grad=True))
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, Kidx):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        rbf_W1, sph = basis
        nEdges = rbf_W1.shape[0]
        Kmax = 0 if sph.shape[2] == 0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m
        sum_k = torch.matmul(sph, m2)
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)
        m_ca = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)
        m_ca = torch.sum(m_ca, dim=0)
        return m_ca


class QuadrupletInteraction(torch.nn.Module):
    """
    Quadruplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_quad: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_edge, emb_size_quad, emb_size_bilinear, emb_size_rbf, emb_size_cbf, emb_size_sbf, activation=None, scale_file=None, name='QuadrupletInteraction', **kwargs):
        super().__init__()
        self.name = name
        self.dense_db = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False, name='dense_db')
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, name='MLP_rbf4_2', bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + '_had_rbf')
        self.mlp_cbf = Dense(emb_size_cbf, emb_size_quad, activation=None, name='MLP_cbf4_2', bias=False)
        self.scale_cbf = ScalingFactor(scale_file=scale_file, name=name + '_had_cbf')
        self.mlp_sbf = EfficientInteractionBilinear(emb_size_quad, emb_size_sbf, emb_size_bilinear, name='MLP_sbf4_2')
        self.scale_sbf_sum = ScalingFactor(scale_file=scale_file, name=name + '_sum_sbf')
        self.down_projection = Dense(emb_size_edge, emb_size_quad, activation=activation, bias=False, name='dense_down')
        self.up_projection_ca = Dense(emb_size_bilinear, emb_size_edge, activation=activation, bias=False, name='dense_up_ca')
        self.up_projection_ac = Dense(emb_size_bilinear, emb_size_edge, activation=activation, bias=False, name='dense_up_ac')
        self.inv_sqrt_2 = 1 / 2.0 ** 0.5

    def forward(self, m, rbf, cbf, sbf, Kidx4, id_swap, id4_reduce_ca, id4_expand_intm_db, id4_expand_abd):
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_db = self.dense_db(m)
        x_db2 = x_db * self.mlp_rbf(rbf)
        x_db = self.scale_rbf(x_db, x_db2)
        x_db = self.down_projection(x_db)
        x_db = x_db[id4_expand_intm_db]
        x_db2 = x_db * self.mlp_cbf(cbf)
        x_db = self.scale_cbf(x_db, x_db2)
        x_db = x_db[id4_expand_abd]
        x = self.mlp_sbf(sbf, x_db, id4_reduce_ca, Kidx4)
        x = self.scale_sbf_sum(x_db, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x4 = x_ca + x_ac
        x4 = x4 * self.inv_sqrt_2
        return x4


class TripletInteraction(torch.nn.Module):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_edge, emb_size_trip, emb_size_bilinear, emb_size_rbf, emb_size_cbf, activation=None, scale_file=None, name='TripletInteraction', **kwargs):
        super().__init__()
        self.name = name
        self.dense_ba = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False, name='dense_ba')
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, name='MLP_rbf3_2', bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + '_had_rbf')
        self.mlp_cbf = EfficientInteractionBilinear(emb_size_trip, emb_size_cbf, emb_size_bilinear, name='MLP_cbf3_2')
        self.scale_cbf_sum = ScalingFactor(scale_file=scale_file, name=name + '_sum_cbf')
        self.down_projection = Dense(emb_size_edge, emb_size_trip, activation=activation, bias=False, name='dense_down')
        self.up_projection_ca = Dense(emb_size_bilinear, emb_size_edge, activation=activation, bias=False, name='dense_up_ca')
        self.up_projection_ac = Dense(emb_size_bilinear, emb_size_edge, activation=activation, bias=False, name='dense_up_ac')
        self.inv_sqrt_2 = 1 / 2.0 ** 0.5

    def forward(self, m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca):
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ba = self.dense_ba(m)
        mlp_rbf = self.mlp_rbf(rbf3)
        x_ba2 = x_ba * mlp_rbf
        x_ba = self.scale_rbf(x_ba, x_ba2)
        x_ba = self.down_projection(x_ba)
        x_ba = x_ba[id3_expand_ba]
        x = self.mlp_cbf(cbf3, x_ba, id3_reduce_ca, Kidx3)
        x = self.scale_cbf_sum(x_ba, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3


class InteractionBlock(torch.nn.Module):
    """
    Interaction block for GemNet-Q/dQ.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_atom, emb_size_edge, emb_size_trip, emb_size_quad, emb_size_rbf, emb_size_cbf, emb_size_sbf, emb_size_bil_trip, emb_size_bil_quad, num_before_skip, num_after_skip, num_concat, num_atom, activation=None, scale_file=None, name='Interaction'):
        super().__init__()
        self.name = name
        block_nr = name.split('_')[-1]
        self.dense_ca = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False, name='dense_ca')
        self.quad_interaction = QuadrupletInteraction(emb_size_edge=emb_size_edge, emb_size_quad=emb_size_quad, emb_size_bilinear=emb_size_bil_quad, emb_size_rbf=emb_size_rbf, emb_size_cbf=emb_size_cbf, emb_size_sbf=emb_size_sbf, activation=activation, scale_file=scale_file, name=f'QuadInteraction_{block_nr}')
        self.trip_interaction = TripletInteraction(emb_size_edge=emb_size_edge, emb_size_trip=emb_size_trip, emb_size_bilinear=emb_size_bil_trip, emb_size_rbf=emb_size_rbf, emb_size_cbf=emb_size_cbf, activation=activation, scale_file=scale_file, name=f'TripInteraction_{block_nr}')
        self.layers_before_skip = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_bef_skip_{i}') for i in range(num_before_skip)])
        self.layers_after_skip = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_aft_skip_{i}') for i in range(num_after_skip)])
        self.atom_update = AtomUpdateBlock(emb_size_atom=emb_size_atom, emb_size_edge=emb_size_edge, emb_size_rbf=emb_size_rbf, nHidden=num_atom, activation=activation, scale_file=scale_file, name=f'AtomUpdate_{block_nr}')
        self.concat_layer = EdgeEmbedding(emb_size_atom, emb_size_edge, emb_size_edge, activation=activation, name='concat')
        self.residual_m = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_m_{i}') for i in range(num_concat)])
        self.inv_sqrt_2 = 1 / 2.0 ** 0.5
        self.inv_sqrt_3 = 1 / 3.0 ** 0.5

    def forward(self, h, m, rbf4, cbf4, sbf4, Kidx4, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca, id4_reduce_ca, id4_expand_intm_db, id4_expand_abd, rbf_h, id_c, id_a):
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ca_skip = self.dense_ca(m)
        x4 = self.quad_interaction(m, rbf4, cbf4, sbf4, Kidx4, id_swap, id4_reduce_ca, id4_expand_intm_db, id4_expand_abd)
        x3 = self.trip_interaction(m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca)
        x = x_ca_skip + x3 + x4
        x = x * self.inv_sqrt_3
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, id_a)
        h = h + h2
        h = h * self.inv_sqrt_2
        m2 = self.concat_layer(h, m, id_c, id_a)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class InteractionBlockTripletsOnly(torch.nn.Module):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_atom, emb_size_edge, emb_size_trip, emb_size_quad, emb_size_rbf, emb_size_cbf, emb_size_bil_trip, num_before_skip, num_after_skip, num_concat, num_atom, activation=None, scale_file=None, name='Interaction', **kwargs):
        super().__init__()
        self.name = name
        block_nr = name.split('_')[-1]
        self.dense_ca = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False, name='dense_ca')
        self.trip_interaction = TripletInteraction(emb_size_edge=emb_size_edge, emb_size_trip=emb_size_trip, emb_size_bilinear=emb_size_bil_trip, emb_size_rbf=emb_size_rbf, emb_size_cbf=emb_size_cbf, activation=activation, scale_file=scale_file, name=f'TripInteraction_{block_nr}')
        self.layers_before_skip = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_bef_skip_{i}') for i in range(num_before_skip)])
        self.layers_after_skip = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_aft_skip_{i}') for i in range(num_after_skip)])
        self.atom_update = AtomUpdateBlock(emb_size_atom=emb_size_atom, emb_size_edge=emb_size_edge, emb_size_rbf=emb_size_rbf, nHidden=num_atom, activation=activation, scale_file=scale_file, name=f'AtomUpdate_{block_nr}')
        self.concat_layer = EdgeEmbedding(emb_size_atom, emb_size_edge, emb_size_edge, activation=activation, name='concat')
        self.residual_m = torch.nn.ModuleList([ResidualLayer(emb_size_edge, activation=activation, name=f'res_m_{i}') for i in range(num_concat)])
        self.inv_sqrt_2 = 1 / 2.0 ** 0.5

    def forward(self, h, m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca, rbf_h, id_c, id_a, **kwargs):
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ca_skip = self.dense_ca(m)
        x3 = self.trip_interaction(m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca)
        x = x_ca_skip + x3
        x = x * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, id_a)
        h = h + h2
        h = h * self.inv_sqrt_2
        m2 = self.concat_layer(h, m, id_c, id_a)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Activation function to use in the dense layers (except for the final dense layer).
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: str
            Kernel initializer of the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, num_targets: int, activation=None, direct_forces=True, output_init='HeOrthogonal', scale_file=None, name: str='output', **kwargs):
        super().__init__(name=name, emb_size_atom=emb_size_atom, emb_size_edge=emb_size_edge, emb_size_rbf=emb_size_rbf, nHidden=nHidden, activation=activation, scale_file=scale_file, **kwargs)
        assert isinstance(output_init, str)
        self.output_init = output_init
        self.direct_forces = direct_forces
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.seq_energy = self.layers
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)
        if self.direct_forces:
            self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + '_had')
            self.seq_forces = self.get_mlp(emb_size_edge, nHidden, activation)
            self.out_forces = Dense(emb_size_edge, num_targets, bias=False, activation=None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init.lower() == 'heorthogonal':
            he_orthogonal_init(self.out_energy.weight)
            if self.direct_forces:
                he_orthogonal_init(self.out_forces.weight)
        elif self.output_init.lower() == 'zeros':
            torch.nn.init.zeros_(self.out_energy.weight)
            if self.direct_forces:
                torch.nn.init.zeros_(self.out_forces.weight)
        else:
            raise UserWarning(f'Unknown output_init: {self.output_init}')

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: Tensor, shape=(nAtoms, num_targets)
            - F: Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        nAtoms = h.shape[0]
        rbf_mlp = self.dense_rbf(rbf)
        x = m * rbf_mlp
        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce='add')
        x_E = self.scale_sum(m, x_E)
        for i, layer in enumerate(self.seq_energy):
            x_E = layer(x_E)
        x_E = self.out_energy(x_E)
        if self.direct_forces:
            x_F = self.scale_rbf(m, x)
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)
            x_F = self.out_forces(x_F)
        else:
            x_F = 0
        return x_E, x_F


def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')
    j = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).

    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).

    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    z = sym.symbols('z')
    P_l_m = [([0] * (2 * l + 1)) for l in range(L)]
    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l)
            return P_l_m
        else:
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify((1 - 2 * l) * (1 - z ** 2) ** 0.5 * P_l_m[l - 1][l - 1])
            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify((2 * m + 1) * z * P_l_m[m][m])
            for l in range(2, L):
                for m in range(l - 1):
                    P_l_m[l][m] = sym.simplify(((2 * l - 1) * z * P_l_m[l - 1][m] - (l + m - 1) * P_l_m[l - 2][m]) / (l - m))
            if not pos_m_only:
                for l in range(1, L):
                    for m in range(1, l + 1):
                        P_l_m[l][-m] = sym.simplify((-1) ** m * np.math.factorial(l - m) / np.math.factorial(l + m) * P_l_m[l][m])
            return P_l_m


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    return ((2 * l + 1) / (4 * np.pi) * np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m))) ** 0.5


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols('z')
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [([0] * (2 * l + 1)) for l in range(L)]
    if spherical_coordinates:
        theta = sym.symbols('theta')
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))
    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])
    if not zero_m_only:
        phi = sym.symbols('phi')
        for l in range(1, L):
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(2 ** 0.5 * (-1) ** m * sph_harm_prefactor(l, m) * P_l_m[l][m] * sym.cos(m * phi))
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(2 ** 0.5 * (-1) ** m * sph_harm_prefactor(l, -m) * P_l_m[l][m] * sym.sin(m * phi))
        if not spherical_coordinates:
            x = sym.symbols('x')
            y = sym.symbols('y')
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class SphericalBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float, envelope_exponent: int=5, efficient: bool=False, name: str='spherical_basis'):
        super().__init__()
        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=True)
        self.sph_funcs = []
        self.bessel_funcs = []
        self.norm_const = self.inv_cutoff ** 1.5
        self.register_buffer('device_buffer', torch.zeros(0), persistent=False)
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos, 'sqrt': torch.sqrt}
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(lambda theta: torch.zeros_like(theta) + first_sph(theta))
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], bessel_formulas[l][n], modules))

    def forward(self, D_ca, Angle_cab, id3_reduce_ca, Kidx):
        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = torch.stack(rbf, dim=1)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf
        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)
        if not self.efficient:
            rbf_env = rbf_env[id3_reduce_ca]
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            sph = sph.view(-1, self.num_spherical, 1)
            out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
            return out
        else:
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = torch.transpose(rbf_env, 0, 1)
            Kmax = 0 if sph.shape[0] == 0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))
            nEdges = d_scaled.shape[0]
            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical, device=self.device_buffer.device, dtype=sph.dtype)
            sph2[id3_reduce_ca, Kidx] = sph
            return rbf_env, sph2


class TensorBasisLayer(torch.nn.Module):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float, envelope_exponent: int=5, efficient=False, name: str='tensor_basis'):
        super().__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.efficient = efficient
        self.inv_cutoff = 1 / cutoff
        self.envelope = Envelope(envelope_exponent)
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []
        self.norm_const = self.inv_cutoff ** 1.5
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos, 'sqrt': torch.sqrt}
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if l == 0:
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(lambda theta, phi: torch.zeros_like(theta) + first_sph(theta, phi))
                else:
                    self.sph_funcs.append(sym.lambdify([theta, phi], Y_lm[l][m], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], bessel_formulas[l][j], modules))
        self.register_buffer('degreeInOrder', torch.arange(num_spherical) * 2 + 1, persistent=False)

    def forward(self, D_ca, Alpha_cab, Theta_cabd, id4_reduce_ca, Kidx):
        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = torch.stack(rbf, dim=1)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf
        rbf_env = rbf_env.view((-1, self.num_spherical, self.num_radial))
        rbf_env = torch.repeat_interleave(rbf_env, self.degreeInOrder, dim=1)
        if not self.efficient:
            rbf_env = rbf_env.view((-1, self.num_spherical ** 2 * self.num_radial))
            rbf_env = rbf_env[id4_reduce_ca]
        sph = [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)
        if not self.efficient:
            sph = torch.repeat_interleave(sph, self.num_radial, axis=1)
            return rbf_env * sph
        else:
            rbf_env = torch.transpose(rbf_env, 0, 1)
            Kmax = 0 if sph.shape[0] == 0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))
            nEdges = d_scaled.shape[0]
            sph2 = torch.zeros(nEdges, Kmax, self.num_spherical ** 2, device=self.degreeInOrder.device, dtype=sph.dtype)
            sph2[id4_reduce_ca, Kidx] = sph
            return rbf_env, sph2


class GemNet(torch.nn.Module):
    """
    Parameters
    ----------
        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.
        triplets_only: bool
            If True use GemNet-T or GemNet-dT.No quadruplet based message passing.
        num_targets: int
            Number of prediction targets.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        int_cutoff: float
            Interaction cutoff for interactomic directions in Angstrom. No effect for GemNet-(d)T
        envelope_exponent: int
            Exponent of the envelope function. Determines the shape of the smooth cutoff.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        forces_coupled: bool
            No effect if direct_forces is False. If True enforce that |F_ac| = |F_ca|
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, num_spherical: int, num_radial: int, num_blocks: int, emb_size_atom: int, emb_size_edge: int, emb_size_trip: int, emb_size_quad: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_sbf: int, emb_size_bil_quad: int, emb_size_bil_trip: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, triplets_only: bool, num_targets: int=1, direct_forces: bool=False, cutoff: float=5.0, int_cutoff: float=10.0, envelope_exponent: int=5, extensive=True, forces_coupled: bool=False, output_init='HeOrthogonal', activation: str='swish', scale_file=None, name='gemnet', **kwargs):
        super().__init__()
        assert num_blocks > 0
        self.num_targets = num_targets
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.forces_coupled = forces_coupled
        AutomaticFit.reset()
        self.direct_forces = direct_forces
        self.triplets_only = triplets_only
        self.rbf_basis = BesselBasisLayer(num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        if not self.triplets_only:
            self.cbf_basis = SphericalBasisLayer(num_spherical, num_radial, cutoff=int_cutoff, envelope_exponent=envelope_exponent, efficient=False)
            self.sbf_basis = TensorBasisLayer(num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent, efficient=True)
        self.cbf_basis3 = SphericalBasisLayer(num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent, efficient=True)
        if not self.triplets_only:
            self.mlp_rbf4 = Dense(num_radial, emb_size_rbf, activation=None, name='MLP_rbf4_shared', bias=False)
            self.mlp_cbf4 = Dense(num_radial * num_spherical, emb_size_cbf, activation=None, name='MLP_cbf4_shared', bias=False)
            self.mlp_sbf4 = EfficientInteractionDownProjection(num_spherical ** 2, num_radial, emb_size_sbf, name='MLP_sbf4_shared')
        self.mlp_rbf3 = Dense(num_radial, emb_size_rbf, activation=None, name='MLP_rbf3_shared', bias=False)
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf, name='MLP_cbf3_shared')
        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, name='MLP_rbfh_shared', bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, name='MLP_rbfout_shared', bias=False)
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(emb_size_atom, num_radial, emb_size_edge, activation=activation)
        out_blocks = []
        int_blocks = []
        interaction_block = InteractionBlockTripletsOnly if self.triplets_only else InteractionBlock
        for i in range(num_blocks):
            int_blocks.append(interaction_block(emb_size_atom=emb_size_atom, emb_size_edge=emb_size_edge, emb_size_trip=emb_size_trip, emb_size_quad=emb_size_quad, emb_size_rbf=emb_size_rbf, emb_size_cbf=emb_size_cbf, emb_size_sbf=emb_size_sbf, emb_size_bil_trip=emb_size_bil_trip, emb_size_bil_quad=emb_size_bil_quad, num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_concat=num_concat, num_atom=num_atom, activation=activation, scale_file=scale_file, name=f'IntBlock_{i + 1}'))
        for i in range(num_blocks + 1):
            out_blocks.append(OutputBlock(emb_size_atom=emb_size_atom, emb_size_edge=emb_size_edge, emb_size_rbf=emb_size_rbf, nHidden=num_atom, num_targets=num_targets, activation=activation, output_init=output_init, direct_forces=direct_forces, scale_file=scale_file, name=f'OutBlock_{i}'))
        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

    @staticmethod
    def calculate_interatomic_vectors(R, id_s, id_t):
        """
        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_s: Tensor, shape = (nEdges,)
                Indices of the source atom of the edges.
            id_t: Tensor, shape = (nEdges,)
                Indices of the target atom of the edges.

        Returns
        -------
            (D_st, V_st): tuple
                D_st: Tensor, shape = (nEdges,)
                    Distance from atom t to s.
                V_st: Tensor, shape = (nEdges,)
                    Unit direction from atom t to s.
        """
        Rt = R[id_t]
        Rs = R[id_s]
        V_st = Rt - Rs
        D_st = torch.sqrt(torch.sum(V_st ** 2, dim=1))
        V_st = V_st / D_st[..., None]
        return D_st, V_st

    @staticmethod
    def calculate_neighbor_angles(R_ac, R_ab):
        """Calculate angles between atoms c <- a -> b.

        Parameters
        ----------
            R_ac: Tensor, shape = (N,3)
                Vector from atom a to c.
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.

        Returns
        -------
            angle_cab: Tensor, shape = (N,)
                Angle between atoms c <- a -> b.
        """
        x = torch.sum(R_ac * R_ab, dim=1)
        y = torch.cross(R_ac, R_ab).norm(dim=-1)
        y = torch.max(y, torch.tensor(1e-09))
        angle = torch.atan2(y, x)
        return angle

    @staticmethod
    def vector_rejection(R_ab, P_n):
        """
        Project the vector R_ab onto a plane with normal vector P_n.

        Parameters
        ----------
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.
            P_n: Tensor, shape = (N,3)
                Normal vector of a plane onto which to project R_ab.

        Returns
        -------
            R_ab_proj: Tensor, shape = (N,3)
                Projected vector (orthogonal to P_n).
        """
        a_x_b = torch.sum(R_ab * P_n, dim=-1)
        b_x_b = torch.sum(P_n * P_n, dim=-1)
        return R_ab - (a_x_b / b_x_b)[:, None] * P_n

    @staticmethod
    def calculate_angles(R, id_c, id_a, id4_int_b, id4_int_a, id4_expand_abd, id4_reduce_cab, id4_expand_intm_db, id4_reduce_intm_ca, id4_expand_intm_ab, id4_reduce_intm_ab):
        """Calculate angles for quadruplet-based message passing.

        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_c: Tensor, shape = (nEdges,)
                Indices of atom c (source atom of edge).
            id_a: Tensor, shape = (nEdges,)
                Indices of atom a (target atom of edge).
            id4_int_b: torch.Tensor, shape (nInterEdges,)
                Indices of the atom b of the interaction edge.
            id4_int_a: torch.Tensor, shape (nInterEdges,)
                Indices of the atom a of the interaction edge.
            id4_expand_abd: torch.Tensor, shape (nQuadruplets,)
                Indices to map from intermediate d->b to quadruplet d->b.
            id4_reduce_cab: torch.Tensor, shape (nQuadruplets,)
                Indices to map from intermediate c->a to quadruplet c->a.
            id4_expand_intm_db: torch.Tensor, shape (intmTriplets,)
                Indices to map d->b to intermediate d->b.
            id4_reduce_intm_ca: torch.Tensor, shape (intmTriplets,)
                Indices to map c->a to intermediate c->a.
            id4_expand_intm_ab: torch.Tensor, shape (intmTriplets,)
                Indices to map b-a to intermediate b-a of the quadruplet's part a-b<-d.
            id4_reduce_intm_ab: torch.Tensor, shape (intmTriplets,)
                Indices to map b-a to intermediate b-a of the quadruplet's part c->a-b.

        Returns
        -------
            angle_cab: Tensor, shape = (nQuadruplets,)
                Angle between atoms c <- a -> b.
            angle_abd: Tensor, shape = (intmTriplets,)
                Angle between atoms a <- b -> d.
            angle_cabd: Tensor, shape = (nQuadruplets,)
                Angle between atoms c <- a-b -> d.
        """
        Ra = R[id4_int_a[id4_expand_intm_ab]]
        Rb = R[id4_int_b[id4_expand_intm_ab]]
        Rd = R[id_c[id4_expand_intm_db]]
        R_ba = Ra - Rb
        R_bd = Rd - Rb
        angle_abd = GemNet.calculate_neighbor_angles(R_ba, R_bd)
        R_bd_proj = GemNet.vector_rejection(R_bd, R_ba)
        R_bd_proj = R_bd_proj[id4_expand_abd]
        Rc = R[id_c[id4_reduce_intm_ca]]
        Ra = R[id_a[id4_reduce_intm_ca]]
        Rb = R[id4_int_b[id4_reduce_intm_ab]]
        R_ac = Rc - Ra
        R_ab = Rb - Ra
        angle_cab = GemNet.calculate_neighbor_angles(R_ab, R_ac)
        angle_cab = angle_cab[id4_reduce_cab]
        R_ac_proj = GemNet.vector_rejection(R_ac, R_ab)
        R_ac_proj = R_ac_proj[id4_reduce_cab]
        angle_cabd = GemNet.calculate_neighbor_angles(R_ac_proj, R_bd_proj)
        return angle_cab, angle_abd, angle_cabd

    @staticmethod
    def calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba):
        """Calculate angles for triplet-based message passing.

        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_c: Tensor, shape = (nEdges,)
                Indices of atom c (source atom of edge).
            id_a: Tensor, shape = (nEdges,)
                Indices of atom a (target atom of edge).
            id3_reduce_ca: Tensor, shape = (nTriplets,)
                Edge indices of edge c -> a of the triplets.
            id3_expand_ba: Tensor, shape = (nTriplets,)
                Edge indices of edge b -> a of the triplets.

        Returns
        -------
            angle_cab: Tensor, shape = (nTriplets,)
                Angle between atoms c <- a -> b.
        """
        Rc = R[id_c[id3_reduce_ca]]
        Ra = R[id_a[id3_reduce_ca]]
        Rb = R[id_c[id3_expand_ba]]
        R_ac = Rc - Ra
        R_ab = Rb - Ra
        return GemNet.calculate_neighbor_angles(R_ac, R_ab)

    def forward(self, inputs):
        Z, R = inputs['Z'], inputs['R']
        id_a, id_c, id_undir, id_swap = inputs['id_a'], inputs['id_c'], inputs['id_undir'], inputs['id_swap']
        id3_expand_ba, id3_reduce_ca = inputs['id3_expand_ba'], inputs['id3_reduce_ca']
        if not self.triplets_only:
            batch_seg, Kidx4, Kidx3 = inputs['batch_seg'], inputs['Kidx4'], inputs['Kidx3']
            id4_int_b, id4_int_a = inputs['id4_int_b'], inputs['id4_int_a']
            id4_reduce_ca, id4_expand_db = inputs['id4_reduce_ca'], inputs['id4_expand_db']
            id4_reduce_cab, id4_expand_abd = inputs['id4_reduce_cab'], inputs['id4_expand_abd']
            id4_reduce_intm_ca, id4_expand_intm_db = inputs['id4_reduce_intm_ca'], inputs['id4_expand_intm_db']
            id4_reduce_intm_ab, id4_expand_intm_ab = inputs['id4_reduce_intm_ab'], inputs['id4_expand_intm_ab']
        else:
            batch_seg, Kidx4, Kidx3 = inputs['batch_seg'], None, inputs['Kidx3']
            id4_int_b, id4_int_a = None, None
            id4_reduce_ca, id4_expand_db = None, None
            id4_reduce_cab, id4_expand_abd = None, None
            id4_reduce_intm_ca, id4_expand_intm_db = None, None
            id4_reduce_intm_ab, id4_expand_intm_ab = None, None
        if not self.direct_forces:
            inputs['R'].requires_grad = True
        D_ca, V_ca = self.calculate_interatomic_vectors(R, id_c, id_a)
        if not self.triplets_only:
            D_ab, _ = self.calculate_interatomic_vectors(R, id4_int_b, id4_int_a)
            Phi_cab, Phi_abd, Theta_cabd = self.calculate_angles(R, id_c, id_a, id4_int_b, id4_int_a, id4_expand_abd, id4_reduce_cab, id4_expand_intm_db, id4_reduce_intm_ca, id4_expand_intm_ab, id4_reduce_intm_ab)
            cbf4 = self.cbf_basis(D_ab, Phi_abd, id4_expand_intm_ab, None)
            sbf4 = self.sbf_basis(D_ca, Phi_cab, Theta_cabd, id4_reduce_ca, Kidx4)
        rbf = self.rbf_basis(D_ca)
        Angles3_cab = self.calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba)
        cbf3 = self.cbf_basis3(D_ca, Angles3_cab, id3_reduce_ca, Kidx3)
        h = self.atom_emb(Z)
        m = self.edge_emb(h, rbf, id_c, id_a)
        if not self.triplets_only:
            rbf4 = self.mlp_rbf4(rbf)
            cbf4 = self.mlp_cbf4(cbf4)
            sbf4 = self.mlp_sbf4(sbf4)
        else:
            rbf4 = None
            cbf4 = None
            sbf4 = None
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(cbf3)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_a, F_ca = self.out_blocks[0](h, m, rbf_out, id_a)
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](h=h, m=m, rbf4=rbf4, cbf4=cbf4, sbf4=sbf4, Kidx4=Kidx4, rbf3=rbf3, cbf3=cbf3, Kidx3=Kidx3, id_swap=id_swap, id3_expand_ba=id3_expand_ba, id3_reduce_ca=id3_reduce_ca, id4_reduce_ca=id4_reduce_ca, id4_expand_intm_db=id4_expand_intm_db, id4_expand_abd=id4_expand_abd, rbf_h=rbf_h, id_c=id_c, id_a=id_a)
            E, F = self.out_blocks[i + 1](h, m, rbf_out, id_a)
            F_ca += F
            E_a += E
        nMolecules = torch.max(batch_seg) + 1
        if self.extensive:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce='add')
        else:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce='mean')
        if self.direct_forces:
            nAtoms = Z.shape[0]
            if self.forces_coupled:
                nEdges = id_c.shape[0]
                F_ca = scatter(F_ca, id_undir, dim=0, dim_size=int(nEdges / 2), reduce='mean')
                F_ca = F_ca[id_undir]
            F_ji = F_ca[:, :, None] * V_ca[:, None, :]
            F_j = scatter(F_ji, id_a, dim=0, dim_size=nAtoms, reduce='add')
        else:
            if self.num_targets > 1:
                forces = []
                for i in range(self.num_targets):
                    forces += [-torch.autograd.grad(E_a[:, i].sum(), inputs['R'], create_graph=True)[0]]
                F_j = torch.stack(forces, dim=1)
            else:
                F_j = -torch.autograd.grad(E_a.sum(), inputs['R'], create_graph=True)[0]
            inputs['R'].requires_grad = False
        return E_a, F_j

    def load_tfmodel(self, path):
        reader = tf.train.load_checkpoint(path)

        def copy_(src, name):
            W = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            if name[-12:] == 'scale_factor':
                W = torch.tensor(W)
            else:
                W = torch.from_numpy(W)
            if name[-6:] == 'kernel':
                if len(W.shape) == 2:
                    W = W.t()
            src.data.copy_(W)
        copy_(self.rbf_basis.frequencies, 'rbf_basis/frequencies')
        copy_(self.atom_emb.embeddings.weight, 'atom_emb/embeddings')
        copy_(self.edge_emb.dense.weight, 'edge_emb/dense/kernel')
        shared_mlps = ['mlp_cbf3', 'mlp_rbf3', 'mlp_rbf_h', 'mlp_rbf_out']
        if not self.triplets_only:
            shared_mlps += ['mlp_rbf4', 'mlp_cbf4', 'mlp_sbf4']
        for layer in shared_mlps:
            copy_(getattr(self, layer).weight, f'{layer}/kernel')
        for i, block in enumerate(self.int_blocks):
            if not self.triplets_only:
                for layer in ['dense_db', 'mlp_rbf', 'mlp_cbf', 'mlp_sbf', 'down_projection', 'up_projection_ca', 'up_projection_ac']:
                    copy_(getattr(block.quad_interaction, layer).weight, f'int_blocks/{i}/quad_interaction/{layer}/kernel')
                for layer in ['rbf', 'cbf', 'sbf_sum']:
                    copy_(getattr(block.quad_interaction, f'scale_{layer}').scale_factor, f'int_blocks/{i}/quad_interaction/scale_{layer}/scale_factor')
            for layer in ['dense_ba', 'mlp_rbf', 'mlp_cbf', 'down_projection', 'up_projection_ac', 'up_projection_ca']:
                copy_(getattr(block.trip_interaction, layer).weight, f'int_blocks/{i}/trip_interaction/{layer}/kernel')
            for layer in ['rbf', 'cbf_sum']:
                copy_(getattr(block.trip_interaction, f'scale_{layer}').scale_factor, f'int_blocks/{i}/trip_interaction/scale_{layer}/scale_factor')
            copy_(block.atom_update.dense_rbf.weight, f'int_blocks/{i}/atom_update/dense_rbf/kernel')
            copy_(block.atom_update.scale_sum.scale_factor, f'int_blocks/{i}/atom_update/scale_sum/scale_factor')
            copy_(block.atom_update.layers[0].weight, f'int_blocks/{i}/atom_update/layers/0/kernel')
            for j, res_layer in enumerate(block.atom_update.layers[1:]):
                j = j + 1
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(layer.weight, f'int_blocks/{i}/atom_update/layers/{j}/dense_mlp/layer_with_weights-{k}/kernel')
            copy_(block.concat_layer.dense.weight, f'int_blocks/{i}/concat_layer/dense/kernel')
            copy_(block.dense_ca.weight, f'int_blocks/{i}/dense_ca/kernel')
            for j, res_layer in enumerate(block.layers_after_skip):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(layer.weight, f'int_blocks/{i}/layers_after_skip/{j}/dense_mlp/layer_with_weights-{k}/kernel')
            for j, res_layer in enumerate(block.layers_before_skip):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(layer.weight, f'int_blocks/{i}/layers_before_skip/{j}/dense_mlp/layer_with_weights-{k}/kernel')
            for j, res_layer in enumerate(block.residual_m):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(layer.weight, f'int_blocks/{i}/residual_m/{j}/dense_mlp/layer_with_weights-{k}/kernel')
        for i, block in enumerate(self.out_blocks):
            copy_(block.dense_rbf.weight, f'out_blocks/{i}/dense_rbf/kernel')
            copy_(block.layers[0].weight, f'out_blocks/{i}/layers/0/kernel')
            for j, res_layer in enumerate(block.layers[1:]):
                j = j + 1
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(layer.weight, f'out_blocks/{i}/layers/{j}/dense_mlp/layer_with_weights-{k}/kernel')
            copy_(block.out_energy.weight, f'out_blocks/{i}/out_energy/kernel')
            copy_(block.scale_sum.scale_factor, f'out_blocks/{i}/scale_sum/scale_factor')
            if self.direct_forces:
                copy_(block.out_forces.weight, f'out_blocks/{i}/out_forces/kernel')
                copy_(block.out_forces.bias, f'out_blocks/{i}/out_forces/bias')
                copy_(block.seq_forces[0].weight, f'out_blocks/{i}/seq_forces/0/kernel')
                copy_(block.scale_rbf.scale_factor, f'out_blocks/{i}/scale_rbf/scale_factor')
                for j, res_layer in enumerate(block.seq_forces[1:]):
                    j = j + 1
                    for k, layer in enumerate(res_layer.dense_mlp):
                        copy_(layer.weight, f'out_blocks/{i}/seq_forces/{j}/dense_mlp/layer_with_weights-{k}/kernel')

    def predict(self, inputs):
        E, F = self(inputs)
        E = E.detach().cpu()
        F = F.detach().cpu()
        return E, F

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


class EfficientInteractionHadamard(torch.nn.Module):
    """
    Efficient reformulation of the hadamard product and subsequent summation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        emb_size: int
            Embedding size.
    """

    def __init__(self, emb_size_interm: int, emb_size: int, name='EfficientHadamard'):
        super().__init__()
        self.emb_size_interm = emb_size_interm
        self.emb_size = emb_size
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.empty((self.emb_size, 1, self.emb_size_interm), requires_grad=True))
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, Kidx):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        rbf_W1, sph = basis
        nEdges = rbf_W1.shape[0]
        if sph.shape[2] == 0:
            Kmax = 0
        else:
            Kmax = torch.max(torch.max(Kidx + 1), torch.tensor(0))
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m
        sum_k = torch.matmul(sph, m2)
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)
        m_ca = torch.matmul(self.weight, rbf_W1_sum_k.permute(2, 1, 0))[:, 0]
        m_ca = torch.transpose(m_ca, 0, 1)
        return m_ca


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BesselBasisLayer,
     lambda: ([], {'num_radial': 4, 'cutoff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dense,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeEmbedding,
     lambda: ([], {'atom_features': 4, 'edge_features': 4, 'out_features': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.rand([4]), torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (Envelope,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualLayer,
     lambda: ([], {'units': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledSiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_TUM_DAML_gemnet_pytorch(_paritybench_base):
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

