import sys
_module = sys.modules[__name__]
del sys
multiclass_utils = _module
sparsemax = _module
tab_model = _module
tab_network = _module
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


from torch import nn


from torch.autograd import Function


import torch.nn.functional as F


import torch


import numpy as np


from scipy.sparse import csc_matrix


import time


from abc import abstractmethod


from sklearn.metrics import roc_auc_score


from sklearn.metrics import mean_squared_error


from sklearn.metrics import accuracy_score


from torch.nn.utils import clip_grad_norm_


from sklearn.base import BaseEstimator


from torch.utils.data import DataLoader


import copy


from torch.nn import Linear


from torch.nn import BatchNorm1d


from torch.nn import ReLU


from torch.utils.data import Dataset


from torch.utils.data import WeightedRandomSampler


import scipy


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Args:
            input: any dimension
            dim: dimension along which to apply the sparsemax
        Returns:
            the threshold value
        """
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2
        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)
        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


entmax15 = Entmax15Function.apply


class Entmax15(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class AttentiveTransformer(torch.nn.Module):

    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.sp_max = sparsemax.Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.sp_max(x)
        return x


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GLU_Layer(torch.nn.Module):

    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        super(GLU_Layer, self).__init__()
        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)
        self.bn = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out


class GLU_Block(torch.nn.Module):
    """
        Independant GLU block, specific to each step
    """

    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None, virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()
        params = {'virtual_batch_size': virtual_batch_size, 'momentum': momentum}
        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]))
        if self.first:
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)
        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class FeatTransformer(torch.nn.Module):

    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size=128, momentum=0.02):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - n_glu_independant
        - shared_blocks : torch.nn.ModuleList
            The shared block that should be common to every step
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """
        params = {'n_glu': n_glu_independent, 'virtual_batch_size': virtual_batch_size, 'momentum': momentum}
        if shared_layers is None:
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(input_dim, output_dim, first=True, shared_layers=shared_layers, n_glu=len(shared_layers), virtual_batch_size=virtual_batch_size, momentum=momentum)
            is_first = False
        if n_glu_independent == 0:
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(spec_input_dim, output_dim, first=is_first, **params)

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class TabNetNoEmbeddings(torch.nn.Module):

    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2, n_shared=2, epsilon=1e-15, virtual_batch_size=128, momentum=0.02):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        - input_dim : int
            Number of features
        - output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        - n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        - n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        - n_steps: int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        - gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        - n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        - n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        - epsilon: float
            Avoid log(0), this should be kept very low
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(self.input_dim, 2 * (n_d + n_a), bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d + n_a, 2 * (n_d + n_a), bias=False))
        else:
            shared_feat_transform = None
        self.initial_splitter = FeatTransformer(self.input_dim, n_d + n_a, shared_feat_transform, n_glu_independent=self.n_independent, virtual_batch_size=self.virtual_batch_size, momentum=momentum)
        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()
        for step in range(n_steps):
            transformer = FeatTransformer(self.input_dim, n_d + n_a, shared_feat_transform, n_glu_independent=self.n_independent, virtual_batch_size=self.virtual_batch_size, momentum=momentum)
            attention = AttentiveTransformer(n_a, self.input_dim, virtual_batch_size=self.virtual_batch_size, momentum=momentum)
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)
        self.final_mapping = Linear(n_d, output_dim, bias=False)
        initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        prior = torch.ones(x.shape)
        M_explain = torch.zeros(x.shape)
        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d:]
        masks = {}
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)) / self.n_steps
            prior = torch.mul(self.gamma - M, prior)
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            res = torch.add(res, d)
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            att = out[:, self.n_d:]
        res = self.final_mapping(res)
        return res, M_loss, M_explain, masks


class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """ This is an embedding module for an entier set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))
        self.embeddings = torch.nn.ModuleList()
        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            return x
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            if is_continuous:
                cols.append(x[:, (feat_init_idx)].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, (feat_init_idx)].long()))
                cat_feat_counter += 1
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


class TabNet(torch.nn.Module):

    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=2, n_shared=2, epsilon=1e-15, virtual_batch_size=128, momentum=0.02, device_name='auto'):
        """
        Defines TabNet network

        Parameters
        ----------
        - input_dim : int
            Initial number of features
        - output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        - n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        - n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        - n_steps: int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        - gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        - cat_idxs : list of int
            Index of each categorical column in the dataset
        - cat_dims : list of int
            Number of categories in each categorical column
        - cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        - n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        - n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        - epsilon: float
            Avoid log(0), this should be kept very low
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        if self.n_steps <= 0:
            raise ValueError('n_steps should be a positive integer.')
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independant can't be both zero.")
        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(self.post_embed_dim, output_dim, n_d, n_a, n_steps, gamma, n_independent, n_shared, epsilon, virtual_batch_size, momentum)
        self.initial_bn = BatchNorm1d(self.post_embed_dim, momentum=0.01)
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        self

    def forward(self, x):
        x = self.embedder(x)
        x = self.initial_bn(x)
        return self.tabnet(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Entmax15,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GBN,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GLU_Block,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GLU_Layer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Sparsemax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_dreamquark_ai_tabnet(_paritybench_base):
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

