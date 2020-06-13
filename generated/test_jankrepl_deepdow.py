import sys
_module = sys.modules[__name__]
del sys
deepdow = _module
benchmarks = _module
callbacks = _module
data = _module
augment = _module
load = _module
synthetic = _module
experiments = _module
explain = _module
layers = _module
allocate = _module
collapse = _module
misc = _module
transform = _module
losses = _module
nn = _module
utils = _module
visualize = _module
conf = _module
softmax_sparsemax = _module
var_model = _module
zoom = _module
setup = _module
tests = _module
conftest = _module
test_benchmarks = _module
test_callbacks = _module
test_data = _module
test_augment = _module
test_load = _module
test_synthetic = _module
test_experiments = _module
test_explain = _module
test_layers = _module
test_losses = _module
test_nn = _module
test_utils = _module
test_visualize = _module

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


import numpy as np


import torch


from torch.distributions import MultivariateNormal


import torch.nn as nn


from itertools import cycle


class AnalyticalMarkowitz(nn.Module):
    """Minimum variance and maximum sharpe ratio with no constraints.

    There exists known analytical solutions so numerical solutions are necessary.

    References
    ----------
    [1] http://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
    """

    def forward(self, covmat, rets=None):
        """Perform forward pass.

        Parameters
        ----------
        covmat : torch.Tensor
            Covariance matrix of shape `(n_samples, n_assets, n_assets)`.

        rets : torch.Tensor or None
            If tensor then of shape `(n_samples, n_assets)` representing expected returns. If provided triggers
            computation of maximum share ratio. Else None triggers computation of minimum variance portfolio.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights. If `rets` provided, then it represents
            maximum sharpe ratio portfolio (tangency portfolio). Otherwise minimum variance portfolio.
        """
        n_samples, n_assets, _ = covmat.shape
        device = covmat.device
        dtype = covmat.dtype
        covmat_inv = torch.inverse(covmat)
        ones = torch.ones(n_samples, n_assets, 1).to(device=device, dtype=dtype
            )
        if rets is not None:
            expected_returns = rets.view(n_samples, n_assets, 1)
        else:
            expected_returns = ones
        w_unscaled = torch.matmul(covmat_inv, expected_returns)
        denominator = torch.matmul(ones.permute(0, 2, 1), w_unscaled)
        w = w_unscaled / denominator
        return w.squeeze(-1)


class NCO(nn.Module):
    """Nested cluster optimization.

    This optimization algorithm performs the following steps:

         1. Divide all assets into clusters
         2. Run standard optimization inside of each of these clusters (intra step)
         3. Run standard optimization on the resulting portfolios (inter step)
         4. Compute the final weights

    Parameters
    ----------
    n_clusters : int
        Number of clusters to find in the data. Note that the underlying clustering model is
        KMeans - ``deepdow.layers.KMeans``.

    n_init : int
        Number of runs of the clustering algorithm.

    init : str, {'random', 'k-means++'}
        Initialization strategy of the clustering algorithm.

    random_state : int or None
        Random state passed to the stochastic k-means clustering.

    See Also
    --------
    deepdow.layers.KMeans : k-means clustering algorithm

    References
    ----------
    [1] M Lopez de Prado.
        "A Robust Estimator of the Efficient Frontier"
        Available at SSRN 3469961, 2019

    """

    def __init__(self, n_clusters, n_init=10, init='random', random_state=None
        ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.cov2corr_layer = Cov2Corr()
        self.kmeans_layer = KMeans(n_clusters=self.n_clusters, n_init=self.
            n_init, init=self.init, random_state=self.random_state)
        self.analytical_markowitz_layer = AnalyticalMarkowitz()

    def forward(self, covmat, rets=None):
        """Perform forward pass.

        Parameters
        ----------
        covmat : torch.Tensor
            Covariance matrix of shape `(n_samples, n_assets, n_assets)`.

        rets : torch.Tensor or None
            If tensor then of shape `(n_samples, n_assets)` representing expected returns. If provided triggers
            computation of maximum share ratio. Else None triggers computation of minimum variance portfolio.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights. If `rets` provided, then
            maximum sharpe ratio portfolio (tangency portfolio) used both on intra and inter cluster level. Otherwise
            minimum variance portfolio.

        Notes
        -----
        Currently there is not batching over the sample dimension - simple for loop is used.

        """
        n_samples, n_assets, _ = covmat.shape
        dtype, device = covmat.dtype, covmat.device
        corrmat = Cov2Corr()(covmat)
        w_l = []
        for i in range(n_samples):
            cluster_ixs, cluster_centers = self.kmeans_layer(corrmat[i])
            w_intra_clusters = torch.zeros((n_assets, self.n_clusters),
                dtype=dtype, device=device)
            for c in range(self.n_clusters):
                in_cluster = torch.where(cluster_ixs == c)[0]
                intra_covmat = covmat[[i]].index_select(1, in_cluster
                    ).index_select(2, in_cluster)
                intra_rets = None if rets is None else rets[[i]].index_select(
                    1, in_cluster)
                w_intra_clusters[in_cluster, c
                    ] = self.analytical_markowitz_layer(intra_covmat,
                    intra_rets)[0]
            inter_covmat = w_intra_clusters.T @ (covmat[i] @ w_intra_clusters)
            inter_rets = None if rets is None else (w_intra_clusters.T @
                rets[i]).view(1, -1)
            w_inter_clusters = self.analytical_markowitz_layer(inter_covmat
                .view(1, self.n_clusters, self.n_clusters), inter_rets)
            w_final = (w_intra_clusters * w_inter_clusters).sum(dim=1)
            w_l.append(w_final)
        res = torch.stack(w_l, dim=0)
        return res


class NumericalMarkowitz(nn.Module):
    """Convex optimization layer stylized into portfolio optimization problem.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Attributes
    ----------
    cvxpylayer : CvxpyLayer
        Custom layer used by a third party package called cvxpylayers.

    References
    ----------
    [1] https://github.com/cvxgrp/cvxpylayers

    """

    def __init__(self, n_assets, max_weight=1):
        """Construct."""
        super().__init__()
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        rets = cp.Parameter(n_assets)
        alpha = cp.Parameter(nonneg=True)
        w = cp.Variable(n_assets)
        ret = rets @ w
        risk = cp.sum_squares(covmat_sqrt @ w)
        reg = alpha * cp.norm(w) ** 2
        prob = cp.Problem(cp.Maximize(ret - risk - reg), [cp.sum(w) == 1, w >=
            0, w <= max_weight])
        assert prob.is_dpp()
        self.cvxpylayer = CvxpyLayer(prob, parameters=[rets, covmat_sqrt,
            alpha], variables=[w])

    def forward(self, rets, covmat_sqrt, gamma_sqrt, alpha):
        """Perform forward pass.

        Parameters
        ----------
        rets : torch.Tensor
            Of shape (n_samples, n_assets) representing expected returns (or whatever the feature extractor decided
            to encode).

        covmat_sqrt : torch.Tensor
            Of shape (n_samples, n_assets, n_assets) representing the square of the covariance matrix.

        gamma_sqrt : torch.Tensor
            Of shape (n_samples,) representing the tradeoff between risk and return - where on efficient frontier
            we are.

        alpha : torch.Tensor
            Of shape (n_samples,) representing how much L2 regularization is applied to weights. Note that
            we pass the absolute value of this variable into the optimizer since when creating the problem
            we asserted it is going to be nonnegative.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights as determined by the convex optimizer.

        """
        n_samples, n_assets = rets.shape
        gamma_sqrt_ = gamma_sqrt.repeat((1, n_assets * n_assets)).view(
            n_samples, n_assets, n_assets)
        alpha_abs = torch.abs(alpha)
        return self.cvxpylayer(rets, gamma_sqrt_ * covmat_sqrt, alpha_abs)[0]


class Resample(nn.Module):
    """Meta allocator that bootstraps the input expected returns and covariance matrix.

    The idea is to take the input covmat and expected returns and view them as parameters of a Multivariate
    Normal distribution. After that, we iterate the below steps `n_portfolios` times:

        1. Sample `n_draws` from the distribution
        2. Estimate expected_returns and covariance matrix
        3. Use the `allocator` to compute weights.

    This will results in `n_portfolios` portfolios that we simply average to get the final weights.

    Parameters
    ----------
    allocator : AnalyticalMarkowitz or NCO or NumericalMarkowitz
        Instance of an allocator.

    n_draws : int or None
        Number of draws. If None then set equal to number of assets to prevent numerical problems.

    n_portfolios : int
        Number of samples.

    sqrt : bool
        If True, then the input array represent the square root of the covariance matrix. Else it is the actual
        covariance matrix.

    random_state : int or None
        Random state (forward passes with same parameters will have same results).

    References
    ----------
    [1] Michaud, Richard O., and Robert Michaud.
        "Estimation error and portfolio optimization: a resampling solution."
        Available at SSRN 2658657 (2007)
    """

    def __init__(self, allocator, n_draws=None, n_portfolios=5, sqrt=False,
        random_state=None):
        super().__init__()
        if not isinstance(allocator, (AnalyticalMarkowitz, NCO,
            NumericalMarkowitz)):
            raise TypeError('Unsupported type of allocator: {}'.format(type
                (allocator)))
        self.allocator = allocator
        self.sqrt = sqrt
        self.n_draws = n_draws
        self.n_portfolios = n_portfolios
        self.random_state = random_state
        mapper = {'AnalyticalMarkowitz': False, 'NCO': True,
            'NumericalMarkowitz': True}
        self.uses_sqrt = mapper[allocator.__class__.__name__]

    def forward(self, matrix, rets=None, **kwargs):
        """Perform forward pass.

        Only accepts keyword arguments to avoid ambiguity.

        Parameters
        ----------
        matrix : torch.Tensor
            Of shape (n_samples, n_assets, n_assets) representing the square of the covariance matrix if
            `self.square=True` else the covariance matrix itself.

        rets : torch.Tensor or None
            Of shape (n_samples, n_assets) representing expected returns (or whatever the feature extractor decided
            to encode). Note that `NCO` and `AnalyticalMarkowitz` allow for `rets=None` (using only minimum variance).

        kwargs : dict
            All additional input arguments the `self.allocator` needs to perform forward pass.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights.

        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        n_samples, n_assets, _ = matrix.shape
        dtype, device = matrix.dtype, matrix.device
        n_draws = self.n_draws or n_assets
        covmat = matrix @ matrix if self.sqrt else matrix
        dist_rets = torch.zeros(n_samples, n_assets, dtype=dtype, device=device
            ) if rets is None else rets
        dist = MultivariateNormal(loc=dist_rets, covariance_matrix=covmat)
        portfolios = []
        for _ in range(self.n_portfolios):
            draws = dist.sample((n_draws,))
            rets_ = draws.mean(dim=0) if rets is not None else None
            covmat_ = CovarianceMatrix(sqrt=self.uses_sqrt)(draws.permute(1,
                0, 2))
            if isinstance(self.allocator, (AnalyticalMarkowitz, NCO)):
                portfolio = self.allocator(covmat=covmat_, rets=rets_)
            elif isinstance(self.allocator, NumericalMarkowitz):
                gamma = kwargs['gamma']
                alpha = kwargs['alpha']
                portfolio = self.allocator(rets_, covmat_, gamma, alpha)
            portfolios.append(portfolio)
        portfolios_t = torch.stack(portfolios, dim=0)
        return portfolios_t.mean(dim=0)


class SoftmaxAllocator(torch.nn.Module):
    """Portfolio creation by computing a softmax over the asset dimension with temperature.

    Parameters
    ----------
    temperature : None or float
        If None, then needs to be provided per sample during forward pass. If ``float`` then assumed
        to be always the same.

    formulation : str, {'analytical', 'variational'}
        Controls what way the problem is solved. If 'analytical' then using an explicit formula,
        however, one cannot decide on a `max_weight` different than 1. If `variational` then solved
        via convex optimization and one can set any `max_weight`.

    n_assets : None or int
        Only required and used if `formulation='variational`.

    max_weight : float
        A float between (0, 1] representing the maximum weight per asset.

    """

    def __init__(self, temperature=1, formulation='analytical', n_assets=
        None, max_weight=1):
        super().__init__()
        self.temperature = temperature
        if formulation not in {'analytical', 'variational'}:
            raise ValueError('Unrecognized formulation {}'.format(formulation))
        if formulation == 'variational' and n_assets is None:
            raise ValueError(
                'One needs to provide n_assets for the variational formulation.'
                )
        if formulation == 'analytical' and max_weight != 1:
            raise ValueError(
                'Cannot constraint weights via max_weight for analytical formulation'
                )
        if formulation == 'variational' and n_assets * max_weight < 1:
            raise ValueError(
                'One cannot create fully invested portfolio with the given max_weight'
                )
        self.formulation = formulation
        if formulation == 'analytical':
            self.layer = torch.nn.Softmax(dim=1)
        else:
            x = cp.Parameter(n_assets)
            w = cp.Variable(n_assets)
            obj = -x @ w - cp.sum(cp.entr(w))
            cons = [cp.sum(w) == 1.0, w <= max_weight]
            prob = cp.Problem(cp.Minimize(obj), cons)
            self.layer = CvxpyLayer(prob, [x], [w])

    def forward(self, x, temperature=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        temperature : None or torch.Tensor
            If None, then using the `temperature` provided at construction time. Otherwise a `torch.Tensor` of shape
            `(n_samples,)` representing a per sample temperature.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        n_samples, _ = x.shape
        device, dtype = x.device, x.dtype
        if not (temperature is None) ^ (self.temperature is None):
            raise ValueError('Not clear which temperature to use')
        if temperature is not None:
            temperature_ = temperature
        else:
            temperature_ = float(self.temperature) * torch.ones(n_samples,
                dtype=dtype, device=device)
        inp = x / temperature_[..., None]
        return self.layer(inp
            ) if self.formulation == 'analytical' else self.layer(inp)[0]


class SparsemaxAllocator(torch.nn.Module):
    """Portfolio creation by computing a sparsemax over the asset dimension with temperature.

    Parameters
    ----------
    n_assets : int
        Number of assets. Note that we require this quantity at construction to make sure
        the underlying cvxpylayer does not need to be reinitialized every forward pass.

    temperature : None or float
        If None, then needs to be provided per sample during forward pass. If ``float`` then
        assumed to be always the same.

    max_weight : float
        A float between (0, 1] representing the maximum weight per asset.

    References
    ----------
    [1] Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention
    and multi-label classification." International Conference on Machine Learning. 2016.

    [2] Malaviya, Chaitanya, Pedro Ferreira, and André FT Martins. "Sparse and constrained attention
    for neural machine translation." arXiv preprint arXiv:1805.08241 (2018)
    """

    def __init__(self, n_assets, temperature=1, max_weight=1):
        super().__init__()
        if n_assets * max_weight < 1:
            raise ValueError(
                'One cannot create fully invested portfolio with the given max_weight'
                )
        self.n_assets = n_assets
        self.temperature = temperature
        x = cp.Parameter(n_assets)
        w = cp.Variable(n_assets)
        obj = cp.sum_squares(x - w)
        cons = [cp.sum(w) == 1, 0.0 <= w, w <= max_weight]
        prob = cp.Problem(cp.Minimize(obj), cons)
        self.layer = CvxpyLayer(prob, parameters=[x], variables=[w])

    def forward(self, x, temperature=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        temperature : None or torch.Tensor
            If None, then using the `temperature` provided at construction time. Otherwise a
            `torch.Tensor` of shape `(n_samples,)` representing a per sample temperature.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        n_samples, _ = x.shape
        device, dtype = x.device, x.dtype
        if not (temperature is None) ^ (self.temperature is None):
            raise ValueError('Not clear which temperature to use')
        if temperature is not None:
            temperature_ = temperature
        else:
            temperature_ = float(self.temperature) * torch.ones(n_samples,
                dtype=dtype, device=device)
        inp = x / temperature_[..., None]
        return self.layer(inp)[0]


class WeightNorm(torch.nn.Module):
    """Allocation via weight normalization.

    We learn a single weight for each asset and make sure that they sum up to one.
    """

    def __init__(self, n_assets):
        super().__init__()
        self.asset_weights = torch.nn.Parameter(torch.ones(n_assets),
            requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, dim_1, ...., dim_N)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        n_samples = x.shape[0]
        clamped = torch.clamp(self.asset_weights, min=0)
        normalized = clamped / clamped.sum()
        return torch.stack(n_samples * [normalized], dim=0)


class AttentionCollapse(nn.Module):
    """Collapsing over the channels with attention.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    Attributes
    ----------
    affine : nn.Module
        Fully connected layer performing linear mapping.

    context_vector : nn.Module
        Fully connected layer encoding direction importance.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.affine = nn.Linear(n_channels, n_channels)
        self.context_vector = nn.Linear(n_channels, 1, bias=False)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_channels, n_assets)`.

        """
        n_samples, n_channels, lookback, n_assets = x.shape
        res_list = []
        for i in range(n_samples):
            inp_single = x[i].permute(2, 1, 0)
            tformed = self.affine(inp_single)
            w = self.context_vector(tformed)
            scaled_w = torch.nn.functional.softmax(w, dim=1)
            weighted_sum = (inp_single * scaled_w).mean(dim=1)
            res_list.append(weighted_sum.permute(1, 0))
        return torch.stack(res_list, dim=0)


class AverageCollapse(nn.Module):
    """Global average collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Average over the removeed dimension.
        """
        return x.mean(dim=self.collapse_dim)


class ElementCollapse(nn.Module):
    """Single element over a specified dimension."""

    def __init__(self, collapse_dim=2, element_ix=-1):
        super().__init__()
        self.collapse_dim = collapse_dim
        self.element_ix = element_ix

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Taking the `self.element_ix` element of the removed dimension.
        """
        return x.unbind(self.collapse_dim)[self.element_ix]


class ExponentialCollapse(nn.Module):
    """Exponential weighted collapsing over a specified dimension.

    The unscaled weights are defined recursively with the following rules:
        - w_{0}=1
        - w_{t+1} = forgetting_factor * w_{t} + 1

    Parameters
    ----------
    collapse_dim : int
        What dimension to remove.

    forgetting_factor : float or None
        If float, then fixed constant. If None this will become learnable.

    """

    def __init__(self, collapse_dim=2, forgetting_factor=None):
        super().__init__()
        self.collapse_dim = collapse_dim
        self.forgetting_factor = forgetting_factor or torch.nn.Parameter(torch
            .Tensor([0.5]), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Exponential Average over the removed dimension.
        """
        n_steps = x.shape[self.collapse_dim]
        n_dim = x.ndim
        view = [(-1 if i == self.collapse_dim else 1) for i in range(n_dim)]
        w_unscaled = [1]
        for _ in range(1, n_steps):
            w_unscaled.append(self.forgetting_factor * w_unscaled[-1] + 1)
        w_unscaled = torch.Tensor(w_unscaled).to(dtype=x.dtype, device=x.device
            )
        w = w_unscaled / w_unscaled.sum()
        return (x * w.view(*view)).sum(dim=self.collapse_dim)


class MaxCollapse(nn.Module):
    """Global max collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Maximum over the removed dimension.
        """
        return x.max(dim=self.collapse_dim)[0]


class SumCollapse(nn.Module):
    """Global sum collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Sum over the removed dimension.
        """
        return x.sum(dim=self.collapse_dim)


class Cov2Corr(nn.Module):
    """Conversion from covariance matrix to correlation matrix."""

    def forward(self, covmat):
        """Convert.

        Parameters
        ----------
        covmat : torch.Tensor
            Covariance matrix of shape (n_samples, n_assets, n_assets).

        Returns
        -------
        corrmat : torch.Tensor
            Correlation matrix of shape (n_samples, n_assets, n_assets).

        """
        n_samples, n_assets, _ = covmat.shape
        stds = torch.sqrt(torch.diagonal(covmat, dim1=1, dim2=2))
        stds_ = stds.view(n_samples, n_assets, 1)
        corr = covmat / torch.matmul(stds_, stds_.permute(0, 2, 1))
        return corr


class CovarianceMatrix(nn.Module):
    """Covariance matrix or its square root.

    Parameters
    ----------
    sqrt : bool
        If True, then returning the square root.

    shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
        Strategy of combining the sample covariance matrix with some more stable matrix.

    shrinkage_coef : float or None
        If ``float`` then in the range [0, 1] representing the weight of the convex combination. If `shrinkage_coef=1`
        then using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.
        If None then needs to be provided dynamically when performing forward pass.
    """

    def __init__(self, sqrt=True, shrinkage_strategy='diagonal',
        shrinkage_coef=0.5):
        """Construct."""
        super().__init__()
        self.sqrt = sqrt
        if shrinkage_strategy is not None:
            if shrinkage_strategy not in {'diagonal', 'identity',
                'scaled_identity'}:
                raise ValueError('Unrecognized shrinkage strategy {}'.
                    format(shrinkage_strategy))
        self.shrinkage_strategy = shrinkage_strategy
        self.shrinkage_coef = shrinkage_coef

    def forward(self, x, shrinkage_coef=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, n_assets).

        shrinkage_coef : None or torch.Tensor
            If None then using the `self.shrinkage_coef` supplied at construction for each sample. Otherwise a
            tensor of shape `(n_shapes,)`.

        Returns
        -------
        covmat : torch.Tensor
            Of shape (n_samples, n_assets, n_assets).

        """
        n_samples = x.shape[0]
        dtype, device = x.dtype, x.device
        if not (shrinkage_coef is None) ^ (self.shrinkage_coef is None):
            raise ValueError('Not clear which shrinkage coefficient to use')
        if shrinkage_coef is not None:
            shrinkage_coef_ = shrinkage_coef
        else:
            shrinkage_coef_ = self.shrinkage_coef * torch.ones(n_samples,
                dtype=dtype, device=device)
        wrapper = self.compute_sqrt if self.sqrt else lambda h: h
        return torch.stack([wrapper(self.compute_covariance(x[i].T.clone(),
            shrinkage_strategy=self.shrinkage_strategy, shrinkage_coef=
            shrinkage_coef_[i])) for i in range(n_samples)], dim=0)

    @staticmethod
    def compute_covariance(m, shrinkage_strategy=None, shrinkage_coef=0.5):
        """Compute covariance matrix for a single sample.

        Parameters
        ----------
        m : torch.Tensor
            Of shape (n_assets, n_channels).

        shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
            Strategy of combining the sample covariance matrix with some more stable matrix.

        shrinkage_coef : torch.Tensor
            A ``torch.Tensor`` scalar (probably in the range [0, 1]) representing the weight of the
            convex combination.

        Returns
        -------
        covmat_single : torch.Tensor
            Covariance matrix of shape (n_assets, n_assets).

        """
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        s = fact * m.matmul(mt)
        if shrinkage_strategy is None:
            return s
        elif shrinkage_strategy == 'identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)
            return shrinkage_coef * s + (1 - shrinkage_coef) * identity
        elif shrinkage_strategy == 'scaled_identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)
            scaled_identity = identity * torch.diag(s).mean()
            return shrinkage_coef * s + (1 - shrinkage_coef) * scaled_identity
        elif shrinkage_strategy == 'diagonal':
            diagonal = torch.diag(torch.diag(s))
            return shrinkage_coef * s + (1 - shrinkage_coef) * diagonal

    @staticmethod
    def compute_sqrt(m):
        """Compute the square root of a single positive definite matrix.

        Parameters
        ----------
        m : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the covariance matrix - needs to be PSD.

        Returns
        -------
        m_sqrt : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the square root of the covariance matrix.

        """
        _, s, v = m.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype
            ).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[(...), :common]
            v = v[(...), :common]
            if unbalanced:
                good = good[(...), :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return v * s.sqrt().unsqueeze(-2) @ v.transpose(-2, -1)


class KMeans(torch.nn.Module):
    """K-means algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to look for.

    init : str, {'random, 'k-means++', 'manual'}
        How to initialize the clusters at the beginning of the algorithm.

    n_init : int
        Number of times the algorithm is run. The best clustering is determined based on the
        potential (sum of distances of all points to the centroids).

    max_iter : int
        Maximum number of iterations of the algorithm. Note that if `norm(new_potential - old_potential) < tol`
        then stop prematurely.

    tol : float
        If `abs(new_potential - old_potential) < tol` then algorithm stopped irrespective of the `max_iter`.

    random_state : int or None
        Setting randomness.

    verbose : bool
        Control level of verbosity.
    """

    def __init__(self, n_clusters=5, init='random', n_init=1, max_iter=30,
        tol=1e-05, random_state=None, verbose=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        if self.init not in {'manual', 'random', 'k-means++'}:
            raise ValueError('Unrecognized initialization {}'.format(self.init)
                )

    def initialize(self, x, manual_init=None):
        """Initialize the k-means algorithm.

        Parameters
        ----------
        x : torch.Tensor
            Feature matrix of shape `(n_samples, n_features)`.

        manual_init : None or torch.Tensor
            If not None then expecting a tensor of shape `(n_clusters, n_features)`. Note that for this feature
            to be used one needs to set `init='manual'` in the constructor.

        Returns
        -------
        cluster_centers : torch.Tensor
            Tensor of shape `(n_clusters, n_features)` representing the initial cluster centers.

        """
        n_samples, n_features = x.shape
        device, dtype = x.device, x.dtype
        if self.init == 'random':
            p = torch.ones(n_samples, dtype=dtype, device=device)
            centroid_samples = torch.multinomial(p, num_samples=self.
                n_clusters, replacement=False)
            cluster_centers = x[centroid_samples]
        elif self.init == 'k-means++':
            p = torch.ones(n_samples, dtype=dtype, device=device)
            cluster_centers_l = []
            centroid_samples_l = []
            while len(cluster_centers_l) < self.n_clusters:
                centroid_sample = torch.multinomial(p, num_samples=1,
                    replacement=False)
                if centroid_sample in centroid_samples_l:
                    continue
                centroid_samples_l.append(centroid_sample)
                cluster_center = x[[centroid_sample]]
                cluster_centers_l.append(cluster_center)
                p = self.compute_distances(x, cluster_center).view(-1)
            cluster_centers = torch.cat(cluster_centers_l, dim=0)
        elif self.init == 'manual':
            if not torch.is_tensor(manual_init):
                raise TypeError('The manual_init needs to be a torch.Tensor')
            if manual_init.shape[0] != self.n_clusters:
                raise ValueError(
                    'The number of manually provided cluster centers is different from n_clusters'
                    )
            if manual_init.shape[1] != x.shape[1]:
                raise ValueError(
                    'The feature size of manually provided cluster centers is different from the input'
                    )
            cluster_centers = manual_init.to(dtype=dtype, device=device)
        return cluster_centers

    def forward(self, x, manual_init=None):
        """Perform clustering.

        Parameters
        ----------
        x : torch.Tensor
            Feature matrix of shape `(n_samples, n_features)`.

        manual_init : None or torch.Tensor
            If not None then expecting a tensor of shape `(n_clusters, n_features)`. Note that for this feature
            to be used one needs to set `init='manual'` in the constructor.

        Returns
        -------
        cluster_ixs : torch.Tensor
            1D array of lenght `n_samples` representing to what cluster each sample belongs.

        cluster_centers : torch.tensor
            Tensor of shape `(n_clusters, n_features)` representing the cluster centers.

        """
        n_samples, n_features = x.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                'The number of samples is lower than the number of clusters.')
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        lowest_potential = float('inf')
        lowest_potential_cluster_ixs = None
        lowest_potential_cluster_centers = None
        for run in range(self.n_init):
            cluster_centers = self.initialize(x, manual_init=manual_init)
            previous_potential = float('inf')
            for it in range(self.max_iter):
                distances = self.compute_distances(x, cluster_centers)
                cluster_ixs = torch.argmin(distances, dim=1)
                cluster_centers = torch.stack([x[cluster_ixs == i].mean(dim
                    =0) for i in range(self.n_clusters)], dim=0)
                current_potential = distances.gather(1, cluster_ixs.view(-1, 1)
                    ).sum()
                if abs(current_potential - previous_potential
                    ) < self.tol or it == self.max_iter - 1:
                    if self.verbose:
                        None
                    break
                previous_potential = current_potential
            if current_potential < lowest_potential:
                lowest_potential = current_potential
                lowest_potential_cluster_ixs = cluster_ixs.clone()
                lowest_potential_cluster_centers = cluster_centers.clone()
        if self.verbose:
            None
        return lowest_potential_cluster_ixs, lowest_potential_cluster_centers

    @staticmethod
    def compute_distances(x, cluster_centers):
        """Compute squared distances of samples to cluster centers.

        Parameters
        ----------
        x : torch.tensor
            Tensor of shape `(n_samples, n_features)`.

        cluster_centers : torch.tensor
            Tensor of shape `(n_clusters, n_features)`.

        Returns
        -------
        distances : torch.tensor
            Tensor of shape `(n_samples, n_clusters)` that provides for each sample (row) the squared distance
            to a given cluster center (column).

        """
        x_n = (x ** 2).sum(dim=1).view(-1, 1)
        c_n = (cluster_centers ** 2).sum(dim=1).view(1, -1)
        distances = x_n + c_n - 2 * torch.mm(x, cluster_centers.permute(1, 0))
        return torch.clamp(distances, min=0)


class MultiplyByConstant(torch.nn.Module):
    """Multiplying constant.

    Parameters
    ----------
    dim_size : int
        Number of input channels. We learn one constant per channel. Therefore `dim_size=n_trainable_parameters`.

    dim_ix : int
        Which dimension to apply the multiplication to.
    """

    def __init__(self, dim_size=1, dim_ix=1):
        super().__init__()
        self.dim_size = dim_size
        self.dim_ix = dim_ix
        self.constant = torch.nn.Parameter(torch.ones(self.dim_size),
            requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1})

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (d_0, d_1, ..., d_{N-1}).

        """
        if self.dim_size != x.shape[self.dim_ix]:
            raise ValueError(
                'The size of dimension {} is {} which is different than {}'
                .format(self.dim_ix, x.shape[self.dim_ix], self.dim_size))
        view = [(self.dim_size if i == self.dim_ix else 1) for i in range(x
            .ndim)]
        return x * self.constant.view(view)


class Conv(nn.Module):
    """Convolutional layer.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.

    n_output_channels : int
        Number of output channels.

    kernel_size : int
        Size of the kernel.

    method : str, {'2D, '1D'}
        What type of convolution is used in the background.
    """

    def __init__(self, n_input_channels, n_output_channels, kernel_size=3,
        method='2D'):
        super().__init__()
        self.method = method
        if method == '2D':
            self.conv = nn.Conv2d(n_input_channels, n_output_channels,
                kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        elif method == '1D':
            self.conv = nn.Conv1d(n_input_channels, n_output_channels,
                kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        else:
            raise ValueError("Invalid method {}, only supports '1D' or '2D'."
                .format(method))

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets) if `self.method='2D'`. Otherwise
            `(n_samples, n_input_channels, lookback)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_output_channels, lookback, n_assets)` if `self.method='2D'`. Otherwise
            `(n_samples, n_output_channels, lookback)`.

        """
        return self.conv(x)


class RNN(nn.Module):
    """Recurrent neural network layer.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

    cell_type : str, {'LSTM', 'RNN'}
        Type of the recurrent cell.

    bidirectional : bool
        If True, then bidirectional. Note that `hidden_size` already takes this parameter into account.

    n_layers : int
        Number of stacked layers.

    """

    def __init__(self, n_channels, hidden_size, cell_type='LSTM',
        bidirectional=True, n_layers=1):
        """Construct."""
        super().__init__()
        if hidden_size % 2 != 0 and bidirectional:
            raise ValueError(
                'Hidden size needs to be divisible by two for bidirectional RNNs.'
                )
        hidden_size_one_direction = int(hidden_size // (1 + int(bidirectional))
            )
        if cell_type == 'RNN':
            self.cell = torch.nn.RNN(n_channels, hidden_size_one_direction,
                bidirectional=bidirectional, num_layers=n_layers)
        elif cell_type == 'LSTM':
            self.cell = torch.nn.LSTM(n_channels, hidden_size_one_direction,
                bidirectional=bidirectional, num_layers=n_layers)
        else:
            raise ValueError('Unsupported cell_type {}'.format(cell_type))

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, self.hidden_size, lookback, n_assets)`.

        """
        n_samples, n_channels, lookback, n_assets = x.shape
        x_swapped = x.permute(0, 2, 3, 1)
        res = []
        for i in range(n_samples):
            all_hidden_ = self.cell(x_swapped[i])[0]
            res.append(all_hidden_.permute(2, 0, 1))
        return torch.stack(res)


class Zoom(torch.nn.Module):
    """Zoom in and out.

    It can dynamically zoom into more recent timesteps and disregard older ones. Conversely,
    it can collapse more timesteps into one. Based on Spatial Transformer Network.

    Parameters
    ----------
    mode : str, {'bilinear', 'nearest'}
        What interpolation to perform.

    padding_mode : str, {'zeros', 'border', 'reflection'}
        How to fill in values that fall outisde of the grid. Relevant in the case when we
        zoom out.

    References
    ----------
    [1] Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks."
        Advances in neural information processing systems. 2015.

    """

    def __init__(self, mode='bilinear', padding_mode='reflection'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, x, scale):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        scale : torch.Tensor
            Tensor of shape `(n_samples,)` representing how much to zoom in (`scale < 1`) or
            zoom out (`scale > 1`).

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)` that is a zoomed
            version of the input. Note that the shape is identical to the input.

        """
        translate = 1 - scale
        theta = torch.stack([torch.tensor([[1, 0, 0], [0, s, t]]) for s, t in
            zip(scale, translate)], dim=0)
        theta = theta.to(device=x.device, dtype=x.dtype)
        grid = nn.functional.affine_grid(theta, x.shape)
        x_zoomed = nn.functional.grid_sample(x, grid, mode=self.mode,
            padding_mode=self.padding_mode, align_corners=False)
        return x_zoomed


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jankrepl_deepdow(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AnalyticalMarkowitz(*[], **{}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(AttentionCollapse(*[], **{'n_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(AverageCollapse(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Conv(*[], **{'n_input_channels': 4, 'n_output_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Cov2Corr(*[], **{}), [torch.rand([4, 4, 4])], {})

    def test_005(self):
        self._check(ElementCollapse(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ExponentialCollapse(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(MaxCollapse(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(NCO(*[], **{'n_clusters': 4}), [torch.rand([4, 4, 4])], {})

    def test_009(self):
        self._check(RNN(*[], **{'n_channels': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(SoftmaxAllocator(*[], **{}), [torch.rand([4, 4])], {})

    def test_011(self):
        self._check(SumCollapse(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(WeightNorm(*[], **{'n_assets': 4}), [torch.rand([4, 4, 4, 4])], {})

