import sys
_module = sys.modules[__name__]
del sys
alg_comparison = _module
alg_comparison_2d = _module
simplicial_complex = _module
time_levelset = _module
examples = _module
noisy_circle = _module
alpha = _module
alpha = _module
levelset = _module
penalties = _module
problems = _module
rips = _module
util = _module
holes = _module
setup = _module
cpp = _module
simplicial_complex = _module
nn = _module
alpha = _module
levelset = _module
rips = _module
topologylayer = _module
functional = _module
alpha_dionysus = _module
flag = _module
levelset_dionysus = _module
rips_dionysus = _module
sublevel = _module
utils_dionysus = _module
alpha = _module
alpha_dionysus = _module
features = _module
levelset = _module
levelset_dionysus = _module
rips = _module
rips_dionysus = _module
construction = _module
flag_dionysus = _module
plot_dionysus = _module
process = _module
star_dionysus = _module

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


import matplotlib.pyplot as plt


import torch


import time


import numpy as np


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import sampler


import torchvision.datasets as dset


import torchvision.transforms as T


from torchvision.utils import save_image


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension


import itertools


from scipy.spatial import Delaunay


from torch.autograd import Variable


from torch.autograd import Function


import matplotlib


ape = 1, 28, 28


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(100, 128), nn.LeakyReLU(0.2, inplace=True), nn.Linear(128, 256), nn.BatchNorm1d(256, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 512), nn.BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, int(np.prod(ape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *ape)
        return img


def remove_filler(dgm, val=np.inf):
    """
    remove filler rows from diagram
    """
    inds = dgm[:, (0)] != val
    return dgm[(inds), :]


class RipsLayer(nn.Module):
    """
    Rips persistence layer
    Parameters:
        maxdim : maximum homology dimension (default=1)
        rmax   : maximum value of filtration (default=inf)
        verbose : print information
    """

    def __init__(self, maxdim=1, rmax=np.inf, verbose=False):
        super(RipsLayer, self).__init__()
        self.rmax = rmax
        self.maxdim = maxdim
        self.verbose = verbose
        self.fnobj = ripsdgm()

    def forward(self, x):
        dgm = self.fnobj.apply(x, self.rmax, self.maxdim, self.verbose)
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.maxdim + 1))
        return dgms, True


def get_start_end(dgm, issublevel):
    """
    get start and endpoints of barcode pairs
    input:
        dgminfo - Tuple consisting of diagram tensor and bool
            bool = true if diagram is of sub-level set type
            bool = false if diagram is of super-level set type
    output - start, end tensors of diagram
    """
    if issublevel:
        start, end = dgm[:, (0)], dgm[:, (1)]
    else:
        end, start = dgm[:, (0)], dgm[:, (1)]
    return start, end


def get_raw_barcode_lengths(dgm, issublevel):
    """
    get barcode lengths from barcode pairs
    no filtering
    """
    start, end = get_start_end(dgm, issublevel)
    lengths = end - start
    return lengths


def get_barcode_lengths(dgm, issublevel):
    """
    get barcode lengths from barcode pairs
    filter out infinite bars
    """
    lengths = get_raw_barcode_lengths(dgm, issublevel)
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths


class SumBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram
    ignores infinite bars, and padding
    Options:
        dim - bardocde dimension to sum over (defualt 0)

    forward input:
        (dgms, issub) tuple, passed from diagram layer
    """

    def __init__(self, dim=0):
        super(SumBarcodeLengths, self).__init__()
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)
        return torch.sum(lengths, dim=0)


class TopLoss(nn.Module):

    def __init__(self):
        super(TopLoss, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


class PartialSumBarcodeLengths(nn.Module):
    """
    Layer that computes a partial sum of barckode lengths

    inputs:
        dim - homology dimension
        skip - skip this number of the longest bars

    ignores infinite bars and padding
    """

    def __init__(self, dim, skip):
        super(PartialSumBarcodeLengths, self).__init__()
        self.skip = skip
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)
        sortl, indl = torch.sort(lengths, descending=True)
        return torch.sum(sortl[self.skip:])


class TopLoss2(nn.Module):

    def __init__(self):
        super(TopLoss2, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


class SobLoss(torch.nn.Module):
    """
    Sobolev norm penalty on function
    (sum |x_{i} - x{i+1}|^p)^{1/p}

    parameters:
        p - dimension of norm
    """

    def __init__(self, p):
        super(SobLoss, self).__init__()
        self.p = p

    def forward(self, beta):
        hdiff = beta[1:] - beta[:-1]
        return torch.norm(hdiff, p=self.p)


class NormLoss(torch.nn.Module):
    """
    Norm penalty on function

    parameters:
        p - dimension of norm
    """

    def __init__(self, p):
        super(NormLoss, self).__init__()
        self.p = p

    def forward(self, beta):
        return torch.norm(beta, p=self.p)


class AlphaLayer(torch.nn.Module):
    """
    Alpha persistence layer for spatial inputs
    Should be equivalent for Rips, but much faster
    Parameters:
        maxdim : maximum homology dimension (defualt=0)
        verbose : print information (default=False)
    """

    def __init__(self, maxdim=0, verbose=False):
        super(AlphaLayer, self).__init__()
        self.verbose = verbose
        self.maxdim = maxdim
        self.fnobj = alphadgm()

    def forward(self, x):
        dgm = self.fnobj.apply(x, self.maxdim, self.verbose)
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.maxdim + 1))
        return dgms, True


def get_barcode_lengths_means(dgm, issublevel):
    """
    return lengths and means of barcode

    set irrelevant or infinite to zero
    """
    start, end = get_start_end(dgm, issublevel)
    lengths = end - start
    means = (end + start) / 2
    means[lengths == np.inf] = 0
    means[lengths != lengths] = 0
    lengths[lengths == np.inf] = 0
    lengths[lengths != lengths] = 0
    return lengths, means


def remove_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:, (0)] != dgm[:, (1)]
    return dgm[(inds), :]


class BarcodePolyFeature(nn.Module):
    """
    applies function
    sum length^p * mean^q
    over lengths and means of barcode
    parameters:
        dim - homology dimension to work over
        p - exponent for lengths
        q - exponent for means
        remove_zero = Flag to remove zero-length bars (default=True)
    """

    def __init__(self, dim, p, q, remove_zero=True):
        super(BarcodePolyFeature, self).__init__()
        self.dim = dim
        self.p = p
        self.q = q
        self.remove_zero = remove_zero

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        dgm = dgms[self.dim]
        if self.remove_zero:
            dgm = remove_zero_bars(dgm)
        lengths, means = get_barcode_lengths_means(dgm, issublevel)
        return torch.sum(torch.mul(torch.pow(lengths, self.p), torch.pow(means, self.q)))


def pad_k(t, k, pad=0.0):
    """
    zero pad tensor t until dimension along axis is k

    if t has dimension greater than k, truncate
    """
    lt = len(t)
    if lt > k:
        return t[:k]
    if lt < k:
        fillt = torch.tensor(pad * np.ones(k - lt), dtype=t.dtype)
        return torch.cat((t, fillt))
    return t


class TopKBarcodeLengths(nn.Module):
    """
    Layer that returns top k lengths of persistence diagram in dimension

    inputs:
        dim - homology dimension
        k - number of lengths

    ignores infinite bars and padding
    """

    def __init__(self, dim, k):
        super(TopKBarcodeLengths, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)
        sortl, indl = torch.sort(lengths, descending=True)
        return pad_k(sortl, self.k, 0.0)


def init_freudenthal_2d(width, height):
    """
    Freudenthal triangulation of 2d grid
    """
    s = d.Filtration()
    for i in range(height):
        for j in range(width):
            ind = i * width + j
            s.append(d.Simplex([ind]))
    for i in range(height):
        for j in range(width - 1):
            ind = i * width + j
            s.append(d.Simplex([ind, ind + 1]))
    for i in range(height - 1):
        for j in range(width):
            ind = i * width + j
            s.append(d.Simplex([ind, ind + width]))
    for i in range(height - 1):
        for j in range(width - 1):
            ind = i * width + j
            s.append(d.Simplex([ind, ind + width + 1]))
            s.append(d.Simplex([ind, ind + 1, ind + width + 1]))
            s.append(d.Simplex([ind, ind + width, ind + width + 1]))
    return s


class LevelSetLayer(nn.Module):
    """
    Level set persistence layer
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : haximum homology dimension (default 1)
        complex :
            "scipy" - use scipy freudenthal triangulation (default)
            "freudenthal" - use canonical freudenthal triangulation
    """

    def __init__(self, size, maxdim=1, complex='scipy'):
        super(LevelSetLayer, self).__init__()
        self.size = size
        self.maxdim = maxdim
        self.fnobj = levelsetdgm()
        width, height = size
        if complex == 'scipy':
            axis_x = np.arange(0, width)
            axis_y = np.arange(0, height)
            grid_axes = np.array(np.meshgrid(axis_x, axis_y))
            grid_axes = np.transpose(grid_axes, (1, 2, 0))
            tri = Delaunay(grid_axes.reshape([-1, 2]))
            faces = tri.simplices.copy()
            self.complex = self.fnobj.init_filtration(faces)
        elif complex == 'freudenthal':
            self.complex = init_freudenthal_2d(width, height)
        else:
            AssertionError('bad complex type')

    def forward(self, img):
        dgm = self.fnobj.apply(img, self.complex)
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.maxdim + 1))
        return dgms, False


def init_grid_2d(width, height):
    """
    initialize 2d grid with diagonal and anti-diagonal
    """
    s = SimplicialComplex()
    for i in range(height):
        for j in range(width):
            ind = i * width + j
            s.append([ind])
    for i in range(height):
        for j in range(width - 1):
            ind = i * width + j
            s.append([ind, ind + 1])
    for i in range(height - 1):
        for j in range(width):
            ind = i * width + j
            s.append([ind, ind + width])
    for i in range(height - 1):
        for j in range(width - 1):
            ind = i * width + j
            s.append([ind, ind + width + 1])
            s.append([ind, ind + 1, ind + width + 1])
            s.append([ind, ind + width, ind + width + 1])
    for i in range(height - 1):
        for j in range(width - 1):
            ind = i * width + j
            s.append([ind + 1, ind + width])
            s.append([ind + 1, ind + width, ind + width + 1])
            s.append([ind, ind + 1, ind + width])
    return s


def unique_simplices(faces, dim):
    """
    obtain unique simplices up to dimension dim from faces
    """
    simplices = [[] for k in range(dim + 1)]
    for face in faces:
        for k in range(dim + 1):
            for s in combinations(face, k + 1):
                simplices[k].append(np.sort(list(s)))
    s = SimplicialComplex()
    for k in range(dim + 1):
        kcells = np.unique(simplices[k], axis=0)
        for cell in kcells:
            s.append(cell)
    return s


def init_tri_complex(width, height):
    """
    initialize 2d complex in dumbest possible way
    """
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y))
    grid_axes = np.transpose(grid_axes, (1, 2, 0))
    tri = Delaunay(grid_axes.reshape([-1, 2]))
    return unique_simplices(tri.simplices, 2)


class LevelSetLayer2D(LevelSetLayer):
    """
    Level set persistence layer for 2D input
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
        complex : method of constructing complex
            "freudenthal" (default) - canonical triangulation of the lattice
            "grid" - includes diagonals and anti-diagonals
            "delaunay" - scipy delaunay triangulation of the lattice.
                Every square will be triangulated, but the diagonal orientation may not be consistent.
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """

    def __init__(self, size, maxdim=1, sublevel=True, complex='freudenthal', alg='hom'):
        width, height = size
        tmpcomplex = None
        if complex == 'freudenthal':
            tmpcomplex = init_freudenthal_2d(width, height)
        elif complex == 'grid':
            tmpcomplex = init_grid_2d(width, height)
        elif complex == 'delaunay':
            tmpcomplex = init_tri_complex(width, height)
        super(LevelSetLayer2D, self).__init__(tmpcomplex, maxdim=maxdim, sublevel=sublevel, alg=alg)
        self.size = size


def init_line_complex(p):
    """
    initialize 1D complex on the line
    Input:
        p - number of 0-simplices
    Will add (p-1) 1-simplices
    """
    f = d.Filtration()
    for i in range(p - 1):
        c = d.closure([d.Simplex([i, i + 1])], 1)
        for j in c:
            f.append(j)
    return f


class LevelSetLayer1D(nn.Module):
    """
    Level set persistence layer
    Parameters:
        size : number of features
    only returns H0
    """

    def __init__(self, size):
        super(LevelSetLayer1D, self).__init__()
        self.size = size
        self.fnobj = levelsetdgm()
        self.complex = init_line_complex(size)

    def forward(self, img):
        dgm = self.fnobj.apply(img, self.complex)
        dgm = dgm[0]
        return (dgm,), False


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NormLoss,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SobLoss,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bruel_gabrielsson_TopologyLayer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

