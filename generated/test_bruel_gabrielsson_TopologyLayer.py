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
nn = _module
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


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import sampler


import numpy as np


from scipy.spatial import Delaunay


import itertools


ape = 1, 28, 28


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(100, 128), nn.LeakyReLU(0.2,
            inplace=True), nn.Linear(128, 256), nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 512), nn.
            BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.
            Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2,
            inplace=True), nn.Linear(1024, int(np.prod(ape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *ape)
        return img


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


class TopLoss(nn.Module):

    def __init__(self):
        super(TopLoss, self).__init__()
        self.pdfn = AlphaLayer(maxdim=0)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgms, issublevel = self.pdfn(beta)
        return self.topfn((dgms[0], issublevel))


class TopLoss2(nn.Module):

    def __init__(self):
        super(TopLoss2, self).__init__()
        self.pdfn = AlphaLayer(maxdim=0)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgms, issublevel = self.pdfn(beta)
        return self.topfn((dgms[0], issublevel))


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


class TopLoss2(nn.Module):

    def __init__(self, p):
        super(TopLoss2, self).__init__()
        self.pdfn = LevelSetLayer1D(p, False)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgms, issublevel = self.pdfn(beta)
        return self.topfn((dgms[0], issublevel))


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


class TopLoss(nn.Module):

    def __init__(self):
        super(TopLoss, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


class TopLoss2(nn.Module):

    def __init__(self):
        super(TopLoss2, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


def delaunay_complex(x, maxdim=2):
    """
    compute Delaunay triangulation
    fill in simplices as appropriate

    if x is 1-dimensional, defaults to 1D Delaunay
    inputs:
        x - pointcloud
        maxdim - maximal simplex dimension (default = 2)
    """
    if x.shape[1] == 1:
        x = x.flatten()
        return alpha_complex_1d(x)
    tri = Delaunay(x)
    return unique_simplices(tri.simplices, maxdim)


def delaunay_complex_1d(x):
    """
    returns Delaunay complex on 1D space
    """
    inds = np.argsort(x)
    s = SimplicialComplex()
    s.append([inds[0]])
    for ii in range(len(inds) - 1):
        s.append([inds[ii + 1]])
        s.append([inds[ii], inds[ii + 1]])
    return s


class FlagDiagram(Function):
    """
    Compute Flag complex persistence using point coordinates

    forward inputs:
        X - simplicial complex
        y - N x D torch.float tensor of coordinates
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """

    @staticmethod
    def forward(ctx, X, y, maxdim, alg='hom'):
        device = y.device
        ctx.device = device
        ycpu = y.cpu()
        X.extendFlag(ycpu)
        if alg == 'hom':
            ret = persistenceForwardHom(X, maxdim, 0)
        elif alg == 'hom2':
            ret = persistenceForwardHom(X, maxdim, 1)
        elif alg == 'cohom':
            ret = persistenceForwardCohom(X, maxdim)
        ctx.X = X
        ctx.save_for_backward(ycpu)
        ret = [r.to(device) for r in ret]
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        X = ctx.X
        device = ctx.device
        ycpu, = ctx.saved_tensors
        grad_ret = [gd.cpu() for gd in grad_dgms]
        grad_y = persistenceBackwardFlag(X, ycpu, grad_ret)
        return None, grad_y.to(device), None, None


class AlphaLayer(nn.Module):
    """
    Alpha persistence layer
    Parameters:
        maxdim : maximum homology dimension (default=0)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """

    def __init__(self, maxdim=0, alg='hom'):
        super(AlphaLayer, self).__init__()
        self.maxdim = maxdim
        self.fnobj = FlagDiagram()
        self.alg = alg

    def forward(self, x):
        xnp = x.cpu().detach().numpy()
        complex = None
        if xnp.shape[1] == 1:
            xnp = xnp.flatten()
            complex = delaunay_complex_1d(xnp)
        else:
            complex = delaunay_complex(xnp, maxdim=self.maxdim + 1)
        complex.initialize()
        dgms = self.fnobj.apply(complex, x, self.maxdim, self.alg)
        return dgms, True


def remove_filler(dgm, val=np.inf):
    """
    remove filler rows from diagram
    """
    inds = dgm[:, (0)] != val
    return dgm[(inds), :]


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
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.
            maxdim + 1))
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


def remove_zero_bars(dgm):
    """
    remove zero bars from diagram
    """
    inds = dgm[:, (0)] != dgm[:, (1)]
    return dgm[(inds), :]


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
        return torch.sum(torch.mul(torch.pow(lengths, self.p), torch.pow(
            means, self.q)))


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


class SubLevelSetDiagram(Function):
    """
    Compute sub-level set persistence on a space
    forward inputs:
        X - simplicial complex
        f - torch.float tensor of function values on vertices of X
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """

    @staticmethod
    def forward(ctx, X, f, maxdim, alg='hom'):
        ctx.retshape = f.shape
        f = f.view(-1)
        device = f.device
        ctx.device = device
        X.extendFloat(f.cpu())
        if alg == 'hom':
            ret = persistenceForwardHom(X, maxdim, 0)
        elif alg == 'hom2':
            ret = persistenceForwardHom(X, maxdim, 1)
        elif alg == 'cohom':
            ret = persistenceForwardCohom(X, maxdim)
        ctx.X = X
        ret = [r.to(device) for r in ret]
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        X = ctx.X
        device = ctx.device
        retshape = ctx.retshape
        grad_ret = [gd.cpu() for gd in grad_dgms]
        grad_f = persistenceBackward(X, grad_ret)
        return None, grad_f.view(retshape).to(device), None, None


class LevelSetLayer(nn.Module):
    """
    Level set persistence layer arbitrary simplicial complex
    Parameters:
        complex : SimplicialComplex
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology

    Note that the complex should be acyclic for the computation to be correct (currently)
    """

    def __init__(self, complex, maxdim=1, sublevel=True, alg='hom'):
        super(LevelSetLayer, self).__init__()
        self.complex = complex
        self.maxdim = maxdim
        self.fnobj = SubLevelSetDiagram()
        self.sublevel = sublevel
        self.alg = alg
        self.complex.initialize()

    def forward(self, f):
        if self.sublevel:
            dgms = self.fnobj.apply(self.complex, f, self.maxdim, self.alg)
            return dgms, True
        else:
            f = -f
            dgms = self.fnobj.apply(self.complex, f, self.maxdim, self.alg)
            dgms = tuple(-dgm for dgm in dgms)
            return dgms, False


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
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.
            maxdim + 1))
        return dgms, False


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


def clique_complex(n, d):
    """
    Create d-skeleton of clique complex on n vertices
    """
    s = SimplicialComplex()
    for k in range(d + 1):
        for cell in combinations(range(n), k + 1):
            s.append(list(cell))
    return s


class RipsLayer(nn.Module):
    """
    Rips persistence layer
    Parameters:
        n : number of points
        maxdim : maximum homology dimension (default=1)
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """

    def __init__(self, n, maxdim=1, alg='hom'):
        super(RipsLayer, self).__init__()
        self.maxdim = maxdim
        self.complex = clique_complex(n, maxdim + 1)
        self.complex.initialize()
        self.fnobj = FlagDiagram()
        self.alg = alg

    def forward(self, x):
        dgms = self.fnobj.apply(self.complex, x, self.maxdim, self.alg)
        return dgms, True


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
        dgms = tuple(remove_filler(dgm[i], -np.inf) for i in range(self.
            maxdim + 1))
        return dgms, True


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_bruel_gabrielsson_TopologyLayer(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(Generator(*[], **{}), [torch.rand([100, 100])], {})

    def test_001(self):
        self._check(SobLoss(*[], **{'p': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(NormLoss(*[], **{'p': 4}), [torch.rand([4, 4, 4, 4])], {})
