import sys
_module = sys.modules[__name__]
del sys
ising = _module
args = _module
ipeps = _module
measure = _module
utils = _module
variational = _module
setup = _module
tensornets = _module
adlib = _module
eigh = _module
power = _module
qr = _module
svd = _module
ctmrg = _module
test_power = _module
test_qr = _module
test_svd = _module
trg = _module

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


import time


import re


from torch.utils.checkpoint import detach_variable


from torch.utils.checkpoint import checkpoint


def safe_inverse(x, epsilon=1e-12):
    return x / (x ** 2 + epsilon)


class SVD(torch.autograd.Function):

    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        F = S - S[:, None]
        F = safe_inverse(F)
        F.diagonal().fill_(0)
        G = S + S[:, None]
        G.diagonal().fill_(np.inf)
        G = 1 / G
        UdU = Ut @ dU
        VdV = Vt @ dV
        Su = (F + G) * (UdU - UdU.t()) / 2
        Sv = (F - G) * (VdV - VdV.t()) / 2
        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if M > NS:
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U @ Ut) @ (dU / S) @ Vt
        if N > NS:
            dA = dA + U / S @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V @ Vt)
        return dA


svd = SVD.apply


def renormalize(*args):
    T, chi, epsilon = args
    D = T.shape[0]
    Ma = T.view(D ** 2, D ** 2)
    Mb = T.permute(1, 2, 0, 3).contiguous().view(D ** 2, D ** 2)
    Ua, Sa, Va = svd(Ma)
    Ub, Sb, Vb = svd(Mb)
    D_new = min(min(D ** 2, chi), min((Sa > epsilon).sum().item(), (Sb > epsilon).sum().item()))
    S1 = (Ua[:, :D_new] * torch.sqrt(Sa[:D_new])).view(D, D, D_new)
    S3 = (Va[:, :D_new] * torch.sqrt(Sa[:D_new])).view(D, D, D_new)
    S2 = (Ub[:, :D_new] * torch.sqrt(Sb[:D_new])).view(D, D, D_new)
    S4 = (Vb[:, :D_new] * torch.sqrt(Sb[:D_new])).view(D, D, D_new)
    return torch.einsum('xwu,yxl,yzd,wzr->uldr', (S2, S3, S4, S1))


def CTMRG(T, chi, max_iter, use_checkpoint=False):
    threshold = 1e-12 if T.dtype is torch.float64 else 1e-06
    C = T.sum((0, 1))
    E = T.sum(1).permute(0, 2, 1)
    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 10.0
    for n in range(max_iter):
        tensors = C, E, T, torch.tensor(chi)
        if use_checkpoint:
            C, E, s, error = checkpoint(renormalize, *tensors)
        else:
            C, E, s, error = renormalize(*tensors)
        Enorm = E.norm()
        E = E / Enorm
        truncation_error += error.item()
        if s.numel() == sold.numel():
            diff = (s - sold).norm().item()
        if diff < threshold:
            break
        sold = s
    return C, E


def get_obs(Asymm, H, Sx, Sy, Sz, C, E):
    Da = Asymm.size()
    Td = torch.einsum('mefgh,nabcd->eafbgchdmn', (Asymm, Asymm)).contiguous().view(Da[1] ** 2, Da[2] ** 2, Da[3] ** 2, Da[4] ** 2, Da[0], Da[0])
    CE = torch.tensordot(C, E, ([1], [0]))
    EL = torch.tensordot(E, CE, ([2], [0]))
    EL = torch.tensordot(EL, Td, ([1, 2], [1, 0]))
    EL = torch.tensordot(EL, CE, ([0, 2], [0, 1]))
    Rho = torch.tensordot(EL, EL, ([0, 1, 4], [0, 1, 4])).permute(0, 2, 1, 3).contiguous().view(Da[0] ** 2, Da[0] ** 2)
    Rho = 0.5 * (Rho + Rho.t())
    Tnorm = Rho.trace()
    Energy = torch.mm(Rho, H).trace() / Tnorm
    Mx = torch.mm(Rho, Sx).trace() / Tnorm
    My = torch.mm(Rho, Sy).trace() / Tnorm
    Mz = torch.mm(Rho, Sz).trace() / Tnorm
    return Energy, Mx, My, Mz


def symmetrize(A):
    """
    A(phy, up, left, down, right)
    left-right, up-down, diagonal symmetrize
    """
    Asymm = (A + A.permute(0, 1, 4, 3, 2)) / 2.0
    Asymm = (Asymm + Asymm.permute(0, 3, 2, 1, 4)) / 2.0
    Asymm = (Asymm + Asymm.permute(0, 4, 3, 2, 1)) / 2.0
    Asymm = (Asymm + Asymm.permute(0, 2, 1, 4, 3)) / 2.0
    return Asymm / Asymm.norm()


class iPEPS(torch.nn.Module):

    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(iPEPS, self).__init__()
        self.d = args.d
        self.D = args.D
        self.chi = args.chi
        self.Niter = args.Niter
        self.use_checkpoint = use_checkpoint
        d, D = self.d, self.D
        B = torch.rand(d, D, D, D, D, dtype=dtype, device=device)
        B = B / B.norm()
        self.A = torch.nn.Parameter(B)

    def forward(self, H, Mpx, Mpy, Mpz, chi):
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter
        Asymm = symmetrize(self.A)
        T = (Asymm.view(d, -1).t() @ Asymm.view(d, -1)).contiguous().view(D, D, D, D, D, D, D, D)
        T = T.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(D ** 2, D ** 2, D ** 2, D ** 2)
        T = T / T.norm()
        C, E = CTMRG(T, chi, Niter, self.use_checkpoint)
        loss, Mx, My, Mz = get_obs(Asymm, H, Mpx, Mpy, Mpz, C, E)
        return loss, Mx, My, Mz

