import sys
_module = sys.modules[__name__]
del sys
asteroid = _module
binarize = _module
complex_nn = _module
data = _module
avspeech_dataset = _module
dampvsep_dataset = _module
dns_dataset = _module
fuss_dataset = _module
kinect_wsj = _module
librimix_dataset = _module
musdb18_dataset = _module
sms_wsj_dataset = _module
utils = _module
vad_dataset = _module
wham_dataset = _module
whamr_dataset = _module
wsj0_mix = _module
dsp = _module
beamforming = _module
consistency = _module
deltas = _module
normalization = _module
overlap_add = _module
spatial = _module
vad = _module
engine = _module
optimizers = _module
schedulers = _module
system = _module
losses = _module
cluster = _module
mixit_wrapper = _module
mse = _module
multi_scale_spectral = _module
pit_wrapper = _module
pmsqe = _module
sdr = _module
sinkpit_wrapper = _module
soft_f1 = _module
stoi = _module
masknn = _module
_dccrn_architectures = _module
_dcunet_architectures = _module
_local = _module
activations = _module
attention = _module
base = _module
convolutional = _module
norms = _module
recurrent = _module
tac = _module
metrics = _module
models = _module
base_models = _module
conv_tasnet = _module
dccrnet = _module
dcunet = _module
demask = _module
dprnn_tasnet = _module
dptnet = _module
fasnet = _module
lstm_tasnet = _module
publisher = _module
sudormrf = _module
x_umx = _module
zenodo = _module
scripts = _module
asteroid_cli = _module
asteroid_versions = _module
separate = _module
deprecation_utils = _module
generic_utils = _module
hub_utils = _module
parser_utils = _module
test_utils = _module
torch_utils = _module
conf = _module
eval = _module
train = _module
eval = _module
parse_data = _module
tac_dataset = _module
train = _module
eval = _module
local = _module
loader = _module
audio_mixer_generator = _module
constants = _module
download = _module
extract_audio = _module
frames = _module
generate_video_embedding = _module
remove_corrupt = _module
remove_empty_audio = _module
postprocess = _module
postprocess_audio = _module
model = _module
train = _module
callbacks = _module
config = _module
metric_utils = _module
trainer = _module
eval = _module
train = _module
demask_dataset = _module
train = _module
denoise = _module
eval_on_synthetic = _module
preprocess_dns = _module
model = _module
train = _module
eval = _module
preprocess_kinect_wsj = _module
train = _module
eval = _module
create_local_metadata = _module
get_text = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
test_dataloader = _module
eval = _module
dataloader = _module
train = _module
start_evaluation = _module
eval = _module
preprocess_wham = _module
train = _module
eval = _module
train = _module
eval = _module
train = _module
eval = _module
augmented_wham = _module
resample_dataset = _module
model = _module
train = _module
get_training_stats = _module
model = _module
train = _module
eval = _module
train = _module
eval = _module
model = _module
system = _module
train = _module
eval = _module
preprocess_whamr = _module
model = _module
train = _module
eval = _module
preprocess_wsj0mix = _module
metrics = _module
model = _module
train = _module
wsj0_mix_variable = _module
eval = _module
model = _module
train = _module
hubconf = _module
setup = _module
binarize_test = _module
cli_setup = _module
cli_test = _module
complex_nn_test = _module
beamforming_test = _module
consistency_test = _module
deltas_tests = _module
normalization_test = _module
overlap_add_test = _module
spatial_test = _module
vad_test = _module
optimizers_test = _module
scheduler_test = _module
system_test = _module
jit = _module
jit_filterbanks_test = _module
jit_masknn_test = _module
jit_models_test = _module
jit_torch_utils_test = _module
loss_functions_test = _module
mixit_wrapper_test = _module
pit_wrapper_test = _module
sinkpit_wrapper_test = _module
activations_test = _module
convolutional_test = _module
norms_test = _module
recurrent_test = _module
metrics_test = _module
demask_test = _module
fasnet_test = _module
models_test = _module
publish_test = _module
xumx_test = _module
deprecation_utils_test = _module
hub_utils_test = _module
torch_utils_test = _module
utils_test = _module
dummy_test = _module

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


from itertools import groupby


import torch


import functools


from torch import nn


import re


import numpy as np


from torch.utils import data


from torch.nn import functional as F


import pandas as pd


from typing import Union


import torch.utils.data


import random


from torch.utils.data import Dataset


from torch import hub


from torch.utils.data import DataLoader


import random as random


from torch.utils.data._utils.collate import default_collate


from typing import Optional


from typing import List


import torch.nn.functional as F


from torch.optim.optimizer import Optimizer


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim import SGD


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adamax


from torch.optim import AdamW


from torch.optim import ASGD


from torch.optim.lr_scheduler import ReduceLROnPlateau


import warnings


from itertools import combinations


from torch.nn.modules.loss import _Loss


import torch.nn as nn


from itertools import permutations


from scipy.optimize import linear_sum_assignment


from torch import tensor


from functools import partial


from math import ceil


from torch.nn.modules.activation import MultiheadAttention


from typing import Tuple


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import fold


from torch.nn.functional import unfold


from collections import Counter


from copy import deepcopy


import math


from torch.nn import LSTM


from torch.nn import Linear


from torch.nn import BatchNorm1d


from torch.nn import Parameter


import itertools


from functools import lru_cache


from typing import Dict


from collections import OrderedDict


import matplotlib.pyplot as plt


import collections


import torchvision


from scipy.signal import fftconvolve


from scipy.signal import firwin2


from torch.utils.data import random_split


import scipy.signal


import copy


import sklearn.preprocessing


from torch.optim.lr_scheduler import ExponentialLR


from torch.nn.parallel import DistributedDataParallel


import torch.utils.data as data


from time import time


from sklearn.cluster import KMeans


from torch.testing import assert_allclose


from torch import optim


from collections.abc import MutableMapping


def count_same_pair(nums):
    """Transform a list of 0 and 1 in a list of (value, num_consecutive_occurences).

    Args:
        nums (list): List of list containing the binary sequences.

    Returns:
        List of values and number consecutive occurences.

    Example:
        >>> nums = [[0,0,1,0]]
        >>> result = count_same_pair(nums)
        >>> print(result)
        [[[0, 2], [1, 1], [0, 1]]]

    """
    result = []
    for num in nums:
        result.append([[i, sum(1 for _ in group)] for i, group in groupby(num)])
    return result


def check_silence_or_voice(active, pair):
    """Check if sequence is fully silence or fully voice.

    Args:
        active (List) : List containing the binary sequence
        pair: (List): list of value and consecutive occurrences

    """
    value, num_consecutive_occurrences = pair[0]
    check = False
    if len(pair) == 1:
        check = True
        if value:
            active = torch.ones(num_consecutive_occurrences)
        else:
            active = torch.zeros(num_consecutive_occurrences)
    return active, check


def resolve_instability(i, pair, stability, sample_rate, actived, not_actived, active):
    """Resolve stability issue in input list of value and num_consecutive_occ

    Args:
        i (int): The index of the considered pair of value and num_consecutive_occ.
        pair (list) : Value and num_consecutive_occ.
        stability (float): Minimal number of seconds to change from 0 to 1 or 1 to 0.
        sample_rate (int): The sample rate of the waveform.
        actived (int) : Number of occurrences of the value 1.
        not_actived (int): Number of occurrences of the value 0.
        active (list) : The binary sequence.

    Returns:
        active (list) : The binary sequence.
         i (int): The index of the considered pair of value and num_consecutive_occ.
    """
    while i < len(pair) and pair[i][1] < int(stability * sample_rate):
        value, num_consecutive_occurrences = pair[i]
        if value:
            actived += num_consecutive_occurrences
            i += 1
        else:
            not_actived += num_consecutive_occurrences
            i += 1
    if actived + not_actived < int(stability * sample_rate) and len(active) > 0:
        if active[-1][0] == 1:
            active.append(torch.ones(actived + not_actived))
        else:
            active.append(torch.zeros(actived + not_actived))
    elif actived + not_actived < int(stability * sample_rate) and len(active) == 0:
        active.append(torch.zeros(actived + not_actived))
    elif actived > not_actived:
        active.append(torch.ones(actived + not_actived))
    else:
        active.append(torch.zeros(actived + not_actived))
    return active, i


def transform_to_binary_sequence(pairs, stability, sample_rate):
    """Transforms list of value and consecutive occurrences into a binary sequence with respect to stability

    Args:
        pairs (List): List of list of value and consecutive occurrences
        stability (Float): Minimal number of seconds to change from 0 to 1 or 1 to 0.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        Torch.tensor : The binary sequences.
    """
    batch_active = []
    for pair in pairs:
        active = []
        active, check = check_silence_or_voice(active, pair)
        if check:
            return active
        i = 0
        while i < len(pair):
            value, num_consecutive_occurrences = pair[i]
            actived = 0
            not_actived = 0
            if num_consecutive_occurrences < int(stability * sample_rate):
                active, i = resolve_instability(i, pair, stability, sample_rate, actived, not_actived, active)
            else:
                if value:
                    active.append(torch.ones(pair[i][1]))
                else:
                    active.append(torch.zeros(pair[i][1]))
                i += 1
        batch_active.append(torch.hstack(active))
    batch_active = torch.vstack(batch_active).unsqueeze(1)
    return batch_active


class Binarize(torch.nn.Module):
    """This module transform a sequence of real numbers between 0 and 1 to a sequence of 0 or 1.
    The logic for transformation is based on thresholding and avoids jumping from 0 to 1 inadvertently.

    Example:

        >>> binarizer = Binarize(threshold=0.5, stability=3, sample_rate=1)
        >>> inputs=torch.Tensor([0.1, 0.6, 0.2, 0.6, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1])
        >>>                    # |------------------|-------------|----------------------------|----|
        >>>                    #    unstable          stable             stable                 irregularity
        >>> result = binarizer(inputs.unsqueeze(0).unsqueeze(0))
        >>> print(result)
        tensor([[[0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.]]])

    """

    def __init__(self, threshold=0.5, stability=0.1, sample_rate=8000):
        """

        Args:
            threshold (float): if x > threshold 0 else 1
            stability (float): Minimum number of seconds of 0 (or 1) required to change from 1 (or 0) to 0 (or 1)
            sample_rate (int): The sample rate of the wave form
        """
        super().__init__()
        self.threshold = threshold
        self.stability = stability
        self.sample_rate = sample_rate

    def forward(self, x):
        active = x > self.threshold
        active = active.squeeze(1).tolist()
        pairs = count_same_pair(active)
        active = transform_to_binary_sequence(pairs, self.stability, self.sample_rate)
        return active


def torch_complex_from_reim(re, im):
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


class OnReIm(nn.Module):
    """Like `on_reim`, but for stateful modules.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Called 2x with *args, **kwargs, to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x):
        return torch_complex_from_reim(self.re_module(x.real), self.im_module(x.imag))


ComplexTensor = torch.Tensor


class ComplexMultiplicationWrapper(nn.Module):
    """Make a complex-valued module `F` from a real-valued module `f` by applying
    complex multiplication rules:

    F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

    where `f1`, `f2` are instances of `f` that do *not* share weights.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Constructor of `f` in the formula above.  Called 2x with `*args`, `**kwargs`,
            to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x: ComplexTensor) ->ComplexTensor:
        return torch_complex_from_reim(self.re_module(x.real) - self.im_module(x.imag), self.re_module(x.imag) + self.im_module(x.real))


class ComplexSingleRNN(nn.Module):
    """Module for a complex RNN block.

    This is similar to :cls:`asteroid.masknn.recurrent.SingleRNN` but uses complex
    multiplication as described in [1]. Arguments are identical to those of `SingleRNN`,
    except for `dropout`, which is not yet supported.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
        dropout: Not yet supported.

    References
        [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        assert not (dropout and n_layers > 1), 'Dropout is not yet supported for complex RNN'
        super().__init__()
        kwargs = {'rnn_type': rnn_type, 'hidden_size': hidden_size, 'n_layers': 1, 'dropout': 0, 'bidirectional': bidirectional}
        first_rnn = ComplexMultiplicationWrapper(SingleRNN, input_size=input_size, **kwargs)
        self.rnns = torch.nn.ModuleList([first_rnn])
        for _ in range(n_layers - 1):
            self.rnns.append(ComplexMultiplicationWrapper(SingleRNN, input_size=first_rnn.re_module.output_size, **kwargs))

    @property
    def output_size(self):
        return self.rnns[-1].re_module.output_size

    def forward(self, x: ComplexTensor) ->ComplexTensor:
        """Input shape [batch, seq, feats]"""
        for rnn in self.rnns:
            x = rnn(x)
        return x


def on_reim(f):
    """Make a complex-valued function callable from a real-valued one by applying it to
    the real and imaginary components independently.

    Return:
        cf(x), complex version of `f`: A function that applies `f` to the real and
        imaginary components of `x` and returns the result as PyTorch complex tensor.
    """

    @functools.wraps(f)
    def cf(x):
        return torch_complex_from_reim(f(x.real), f(x.imag))
    cf.__name__ == f'{f.__name__} (complex)'
    cf.__qualname__ == f'{f.__qualname__} (complex)'
    return cf


def torch_complex_from_magphase(mag, phase):
    return torch.view_as_complex(torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1))


def bound_complex_mask(mask: ComplexTensor, bound_type='tanh'):
    """Bound a complex mask, as proposed in [1], section 3.2.

    Valid bound types, for a complex mask :math:`M = |M| ⋅ e^{i φ(M)}`:

    - Unbounded ("UBD"): :math:`M_{\\mathrm{UBD}} = M`
    - Sigmoid ("BDSS"): :math:`M_{\\mathrm{BDSS}} = σ(|M|) e^{i σ(φ(M))}`
    - Tanh ("BDT"): :math:`M_{\\mathrm{BDT}} = \\mathrm{tanh}(|M|) e^{i φ(M)}`

    Args:
        bound_type (str or None): The type of bound to use, either of
            "tanh"/"bdt" (default), "sigmoid"/"bdss" or None/"bdt".

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    if bound_type in {'BDSS', 'sigmoid'}:
        return on_reim(torch.sigmoid)(mask)
    elif bound_type in {'BDT', 'tanh', 'UBD', None}:
        mask_mag, mask_phase = transforms.magphase(transforms.from_torch_complex(mask))
        if bound_type in {'BDT', 'tanh'}:
            mask_mag_bounded = torch.tanh(mask_mag)
        else:
            mask_mag_bounded = mask_mag
        return torch_complex_from_magphase(mask_mag_bounded, mask_phase)
    else:
        raise ValueError(f'Unknown mask bound {bound_type}')


class BoundComplexMask(nn.Module):
    """Module version of `bound_complex_mask`"""

    def __init__(self, bound_type):
        super().__init__()
        self.bound_type = bound_type

    def forward(self, mask: ComplexTensor):
        return bound_complex_mask(mask, self.bound_type)


def compute_scm(x: torch.Tensor, mask: torch.Tensor=None, normalize: bool=True):
    """Compute the spatial covariance matrix from a STFT signal x.

    Args:
        x (torch.ComplexTensor): shape  [batch, mics, freqs, frames]
        mask (torch.Tensor): [batch, 1, freqs, frames] or [batch, 1, freqs, frames]. Optional
        normalize (bool): Whether to normalize with the mask mean per bin.

    Returns:
        torch.ComplexTensor, the SCM with shape (batch, mics, mics, freqs)
    """
    batch, mics, freqs, frames = x.shape
    if mask is None:
        mask = torch.ones(batch, 1, freqs, frames)
    if mask.ndim == 3:
        mask = mask[:, None]
    scm = torch.einsum('bmft,bnft->bmnf', mask * x, x.conj())
    if normalize:
        scm /= mask.sum(-1, keepdim=True).transpose(-1, -2)
    return scm


class SCM(nn.Module):

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, normalize: bool=True):
        """See :func:`compute_scm`."""
        return compute_scm(x, mask=mask, normalize=normalize)


def get_optimal_reference_mic(bf_mat: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor, eps: float=1e-06):
    """Compute the optimal reference mic given the a posteriori SNR, see [1].

    Args:
        bf_mat: (batch, freq, mics, mics)
        target_scm (torch.ComplexTensor): (batch, freqs, mics, mics)
        noise_scm (torch.ComplexTensor): (batch, freqs, mics, mics)
        eps: value to clip the denominator.

    Returns:
        torch.

    References
        Erdogan et al. 2016: "Improved MVDR beamforming using single-channel maskprediction networks"
            https://www.merl.com/publications/docs/TR2016-072.pdf
    """
    den = torch.clamp(torch.einsum('...flm,...fln,...fnm->...m', bf_mat.conj(), noise_scm, bf_mat).real, min=eps)
    snr_post = torch.einsum('...flm,...fln,...fnm->...m', bf_mat.conj(), target_scm, bf_mat).real / den
    assert torch.all(torch.isfinite(snr_post)), snr_post
    return torch.argmax(snr_post, dim=-1)


class Beamformer(nn.Module):
    """Base class for beamforming modules."""

    @staticmethod
    def apply_beamforming_vector(bf_vector: torch.Tensor, mix: torch.Tensor):
        """Apply the beamforming vector to the mixture. Output (batch, freqs, frames).

        Args:
            bf_vector: shape (batch, mics, freqs)
            mix: shape (batch, mics, freqs, frames).
        """
        return torch.einsum('...mf,...mft->...ft', bf_vector.conj(), mix)

    @staticmethod
    def get_reference_mic_vects(ref_mic, bf_mat: torch.Tensor, target_scm: torch.Tensor=None, noise_scm: torch.Tensor=None):
        """Return the reference channel indices over the batch.

        Args:
            ref_mic (Optional[Union[int, torch.Tensor]]): The reference channel.
                If torch.Tensor (ndim>1), return it, it is the reference mic vector,
                If torch.LongTensor of size `batch`, select independent reference mic of the batch.
                If int, select the corresponding reference mic,
                If None, the optimal reference mics are computed with :func:`get_optimal_reference_mic`,
                If None, and either SCM is None, `ref_mic` is set to `0`,
            bf_mat: beamforming matrix of shape (batch, freq, mics, mics).
            target_scm (torch.ComplexTensor): (batch, freqs, mics, mics).
            noise_scm (torch.ComplexTensor): (batch, freqs, mics, mics).

        Returns:
            torch.LongTensor of size ``batch`` to select with the reference channel indices.
        """
        if isinstance(ref_mic, torch.Tensor) and ref_mic.ndim > 1:
            return ref_mic
        if (target_scm is None or noise_scm is None) and ref_mic is None:
            ref_mic = 0
        if ref_mic is None:
            batch_mic_idx = get_optimal_reference_mic(bf_mat=bf_mat, target_scm=target_scm, noise_scm=noise_scm)
        elif isinstance(ref_mic, int):
            batch_mic_idx = torch.LongTensor([ref_mic] * bf_mat.shape[0])
        elif isinstance(ref_mic, torch.Tensor):
            batch_mic_idx = ref_mic
        else:
            raise ValueError(f'Unsupported reference microphone format. Support None, int and 1D torch.LongTensor and torch.Tensor, received {type(ref_mic)}.')
        ref_mic_vects = F.one_hot(batch_mic_idx, num_classes=bf_mat.shape[-1])[:, None, :, None]
        return ref_mic_vects.to(bf_mat.dtype)


def _common_dtype(*args):
    all_dtypes = [a.dtype for a in args]
    if len(set(all_dtypes)) > 1:
        raise RuntimeError(f'Expected inputs from the same dtype. Received {all_dtypes}.')
    return all_dtypes[0]


USE_DOUBLE = True


def _precision_mapping():
    has_complex32 = hasattr(torch, 'complex32')
    if USE_DOUBLE:
        precision_map = {torch.float16: torch.float64, torch.float32: torch.float64, torch.complex64: torch.complex128}
        if has_complex32:
            precision_map[torch.complex32] = torch.complex128
    else:
        precision_map = {torch.float16: torch.float16, torch.float32: torch.float32, torch.complex64: torch.complex64}
        if has_complex32:
            precision_map[torch.complex32] = torch.complex32
    return precision_map


def batch_trace(x, dim1=-2, dim2=-1):
    """Compute the trace along `dim1` and `dim2` for a any matrix `ndim>=2`."""
    return torch.diagonal(x, dim1=dim1, dim2=dim2).sum(-1)


def condition_scm(x, eps=1e-06, dim1=-2, dim2=-1):
    """Condition input SCM with (x + eps tr(x) I) / (1 + eps) along `dim1` and `dim2`.

    See https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3).
    """
    if dim1 != -2 or dim2 != -1:
        raise NotImplementedError
    scale = eps * batch_trace(x, dim1=dim1, dim2=dim2)[..., None, None] / x.shape[dim1]
    scaled_eye = torch.eye(x.shape[dim1], device=x.device)[None, None] * scale
    return (x + scaled_eye) / (1 + eps)


def _stable_solve(b, a, eps=1e-06):
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        a = condition_scm(a, eps)
        return torch.linalg.solve(a, b)


def stable_solve(b, a):
    """Return torch.solve if `a` is non-singular, else regularize `a` and return torch.solve."""
    input_dtype = _common_dtype(b, a)
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    return _stable_solve(b.to(solve_dtype), a.to(solve_dtype))


class RTFMVDRBeamformer(Beamformer):

    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute and apply MVDR beamformer from the speech and noise SCM matrices.

        :math:`\\mathbf{w} =  \\displaystyle \\frac{\\Sigma_{nn}^{-1} \\mathbf{a}}{
        \\mathbf{a}^H \\Sigma_{nn}^{-1} \\mathbf{a}}` where :math:`\\mathbf{a}` is the
        ATF estimated from the target SCM.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        e_val, e_vec = torch.symeig(target_scm.permute(0, 3, 1, 2), eigenvectors=True)
        rtf_vect = e_vec[..., -1]
        return self.from_rtf_vect(mix=mix, rtf_vec=rtf_vect.transpose(-1, -2), noise_scm=noise_scm)

    def from_rtf_vect(self, mix: torch.Tensor, rtf_vec: torch.Tensor, noise_scm: torch.Tensor):
        """Compute and apply MVDR beamformer from the ATF vector and noise SCM matrix.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            rtf_vec (torch.ComplexTensor): (batch, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)
        rtf_vec_t = rtf_vec.transpose(-1, -2).unsqueeze(-1)
        numerator = stable_solve(rtf_vec_t, noise_scm_t)
        denominator = torch.matmul(rtf_vec_t.conj().transpose(-1, -2), numerator)
        bf_vect = (numerator / denominator).squeeze(-1).transpose(-1, -2)
        output = self.apply_beamforming_vector(bf_vect, mix=mix)
        return output


class SoudenMVDRBeamformer(Beamformer):

    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor, ref_mic: Union[torch.Tensor, torch.LongTensor, int]=0, eps=1e-08):
        """Compute and apply MVDR beamformer from the speech and noise SCM matrices.
        This class uses Souden's formulation [1].

        :math:`\\mathbf{w} =  \\displaystyle \\frac{\\Sigma_{nn}^{-1} \\Sigma_{ss}}{
        Tr\\left( \\Sigma_{nn}^{-1} \\Sigma_{ss} \\right) }\\mathbf{u}` where :math:`\\mathbf{a}`
        is the steering vector.


        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            ref_mic (int): reference microphone.
            eps: numerical stabilizer.

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)

        References
            [1] Souden, M., Benesty, J., & Affes, S. (2009). On optimal frequency-domain multichannel
            linear filtering for noise reduction. IEEE Transactions on audio, speech, and language processing, 18(2), 260-276.
        """
        noise_scm = noise_scm.permute(0, 3, 1, 2)
        target_scm = target_scm.permute(0, 3, 1, 2)
        numerator = stable_solve(target_scm, noise_scm)
        bf_mat = numerator / (batch_trace(numerator)[..., None, None] + eps)
        batch_mic_vects = self.get_reference_mic_vects(ref_mic, bf_mat, target_scm=target_scm, noise_scm=noise_scm)
        bf_vect = torch.matmul(bf_mat, batch_mic_vects)
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)
        output = self.apply_beamforming_vector(bf_vect, mix=mix)
        return output


class SDWMWFBeamformer(Beamformer):

    def __init__(self, mu=1.0):
        super().__init__()
        self.mu = mu

    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor, ref_mic: Union[torch.Tensor, torch.LongTensor, int]=None):
        """Compute and apply SDW-MWF beamformer.

        :math:`\\mathbf{w} =  \\displaystyle (\\Sigma_{ss} + \\mu \\Sigma_{nn})^{-1} \\Sigma_{ss}`.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            ref_mic (int): reference microphone.

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)
        target_scm_t = target_scm.permute(0, 3, 1, 2)
        denominator = target_scm_t + self.mu * noise_scm_t
        bf_mat = stable_solve(target_scm_t, denominator)
        batch_mic_vects = self.get_reference_mic_vects(ref_mic, bf_mat, target_scm=target_scm_t, noise_scm=noise_scm_t)
        bf_vect = torch.matmul(bf_mat, batch_mic_vects)
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)
        output = self.apply_beamforming_vector(bf_vect, mix=mix)
        return output


def _stable_cholesky(input, upper=False, out=None, eps=1e-06):
    try:
        if upper:
            return torch.linalg.cholesky(input, out=out).mH
        return torch.linalg.cholesky(input, out=out)
    except RuntimeError:
        input = condition_scm(input, eps)
        if upper:
            return torch.linalg.cholesky(input, out=out).mH
        return torch.linalg.cholesky(input, out=out)


def stable_cholesky(input, upper=False, out=None, eps=1e-06):
    """Compute the Cholesky decomposition of ``input``.
    If ``input`` is only p.s.d, add a small jitter to the diagonal.

    Args:
        input (Tensor): The tensor to compute the Cholesky decomposition of
        upper (bool, optional): See torch.cholesky
        out (Tensor, optional): See torch.cholesky
        eps (int): small jitter added to the diagonal if PD.
    """
    input_dtype = input.dtype
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    return _stable_cholesky(input.to(solve_dtype), upper=upper, out=out, eps=eps)


def _generalized_eigenvalue_decomposition(a, b):
    cholesky = stable_cholesky(b)
    inv_cholesky = torch.inverse(cholesky)
    cmat = inv_cholesky @ a @ inv_cholesky.conj().transpose(-1, -2)
    e_val, e_vec = torch.symeig(cmat, eigenvectors=True)
    e_vec = torch.matmul(inv_cholesky.conj().transpose(-1, -2), e_vec)
    return e_val, e_vec


def generalized_eigenvalue_decomposition(a, b):
    """Solves the generalized eigenvalue decomposition through Cholesky decomposition.
    Returns eigen values and eigen vectors (ascending order).
    """
    input_dtype = _common_dtype(a, b)
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    e_val, e_vec = _generalized_eigenvalue_decomposition(a, b)
    return e_val.real, e_vec


class GEVBeamformer(Beamformer):

    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute and apply the GEV beamformer.

        :math:`\\mathbf{w} =  \\displaystyle MaxEig\\{ \\Sigma_{nn}^{-1}\\Sigma_{ss} \\}`, where
        MaxEig extracts the eigenvector corresponding to the maximum eigenvalue
        (using the GEV decomposition).

        Args:
            mix: shape (batch, mics, freqs, frames)
            target_scm: (batch, mics, mics, freqs)
            noise_scm: (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        bf_vect = self.compute_beamforming_vector(target_scm, noise_scm)
        output = self.apply_beamforming_vector(bf_vect, mix=mix)
        return output

    @staticmethod
    def compute_beamforming_vector(target_scm: torch.Tensor, noise_scm: torch.Tensor):
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)
        noise_scm_t = condition_scm(noise_scm_t, 1e-06)
        e_val, e_vec = generalized_eigenvalue_decomposition(target_scm.permute(0, 3, 1, 2), noise_scm_t)
        bf_vect = e_vec[..., -1]
        bf_vect /= torch.norm(bf_vect, dim=-1, keepdim=True)
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)
        return bf_vect


class GEVDBeamformer(Beamformer):
    """Generalized eigenvalue decomposition speech distortion weighted multichannel Wiener filter.

        Compare to SDW-MWF, spatial covariance matrix are computed from low rank approximation
        based on eigen values decomposition,
        see equation 62 in `[1] <https://hal.inria.fr/hal-01390918/file/14-1.pdf>`_.

    Attributes:
        mu (float): Speech distortion constant.
        rank (int): Rank for the approximation of target covariance matrix,
            no approximation is made if `rank` is None.

    References:
        [1] R. Serizel, M. Moonen, B. Van Dijk and J. Wouters,
        "Low-rank Approximation Based Multichannel Wiener Filter Algorithms for
        Noise Reduction with Application in Cochlear Implants,"
        in IEEE/ACM Transactions on Audio, Speech, and Language Processing, April 2014.
    """

    def __init__(self, mu: float=1.0, rank: int=1):
        self.mu = mu
        self.rank = rank

    def compute_beamforming_vector(self, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute beamforming vectors for GEVD beamFormer.

        Args:
            target_scm (torch.ComplexTensor): shape (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): shape (batch, mics, mics, freqs)

        Returns:
            torch.ComplexTensor: shape (batch, mics, freqs)

        """
        e_values, e_vectors = _generalized_eigenvalue_decomposition(target_scm.permute(0, 3, 1, 2), noise_scm.permute(0, 3, 1, 2))
        eps = torch.finfo(e_values.dtype).eps
        e_values = torch.clamp(e_values, min=eps, max=1000000.0)
        e_values = torch.diag_embed(torch.flip(e_values, [-1]))
        e_vectors = torch.flip(e_vectors, [-1])
        if self.rank:
            e_values[..., self.rank:, :] = 0.0
        complex_type = e_vectors.dtype
        ev_plus_mu = e_values + self.mu * torch.eye(e_values.shape[-1]).expand_as(e_values)
        bf_vect = e_vectors @ e_values @ torch.linalg.inv(e_vectors @ ev_plus_mu)
        return bf_vect[..., 0].permute(0, 2, 1)

    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute and apply the GEVD beamformer.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        bf_vect = self.compute_beamforming_vector(target_scm, noise_scm)
        return self.apply_beamforming_vector(bf_vect, mix=mix)


class PITLossWrapper(nn.Module):
    """Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        pit_from (str): Determines how PIT is applied.

            * ``'pw_mtx'`` (pairwise matrix): `loss_func` computes pairwise
              losses and returns a torch.Tensor of shape
              :math:`(batch, n\\_src, n\\_src)`. Each element
              :math:`(batch, i, j)` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\\_targets[:, j]`
            * ``'pw_pt'`` (pairwise point): `loss_func` computes the loss for
              a batch of single source and single estimates (tensors won't
              have the source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'perm_avg'`` (permutation average): `loss_func` computes the
              average loss for a given permutations of the sources and
              estimates. Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_perm_avg_loss`.

            In terms of efficiency, ``'perm_avg'`` is the least efficicient.

        perm_reduce (Callable): torch function to reduce permutation losses.
            Defaults to None (equivalent to mean). Signature of the func
            (pwl_set, **kwargs) : :math:`(B, n\\_src!, n\\_src) --> (B, n\\_src!)`.
            `perm_reduce` can receive **kwargs during forward using the
            `reduce_kwargs` argument (dict). If those argument are static,
            consider defining a small function or using `functools.partial`.
            Only used in `'pw_mtx'` and `'pw_pt'` `pit_from` modes.

    For each of these modes, the best permutation and reordering will be
    automatically computed. When either ``'pw_mtx'`` or ``'pw_pt'`` is used,
    and the number of sources is larger than three, the hungarian algorithm is
    used to find the best permutation.

    Examples
        >>> import torch
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute PIT loss based on pairwise losses
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        >>> loss_val = loss_func(est_sources, sources)
        >>>
        >>> # Using reduce
        >>> def reduce(perm_loss, src):
        >>>     weighted = perm_loss * src.norm(dim=-1, keepdim=True)
        >>>     return torch.mean(weighted, dim=-1)
        >>>
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx',
        >>>                            perm_reduce=reduce)
        >>> reduce_kwargs = {'src': sources}
        >>> loss_val = loss_func(est_sources, sources,
        >>>                      reduce_kwargs=reduce_kwargs)
    """

    def __init__(self, loss_func, pit_from='pw_mtx', perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        if self.pit_from not in ['pw_mtx', 'pw_pt', 'perm_avg']:
            raise ValueError('Unsupported loss function type for now. Expectedone of [`pw_mtx`, `pw_pt`, `perm_avg`]')

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        """Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
              the batch.
            - The reordered targets estimates if ``return_est`` is True.
              :class:`torch.Tensor` of shape $(batch, nsrc, ...)$.
        """
        n_src = targets.shape[1]
        assert n_src < 10, f'Expected source axis along dim 1, found {n_src}'
        if self.pit_from == 'pw_mtx':
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == 'pw_pt':
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == 'perm_avg':
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(self.loss_func, est_targets, targets, **kwargs)
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered
        else:
            return
        assert pw_losses.ndim == 3, 'Something went wrong with the loss function, please read the docs.'
        assert pw_losses.shape[0] == targets.shape[0], 'PIT loss needs same batch dim as input'
        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs)
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        """Get pair-wise losses between the training targets and its estimate
        for a given loss function.

        Args:
            loss_func: function with signature (est_targets, targets, **kwargs)
                The loss function to get pair-wise losses from.
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            torch.Tensor or size $(batch, nsrc, nsrc)$, losses computed for
            all permutations of the targets and est_targets.

        This function can be called on a loss function which returns a tensor
        of size :math:`(batch)`. There are more efficient ways to compute pair-wise
        losses using broadcasting.
        """
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_perm_avg_loss(loss_func, est_targets, targets, **kwargs):
        """Find best permutation from loss function with source axis.

        Args:
            loss_func: function with signature $(est_targets, targets, **kwargs)$
                The loss function batch losses from.
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, *)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, *)$.
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
                The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
                The indices of the best permutations.
        """
        n_src = targets.shape[1]
        perms = torch.tensor(list(permutations(range(n_src))), dtype=torch.long)
        loss_set = torch.stack([loss_func(est_targets[:, perm], targets, **kwargs) for perm in perms], dim=1)
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    @staticmethod
    def find_best_perm(pair_wise_losses, perm_reduce=None, **kwargs):
        """Find the best permutation, given the pair-wise losses.

        Dispatch between factorial method if number of sources is small (<3)
        and hungarian method for more sources. If ``perm_reduce`` is not None,
        the factorial method is always used.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n\\_src, n\\_src)`. Pairwise losses.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : :math:`(B, n\\_src!, n\\_src) -> (B, n\\_src!)`
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
              The indices of the best permutations.
        """
        n_src = pair_wise_losses.shape[-1]
        if perm_reduce is not None or n_src <= 3:
            min_loss, batch_indices = PITLossWrapper.find_best_perm_factorial(pair_wise_losses, perm_reduce=perm_reduce, **kwargs)
        else:
            min_loss, batch_indices = PITLossWrapper.find_best_perm_hungarian(pair_wise_losses)
        return min_loss, batch_indices

    @staticmethod
    def reorder_source(source, batch_indices):
        """Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape :math:`(batch, n_src, time)`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.

        Returns:
            :class:`torch.Tensor`: Reordered sources.
        """
        reordered_sources = torch.stack([torch.index_select(s, 0, b) for s, b in zip(source, batch_indices)])
        return reordered_sources

    @staticmethod
    def find_best_perm_factorial(pair_wise_losses, perm_reduce=None, **kwargs):
        """Find the best permutation given the pair-wise losses by looping
        through all the permutations.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n_src, n_src)`. Pairwise losses.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : :math:`(B, n\\_src!, n\\_src) -> (B, n\\_src!)`
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
              The indices of the best permutations.

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        n_src = pair_wise_losses.shape[-1]
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        idx = torch.unsqueeze(perms, 2)
        if perm_reduce is None:
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            loss_set = perm_reduce(pwl_set, **kwargs)
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    @staticmethod
    def find_best_perm_hungarian(pair_wise_losses: torch.Tensor):
        """
        Find the best permutation given the pair-wise losses, using the Hungarian algorithm.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.Tensor`:
              The indices of the best permutations.
        """
        pwl = pair_wise_losses.transpose(-1, -2)
        pwl_copy = pwl.detach().cpu()
        batch_indices = torch.tensor([linear_sum_assignment(pwl)[1] for pwl in pwl_copy])
        min_loss = torch.gather(pwl, 2, batch_indices[..., None]).mean([-1, -2])
        return min_loss, batch_indices


class PITReorder(PITLossWrapper):
    """Permutation invariant reorderer. Only returns the reordered estimates.
    See `:py:class:asteroid.losses.PITLossWrapper`."""

    def forward(self, est_targets, targets, reduce_kwargs=None, **kwargs):
        _, reordered = super().forward(est_targets=est_targets, targets=targets, return_est=True, reduce_kwargs=reduce_kwargs, **kwargs)
        return reordered


def _reorder_sources(current: torch.FloatTensor, previous: torch.FloatTensor, n_src: int, window_size: int, hop_size: int):
    """
     Reorder sources in current chunk to maximize correlation with previous chunk.
     Used for Continuous Source Separation. Standard dsp correlation is used
     for reordering.


    Args:
        current (:class:`torch.Tensor`): current chunk, tensor
                                        of shape (batch, n_src, window_size)
        previous (:class:`torch.Tensor`): previous chunk, tensor
                                        of shape (batch, n_src, window_size)
        n_src (:class:`int`): number of sources.
        window_size (:class:`int`): window_size, equal to last dimension of
                                    both current and previous.
        hop_size (:class:`int`): hop_size between current and previous tensors.

    """
    batch, frames = current.size()
    current = current.reshape(-1, n_src, frames)
    previous = previous.reshape(-1, n_src, frames)
    overlap_f = window_size - hop_size

    def reorder_func(x, y):
        x = x[..., :overlap_f]
        y = y[..., -overlap_f:]
        x = x - x.mean(-1, keepdim=True)
        y = y - y.mean(-1, keepdim=True)
        return -torch.sum(x.unsqueeze(1) * y.unsqueeze(2), dim=-1)
    pit = PITReorder(reorder_func)
    current = pit(current, previous)
    return current.reshape(batch, frames)


class LambdaOverlapAdd(torch.nn.Module):
    """Overlap-add with lambda transform on segments (not scriptable).

    Segment input signal, apply lambda function (a neural network for example)
    and combine with OLA.

    `LambdaOverlapAdd` can be used with :mod:`asteroid.separate` and the
    `asteroid-infer` CLI.

    Args:
        nnet (callable): Function to apply to each segment.
        n_src (Optional[int]): Number of sources in the output of nnet.
            If None, the number of sources is determined by the network's output,
            but some correctness checks cannot be performed.
        window_size (int): Size of segmenting window.
        hop_size (int): Segmentation hop size.
        window (str): Name of the window (see scipy.signal.get_window) used
            for the synthesis.
        reorder_chunks (bool): Whether to reorder each consecutive segment.
            This might be useful when `nnet` is permutation invariant, as
            source assignements might change output channel from one segment
            to the next (in classic speech separation for example).
            Reordering is performed based on the correlation between
            the overlapped part of consecutive segment.

     Examples
        >>> from asteroid import ConvTasNet
        >>> nnet = ConvTasNet(n_src=2)
        >>> continuous_nnet = LambdaOverlapAdd(
        >>>     nnet=nnet,
        >>>     n_src=2,
        >>>     window_size=64000,
        >>>     hop_size=None,
        >>>     window="hanning",
        >>>     reorder_chunks=True,
        >>>     enable_grad=False,
        >>> )

        >>> # Process wav tensor:
        >>> wav = torch.randn(1, 1, 500000)
        >>> out_wavs = continuous_nnet.forward(wav)
        >>> # asteroid.separate.Separatable support:
        >>> from asteroid.separate import file_separate
        >>> file_separate(continuous_nnet, "example.wav")
    """

    def __init__(self, nnet, n_src, window_size, hop_size=None, window='hanning', reorder_chunks=True, enable_grad=False):
        super().__init__()
        assert window_size % 2 == 0, 'Window size must be even'
        self.nnet = nnet
        self.window_size = window_size
        self.hop_size = hop_size if hop_size is not None else window_size // 2
        self.n_src = n_src
        self.in_channels = getattr(nnet, 'in_channels', None)
        if window:
            from scipy.signal import get_window
            window = get_window(window, self.window_size).astype('float32')
            window = torch.from_numpy(window)
            self.use_window = True
        else:
            self.use_window = False
        self.register_buffer('window', window)
        self.reorder_chunks = reorder_chunks
        self.enable_grad = enable_grad

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""
        assert x.ndim == 3
        batch, channels, n_frames = x.size()
        unfolded = torch.nn.functional.unfold(x.unsqueeze(-1), kernel_size=(self.window_size, 1), padding=(self.window_size, 0), stride=(self.hop_size, 1))
        out = []
        n_chunks = unfolded.shape[-1]
        for frame_idx in range(n_chunks):
            frame = self.nnet(unfolded[..., frame_idx])
            if frame_idx == 0:
                assert frame.ndim == 3, 'nnet should return (batch, n_src, time)'
                if self.n_src is not None:
                    assert frame.shape[1] == self.n_src, 'nnet should return (batch, n_src, time)'
                n_src = frame.shape[1]
            frame = frame.reshape(batch * n_src, -1)
            if frame_idx != 0 and self.reorder_chunks:
                frame = _reorder_sources(frame, out[-1], n_src, self.window_size, self.hop_size)
            if self.use_window:
                frame = frame * self.window
            else:
                frame = frame / (self.window_size / self.hop_size)
            out.append(frame)
        out = torch.stack(out).reshape(n_chunks, batch * n_src, self.window_size)
        out = out.permute(1, 2, 0)
        out = torch.nn.functional.fold(out, (n_frames, 1), kernel_size=(self.window_size, 1), padding=(self.window_size, 0), stride=(self.hop_size, 1))
        return out.squeeze(-1).reshape(batch, n_src, -1)

    def forward(self, x):
        """Forward module: segment signal, apply func, combine with OLA.

        Args:
            x (:class:`torch.Tensor`): waveform signal of shape (batch, 1, time).

        Returns:
            :class:`torch.Tensor`: The output of the lambda OLA.
        """
        with torch.autograd.set_grad_enabled(self.enable_grad):
            olad = self.ola_forward(x)
            return olad

    @property
    def sample_rate(self):
        return self.nnet.sample_rate

    def _separate(self, wav, *args, **kwargs):
        return self.forward(wav, *args, **kwargs)


class DualPathProcessing(nn.Module):
    """
    Perform Dual-Path processing via overlap-add as in DPRNN [1].

    Args:
        chunk_size (int): Size of segmenting window.
        hop_size (int): segmentation hop size.

    References
        [1] Yi Luo, Zhuo Chen and Takuya Yoshioka. "Dual-path RNN: efficient
        long sequence modeling for time-domain single-channel speech separation"
        https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size, hop_size):
        super(DualPathProcessing, self).__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        """
        Unfold the feature tensor from $(batch, channels, time)$ to
        $(batch, channels, chunksize, nchunks)$.

        Args:
            x (:class:`torch.Tensor`): feature tensor of shape $(batch, channels, time)$.

        Returns:
            :class:`torch.Tensor`: spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        """
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(x.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        return unfolded.reshape(batch, chan, self.chunk_size, -1)

    def fold(self, x, output_size=None):
        """
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.

        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.

        .. note:: `fold` caches the original length of the input.

        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(to_unfold, (output_size, 1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        x /= float(self.chunk_size) / self.hop_size
        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x, module):
        """Performs intra-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.

        Returns:
            :class:`torch.Tensor`: processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """
        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, channels).transpose(1, -1)
        x = module(x)
        x = x.reshape(batch, n_chunks, channels, chunk_size).transpose(1, -1).transpose(1, 2)
        return x

    @staticmethod
    def inter_process(x, module):
        """Performs inter-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.


        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """
        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x


class MixITLossWrapper(nn.Module):
    """Mixture invariant loss wrapper.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        generalized (bool): Determines how MixIT is applied. If False ,
            apply MixIT for any number of mixtures as soon as they contain
            the same number of sources (:meth:`~MixITLossWrapper.best_part_mixit`.)
            If True (default), apply MixIT for two mixtures, but those mixtures do not
            necessarly have to contain the same number of sources.
            See :meth:`~MixITLossWrapper.best_part_mixit_generalized`.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    For each of these modes, the best partition and reordering will be
    automatically computed.

    Examples:
        >>> import torch
        >>> from asteroid.losses import multisrc_mse
        >>> mixtures = torch.randn(10, 2, 16000)
        >>> est_sources = torch.randn(10, 4, 16000)
        >>> # Compute MixIT loss based on pairwise losses
        >>> loss_func = MixITLossWrapper(multisrc_mse)
        >>> loss_val = loss_func(est_sources, mixtures)

    References
        [1] Scott Wisdom et al. "Unsupervised sound separation using
        mixtures of mixtures." arXiv:2006.12701 (2020)
    """

    def __init__(self, loss_func, generalized=True, reduction='mean'):
        super().__init__()
        self.loss_func = loss_func
        self.generalized = generalized
        self.reduction = reduction

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """Find the best partition and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, *)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets
            return_est: Boolean. Whether to return the estimated mixtures
                estimates (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best partition loss for each batch sample, average over
              the batch. torch.Tensor(loss_value)
            - The estimated mixtures (estimated sources summed according to the partition)
              if return_est is True. torch.Tensor of shape :math:`(batch, nmix, ...)`.
        """
        assert est_targets.shape[0] == targets.shape[0]
        assert est_targets.shape[2] == targets.shape[2]
        if not self.generalized:
            min_loss, min_loss_idx, parts = self.best_part_mixit(self.loss_func, est_targets, targets, **kwargs)
        else:
            min_loss, min_loss_idx, parts = self.best_part_mixit_generalized(self.loss_func, est_targets, targets, **kwargs)
        returned_loss = min_loss.mean() if self.reduction == 'mean' else min_loss
        if not return_est:
            return returned_loss
        reordered = self.reorder_source(est_targets, targets, min_loss_idx, parts)
        return returned_loss, reordered

    @staticmethod
    def best_part_mixit(loss_func, est_targets, targets, **kwargs):
        """Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid for any number of
        mixtures as soon as they contain the same number of sources.

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indices of the best partition.

            - :class:`list`:
              list of the possible partitions of the sources.

        """
        nmix = targets.shape[1]
        nsrc = est_targets.shape[1]
        if nsrc % nmix != 0:
            raise ValueError('The mixtures are assumed to contain the same number of sources')
        nsrcmix = nsrc // nmix

        def parts_mixit(lst, k, l):
            if l == 0:
                yield []
            else:
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    for r in parts_mixit(rest, k, l - 1):
                        yield [list(c), *r]
        parts = list(parts_mixit(range(nsrc), nsrcmix, nmix))
        loss_set = MixITLossWrapper.loss_set_from_parts(loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs)
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def best_part_mixit_generalized(loss_func, est_targets, targets, **kwargs):
        """Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid only for two mixtures,
        but those mixtures do not necessarly have to contain the same number of
        sources e.g the case where one mixture is silent is allowed..

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indexes of the best permutations.

            - :class:`list`:
              list of the possible partitions of the sources.
        """
        nmix = targets.shape[1]
        nsrc = est_targets.shape[1]
        if nmix != 2:
            raise ValueError('Works only with two mixtures')

        def parts_mixit_gen(lst):
            partitions = []
            for k in range(len(lst) + 1):
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    partitions.append([list(c), rest])
            return partitions
        parts = parts_mixit_gen(range(nsrc))
        loss_set = MixITLossWrapper.loss_set_from_parts(loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs)
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def loss_set_from_parts(loss_func, est_targets, targets, parts, **kwargs):
        """Common loop between both best_part_mixit"""
        loss_set = []
        for partition in parts:
            est_mixes = torch.stack([est_targets[:, idx, :].sum(1) for idx in partition], dim=1)
            loss_partition = loss_func(est_mixes, targets, **kwargs)
            if loss_partition.ndim != 1:
                raise ValueError('Loss function return value should be of size (batch,).')
            loss_set.append(loss_partition[:, None])
        loss_set = torch.cat(loss_set, dim=1)
        return loss_set

    @staticmethod
    def reorder_source(est_targets, targets, min_loss_idx, parts):
        """Reorder sources according to the best partition.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets.
            min_loss_idx: torch.LongTensor. The indexes of the best permutations.
            parts: list of the possible partitions of the sources.

        Returns:
            :class:`torch.Tensor`: Reordered sources of shape :math:`(batch, nmix, time)`.

        """
        ordered = torch.zeros_like(targets)
        for b, idx in enumerate(min_loss_idx):
            right_partition = parts[idx]
            ordered[b, :, :] = torch.stack([est_targets[b, idx, :][None, :, :].sum(1) for idx in right_partition], dim=1)
        return ordered


class PairwiseMSE(_Loss):
    """Measure pairwise mean square error on a batch.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseMSE(), pit_from='pairwise')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 3:
            raise TypeError(f'Inputs must be of shape [batch, n_src, *], got {targets.size()} and {est_targets.size()} instead')
        targets = targets.unsqueeze(1)
        est_targets = est_targets.unsqueeze(2)
        pw_loss = (targets - est_targets) ** 2
        mean_over = list(range(3, pw_loss.ndim))
        return pw_loss.mean(dim=mean_over)


class SingleSrcMSE(_Loss):
    """Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        - est_targets: :math:`(batch, ...)`.
        - targets: :math:`(batch, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)`

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # singlesrc_mse / multisrc_mse support both 'pw_pt' and 'perm_avg'.
        >>> loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(f'Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead')
        loss = (targets - est_targets) ** 2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)


class SingleSrcMultiScaleSpectral(_Loss):
    """Measure multi-scale spectral loss as described in [1]

    Args:
        n_filters (list): list containing the number of filter desired for
            each STFT
        windows_size (list): list containing the size of the window desired for
            each STFT
        hops_size (list): list containing the size of the hop desired for
            each STFT

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples
        >>> import torch
        >>> targets = torch.randn(10, 32000)
        >>> est_targets = torch.randn(10, 32000)
        >>> # Using it by itself on a pair of source/estimate
        >>> loss_func = SingleSrcMultiScaleSpectral()
        >>> loss = loss_func(est_targets, targets)

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # Using it with PITLossWrapper with sets of source/estimates
        >>> loss_func = PITLossWrapper(SingleSrcMultiScaleSpectral(),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and
        Adam Roberts "DDSP: Differentiable Digital Signal Processing" ICLR 2020.
    """

    def __init__(self, n_filters=None, windows_size=None, hops_size=None, alpha=1.0):
        super().__init__()
        if windows_size is None:
            windows_size = [2048, 1024, 512, 256, 128, 64, 32]
        if n_filters is None:
            n_filters = [2048, 1024, 512, 256, 128, 64, 32]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128, 64, 32, 16]
        self.windows_size = windows_size
        self.n_filters = n_filters
        self.hops_size = hops_size
        self.alpha = alpha
        self.encoders = nn.ModuleList(Encoder(STFTFB(n_filters[i], windows_size[i], hops_size[i])) for i in range(len(self.n_filters)))

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)
        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)
        return loss

    def compute_spectral_loss(self, encoder, est_target, target, EPS=1e-08):
        batch_size = est_target.shape[0]
        spect_est_target = mag(encoder(est_target)).view(batch_size, -1)
        spect_target = mag(encoder(target)).view(batch_size, -1)
        linear_loss = self.norm1(spect_est_target - spect_target)
        log_loss = self.norm1(torch.log(spect_est_target + EPS) - torch.log(spect_target + EPS))
        return linear_loss + self.alpha * log_loss

    @staticmethod
    def norm1(a):
        return torch.norm(a, p=1, dim=1)


class SingleSrcPMSQE(nn.Module):
    """Computes the Perceptual Metric for Speech Quality Evaluation (PMSQE)
    as described in [1].
    This version is only designed for 16 kHz (512 length DFT).
    Adaptation to 8 kHz could be done by changing the parameters of the
    class (see Tensorflow implementation).
    The SLL, frequency and gain equalization are applied in each
    sequence independently.

    Parameters:
        window_name (str): Select the used window function for the correct
            factor to be applied. Defaults to sqrt hanning window.
            Among ['rect', 'hann', 'sqrt_hann', 'hamming', 'flatTop'].
        window_weight (float, optional): Correction to the window factor
            applied.
        bark_eq (bool, optional): Whether to apply bark equalization.
        gain_eq (bool, optional): Whether to apply gain equalization.
        sample_rate (int): Sample rate of the input audio.

    References
        [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning
        Loss Function based on the Perceptual Evaluation of the
        Speech Quality', IEEE Signal Processing Letters, 2018.
        Implemented by Juan M. Martin. Contact: mdjuamart@ugr.es

        Copyright 2019: University of Granada, Signal Processing, Multimedia
        Transmission and Speech/Audio Technologies (SigMAT) Group.

    .. note:: Inspired on the Perceptual Evaluation of the Speech Quality (PESQ)
        algorithm, this function consists of two regularization factors :
        the symmetrical and asymmetrical distortion in the loudness domain.

    Examples
        >>> import torch
        >>> from asteroid_filterbanks import STFTFB, Encoder, transforms
        >>> from asteroid.losses import PITLossWrapper, SingleSrcPMSQE
        >>> stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
        >>> # Usage by itself
        >>> ref, est = torch.randn(2, 1, 16000), torch.randn(2, 1, 16000)
        >>> ref_spec = transforms.mag(stft(ref))
        >>> est_spec = transforms.mag(stft(est))
        >>> loss_func = SingleSrcPMSQE()
        >>> loss_value = loss_func(est_spec, ref_spec)
        >>> # Usage with PITLossWrapper
        >>> loss_func = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_pt')
        >>> ref, est = torch.randn(2, 3, 16000), torch.randn(2, 3, 16000)
        >>> ref_spec = transforms.mag(stft(ref))
        >>> est_spec = transforms.mag(stft(est))
        >>> loss_value = loss_func(ref_spec, est_spec)
    """

    def __init__(self, window_name='sqrt_hann', window_weight=1.0, bark_eq=True, gain_eq=True, sample_rate=16000):
        super().__init__()
        self.window_name = window_name
        self.window_weight = window_weight
        self.bark_eq = bark_eq
        self.gain_eq = gain_eq
        if sample_rate not in [16000, 8000]:
            raise ValueError('Unsupported sample rate {}'.format(sample_rate))
        self.sample_rate = sample_rate
        if sample_rate == 16000:
            self.Sp = 6.910853e-06
            self.Sl = 0.1866055
            self.nbins = 512
            self.nbark = 49
        else:
            self.Sp = 2.764344e-05
            self.Sl = 0.1866055
            self.nbins = 256
            self.nbark = 42
        self.alpha = 0.1
        self.beta = 0.309 * self.alpha
        pow_correc_factor = self.get_correction_factor(window_name)
        self.pow_correc_factor = pow_correc_factor * self.window_weight
        self.abs_thresh_power = None
        self.modified_zwicker_power = None
        self.width_of_band_bark = None
        self.bark_matrix = None
        self.mask_sll = None
        self.populate_constants(self.sample_rate)
        self.sqrt_total_width = torch.sqrt(torch.sum(self.width_of_band_bark))
        self.EPS = 1e-08

    def forward(self, est_targets, targets, pad_mask=None):
        """
        Args
            est_targets (torch.Tensor): Dimensions (B, T, F).
                Padded degraded power spectrum in time-frequency domain.
            targets (torch.Tensor): Dimensions (B, T, F).
                Zero-Padded reference power spectrum in time-frequency domain.
            pad_mask (torch.Tensor, optional):  Dimensions (B, T, 1). Mask
                to indicate the padding frames. Defaults to all ones.

        Dimensions
            B: Number of sequences in the batch.
            T: Number of time frames.
            F: Number of frequency bins.

        Returns
            torch.tensor of shape (B, ), wD + 0.309 * wDA

        ..note:: Dimensions (B, F, T) are also supported by SingleSrcPMSQE but are
            less efficient because input tensors are transposed (not inplace).

        """
        assert est_targets.shape == targets.shape
        try:
            freq_idx = est_targets.shape.index(self.nbins // 2 + 1)
        except ValueError:
            raise ValueError('Could not find dimension with {} elements in input tensors, verify your inputs'.format(self.nbins // 2 + 1))
        if freq_idx == 1:
            est_targets = est_targets.transpose(1, 2)
            targets = targets.transpose(1, 2)
        if pad_mask is not None:
            pad_mask = pad_mask.transpose(1, 2) if freq_idx == 1 else pad_mask
        else:
            pad_mask = torch.ones(est_targets.shape[0], est_targets.shape[1], 1, device=est_targets.device)
        ref_spectra = self.magnitude_at_sll(targets, pad_mask)
        deg_spectra = self.magnitude_at_sll(est_targets, pad_mask)
        ref_bark_spectra = self.bark_computation(ref_spectra)
        deg_bark_spectra = self.bark_computation(deg_spectra)
        if self.bark_eq:
            deg_bark_spectra = self.bark_freq_equalization(ref_bark_spectra, deg_bark_spectra)
        if self.gain_eq:
            deg_bark_spectra = self.bark_gain_equalization(ref_bark_spectra, deg_bark_spectra)
        sym_d, asym_d = self.compute_distortion_tensors(ref_bark_spectra, deg_bark_spectra)
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        wd_frame, wda_frame = self.per_frame_distortion(sym_d, asym_d, audible_power_ref)
        dims = [-1, -2]
        pmsqe_frame = (self.alpha * wd_frame + self.beta * wda_frame) * pad_mask
        pmsqe = torch.sum(pmsqe_frame, dim=dims) / pad_mask.sum(dims)
        return pmsqe

    def magnitude_at_sll(self, spectra, pad_mask):
        masked_spectra = spectra * pad_mask * self.mask_sll
        freq_mean_masked_spectra = torch.mean(masked_spectra, dim=-1, keepdim=True)
        sum_spectra = torch.sum(freq_mean_masked_spectra, dim=-2, keepdim=True)
        seq_len = torch.sum(pad_mask, dim=-2, keepdim=True)
        mean_pow = sum_spectra / seq_len
        return 10000000.0 * spectra / mean_pow

    def bark_computation(self, spectra):
        return self.Sp * torch.matmul(spectra, self.bark_matrix)

    def compute_audible_power(self, bark_spectra, factor=1.0):
        thr_bark = torch.where(bark_spectra > self.abs_thresh_power * factor, bark_spectra, torch.zeros_like(bark_spectra))
        return torch.sum(thr_bark, dim=-1, keepdim=True)

    def bark_gain_equalization(self, ref_bark_spectra, deg_bark_spectra):
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        audible_power_deg = self.compute_audible_power(deg_bark_spectra, 1.0)
        gain = (audible_power_ref + 5000.0) / (audible_power_deg + 5000.0)
        limited_gain = torch.min(gain, 5.0 * torch.ones_like(gain))
        limited_gain = torch.max(limited_gain, 0.0003 * torch.ones_like(limited_gain))
        return limited_gain * deg_bark_spectra

    def bark_freq_equalization(self, ref_bark_spectra, deg_bark_spectra):
        """This version is applied in the degraded directly."""
        audible_power_x100 = self.compute_audible_power(ref_bark_spectra, 100.0)
        not_silent = audible_power_x100 >= 10000000.0
        cond_thr = ref_bark_spectra >= self.abs_thresh_power * 100.0
        ref_thresholded = torch.where(cond_thr, ref_bark_spectra, torch.zeros_like(ref_bark_spectra))
        deg_thresholded = torch.where(cond_thr, deg_bark_spectra, torch.zeros_like(deg_bark_spectra))
        avg_ppb_ref = torch.sum(torch.where(not_silent, ref_thresholded, torch.zeros_like(ref_thresholded)), dim=-2, keepdim=True)
        avg_ppb_deg = torch.sum(torch.where(not_silent, deg_thresholded, torch.zeros_like(deg_thresholded)), dim=-2, keepdim=True)
        equalizer = (avg_ppb_ref + 1000.0) / (avg_ppb_deg + 1000.0)
        equalizer = torch.min(equalizer, 100.0 * torch.ones_like(equalizer))
        equalizer = torch.max(equalizer, 0.01 * torch.ones_like(equalizer))
        return equalizer * deg_bark_spectra

    def loudness_computation(self, bark_spectra):
        aterm = torch.pow(self.abs_thresh_power / 0.5, self.modified_zwicker_power)
        bterm = torch.pow(0.5 + 0.5 * bark_spectra / self.abs_thresh_power, self.modified_zwicker_power) - 1.0
        loudness_dens = self.Sl * aterm * bterm
        cond = bark_spectra < self.abs_thresh_power
        return torch.where(cond, torch.zeros_like(loudness_dens), loudness_dens)

    def compute_distortion_tensors(self, ref_bark_spec, deg_bark_spec):
        original_loudness = self.loudness_computation(ref_bark_spec)
        distorted_loudness = self.loudness_computation(deg_bark_spec)
        r = torch.abs(distorted_loudness - original_loudness)
        m = 0.25 * torch.min(original_loudness, distorted_loudness)
        sym_d = torch.max(r - m, torch.ones_like(r) * self.EPS)
        asym = torch.pow((deg_bark_spec + 50.0) / (ref_bark_spec + 50.0), 1.2)
        cond = asym < 3.0 * torch.ones_like(asym)
        asym_factor = torch.where(cond, torch.zeros_like(asym), torch.min(asym, 12.0 * torch.ones_like(asym)))
        asym_d = asym_factor * sym_d
        return sym_d, asym_d

    def per_frame_distortion(self, sym_d, asym_d, total_power_ref):
        d_frame = torch.sum(torch.pow(sym_d * self.width_of_band_bark, 2.0) + self.EPS, dim=-1, keepdim=True)
        d_frame = torch.sqrt(d_frame) * self.sqrt_total_width
        da_frame = torch.sum(asym_d * self.width_of_band_bark, dim=-1, keepdim=True)
        weights = torch.pow((total_power_ref + 100000.0) / 10000000.0, 0.04)
        wd_frame = torch.min(d_frame / weights, 45.0 * torch.ones_like(d_frame))
        wda_frame = torch.min(da_frame / weights, 45.0 * torch.ones_like(da_frame))
        return wd_frame, wda_frame

    @staticmethod
    def get_correction_factor(window_name):
        """Returns the power correction factor depending on the window."""
        if window_name == 'rect':
            return 1.0
        elif window_name == 'hann':
            return 2.666666666666754
        elif window_name == 'sqrt_hann':
            return 2.0
        elif window_name == 'hamming':
            return 2.51635879188799
        elif window_name == 'flatTop':
            return 5.70713295690759
        else:
            raise ValueError('Unexpected window type {}'.format(window_name))

    def populate_constants(self, sample_rate):
        if sample_rate == 8000:
            self.register_8k_constants()
        elif sample_rate == 16000:
            self.register_16k_constants()
        mask_sll = np.zeros(shape=[self.nbins // 2 + 1], dtype=np.float32)
        mask_sll[11] = 0.5 * 25.0 / 31.25
        mask_sll[12:104] = 1.0
        mask_sll[104] = 0.5
        correction = self.pow_correc_factor * (self.nbins + 2.0) / self.nbins ** 2
        mask_sll = mask_sll * correction
        self.mask_sll = nn.Parameter(tensor(mask_sll), requires_grad=False)

    def register_16k_constants(self):
        abs_thresh_power = [51286152.0, 2454709.5, 70794.59375, 4897.788574, 1174.897705, 389.045166, 104.71286, 45.70882, 17.782795, 9.772372, 4.897789, 3.090296, 1.905461, 1.258925, 0.977237, 0.724436, 0.562341, 0.457088, 0.389045, 0.331131, 0.295121, 0.269153, 0.25704, 0.251189, 0.251189, 0.251189, 0.251189, 0.263027, 0.288403, 0.30903, 0.338844, 0.371535, 0.398107, 0.436516, 0.467735, 0.489779, 0.501187, 0.501187, 0.512861, 0.524807, 0.524807, 0.524807, 0.512861, 0.47863, 0.42658, 0.371535, 0.363078, 0.416869, 0.537032]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        modif_zwicker_power = [0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25168783742879913, 0.2480666573186961, 0.244767379124259, 0.24173800119368227, 0.23893798876066405, 0.23633516221479894, 0.23390360348392067, 0.23162209128929445, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        width_of_band_bark = [0.157344, 0.317994, 0.322441, 0.326934, 0.331474, 0.336061, 0.340697, 0.345381, 0.350114, 0.354897, 0.359729, 0.364611, 0.369544, 0.374529, 0.379565, 0.384653, 0.389794, 0.394989, 0.400236, 0.405538, 0.410894, 0.416306, 0.421773, 0.427297, 0.432877, 0.438514, 0.444209, 0.449962, 0.455774, 0.461645, 0.467577, 0.473569, 0.479621, 0.485736, 0.491912, 0.498151, 0.504454, 0.510819, 0.51725, 0.523745, 0.530308, 0.536934, 0.543629, 0.55039, 0.55722, 0.564119, 0.571085, 0.578125, 0.585232]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, 'bark_matrix_16k.mat')
        bark_matrix = self.load_mat(bark_path)['Bark_matrix_16k'].astype('float32')
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)

    def register_8k_constants(self):
        abs_thresh_power = [51286152, 2454709.5, 70794.59375, 4897.788574, 1174.897705, 389.045166, 104.71286, 45.70882, 17.782795, 9.772372, 4.897789, 3.090296, 1.905461, 1.258925, 0.977237, 0.724436, 0.562341, 0.457088, 0.389045, 0.331131, 0.295121, 0.269153, 0.25704, 0.251189, 0.251189, 0.251189, 0.251189, 0.263027, 0.288403, 0.30903, 0.338844, 0.371535, 0.398107, 0.436516, 0.467735, 0.489779, 0.501187, 0.501187, 0.512861, 0.524807, 0.524807, 0.524807]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        modif_zwicker_power = [0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25168783742879913, 0.2480666573186961, 0.244767379124259, 0.24173800119368227, 0.23893798876066405, 0.23633516221479894, 0.23390360348392067, 0.23162209128929445, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        width_of_band_bark = [0.157344, 0.317994, 0.322441, 0.326934, 0.331474, 0.336061, 0.340697, 0.345381, 0.350114, 0.354897, 0.359729, 0.364611, 0.369544, 0.374529, 0.379565, 0.384653, 0.389794, 0.394989, 0.400236, 0.405538, 0.410894, 0.416306, 0.421773, 0.427297, 0.432877, 0.438514, 0.444209, 0.449962, 0.455774, 0.461645, 0.467577, 0.473569, 0.479621, 0.485736, 0.491912, 0.498151, 0.504454, 0.510819, 0.51725, 0.523745, 0.530308, 0.536934]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, 'bark_matrix_8k.mat')
        bark_matrix = self.load_mat(bark_path)['Bark_matrix_8k'].astype('float32')
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)

    def load_mat(self, *args, **kwargs):
        from scipy.io import loadmat
        return loadmat(*args, **kwargs)


class PairwiseNegSDR(_Loss):
    """Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
        >>>                            pit_from='pairwise')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-08):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(f'Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead')
        assert targets.size() == est_targets.size()
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)
        if self.sdr_type in ['sisdr', 'sdsdr']:
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    """Base class for single-source negative SI-SDR, SD-SDR and SNR.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction='none', EPS=1e-08):
        assert reduction != 'sum', NotImplementedError
        super().__init__(reduction=reduction)
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-08

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(f'Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead')
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        if self.sdr_type in ['sisdr', 'sdsdr']:
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.EPS
            scaled_target = dot * target / s_target_energy
        else:
            scaled_target = target
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == 'mean' else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    """Base class for computing negative SI-SDR, SD-SDR and SNR for a given
    permutation of source and their estimates.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, time)`.
        - targets: :math:`(batch, nsrc, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
        >>>                            pit_from='perm_avg')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.

    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-08):
        super().__init__()
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-08

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(f'Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead')
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        if self.sdr_type in ['sisdr', 'sdsdr']:
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            scaled_targets = targets
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + self.EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)


class SinkPITLossWrapper(nn.Module):
    """Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        n_iter (int): number of the Sinkhorn iteration (default = 200).
            Supposed to be an even number.
        hungarian_validation (boolean) : Whether to use the Hungarian algorithm
            for the validation. (default = True)

    ``loss_func`` computes pairwise losses and returns a torch.Tensor of shape
    :math:`(batch, n\\_src, n\\_src)`. Each element :math:`(batch, i, j)` corresponds to
    the loss between :math:`targets[:, i]` and :math:`est\\_targets[:, j]`
    It evaluates an approximate value of the PIT loss using Sinkhorn's iterative algorithm.
    See :meth:`~PITLossWrapper.best_softperm_sinkhorn` and http://arxiv.org/abs/2010.11871

    Examples
        >>> import torch
        >>> import pytorch_lightning as pl
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute SinkPIT loss based on pairwise losses
        >>> loss_func = SinkPITLossWrapper(pairwise_neg_sisdr)
        >>> loss_val = loss_func(est_sources, sources)
        >>> # A fixed temperature parameter `beta` (=10) is used
        >>> # unless a cooling callback is set. The value can be
        >>> # dynamically changed using a cooling callback module as follows.
        >>> model = NeuralNetworkModel()
        >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
        >>> dataset = YourDataset()
        >>> loader = data.DataLoader(dataset, batch_size=16)
        >>> system = System(
        >>>     model,
        >>>     optimizer,
        >>>     loss_func=SinkPITLossWrapper(pairwise_neg_sisdr),
        >>>     train_loader=loader,
        >>>     val_loader=loader,
        >>>     )
        >>>
        >>> trainer = pl.Trainer(
        >>>     max_epochs=100,
        >>>     callbacks=[SinkPITBetaScheduler(lambda epoch : 1.02 ** epoch)],
        >>>     )
        >>>
        >>> trainer.fit(system)
    """

    def __init__(self, loss_func, n_iter=200, hungarian_validation=True):
        super().__init__()
        self.loss_func = loss_func
        self._beta = 10
        self.n_iter = n_iter
        self.hungarian_validation = hungarian_validation

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        assert beta > 0
        self._beta = beta

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """Evaluate the loss using Sinkhorn's algorithm.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
                the batch. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape :math:`(batch, nsrc, ...)`.
        """
        n_src = targets.shape[1]
        assert n_src < 100, f'Expected source axis along dim 1, found {n_src}'
        pw_losses = self.loss_func(est_targets, targets, **kwargs)
        assert pw_losses.ndim == 3, 'Something went wrong with the loss function, please read the docs.'
        assert pw_losses.shape[0] == targets.shape[0], 'PIT loss needs same batch dim as input'
        if not return_est:
            if self.training or not self.hungarian_validation:
                min_loss, soft_perm = self.best_softperm_sinkhorn(pw_losses, self._beta, self.n_iter)
                mean_loss = torch.mean(min_loss)
                return mean_loss
            else:
                min_loss, batch_indices = PITLossWrapper.find_best_perm(pw_losses)
                mean_loss = torch.mean(min_loss)
                return mean_loss
        else:
            min_loss, batch_indices = PITLossWrapper.find_best_perm(pw_losses)
            mean_loss = torch.mean(min_loss)
            reordered = PITLossWrapper.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered

    @staticmethod
    def best_softperm_sinkhorn(pair_wise_losses, beta=10, n_iter=200):
        """Compute an approximate PIT loss using Sinkhorn's algorithm.
        See http://arxiv.org/abs/2010.11871

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n_src, n_src)`. Pairwise losses.
            beta (float) : Inverse temperature parameter. (default = 10)
            n_iter (int) : Number of iteration. Even number. (default = 200)

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.Tensor`:
              A soft permutation matrix.
        """
        C = pair_wise_losses.transpose(-1, -2)
        n_src = C.shape[-1]
        Z = -beta * C
        for it in range(n_iter // 2):
            Z = Z - torch.logsumexp(Z, axis=1, keepdim=True)
            Z = Z - torch.logsumexp(Z, axis=2, keepdim=True)
        min_loss = torch.einsum('bij,bij->b', C + Z / beta, torch.exp(Z))
        min_loss = min_loss / n_src
        return min_loss, torch.exp(Z)


class F1_loss(_Loss):
    """Calculate F1 score"""

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, estimates, targets):
        tp = (targets * estimates).sum()
        fp = ((1 - targets) * estimates).sum()
        fn = (targets * (1 - estimates)).sum()
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        return 1 - f1.mean()


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


def is_tracing():
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()


def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Arguments:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `
        `torch.jit.script`` is returned. Otherwise, the original function ``fn`` is returned.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            return fn(*args, **kwargs)
        compiled_fn = torch.jit.script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)
    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True
    return wrapper


def z_norm(x, dims: List[int], eps: float=1e-08):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt(var2 + eps)
    return value


@script_if_tracing
def _glob_norm(x, eps: float=1e-08):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x, EPS: float=1e-08):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class _ConvNormAct(nn.Module):
    """Convolution layer with normalization and a PReLU activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

    Args
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: stride rate for down-sampling. Default is 1
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, use_globln=False):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        if use_globln:
            self.norm = GlobLN(nOut)
            self.act = nn.PReLU()
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
            self.act = nn.PReLU(nOut)

    def forward(self, inp):
        output = self.conv(inp)
        output = self.norm(output)
        return self.act(output)


class _ConvNorm(nn.Module):
    """Convolution layer with normalization without activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE


    Args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: stride rate for down-sampling. Default is 1
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, inp):
        output = self.conv(inp)
        return self.norm(output)


class _NormAct(nn.Module):
    """Normalization and PReLU activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

    Args:
         nOut: number of output channels
    """

    def __init__(self, nOut, use_globln=False):
        super().__init__()
        if use_globln:
            self.norm = GlobLN(nOut)
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, inp):
        output = self.norm(inp)
        return self.act(output)


class _DilatedConvNorm(nn.Module):
    """Dilated convolution with normalized output.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

    Args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: optional stride rate for down-sampling
        d: optional dilation rate
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, use_globln=False):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d, padding=(kSize - 1) // 2 * d, groups=groups)
        if use_globln:
            self.norm = GlobLN(nOut)
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, inp):
        output = self.conv(inp)
        return self.norm(output)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ImprovedTransformedLayer(nn.Module):
    """
    Improved Transformer module as used in [1].
    It is Multi-Head self-attention followed by LSTM, activation and linear projection layer.

    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(self, embed_dim, n_heads, dim_ff, dropout=0.0, activation='relu', bidirectional=True, norm='gLN'):
        super(ImprovedTransformedLayer, self).__init__()
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.recurrent = nn.LSTM(embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True)
        ff_inner_dim = 2 * dim_ff if bidirectional else dim_ff
        self.linear = nn.Linear(ff_inner_dim, embed_dim)
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)

    def forward(self, x):
        tomha = x.permute(2, 0, 1)
        out = self.mha(tomha, tomha, tomha)[0]
        x = self.dropout(out.permute(1, 2, 0)) + x
        x = self.norm_mha(x)
        out = self.linear(self.dropout(self.activation(self.recurrent(x.transpose(1, -1))[0])))
        x = self.dropout(out.transpose(1, -1)) + x
        return self.norm_ff(x)


def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if ``fn`` can be called with ``name`` as a keyword
            argument.

    Returns:
        bool: whether ``fn`` accepts a ``name`` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class DPTransformer(nn.Module):
    """Dual-path Transformer introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(self, in_chan, n_src, n_heads=4, ff_hid=256, chunk_size=100, hop_size=None, n_repeats=6, norm_type='gLN', ff_activation='relu', mask_act='relu', bidirectional=True, dropout=0):
        super(DPTransformer, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mha_in_dim = ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            warnings.warn(f'DPTransformer input dim ({self.in_chan}) is not a multiple of the number of heads ({self.n_heads}). Adding extra linear layer at input to accomodate (size [{self.in_chan} x {self.mha_in_dim}])')
            self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        else:
            self.input_layer = None
        self.in_norm = norms.get(norm_type)(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(nn.ModuleList([ImprovedTransformedLayer(self.mha_in_dim, self.n_heads, self.ff_hid, self.dropout, self.ff_activation, True, self.norm_type), ImprovedTransformedLayer(self.mha_in_dim, self.n_heads, self.ff_hid, self.dropout, self.ff_activation, self.bidirectional, self.norm_type)]))
        net_out_conv = nn.Conv2d(self.mha_in_dim, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        if self.input_layer is not None:
            mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)
        mixture_w = self.in_norm(mixture_w)
        n_orig_frames = mixture_w.shape[-1]
        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()
        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)
        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        output = self.ola.fold(output, output_size=n_orig_frames)
        output = self.net_out(output) * self.net_gate(output)
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        return est_mask

    def get_config(self):
        config = {'in_chan': self.in_chan, 'ff_hid': self.ff_hid, 'n_heads': self.n_heads, 'chunk_size': self.chunk_size, 'hop_size': self.hop_size, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'ff_activation': self.ff_activation, 'mask_act': self.mask_act, 'bidirectional': self.bidirectional, 'dropout': self.dropout}
        return config


class BaseUNet(torch.nn.Module):
    """Base class for u-nets with skip connections between encoders and decoders.

    (For u-nets without skip connections, simply use a `nn.Sequential`.)

    Args:
        encoders (List[torch.nn.Module] of length `N`): List of encoders
        decoders (List[torch.nn.Module] of length `N - 1`): List of decoders
        output_layer (Optional[torch.nn.Module], optional):
            Layer after last decoder.
    """

    def __init__(self, encoders, decoders, *, output_layer=None):
        assert len(encoders) == len(decoders) + 1
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.output_layer = output_layer or torch.nn.Identity()

    def forward(self, x):
        enc_outs = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x)
            enc_outs.append(x)
        for idx, (enc_out, dec) in enumerate(zip(reversed(enc_outs[:-1]), self.decoders)):
            x = dec(x)
            x = torch.cat([x, enc_out], dim=1)
        return self.output_layer(x)


def _none_sequential(*args):
    return torch.nn.Sequential(*[x for x in args if x is not None])


class BaseDCUMaskNet(BaseUNet):
    """Base class for DCU-style mask nets. Used for DCUMaskNet and DCCRMaskNet.

    The preferred way to instantiate this class is to use the ``default_architecture()``
    classmethod.

    Args:
        encoders (List[torch.nn.Module]): List of encoders
        decoders (List[torch.nn.Module]): List of decoders
        output_layer (Optional[torch.nn.Module], optional):
            Layer after last decoder, before mask application.
        mask_bound (Optional[str], optional): Type of mask bound to use, as defined in [1].
            Valid values are "tanh" ("BDT mask"), "sigmoid" ("BDSS mask"), None (unbounded mask).

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    _architectures = NotImplemented

    @classmethod
    def default_architecture(cls, architecture: str, n_src=1, **kwargs):
        """Create a masknet instance from a predefined, named architecture.

        Args:
            architecture (str): Name of predefined architecture. Valid values
                are dependent on the concrete subclass of ``BaseDCUMaskNet``.
            n_src (int, optional): Number of sources
            kwargs (optional): Passed to ``__init__``.
        """
        encoders, decoders = cls._architectures[architecture]
        in_chan, _ignored_out_chan, *rest = decoders[-1]
        decoders = *decoders[:-1], (in_chan, n_src, *rest)
        return cls(encoders, decoders, **kwargs)

    def __init__(self, encoders, decoders, output_layer=None, mask_bound='tanh', **kwargs):
        self.mask_bound = mask_bound
        super().__init__(encoders=encoders, decoders=decoders, output_layer=_none_sequential(output_layer, complex_nn.BoundComplexMask(mask_bound)), **kwargs)

    def forward(self, x):
        fixed_x = self.fix_input_dims(x)
        out = super().forward(fixed_x.unsqueeze(1))
        out = self.fix_output_dims(out, x)
        return out

    def fix_input_dims(self, x):
        """Overwrite this in subclasses to implement input dimension checks."""
        return x

    def fix_output_dims(self, y, x):
        """Overwrite this in subclasses to implement output dimension checks.
        y is the output and x was the input (passed to use the shape)."""
        return y


class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., :-self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm.
            -  ``'cLN'``: channelwise Layernorm.
            -  ``'cgLN'``: cumulative global Layernorm.
            -  Any norm supported by :func:`~.norms.get`
        causal (bool, optional) : Whether or not the convolutions are causal


    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(self, in_chan, hid_chan, skip_out_chan, kernel_size, padding, dilation, norm_type='gLN', causal=False):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(), conv_norm(hid_chan), depth_conv1d, nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        """Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        causal (bool, optional) : Whether or not the convolutions are causal.

    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type='gLN', mask_act='relu', causal=False):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2 ** x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2 ** x
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan, conv_kernel_size, padding=padding, dilation=2 ** x, norm_type=norm_type, causal=causal))
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {'in_chan': self.in_chan, 'out_chan': self.out_chan, 'bn_chan': self.bn_chan, 'hid_chan': self.hid_chan, 'skip_chan': self.skip_chan, 'conv_kernel_size': self.conv_kernel_size, 'n_blocks': self.n_blocks, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'mask_act': self.mask_act, 'causal': self.causal}
        return config


class TDConvNetpp(nn.Module):
    """Improved Temporal Convolutional network used in [1] (TDCN++)

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.

    References
        [1] : Kavalerov, Ilya et al. “Universal Sound Separation.” in WASPAA 2019

    .. note::
        The differences wrt to ConvTasnet's TCN are:

        1. Channel wise layer norm instead of global
        2. Longer-range skip-residual connections from earlier repeat inputs
           to later repeat inputs after passing them through dense layer.
        3. Learnable scaling parameter after each dense layer. The scaling
           parameter for the second dense  layer  in  each  convolutional
           block (which  is  applied  rightbefore the residual connection) is
           initialized to an exponentially decaying scalar equal to 0.9**L,
           where L is the layer or block index.

    """

    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type='fgLN', mask_act='relu'):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan, conv_kernel_size, padding=padding, dilation=2 ** x, norm_type=norm_type))
        self.dense_skip = nn.ModuleList()
        for r in range(n_repeats - 1):
            self.dense_skip.append(nn.Conv1d(bn_chan, bn_chan, 1))
        scaling_param = torch.Tensor([(0.9 ** l) for l in range(1, n_blocks)])
        scaling_param = scaling_param.unsqueeze(0).expand(n_repeats, n_blocks - 1).clone()
        self.scaling_param = nn.Parameter(scaling_param, requires_grad=True)
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()
        out_size = skip_chan if skip_chan else bn_chan
        self.consistency = nn.Linear(out_size, n_src)

    def forward(self, mixture_w):
        """Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output_copy = output
        skip_connection = 0.0
        for r in range(self.n_repeats):
            if r != 0:
                output = self.dense_skip[r - 1](output_copy) + output
                output_copy = output
            for x in range(self.n_blocks):
                i = r * self.n_blocks + x
                tcn_out = self.TCN[i](output)
                if self.skip_chan:
                    residual, skip = tcn_out
                    skip_connection = skip_connection + skip
                else:
                    residual, _ = tcn_out
                scale = self.scaling_param[r, x - 1] if x > 0 else 1.0
                residual = residual * scale
                output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        weights = self.consistency(mask_inp.mean(-1))
        weights = torch.nn.functional.softmax(weights, -1)
        return est_mask, weights

    def get_config(self):
        config = {'in_chan': self.in_chan, 'out_chan': self.out_chan, 'bn_chan': self.bn_chan, 'hid_chan': self.hid_chan, 'skip_chan': self.skip_chan, 'conv_kernel_size': self.conv_kernel_size, 'n_blocks': self.n_blocks, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'mask_act': self.mask_act}
        return config


class DCUNetComplexEncoderBlock(nn.Module):
    """Encoder block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (Tuple[int, int]): Convolution kernel size.
        stride (Tuple[int, int]): Convolution stride.
        padding (Tuple[int, int]): Convolution padding.
        norm_type (str, optional): Type of normalization to use.
            See :mod:`~asteroid.masknn.norms` for valid values.
        activation (str, optional): Type of activation to use.
            See :mod:`~asteroid.masknn.activations` for valid values.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    def __init__(self, in_chan, out_chan, kernel_size, stride, padding, norm_type='bN', activation='leaky_relu'):
        super().__init__()
        self.conv = complex_nn.ComplexConv2d(in_chan, out_chan, kernel_size, stride, padding, bias=norm_type is None)
        self.norm = norms.get_complex(norm_type)(out_chan)
        activation_class = activations.get_complex(activation)
        self.activation = activation_class()

    def forward(self, x: complex_nn.ComplexTensor):
        return self.activation(self.norm(self.conv(x)))


class DCUNetComplexDecoderBlock(nn.Module):
    """Decoder block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (Tuple[int, int]): Convolution kernel size.
        stride (Tuple[int, int]): Convolution stride.
        padding (Tuple[int, int]): Convolution padding.
        norm_type (str, optional): Type of normalization to use.
            See :mod:`~asteroid.masknn.norms` for valid values.
        activation (str, optional): Type of activation to use.
            See :mod:`~asteroid.masknn.activations` for valid values.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    def __init__(self, in_chan, out_chan, kernel_size, stride, padding, output_padding=(0, 0), norm_type='bN', activation='leaky_relu'):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.deconv = complex_nn.ComplexConvTranspose2d(in_chan, out_chan, kernel_size, stride, padding, output_padding, bias=norm_type is None)
        self.norm = norms.get_complex(norm_type)(out_chan)
        activation_class = activations.get_complex(activation)
        self.activation = activation_class()

    def forward(self, x: complex_nn.ComplexTensor):
        return self.activation(self.norm(self.deconv(x)))


def unet_decoder_args(encoders, *, skip_connections):
    """Get list of decoder arguments for upsampling (right) side of a symmetric u-net,
    given the arguments used to construct the encoder.

    Args:
        encoders (tuple of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding)):
            List of arguments used to construct the encoders
        skip_connections (bool): Whether to include skip connections in the
            calculation of decoder input channels.

    Return:
        tuple of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding):
            Arguments to be used to construct decoders
    """
    decoder_args = []
    for enc_in_chan, enc_out_chan, enc_kernel_size, enc_stride, enc_padding in reversed(encoders):
        if skip_connections and decoder_args:
            skip_in_chan = enc_out_chan
        else:
            skip_in_chan = 0
        decoder_args.append((enc_out_chan + skip_in_chan, enc_in_chan, enc_kernel_size, enc_stride, enc_padding))
    return tuple(decoder_args)


def make_unet_encoder_decoder_args(encoder_args, decoder_args):
    encoder_args = tuple((in_chan, out_chan, kernel_size, stride, tuple([(n // 2) for n in kernel_size]) if padding == 'auto' else padding) for in_chan, out_chan, kernel_size, stride, padding in encoder_args)
    if decoder_args == 'auto':
        decoder_args = unet_decoder_args(encoder_args, skip_connections=True)
    else:
        decoder_args = tuple((in_chan, out_chan, kernel_size, stride, tuple([(n // 2) for n in kernel_size]) if padding == 'auto' else padding, output_padding) for in_chan, out_chan, kernel_size, stride, padding, output_padding in decoder_args)
    return encoder_args, decoder_args


DCUNET_ARCHITECTURES = {'DCUNet-10': make_unet_encoder_decoder_args(((1, 32, (7, 5), (2, 2), 'auto'), (32, 64, (7, 5), (2, 2), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto')), 'auto'), 'DCUNet-16': make_unet_encoder_decoder_args(((1, 32, (7, 5), (2, 2), 'auto'), (32, 32, (7, 5), (2, 1), 'auto'), (32, 64, (7, 5), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto')), 'auto'), 'DCUNet-20': make_unet_encoder_decoder_args(((1, 32, (7, 1), (1, 1), 'auto'), (32, 32, (1, 7), (1, 1), 'auto'), (32, 64, (7, 5), (2, 2), 'auto'), (64, 64, (7, 5), (2, 1), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 64, (5, 3), (2, 1), 'auto'), (64, 64, (5, 3), (2, 2), 'auto'), (64, 90, (5, 3), (2, 1), 'auto')), 'auto'), 'Large-DCUNet-20': make_unet_encoder_decoder_args(((1, 45, (7, 1), (1, 1), 'auto'), (45, 45, (1, 7), (1, 1), 'auto'), (45, 90, (7, 5), (2, 2), 'auto'), (90, 90, (7, 5), (2, 1), 'auto'), (90, 90, (5, 3), (2, 2), 'auto'), (90, 90, (5, 3), (2, 1), 'auto'), (90, 90, (5, 3), (2, 2), 'auto'), (90, 90, (5, 3), (2, 1), 'auto'), (90, 90, (5, 3), (2, 2), 'auto'), (90, 128, (5, 3), (2, 1), 'auto')), ((128, 90, (5, 3), (2, 1), 'auto', (0, 0)), (180, 90, (5, 3), (2, 2), 'auto', (0, 0)), (180, 90, (5, 3), (2, 1), 'auto', (0, 0)), (180, 90, (5, 3), (2, 2), 'auto', (0, 0)), (180, 90, (5, 3), (2, 1), 'auto', (0, 0)), (180, 90, (5, 3), (2, 2), 'auto', (0, 0)), (180, 90, (7, 5), (2, 1), 'auto', (0, 0)), (180, 90, (7, 5), (2, 2), 'auto', (0, 0)), (135, 90, (1, 7), (1, 1), 'auto', (0, 0)), (135, 1, (7, 1), (1, 1), 'auto', (0, 0)))), 'mini': make_unet_encoder_decoder_args(((1, 4, (7, 5), (2, 2), 'auto'), (4, 8, (7, 5), (2, 2), 'auto'), (8, 16, (5, 3), (2, 2), 'auto')), 'auto')}


@script_if_tracing
def _fix_dcu_input_dims(fix_length_mode: Optional[str], x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[1] - 1) % freq_prod:
        raise TypeError(f'Input shape must be [batch, freq + 1, time + 1] with freq divisible by {freq_prod}, got {x.shape} instead')
    time_remainder = (x.shape[2] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(f"Input shape must be [batch, freq + 1, time + 1] with time divisible by {time_prod}, got {x.shape} instead. Set the 'fix_length_mode' argument in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically.")
        elif fix_length_mode == 'pad':
            pad_shape = [0, time_prod - time_remainder]
            x = nn.functional.pad(x, pad_shape, mode='constant')
        elif fix_length_mode == 'trim':
            pad_shape = [0, -time_remainder]
            x = nn.functional.pad(x, pad_shape, mode='constant')
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


@script_if_tracing
def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int=-1) ->torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


@script_if_tracing
def _fix_dcu_output_dims(fix_length_mode: Optional[str], out, x):
    """Fix shape of `out` to the original shape of `x`."""
    return pad_x_to_y(out, x)


class DCUMaskNet(BaseDCUMaskNet):
    """Masking part of DCUNet, as proposed in [1].

    Valid `architecture` values for the ``default_architecture`` classmethod are:
    "Large-DCUNet-20", "DCUNet-20", "DCUNet-16", "DCUNet-10" and "mini".

    Valid `fix_length_mode` values are [None, "pad", "trim"].

    Input shape is expected to be $(batch, nfreqs, time)$, with $nfreqs - 1$ divisible
    by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides of the encoders,
    and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$ where $t_N$ are the time
    strides of the encoders.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    _architectures = DCUNET_ARCHITECTURES

    def __init__(self, encoders, decoders, fix_length_mode=None, **kwargs):
        self.fix_length_mode = fix_length_mode
        self.encoders_stride_product = np.prod([enc_stride for _, _, _, enc_stride, _ in encoders], axis=0)
        super().__init__(encoders=[DCUNetComplexEncoderBlock(*args) for args in encoders], decoders=[DCUNetComplexDecoderBlock(*args) for args in decoders[:-1]], output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]), **kwargs)

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product))

    def fix_output_dims(self, out, x):
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


class _BaseUBlock(nn.Module):

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4, use_globln=False):
        super().__init__()
        self.proj_1x1 = _ConvNormAct(out_chan, in_chan, 1, stride=1, groups=1, use_globln=use_globln)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(_DilatedConvNorm(in_chan, in_chan, kSize=5, stride=1, groups=in_chan, d=1, use_globln=use_globln))
        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(_DilatedConvNorm(in_chan, in_chan, kSize=2 * stride + 1, stride=stride, groups=in_chan, d=1, use_globln=use_globln))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2)


class UBlock(_BaseUBlock):
    """Upsampling block.

    Based on the following principle: ``REDUCE ---> SPLIT ---> TRANSFORM --> MERGE``
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=False)
        self.conv_1x1_exp = _ConvNorm(in_chan, out_chan, 1, 1, groups=1)
        self.final_norm = _NormAct(in_chan)
        self.module_act = _NormAct(out_chan)

    def forward(self, x):
        """
        Args:
            x: input feature map

        Returns:
            transformed feature map
        """
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k[..., :output[-1].shape[-1]]
        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))
        return self.module_act(expanded + x)


class SuDORMRF(nn.Module):
    """SuDORMRF mask network, as described in [1].

    Args:
        in_chan (int): Number of input channels. Also number of output channels.
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.

    References
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
        Tzinis et al. MLSP 2020.
    """

    def __init__(self, in_chan, n_src, bn_chan=128, num_blocks=16, upsampling_depth=4, mask_act='softmax'):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.bn_chan = bn_chan
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.mask_act = mask_act
        self.ln = nn.GroupNorm(1, in_chan, eps=1e-08)
        self.l1 = nn.Conv1d(in_chan, bn_chan, kernel_size=1)
        self.sm = nn.Sequential(*[UBlock(out_chan=bn_chan, in_chan=in_chan, upsampling_depth=upsampling_depth) for _ in range(num_blocks)])
        if bn_chan != in_chan:
            self.reshape_before_masks = nn.Conv1d(bn_chan, in_chan, kernel_size=1)
        self.m = nn.Conv2d(1, n_src, kernel_size=(in_chan + 1, 1), padding=(in_chan - in_chan // 2, 0))
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)
        if self.bn_chan != self.in_chan:
            x = self.reshape_before_masks(x)
        x = self.m(x.unsqueeze(1))
        x = self.output_act(x)
        return x

    def get_config(self):
        config = {'in_chan': self.in_chan, 'n_src': self.n_src, 'bn_chan': self.bn_chan, 'num_blocks': self.num_blocks, 'upsampling_depth': self.upsampling_depth, 'mask_act': self.mask_act}
        return config


class UConvBlock(_BaseUBlock):
    """Block which performs successive downsampling and upsampling
    in order to be able to analyze the input features in multiple resolutions.
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=True)
        self.final_norm = _NormAct(in_chan, use_globln=True)
        self.res_conv = nn.Conv1d(in_chan, out_chan, 1)

    def forward(self, x):
        """
        Args
            x: input feature map

        Returns:
            transformed feature map
        """
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k[..., :output[-1].shape[-1]]
        expanded = self.final_norm(output[-1])
        return self.res_conv(expanded) + residual


class SuDORMRFImproved(nn.Module):
    """Improved SuDORMRF mask network, as described in [1].

    Args:
        in_chan (int): Number of input channels. Also number of output channels.
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks
        upsampling_depth (int): Depth of upsampling
        mask_act (str): Name of output activation.


    References
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
        Tzinis et al. MLSP 2020.
    """

    def __init__(self, in_chan, n_src, bn_chan=128, num_blocks=16, upsampling_depth=4, mask_act='relu'):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.bn_chan = bn_chan
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.mask_act = mask_act
        self.ln = GlobLN(in_chan)
        self.bottleneck = nn.Conv1d(in_chan, bn_chan, kernel_size=1)
        self.sm = nn.Sequential(*[UConvBlock(out_chan=bn_chan, in_chan=in_chan, upsampling_depth=upsampling_depth) for _ in range(num_blocks)])
        mask_conv = nn.Conv1d(bn_chan, n_src * in_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, x):
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)
        x = self.mask_net(x)
        x = x.view(x.shape[0], self.n_src, self.in_chan, -1)
        x = self.output_act(x)
        return x

    def get_config(self):
        config = {'in_chan': self.in_chan, 'n_src': self.n_src, 'bn_chan': self.bn_chan, 'num_blocks': self.num_blocks, 'upsampling_depth': self.upsampling_depth, 'mask_act': self.mask_act}
        return config


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""

    def forward(self, x, EPS: float=1e-08):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, *]`
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""

    def forward(self, x, EPS: float=1e-08):
        """

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             :class:`torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


@script_if_tracing
def _feat_glob_norm(x, eps: float=1e-08):
    dims: List[int] = torch.arange(2, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


class FeatsGlobLN(_LayerNorm):
    """Feature-wise global Layer Normalization (FeatsGlobLN).
    Applies normalization over frames for each channel."""

    def forward(self, x, EPS: float=1e-08):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, time]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, time]`
        """
        value = _feat_glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError('expected 4D or 3D input (got {}D input)'.format(input.dim()))


class SingleRNN(nn.Module):
    """Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ['RNN', 'LSTM', 'GRU']
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=bool(bidirectional))

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        self.rnn.flatten_parameters()
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class MulCatRNN(nn.Module):
    """MulCat RNN block from [1].

    Composed of two RNNs, returns ``cat([RNN_1(x) * RNN_2(x), x])``.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    References
        [1] Eliya Nachmani, Yossi Adi, & Lior Wolf. (2020). Voice Separation with an Unknown Number of Multiple Speakers.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        super(MulCatRNN, self).__init__()
        assert rnn_type.upper() in ['RNN', 'LSTM', 'GRU']
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn1 = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=bool(bidirectional))
        self.rnn2 = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=bool(bidirectional))

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1) + self.input_size

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        rnn_output1, _ = self.rnn1(inp)
        rnn_output2, _ = self.rnn2(inp)
        return torch.cat((rnn_output1 * rnn_output2, inp), 2)


class StackedResidualRNN(nn.Module):
    """Stacked RNN with builtin residual connection.
    Only supports forward RNNs.
    See StackedResidualBiRNN for bidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=False):
        super(StackedResidualRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is False, 'Bidirectional not supported yet'
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SingleRNN(rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """Builtin residual connections + dropout applied before residual.
        Input shape : [batch, time_axis, feat_axis]
        """
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class StackedResidualBiRNN(nn.Module):
    """Stacked Bidirectional RNN with builtin residual connection.
    Residual connections are applied on both RNN directions.
    Only supports bidiriectional RNNs.
    See StackedResidualRNN for unidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is True, 'Only bidirectional not supported yet'
        self.bidirectional = bidirectional
        self.first_layer = SingleRNN(rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional)
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            input_size = 2 * n_units
            self.layers.append(SingleRNN(rnn_type, input_size=input_size, hidden_size=n_units, bidirectional=bidirectional))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """Builtin residual connections + dropout applied before residual.
        Input shape : [batch, time_axis, feat_axis]
        """
        rnn_out = self.first_layer(x)
        dropped_out = self.dropout_layer(rnn_out)
        x = torch.cat([x, x], dim=-1) + dropped_out
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class DPRNNBlock(nn.Module):
    """Dual-Path RNN Block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_size (int): Number of hidden neurons in the RNNs.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN.
        rnn_type (str, optional): Type of RNN used. Choose from ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers used in each RNN.
        dropout (float, optional): Dropout ratio. Must be in [0, 1].

    References
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(self, in_chan, hid_size, norm_type='gLN', bidirectional=True, rnn_type='LSTM', use_mulcat=False, num_layers=1, dropout=0):
        super(DPRNNBlock, self).__init__()
        if use_mulcat:
            self.intra_RNN = MulCatRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True)
            self.inter_RNN = MulCatRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            self.intra_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True)
            self.inter_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, in_chan)
        self.intra_norm = norms.get(norm_type)(in_chan)
        self.inter_linear = nn.Linear(self.inter_RNN.output_size, in_chan)
        self.inter_norm = norms.get(norm_type)(in_chan)

    def forward(self, x):
        """Input shape : [batch, feats, chunk_size, num_chunks]"""
        B, N, K, L = x.size()
        output = x
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    """Dual-path RNN Network for Single-Channel Source Separation
        introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(self, in_chan, n_src, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type='gLN', mask_act='relu', bidirectional=True, rnn_type='LSTM', use_mulcat=False, num_layers=1, dropout=0):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        net = []
        for x in range(self.n_repeats):
            net += [DPRNNBlock(bn_chan, hid_size, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type, use_mulcat=use_mulcat, num_layers=num_layers, dropout=dropout)]
        self.net = nn.Sequential(*net)
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        output = self.net(output)
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(output.reshape(batch * self.n_src, to_unfold, n_chunks), (n_frames, 1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask

    def get_config(self):
        config = {'in_chan': self.in_chan, 'out_chan': self.out_chan, 'bn_chan': self.bn_chan, 'hid_size': self.hid_size, 'chunk_size': self.chunk_size, 'hop_size': self.hop_size, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'mask_act': self.mask_act, 'bidirectional': self.bidirectional, 'rnn_type': self.rnn_type, 'num_layers': self.num_layers, 'dropout': self.dropout, 'use_mulcat': self.use_mulcat}
        return config


class LSTMMasker(nn.Module):
    """LSTM mask network introduced in [1], without skip connections.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        n_layers (int, optional): Number of layers in each RNN.
        hid_size (int): Number of neurons in the RNNs cell state.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): Whether to use BiLSTM
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1]: Yi Luo et al. "Real-time Single-channel Dereverberation and Separation
        with Time-domain Audio Separation Network", Interspeech 2018
    """

    def __init__(self, in_chan, n_src, out_chan=None, rnn_type='lstm', n_layers=4, hid_size=512, dropout=0.3, mask_act='sigmoid', bidirectional=True):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hid_size = hid_size
        self.dropout = dropout
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()
        out_size = hid_size * (int(bidirectional) + 1)
        if bidirectional:
            self.bn_layer = GlobLN(in_chan)
        else:
            self.bn_layer = CumLN(in_chan)
        self.masker = nn.Sequential(SingleRNN('lstm', in_chan, hidden_size=hid_size, n_layers=n_layers, bidirectional=bidirectional, dropout=dropout), nn.Linear(out_size, self.n_src * out_chan), self.output_act)

    def forward(self, x):
        batch_size = x.shape[0]
        to_sep = self.bn_layer(x)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.out_chan, -1)
        return est_masks

    def get_config(self):
        config = {'in_chan': self.in_chan, 'n_src': self.n_src, 'out_chan': self.out_chan, 'rnn_type': self.rnn_type, 'n_layers': self.n_layers, 'hid_size': self.hid_size, 'dropout': self.dropout, 'mask_act': self.mask_act, 'bidirectional': self.bidirectional}
        return config


class DCCRMaskNetRNN(nn.Module):
    """RNN (LSTM) layer between encoders and decoders introduced in [1].

    Args:
        in_size (int): Number of inputs to the RNN. Must be the product of non-batch,
            non-time dimensions of output shape of last encoder, i.e. if the last
            encoder output shape is $(batch, nchans, nfreqs, time)$, `in_size` must be
            $nchans * nfreqs$.
        hid_size (int, optional): Number of units in RNN.
        rnn_type (str, optional): Type of RNN to use. See ``SingleRNN`` for valid values.
        n_layers (int, optional): Number of layers used in RNN.
        norm_type (Optional[str], optional): Norm to use after linear.
            See ``asteroid.masknn.norms`` for valid values. (Not used in [1]).
        rnn_kwargs (optional): Passed to :func:`~.recurrent.SingleRNN`.

    References
        [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    def __init__(self, in_size, hid_size=128, rnn_type='LSTM', n_layers=2, norm_type=None, **rnn_kwargs):
        super().__init__()
        self.rnn = complex_nn.ComplexSingleRNN(rnn_type, in_size, hid_size, n_layers=n_layers, **rnn_kwargs)
        self.linear = complex_nn.ComplexMultiplicationWrapper(nn.Linear, self.rnn.output_size, in_size)
        self.norm = norms.get_complex(norm_type)

    def forward(self, x: complex_nn.ComplexTensor):
        """Input shape: [batch, ..., time]"""
        x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
        x = self.linear(self.rnn(x.reshape(*x.shape[:2], -1))).reshape(*x.shape)
        x = x.permute(0, *range(2, x.ndim), 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


DCCRN_ARCHITECTURES = {'DCCRN-CL': (((1, 16, (5, 2), (2, 1), (2, 0)), (16, 32, (5, 2), (2, 1), (2, 0)), (32, 64, (5, 2), (2, 1), (2, 0)), (64, 128, (5, 2), (2, 1), (2, 0)), (128, 128, (5, 2), (2, 1), (2, 0)), (128, 128, (5, 2), (2, 1), (2, 0))), ((256, 128, (5, 2), (2, 1), (2, 0), (1, 0)), (256, 128, (5, 2), (2, 1), (2, 0), (1, 0)), (256, 64, (5, 2), (2, 1), (2, 0), (1, 0)), (128, 32, (5, 2), (2, 1), (2, 0), (1, 0)), (64, 16, (5, 2), (2, 1), (2, 0), (1, 0)), (32, 1, (5, 2), (2, 1), (2, 0), (1, 0)))), 'mini': (((1, 4, (5, 2), (2, 1), (2, 0)), (4, 8, (5, 2), (2, 1), (2, 0))), ((16, 4, (5, 2), (2, 1), (2, 0), (1, 0)), (8, 1, (5, 2), (2, 1), (2, 0), (1, 0))))}


class DCCRMaskNet(BaseDCUMaskNet):
    """Masking part of DCCRNet, as proposed in [1].

    Valid `architecture` values for the ``default_architecture`` classmethod are:
    "DCCRN" and "mini".

    Args:
        encoders (list of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding)):
            Arguments of encoders of the u-net
        decoders (list of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding))
            Arguments of decoders of the u-net
        n_freqs (int): Number of frequencies (dim 1) of input to ``.forward()``.
            Must be divisible by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency
            strides of the encoders.

    Input shape is expected to be $(batch, nfreqs, time)$, with $nfreqs$ divisible
    by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides of the encoders.

    References
        [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """
    _architectures = DCCRN_ARCHITECTURES

    def __init__(self, encoders, decoders, n_freqs, **kwargs):
        self.encoders_stride_product = np.prod([enc_stride for _, _, _, enc_stride, _ in encoders], axis=0)
        freq_prod, _ = self.encoders_stride_product
        last_encoder_out_shape = encoders[-1][1], int(np.ceil(n_freqs / freq_prod))
        super().__init__(encoders=[*(DCUNetComplexEncoderBlock(*args, activation='prelu') for args in encoders), DCCRMaskNetRNN(np.prod(last_encoder_out_shape))], decoders=[torch.nn.Identity(), *(DCUNetComplexDecoderBlock(*args, activation='prelu') for args in decoders[:-1])], output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]), **kwargs)

    def fix_input_dims(self, x):
        freq_prod, _ = self.encoders_stride_product
        if x.shape[1] % freq_prod:
            raise TypeError(f'Input shape must be [batch, freq, time] with freq divisible by {freq_prod}, got {x.shape} instead')
        return x


class TAC(nn.Module):
    """Transform-Average-Concatenate inter-microphone-channel permutation invariant communication block [1].

    Args:
        input_dim (int): Number of features of input representation.
        hidden_dim (int, optional): size of hidden layers in TAC operations.
        activation (str, optional): type of activation used. See asteroid.masknn.activations.
        norm_type (str, optional): type of normalization layer used. See asteroid.masknn.norms.

    .. note:: Supports inputs of shape :math:`(batch, mic\\_channels, features, chunk\\_size, n\\_chunks)`
        as in FasNet-TAC. The operations are applied for each element in ``chunk_size`` and ``n_chunks``.
        Output is of same shape as input.

    References
        [1] : Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel
        speech separation." ICASSP 2020.
    """

    def __init__(self, input_dim, hidden_dim=384, activation='prelu', norm_type='gLN'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_tf = nn.Sequential(nn.Linear(input_dim, hidden_dim), activations.get(activation)())
        self.avg_tf = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activations.get(activation)())
        self.concat_tf = nn.Sequential(nn.Linear(2 * hidden_dim, input_dim), activations.get(activation)())
        self.norm = norms.get(norm_type)(input_dim)

    def forward(self, x, valid_mics=None):
        """
        Args:
            x: (:class:`torch.Tensor`): Input multi-channel DPRNN features.
                Shape: :math:`(batch, mic\\_channels, features, chunk\\_size, n\\_chunks)`.
            valid_mics: (:class:`torch.LongTensor`): tensor containing effective number of microphones on each batch.
                Batches can be composed of examples coming from arrays with a different
                number of microphones and thus the ``mic_channels`` dimension is padded.
                E.g. torch.tensor([4, 3]) means first example has 4 channels and the second 3.
                Shape:  :math`(batch)`.

        Returns:
            output (:class:`torch.Tensor`): features for each mic_channel after TAC inter-channel processing.
                Shape :math:`(batch, mic\\_channels, features, chunk\\_size, n\\_chunks)`.
        """
        batch_size, nmics, channels, chunk_size, n_chunks = x.size()
        if valid_mics is None:
            valid_mics = torch.LongTensor([nmics] * batch_size)
        output = self.input_tf(x.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics * chunk_size * n_chunks, channels)).reshape(batch_size, chunk_size, n_chunks, nmics, self.hidden_dim)
        if valid_mics.max() == 0:
            mics_mean = output.mean(1)
        else:
            mics_mean = [output[b, :, :, :valid_mics[b]].mean(2).unsqueeze(0) for b in range(batch_size)]
            mics_mean = torch.cat(mics_mean, 0)
        mics_mean = self.avg_tf(mics_mean.reshape(batch_size * chunk_size * n_chunks, self.hidden_dim))
        mics_mean = mics_mean.reshape(batch_size, chunk_size, n_chunks, self.hidden_dim).unsqueeze(3).expand_as(output)
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(output.reshape(batch_size * chunk_size * n_chunks * nmics, -1)).reshape(batch_size, chunk_size, n_chunks, nmics, -1)
        output = self.norm(output.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics, -1, chunk_size, n_chunks)).reshape(batch_size, nmics, -1, chunk_size, n_chunks)
        output += x
        return output


class F1Tracker(nn.Module):
    """F1 score tracker."""

    def __init__(self, epsilon=1e-07):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        tp = torch.sum(torch.logical_and(y_pred, y_true))
        tn = torch.sum(torch.logical_and(torch.logical_not(y_pred), torch.logical_not(y_true)))
        fp = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_pred))
        fn = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_true))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return {'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1)}


MODELS_URLS_HASHTABLE = {'mpariente/ConvTasNet_WHAM!_sepclean': 'https://zenodo.org/record/3862942/files/model.pth?download=1', 'mpariente/DPRNNTasNet_WHAM!_sepclean': 'https://zenodo.org/record/3873670/files/model.pth?download=1', 'mpariente/DPRNNTasNet(ks=16)_WHAM!_sepclean': 'https://zenodo.org/record/3903795/files/model.pth?download=1', 'Cosentino/ConvTasNet_LibriMix_sep_clean': 'https://zenodo.org/record/3873572/files/model.pth?download=1', 'Cosentino/ConvTasNet_LibriMix_sep_noisy': 'https://zenodo.org/record/3874420/files/model.pth?download=1', 'brijmohan/ConvTasNet_Libri1Mix_enhsingle': 'https://zenodo.org/record/3970768/files/model.pth?download=1', 'groadabike/ConvTasNet_DAMP-VSEP_enhboth': 'https://zenodo.org/record/3994193/files/model.pth?download=1', 'popcornell/DeMask_Surgical_mask_speech_enhancement_v1': 'https://zenodo.org/record/3997047/files/model.pth?download=1', 'popcornell/DPRNNTasNet_WHAM_enhancesingle': 'https://zenodo.org/record/3998647/files/model.pth?download=1', 'tmirzaev-dotcom/ConvTasNet_Libri3Mix_sepnoisy': 'https://zenodo.org/record/4020529/files/model.pth?download=1', 'mhu-coder/ConvTasNet_Libri1Mix_enhsingle': 'https://zenodo.org/record/4301955/files/model.pth?download=1', 'r-sawata/XUMX_MUSDB18_music_separation': 'https://zenodo.org/record/4704231/files/pretrained_xumx.pth?download=1'}


SR_HASHTABLE = {k: (8000.0 if not 'DeMask' in k else 16000.0) for k in MODELS_URLS_HASHTABLE}


def get_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def url_to_filename(url):
    """Consistently convert ``url`` into a filename."""
    _bytes = url.encode('utf-8')
    _hash = sha256(_bytes)
    filename = _hash.hexdigest()
    return filename


def cached_download(filename_or_url):
    """Download from URL and cache the result in ASTEROID_CACHE.

    Args:
        filename_or_url (str): Name of a model as named on the Zenodo Community
            page (ex: ``"mpariente/ConvTasNet_WHAM!_sepclean"``), or model id from
            the Hugging Face model hub (ex: ``"julien-c/DPRNNTasNet-ks16_WHAM_sepclean"``),
            or a URL to a model file (ex: ``"https://zenodo.org/.../model.pth"``), or a filename
            that exists locally (ex: ``"local/tmp_model.pth"``)

    Returns:
        str, normalized path to the downloaded (or not) model
    """
    if os.path.isfile(filename_or_url):
        return filename_or_url
    if filename_or_url.startswith(huggingface_hub.HUGGINGFACE_CO_URL_HOME):
        filename_or_url = filename_or_url[len(huggingface_hub.HUGGINGFACE_CO_URL_HOME):]
    if filename_or_url.startswith(('http://', 'https://')):
        url = filename_or_url
    elif filename_or_url in MODELS_URLS_HASHTABLE:
        url = MODELS_URLS_HASHTABLE[filename_or_url]
    else:
        if '@' in filename_or_url:
            model_id = filename_or_url.split('@')[0]
            revision = filename_or_url.split('@')[1]
        else:
            model_id = filename_or_url
            revision = None
        return huggingface_hub.hf_hub_download(repo_id=model_id, filename=huggingface_hub.PYTORCH_WEIGHTS_NAME, cache_dir=get_cache_dir(), revision=revision, library_name='asteroid', library_version=asteroid_version)
    cached_filename = url_to_filename(url)
    cached_dir = os.path.join(get_cache_dir(), cached_filename)
    cached_path = os.path.join(cached_dir, 'model.pth')
    os.makedirs(cached_dir, exist_ok=True)
    if not os.path.isfile(cached_path):
        hub.download_url_to_file(url, cached_path)
        return cached_path
    None
    return cached_path


class BaseModel(torch.nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures, and separation interface to `separate`.
    Need to overwrite the `forward` and `get_model_args` methods.

    Models inheriting from `BaseModel` can be used by :mod:`asteroid.separate`
    and by the `asteroid-infer` CLI. For models whose `forward` doesn't go from
    waveform to waveform tensors, overwrite `forward_wav` to return
    waveform tensors.

    Args:
        sample_rate (float): Operating sample rate of the model.
        in_channels: Number of input channels in the signal.
            If None, no checks will be performed.
    """

    def __init__(self, sample_rate: float, in_channels: Optional[int]=1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn('Other sub-components of the model might have a `sample_rate` attribute, be sure to modify them for consistency.', UserWarning)
        self.__sample_rate = new_sample_rate

    def separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.separate`."""
        return separate.separate(self, *args, **kwargs)

    def torch_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.torch_separate`."""
        return separate.torch_separate(self, *args, **kwargs)

    def numpy_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.numpy_separate`."""
        return separate.numpy_separate(self, *args, **kwargs)

    def file_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.file_separate`."""
        return separate.file_separate(self, *args, **kwargs)

    def forward_wav(self, wav, *args, **kwargs):
        """Separation method for waveforms.

        In case the network's `forward` doesn't have waveforms as input/output,
        overwrite this method to separate from waveform to waveform.
        Should return a single torch.Tensor, the separated waveforms.

        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location='cpu')
        else:
            conf = pretrained_model_conf_or_path
        if 'model_name' not in conf.keys():
            raise ValueError('Expected config dictionary to have field model_name`. Found only: {}'.format(conf.keys()))
        if 'state_dict' not in conf.keys():
            raise ValueError('Expected config dictionary to have field state_dict`. Found only: {}'.format(conf.keys()))
        if 'model_args' not in conf.keys():
            raise ValueError('Expected config dictionary to have field model_args`. Found only: {}'.format(conf.keys()))
        conf['model_args'].update(kwargs)
        if 'sample_rate' not in conf['model_args'] and isinstance(pretrained_model_conf_or_path, str):
            conf['model_args']['sample_rate'] = SR_HASHTABLE.get(pretrained_model_conf_or_path, None)
        try:
            model_class = get(conf['model_name'])
        except ValueError:
            model = cls(*args, **conf['model_args'])
        else:
            model = model_class(*args, **conf['model_args'])
        model.load_state_dict(conf['state_dict'])
        return model

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        model_conf = dict(model_name=self.__class__.__name__, state_dict=self.get_state_dict(), model_args=self.get_model_args())
        infos = dict()
        infos['software_versions'] = dict(torch_version=torch.__version__, pytorch_lightning_version=pl.__version__, asteroid_version=asteroid_version)
        model_conf['infos'] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


@script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


@script_if_tracing
def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


@script_if_tracing
def jitable_shape(tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler

    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, 'sample_rate', None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or 'linear')()

    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def forward_encoder(self, wav: torch.Tensor) ->torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) ->torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) ->torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) ->torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError('Filterbank and Mask network config share common keys. Merging them is not safe.')
        model_args = {**fb_config, **masknet_config, 'encoder_activation': self.encoder_activation}
        return model_args


class ConvTasNet(BaseEncoderMaskerDecoder):
    """ConvTasNet separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """

    def __init__(self, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type='gLN', mask_act='sigmoid', in_chan=None, causal=False, fb_name='free', kernel_size=16, n_filters=512, stride=8, encoder_activation=None, sample_rate=8000, **fb_kwargs):
        encoder, decoder = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, sample_rate=sample_rate, **fb_kwargs)
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, f'Number of filterbank output channels and number of input channels should be the same. Received {n_feats} and {in_chan}'
        if causal and norm_type not in ['cgLN', 'cLN']:
            norm_type = 'cLN'
            warnings.warn(f'In causal configuration cumulative layer normalization (cgLN)or channel-wise layer normalization (chanLN)  must be used. Changing {norm_type} to cLN')
        masker = TDConvNet(n_feats, n_src, out_chan=out_chan, n_blocks=n_blocks, n_repeats=n_repeats, bn_chan=bn_chan, hid_chan=hid_chan, skip_chan=skip_chan, conv_kernel_size=conv_kernel_size, norm_type=norm_type, mask_act=mask_act, causal=causal)
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


class VADNet(ConvTasNet):

    def forward_decoder(self, masked_tf_rep: torch.Tensor) ->torch.Tensor:
        return torch.nn.functional.sigmoid(self.decoder(masked_tf_rep))


def build_demask_masker(n_in, n_out, activation='relu', dropout=0.0, hidden_dims=(1024,), mask_act='relu', norm_type='gLN'):
    make_layer_norm = norms.get(norm_type)
    net = [make_layer_norm(n_in)]
    layer_activation = activations.get(activation)()
    in_chan = n_in
    for hidden_dim in hidden_dims:
        net.extend([nn.Conv1d(in_chan, hidden_dim, 1), make_layer_norm(hidden_dim), layer_activation, nn.Dropout(dropout)])
        in_chan = hidden_dim
    net.extend([nn.Conv1d(in_chan, n_out, 1), activations.get(mask_act)()])
    return nn.Sequential(*net)


class DeMask(BaseEncoderMaskerDecoder):
    """
    Simple MLP model for surgical mask speech enhancement A transformed-domain masking approach is used.

    Args:
        input_type (str, optional): whether the magnitude spectrogram "mag" or both real imaginary parts "reim" are
                    passed as features to the masker network.
                    Concatenation of "mag" and "reim" also can be used by using "cat".
        output_type (str, optional): whether the masker ouputs a mask
                    for magnitude spectrogram "mag" or both real imaginary parts "reim".

        hidden_dims (list, optional): list of MLP hidden layer sizes.
        dropout (float, optional): dropout probability.
        activation (str, optional): type of activation used in hidden MLP layers.
        mask_act (str, optional): Which non-linear function to generate mask.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.

        fb_name (str): type of analysis and synthesis filterbanks used,
                            choose between ["stft", "free", "analytic_free"].
        n_filters (int): number of filters in the analysis and synthesis filterbanks.
        stride (int): filterbank filters stride.
        kernel_size (int): length of filters in the filterbank.
        encoder_activation (str)
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """

    def __init__(self, input_type='mag', output_type='mag', hidden_dims=(1024,), dropout=0.0, activation='relu', mask_act='relu', norm_type='gLN', fb_name='stft', n_filters=512, stride=256, kernel_size=512, sample_rate=16000, **fb_kwargs):
        encoder, decoder = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, sample_rate=sample_rate, **fb_kwargs)
        n_masker_in = self._get_n_feats_input(input_type, encoder.n_feats_out)
        n_masker_out = self._get_n_feats_output(output_type, encoder.n_feats_out)
        masker = build_demask_masker(n_masker_in, n_masker_out, norm_type=norm_type, activation=activation, hidden_dims=hidden_dims, dropout=dropout, mask_act=mask_act)
        super().__init__(encoder, masker, decoder)
        self.input_type = input_type
        self.output_type = output_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.mask_act = mask_act
        self.norm_type = norm_type

    def _get_n_feats_input(self, input_type, encoder_n_out):
        if input_type == 'reim':
            return encoder_n_out
        if input_type not in {'mag', 'cat'}:
            raise NotImplementedError('Input type should be either mag, reim or cat')
        n_feats_input = encoder_n_out // 2
        if input_type == 'cat':
            n_feats_input += encoder_n_out
        return n_feats_input

    def _get_n_feats_output(self, output_type, encoder_n_out):
        if output_type == 'mag':
            return encoder_n_out // 2
        if output_type == 'reim':
            return encoder_n_out
        raise NotImplementedError('Output type should be either mag or reim')

    def forward_masker(self, tf_rep):
        """Estimates masks based on time-frequency representations.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in
                (batch, freq, seq).

        Returns:
            torch.Tensor: Estimated masks in (batch, freq, seq).
        """
        masker_input = tf_rep
        if self.input_type == 'mag':
            masker_input = mag(masker_input)
        elif self.input_type == 'cat':
            masker_input = magreim(masker_input)
        est_masks = self.masker(masker_input)
        if self.output_type == 'mag':
            est_masks = est_masks.repeat(1, 2, 1)
        return est_masks

    def apply_masks(self, tf_rep, est_masks):
        """Applies masks to time-frequency representations.

        Args:
            tf_rep (torch.Tensor): Time-frequency representations in
                (batch, freq, seq).
            est_masks (torch.Tensor): Estimated masks in (batch, freq, seq).

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        if self.output_type == 'reim':
            tf_rep = tf_rep.unsqueeze(1)
        return est_masks * tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {'input_type': self.input_type, 'output_type': self.output_type, 'hidden_dims': self.hidden_dims, 'dropout': self.dropout, 'activation': self.activation, 'mask_act': self.mask_act, 'norm_type': self.norm_type}
        model_args.update(self.encoder.filterbank.get_config())
        return model_args


def xcorr(inp, ref, normalized=True, eps=1e-08):
    """Multi-channel cross correlation.

    The two signals can have different lengths but the input signal should be shorter than the reference signal.

    .. note:: The cross correlation is computed between each pair of microphone channels and not
        between all possible pairs e.g. if both input and ref have shape ``(1, 2, 100)``
        the output will be ``(1, 2, 1)`` the first element is the xcorr between
        the first mic channel of input and the first mic channel of ref.
        If either input and ref have only one channel e.g. input: (1, 3, 100) and ref: ``(1, 1, 100)``
        then output will be ``(1, 3, 1)`` as ref will be broadcasted to have same shape as input.

    Args:
        inp (:class:`torch.Tensor`): multi-channel input signal. Shape: :math:`(batch, mic\\_channels, seq\\_len)`.
        ref (:class:`torch.Tensor`): multi-channel reference signal. Shape: :math:`(batch, mic\\_channels, seq\\_len)`.
        normalized (bool, optional): whether to normalize the cross-correlation with the l2 norm of input signals.
        eps (float, optional): machine epsilon used for numerical stabilization when normalization is used.

    Returns:
        out (:class:`torch.Tensor`): cross correlation between the two multi-channel signals.
            Shape: :math:`(batch, mic\\_channels, seq\\_len\\_ref - seq\\_len\\_input + 1)`.

    """
    assert inp.size(0) == ref.size(0), 'ref and inp signals should have same batch size.'
    assert inp.size(2) >= ref.size(2), 'Input signal should be shorter than the ref signal.'
    inp = inp.permute(1, 0, 2).contiguous()
    ref = ref.permute(1, 0, 2).contiguous()
    bsz = inp.size(1)
    inp_mics = inp.size(0)
    if ref.size(0) > inp.size(0):
        inp = inp.expand(ref.size(0), inp.size(1), inp.size(2)).contiguous()
        inp_mics = ref.size(0)
    elif ref.size(0) < inp.size(0):
        ref = ref.expand(inp.size(0), ref.size(1), ref.size(2)).contiguous()
    out = F.conv1d(inp.view(1, -1, inp.size(2)), ref.view(-1, 1, ref.size(2)), groups=inp_mics * bsz)
    if normalized:
        inp_norm = F.conv1d(inp.view(1, -1, inp.size(2)).pow(2), torch.ones(inp.size(0) * inp.size(1), 1, ref.size(2)).type(inp.type()), groups=inp_mics * bsz)
        inp_norm = inp_norm.sqrt() + eps
        ref_norm = ref.norm(2, dim=2).view(1, -1, 1) + eps
        out = out / (inp_norm * ref_norm)
    return out.view(inp_mics, bsz, -1).permute(1, 0, 2).contiguous()


class FasNetTAC(BaseModel):
    """FasNetTAC separation model with optional Transform-Average-Concatenate (TAC) module[1].

    Args:
        n_src (int): Maximum number of sources the model can separate.
        enc_dim (int, optional): Length of analysis filter. Defaults to 64.
        feature_dim (int, optional): Size of hidden representation in DPRNN blocks after bottleneck.
            Defaults to 64.
        hidden_dim (int, optional): Number of neurons in the RNNs cell state in DPRNN blocks.
            Defaults to 128.
        n_layers (int, optional): Number of DPRNN blocks. Default to 4.
        window_ms (int, optional): Beamformer window_length in milliseconds. Defaults to 4.
        stride (int, optional): Stride for Beamforming windows. Defaults to window_ms // 2.
        context_ms (int, optional): Context for each Beamforming window. Defaults to 16.
            Effective window is 2*context_ms+window_ms.
        sample_rate (int, optional): Samplerate of input signal.
        tac_hidden_dim (int, optional): Size for TAC module hidden dimensions. Default to 384 neurons.
        norm_type (str, optional): Normalization layer used. Default is Layer Normalization.
        chunk_size (int, optional): Chunk size used for dual-path processing in DPRNN blocks.
            Default to 50 samples.
        hop_size (int, optional): Hop-size used for dual-path processing in DPRNN blocks.
            Default to `chunk_size // 2` (50% overlap).
        bidirectional (bool, optional):  True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional):  Type of RNN used. Choose between ``'RNN'``, ``'LSTM'`` and ``'GRU'``.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        use_tac (bool, optional): whether to use Transform-Average-Concatenate for inter-mic-channels
            communication. Defaults to True.

    References
        [1] Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel
        speech separation." ICASSP 2020.
    """

    def __init__(self, n_src, enc_dim=64, feature_dim=64, hidden_dim=128, n_layers=4, window_ms=4, stride=None, context_ms=16, sample_rate=16000, tac_hidden_dim=384, norm_type='gLN', chunk_size=50, hop_size=25, bidirectional=True, rnn_type='LSTM', dropout=0.0, use_tac=True):
        super().__init__(sample_rate=sample_rate, in_channels=None)
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_src = n_src
        assert window_ms % 2 == 0, 'Window length should be even'
        self.window_ms = window_ms
        self.context_ms = context_ms
        self.window = int(self.sample_rate * window_ms / 1000)
        self.context = int(self.sample_rate * context_ms / 1000)
        if not stride:
            self.stride = self.window // 2
        else:
            self.stride = int(self.sample_rate * stride / 1000)
        self.filter_dim = self.context * 2 + 1
        self.output_dim = self.context * 2 + 1
        self.tac_hidden_dim = tac_hidden_dim
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.use_tac = use_tac
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.window, bias=False)
        self.enc_LN = norms.get(norm_type)(self.enc_dim)
        self.bottleneck = nn.Conv1d(self.filter_dim + self.enc_dim, self.feature_dim, 1, bias=False)
        self.DPRNN_TAC = nn.ModuleList([])
        for i in range(self.n_layers):
            tmp = nn.ModuleList([DPRNNBlock(self.feature_dim, self.hidden_dim, norm_type, bidirectional, rnn_type, dropout=dropout)])
            if self.use_tac:
                tmp.append(TAC(self.feature_dim, tac_hidden_dim, norm_type=norm_type))
            self.DPRNN_TAC.append(tmp)
        self.conv_2D = nn.Sequential(nn.PReLU(), nn.Conv2d(self.feature_dim, self.n_src * self.feature_dim, 1))
        self.tanh = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid())

    @staticmethod
    def windowing_with_context(x, window, context):
        batch_size, nmic, nsample = x.shape
        unfolded = F.unfold(x.unsqueeze(-1), kernel_size=(window + 2 * context, 1), padding=(context + window, 0), stride=(window // 2, 1))
        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size, nmic, window + 2 * context, n_chunks)
        return unfolded[:, :, context:context + window].transpose(2, -1), unfolded.transpose(2, -1)

    def forward(self, x, valid_mics=None):
        """
        Args:
            x: (:class:`torch.Tensor`): multi-channel input signal. Shape: :math:`(batch, mic\\_channels, samples)`.
            valid_mics: (:class:`torch.LongTensor`): tensor containing effective number of microphones on each batch.
                Batches can be composed of examples coming from arrays with a different
                number of microphones and thus the ``mic_channels`` dimension is padded.
                E.g. torch.tensor([4, 3]) means first example has 4 channels and the second 3.
                Shape: :math`(batch)`.

        Returns:
            bf_signal (:class:`torch.Tensor`): beamformed signal with shape :math:`(batch, n\\_src, samples)`.
        """
        if valid_mics is None:
            valid_mics = torch.LongTensor([x.shape[1]] * x.shape[0])
        n_samples = x.size(-1)
        all_seg, all_mic_context = self.windowing_with_context(x, self.window, self.context)
        batch_size, n_mics, seq_length, feats = all_mic_context.size()
        enc_output = self.encoder(all_mic_context.reshape(batch_size * n_mics * seq_length, 1, feats)).reshape(batch_size * n_mics, seq_length, self.enc_dim).transpose(1, 2).contiguous()
        enc_output = self.enc_LN(enc_output).reshape(batch_size, n_mics, self.enc_dim, seq_length)
        ref_seg = all_seg[:, 0].reshape(batch_size * seq_length, self.window).unsqueeze(1)
        all_context = all_mic_context.transpose(1, 2).reshape(batch_size * seq_length, n_mics, self.context * 2 + self.window)
        all_cos_sim = xcorr(all_context, ref_seg)
        all_cos_sim = all_cos_sim.reshape(batch_size, seq_length, n_mics, self.context * 2 + 1).permute(0, 2, 3, 1).contiguous()
        input_feature = torch.cat([enc_output, all_cos_sim], 2)
        input_feature = self.bottleneck(input_feature.reshape(batch_size * n_mics, -1, seq_length))
        unfolded = F.unfold(input_feature.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size * n_mics, self.feature_dim, self.chunk_size, n_chunks)
        for i in range(self.n_layers):
            dprnn = self.DPRNN_TAC[i][0]
            unfolded = dprnn(unfolded)
            if self.use_tac:
                b, ch, chunk_size, n_chunks = unfolded.size()
                tac = self.DPRNN_TAC[i][1]
                unfolded = unfolded.reshape(-1, n_mics, ch, chunk_size, n_chunks)
                unfolded = tac(unfolded, valid_mics).reshape(batch_size * n_mics, self.feature_dim, self.chunk_size, n_chunks)
        unfolded = self.conv_2D(unfolded).reshape(batch_size * n_mics * self.n_src, self.feature_dim * self.chunk_size, n_chunks)
        folded = F.fold(unfolded, (seq_length, 1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        folded = folded.squeeze(-1) / (self.chunk_size / self.hop_size)
        folded = self.tanh(folded) * self.gate(folded)
        folded = folded.view(batch_size, n_mics, self.n_src, -1, seq_length)
        all_mic_context = all_mic_context.unsqueeze(2).repeat(1, 1, self.n_src, 1, 1)
        all_bf_output = F.conv1d(all_mic_context.view(1, -1, self.context * 2 + self.window), folded.transpose(3, -1).contiguous().view(-1, 1, self.filter_dim), groups=batch_size * n_mics * self.n_src * seq_length)
        all_bf_output = all_bf_output.view(batch_size, n_mics, self.n_src, seq_length, self.window)
        all_bf_output = F.fold(all_bf_output.reshape(batch_size * n_mics * self.n_src, seq_length, self.window).transpose(1, -1), (n_samples, 1), kernel_size=(self.window, 1), padding=(self.window, 0), stride=(self.window // 2, 1))
        bf_signal = all_bf_output.reshape(batch_size, n_mics, self.n_src, n_samples)
        if valid_mics.max() == 0:
            bf_signal = bf_signal.mean(1)
        else:
            bf_signal = [bf_signal[b, :valid_mics[b]].mean(0).unsqueeze(0) for b in range(batch_size)]
            bf_signal = torch.cat(bf_signal, 0)
        return bf_signal

    def get_model_args(self):
        config = {'n_src': self.n_src, 'enc_dim': self.enc_dim, 'feature_dim': self.feature_dim, 'hidden_dim': self.hidden_dim, 'n_layers': self.n_layers, 'window_ms': self.window_ms, 'stride': self.stride, 'context_ms': self.context_ms, 'sample_rate': self.sample_rate, 'tac_hidden_dim': self.tac_hidden_dim, 'norm_type': self.norm_type, 'chunk_size': self.chunk_size, 'hop_size': self.hop_size, 'bidirectional': self.bidirectional, 'rnn_type': self.rnn_type, 'dropout': self.dropout, 'use_tac': self.use_tac}
        return config


class _GatedEncoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.filterbank = encoder.filterbank
        self.sample_rate = getattr(encoder.filterbank, 'sample_rate', None)
        self.encoder_relu = encoder
        self.encoder_sig = deepcopy(encoder)

    def forward(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


class LSTMTasNet(BaseEncoderMaskerDecoder):
    """TasNet separation model, as described in [1].

    Args:
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        n_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1]: Yi Luo et al. "Real-time Single-channel Dereverberation and Separation
          with Time-domain Audio Separation Network", Interspeech 2018
    """

    def __init__(self, n_src, out_chan=None, rnn_type='lstm', n_layers=4, hid_size=512, dropout=0.3, mask_act='sigmoid', bidirectional=True, in_chan=None, fb_name='free', n_filters=64, kernel_size=16, stride=8, encoder_activation=None, sample_rate=8000, **fb_kwargs):
        encoder, decoder = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, sample_rate=sample_rate, **fb_kwargs)
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, f'Number of filterbank output channels and number of input channels should be the same. Received {n_feats} and {in_chan}'
        encoder = _GatedEncoder(encoder)
        masker = LSTMMasker(n_feats, n_src, out_chan=out_chan, hid_size=hid_size, mask_act=mask_act, bidirectional=bidirectional, rnn_type=rnn_type, n_layers=n_layers, dropout=dropout)
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


@script_if_tracing
def pad(x, lcm: int):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padding = torch.zeros(list(appropriate_shape[:-1]) + [lcm - values_to_pad], dtype=x.dtype, device=x.device)
        padded_x = torch.cat([x, padding], dim=-1)
        return padded_x
    return x


class _Padder(nn.Module):

    def __init__(self, encoder, upsampling_depth=4, kernel_size=21):
        super().__init__()
        self.encoder = encoder
        self.upsampling_depth = upsampling_depth
        self.kernel_size = kernel_size
        self.lcm = abs(self.kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(self.kernel_size // 2, 2 ** self.upsampling_depth)
        self.filterbank = self.encoder.filterbank
        self.sample_rate = getattr(self.encoder.filterbank, 'sample_rate', None)

    def forward(self, x):
        x = pad(x, self.lcm)
        return self.encoder(x)


class SuDORMRFNet(BaseEncoderMaskerDecoder):
    """SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
          Tzinis et al. MLSP 2020.
    """

    def __init__(self, n_src, bn_chan=128, num_blocks=16, upsampling_depth=4, mask_act='softmax', in_chan=None, fb_name='free', kernel_size=21, n_filters=512, stride=None, sample_rate=8000, **fb_kwargs):
        stride = kernel_size // 2 if not stride else stride
        enc, dec = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=kernel_size // 2, sample_rate=sample_rate, padding=kernel_size // 2, output_padding=kernel_size // 2 - 1, **fb_kwargs)
        n_feats = enc.n_feats_out
        enc = _Padder(enc, upsampling_depth=upsampling_depth, kernel_size=kernel_size)
        if in_chan is not None:
            assert in_chan == n_feats, f'Number of filterbank output channels and number of input channels should be the same. Received {n_feats} and {in_chan}'
        masker = SuDORMRF(n_feats, n_src, bn_chan=bn_chan, num_blocks=num_blocks, upsampling_depth=upsampling_depth, mask_act=mask_act)
        super().__init__(enc, masker, dec, encoder_activation='relu')


class SuDORMRFImprovedNet(BaseEncoderMaskerDecoder):
    """Improved SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
          Tzinis et al. MLSP 2020.
    """

    def __init__(self, n_src, bn_chan=128, num_blocks=16, upsampling_depth=4, mask_act='relu', in_chan=None, fb_name='free', kernel_size=21, n_filters=512, stride=None, sample_rate=8000, **fb_kwargs):
        stride = kernel_size // 2 if not stride else stride
        enc, dec = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, sample_rate=sample_rate, padding=kernel_size // 2, output_padding=kernel_size // 2 - 1, **fb_kwargs)
        n_feats = enc.n_feats_out
        enc = _Padder(enc, upsampling_depth=upsampling_depth, kernel_size=kernel_size)
        if in_chan is not None:
            assert in_chan == n_feats, f'Number of filterbank output channels and number of input channels should be the same. Received {n_feats} and {in_chan}'
        masker = SuDORMRFImproved(n_feats, n_src, bn_chan=bn_chan, num_blocks=num_blocks, upsampling_depth=upsampling_depth, mask_act=mask_act)
        super().__init__(enc, masker, dec, encoder_activation=None)


class _ISTFT(nn.Module):

    def __init__(self, window, n_fft=4096, hop_length=1024, center=True):
        super(_ISTFT, self).__init__()
        self.window = window
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def forward(self, spec, ang):
        sources, bsize, channels, fbins, frames = spec.shape
        x_r = spec * torch.cos(ang)
        x_i = spec * torch.sin(ang)
        x = torch.stack([x_r, x_i], dim=-1)
        x = x.view(sources * bsize * channels, fbins, frames, 2)
        wav = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=self.center)
        wav = wav.view(sources, bsize, channels, wav.shape[-1])
        return wav


class _InstrumentBackboneDec(nn.Module):
    """Decoder structure that maps output of LSTM layers to
    magnitude estimate of an instrument.

    Args:
        nb_output_bins (int): Number of frequency bins of the instrument estimate.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): Number of output bins depending on STFT size.
            It is generally calculated ``(STFT size) // 2 + 1''.
    """

    def __init__(self, nb_output_bins, hidden_size=512, nb_channels=2):
        super().__init__()
        self.nb_output_bins = nb_output_bins
        self.dec = nn.Sequential(Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False), BatchNorm1d(hidden_size), nn.ReLU(), Linear(in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False), BatchNorm1d(self.nb_output_bins * nb_channels))

    def forward(self, x, shapes):
        nb_frames, nb_samples, nb_channels, _ = shapes
        x = self.dec(x.reshape(-1, x.shape[-1]))
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        return x


class _InstrumentBackboneEnc(nn.Module):
    """Encoder structure that maps the mixture magnitude spectrogram to
    smaller-sized features which are the input for the LSTM layers.

    Args:
        nb_bins (int): Number of frequency bins of the mixture.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model
            (1 for mono (spectral downmix is applied,) 2 for stereo).
    """

    def __init__(self, nb_bins, hidden_size=512, nb_channels=2):
        super().__init__()
        self.max_bin = nb_bins
        self.hidden_size = hidden_size
        self.enc = nn.Sequential(Linear(self.max_bin * nb_channels, hidden_size, bias=False), BatchNorm1d(hidden_size))

    def forward(self, x, shapes):
        nb_frames, nb_samples, nb_channels, _ = shapes
        x = self.enc(x.reshape(-1, nb_channels * self.max_bin))
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        return x


class _STFT(nn.Module):

    def __init__(self, window_length, n_fft=4096, n_hop=1024, center=True):
        super(_STFT, self).__init__()
        self.window = Parameter(torch.hann_window(window_length), requires_grad=False)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        nb_samples, nb_channels, nb_timesteps = x.size()
        x = x.reshape(nb_samples * nb_channels, -1)
        stft_f = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=self.center, normalized=False, onesided=True, pad_mode='reflect', return_complex=False)
        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2)
        return stft_f


class _Spectrogram(nn.Module):

    def __init__(self, spec_power=1, mono=True):
        super(_Spectrogram, self).__init__()
        self.spec_power = spec_power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram and the corresponding phase
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        phase = stft_f.detach().clone()
        phase = torch.atan2(phase[Ellipsis, 1], phase[Ellipsis, 0])
        stft_f = stft_f.transpose(2, 3)
        stft_f = stft_f.pow(2).sum(-1).pow(self.spec_power / 2.0)
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
            phase = torch.mean(phase, 1, keepdim=True)
        return [stft_f.permute(2, 0, 1, 3), phase]


class XUMX(BaseModel):
    """CrossNet-Open-Unmix (X-UMX) for Music Source Separation introduced in [1].
        There are two notable contributions with no effect on inference:
            a) Multi Domain Losses
                - Considering not only spectrograms but also time signals
            b) Combination Scheme
                - Considering possible combinations of output instruments
        When starting to train X-UMX, you can optionally use the above by setting
        ``loss_use_multidomain'' and ``loss_combine_sources'' which are both set in conf.yml.

    Args:
        sources (list): The list of instruments, e.g., ["bass", "drums", "vocals"],
            defined in conf.yml.
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        sample_rate (int): sampling rate of input wavs
        nb_layers (int): Number of (B)LSTM layers in network.
        input_mean (torch.tensor): Mean for each frequency bin calculated
            in advance to normalize the mixture magnitude spectrogram.
        input_scale (torch.tensor): Standard deviation for each frequency bin
            calculated in advance to normalize the mixture magnitude spectrogram.
        max_bin (int): Maximum frequency bin index of the mixture that X-UMX
            should consider. Set to None to use all frequency bins.
        bidirectional (bool): whether we use LSTM or BLSTM.
        spec_power (int): Exponent for spectrogram calculation.
        return_time_signals (bool): Set to true if you are using a time-domain
            loss., i.e., applies ISTFT. If you select ``MDL=True'' via
            conf.yml, this is set as True.

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(self, sources, window_length=4096, in_chan=4096, n_hop=1024, hidden_size=512, nb_channels=2, sample_rate=44100, nb_layers=3, input_mean=None, input_scale=None, max_bin=None, bidirectional=True, spec_power=1, return_time_signals=False):
        super().__init__(sample_rate)
        self.window_length = window_length
        self.in_chan = in_chan
        self.n_hop = n_hop
        self.sources = sources
        self._return_time_signals = return_time_signals
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.bidirectional = bidirectional
        self.nb_output_bins = in_chan // 2 + 1
        if max_bin:
            self.max_bin = max_bin
        else:
            self.max_bin = self.nb_output_bins
        self.hidden_size = hidden_size
        self.spec_power = spec_power
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[:self.max_bin]).float()
        else:
            input_mean = torch.zeros(self.max_bin)
        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[:self.max_bin]).float()
        else:
            input_scale = torch.ones(self.max_bin)
        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=nb_channels == 1)
        self.encoder = nn.Sequential(stft, spec)
        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
        src_enc = {}
        src_lstm = {}
        src_dec = {}
        mean_scale = {}
        for src in sources:
            src_enc[src] = _InstrumentBackboneEnc(nb_bins=self.max_bin, hidden_size=hidden_size, nb_channels=nb_channels)
            src_lstm[src] = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=nb_layers, bidirectional=bidirectional, batch_first=False, dropout=0.4)
            src_dec[src] = _InstrumentBackboneDec(nb_output_bins=self.nb_output_bins, hidden_size=hidden_size, nb_channels=nb_channels)
            mean_scale['input_mean_{}'.format(src)] = Parameter(input_mean.clone())
            mean_scale['input_scale_{}'.format(src)] = Parameter(input_scale.clone())
            mean_scale['output_mean_{}'.format(src)] = Parameter(torch.ones(self.nb_output_bins).float())
            mean_scale['output_scale_{}'.format(src)] = Parameter(torch.ones(self.nb_output_bins).float())
        self.layer_enc = nn.ModuleDict(src_enc)
        self.layer_lstm = nn.ModuleDict(src_lstm)
        self.layer_dec = nn.ModuleDict(src_dec)
        self.mean_scale = nn.ParameterDict(mean_scale)
        self.decoder = _ISTFT(window=stft.window, n_fft=in_chan, hop_length=n_hop, center=True)

    def forward(self, wav):
        """Model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            masked_mixture (torch.Tensor): estimated spectrograms masked by
                X-UMX's output of shape $(sources, frames, batch_size, channels, bins)$
            time_signals (torch.Tensor): estimated time signals of shape $(sources, batch_size, channels, time_length)$ if `return_time_signals` is `True`
        """
        mixture, ang = self.encoder(wav)
        est_masks = self.forward_masker(mixture.clone())
        masked_mixture = self.apply_masks(mixture, est_masks)
        if self._return_time_signals:
            spec = masked_mixture.permute(0, 2, 3, 4, 1)
            time_signals = self.decoder(spec, ang)
        else:
            time_signals = None
        return masked_mixture, time_signals

    def forward_masker(self, input_spec):
        shapes = input_spec.data.shape
        x = input_spec[..., :self.max_bin]
        inputs = [x]
        for i in range(1, len(self.sources)):
            inputs.append(x.clone())
        for i, src in enumerate(self.sources):
            inputs[i] += self.mean_scale['input_mean_{}'.format(src)]
            inputs[i] *= self.mean_scale['input_scale_{}'.format(src)]
            inputs[i] = self.layer_enc[src](inputs[i], shapes)
        cross_1 = sum(inputs) / len(self.sources)
        cross_2 = 0.0
        for i, src in enumerate(self.sources):
            tmp_lstm_out = self.layer_lstm[src](cross_1)
            cross_2 += torch.cat([inputs[i], tmp_lstm_out[0]], -1)
        cross_2 /= len(self.sources)
        mask_list = []
        for src in self.sources:
            x_tmp = self.layer_dec[src](cross_2, shapes)
            x_tmp *= self.mean_scale['output_scale_{}'.format(src)]
            x_tmp += self.mean_scale['output_mean_{}'.format(src)]
            mask_list.append(F.relu(x_tmp))
        est_masks = torch.stack(mask_list, dim=0)
        return est_masks

    def apply_masks(self, mixture, est_masks):
        masked_tf_rep = torch.stack([(mixture * est_masks[i]) for i in range(len(self.sources))])
        return masked_tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = {'window_length': self.window_length, 'in_chan': self.in_chan, 'n_hop': self.n_hop, 'sample_rate': self.sample_rate}
        net_config = {'sources': self.sources, 'hidden_size': self.hidden_size, 'nb_channels': self.nb_channels, 'input_mean': None, 'input_scale': None, 'max_bin': self.max_bin, 'nb_layers': self.nb_layers, 'bidirectional': self.bidirectional, 'spec_power': self.spec_power, 'return_time_signals': False}
        model_args = {**fb_config, **net_config}
        return model_args


class Audio_Model(nn.Module):

    def __init__(self, last_shape=8):
        super(Audio_Model, self).__init__()
        self.conv1 = nn.Conv2d(2, 96, kernel_size=(1, 7), padding=self.get_padding((1, 7), (1, 1)), dilation=(1, 1))
        self.conv2 = nn.Conv2d(96, 96, kernel_size=(7, 1), padding=self.get_padding((7, 1), (1, 1)), dilation=(1, 1))
        self.conv3 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (1, 1)), dilation=(1, 1))
        self.conv4 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (2, 1)), dilation=(2, 1))
        self.conv5 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (4, 1)), dilation=(4, 1))
        self.conv6 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (8, 1)), dilation=(8, 1))
        self.conv7 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (16, 1)), dilation=(16, 1))
        self.conv8 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (32, 1)), dilation=(32, 1))
        self.conv9 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (1, 1)), dilation=(1, 1))
        self.conv10 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (2, 2)), dilation=(2, 2))
        self.conv11 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (4, 4)), dilation=(4, 4))
        self.conv12 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (8, 8)), dilation=(8, 8))
        self.conv13 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (16, 16)), dilation=(16, 16))
        self.conv14 = nn.Conv2d(96, 96, kernel_size=(5, 5), padding=self.get_padding((5, 5), (32, 32)), dilation=(32, 32))
        self.conv15 = nn.Conv2d(96, last_shape, kernel_size=(1, 1), padding=self.get_padding((1, 1), (1, 1)), dilation=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(96)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.batch_norm4 = nn.BatchNorm2d(96)
        self.batch_norm5 = nn.BatchNorm2d(96)
        self.batch_norm6 = nn.BatchNorm2d(96)
        self.batch_norm7 = nn.BatchNorm2d(96)
        self.batch_norm8 = nn.BatchNorm2d(96)
        self.batch_norm9 = nn.BatchNorm2d(96)
        self.batch_norm10 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm12 = nn.BatchNorm2d(96)
        self.batch_norm13 = nn.BatchNorm2d(96)
        self.batch_norm14 = nn.BatchNorm2d(96)
        self.batch_norm15 = nn.BatchNorm2d(last_shape)

    def get_padding(self, kernel_size, dilation):
        padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2
        return padding

    def forward(self, input_audio):
        output_layer = F.relu(self.batch_norm1(self.conv1(input_audio)))
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))
        output_layer = F.relu(self.batch_norm7(self.conv7(output_layer)))
        output_layer = F.relu(self.batch_norm8(self.conv8(output_layer)))
        output_layer = F.relu(self.batch_norm9(self.conv9(output_layer)))
        output_layer = F.relu(self.batch_norm10(self.conv10(output_layer)))
        output_layer = F.relu(self.batch_norm11(self.conv11(output_layer)))
        output_layer = F.relu(self.batch_norm12(self.conv12(output_layer)))
        output_layer = F.relu(self.batch_norm13(self.conv13(output_layer)))
        output_layer = F.relu(self.batch_norm14(self.conv14(output_layer)))
        output_layer = F.relu(self.batch_norm15(self.conv15(output_layer)))
        batch_size = output_layer.size(0)
        height = output_layer.size(2)
        output_layer = output_layer.transpose(-1, -2).reshape((batch_size, -1, height, 1))
        return output_layer


class Video_Model(nn.Module):

    def __init__(self, last_shape=256):
        super(Video_Model, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(7, 1), padding=self.get_padding((7, 1), (1, 1)), dilation=(1, 1))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=self.get_padding((5, 1), (1, 1)), dilation=(1, 1))
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=self.get_padding((5, 1), (2, 1)), dilation=(2, 1))
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=self.get_padding((5, 1), (4, 1)), dilation=(4, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=self.get_padding((5, 1), (8, 1)), dilation=(8, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=self.get_padding((5, 1), (16, 1)), dilation=(16, 1))
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(last_shape)

    def get_padding(self, kernel_size, dilation):
        padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2
        return padding

    def forward(self, input_video):
        if len(input_video.shape) == 3:
            input_video = input_video.unsqueeze(1)
        input_video = torch.transpose(input_video, 1, 3)
        output_layer = F.relu(self.batch_norm1(self.conv1(input_video)))
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))
        output_layer = nn.functional.interpolate(output_layer, size=(298, 1), mode='nearest')
        return output_layer


def cRM_tanh_recover(O, K=10, C=0.1):
    """CRM tanh recover.

    Args:
        O (torch.Tensor): predicted compressed crm.
        K (torch.Tensor): parameter to control the compression.
        C (torch.Tensor): parameter to control the compression.

    Returns:
        M (torch.Tensor): uncompressed crm.

    """
    numerator = K - O
    denominator = K + O
    M = -(1.0 / C * torch.log(numerator / denominator))
    return M


def fast_icRM(Y, crm, K=10, C=0.1):
    """fast iCRM.

    Args:
        Y (torch.Tensor): mixed/noised stft.
        crm (torch.Tensor): DNN output of compressed crm.
        K (torch.Tensor): parameter to control the compression.
        C (torch.Tensor): parameter to control the compression.

    Returns:
        S (torch.Tensor): clean stft.

    """
    M = cRM_tanh_recover(crm, K, C)
    S = torch.zeros(M.shape)
    S[:, 0, ...] = M[:, 0, ...] * Y[:, 0, ...] - M[:, 1, ...] * Y[:, 1, ...]
    S[:, 1, ...] = M[:, 0, ...] * Y[:, 1, ...] + M[:, 1, ...] * Y[:, 0, ...]
    return S


class Audio_Visual_Fusion(nn.Module):
    """Audio Visual Speech Separation model as described in [1].
    All default values are the same as paper.

        Args:
            num_person (int): total number of persons (as i/o).
            device (torch.Device): device used to return the final tensor.
            audio_last_shape (int): relevant last shape for tensor in audio network.
            video_last_shape (int): relevant last shape for tensor in video network.
            input_spectrogram_shape (tuple(int)): shape of input spectrogram.

        References:
            [1]: 'Looking to Listen at the Cocktail Party:
            A Speaker-Independent Audio-Visual Model for Speech Separation' Ephrat et. al
            https://arxiv.org/abs/1804.03619
    """

    def __init__(self, num_person=2, device=None, audio_last_shape=8, video_last_shape=256, input_spectrogram_shape=(298, 257, 2)):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        super(Audio_Visual_Fusion, self).__init__()
        self.num_person = num_person
        self.input_dim = audio_last_shape * input_spectrogram_shape[1] + video_last_shape * self.num_person
        self.audio_output = Audio_Model(last_shape=audio_last_shape)
        self.video_output = Video_Model(last_shape=video_last_shape)
        self.lstm = nn.LSTM(self.input_dim, 400, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(400, 600)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(600, 600)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(600, 600)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.complex_mask_layer = nn.Linear(600, 2 * 257 * self.num_person)
        torch.nn.init.xavier_uniform_(self.complex_mask_layer.weight)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(298)
        self.batch_norm2 = nn.BatchNorm1d(298)
        self.batch_norm3 = nn.BatchNorm1d(298)

    def forward(self, input_audio, input_video):
        input_audio = transforms.to_torchaudio(input_audio).transpose(1, 3)
        audio_out = self.audio_output(input_audio)
        AVFusion = [audio_out]
        for i in range(self.num_person):
            video_out = self.video_output(input_video[i])
            AVFusion.append(video_out)
        mixed_av = torch.cat(AVFusion, dim=1)
        mixed_av = mixed_av.squeeze(3)
        mixed_av = torch.transpose(mixed_av, 1, 2)
        self.lstm.flatten_parameters()
        mixed_av, (h, c) = self.lstm(mixed_av)
        mixed_av = mixed_av[..., :400] + mixed_av[..., 400:]
        mixed_av = self.batch_norm1(F.relu(self.fc1(mixed_av)))
        mixed_av = self.drop1(mixed_av)
        mixed_av = self.batch_norm2(F.relu(self.fc2(mixed_av)))
        mixed_av = self.drop2(mixed_av)
        mixed_av = self.batch_norm3(F.relu(self.fc3(mixed_av)))
        mixed_av = self.drop3(mixed_av)
        complex_mask = torch.sigmoid(self.complex_mask_layer(mixed_av))
        batch_size = complex_mask.size(0)
        complex_mask = complex_mask.view(batch_size, 298, 2, 257, self.num_person).transpose(1, 2)
        output_audio = torch.zeros_like(complex_mask, device=self.device)
        for i in range(self.num_person):
            output_audio[..., i] = fast_icRM(input_audio, complex_mask[..., i])
        output_audio = output_audio.permute(0, 4, 1, 3, 2).reshape(batch_size, self.num_person, 514, 298)
        return output_audio


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, n_src=2, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()
        self.n_src = n_src
        self.gamma = gamma

    def forward(self, input, target):
        sum_mtr = torch.zeros_like(input[:, 0, ...])
        for i in range(self.n_src):
            sum_mtr += (target[:, i, ...] - input[:, i, ...]) ** 2
            for j in range(self.n_src):
                if i != j:
                    sum_mtr -= self.gamma * (target[:, i, ...] - input[:, j, ...]) ** 2
        sum_mtr = torch.mean(sum_mtr.view(-1))
        return sum_mtr


class Combine_Loss(torch.nn.Module):
    """
    Loss function combines L1 loss and STOI loss to focus the
    separation on the vocal segment. This has relevance specially
    when ORIGINAL mixture is selected.
    """

    def __init__(self, alpha=0.5, sample_rate=16000):
        super(Combine_Loss, self).__init__()
        self.alpha = alpha
        self.loss_vocal = SingleSrcNegSTOI(sample_rate=sample_rate, extended=False, use_vad=False)
        self.loss_background = torch.nn.L1Loss()

    def forward(self, est_targets, targets):
        l_vocal = self.loss_vocal(est_targets[:, 0, :], targets[:, 0, :])
        l_back = self.loss_background(est_targets[:, 1, :], targets[:, 1, :])
        loss = (1 - self.alpha) * l_back + self.alpha * torch.mean(l_vocal)
        return loss


@script_if_tracing
def ebased_vad(mag_spec, th_db: int=40):
    """Compute energy-based VAD from a magnitude spectrogram (or equivalent).

    Args:
        mag_spec (torch.Tensor): the spectrogram to perform VAD on.
            Expected shape (batch, *, freq, time).
            The VAD mask will be computed independently for all the leading
            dimensions until the last two. Independent of the ordering of the
            last two dimensions.
        th_db (int): The threshold in dB from which a TF-bin is considered
            silent.

    Returns:
        :class:`torch.BoolTensor`, the VAD mask.


    Examples
        >>> import torch
        >>> mag_spec = torch.abs(torch.randn(10, 2, 65, 16))
        >>> batch_src_mask = ebased_vad(mag_spec)
    """
    log_mag = 20 * torch.log10(mag_spec)
    to_view = list(mag_spec.shape[:-2]) + [1, -1]
    max_log_mag = torch.max(log_mag.view(to_view), -1, keepdim=True)[0]
    return log_mag > max_log_mag - th_db


class Model(nn.Module):

    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        final_proj, mask_out = self.masker(mag(tf_rep))
        return final_proj, mask_out

    def separate(self, x):
        """Separate with mask-inference head, output waveforms"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        proj, mask_out = self.masker(mag(tf_rep))
        masked = apply_mag_mask(tf_rep.unsqueeze(1), mask_out)
        wavs = torch_utils.pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out

    def dc_head_separate(self, x):
        """Cluster embeddings to produce binary masks, output waveforms"""
        kmeans = KMeans(n_clusters=self.masker.n_src)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        mag_spec = mag(tf_rep)
        proj, mask_out = self.masker(mag_spec)
        active_bins = ebased_vad(mag_spec)
        active_proj = proj[active_bins.view(1, -1)]
        bin_clusters = kmeans.fit_predict(active_proj.cpu().data.numpy())
        est_mask_list = []
        for i in range(self.masker.n_src):
            mask = ~active_bins
            mask[active_bins] = torch.from_numpy(bin_clusters == i)
            est_mask_list.append(mask.float())
        est_masks = torch.stack(est_mask_list, dim=1)
        masked = apply_mag_mask(tf_rep, est_masks)
        wavs = pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out


class SimpleModel(nn.Module):
    """Simple recurrent model for the DNS challenge.

    Args:
        input_size (int): input size along the features dimension
        hidden_size (int): hidden size in the recurrent net
        output_size (int): output size, defaults to `:attr:` input_size
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can also
            be passed in lowercase letters.
        n_layers (int): Number of recurrent layers.
        dropout (float): dropout value between recurrent layers.
    """

    def __init__(self, input_size, hidden_size, output_size=None, rnn_type='gru', n_layers=3, dropout=0.3):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size if output_size is None else output_size
        self.output_size = output_size
        self.in_proj_layer = nn.Linear(input_size, hidden_size)
        self.residual_rec = StackedResidualRNN(rnn_type, hidden_size, n_layers=n_layers, dropout=dropout)
        self.out_proj_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Mask estimator's forward pass. Expects [batch, time, input_size]"""
        out_rec = self.residual_rec(torch.relu(self.in_proj_layer(x)))
        return torch.relu(self.out_proj_layer(out_rec))


def batch_matrix_norm(matrix, norm_order=2):
    """Normalize a matrix according to `norm_order`

    Args:
        matrix (torch.Tensor): Expected shape [batch, *]
        norm_order (int): Norm order.

    Returns:
        torch.Tensor, normed matrix of shape [batch]
    """
    keep_batch = list(range(1, matrix.ndim))
    return torch.norm(matrix, p=norm_order, dim=keep_batch) ** norm_order


def deep_clustering_loss(embedding, tgt_index, binary_mask=None):
    """Compute the deep clustering loss defined in [1].

    Args:
        embedding (torch.Tensor): Estimated embeddings.
            Expected shape  :math:`(batch, frequency * frame, embedding\\_dim)`.
        tgt_index (torch.Tensor): Dominating source index in each TF bin.
            Expected shape: :math:`(batch, frequency, frame)`.
        binary_mask (torch.Tensor): VAD in TF plane. Bool or Float.
            See asteroid.dsp.vad.ebased_vad.

    Returns:
         `torch.Tensor`. Deep clustering loss for every batch sample.

    Examples
        >>> import torch
        >>> from asteroid.losses.cluster import deep_clustering_loss
        >>> spk_cnt = 3
        >>> embedding = torch.randn(10, 5*400, 20)
        >>> targets = torch.LongTensor(10, 400, 5).random_(0, spk_cnt)
        >>> loss = deep_clustering_loss(embedding, targets)

    Reference
        [1] Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey
        "ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING"

    .. note::
        Be careful in viewing the embedding tensors. The target indices
        ``tgt_index`` are of shape :math:`(batch, freq, frames)`. Even if the embedding
        is of shape :math:`(batch, freq * frames, emb)`, the underlying view should be
        :math:`(batch, freq, frames, emb)` and not :math:`(batch, frames, freq, emb)`.
    """
    spk_cnt = len(tgt_index.unique())
    batch, bins, frames = tgt_index.shape
    if binary_mask is None:
        binary_mask = torch.ones(batch, bins * frames, 1)
    binary_mask = binary_mask.float()
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.view(batch, bins * frames, 1)
    binary_mask = binary_mask
    tgt_embedding = torch.zeros(batch, bins * frames, spk_cnt, device=tgt_index.device)
    tgt_embedding.scatter_(2, tgt_index.view(batch, bins * frames, 1), 1)
    tgt_embedding = tgt_embedding * binary_mask
    embedding = embedding * binary_mask
    est_proj = torch.einsum('ijk,ijl->ikl', embedding, embedding)
    true_proj = torch.einsum('ijk,ijl->ikl', tgt_embedding, tgt_embedding)
    true_est_proj = torch.einsum('ijk,ijl->ikl', embedding, tgt_embedding)
    cost = batch_matrix_norm(est_proj) + batch_matrix_norm(true_proj)
    cost = cost - 2 * batch_matrix_norm(true_est_proj)
    return cost / torch.sum(binary_mask, dim=[1, 2])


pairwise_mse = PairwiseMSE()


class ChimeraLoss(nn.Module):
    """Combines Deep clustering loss and mask inference loss for ChimeraNet.

    Args:
        alpha (float): loss weight. Total loss will be :
            `alpha` * dc_loss + (1 - `alpha`) * mask_mse_loss.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha >= 0, "Negative alpha values don't make sense."
        assert alpha <= 1, "Alpha values above 1 don't make sense."
        self.src_mse = PITLossWrapper(pairwise_mse, pit_from='pw_mtx')
        self.alpha = alpha

    def forward(self, est_embeddings, target_indices, est_src=None, target_src=None, mix_spec=None):
        """

        Args:
            est_embeddings (torch.Tensor): Estimated embedding from the DC head.
            target_indices (torch.Tensor): Target indices that'll be passed to
                the DC loss.
            est_src (torch.Tensor): Estimated magnitude spectrograms (or masks).
            target_src (torch.Tensor): Target magnitude spectrograms (or masks).
            mix_spec (torch.Tensor): The magnitude spectrogram of the mixture
                from which VAD will be computed. If None, no VAD is used.

        Returns:
            torch.Tensor, the total loss, averaged over the batch.
            dict with `dc_loss` and `pit_loss` keys, unweighted losses.
        """
        if self.alpha != 0 and (est_src is None or target_src is None):
            raise ValueError('Expected target and estimated spectrograms to compute the PIT loss, found None.')
        binary_mask = None
        if mix_spec is not None:
            binary_mask = ebased_vad(mix_spec)
        dc_loss = deep_clustering_loss(embedding=est_embeddings, tgt_index=target_indices, binary_mask=binary_mask)
        src_pit_loss = self.src_mse(est_src, target_src)
        tot = self.alpha * dc_loss.mean() + (1 - self.alpha) * src_pit_loss
        loss_dict = dict(dc_loss=dc_loss.mean(), pit_loss=src_pit_loss)
        return tot, loss_dict


singlesrc_mse = SingleSrcMSE()


def freq_domain_loss(s_hat, gt_spec, combination=True):
    """Calculate frequency-domain loss between estimated and reference spectrograms.
    MSE between target and estimated target spectrograms is adopted as frequency-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        estimated spectrograms
            (Sources, Freq. bins, Batch size, Channels, Frames)
        reference spectrograms
            (Freq. bins, Batch size, Sources x Channels, Frames)
        whether use combination or not (optional)
    Output:
        calculated frequency-domain loss
    """
    n_src = len(s_hat)
    idx_list = [i for i in range(n_src)]
    inferences = []
    refrences = []
    for i, s in enumerate(s_hat):
        inferences.append(s)
        refrences.append(gt_spec[..., 2 * i:2 * i + 2, :])
    assert inferences[0].shape == refrences[0].shape
    _loss_mse = 0.0
    cnt = 0.0
    for i in range(n_src):
        _loss_mse += singlesrc_mse(inferences[i], refrences[i]).mean()
        cnt += 1.0
    if combination:
        for c in range(2, n_src):
            patterns = list(itertools.combinations(idx_list, c))
            for indices in patterns:
                tmp_loss = singlesrc_mse(sum(itemgetter(*indices)(inferences)), sum(itemgetter(*indices)(refrences))).mean()
                _loss_mse += tmp_loss
                cnt += 1.0
    _loss_mse /= cnt
    return _loss_mse


def weighted_sdr(input, gt, mix, weighted=True, eps=1e-10):
    assert input.shape == gt.shape
    assert mix.shape == gt.shape
    ns = mix - gt
    ns_hat = mix - input
    if weighted:
        alpha_num = (gt * gt).sum(1, keepdims=True)
        alpha_denom = (gt * gt).sum(1, keepdims=True) + (ns * ns).sum(1, keepdims=True)
        alpha = alpha_num / (alpha_denom + eps)
    else:
        alpha = 0.5
    num_cln = (input * gt).sum(1, keepdims=True)
    denom_cln = torch.sqrt(eps + (input * input).sum(1, keepdims=True)) * torch.sqrt(eps + (gt * gt).sum(1, keepdims=True))
    sdr_cln = num_cln / (denom_cln + eps)
    num_noise = (ns * ns_hat).sum(1, keepdims=True)
    denom_noise = torch.sqrt(eps + (ns_hat * ns_hat).sum(1, keepdims=True)) * torch.sqrt(eps + (ns * ns).sum(1, keepdims=True))
    sdr_noise = num_noise / (denom_noise + eps)
    return torch.mean(-alpha * sdr_cln - (1.0 - alpha) * sdr_noise)


def time_domain_loss(mix, time_hat, gt_time, combination=True):
    """Calculate weighted time-domain loss between estimated and reference time signals.
    weighted SDR [1] between target and estimated target signals is adopted as time-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        mixture time signal
            (Batch size, Channels, Time Length (samples))
        estimated time signals
            (Sources, Batch size, Channels, Time Length (samples))
        reference time signals
            (Batch size, Sources x Channels, Time Length (samples))
        whether use combination or not (optional)
    Output:
        calculated time-domain loss

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    n_src, n_batch, n_channel, time_length = time_hat.shape
    idx_list = [i for i in range(n_src)]
    mix = mix[Ellipsis, :time_length]
    gt_time = gt_time[Ellipsis, :time_length]
    mix_ref = [mix]
    mix_ref.extend([gt_time[..., 2 * i:2 * i + 2, :] for i in range(n_src)])
    mix_ref = torch.stack(mix_ref)
    mix_ref = mix_ref.view(-1, time_length)
    time_hat = time_hat.view(n_batch * n_channel * time_hat.shape[0], time_hat.shape[-1])
    if combination:
        indices = []
        for c in range(2, n_src):
            indices.extend(list(itertools.combinations(idx_list, c)))
        for tr in indices:
            sp = [(n_batch * n_channel * (tr[i] + 1)) for i in range(len(tr))]
            ep = [(n_batch * n_channel * (tr[i] + 2)) for i in range(len(tr))]
            spi = [(n_batch * n_channel * tr[i]) for i in range(len(tr))]
            epi = [(n_batch * n_channel * (tr[i] + 1)) for i in range(len(tr))]
            tmp = sum([mix_ref[sp[i]:ep[i], ...].clone() for i in range(len(tr))])
            tmpi = sum([time_hat[spi[i]:epi[i], ...].clone() for i in range(len(tr))])
            mix_ref = torch.cat([mix_ref, tmp], dim=0)
            time_hat = torch.cat([time_hat, tmpi], dim=0)
        mix_t = mix_ref[:n_batch * n_channel, Ellipsis].repeat(n_src + len(indices), 1)
        refrences_t = mix_ref[n_batch * n_channel:, Ellipsis]
    else:
        mix_t = mix_ref[:n_batch * n_channel, Ellipsis].repeat(n_src, 1)
        refrences_t = mix_ref[n_batch * n_channel:, Ellipsis]
    _loss_sdr = weighted_sdr(time_hat, refrences_t, mix_t)
    return 1.0 + _loss_sdr


class MultiDomainLoss(_Loss):
    """A class for calculating loss functions of X-UMX.

    Args:
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        spec_power (int): Exponent for spectrogram calculation.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        loss_combine_sources (bool): Set to true if you are using the combination scheme
            proposed in [1]. If you select ``loss_combine_sources: yes'' via
            conf.yml, this is set as True.
        loss_use_multidomain (bool): Set to true if you are using a frequency- and time-domain
            losses collaboratively, i.e., Multi Domain Loss (MDL) proposed in [1].
            If you select ``loss_use_multidomain: yes'' via conf.yml, this is set as True.
        mix_coef (float): A mixing parameter for multi domain losses

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(self, window_length, in_chan, n_hop, spec_power, nb_channels, loss_combine_sources, loss_use_multidomain, mix_coef):
        super().__init__()
        self.transform = nn.Sequential(_STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop), _Spectrogram(spec_power=spec_power, mono=nb_channels == 1))
        self._combi = loss_combine_sources
        self._multi = loss_use_multidomain
        self.coef = mix_coef
        None
        if self._multi:
            None
        else:
            None
        self.cnt = 0

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)

        targets: (Batch, Source, Channels, TimeLen)
        """
        spec_hat = est_targets[0]
        time_hat = est_targets[1]
        n_batch, n_src, n_channel, time_length = targets.shape
        targets = targets.view(n_batch, n_src * n_channel, time_length)
        Y = self.transform(targets)[0]
        if self._multi:
            n_src = spec_hat.shape[0]
            mixture_t = sum([targets[:, 2 * i:2 * i + 2, ...] for i in range(n_src)])
            loss_f = freq_domain_loss(spec_hat, Y, combination=self._combi)
            loss_t = time_domain_loss(mixture_t, time_hat, targets, combination=self._combi)
            loss = float(self.coef) * loss_t + loss_f
        else:
            loss = freq_domain_loss(spec_hat, Y, combination=self._combi)
        return loss


class SeparableDilatedConv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1] without skip
        output. As used in the two step approach [2]. This block uses the
        groupnorm across features and also produces always a padded output.

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        dilation (int): Dilation of the depth-wise convolution.

    References:
        [1]: "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
             for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
             https://arxiv.org/abs/1809.07454
        [2]: Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self, in_chan=256, hid_chan=512, kernel_size=3, dilation=1):
        super(SeparableDilatedConv1DBlock, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(in_channels=in_chan, out_channels=hid_chan, kernel_size=1), nn.PReLU(), nn.GroupNorm(1, hid_chan, eps=1e-08), nn.Conv1d(in_channels=hid_chan, out_channels=hid_chan, kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation, groups=hid_chan), nn.PReLU(), nn.GroupNorm(1, hid_chan, eps=1e-08), nn.Conv1d(in_channels=hid_chan, out_channels=in_chan, kernel_size=1))

    def forward(self, x):
        """Input shape [batch, feats, seq]"""
        y = x.clone()
        return x + self.module(y)


class TwoStepTDCN(nn.Module):
    """
    A time-dilated convolutional network (TDCN) similar to the initial
    ConvTasNet architecture where the encoder and decoder have been
    pre-trained separately. The TwoStepTDCN infers masks directly on the
    latent space and works using an signal to distortion ratio (SDR) loss
    directly on the ideal latent masks.
    Adaptive basis encoder and decoder with inference of ideal masks.
    Copied from: https://github.com/etzinis/two_step_mask_learning/

    Args:
        pretrained_filterbank: A pretrained encoder decoder like the one
            implemented in asteroid_filterbanks.simple_adaptive
        n_sources (int, optional): Number of masks to estimate.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 4.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        kernel_size (int, optional): Kernel size in convolutional blocks.
            n_sources: The number of sources
    References:
        Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
        Smaragdis, P., "Two-Step Sound Source Separation:
        Training on Learned Latent Targets." In Acoustics, Speech
        and Signal Processing (ICASSP), 2020 IEEE International Conference.
        https://arxiv.org/abs/1910.09804
    """

    def __init__(self, pretrained_filterbank, bn_chan=256, hid_chan=512, kernel_size=3, n_blocks=8, n_repeats=4, n_sources=2):
        super(TwoStepTDCN, self).__init__()
        try:
            self.pretrained_filterbank = pretrained_filterbank
            self.encoder = self.pretrained_filterbank.mix_encoder
            self.decoder = self.pretrained_filterbank.decoder
            self.fbank_basis = self.encoder.conv.out_channels
            self.fbank_kernel_size = self.encoder.conv.kernel_size[0]
            self.encoder.conv.weight.requires_grad = False
            self.encoder.conv.bias.requires_grad = False
            self.decoder.deconv.weight.requires_grad = False
            self.decoder.deconv.bias.requires_grad = False
        except Exception as e:
            None
            raise ValueError('Could not load features form the pretrained adaptive filterbank.')
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.n_sources = n_sources
        self.ln_in = nn.BatchNorm1d(self.fbank_basis)
        self.l1 = nn.Conv1d(in_channels=self.fbank_basis, out_channels=self.bn_chan, kernel_size=1)
        self.separator = nn.Sequential(*[SeparableDilatedConv1DBlock(in_chan=self.bn_chan, hid_chan=self.hid_chan, kernel_size=self.kernel_size, dilation=2 ** d) for _ in range(self.n_blocks) for d in range(self.n_repeats)])
        self.mask_layer = nn.Conv2d(in_channels=1, out_channels=self.n_sources, kernel_size=(self.fbank_basis + 1, 1), padding=(self.fbank_basis - self.fbank_basis // 2, 0))
        if self.bn_chan != self.fbank_basis:
            self.out_reshape = nn.Conv1d(in_channels=self.bn_chan, out_channels=self.fbank_basis, kernel_size=1)
        self.ln_mask_in = nn.BatchNorm1d(self.fbank_basis)

    def forward(self, x):
        x = self.encoder(x)
        encoded_mixture = x.clone()
        x = self.ln_in(x)
        x = self.l1(x)
        x = self.separator(x)
        if self.bn_chan != self.fbank_basis:
            x = self.out_reshape(x)
        x = self.ln_mask_in(x)
        x = nn.functional.relu(x)
        x = self.mask_layer(x.unsqueeze(1))
        masks = nn.functional.softmax(x, dim=1)
        return masks * encoded_mixture.unsqueeze(1)

    def infer_source_signals(self, mixture_wav):
        adfe_sources = self.forward(mixture_wav)
        rec_wavs = self.decoder(adfe_sources.view(adfe_sources.shape[0], -1, adfe_sources.shape[-1]))
        return rec_wavs


class AdaptiveEncoder1D(nn.Module):
    """
    A 1D convolutional block that transforms signal in wave form into higher
    dimension.

    Args:
        input shape: [batch, 1, n_samples]
        output shape: [batch, freq_res, n_samples//sample_res]
        freq_res: number of output frequencies for the encoding convolution
        sample_res: int, length of the encoding filter
    """

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1, freq_res, sample_res, stride=sample_res // 2, padding=sample_res // 2)

    def forward(self, s):
        return F.relu(self.conv(s))


class AdaptiveDecoder1D(nn.Module):
    """A 1D deconvolutional block that transforms encoded representation
    into wave form.
    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]
    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    """

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_sources * freq_res, n_sources, sample_res, padding=sample_res // 2, stride=sample_res // 2, groups=n_sources, output_padding=sample_res // 2 - 1)

    def forward(self, x):
        return self.deconv(x)


class AdaptiveEncoderDecoder(nn.Module):
    """
    Adaptive basis encoder and decoder with inference of ideal masks.
    Copied from: https://github.com/etzinis/two_step_mask_learning/

    Args:
        freq_res: The number of frequency like representations
        sample_res: The number of samples in kernel 1D convolutions
        n_sources: The number of sources
    References:
        Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
        Smaragdis, P., "Two-Step Sound Source Separation:
        Training on Learned Latent Targets." In Acoustics, Speech
        and Signal Processing (ICASSP), 2020 IEEE International Conference.
        https://arxiv.org/abs/1910.09804
    """

    def __init__(self, freq_res=256, sample_res=21, n_sources=2):
        super().__init__()
        self.freq_res = freq_res
        self.sample_res = sample_res
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources

    def get_target_masks(self, clean_sources):
        """
        Get target masks for the given clean sources
        :param clean_sources: [batch, n_sources, time_samples]
        :return: Ideal masks for the given sources:
        [batch, n_sources, time_samples//(sample_res // 2)]
        """
        enc_mask_list = [self.mix_encoder(clean_sources[:, i, :].unsqueeze(1)) for i in range(self.n_sources)]
        total_mask = torch.stack(enc_mask_list, dim=1)
        return F.softmax(total_mask, dim=1)

    def reconstruct(self, mixture):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        return self.decoder(enc_mixture)

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

    def forward(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        recon_sources = self.decoder(s_recon_enc.view(s_recon_enc.shape[0], -1, s_recon_enc.shape[-1]))
        return recon_sources, enc_masks


class TasNet(nn.Module):
    """Some kind of TasNet, but not the original one
    Differences:
        - Overlap-add support (strided convolutions)
        - No frame-wise normalization on the wavs
        - GlobLN as bottleneck layer.
        - No skip connection.

    Args:
        fb_conf (dict): see local/conf.yml
        mask_conf (dict): see local/conf.yml
    """

    def __init__(self, fb_conf, mask_conf):
        super().__init__()
        self.n_src = mask_conf['n_src']
        self.n_filters = fb_conf['n_filters']
        self.encoder_sig = Encoder(FreeFB(**fb_conf))
        self.encoder_relu = Encoder(FreeFB(**fb_conf))
        self.decoder = Decoder(FreeFB(**fb_conf))
        self.bn_layer = GlobLN(fb_conf['n_filters'])
        self.masker = nn.Sequential(SingleRNN('lstm', fb_conf['n_filters'], hidden_size=mask_conf['n_units'], n_layers=mask_conf['n_layers'], bidirectional=True, dropout=mask_conf['dropout']), nn.Linear(2 * mask_conf['n_units'], self.n_src * self.n_filters), nn.Sigmoid())

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encode(x)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks
        return torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


class PairwiseNegSDR_Loss(_Loss):
    """
    Same as asteroid.losses.PairwiseNegSDR, but supports speaker number mismatch
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-08):
        super(PairwiseNegSDR_Loss, self).__init__()
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets):
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)
        if self.sdr_type in ['sisdr', 'sdsdr']:
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class Penalized_PIT_Wrapper(nn.Module):
    """
    Implementation of P-Si-SNR, as purposed in [1]
    References:
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(self, loss_func, penalty=30, perm_reduce=None):
        super().__init__()
        assert penalty > 0, 'penalty term should be positive'
        self.neg_penalty = -penalty
        self.perm_reduce = perm_reduce
        self.loss_func = loss_func

    def forward(self, est_targets, targets, **kwargs):
        """
        est_targets: torch.Tensor, $(est_nsrc, ...)$
        targets: torch.Tensor, $(gt_nsrc, ...)$
        """
        est_nsrc, T = est_targets.size()
        gt_nsrc = est_targets.size(0)
        pw_losses = self.loss_func(est_targets.unsqueeze(0), targets.unsqueeze(0)).squeeze(0)
        pwl = pw_losses.transpose(-1, -2)
        row, col = [torch.Tensor(x).long() for x in linear_sum_assignment(pwl.detach().cpu())]
        avg_neg_sdr = pwl[row, col].mean()
        p_si_snr = (-avg_neg_sdr * min(est_nsrc, gt_nsrc) + self.neg_penalty * abs(est_nsrc - gt_nsrc)) / max(est_nsrc, gt_nsrc)
        return p_si_snr


class DPRNN_MultiStage(nn.Module):
    """Implementation of the Dual-Path-RNN model,
    with multi-stage output, without Conv2D projection
    """

    def __init__(self, in_chan, bn_chan, hid_size, chunk_size, hop_size, n_repeats, norm_type, bidirectional, rnn_type, use_mulcat, num_layers, dropout):
        super(DPRNN_MultiStage, self).__init__()
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.net = nn.ModuleList([])
        for i in range(self.n_repeats):
            self.net.append(DPRNNBlock(bn_chan, hid_size, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type, use_mulcat=use_mulcat, num_layers=num_layers, dropout=dropout))

    def forward(self, mixture_w):
        """Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            list of (:class:`torch.Tensor`): Tensor of shape $(batch, bn_chan, chunk_size, n_chunks)
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        output_list = []
        for i in range(self.n_repeats):
            output = self.net[i](output)
            output_list.append(output)
        return output_list


class SingleDecoder(nn.Module):
    """
    Base decoder module, including the projection layer from (bn_chan) to (n_src * bn_chan).
    Takes a single example mask and encoding, outputs waveform
    """

    def __init__(self, kernel_size, stride, in_chan, n_src, bn_chan, chunk_size, hop_size, mask_act):
        super(SingleDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_src = n_src
        self.mask_act = mask_act
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, in_chan, 1, bias=False)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()
        _, self.trans_conv = make_enc_dec('free', kernel_size=kernel_size, stride=stride, n_filters=in_chan)

    def forward(self, output, mixture_w):
        """
        Args:
            output: LSTM output, Tensor of shape $(num_stages, bn_chan, chunk_size, n_chunks)$
            mixture_w: Encoder output, Tensor of shape $(num_stages, in_chan, nframes)
        outputs:
            est_wavs: Signal, Tensor of shape $(num_stages, n_src, T)
        """
        batch, bn_chan, chunk_size, n_chunks = output.size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.bn_chan == bn_chan
        assert self.in_chan == in_chan
        assert self.chunk_size == chunk_size
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(output.reshape(batch * self.n_src, to_unfold, n_chunks), (n_frames, 1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.reshape(batch, self.n_src, self.in_chan, n_frames)
        mixture_w = mixture_w.unsqueeze(1)
        source_w = est_mask * mixture_w
        source_w = source_w.reshape(batch * self.n_src, self.in_chan, n_frames)
        est_wavs = self.trans_conv(source_w)
        est_wavs = est_wavs.reshape(batch, self.n_src, -1)
        return est_wavs


class Decoder_Select(nn.Module):
    """Selects which SingleDecoder to use, as well as whether to use multiloss, as proposed in [1]
    References
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(self, kernel_size, stride, in_chan, n_srcs, bn_chan, chunk_size, hop_size, mask_act):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.n_srcs = n_srcs
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.mask_act = mask_act
        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.decoders = torch.nn.ModuleList()
        for n_src in n_srcs:
            self.decoders.append(SingleDecoder(kernel_size=kernel_size, stride=stride, in_chan=in_chan, n_src=n_src, bn_chan=bn_chan, chunk_size=chunk_size, hop_size=hop_size, mask_act=mask_act))
        self.selector = nn.Sequential(nn.Conv2d(bn_chan, in_chan, 1), nn.AdaptiveAvgPool2d(1), nn.ReLU(), nn.Conv2d(in_chan, len(n_srcs), 1))

    def forward(self, output_list, mixture_w, ground_truth):
        """Forward
        Args:
            output_list: list of $(batch, bn_chan, chunk_size, n_chunks)$
            mixture_w: torch.Tensor, $(batch, in_chan, n_frames)$
            ground_truth: None, or list of [B] ints, or Long Tensor of $(B)
                if None, use inferred number of speakers to determine output shape
        Output:
            output_wavs: torch.Tensor, $(batch, num_stages, max_spks, T)$
                where the speaker dimension is padded for examples with num_spks < max_spks
                if training, num_stages=n_repeats; otherwise, num_stages=1
            selector_output: output logits from selector module. torch.Tensor, $(batch, num_stages, num_decoders)$
        """
        batch, bn_chan, chunk_size, n_chunks = output_list[0].size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.chunk_size == chunk_size
        if not self.training:
            output_list = output_list[-1:]
        num_stages = len(output_list)
        output = torch.stack(output_list, 1).reshape(batch * num_stages, bn_chan, chunk_size, n_chunks)
        selector_output = self.selector(output).reshape(batch, num_stages, -1)
        output = output.reshape(batch, num_stages, bn_chan, chunk_size, n_chunks)
        mixture_w = mixture_w.unsqueeze(1).repeat(1, num_stages, 1, 1)
        if ground_truth is not None:
            decoder_selected = torch.LongTensor([self.n_src2idx[truth] for truth in ground_truth])
        else:
            assert num_stages == 1
            decoder_selected = selector_output.reshape(batch, -1).argmax(1)
        T = self.kernel_size + self.stride * (n_frames - 1)
        output_wavs = torch.zeros(batch, num_stages, max(self.n_srcs), T)
        for i in range(batch):
            output_wavs[i, :, :self.n_srcs[decoder_selected[i]], :] = self.decoders[decoder_selected[i]](output[i], mixture_w[i])
        return output_wavs, selector_output


pairwise_neg_sisdr = PairwiseNegSDR('sisdr')


class MultiDecoderDPRNN(BaseModel):
    """Multi-Decoder Dual-Path RNN as proposed in [1].

    Args:
        n_srcs (list of int): range of possible number of sources
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        kernel_size (int): Length of the filters.
        n_filters (int): Number of filters / Input dimension of the masker net.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.

    References
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(self, n_srcs, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type='gLN', mask_act='sigmoid', bidirectional=True, rnn_type='LSTM', num_layers=1, dropout=0, kernel_size=16, n_filters=64, stride=8, encoder_activation=None, use_mulcat=False, sample_rate=8000):
        super().__init__(sample_rate=sample_rate)
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or 'linear')()
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.encoder, _ = make_enc_dec('free', kernel_size=kernel_size, n_filters=n_filters, stride=stride)
        self.masker = DPRNN_MultiStage(in_chan=n_filters, bn_chan=bn_chan, hid_size=hid_size, chunk_size=chunk_size, hop_size=hop_size, n_repeats=n_repeats, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type, use_mulcat=use_mulcat, num_layers=num_layers, dropout=dropout)
        self.decoder_select = Decoder_Select(kernel_size=kernel_size, stride=stride, in_chan=n_filters, n_srcs=n_srcs, bn_chan=bn_chan, chunk_size=chunk_size, hop_size=hop_size, mask_act=mask_act)
        """
        Args:
            wav: 2D or 3D Tensor, Tensor of shape $(batch, T)$
            ground_truth: oracle number of speakers, None or list of $(batch)$ ints 
        Return:
            reconstructed: torch.Tensor, $(batch, num_stages, max_spks, T)$
                where max_spks is the maximum possible number of speakers.
                if training, num_stages=n_repeats; otherwise num_stages=0
            Speaker dimension is zero-padded for examples with num_spks < max_spks
        """

    def forward(self, wav, ground_truth=None):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks_list = self.masker(tf_rep)
        decoded, selector_output = self.decoder_select(est_masks_list, tf_rep, ground_truth=ground_truth)
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), _shape_reconstructed(selector_output, shape)

    def forward_wav(self, wav, slice_size=32000, *args, **kwargs):
        """Separation method for waveforms.
        Unfolds a full audio into slices, estimate
        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        Return:
            output_cat (torch.Tensor): concatenated output tensor.
                [num_spks, T]
        """
        assert not self.training, 'forward_wav is only used for test mode'
        T = wav.size(-1)
        if wav.ndim == 1:
            wav = wav.reshape(1, wav.size(0))
        assert wav.ndim == 2
        slice_stride = slice_size // 2
        T_padded = max(int(np.ceil(T / slice_stride)), 2) * slice_stride
        wav = F.pad(wav, (0, T_padded - T))
        slices = wav.unfold(dimension=-1, size=slice_size, step=slice_stride)
        slice_nb = slices.size(1)
        slices = slices.squeeze(0).unsqueeze(1)
        tf_rep = self.enc_activation(self.encoder(slices))
        est_masks_list = self.masker(tf_rep)
        selector_input = est_masks_list[-1]
        selector_output = self.decoder_select.selector(selector_input).reshape(slice_nb, -1)
        est_idx, _ = selector_output.argmax(-1).mode()
        est_spks = self.decoder_select.n_srcs[est_idx]
        output_wavs, _ = self.decoder_select(est_masks_list, tf_rep, ground_truth=[est_spks] * slice_nb)
        output_wavs = output_wavs.squeeze(1)[:, :est_spks, :]
        output_cat = output_wavs.new_zeros(est_spks, slice_nb * slice_size)
        output_cat[:, :slice_size] = output_wavs[0]
        start = slice_stride
        for i in range(1, slice_nb):
            end = start + slice_size
            overlap_prev = output_cat[:, start:start + slice_stride].unsqueeze(0)
            overlap_next = output_wavs[i:i + 1, :, :slice_stride]
            pw_losses = pairwise_neg_sisdr(overlap_next, overlap_prev)
            _, best_indices = PITLossWrapper.find_best_perm(pw_losses)
            reordered = PITLossWrapper.reorder_source(output_wavs[i:i + 1, :, :], best_indices)
            output_cat[:, start:start + slice_size] += reordered.squeeze(0)
            output_cat[:, start:start + slice_stride] /= 2
            start += slice_stride
        return output_cat[:, :T]


class WeightedPITLoss(nn.Module):
    """
    This loss has two components. One is the standard PIT loss, with Si-SDR summed(not mean, but sum) over each source
    under the best matching permutation. The other component is the classification loss, which is cross entropy for the
    speaker number classification head network.
    """

    def __init__(self, n_srcs, lamb=0.05):
        super().__init__()
        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.cce = nn.CrossEntropyLoss(reduction='none')
        self.lamb = lamb

    def forward(self, est_src, logits, src):
        """Forward
        Args:
            est_src: $(num_stages, n_src, T)
            logits: $(num_stages, num_decoders)
            src: $(n_src, T)
        """
        assert est_src.size()[1:] == src.size()
        num_stages, n_src, T = est_src.size()
        target_src = src.unsqueeze(0).repeat(num_stages, 1, 1)
        target_idx = self.n_src2idx[n_src]
        pw_losses = pairwise_neg_sisdr(est_src, target_src)
        sdr_loss, _ = PITLossWrapper.find_best_perm(pw_losses)
        pos_sdr = -sdr_loss[-1]
        cls_target = torch.LongTensor([target_idx] * num_stages)
        cls_loss = self.cce(logits, cls_target)
        correctness = logits[-1].argmax().item() == target_idx
        coeffs = torch.Tensor([((c_idx + 1) * (1 / num_stages)) for c_idx in range(num_stages)])
        assert coeffs.size() == sdr_loss.size() == cls_loss.size()
        loss = torch.sum(coeffs * (sdr_loss * n_src + cls_loss * self.lamb))
        return loss, pos_sdr, correctness


class Chimera(nn.Module):

    def __init__(self, in_chan, n_src, rnn_type='lstm', n_layers=2, hidden_size=600, bidirectional=True, dropout=0.3, embedding_dim=20, take_log=False, EPS=1e-08):
        super().__init__()
        self.input_dim = in_chan
        self.n_src = n_src
        self.take_log = take_log
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(rnn_type, in_chan, hidden_size, n_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * self.n_src)
        self.mask_act = nn.Sigmoid()
        self.embedding_layer = nn.Linear(rnn_out_dim, in_chan * embedding_dim)
        self.embedding_act = nn.Tanh()
        self.EPS = EPS

    def forward(self, input_data):
        batch, _, n_frames = input_data.shape
        if self.take_log:
            input_data = torch.log(input_data + self.EPS)
        out = self.rnn(input_data.permute(0, 2, 1))
        out = self.dropout(out)
        proj = self.embedding_layer(out)
        proj = self.embedding_act(proj)
        proj = proj.view(batch, n_frames, -1, self.embedding_dim).transpose(1, 2)
        proj = proj.reshape(batch, -1, self.embedding_dim)
        proj_norm = torch.norm(proj, p=2, dim=-1, keepdim=True)
        projection_final = proj / (proj_norm + self.EPS)
        mask_out = self.mask_layer(out).view(batch, n_frames, self.n_src, self.input_dim)
        mask_out = mask_out.permute(0, 2, 3, 1)
        mask_out = self.mask_act(mask_out)
        return projection_final, mask_out


class DummyModel(BaseEncoderMaskerDecoder):

    def __init__(self, fb_name='free', kernel_size=16, n_filters=32, stride=8, encoder_activation=None, **fb_kwargs):
        encoder, decoder = make_enc_dec(fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs)
        masker = torch.nn.Identity()
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveEncoder1D,
     lambda: ([], {'freq_res': 4, 'sample_res': 4}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     True),
    (Audio_Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64])], {}),
     True),
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Binarize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChanLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Chimera,
     lambda: ([], {'in_chan': 4, 'n_src': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CumLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DiscriminativeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (F1Tracker,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (F1_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatsGlobLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MulCatRNN,
     lambda: ([], {'rnn_type': 'gru', 'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MultiSrcNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PairwiseMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairwiseNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PairwiseNegSDR_Loss,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SCM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleModel,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SingleRNN,
     lambda: ([], {'rnn_type': 'gru', 'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SingleSrcMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleSrcNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (StackedResidualBiRNN,
     lambda: ([], {'rnn_type': 'gru', 'n_units': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (StackedResidualRNN,
     lambda: ([], {'rnn_type': 'gru', 'n_units': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Video_Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 512, 512])], {}),
     True),
    (_Chop1d,
     lambda: ([], {'chop_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ConvNorm,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (_ConvNormAct,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (_DilatedConvNorm,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (_NormAct,
     lambda: ([], {'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_asteroid_team_asteroid(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

