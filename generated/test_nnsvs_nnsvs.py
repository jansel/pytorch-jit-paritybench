import sys
_module = sys.modules[__name__]
del sys
conf = _module
nnsvs = _module
acoustic_models = _module
multistream = _module
sinsy = _module
tacotron = _module
tacotron_f0 = _module
util = _module
base = _module
bin = _module
anasyn = _module
fit_scaler = _module
gen_static_features = _module
generate = _module
prepare_features = _module
prepare_static_features = _module
preprocess_normalize = _module
synthesis = _module
train = _module
train_acoustic = _module
train_postfilter = _module
gen_static_features = _module
generate = _module
prepare_voc_features = _module
synthesis = _module
train = _module
train_acoustic = _module
train_postfilter = _module
data = _module
data_source = _module
diffsinger = _module
denoiser = _module
diffusion = _module
fs2 = _module
pe = _module
discriminators = _module
dsp = _module
frontend = _module
ja = _module
zh = _module
gen = _module
io = _module
hts = _module
layers = _module
conv = _module
layer_norm = _module
logger = _module
mdn = _module
model = _module
multistream = _module
pitch = _module
postfilters = _module
pretrained = _module
svs = _module
decoder = _module
encoder = _module
postnet = _module
train_util = _module
transformer = _module
attentions = _module
encoder = _module
usfgan = _module
cheaptrick = _module
residual_block = _module
upsample = _module
models = _module
discriminator = _module
generator = _module
utils = _module
features = _module
filters = _module
index = _module
util = _module
version = _module
wavenet = _module
conv = _module
modules = _module
wavenet = _module
clean_checkpoint_state = _module
data_prep = _module
align_lab = _module
finalize_lab = _module
musicxml2lab = _module
perf_segmentation = _module
round_lab = _module
ust2lab = _module
scaler_joblib2npy = _module
scaler_joblib2npy_voc = _module
extract_static_scaler = _module
setup = _module
app = _module
tests = _module
test_acoustic_models = _module
test_compat = _module
test_diffusion = _module
test_discriminators = _module
test_dsp = _module
test_frontend = _module
test_gen = _module
test_logger = _module
test_mdn = _module
test_model = _module
test_model_configs = _module
test_pitch = _module
test_postfilters = _module
test_preprocess = _module
test_svs = _module
test_util = _module
test_wavenet = _module
util = _module
enunu2nnsvs = _module
make_graph = _module
merge_postfilters = _module
nnsvs2opencpop = _module
nnsvs2usfgan = _module
opencpop2nnsvs = _module
pitch_augmentation = _module
sv56 = _module
sv56_inplace = _module
visualize_vibrato = _module

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


from functools import partial


from torch import nn


import torch


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import numpy as np


from torch.nn import functional as F


from enum import Enum


from scipy.io import wavfile


import torch.distributed as dist


from torch.cuda.amp import autocast


from torch.nn.parallel import DistributedDataParallel as DDP


import math


from math import sqrt


import torch.nn as nn


import torch.nn.functional as F


from collections import deque


from torch.nn import Parameter


from scipy import signal


from sklearn.preprocessing import MinMaxScaler


from torch.nn.utils import weight_norm


from warnings import warn


from scipy.signal import argrelmax


from scipy.signal import argrelmin


import random


import types


import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler


from torch import optim


from torch.cuda.amp import GradScaler


from torch.utils import data as data_utils


from torch.utils.data.sampler import BatchSampler


from torch.utils.tensorboard import SummaryWriter


import torch.fft


from logging import getLogger


import copy


from torchaudio.functional import spectrogram


from torch.nn.functional import interpolate


from torch.nn import ConstantPad1d as pad1d


from typing import Any


import torch.optim as optim


from sklearn.preprocessing import StandardScaler


class PredictionType(Enum):
    """Prediction types"""
    DETERMINISTIC = 1
    """Deterministic prediction

    Non-MDN single-stream models should use this type.

    Pseudo code:

    .. code-block::

        # training
        y = model(x)
        # inference
        y = model.inference(x)
    """
    PROBABILISTIC = 2
    """Probabilistic prediction with mixture density networks

    MDN-based models should use this type.

    Pseudo code:

    .. code-block::

        # training
        mdn_params = model(x)
        # inference
        mu, sigma = model.inference(x)
    """
    MULTISTREAM_HYBRID = 3
    """Multi-stream preodictions where each prediction can be
    detereministic or probabilistic

    Multi-stream models should use this type.

    Pseudo code:

    .. code-block::

        # training
        feature_streams = model(x) # e.g. (mgc, lf0, vuv, bap) or (mel, lf0, vuv)
        # inference
        y = model.inference(x)

    Note that concatenated features are assumed to be returned during inference.
    """
    DIFFUSION = 4
    """Diffusion model's prediction

    NOTE: may subject to change in the future

    Pseudo code:

    .. code-block::

        # training
        noise, x_recon = model(x)

        # inference
        y = model.inference(x)
    """


class BaseModel(nn.Module):
    """Base class for all models

    If you want to implement your custom model, you should inherit from this class.
    You must need to implement the forward method. Other methods are optional.
    """

    def forward(self, x, lengths=None, y=None):
        """Forward pass

        Args:
            x (tensor): input features
            lengths (tensor): lengths of the input features
            y (tensor): optional target features

        Returns:
            tensor: output features
        """
        pass

    def inference(self, x, lengths=None):
        """Inference method

        If you want to implement custom inference method such as autoregressive sampling,
        please override this method.

        Defaults to call the forward method.

        Args:
            x (tensor): input features
            lengths (tensor): lengths of the input features

        Returns:
            tensor: output features
        """
        return self(x, lengths)

    def preprocess_target(self, y):
        """Preprocess target signals at training time

        This is useful for shallow AR models in which a FIR filter
        is used for the target signals. For other types of model, you don't need to
        implement this method.

        Defaults to do nothing.

        Args:
            y (tensor): target features

        Returns:
            tensor: preprocessed target features
        """
        return y

    def prediction_type(self):
        """Prediction type.

        If your model has a MDN layer, please return ``PredictionType.PROBABILISTIC``.

        Returns:
            PredictionType: Determisitic or probabilistic. Default is deterministic.
        """
        return PredictionType.DETERMINISTIC

    def is_autoregressive(self):
        """Is autoregressive or not

        If your custom model is an autoregressive model, please return ``True``. In that case,
        you would need to implement autoregressive sampling in :py:meth:`inference`.

        Returns:
            bool: True if autoregressive. Default is False.
        """
        return False

    def has_residual_lf0_prediction(self):
        """Whether the model has residual log-F0 prediction or not.

        This should only be used for acoustic models.

        Returns:
            bool: True if the model has residual log-F0 prediction. Default is False.
        """
        return False


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs):
    """Weight-normalized Conv1d layer."""
    m = conv.Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs)
    return nn.utils.weight_norm(m)


class ResidualBlock(nn.Module):

    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):

    def __init__(self, in_dim=80, encoder_hidden_dim=256, residual_layers=20, residual_channels=256, dilation_cycle_length=4):
        super().__init__()
        self.in_dim = in_dim
        self.input_projection = Conv1d(in_dim, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.residual_layers = nn.ModuleList([ResidualBlock(encoder_hidden_dim, residual_channels, 2 ** (i % dilation_cycle_length)) for i in range(residual_layers)])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, in_dim, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for _, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x[:, None, :, :]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps, max_beta=0.06):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, max_beta, timesteps)
    return betas


beta_schedule = {'cosine': cosine_beta_schedule, 'linear': linear_beta_schedule}


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = 1, *shape[1:]
        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)
    else:
        return noise_fn(*shape, device=device)


class GaussianDiffusion(BaseModel):

    def __init__(self, in_dim, out_dim, denoise_fn, encoder=None, K_step=100, betas=None, schedule_type='linear', scheduler_params=None, norm_scale=10, pndm_speedup=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.denoise_fn = denoise_fn
        self.K_step = K_step
        self.pndm_speedup = pndm_speedup
        self.encoder = encoder
        self.norm_scale = norm_scale
        if scheduler_params is None:
            if schedule_type == 'linear':
                scheduler_params = {'max_beta': 0.06}
            elif schedule_type == 'cosine':
                scheduler_params = {'s': 0.008}
        if encoder is not None:
            assert encoder.in_dim == in_dim, 'encoder input dim must match in_dim'
        assert out_dim == denoise_fn.in_dim, 'denoise_fn input dim must match out_dim'
        if pndm_speedup:
            raise NotImplementedError('pndm_speedup is not implemented yet')
        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = beta_schedule[schedule_type](K_step, **scheduler_params)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.noise_list = deque(maxlen=4)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    def _norm(self, x, a_max=10):
        return x / a_max

    def _denorm(self, x, a_max=10):
        return x * a_max

    def prediction_type(self):
        return PredictionType.DIFFUSION

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, noise_fn=torch.randn, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, noise_fn, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond):
        """
        Use the PLMS method from Pseudo Numerical Methods for Diffusion Models on Manifolds
        https://arxiv.org/abs/2202.09778.
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta
            return x_pred
        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)
        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, torch.max(t - interval, torch.zeros_like(t)), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24
        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)
        return x_prev

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def forward(self, cond, lengths=None, y=None):
        """Forward step

        Args:
            cond (torch.Tensor): conditioning features of shaep (B, T, encoder_hidden_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): ground truth of shape (B, T, C)

        Returns:
            tuple of tensors (B, T, in_dim), (B, T, in_dim)
        """
        B = cond.shape[0]
        device = cond.device
        if self.encoder is not None:
            cond = self.encoder(cond, lengths)
        cond = cond.transpose(1, 2)
        t = torch.randint(0, self.K_step, (B,), device=device).long()
        x = self._norm(y, self.norm_scale)
        x = x.transpose(1, 2)[:, None, :, :]
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)
        noise = noise.squeeze(1).transpose(1, 2)
        x_recon = x_recon.squeeze(1).transpose(1, 2)
        return noise, x_recon

    def inference(self, cond, lengths=None):
        B = cond.shape[0]
        device = cond.device
        if self.encoder is not None:
            cond = self.encoder(cond, lengths)
        cond = cond.transpose(1, 2)
        t = self.K_step
        shape = cond.shape[0], 1, self.out_dim, cond.shape[2]
        x = torch.randn(shape, device=device)
        if self.pndm_speedup:
            self.noise_list = deque(maxlen=4)
            iteration_interval = int(self.pndm_speedup)
            for i in tqdm(reversed(range(0, t, iteration_interval)), desc='sample time step', total=t // iteration_interval):
                x = self.p_sample_plms(x, torch.full((B,), i, device=device, dtype=torch.long), iteration_interval, cond)
        else:
            for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                x = self.p_sample(x, torch.full((B,), i, device=device, dtype=torch.long), cond)
        x = self._denorm(x[:, 0].transpose(1, 2), self.norm_scale)
        return x


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, :x.size(1)]
        return self.dropout(x) + self.dropout(pos_emb)


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.enable_torch_version = False
        if hasattr(F, 'multi_head_attention_forward'):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None, before_softmax=False, need_head_weights=False, enc_dec_attn_constraint_mask=None, reset_attn_weight=None):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if self.enable_torch_version and incremental_state is None and not static_kv and reset_attn_weight is None:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        if incremental_state is not None:
            raise NotImplementedError()
        else:
            saved_state = None
        if self.self_attention:
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            raise NotImplementedError()
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(enc_dec_attn_constraint_mask.unsqueeze(2).bool(), -1000000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1000000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        if reset_attn_weight is not None:
            if reset_attn_weight:
                self.last_attn_probs = attn_probs.detach()
            else:
                assert self.last_attn_probs is not None
                attn_probs = self.last_attn_probs
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None
        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights


class BatchNorm1dTBC(nn.Module):

    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        x = x.permute(2, 0, 1)
        return x


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):

    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerFFNLayer(nn.Module):

    def __init__(self, hidden_size, filter_size, padding='SAME', kernel_size=1, dropout=0.0, act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(nn.ConstantPad1d((kernel_size - 1, 0), 0.0), nn.Conv1d(hidden_size, filter_size, kernel_size))
        self.ffn_2 = Linear(filter_size, hidden_size)
        if self.act == 'swish':
            self.swish_fn = CustomSwish()

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            assert incremental_state is None, 'Nar-generation does not allow this.'
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5
        if incremental_state is not None:
            x = x[-1:]
        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        if self.act == 'swish':
            x = self.swish_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-05):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class EncSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln', padding='SAME', act='gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(hidden_size, num_heads, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=kernel_size, padding=padding, norm=norm, act=act)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be
            the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    """
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs)
    return mask


class FFTBlocks(nn.Module):

    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=0.1, num_heads=2, use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(self.hidden_size, self.dropout, kernel_size=ffn_kernel_size, num_heads=num_heads) for _ in range(self.num_layers)])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, lengths, padding_mask=None):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        if padding_mask is None:
            padding_mask = make_pad_mask(lengths)
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x)
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        x = x.transpose(0, 1)
        return x


class FFTBlocksEncoder(BaseModel):

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2, ffn_kernel_size=9, dropout=0.1, num_heads=2, use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True, reduction_factor=1, downsample_by_conv=True, in_ph_start_idx: int=1, in_ph_end_idx: int=50, embed_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.reduction_factor = reduction_factor
        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            self.fc = nn.Linear(embed_dim, hidden_dim)
        else:
            self.emb = None
            self.fc_in = None
            self.fc = nn.Linear(in_dim, hidden_dim)
        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(in_dim, in_dim, kernel_size=reduction_factor, stride=reduction_factor, groups=in_dim)
        else:
            self.conv_downsample = None
        self.encoder = FFTBlocks(hidden_size=hidden_dim, num_layers=num_layers, ffn_kernel_size=ffn_kernel_size, dropout=dropout, num_heads=num_heads, use_pos_embed=use_pos_embed, use_last_norm=use_last_norm, norm=norm, use_pos_embed_alpha=use_pos_embed_alpha)
        self.fc_out = nn.Linear(hidden_dim, out_dim * reduction_factor)

    def forward(self, x, lengths, y=None):
        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(x, [self.in_ph_start_idx, self.num_vocab, self.in_dim - self.num_vocab - self.in_ph_start_idx], dim=-1)
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))
        if self.reduction_factor > 1:
            lengths = (lengths / self.reduction_factor).long()
            if self.conv_downsample is not None:
                x = self.conv_downsample(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = x[:, self.reduction_factor - 1::self.reduction_factor]
        x = self.fc(x)
        x = self.encoder(x, lengths)
        x = self.fc_out(x).view(x.shape[0], -1, self.out_dim)
        return x


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(100000.0)


class FSLayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(FSLayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(FSLayerNorm, self).forward(x)
        return super(FSLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class PitchPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1, padding='SAME'):
        """Initialize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding == 'SAME' else (kernel_size - 1, 0), 0), torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), FSLayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1))
        return xs


class Prenet(nn.Module):
    """Pre-Net of Tacotron.

    Args:
        in_dim (int) : dimension of input
        layers (int) : number of pre-net layers
        hidden_dim (int) : dimension of hidden layer
        dropout (float) : dropout rate
    """

    def __init__(self, in_dim, layers=2, hidden_dim=256, dropout=0.5, eval_dropout=True):
        super().__init__()
        self.dropout = dropout
        self.eval_dropout = eval_dropout
        prenet = nn.ModuleList()
        for layer in range(layers):
            prenet += [nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim), nn.ReLU()]
        self.prenet = nn.Sequential(*prenet)

    def forward(self, x):
        """Forward step

        Args:
            x (torch.Tensor) : input tensor

        Returns:
            torch.Tensor : output tensor
        """
        for layer in self.prenet:
            if self.eval_dropout:
                x = F.dropout(layer(x), self.dropout, training=True)
            else:
                x = F.dropout(layer(x), self.dropout, training=self.training)
        return x


class ConvBlock(nn.Module):

    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == 'ln':
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == 'wn':
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == 'none':
                pass
            elif self.norm == 'ln':
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):

    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0, strides=None, res=True):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.res = res
        self.in_proj = Linear(idim, n_chans)
        if strides is None:
            strides = [1] * n_layers
        else:
            assert len(strides) == n_layers
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=strides[idx], norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)
        hiddens = []
        for f in self.conv:
            x_ = f(x)
            x = x + x_ if self.res else x_
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)
            return x, hiddens
        return x


class PitchExtractor(nn.Module):

    def __init__(self, n_mel_bins=80, conv_layers=2, hidden_size=256, predictor_hidden=-1, ffn_padding='SAME', predictor_kernel=5, pitch_type='frame', use_uv=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.pitch_type = pitch_type
        assert pitch_type == 'log'
        self.use_uv = use_uv
        self.predictor_hidden = predictor_hidden if predictor_hidden > 0 else self.hidden_size
        self.conv_layers = conv_layers
        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=[1, 1, 1])
        if self.conv_layers > 0:
            self.mel_encoder = ConvStacks(idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size, n_layers=self.conv_layers)
        self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=self.predictor_hidden, n_layers=5, dropout_rate=0.1, odim=2, padding=ffn_padding, kernel_size=predictor_kernel)

    def forward(self, mel_input=None):
        mel_hidden = self.mel_prenet(mel_input)[1]
        if self.conv_layers > 0:
            mel_hidden = self.mel_encoder(mel_hidden)
        pitch_pred = self.pitch_predictor(mel_hidden)
        lf0, uv = pitch_pred[:, :, 0], pitch_pred[:, :, 1]
        f0 = 2 ** lf0
        lf0 = torch.log(f0)
        lf0[uv > 0] = 0
        return lf0


class PitchExtractorWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = PitchExtractor(**kwargs)

    def forward(self, x, lengths=None, y=None):
        return self.pitch_extractor(x)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Args:
        net (torch.nn.Module): network to initialize
        init_type (str): the name of an initialization method:
            normal | xavier | kaiming | orthogonal | none.
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    if init_type == 'none':
        return

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class Conv2dD(nn.Module):
    """Conv2d-based discriminator


    The implementation follows the discrimiantor of the GAN-based post-filters
    in :cite:t:`Kaneko2017Interspeech`.

    Args:
        in_dim (int): Input feature dim
        channels (int): Number of channels
        kernel_size (tuple): Kernel size for 2d-convolution
        padding (tuple): Padding for 2d-convolution
        last_sigmoid (bool): If True, apply sigmoid on the output
        init_type (str): Initialization type
        padding_mode (str): Padding mode
    """

    def __init__(self, in_dim=None, channels=64, kernel_size=(5, 3), padding=(0, 0), last_sigmoid=False, init_type='kaiming_normal', padding_mode='zeros'):
        super().__init__()
        self.last_sigmoid = last_sigmoid
        C = channels
        ks = np.asarray(list(kernel_size))
        if padding is None:
            padding = (ks - 1) // 2
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(nn.Conv2d(1, C, kernel_size=ks, padding=padding, stride=(1, 1), padding_mode=padding_mode), nn.LeakyReLU(0.2)))
        self.convs.append(nn.Sequential(nn.Conv2d(C, 2 * C, kernel_size=ks, padding=padding, stride=(2, 1), padding_mode=padding_mode), nn.LeakyReLU(0.2)))
        self.convs.append(nn.Sequential(nn.Conv2d(2 * C, 4 * C, kernel_size=ks, padding=padding, stride=(2, 1), padding_mode=padding_mode), nn.LeakyReLU(0.2)))
        self.convs.append(nn.Sequential(nn.Conv2d(4 * C, 2 * C, kernel_size=ks, padding=padding, stride=(2, 1), padding_mode=padding_mode), nn.LeakyReLU(0.2)))
        self.last_conv = nn.Conv2d(2 * C, 1, kernel_size=ks, padding=padding, stride=(1, 1), padding_mode=padding_mode)
        init_weights(self, init_type)

    def forward(self, x, c=None, lengths=None):
        """Forward step

        Args:
            x (torch.Tensor): Input tensor
            c (torch.Tensor): Optional conditional features
            lengths (torch.Tensor): Optional lengths of the input

        Returns:
            list: List of output tensors
        """
        outs = []
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            outs.append(x)
        y = self.last_conv(x)
        y = torch.sigmoid(y) if self.last_sigmoid else y
        y = y.squeeze(1)
        outs.append(y)
        return [outs]


class TimeInvFIRFilter(nn.Conv1d):
    """Time-invatiant FIR filter implementation

    Args:
        channels (int): input channels
        filt_coef (torch.Tensor): FIR filter coefficients
        causal (bool): causal
        requires_grad (bool): trainable kernel or not
    """

    def __init__(self, channels, filt_coef, causal=True, requires_grad=False):
        assert len(filt_coef.shape) == 1
        kernel_size = len(filt_coef)
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * 1
        else:
            padding = (kernel_size - 1) // 2 * 1
        super(TimeInvFIRFilter, self).__init__(channels, channels, kernel_size, padding=padding, groups=channels, bias=None)
        self.weight.data[:, :, :] = filt_coef.flip(-1)
        self.weight.requires_grad = requires_grad

    def forward(self, x):
        out = super(TimeInvFIRFilter, self).forward(x)
        out = out[:, :, :-self.padding[0]] if self.causal else out
        return out


class TrTimeInvFIRFilter(nn.Conv1d):
    """Trainable Time-invatiant FIR filter implementation

    H(z) = \\sigma_{k=0}^{filt_dim} b_{k}z_{-k}

    Note that b_{0} is fixed to 1 if fixed_0th is True.

    Args:
        channels (int): input channels
        filt_dim (int): FIR filter dimension
        causal (bool): causal
        tanh (bool): apply tanh to filter coef or not.
        fixed_0th (bool): fix the first filt coef to 1 or not.
    """

    def __init__(self, channels, filt_dim, causal=True, tanh=True, fixed_0th=True):
        init_filt_coef = torch.randn(filt_dim) * (1 / filt_dim)
        kernel_size = len(init_filt_coef)
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * 1
        else:
            padding = (kernel_size - 1) // 2 * 1
        super(TrTimeInvFIRFilter, self).__init__(channels, channels, kernel_size, padding=padding, groups=channels, bias=None)
        self.weight.data[:, :, :] = init_filt_coef.flip(-1)
        self.weight.requires_grad = True
        self.tanh = tanh
        self.fixed_0th = fixed_0th

    def get_filt_coefs(self):
        b = torch.tanh(self.weight) if self.tanh else self.weight
        b = b.clone()
        if self.fixed_0th:
            b[:, :, -1] = 1
        return b

    def forward(self, x):
        b = self.get_filt_coefs()
        out = F.conv1d(x, b, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.padding[0] > 0:
            out = out[:, :, :-self.padding[0]] if self.causal else out
        return out


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(dilation), WNConv1d(dim, dim, kernel_size=3, dilation=dilation), nn.LeakyReLU(0.2), WNConv1d(dim, dim, kernel_size=1))
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class MDNLayer(nn.Module):
    """Mixture Density Network layer

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.
    If dim_wise is True, features for each dimension are modeld by independent 1-D GMMs
    instead of modeling jointly. This would workaround training difficulty
    especially for high dimensional data.

    Implementation references:
    1. Mixture Density Networks by Mike Dusenberry
    https://mikedusenberry.com/mixture-density-networks
    2. PRML book
    https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
    3. sagelywizard/pytorch-mdn
    https://github.com/sagelywizard/pytorch-mdn
    4. sksq96/pytorch-mdn
    https://github.com/sksq96/pytorch-mdn

    Attributes:
        in_dim (int): the number of dimensions in the input
        out_dim (int): the number of dimensions in the output
        num_gaussians (int): the number of mixture component
        dim_wise (bool): whether to model data for each dimension separately
    """

    def __init__(self, in_dim, out_dim, num_gaussians=30, dim_wise=False):
        super(MDNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians
        self.dim_wise = dim_wise
        odim_log_pi = out_dim * num_gaussians if dim_wise else num_gaussians
        self.log_pi = nn.Linear(in_dim, odim_log_pi)
        self.log_sigma = nn.Linear(in_dim, out_dim * num_gaussians)
        self.mu = nn.Linear(in_dim, out_dim * num_gaussians)

    def forward(self, minibatch):
        """Forward for MDN

        Args:
            minibatch (torch.Tensor): tensor of shape (B, T, D_in)
                B is the batch size and T is data lengths of this batch,
                and D_in is in_dim.

        Returns:
            torch.Tensor: Tensor of shape (B, T, G) or (B, T, G, D_out)
                Log of mixture weights. G is num_gaussians and D_out is out_dim.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                the log of standard deviation of each Gaussians.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                mean of each Gaussians
        """
        B = len(minibatch)
        if self.dim_wise:
            log_pi = self.log_pi(minibatch).view(B, -1, self.num_gaussians, self.out_dim)
            log_pi = F.log_softmax(log_pi, dim=2)
        else:
            log_pi = F.log_softmax(self.log_pi(minibatch), dim=2)
        log_sigma = self.log_sigma(minibatch)
        log_sigma = log_sigma.view(B, -1, self.num_gaussians, self.out_dim)
        mu = self.mu(minibatch)
        mu = mu.view(B, -1, self.num_gaussians, self.out_dim)
        return log_pi, log_sigma, mu


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu):
    """Return the mean and standard deviation of the Gaussian component
    whose weight coefficient is the largest as the most probable predictions.

    Args:
        log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
            The log of multinomial distribution of the Gaussians.
            B is the batch size, T is data length of this batch,
            G is num_gaussians of class MDNLayer.
        log_sigma (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The standard deviation of the Gaussians. D_out is out_dim of class
            MDNLayer.
        mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The means of the Gaussians. D_out is out_dim of class MDNLayer.

    Returns:
        tuple: tuple of torch.Tensor
            torch.Tensor of shape (B, T, D_out). The standardd deviations
            of the most probable Gaussian component.
            torch.Tensor of shape (B, T, D_out). Means of the Gaussians.
    """
    dim_wise = len(log_pi.shape) == 4
    _, _, num_gaussians, _ = mu.shape
    _, max_component = torch.max(log_pi, dim=2)
    one_hot = to_one_hot(max_component, num_gaussians)
    if dim_wise:
        one_hot = one_hot.transpose(2, 3)
        assert one_hot.shape == mu.shape
    else:
        one_hot = one_hot.unsqueeze(3).expand_as(mu)
    max_mu = torch.sum(mu * one_hot, dim=2)
    max_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))
    return max_sigma, max_mu


class Conv1dResnet(BaseModel):
    """Conv1d + Resnet

    The model is inspired by the MelGAN's model architecture (:cite:t:`kumar2019melgan`).
    MDN layer is added if use_mdn is True.

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        init_type (str): the type of weight initialization
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): the number of gaussians in MDN
        dim_wise (bool): whether to use dim-wise or not
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, init_type='none', use_mdn=False, num_gaussians=8, dim_wise=False, **kwargs):
        super().__init__()
        self.use_mdn = use_mdn
        if 'dropout' in kwargs:
            warn('dropout argument in Conv1dResnet is deprecated and will be removed in future versions')
        model = [nn.ReflectionPad1d(3), WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0)]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        last_conv_out_dim = hidden_dim if use_mdn else out_dim
        model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(hidden_dim, last_conv_out_dim, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        if self.use_mdn:
            self.mdn_layer = MDNLayer(in_dim=hidden_dim, out_dim=out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)
        else:
            self.mdn_layer = None
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC if self.use_mdn else PredictionType.DETERMINISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        if self.use_mdn:
            return self.mdn_layer(out)
        else:
            return out

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance if use_mdn is True

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


def split_streams(inputs, stream_sizes=None):
    """Split streams from multi-stream features

    Args:
        inputs (array like): input 3-d array
        stream_sizes (list): sizes for each stream

    Returns:
        list: list of stream features
    """
    if stream_sizes is None:
        stream_sizes = [60, 1, 1, 1]
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size in zip(start_indices, stream_sizes):
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx:start_idx + size]
        else:
            s = inputs[:, start_idx:start_idx + size]
        ret.append(s)
    return ret


@torch.no_grad()
def _shallow_ar_inference(out, stream_sizes, analysis_filts):
    from torchaudio.functional import lfilter
    out_streams = split_streams(out, stream_sizes)
    out_streams = map(lambda x: x.transpose(1, 2), out_streams)
    out_syn = []
    for sidx, os in enumerate(out_streams):
        out_stream_syn = torch.zeros_like(os)
        a = analysis_filts[sidx].get_filt_coefs()
        for idx in range(os.shape[1]):
            ai = a[idx].view(-1).flip(0)
            bi = torch.zeros_like(ai)
            bi[0] = 1
            out_stream_syn[:, idx, :] = lfilter(os[:, idx, :], ai, bi, clamp=False)
        out_syn += [out_stream_syn]
    out_syn = torch.cat(out_syn, 1)
    return out_syn.transpose(1, 2)


class Conv1dResnetSAR(Conv1dResnet):
    """Conv1dResnet with shallow AR structure

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
        init_type (str): the type of weight initialization
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, stream_sizes=None, ar_orders=None, init_type='none', **kwargs):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers)
        if 'dropout' in kwargs:
            warn('dropout argument in Conv1dResnetSAR is deprecated and will be removed in future versions')
        if stream_sizes is None:
            stream_sizes = [180, 3, 1, 15]
        if ar_orders is None:
            ar_orders = [20, 200, 20, 20]
        self.stream_sizes = stream_sizes
        init_weights(self, init_type)
        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K + 1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1, 2)).transpose(1, 2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None, y=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


def convert_pad_shape(pad_shape):
    ll = pad_shape[::-1]
    pad_shape = [item for sublist in ll for item in sublist]
    return pad_shape


class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == 'gelu':
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x


class LSTMRNN(BaseModel):
    """LSTM-based recurrent neural network

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        init_type (str): the type of weight initialization
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0, init_type='none'):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, out_dim)
        init_weights(self, init_type)

    def forward(self, x, lengths, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        if isinstance(lengths, torch.Tensor):
            lengths = lengths
        x = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


class LSTMRNNSAR(LSTMRNN):
    """LSTM-RNN with shallow AR structure

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
        init_type (str): the type of weight initialization
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0, stream_sizes=None, ar_orders=None, init_type='none'):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, bidirectional, dropout, init_type)
        if stream_sizes is None:
            stream_sizes = [180, 3, 1, 15]
        if ar_orders is None:
            ar_orders = [20, 200, 20, 20]
        self.stream_sizes = stream_sizes
        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K + 1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1, 2)).transpose(1, 2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None, y=None):
        out = self.forward(x, lengths)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class RMDN(BaseModel):
    """RNN-based mixture density networks (MDN)

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dimension-wise or not
        init_type (str): the type of weight initialization
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0, num_gaussians=8, dim_wise=False, init_type='none'):
        super(RMDN, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.mdn = MDNLayer(in_dim=self.num_direction * hidden_dim, out_dim=out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        if isinstance(lengths, torch.Tensor):
            lengths = lengths
        out = self.linear(x)
        sequence = pack_padded_sequence(self.relu(out), lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.mdn(out)
        return out

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class MDN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, num_gaussians=30):
        super(MDN, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.tanh = nn.Tanh()
        self.mdn = mdn.MDNLayer(hidden_dim, out_dim, num_gaussians=num_gaussians)

    def forward(self, x, lengths=None):
        out = self.tanh(self.first_linear(x))
        for hl in self.hidden_layers:
            out = self.tanh(hl(out))
        return self.mdn(out)


class MDNv2(BaseModel):
    """Mixture density networks (MDN) with FFN

    MDN (v1) + Dropout

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        dropout (float): dropout rate
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dimension-wise or not
        init_type (str): the type of weight initialization
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, dropout=0.5, num_gaussians=8, dim_wise=False, init_type='none'):
        super(MDNv2, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        model += [MDNLayer(in_dim=hidden_dim, out_dim=out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class Conv1dResnetMDN(BaseModel):
    """Conv1dResnet with MDN output layer

    .. warning::

        Will be removed in v0.1.0. Use Conv1dResNet with ``use_mdn=True`` instead.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, num_gaussians=8, dim_wise=False, init_type='none', **kwargs):
        super().__init__()
        if 'dropout' in kwargs:
            warn('dropout argument in Conv1dResnet is deprecated and will be removed in future versions')
        model = [Conv1dResnet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=num_layers), nn.ReLU(), MDNLayer(in_dim=hidden_dim, out_dim=out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class FFConvLSTM(BaseModel):
    """FFN + Conv1d + LSTM

    A model proposed in :cite:t:`hono2021sinsy` without residual F0 prediction.

    Args:
        in_dim (int): the dimension of the input
        ff_hidden_dim (int): the dimension of the hidden state of the FFN
        conv_hidden_dim (int): the dimension of the hidden state of the conv1d
        lstm_hidden_dim (int): the dimension of the hidden state of the LSTM
        out_dim (int): the dimension of the output
        dropout (float): dropout rate
        num_lstm_layers (int): the number of layers of the LSTM
        bidirectional (bool): whether to use bidirectional LSTM
        init_type (str): the type of weight initialization
        use_mdn (bool): whether to use MDN or not
        dim_wise (bool): whether to use dimension-wise or not
        num_gaussians (int): the number of gaussians
        in_ph_start_idx (int): the start index of phoneme identity in a hed file
        in_ph_end_idx (int): the end index of phoneme identity in a hed file
        embed_dim (int): the dimension of the phoneme embedding
    """

    def __init__(self, in_dim, ff_hidden_dim=2048, conv_hidden_dim=1024, lstm_hidden_dim=256, out_dim=67, dropout=0.0, num_lstm_layers=2, bidirectional=True, init_type='none', use_mdn=False, dim_wise=True, num_gaussians=4, in_ph_start_idx: int=1, in_ph_end_idx: int=50, embed_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.use_mdn = use_mdn
        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim
        self.ff = nn.Sequential(nn.Linear(ff_in_dim, ff_hidden_dim), nn.ReLU(), nn.Linear(ff_hidden_dim, ff_hidden_dim), nn.ReLU(), nn.Linear(ff_hidden_dim, ff_hidden_dim), nn.ReLU())
        self.conv = nn.Sequential(nn.ReflectionPad1d(3), nn.Conv1d(ff_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0), nn.BatchNorm1d(conv_hidden_dim), nn.ReLU(), nn.ReflectionPad1d(3), nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0), nn.BatchNorm1d(conv_hidden_dim), nn.ReLU(), nn.ReflectionPad1d(3), nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0), nn.BatchNorm1d(conv_hidden_dim), nn.ReLU())
        num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(conv_hidden_dim, lstm_hidden_dim, num_lstm_layers, bidirectional=True, batch_first=True, dropout=dropout)
        last_in_dim = num_direction * lstm_hidden_dim
        if self.use_mdn:
            assert dim_wise
            self.fc = MDNLayer(in_dim=last_in_dim, out_dim=out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)
        else:
            self.fc = nn.Linear(last_in_dim, out_dim)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC if self.use_mdn else PredictionType.DETERMINISTIC

    def forward(self, x, lengths=None, y=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths
        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(x, [self.in_ph_start_idx, self.num_vocab, self.in_dim - self.num_vocab - self.in_ph_start_idx], dim=-1)
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))
        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)

    def inference(self, x, lengths=None):
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


class VariancePredictor(BaseModel):
    """Variance predictor in :cite:t:`ren2020fastspeech`.

    The model is composed of stacks of Conv1d + ReLU + LayerNorm layers.
    The model can be used for duration or pitch prediction.

    Args:
        in_dim (int): the input dimension
        out_dim (int): the output dimension
        num_layers (int): the number of layers
        hidden_dim (int): the hidden dimension
        kernel_size (int): the kernel size
        dropout (float): the dropout rate
        init_type (str): the initialization type
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dim-wise or not
    """

    def __init__(self, in_dim, out_dim, num_layers=5, hidden_dim=256, kernel_size=5, dropout=0.5, init_type='none', use_mdn=False, num_gaussians=1, dim_wise=False):
        super().__init__()
        self.use_mdn = use_mdn
        conv = nn.ModuleList()
        for idx in range(num_layers):
            in_channels = in_dim if idx == 0 else hidden_dim
            conv += [nn.Sequential(nn.Conv1d(in_channels, hidden_dim, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.ReLU(), LayerNorm(hidden_dim, dim=1), nn.Dropout(dropout))]
        self.conv = nn.Sequential(*conv)
        if self.use_mdn:
            self.mdn_layer = MDNLayer(hidden_dim, out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise)
        else:
            self.fc = nn.Linear(hidden_dim, out_dim)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC if self.use_mdn else PredictionType.DETERMINISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.use_mdn:
            return self.mdn_layer(out)
        else:
            return self.fc(out)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance if use_mdn is True

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


class LSTMEncoder(BaseModel):
    """LSTM encoder

    A simple LSTM-based encoder

    Args:
        in_dim (int): the input dimension
        hidden_dim (int): the hidden dimension
        out_dim (int): the output dimension
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional or not
        dropout (float): the dropout rate
        init_type (str): the initialization type
        in_ph_start_idx (int): the start index of phonetic context in a hed file
        in_ph_end_idx (int): the end index of phonetic context in a hed file
        embed_dim (int): the embedding dimension
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int=1, bidirectional: bool=True, dropout: float=0.0, init_type: str='none', in_ph_start_idx: int=1, in_ph_end_idx: int=50, embed_dim=None):
        super(LSTMEncoder, self).__init__()
        self.in_dim = in_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            lstm_in_dim = embed_dim
        else:
            lstm_in_dim = in_dim
        self.num_layers = num_layers
        num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(lstm_in_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(num_direction * hidden_dim, out_dim)
        init_weights(self, init_type)

    def forward(self, x, lengths, y=None):
        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(x, [self.in_ph_start_idx, self.num_vocab, self.in_dim - self.num_vocab - self.in_ph_start_idx], dim=-1)
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))
        if isinstance(lengths, torch.Tensor):
            lengths = lengths
        x = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class TransformerEncoder(BaseModel):
    """Transformer encoder


    .. warning::

        So far this is not well tested. Maybe be removed in the future.

    Args:
        in_dim (int): the input dimension
        out_dim (int): the output dimension
        hidden_dim (int): the hidden dimension
        attention_dim (int): the attention dimension
        num_heads (int): the number of heads
        num_layers (int): the number of layers
        kernel_size (int): the kernel size
        dropout (float): the dropout rate
        reduction_factor (int): the reduction factor
        init_type (str): the initialization type
        downsample_by_conv (bool): whether to use convolutional downsampling or not
        in_ph_start_idx (int): the start index of phonetic context in a hed file
        in_ph_end_idx (int): the end index of phonetic context in a hed file
        embed_dim (int): the embedding dimension
    """

    def __init__(self, in_dim, out_dim, hidden_dim, attention_dim, num_heads=2, num_layers=2, kernel_size=3, dropout=0.1, reduction_factor=1, init_type='none', downsample_by_conv=False, in_ph_start_idx: int=1, in_ph_end_idx: int=50, embed_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            self.fc = nn.Linear(embed_dim, hidden_dim)
        else:
            self.emb = None
            self.fc_in = None
            self.fc = nn.Linear(in_dim, hidden_dim)
        self.reduction_factor = reduction_factor
        self.encoder = _TransformerEncoder(hidden_channels=hidden_dim, filter_channels=attention_dim, n_heads=num_heads, n_layers=num_layers, kernel_size=kernel_size, p_dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, out_dim * reduction_factor)
        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(in_dim, in_dim, kernel_size=reduction_factor, stride=reduction_factor, groups=in_dim)
        else:
            self.conv_downsample = None
        for f in [self.fc_in, self.emb, self.fc, self.fc_out]:
            if f is not None:
                init_weights(f, init_type)

    def forward(self, x, lengths=None, y=None):
        """Forward pass

        Args:
            x (torch.Tensor): input tensor
            lengths (torch.Tensor): input sequence lengths
            y (torch.Tensor): target tensor (optional)

        Returns:
            torch.Tensor: output tensor
        """
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(x, [self.in_ph_start_idx, self.num_vocab, self.in_dim - self.num_vocab - self.in_ph_start_idx], dim=-1)
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))
        if self.reduction_factor > 1:
            lengths = (lengths / self.reduction_factor).long()
            if self.conv_downsample is not None:
                x = self.conv_downsample(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = x[:, self.reduction_factor - 1::self.reduction_factor]
        x = self.fc(x)
        x = x.transpose(1, 2)
        x_mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1)
        x = self.encoder(x * x_mask, x_mask)
        x = self.fc_out(x.transpose(1, 2)).view(x.shape[0], -1, self.out_dim)
        return x


class MovingAverage1d(nn.Conv1d):
    """Moving average filter on 1-D signals

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): kernel size
        padding_mode (str): padding mode
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, groups=in_channels, bias=False, padding='same', padding_mode=padding_mode)
        nn.init.constant_(self.weight, 1 / kernel_size)
        for p in self.parameters():
            p.requires_grad = False


class Conv2dPostFilter(BaseModel):
    """A post-filter based on Conv2d

    A model proposed in :cite:t:`kaneko2017generative`.

    Args:
        channels (int): number of channels
        kernel_size (tuple): kernel sizes for Conv2d
        init_type (str): type of initialization
        noise_scale (float): scale of noise
        noise_type (str): type of noise. "frame_wise" or "bin_wise"
        padding_mode (str): padding mode
        smoothing_width (int): Width of smoothing window.
            The larger the smoother. Only used at inference time.
    """

    def __init__(self, in_dim=None, channels=128, kernel_size=(5, 5), init_type='kaiming_normal', noise_scale=1.0, noise_type='bin_wise', padding_mode='zeros', smoothing_width=-1):
        super().__init__()
        self.in_dim = in_dim
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        C = channels
        self.smoothing_width = smoothing_width
        assert len(kernel_size) == 2
        ks = np.asarray(list(kernel_size))
        padding = (ks - 1) // 2
        self.conv1 = nn.Sequential(nn.Conv2d(2, C, kernel_size=ks, padding=padding, padding_mode=padding_mode), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(C + 1, C * 2, kernel_size=ks, padding=padding, padding_mode=padding_mode), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(C * 2 + 1, C, kernel_size=ks, padding=padding, padding_mode=padding_mode), nn.ReLU())
        self.conv4 = nn.Conv2d(C + 1, 1, kernel_size=ks, padding=padding, padding_mode=padding_mode)
        if self.noise_type == 'frame_wise':
            self.fc = nn.Linear(1, in_dim)
        elif self.noise_type == 'bin_wise':
            self.fc = None
        else:
            raise ValueError('Unknown noise type: {}'.format(self.noise_type))
        init_weights(self, init_type)

    def forward(self, x, lengths=None, y=None, is_inference=False):
        """Forward step

        Args:
            x (torch.Tensor): input tensor of shape (B, T, C)
            lengths (torch.Tensor): lengths of shape (B,)

        Returns:
            torch.Tensor: output tensor of shape (B, T, C)
        """
        x = x.unsqueeze(1)
        if self.noise_type == 'bin_wise':
            z = torch.randn_like(x).squeeze(1).transpose(1, 2) * self.noise_scale
            if is_inference and self.smoothing_width > 0:
                ave_filt = MovingAverage1d(self.in_dim, self.in_dim, self.smoothing_width)
                z = ave_filt(z)
            z = z.transpose(1, 2).unsqueeze(1)
        elif self.noise_type == 'frame_wise':
            z = torch.randn(x.shape[0], 1, x.shape[2]) * self.noise_scale
            if is_inference and self.smoothing_width > 0:
                ave_filt = MovingAverage1d(1, 1, self.smoothing_width)
                z = ave_filt(z)
            z = z.unsqueeze(-1)
            z = self.fc(z)
        x_syn = x
        y = self.conv1(torch.cat([x_syn, z], dim=1))
        y = self.conv2(torch.cat([x_syn, y], dim=1))
        y = self.conv3(torch.cat([x_syn, y], dim=1))
        residual = self.conv4(torch.cat([x_syn, y], dim=1))
        out = x_syn + residual
        out = out.squeeze(1)
        return out

    def inference(self, x, lengths=None):
        return self(x, lengths, is_inference=True)


class MultistreamPostFilter(BaseModel):
    """A multi-stream post-filter that applies post-filtering for each feature stream

    Currently, post-filtering for MGC, BAP and log-F0 are supported.
    Note that it doesn't make much sense to apply post-filtering for other features.

    Args:
        mgc_postfilter (nn.Module): post-filter for MGC
        bap_postfilter (nn.Module): post-filter for BAP
        lf0_postfilter (nn.Module): post-filter for log-F0
        stream_sizes (list): sizes of each feature stream
        mgc_offset (int): offset for MGC. Defaults to 2.
        bap_offset (int): offset for BAP. Defaults to 0.
    """

    def __init__(self, mgc_postfilter: nn.Module, bap_postfilter: nn.Module, lf0_postfilter: nn.Module, stream_sizes: list, mgc_offset: int=2, bap_offset: int=0):
        super().__init__()
        self.mgc_postfilter = mgc_postfilter
        self.bap_postfilter = bap_postfilter
        self.lf0_postfilter = lf0_postfilter
        self.stream_sizes = stream_sizes
        self.mgc_offset = mgc_offset
        self.bap_offset = bap_offset

    def forward(self, x, lengths=None, y=None, is_inference=False):
        """Forward step

        Each feature stream is processed independently.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, C)
            lengths (torch.Tensor): lengths of shape (B,)

        Returns:
            torch.Tensor: output tensor of shape (B, T, C)
        """
        streams = split_streams(x, self.stream_sizes)
        if len(streams) == 4:
            mgc, lf0, vuv, bap = streams
        elif len(streams) == 5:
            mgc, lf0, vuv, bap, vuv = streams
        elif len(streams) == 6:
            mgc, lf0, vuv, bap, vib, vib_flags = streams
        else:
            raise ValueError('Invalid number of streams')
        if self.mgc_postfilter is not None:
            if self.mgc_offset > 0:
                mgc0 = mgc[:, :, :self.mgc_offset]
                if is_inference:
                    mgc_pf = self.mgc_postfilter.inference(mgc[:, :, self.mgc_offset:], lengths)
                else:
                    mgc_pf = self.mgc_postfilter(mgc[:, :, self.mgc_offset:], lengths)
                mgc_pf = torch.cat([mgc0, mgc_pf], dim=-1)
            elif is_inference:
                mgc_pf = self.mgc_postfilter.inference(mgc, lengths)
            else:
                mgc_pf = self.mgc_postfilter(mgc, lengths)
            mgc = mgc_pf
        if self.bap_postfilter is not None:
            if self.bap_offset > 0:
                bap0 = bap[:, :, :self.bap_offset]
                if is_inference:
                    bap_pf = self.bap_postfilter.inference(bap[:, :, self.bap_offset:], lengths)
                else:
                    bap_pf = self.bap_postfilter(bap[:, :, self.bap_offset:], lengths)
                bap_pf = torch.cat([bap0, bap_pf], dim=-1)
            elif is_inference:
                bap_pf = self.bap_postfilter.inference(bap, lengths)
            else:
                bap_pf = self.bap_postfilter(bap, lengths)
            bap = bap_pf
        if self.lf0_postfilter is not None:
            if is_inference:
                lf0 = self.lf0_postfilter.inference(lf0, lengths)
            else:
                lf0 = self.lf0_postfilter(lf0, lengths)
        if len(streams) == 4:
            out = torch.cat([mgc, lf0, vuv, bap], dim=-1)
        elif len(streams) == 5:
            out = torch.cat([mgc, lf0, vuv, bap, vib], dim=-1)
        elif len(streams) == 6:
            out = torch.cat([mgc, lf0, vuv, bap, vib, vib_flags], dim=-1)
        return out

    def inference(self, x, lengths):
        return self(x, lengths, is_inference=True)


class MelF0MultistreamPostFilter(BaseModel):

    def __init__(self, mel_postfilter: nn.Module, lf0_postfilter: nn.Module, stream_sizes: list, mel_offset: int=0):
        super().__init__()
        self.mel_postfilter = mel_postfilter
        self.lf0_postfilter = lf0_postfilter
        self.stream_sizes = stream_sizes
        self.mel_offset = mel_offset

    def forward(self, x, lengths=None, y=None, is_inference=False):
        """Forward step

        Each feature stream is processed independently.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, C)
            lengths (torch.Tensor): lengths of shape (B,)

        Returns:
            torch.Tensor: output tensor of shape (B, T, C)
        """
        streams = split_streams(x, self.stream_sizes)
        assert len(streams) == 3
        mel, lf0, vuv = streams
        if self.mel_postfilter is not None:
            if self.mel_offset > 0:
                mel0 = mel[:, :, :self.mel_offset]
                if is_inference:
                    mel_pf = self.mel_postfilter.inference(mel[:, :, self.mel_offset:], lengths)
                else:
                    mel_pf = self.mel_postfilter(mel[:, :, self.mel_offset:], lengths)
                mel_pf = torch.cat([mel0, mel_pf], dim=-1)
            elif is_inference:
                mel_pf = self.mel_postfilter.inference(mel, lengths)
            else:
                mel_pf = self.mel_postfilter(mel, lengths)
            mel = mel_pf
        if self.lf0_postfilter is not None:
            if is_inference:
                lf0 = self.lf0_postfilter.inference(lf0, lengths)
            else:
                lf0 = self.lf0_postfilter(lf0, lengths)
        out = torch.cat([mel, lf0, vuv], dim=-1)
        return out

    def inference(self, x, lengths):
        return self(x, lengths, is_inference=True)


class _PadConv2dPostFilter(BaseModel):

    def __init__(self, in_dim=None, channels=128, kernel_size=5, init_type='kaiming_normal', padding_side='left'):
        super().__init__()
        assert not isinstance(kernel_size, list)
        C = channels
        ks = kernel_size
        padding = (ks - 1) // 2
        self.padding = padding
        self.padding_side = padding_side
        if padding_side == 'left':
            self.pad = nn.ReflectionPad2d((padding, 0, padding, padding))
        elif padding_side == 'none':
            self.pad = nn.ReflectionPad2d((0, 0, padding, padding))
        elif padding_side == 'right':
            self.pad = nn.ReflectionPad2d((0, padding, padding, padding))
        else:
            raise ValueError('Invalid padding side')
        self.conv1 = nn.Sequential(nn.Conv2d(2, C, kernel_size=(ks, ks)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(C + 1, C * 2, kernel_size=(ks, 3), padding=(padding, 1), padding_mode='reflect'), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(C * 2 + 1, C, kernel_size=(ks, 3), padding=(padding, 1), padding_mode='reflect'), nn.ReLU())
        self.conv4 = nn.Conv2d(C + 1, 1, kernel_size=(ks, 1), padding=(padding, 0), padding_mode='reflect')
        self.fc = nn.Linear(1, in_dim)
        init_weights(self, init_type)

    def forward(self, x, z, lengths=None):
        x = x.unsqueeze(1)
        z = z.unsqueeze(1)
        z = self.fc(z)
        x_syn = x
        y = self.conv1(torch.cat([self.pad(x_syn), self.pad(z)], dim=1))
        if self.padding_side == 'left':
            x_syn = x[:, :, :, :-self.padding]
        elif self.padding_side == 'none':
            x_syn = x[:, :, :, self.padding:-self.padding]
        elif self.padding_side == 'right':
            x_syn = x[:, :, :, self.padding:]
        y = self.conv2(torch.cat([x_syn, y], dim=1))
        y = self.conv3(torch.cat([x_syn, y], dim=1))
        residual = self.conv4(torch.cat([x_syn, y], dim=1))
        out = x_syn + residual
        out = out.squeeze(1)
        return out


class MultistreamConv2dPostFilter(nn.Module):
    """Conv2d-based multi-stream post-filter designed for MGC

    Divide the MGC transformation into low/mid/high dim transfomations
    with small overlaps. Overlap is determined by the kernel size.
    """

    def __init__(self, in_dim=None, channels=128, kernel_size=5, init_type='kaiming_normal', noise_scale=1.0, stream_sizes=(8, 20, 30)):
        super().__init__()
        assert len(stream_sizes) == 3
        self.padding = (kernel_size - 1) // 2
        self.noise_scale = noise_scale
        self.stream_sizes = stream_sizes
        self.low_postfilter = _PadConv2dPostFilter(stream_sizes[0] + self.padding, channels=channels, kernel_size=kernel_size, init_type=init_type, padding_side='left')
        self.mid_postfilter = _PadConv2dPostFilter(stream_sizes[1] + 2 * self.padding, channels=channels, kernel_size=kernel_size, init_type=init_type, padding_side='none')
        self.high_postfilter = _PadConv2dPostFilter(stream_sizes[2] + self.padding, channels=channels, kernel_size=kernel_size, init_type=init_type, padding_side='right')

    def forward(self, x, lengths=None, y=None):
        assert x.shape[-1] == sum(self.stream_sizes)
        z = torch.randn(x.shape[0], x.shape[1], 1) * self.noise_scale
        out1 = self.low_postfilter(x[:, :, :self.stream_sizes[0] + self.padding], z)
        out2 = self.mid_postfilter(x[:, :, self.stream_sizes[0] - self.padding:sum(self.stream_sizes[:2]) + self.padding], z)
        out3 = self.high_postfilter(x[:, :, sum(self.stream_sizes[:2]) - self.padding:], z)
        out = torch.cat([out1, out2, out3], dim=-1)
        return out


class ZoneOutCell(nn.Module):

    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            if prob > 0.0:
                mask = h.new(*h.size()).bernoulli_(prob)
            else:
                mask = 0
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class NonAttentiveDecoder(BaseModel):
    """Decoder of Tacotron w/o attention mechanism

    Args:
        in_dim (int) : dimension of encoder hidden layer
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        attention_hidden_dim (int) : dimension of attention hidden layer
        attention_conv_channel (int) : number of attention convolution channels
        attention_conv_kernel_size (int) : kernel size of attention convolution
        downsample_by_conv (bool) : if True, downsample by convolution
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(self, in_dim=512, out_dim=80, layers=2, hidden_dim=1024, prenet_layers=2, prenet_hidden_dim=256, prenet_dropout=0.5, zoneout=0.1, reduction_factor=1, downsample_by_conv=False, init_type='none', eval_dropout=True, prenet_noise_std=0.0, initial_value=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.prenet_noise_std = prenet_noise_std
        self.initial_value = initial_value
        if prenet_layers > 0:
            self.prenet = Prenet(out_dim, prenet_layers, prenet_hidden_dim, prenet_dropout, eval_dropout=eval_dropout)
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim
        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(lstm_in_dim if layer == 0 else hidden_dim, hidden_dim)
            self.lstm += [ZoneOutCell(lstm, zoneout)]
        proj_in_dim = in_dim + hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)
        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(in_dim, in_dim, kernel_size=reduction_factor, stride=reduction_factor, groups=in_dim)
        else:
            self.conv_downsample = None
        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def is_autoregressive(self):
        return True

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        """Forward step

        Args:
            encoder_outs (torch.Tensor): encoder outputs (B, T, C)
            in_lens (torch.Tensor): input lengths
            decoder_targets (torch.Tensor): decoder targets for teacher-forcing. (B, T, C)

        Returns:
            torch.Tensor: the output (B, C, T)
        """
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[:, self.reduction_factor - 1::self.reduction_factor]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(encoder_outs.transpose(1, 2)).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[:, self.reduction_factor - 1::self.reduction_factor]
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim) + self.initial_value
        prev_out = go_frame
        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)
        outs = []
        for t in range(encoder_outs.shape[1]):
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            elif self.prenet_noise_std > 0:
                prenet_out = prev_out + torch.randn_like(prev_out) * self.prenet_noise_std
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](h_list[i - 1], (h_list[i], c_list[i]))
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            outs.append(self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1))
            if is_inference:
                prev_out = outs[-1][:, :, -1]
            else:
                prev_out = decoder_targets[:, t, :]
        outs = torch.cat(outs, dim=2)
        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.out_dim, -1)
        return outs.transpose(1, 2)


def mdn_get_sample(log_pi, log_sigma, mu):
    """Sample from mixture of the Gaussian component whose weight coefficient is
    the largest as the most probable predictions.

    Args:
        log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
            The log of multinomial distribution of the Gaussians.
            B is the batch size, T is data length of this batch,
            G is num_gaussians of class MDNLayer.
        log_sigma (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The log of standard deviation of the Gaussians.
            D_out is out_dim of class MDNLayer.
        mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The means of the Gaussians. D_out is out_dim of class MDNLayer.

    Returns:
        torch.Tensor: Tensor of shape (B, T, D_out)
            Sample from the mixture of the Gaussian component.
    """
    max_sigma, max_mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
    dist = torch.distributions.Normal(loc=max_mu, scale=max_sigma)
    sample = dist.sample()
    return sample


class MDNNonAttentiveDecoder(BaseModel):
    """Non-atteive decoder with MDN

    Each decoder step outputs the parameters of MDN.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        layers (int): number of LSTM layers
        hidden_dim (int): hidden dimension
        prenet_layers (int): number of prenet layers
        prenet_hidden_dim (int): prenet hidden dimension
        prenet_dropout (float): prenet dropout rate
        zoneout (float): zoneout rate
        reduction_factor (int): reduction factor
        downsample_by_conv (bool): if True, use conv1d to downsample the input
        num_gaussians (int): number of Gaussians
        sampling_mode (str): sampling mode
        init_type (str): initialization type
        eval_dropout (bool): if True, use dropout in evaluation
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(self, in_dim=512, out_dim=80, layers=2, hidden_dim=1024, prenet_layers=2, prenet_hidden_dim=256, prenet_dropout=0.5, zoneout=0.1, reduction_factor=1, downsample_by_conv=False, num_gaussians=8, sampling_mode='mean', init_type='none', eval_dropout=True, prenet_noise_std=0.0, initial_value=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.prenet_noise_std = prenet_noise_std
        self.num_gaussians = num_gaussians
        self.sampling_mode = sampling_mode
        assert sampling_mode in ['mean', 'random']
        self.initial_value = initial_value
        if prenet_layers > 0:
            self.prenet = Prenet(out_dim, prenet_layers, prenet_hidden_dim, prenet_dropout, eval_dropout=eval_dropout)
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim
        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(lstm_in_dim if layer == 0 else hidden_dim, hidden_dim)
            self.lstm += [ZoneOutCell(lstm, zoneout)]
        proj_in_dim = in_dim + hidden_dim
        self.feat_out = MDNLayer(proj_in_dim, out_dim * reduction_factor, num_gaussians=num_gaussians, dim_wise=True)
        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(in_dim, in_dim, kernel_size=reduction_factor, stride=reduction_factor, groups=in_dim)
        else:
            self.conv_downsample = None
        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def is_autoregressive(self):
        return True

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[:, self.reduction_factor - 1::self.reduction_factor]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(encoder_outs.transpose(1, 2)).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[:, self.reduction_factor - 1::self.reduction_factor]
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim) + self.initial_value
        prev_out = go_frame
        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)
        mus = []
        log_pis = []
        log_sigmas = []
        mus_inf = []
        for t in range(encoder_outs.shape[1]):
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            elif self.prenet_noise_std > 0:
                prenet_out = prev_out + torch.randn_like(prev_out) * self.prenet_noise_std
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](h_list[i - 1], (h_list[i], c_list[i]))
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            log_pi, log_sigma, mu = self.feat_out(hcs.unsqueeze(1))
            log_pi = log_pi.transpose(1, 2).view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim).transpose(1, 2)
            log_sigma = log_sigma.transpose(1, 2).view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim).transpose(1, 2)
            mu = mu.transpose(1, 2).view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim).transpose(1, 2)
            mus.append(mu)
            log_pis.append(log_pi)
            log_sigmas.append(log_sigma)
            if is_inference:
                if self.sampling_mode == 'mean':
                    _, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
                elif self.sampling_mode == 'random':
                    mu = mdn_get_sample(log_pi, log_sigma, mu)
                prev_out = mu[:, -1]
                mus_inf.append(mu)
            else:
                prev_out = decoder_targets[:, t, :]
        mus = torch.cat(mus, dim=1)
        log_pis = torch.cat(log_pis, dim=1)
        log_sigmas = torch.cat(log_sigmas, dim=1)
        if is_inference:
            mu = torch.cat(mus_inf, dim=1)
            return mu, mu
        else:
            return log_pis, log_sigmas, mus


class MultiHeadAttention(nn.Module):

    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = *key.size(), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, 'Relative attention is only available for self-attention.'
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias is only available for self-attention.'
            scores = scores + self._attention_bias_proximal(t_s)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10000.0)
            if self.block_length is not None:
                assert t_s == t_t, 'Local attention is only available for self-attention.'
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -10000.0)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class Encoder(nn.Module):

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Postnet(nn.Module):
    """Post-Net of Tacotron 2

    Args:
        in_dim (int): dimension of input
        layers (int): number of layers
        channels (int): number of channels
        kernel_size (int): kernel size
        dropout (float): dropout rate
    """

    def __init__(self, in_dim, layers=5, channels=512, kernel_size=5, dropout=0.5):
        super().__init__()
        postnet = nn.ModuleList()
        for layer in range(layers):
            in_channels = in_dim if layer == 0 else channels
            out_channels = in_dim if layer == layers - 1 else channels
            postnet += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm1d(out_channels)]
            if layer != layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, xs):
        """Forward step

        Args:
            xs (torch.Tensor): input sequence

        Returns:
            torch.Tensor: output sequence
        """
        return self.postnet(xs)


class SignalGenerator:
    """Input signal generator module."""

    def __init__(self, sample_rate=24000, hop_size=120, sine_amp=0.1, noise_amp=0.003, signal_types=['sine', 'noise']):
        """Initialize WaveNetResidualBlock module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if 'noise' == typ:
                signals.append(self.random_noise(f0))
            if 'sine' == typ:
                signals.append(self.sinusoid(f0))
            if 'uv' == typ:
                signals.append(self.vuv_binary(f0))
        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)
        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Gaussian noise signals (B, 1, T).

        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)
        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = interpolate(f0, T * self.hop_size) / self.sample_rate % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise
        return sine

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: V/UV binary sequences (B, 1, T).

        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        return uv


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor

    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle

    Return:
        dilated_factors(np array):
            float array of the pitch-dependent dilated factors (T)

    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(batch_f0.shape) * fs
    dilated_factors /= batch_f0
    dilated_factors /= dense_factor
    assert np.all(dilated_factors > 0)
    return dilated_factors


class USFGANWrapper(nn.Module):

    def __init__(self, config, generator):
        super().__init__()
        self.generator = generator
        self.config = config

    def inference(self, f0, aux_feats):
        """Inference for USFGAN

        Args:
            f0 (numpy.ndarray): F0 (T, 1)
            aux_feats (Tensor): Auxiliary features (T, C)

        """
        signal_generator = SignalGenerator(sample_rate=self.config.data.sample_rate, hop_size=self.config.data.hop_size, sine_amp=self.config.data.sine_amp, noise_amp=self.config.data.noise_amp, signal_types=self.config.data.signal_types)
        assert self.config.data.sine_f0_type in ['contf0', 'cf0', 'f0']
        assert self.config.data.df_f0_type in ['contf0', 'cf0', 'f0']
        device = aux_feats.device
        is_sifigan = 'aux_context_window' not in self.config.generator
        if is_sifigan:
            dfs = []
            for df, us in zip(self.config.data.dense_factors, np.cumprod(self.config.generator.upsample_scales)):
                dfs += [np.repeat(dilated_factor(f0.copy(), self.config.data.sample_rate, df), us)]
            df = [torch.FloatTensor(np.array(df)).view(1, 1, -1) for df in dfs]
            c = aux_feats.unsqueeze(0).transpose(2, 1)
        else:
            df = dilated_factor(np.squeeze(f0.copy()), self.config.data.sample_rate, self.config.data.dense_factor)
            df = df.repeat(self.config.data.hop_size, axis=0)
            pad_fn = nn.ReplicationPad1d(self.config.generator.aux_context_window)
            c = pad_fn(aux_feats.unsqueeze(0).transpose(2, 1))
            df = torch.FloatTensor(df).view(1, 1, -1)
        f0 = torch.FloatTensor(f0).unsqueeze(0).transpose(2, 1)
        in_signal = signal_generator(f0)
        y = self.generator(in_signal, c, df)[0]
        return y


class AdaptiveWindowing(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(self, sampling_rate, hop_size, fft_size, f0_floor, f0_ceil):
        """Initialize AdaptiveWindowing module.

        Args:
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.

        """
        super(AdaptiveWindowing, self).__init__()
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.register_buffer('window', torch.zeros((f0_ceil + 1, fft_size)))
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sampling_rate / f0)
            base_index = torch.arange(-half_win_len, half_win_len + 1, dtype=torch.int64)
            position = base_index / 1.5 / self.sampling_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window = torch.zeros(fft_size)
            window[left:right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = window / average

    def forward(self, x, f, power=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Waveform (B, fft_size // 2 + 1, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude.

        Returns:
            Tensor: Power spectrogram (B, bin_size, T').

        """
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f]
        x = torch.abs(torch.fft.rfft(x[:, :-1, :] * windows))
        x = x.pow(2) if power else x
        return x


class AdaptiveLiftering(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(self, sampling_rate, fft_size, f0_floor, f0_ceil, q1=-0.15):
        """Initialize AdaptiveLiftering module.

        Args:
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            q1 (float): Parameter to remove effect of adjacent harmonics.

        """
        super(AdaptiveLiftering, self).__init__()
        self.sampling_rate = sampling_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.register_buffer('smoothing_lifter', torch.zeros((f0_ceil + 1, self.bin_size)))
        self.register_buffer('compensation_lifter', torch.zeros((f0_ceil + 1, self.bin_size)))
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = torch.arange(1, self.bin_size) / sampling_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (math.pi * f0 * quefrency)
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(2.0 * math.pi * f0 * quefrency)
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f, elim_0th=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Power spectrogram (B, bin_size, T').
            f (Tensor): F0 sequence (B, T').
            elim_0th (bool): Whether to eliminate cepstram 0th component.

        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').

        """
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(torch.clamp(tmp, min=1e-07))).real
        if elim_0th:
            cepstrum[..., 0] = 0
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter
        x = torch.fft.irfft(liftered_cepstrum)[:, :, :self.bin_size]
        return x


class CheapTrick(nn.Module):
    """CheapTrick based spectral envelope estimation module."""

    def __init__(self, sampling_rate, hop_size, fft_size, f0_floor=70, f0_ceil=340, uv_threshold=0, q1=-0.15):
        """Initialize AdaptiveLiftering module.

        Args:
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            uv_threshold (float): V/UV determining threshold.
            q1 (float): Parameter to remove effect of adjacent harmonics.

        """
        super(CheapTrick, self).__init__()
        assert fft_size > 3.0 * sampling_rate / f0_floor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold
        self.ada_wind = AdaptiveWindowing(sampling_rate, hop_size, fft_size, f0_floor, f0_ceil)
        self.ada_lift = AdaptiveLiftering(sampling_rate, fft_size, f0_floor, f0_ceil, q1)

    def forward(self, x, f, power=False, elim_0th=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Power spectrogram (B, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to eliminate cepstram 0th component.

        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').

        """
        voiced = (f > self.uv_threshold) * torch.ones_like(f)
        f = voiced * f + (1.0 - voiced) * self.f0_ceil
        f = torch.round(torch.clamp(f, min=self.f0_floor, max=self.f0_ceil))
        x = self.ada_wind(x, f, power)
        x = self.ada_lift(x, f, elim_0th)
        return x


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1x1 Weight-normalized Conv1d layer."""
    return Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv2d1x1(Conv2d):
    """1x1 Conv2d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv2d module."""
        super(Conv2d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class FixedBlock(nn.Module):
    """Fixed block module in QPPWG."""

    def __init__(self, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, kernel_size=3, dilation=1, bias=True):
        """Initialize Fixed ResidualBlock module.

        Args:
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dilation (int): Dilation size.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super(FixedBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, padding=padding, padding_mode='reflect', dilation=dilation, bias=bias)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = self.conv(x)
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, s


class AdaptiveBlock(nn.Module):
    """Adaptive block module in QPPWG."""

    def __init__(self, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, bias=True):
        """Initialize Adaptive ResidualBlock module.

        Args:
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super(AdaptiveBlock, self).__init__()
        self.convP = Conv1d1x1(residual_channels, gate_channels, bias=bias)
        self.convC = Conv1d1x1(residual_channels, gate_channels, bias=bias)
        self.convF = Conv1d1x1(residual_channels, gate_channels, bias=bias)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, xC, xP, xF, c):
        """Calculate forward propagation.

        Args:
            xC (Tensor): Current input tensor (B, residual_channels, T).
            xP (Tensor): Past input tensor (B, residual_channels, T).
            xF (Tensor): Future input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = xC
        x = self.convC(xC) + self.convP(xP) + self.convF(xF)
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, s


logger = getLogger(__name__)


def pd_indexing(x, d, dilation, batch_index, ch_index):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.
        batch_index (Tensor): Batch index
        ch_index (Tensor): Channel index

    Returns:
        Tensor: Past output tensor (B, out_channels, T)
        Tensor: Future output tensor (B, out_channels, T)

    """
    _, _, batch_length = d.size()
    dilations = d * dilation
    idxP = torch.arange(-batch_length, 0).float()
    if torch.cuda.is_available():
        idxP = idxP
    idxP = torch.add(-dilations, idxP)
    idxP = idxP.round().long()
    maxP = -(torch.min(idxP) + batch_length)
    assert maxP >= 0
    idxP = batch_index, ch_index, idxP
    xP = pad1d((maxP, 0), 0)(x)
    idxF = torch.arange(0, batch_length).float()
    if torch.cuda.is_available():
        idxF = idxF
    idxF = torch.add(dilations, idxF)
    idxF = idxF.round().long()
    maxF = torch.max(idxF) - (batch_length - 1)
    assert maxF >= 0
    idxF = batch_index, ch_index, idxF
    xF = pad1d((0, maxF), 0)(x)
    return xP[idxP], xF[idxF]


class ResidualBlocks(nn.Module):
    """Multiple residual blocks stacking module."""

    def __init__(self, blockA, cycleA, blockF, cycleF, cascade_mode=0, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80):
        """Initialize ResidualBlocks module.

        Args:
            blockA (int): Number of adaptive residual blocks.
            cycleA (int): Number of dilation cycles of adaptive residual blocks.
            blockF (int): Number of fixed residual blocks.
            cycleF (int): Number of dilation cycles of fixed residual blocks.
            cascade_mode (int): Cascaded mode (0: Adaptive->Fixed; 1: Fixed->Adaptive).
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.

        """
        super(ResidualBlocks, self).__init__()
        cycleA = max(cycleA, 1)
        cycleF = max(cycleF, 1)
        assert blockA % cycleA == 0
        self.blockA_per_cycle = blockA // cycleA
        assert blockF % cycleF == 0
        blockF_per_cycle = blockF // cycleF
        adaptive_blocks = nn.ModuleList()
        for block in range(blockA):
            conv = AdaptiveBlock(residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels)
            adaptive_blocks += [conv]
        fixed_blocks = nn.ModuleList()
        for block in range(blockF):
            dilation = 2 ** (block % blockF_per_cycle)
            conv = FixedBlock(residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels, dilation=dilation)
            fixed_blocks += [conv]
        if cascade_mode == 0:
            self.conv_dilated = adaptive_blocks.extend(fixed_blocks)
            self.block_modes = [True] * blockA + [False] * blockF
        elif cascade_mode == 1:
            self.conv_dilated = fixed_blocks.extend(adaptive_blocks)
            self.block_modes = [False] * blockF + [True] * blockA
        else:
            logger.error(f'Cascaded mode {cascade_mode} is not supported!')
            sys.exit(0)

    def forward(self, x, c, d, batch_index, ch_index):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T).
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, residual_channels, T).

        """
        skips = 0
        blockA_idx = 0
        for f, mode in zip(self.conv_dilated, self.block_modes):
            if mode:
                dilation = 2 ** (blockA_idx % self.blockA_per_cycle)
                xP, xF = pd_indexing(x, d, dilation, batch_index, ch_index)
                x, h = f(x, xP, xF, c)
                blockA_idx += 1
            else:
                x, h = f(x, c)
            skips = h + skips
        skips *= math.sqrt(1.0 / len(self.conv_dilated))
        return x


class PeriodicityEstimator(nn.Module):
    """Periodicity estimator module."""

    def __init__(self, in_channels, residual_channels=64, conv_layers=3, kernel_size=5, dilation=1, padding_mode='replicate'):
        """Initialize USFGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            residual_channels (int): Number of channels in residual conv.
            conv_layers (int):  # Number of convolution layers.
            kernel_size (int): Kernel size.
            dilation (int): Dilation size.
            padding_mode (str): Padding mode.

        """
        super(PeriodicityEstimator, self).__init__()
        modules = []
        for idx in range(conv_layers):
            conv1d = Conv1d(in_channels, residual_channels, kernel_size=kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation, padding_mode=padding_mode)
            if idx != conv_layers - 1:
                nonlinear = nn.ReLU(inplace=True)
            else:
                nn.init.normal_(conv1d.weight, std=0.0001)
                nonlinear = nn.Sigmoid()
            modules += [conv1d, nonlinear]
            in_channels = residual_channels
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input auxiliary features (B, C ,T).

        Returns:
            Tensor: Output tensor (B, residual_channels, T).

        """
        return self.layers(x)


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode='nearest'):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.

        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, C, F, T).

        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale)

        """
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, use_causal_conv=False):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]
            assert (freq_axis_kernel_size - 1) % 2 == 0, 'Not support even number freq axis kernel size.'
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = freq_axis_kernel_size, scale * 2 + 1
            if use_causal_conv:
                padding = freq_axis_padding, scale * 2
            else:
                padding = freq_axis_padding, scale
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., :c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, aux_channels=80, aux_context_window=0, use_causal_conv=False):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.

        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales=upsample_scales, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, interpolate_mode=interpolate_mode, freq_axis_kernel_size=freq_axis_kernel_size, use_causal_conv=use_causal_conv)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T').

        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).

        Note:
            The length of inputs considers the context window size.

        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class PWGDiscriminator(nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=10, conv_channels=64, dilation_factor=1, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, bias=True, use_weight_norm=True):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(PWGDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        assert dilation_factor > 0, 'Dilation factor must be > 0.'
        self.conv_layers = nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [Conv1d(conv_in_channels, conv_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias), getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        conv_last_layer = Conv1d(conv_in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [conv_last_layer]
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        fmaps = []
        for f in self.conv_layers:
            x = f(x)
            if return_fmaps:
                fmaps.append(x)
        if return_fmaps:
            return [x], fmaps
        else:
            return [x]

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f'Weight norm is removed from {m}.')
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)


class HiFiGANPeriodDiscriminator(nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, period=3, kernel_sizes=[5, 3], channels=32, downsample_scales=[3, 3, 3, 3, 1], max_downsample_channels=1024, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, use_weight_norm=True, use_spectral_norm=False):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, 'Kernel size must be odd number.'
        assert kernel_sizes[1] % 2 == 1, 'Kernel size must be odd number.'
        self.period = period
        self.convs = nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [nn.Sequential(nn.Conv2d(in_chs, out_chs, (kernel_sizes[0], 1), (downsample_scale, 1), padding=((kernel_sizes[0] - 1) // 2, 0)), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = nn.Conv2d(out_chs, out_channels, (kernel_sizes[1] - 1, 1), 1, padding=((kernel_sizes[1] - 1) // 2, 0))
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Either use use_weight_norm or use_spectral_norm.')
        if use_weight_norm:
            self.apply_weight_norm()
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            list: List of each layer's tensors.

        """
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for f in self.convs:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        x = self.output_conv(x)
        out = torch.flatten(x, 1, -1)
        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logger.debug(f'Spectral norm is applied to {m}.')
        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(self, periods=[2, 3, 5, 7, 11], discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params['period'] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)
        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANScaleDiscriminator(nn.Module):
    """HiFi-GAN scale discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[15, 41, 5, 3], channels=128, max_downsample_channels=1024, max_groups=16, bias=True, downsample_scales=[2, 2, 4, 4, 1], nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, use_weight_norm=True, use_spectral_norm=False):
        """Initialize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.layers = nn.ModuleList()
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1
        self.layers += [nn.Sequential(nn.Conv1d(in_channels, channels, kernel_sizes[0], bias=bias, padding=(kernel_sizes[0] - 1) // 2), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
        in_chs = channels
        out_chs = channels
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [nn.Sequential(nn.Conv1d(in_chs, out_chs, kernel_size=kernel_sizes[1], stride=downsample_scale, padding=(kernel_sizes[1] - 1) // 2, groups=groups, bias=bias), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
            out_chs = min(in_chs * 2, max_downsample_channels)
            groups = min(groups * 4, max_groups)
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [nn.Sequential(nn.Conv1d(in_chs, out_chs, kernel_size=kernel_sizes[2], stride=1, padding=(kernel_sizes[2] - 1) // 2, bias=bias), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.last_layer = nn.Conv1d(out_chs, out_channels, kernel_size=kernel_sizes[3], stride=1, padding=(kernel_sizes[3] - 1) // 2, bias=bias)
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Either use use_weight_norm or use_spectral_norm.')
        if use_weight_norm:
            self.apply_weight_norm()
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of output tensors of each layer.

        """
        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        out = self.last_layer(x)
        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logger.debug(f'Spectral norm is applied to {m}.')
        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(nn.Module):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(self, scales=3, downsample_pooling='AvgPool1d', downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 2}, discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [15, 41, 5, 3], 'channels': 128, 'max_downsample_channels': 1024, 'max_groups': 16, 'bias': True, 'downsample_scales': [2, 2, 4, 4, 1], 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}}, follow_official_norm=False):
        """Initialize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementation. The first discriminator uses spectral norm and the other
                discriminators use weight norm.

        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params['use_weight_norm'] = False
                    params['use_spectral_norm'] = True
                else:
                    params['use_weight_norm'] = True
                    params['use_spectral_norm'] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(nn, downsample_pooling)(**downsample_pooling_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)
            x = self.pooling(x)
        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(self, scales=3, scale_downsample_pooling='AvgPool1d', scale_downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 2}, scale_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [15, 41, 5, 3], 'channels': 128, 'max_downsample_channels': 1024, 'max_groups': 16, 'bias': True, 'downsample_scales': [2, 2, 4, 4, 1], 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}}, follow_official_norm=True, periods=[2, 3, 5, 7, 11], period_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        """Initialize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementation. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(scales=scales, downsample_pooling=scale_downsample_pooling, downsample_pooling_params=scale_downsample_pooling_params, discriminator_params=scale_discriminator_params, follow_official_norm=follow_official_norm)
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods, discriminator_params=period_discriminator_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        if return_fmaps:
            msd_outs, msd_fmaps = self.msd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = msd_outs + mpd_outs
            fmaps = msd_fmaps + mpd_fmaps
            return outs, fmaps
        else:
            msd_outs = self.msd(x)
            mpd_outs = self.mpd(x)
            outs = msd_outs + mpd_outs
            return outs


class UnivNetSpectralDiscriminator(nn.Module):
    """UnivNet spectral discriminator module."""

    def __init__(self, fft_size, hop_size, win_length, window='hann_window', kernel_sizes=[(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)], strides=[(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)], channels=32, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, use_weight_norm=True):
        """Initialize HiFiGAN scale discriminator module.

        Args:
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            strides (list):
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))
        self.layers = nn.ModuleList()
        assert len(kernel_sizes) == len(strides)
        self.layers += [nn.Sequential(nn.Conv2d(1, channels, kernel_sizes[0], stride=strides[0], bias=bias), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
        for i in range(1, len(kernel_sizes) - 2):
            self.layers += [nn.Sequential(nn.Conv2d(channels, channels, kernel_size=kernel_sizes[i], stride=strides[i], bias=bias), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.layers += [nn.Sequential(nn.Conv2d(channels, channels, kernel_size=kernel_sizes[-2], stride=strides[-2], bias=bias), getattr(nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.layers += [nn.Conv2d(channels, 1, kernel_size=kernel_sizes[-1], stride=strides[-1], bias=bias)]
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of output tensors of each layer.

        """
        x = spectrogram(x, pad=self.win_length // 2, window=self.window, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length, power=1.0, normalized=False).transpose(-1, -2)
        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        if return_fmaps:
            return x, fmap
        else:
            return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)


class UnivNetMultiResolutionSpectralDiscriminator(nn.Module):
    """UnivNet multi-resolution spectral discriminator module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window', discriminator_params={'channels': 32, 'kernel_sizes': [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)], 'strides': [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)], 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.2}}):
        """Initialize UnivNetMultiResolutionSpectralDiscriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementation. The first discriminator uses spectral norm and the other
                discriminators use weight norm.

        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.discriminators = nn.ModuleList()
        for i in range(len(fft_sizes)):
            params = copy.deepcopy(discriminator_params)
            self.discriminators += [UnivNetSpectralDiscriminator(fft_size=fft_sizes[i], hop_size=hop_sizes[i], win_length=win_lengths[i], window=window, **params)]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)
        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class UnivNetMultiResolutionMultiPeriodDiscriminator(nn.Module):
    """UnivNet multi-resolution + multi-period discriminator module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window', spectral_discriminator_params={'channels': 32, 'kernel_sizes': [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)], 'strides': [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)], 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.2}}, periods=[2, 3, 5, 7, 11], period_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        """Initialize UnivNetMultiResolutionMultiPeriodDiscriminator module.

        Args:
            sperctral_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.mrd = UnivNetMultiResolutionSpectralDiscriminator(fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths, window=window, discriminator_params=spectral_discriminator_params)
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods, discriminator_params=period_discriminator_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        if return_fmaps:
            mrd_outs, mrd_fmaps = self.mrd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = mrd_outs + mpd_outs
            fmaps = mrd_fmaps + mpd_fmaps
            return outs, fmaps
        else:
            mrd_outs = self.mrd(x)
            mpd_outs = self.mpd(x)
            outs = mrd_outs + mpd_outs
            return outs


def index_initial(n_batch, n_ch, tensor=True):
    """Tensor batch and channel index initialization.

    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array

    Returns:
        Tensor: Batch index
        Tensor: Channel index

    """
    batch_index = []
    for i in range(n_batch):
        batch_index.append([[i]] * n_ch)
    ch_index = []
    for i in range(n_ch):
        ch_index += [[i]]
    ch_index = [ch_index] * n_batch
    if tensor:
        batch_index = torch.tensor(batch_index)
        ch_index = torch.tensor(ch_index)
        if torch.cuda.is_available():
            batch_index = batch_index
            ch_index = ch_index
    return batch_index, ch_index


class USFGANGenerator(nn.Module):
    """Unified Source-Filter GAN Generator module."""

    def __init__(self, source_network_params={'blockA': 30, 'cycleA': 3, 'blockF': 0, 'cycleF': 0, 'cascade_mode': 0}, filter_network_params={'blockA': 0, 'cycleA': 0, 'blockF': 30, 'cycleF': 3, 'cascade_mode': 0}, in_channels=1, out_channels=1, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, use_weight_norm=True, upsample_params={'upsample_scales': [5, 4, 3, 2]}):
        """Initialize USFGANGenerator module.

        Args:
            source_network_params (dict): Source-network parameters.
            filter_network_params (dict): Filter-network parameters.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            upsample_params (dict): Upsampling network parameters.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.n_ch = residual_channels
        self.conv_first = Conv1d1x1(in_channels, residual_channels)
        self.upsample_net = getattr(upsample, 'ConvInUpsampleNetwork')(**upsample_params, aux_channels=aux_channels, aux_context_window=aux_context_window)
        for params in [source_network_params, filter_network_params]:
            params.update({'residual_channels': residual_channels, 'gate_channels': gate_channels, 'skip_channels': skip_channels, 'aux_channels': aux_channels})
        self.source_network = ResidualBlocks(**source_network_params)
        self.filter_network = ResidualBlocks(**filter_network_params)
        self.conv_mid = Conv1d1x1(out_channels, skip_channels)
        self.conv_last = nn.Sequential(nn.ReLU(), Conv1d1x1(skip_channels, skip_channels), nn.ReLU(), Conv1d1x1(skip_channels, out_channels))
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        batch_index, ch_index = index_initial(x.size(0), self.n_ch)
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)
        x = self.conv_first(x)
        x = self.source_network(x, c, d, batch_index, ch_index)
        s = self.conv_last(x)
        x = self.conv_mid(s)
        x = self.filter_network(x, c, d, batch_index, ch_index)
        x = self.conv_last(x)
        return x, s

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f'Weight norm is removed from {m}.')
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)


class CascadeHnUSFGANGenerator(nn.Module):
    """Cascade hn-uSFGAN Generator module."""

    def __init__(self, harmonic_network_params={'blockA': 20, 'cycleA': 4, 'blockF': 0, 'cycleF': 0, 'cascade_mode': 0}, noise_network_params={'blockA': 0, 'cycleA': 0, 'blockF': 5, 'cycleF': 5, 'cascade_mode': 0}, filter_network_params={'blockA': 0, 'cycleA': 0, 'blockF': 30, 'cycleF': 3, 'cascade_mode': 0}, periodicity_estimator_params={'conv_blocks': 3, 'kernel_size': 5, 'dilation': 1, 'padding_mode': 'replicate'}, in_channels=1, out_channels=1, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, use_weight_norm=True, upsample_params={'upsample_scales': [5, 4, 3, 2]}):
        """Initialize CascadeHnUSFGANGenerator module.

        Args:
            harmonic_network_params (dict): Periodic source generation network parameters.
            noise_network_params (dict): Aperiodic source generation network parameters.
            filter_network_params (dict): Filter network parameters.
            periodicity_estimator_params (dict): Periodicity estimation network parameters.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            upsample_params (dict): Upsampling network parameters.

        """
        super(CascadeHnUSFGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.n_ch = residual_channels
        self.conv_first_sine = Conv1d1x1(in_channels, residual_channels)
        self.conv_first_noise = Conv1d1x1(in_channels, residual_channels)
        self.conv_merge = Conv1d1x1(residual_channels * 2, residual_channels)
        self.upsample_net = getattr(upsample, 'ConvInUpsampleNetwork')(**upsample_params, aux_channels=aux_channels, aux_context_window=aux_context_window)
        for params in [harmonic_network_params, noise_network_params, filter_network_params]:
            params.update({'residual_channels': residual_channels, 'gate_channels': gate_channels, 'skip_channels': skip_channels, 'aux_channels': aux_channels})
        self.harmonic_network = ResidualBlocks(**harmonic_network_params)
        self.noise_network = ResidualBlocks(**noise_network_params)
        self.filter_network = ResidualBlocks(**filter_network_params)
        self.periodicity_estimator = PeriodicityEstimator(**periodicity_estimator_params, in_channels=aux_channels)
        self.conv_last = nn.Sequential(nn.ReLU(), Conv1d1x1(skip_channels, skip_channels), nn.ReLU(), Conv1d1x1(skip_channels, out_channels))
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        batch_index, ch_index = index_initial(x.size(0), self.n_ch)
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)
        a = self.periodicity_estimator(c)
        sine, noise = torch.chunk(x, 2, 1)
        h = self.conv_first_sine(sine)
        n = self.conv_first_noise(noise)
        h = self.harmonic_network(h, c, d, batch_index, ch_index)
        h = a * h
        n = self.conv_merge(torch.cat([h, n], dim=1))
        n = self.noise_network(n, c, d, batch_index, ch_index)
        n = (1.0 - a) * n
        s = h + n
        x = self.filter_network(s, c, d, batch_index, ch_index)
        x = self.conv_last(x)
        s = self.conv_last(s)
        with torch.no_grad():
            h = self.conv_last(h)
            n = self.conv_last(n)
        return x, s, h, n, a

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f'Weight norm is removed from {m}.')
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)


class ParallelHnUSFGANGenerator(nn.Module):
    """Parallel hn-uSFGAN Generator module."""

    def __init__(self, harmonic_network_params={'blockA': 20, 'cycleA': 4, 'blockF': 0, 'cycleF': 0, 'cascade_mode': 0}, noise_network_params={'blockA': 0, 'cycleA': 0, 'blockF': 5, 'cycleF': 5, 'cascade_mode': 0}, filter_network_params={'blockA': 0, 'cycleA': 0, 'blockF': 30, 'cycleF': 3, 'cascade_mode': 0}, periodicity_estimator_params={'conv_blocks': 3, 'kernel_size': 5, 'dilation': 1, 'padding_mode': 'replicate'}, in_channels=1, out_channels=1, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, use_weight_norm=True, upsample_params={'upsample_scales': [5, 4, 3, 2]}):
        """Initialize ParallelHnUSFGANGenerator module.

        Args:
            harmonic_network_params (dict): Periodic source generation network parameters.
            noise_network_params (dict): Aperiodic source generation network parameters.
            filter_network_params (dict): Filter network parameters.
            periodicity_estimator_params (dict): Periodicity estimation network parameters.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ParallelHnUSFGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.n_ch = residual_channels
        self.conv_first_sine = Conv1d1x1(in_channels, residual_channels)
        self.conv_first_noise = Conv1d1x1(in_channels, residual_channels)
        self.upsample_net = getattr(upsample, 'ConvInUpsampleNetwork')(**upsample_params, aux_channels=aux_channels, aux_context_window=aux_context_window)
        for params in [harmonic_network_params, noise_network_params, filter_network_params]:
            params.update({'residual_channels': residual_channels, 'gate_channels': gate_channels, 'skip_channels': skip_channels, 'aux_channels': aux_channels})
        self.harmonic_network = ResidualBlocks(**harmonic_network_params)
        self.noise_network = ResidualBlocks(**noise_network_params)
        self.filter_network = ResidualBlocks(**filter_network_params)
        self.periodicity_estimator = PeriodicityEstimator(**periodicity_estimator_params, in_channels=aux_channels)
        self.conv_last = nn.Sequential(nn.ReLU(), Conv1d1x1(skip_channels, skip_channels), nn.ReLU(), Conv1d1x1(skip_channels, out_channels))
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        batch_index, ch_index = index_initial(x.size(0), self.n_ch)
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)
        a = self.periodicity_estimator(c)
        sine, noise = torch.chunk(x, 2, 1)
        h = self.conv_first_sine(sine)
        n = self.conv_first_noise(noise)
        h = self.harmonic_network(h, c, d, batch_index, ch_index)
        n = self.noise_network(n, c, d, batch_index, ch_index)
        h = a * h
        n = (1.0 - a) * n
        s = h + n
        x = self.filter_network(s, c, d, batch_index, ch_index)
        x = self.conv_last(x)
        s = self.conv_last(s)
        with torch.no_grad():
            h = self.conv_last(h)
            n = self.conv_last(n)
        return x, s, h, n, a

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f'Weight norm is removed from {m}.')
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)


class PyTorchStandardScaler(nn.Module):
    """PyTorch module for standardization.

    Args:
        mean (torch.Tensor): mean
        scale (torch.Tensor): scale
    """

    def __init__(self, mean, scale):
        super().__init__()
        self.mean_ = nn.Parameter(mean, requires_grad=False)
        self.scale_ = nn.Parameter(scale, requires_grad=False)

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


class ResSkipBlock(nn.Module):
    """Convolution block with residual and skip connections.

    Args:
        residual_channels (int): Residual connection channels.
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels.
        dilation (int): Dilation factor.
        cin_channels (int): Local conditioning channels.
        args (list): Additional arguments for Conv1d.
        kwargs (dict): Additional arguments for Conv1d.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels, dilation=1, cin_channels=80, *args, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, *args, padding=self.padding, dilation=dilation, **kwargs)
        self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels)

    def forward(self, x, c):
        """Forward step

        Args:
            x (torch.Tensor): Input signal.
            c (torch.Tensor): Local conditioning signal.

        Returns:
            tuple: Tuple of output signal and skip connection signal
        """
        return self._forward(x, c, False)

    def incremental_forward(self, x, c):
        """Incremental forward

        Args:
            x (torch.Tensor): Input signal.
            c (torch.Tensor): Local conditioning signal.

        Returns:
            tuple: Tuple of output signal and skip connection signal
        """
        return self._forward(x, c, True)

    def _forward(self, x, c, is_incremental):
        residual = x
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :-self.padding]
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        c = self._conv1x1_forward(self.conv1x1c, c, is_incremental)
        ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
        a, b = a + ca, b + cb
        x = torch.tanh(a) * torch.sigmoid(b)
        s = self._conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        x = self._conv1x1_forward(self.conv1x1_out, x, is_incremental)
        x = x + residual
        return x, s

    def _conv1x1_forward(self, conv, x, is_incremental):
        if is_incremental:
            x = conv.incremental_forward(x)
        else:
            x = conv(x)
        return x

    def clear_buffer(self):
        """Clear input buffer."""
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip, self.conv1x1c]:
            if c is not None:
                c.clear_buffer()


class WaveNet(nn.Module):
    """WaveNet

    Args:
        in_dim (int): the dimension of the input
        out_dim (int): the dimension of the output
        layers (int): the number of layers
        stacks (int): the number of residual stacks
        residual_channels (int): the number of residual channels
        gate_channels (int): the number of channels for the gating function
        skip_out_channels (int): the number of channels in the skip output
        kernel_size (int): the size of the convolutional kernel
    """

    def __init__(self, in_dim=334, out_dim=206, layers=10, stacks=1, residual_channels=64, gate_channels=128, skip_out_channels=64, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_conv = Conv1d1x1(out_dim, residual_channels)
        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResSkipBlock(residual_channels, gate_channels, kernel_size, skip_out_channels, dilation=dilation, cin_channels=in_dim)
            self.main_conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([nn.ReLU(), Conv1d1x1(skip_out_channels, skip_out_channels), nn.ReLU(), Conv1d1x1(skip_out_channels, out_dim)])

    def forward(self, c, x, lengths=None):
        """Forward step

        Args:
            c (torch.Tensor): the conditional features (B, T, C)
            x (torch.Tensor): the target features (B, T, C)

        Returns:
            torch.Tensor: the output waveform
        """
        x = x.transpose(1, 2)
        c = c.transpose(1, 2)
        x = self.first_conv(x)
        skips = 0
        for f in self.main_conv_layers:
            x, h = f(x, c)
            skips += h
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = x.transpose(1, 2)
        return x

    def inference(self, c, num_time_steps=100, tqdm=lambda x: x):
        """Inference step

        Args:
            c (torch.Tensor): the local conditioning feature (B, T, C)
            num_time_steps (int): the number of time steps to generate
            tqdm (lambda): a tqdm function to track progress

        Returns:
            torch.Tensor: the output waveform
        """
        self.clear_buffer()
        B = c.shape[0]
        outputs = []
        current_input = torch.zeros(B, 1, self.out_dim)
        if tqdm is None:
            ts = range(num_time_steps)
        else:
            ts = tqdm(range(num_time_steps))
        for t in ts:
            if t > 0:
                current_input = outputs[-1]
            ct = c[:, t, :].unsqueeze(1)
            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.main_conv_layers:
                x, h = f.incremental_forward(x, ct)
                skips += h
            x = skips
            for f in self.last_conv_layers:
                if hasattr(f, 'incremental_forward'):
                    x = f.incremental_forward(x)
                else:
                    x = f(x)
            x = F.softmax(x.view(B, -1), dim=1)
            x = torch.distributions.OneHotCategorical(x).sample()
            outputs += [x.data]
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1).contiguous()
        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        """Clear the internal buffer."""
        self.first_conv.clear_buffer()
        for f in self.main_conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def remove_weight_norm_(self):
        """Remove weight normalization of the model"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm1dTBC,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dPostFilter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (CustomSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'hidden_channels': 4, 'filter_channels': 4, 'n_heads': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (FFN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'filter_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FSLayerNorm,
     lambda: ([], {'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HiFiGANMultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (HiFiGANMultiScaleMultiPeriodDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (HiFiGANPeriodDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4])], {}),
     False),
    (HiFiGANScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (LSTMEncoder,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (LSTMRNN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (LSTMRNNSAR,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (LayerNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MDN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MDNLayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MDNv2,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MovingAverage1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'channels': 4, 'out_channels': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PitchPredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Postnet,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Prenet,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMDN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SinusoidalPositionalEmbedding,
     lambda: ([], {'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Stretch2d,
     lambda: ([], {'x_scale': 1.0, 'y_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TrTimeInvFIRFilter,
     lambda: ([], {'channels': 4, 'filt_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (TransformerFFNLayer,
     lambda: ([], {'hidden_size': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_nnsvs_nnsvs(_paritybench_base):
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

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

