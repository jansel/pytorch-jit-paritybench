import sys
_module = sys.modules[__name__]
del sys
data = _module
constants = _module
core = _module
dsp = _module
audio_processing = _module
core = _module
stft = _module
torch_dsp = _module
g2p = _module
preprocess = _module
serialize = _module
inference = _module
inference_ob = _module
model = _module
core = _module
modules = _module
train = _module
trainer = _module
base = _module
core = _module
scheduler = _module
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


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import numpy as np


from scipy.signal import get_window


import torchaudio


from scipy.io.wavfile import read


import torch.nn.functional as F


from torch.autograd import Variable


from torchaudio.transforms import Resample


from torchaudio.transforms import Spectrogram


import warnings


from torch import nn


from torch.nn import functional as F


from torch.utils.tensorboard import SummaryWriter


from torch.optim.lr_scheduler import _LRScheduler


import random


def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0, max_wav_value=32768.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.frame_len = hop_length / sampling_rate * 1000
        self.max_wav_value = max_wav_value

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (1, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, _ = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        return self.lin2mel(magnitudes)

    def lin2mel(self, lin):
        if not torch.is_tensor(lin):
            lin = torch.from_numpy(lin)
        mel_output = torch.matmul(self.mel_basis, lin)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output.squeeze(0).transpose(0, 1)

    def load_wav(self, path, normalize=True):
        source_rate, audio = read(path)
        audio = torch.FloatTensor(audio.astype(np.float32))
        if audio.ndim > 1 and len(audio) > 1:
            audio = audio.mean(-1)
        if source_rate != self.sampling_rate:
            resample = torchaudio.transforms.Resample(source_rate, self.sampling_rate)
            audio = resample(audio)
        if normalize:
            audio = audio / self.max_wav_value
        audio = audio.unsqueeze(0)
        return audio


class FeedForward(nn.Module):

    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class ChannelMixer(nn.Module):

    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out


class TokenMixer(nn.Module):

    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(seq_len, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = x + residual
        return out


class MixerBlock(nn.Module):

    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(d_model, seq_len, expansion_factor, dropout)
        self.channel_mixer = ChannelMixer(d_model, expansion_factor, dropout)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, d_model=256, seq_len=256, expansion_factor=2, dropout=0.5, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(*[MixerBlock(d_model, seq_len, expansion_factor, dropout) for _ in range(num_layers)])

    def forward(self, x):
        return self.model(x)


class MLPSinger(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        d_model = config.d_pitch + config.d_phone
        self.text_embed = nn.Embedding(config.num_phone, config.d_phone)
        self.pitch_embed = nn.Embedding(config.num_pitch, config.d_pitch)
        self.embed = nn.Linear(d_model, d_model)
        self.decoder = MLPMixer(d_model=d_model, seq_len=config.seq_len, expansion_factor=config.expansion_factor, num_layers=config.num_layers, dropout=config.dropout)
        self.proj = nn.Linear(d_model, config.mel_dim)

    def forward(self, pitch, phonemes):
        pitch_embedding = self.pitch_embed(pitch)
        text_embedding = self.text_embed(phonemes)
        x = torch.cat((text_embedding, pitch_embedding), dim=-1)
        x = self.embed(x)
        x = self.decoder(x)
        out = self.proj(x)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChannelMixer,
     lambda: ([], {'d_model': 4, 'expansion_factor': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'num_features': 4, 'expansion_factor': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPMixer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     True),
    (MixerBlock,
     lambda: ([], {'d_model': 4, 'seq_len': 4, 'expansion_factor': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TokenMixer,
     lambda: ([], {'d_model': 4, 'seq_len': 4, 'expansion_factor': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_neosapience_mlp_singer(_paritybench_base):
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

