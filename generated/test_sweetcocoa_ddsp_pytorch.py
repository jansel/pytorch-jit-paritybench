import sys
_module = sys.modules[__name__]
del sys
filtered_noise = _module
harmonic_oscillator = _module
loudness_extractor = _module
crepe = _module
utils = _module
reverb = _module
audiodata = _module
mss_loss = _module
autoencoder = _module
decoder = _module
encoder = _module
radam = _module
test = _module
train = _module
logging = _module
trainer = _module
io = _module
trainer = _module

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


import torch


import torch.nn as nn


import torchaudio


import pandas as pd


from torch.utils.data import Dataset


import torch.nn.functional as F


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.utils.data.dataloader import DataLoader


import torch.optim as optim


from torch.optim import Adam


from torch.nn.utils import clip_grad_norm_


from torch.optim.lr_scheduler import ReduceLROnPlateau


from time import time


from collections import defaultdict


import logging


class FilteredNoise(nn.Module):

    def __init__(self, frame_length=64, attenuate_gain=0.01, device='cuda'):
        super(FilteredNoise, self).__init__()
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain

    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varient in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        
        """
        batch_num, frame_num, filter_coeff_length = z['H'].shape
        self.filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype=torch.float32), requires_grad=False)
        INPUT_FILTER_COEFFICIENT = z['H']
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT.unsqueeze(-1).expand(batch_num, frame_num, filter_coeff_length, 2).contiguous()
        ZERO_PHASE_FR_BANK[..., 1] = 0
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length, 2)
        zero_phase_ir_bank = torch.irfft(ZERO_PHASE_FR_BANK, 1, signal_sizes=(filter_coeff_length * 2 - 1,))
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.rfft(zero_paded_windowed_linear_phase_ir_bank, 1)
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype=torch.float32).view(-1, self.frame_length) * 2 - 1
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.rfft(zero_paded_noise, 1)
        FILTERED_NOISE = torch.zeros_like(ZERO_PADED_NOISE)
        FILTERED_NOISE[:, :, 0] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0] - ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1]
        FILTERED_NOISE[:, :, 1] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1] + ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0]
        filtered_noise = torch.irfft(FILTERED_NOISE, 1).view(batch_num, frame_num, -1) * self.attenuate_gain
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad=False).unsqueeze(1)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), overlap_add_filter, stride=self.frame_length, padding=0).squeeze(1)
        return output_signal


class HarmonicOscillator(nn.Module):

    def __init__(self, sr=16000, frame_length=64, attenuate_gain=0.02, device='cuda'):
        super(HarmonicOscillator, self).__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        self.device = device
        self.framerate_to_audiorate = nn.Upsample(scale_factor=self.frame_length, mode='linear', align_corners=False)

    def forward(self, z):
        """
        Compute Addictive Synthesis
        Argument: 
            z['f0'] : fundamental frequency envelope for each sample
                - dimension (batch_num, frame_rate_time_samples)
            z['c'] : harmonic distribution of partials for each sample 
                - dimension (batch_num, partial_num, frame_rate_time_samples)
            z['a'] : loudness of entire sound for each sample
                - dimension (batch_num, frame_rate_time_samples)
        Returns:
            addictive_output : synthesized sinusoids for each sample 
                - dimension (batch_num, audio_rate_time_samples)
        """
        fundamentals = z['f0']
        framerate_c_bank = z['c']
        num_osc = framerate_c_bank.shape[1]
        partial_mult = torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1)
        framerate_f0_bank = fundamentals.unsqueeze(-1).expand(-1, -1, num_osc).transpose(1, 2) * partial_mult
        mask_filter = (framerate_f0_bank < self.sr / 2).float()
        antialiased_framerate_c_bank = framerate_c_bank * mask_filter
        audiorate_f0_bank = self.framerate_to_audiorate(framerate_f0_bank)
        audiorate_phase_bank = torch.cumsum(audiorate_f0_bank / self.sr, 2)
        audiorate_a_bank = self.framerate_to_audiorate(antialiased_framerate_c_bank)
        sinusoid_bank = torch.sin(2 * np.pi * audiorate_phase_bank) * audiorate_a_bank * self.attenuate_gain
        framerate_loudness = z['a']
        audiorate_loudness = self.framerate_to_audiorate(framerate_loudness.unsqueeze(0)).squeeze(0)
        addictive_output = torch.sum(sinusoid_bank, 1) * audiorate_loudness
        return addictive_output


class LoudnessExtractor(nn.Module):

    def __init__(self, sr=16000, frame_length=64, attenuate_gain=2.0, device='cuda'):
        super(LoudnessExtractor, self).__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = self.frame_length * 5
        self.device = device
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype=torch.float32), requires_grad=False)

    def torch_A_weighting(self, FREQUENCIES, min_db=-45.0):
        """
        Compute A-weighting weights in Decibel scale (codes from librosa) and 
        transform into amplitude domain (with DB-SPL equation).
        
        Argument: 
            FREQUENCIES : tensor of frequencies to return amplitude weight
            min_db : mininum decibel weight. appropriate min_db value is important, as 
                exp/log calculation might raise numeric error with float32 type. 
        
        Returns:
            weights : tensor of amplitude attenuation weights corresponding to the FREQUENCIES tensor.
        """
        FREQUENCY_SQUARED = FREQUENCIES ** 2
        const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
        WEIGHTS_IN_DB = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(FREQUENCIES) - torch.log10(FREQUENCY_SQUARED + const[0]) - torch.log10(FREQUENCY_SQUARED + const[1]) - 0.5 * torch.log10(FREQUENCY_SQUARED + const[2]) - 0.5 * torch.log10(FREQUENCY_SQUARED + const[3]))
        if min_db is not None:
            WEIGHTS_IN_DB = torch.max(WEIGHTS_IN_DB, torch.tensor([min_db], dtype=torch.float32))
        weights = torch.exp(torch.log(torch.tensor([10.0], dtype=torch.float32)) * WEIGHTS_IN_DB / 10)
        return weights

    def forward(self, z):
        """
        Compute A-weighted Loudness Extraction
        Input:
            z['audio'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """
        input_signal = z['audio']
        paded_input_signal = nn.functional.pad(input_signal, (self.frame_length * 2, self.frame_length * 2))
        sliced_signal = paded_input_signal.unfold(1, self.n_fft, self.frame_length)
        sliced_windowed_signal = sliced_signal * self.smoothing_window
        SLICED_SIGNAL = torch.rfft(sliced_windowed_signal, 1, onesided=False)
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = torch.zeros(SLICED_SIGNAL.shape[:-1])
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL[:, :, :, 0] ** 2 + SLICED_SIGNAL[:, :, :, 1] ** 2
        freq_bin_size = self.sr / self.n_fft
        FREQUENCIES = torch.tensor([(freq_bin_size * i % (0.5 * self.sr)) for i in range(self.n_fft)])
        A_WEIGHTS = self.torch_A_weighting(FREQUENCIES)
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL_LOUDNESS_SPECTRUM * A_WEIGHTS
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS = torch.sqrt(torch.sum(A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM, 2)) / self.n_fft * self.attenuate_gain
        return A_WEIGHTED_SLICED_SIGNAL_LOUDNESS


class ConvBlock(nn.Module):

    def __init__(self, f, w, s, in_channels):
        super().__init__()
        p1 = (w - 1) // 2
        p2 = w - 1 - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        to_local_average_cents.mapping = torch.tensor(np.linspace(0, 7180, 360)) + 1997.379408437619
    if isinstance(salience, np.ndarray):
        salience = torch.from_numpy(salience)
    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = torch.sum(salience * to_local_average_cents.mapping[start:end])
        weight_sum = torch.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return torch.tensor([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])
    raise Exception('label should be either 1d or 2d Tensor')


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.

    * Note : This is NOT implemented with pytorch.
    """
    starting = np.ones(360) / 360
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]
    self_emission = 0.1
    emission = np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 360)
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = starting, transition, emission
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])
    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])


def to_freq(activation, viterbi=False):
    if viterbi:
        cents = to_viterbi_cents(activation.detach().numpy())
        cents = torch.tensor(cents)
    else:
        cents = to_local_average_cents(activation)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    return frequency


class CREPE(nn.Module):

    def __init__(self, model_capacity='full'):
        super().__init__()
        capacity_multiplier = {'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32}[model_capacity]
        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [(n * capacity_multiplier) for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        for i in range(len(self.layers)):
            f, w, s, in_channel = filters[i + 1], widths[i], strides[i], filters[i]
            self.add_module('conv%d' % i, ConvBlock(f, w, s, in_channel))
        self.linear = nn.Linear(64 * capacity_multiplier, 360)
        self.load_weight(model_capacity)
        self.eval()

    def load_weight(self, model_capacity):
        download_weights(model_capacity)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = 'crepe-{}.pth'.format(model_capacity)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__('conv%d' % i)(x)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def get_activation(self, audio, sr, center=True, step_size=10, batch_size=128):
        """     
        audio : (N,) or (C, N)
        """
        if sr != 16000:
            rs = torchaudio.transforms.Resample(sr, 16000)
            audio = rs(audio)
        if len(audio.shape) == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = audio.mean(dim=0)

        def get_frame(audio, step_size, center):
            if center:
                audio = nn.functional.pad(audio, pad=(512, 512))
            hop_length = int(16000 * step_size / 1000)
            n_frames = 1 + int((len(audio) - 1024) / hop_length)
            assert audio.dtype == torch.float32
            itemsize = 1
            frames = torch.as_strided(audio, size=(1024, n_frames), stride=(itemsize, hop_length * itemsize))
            frames = frames.transpose(0, 1).clone()
            frames -= torch.mean(frames, axis=1).unsqueeze(-1)
            frames /= torch.std(frames, axis=1).unsqueeze(-1)
            return frames
        frames = get_frame(audio, step_size, center)
        activation_stack = []
        device = self.linear.weight.device
        for i in range(0, len(frames), batch_size):
            f = frames[i:min(i + batch_size, len(frames))]
            f = f
            act = self.forward(f)
            activation_stack.append(act.cpu())
        activation = torch.cat(activation_stack, dim=0)
        return activation

    def predict(self, audio, sr, viterbi=False, center=True, step_size=10, batch_size=128):
        activation = self.get_activation(audio, sr, batch_size=batch_size, step_size=step_size)
        frequency = to_freq(activation, viterbi=viterbi)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence, activation

    def process_file(self, file, output=None, viterbi=False, center=True, step_size=10, save_plot=False, batch_size=128):
        try:
            audio, sr = torchaudio.load(file)
        except ValueError:
            None
        with torch.no_grad():
            time, frequency, confidence, activation = self.predict(audio, sr, viterbi=viterbi, center=center, step_size=step_size, batch_size=batch_size)
        time, frequency, confidence, activation = time.numpy(), frequency.numpy(), confidence.numpy(), activation.numpy()
        f0_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + '.f0.csv'
        f0_data = np.vstack([time, frequency, confidence]).transpose()
        np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',', header='time,frequency,confidence', comments='')
        if save_plot:
            import matplotlib.cm
            plot_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + '.activation.png'
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())
            imwrite(plot_file, (255 * image).astype(np.uint8))


class TrainableFIRReverb(nn.Module):

    def __init__(self, reverb_length=48000, device='cuda'):
        super(TrainableFIRReverb, self).__init__()
        self.reverb_length = reverb_length
        self.device = device
        self.fir = nn.Parameter(torch.rand(1, self.reverb_length, dtype=torch.float32) * 2 - 1, requires_grad=True)
        self.drywet = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        self.decay = nn.Parameter(torch.tensor([3.0], dtype=torch.float32), requires_grad=True)

    def forward(self, z):
        """
        Compute FIR Reverb
        Input:
            z['audio_synth'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """
        input_signal = z['audio_synth']
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.rfft(zero_pad_input_signal, 1)
        """ TODO 
        Not numerically stable decay method?
        """
        decay_envelope = torch.exp(-(torch.exp(self.decay) + 2) * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32))
        decay_fir = self.fir * decay_envelope
        ir_identity = torch.zeros(1, decay_fir.shape[-1])
        ir_identity[:, 0] = 1
        """ TODO
        Equal-loudness(intensity) crossfade between to ir.
        """
        final_fir = torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))
        FIR = torch.rfft(zero_pad_final_fir, 1)
        OUTPUT_SIGNAL = torch.zeros_like(INPUT_SIGNAL)
        OUTPUT_SIGNAL[:, :, 0] = INPUT_SIGNAL[:, :, 0] * FIR[:, :, 0] - INPUT_SIGNAL[:, :, 1] * FIR[:, :, 1]
        OUTPUT_SIGNAL[:, :, 1] = INPUT_SIGNAL[:, :, 0] * FIR[:, :, 1] + INPUT_SIGNAL[:, :, 1] * FIR[:, :, 0]
        output_signal = torch.irfft(OUTPUT_SIGNAL, 1)
        return output_signal


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft, alpha=1.0, overlap=0.75, eps=1e-07):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)

    def forward(self, x_pred, x_true):
        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())
        loss = linear_term + self.alpha * log_term
        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.

    Usage ::

    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)

    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    """

    def __init__(self, n_ffts: list, alpha=1.0, overlap=0.75, eps=1e-07, use_reverb=True):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        if use_reverb:
            self.signal_key = 'audio_reverb'
        else:
            self.signal_key = 'audio_synth'

    def forward(self, x_pred, x_true):
        if isinstance(x_pred, dict):
            x_pred = x_pred[self.signal_key]
        if isinstance(x_true, dict):
            x_true = x_true['audio']
        x_pred = x_pred[..., :x_true.shape[-1]]
        losses = [loss(x_pred, x_true) for loss in self.losses]
        return sum(losses).sum()


class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 

    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU

    constructor arguments :
        n_input : dimension of input
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)

    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace
        self.add_module(f'mlp_layer1', nn.Sequential(nn.Linear(n_input, n_units), nn.LayerNorm(normalized_shape=n_units), relu(inplace=self.inplace)))
        for i in range(2, n_layer + 1):
            self.add_module(f'mlp_layer{i}', nn.Sequential(nn.Linear(n_units, n_units), nn.LayerNorm(normalized_shape=n_units), relu(inplace=self.inplace)))

    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f'mlp_layer{i}')(x)
        return x


class Decoder(nn.Module):
    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp_f0 = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        if config.use_z:
            self.mlp_z = MLP(n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.num_mlp = 3
        else:
            self.num_mlp = 2
        self.gru = nn.GRU(input_size=self.num_mlp * config.mlp_units, hidden_size=config.gru_units, num_layers=1, batch_first=True, bidirectional=config.bidirectional)
        self.mlp_gru = MLP(n_input=config.gru_units * 2 if config.bidirectional else config.gru_units, n_units=config.mlp_units, n_layer=config.mlp_layers, inplace=True)
        self.dense_harmonic = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_filter = nn.Linear(config.mlp_units, config.n_freq)

    def forward(self, batch):
        f0 = batch['f0'].unsqueeze(-1)
        loudness = batch['loudness'].unsqueeze(-1)
        if self.config.use_z:
            z = batch['z']
            latent_z = self.mlp_z(z)
        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)
        if self.config.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)
        latent, h = self.gru(latent)
        latent = self.mlp_gru(latent)
        amplitude = self.dense_harmonic(latent)
        a = amplitude[..., 0]
        a = Decoder.modified_sigmoid(a)
        c = F.softmax(amplitude[..., 1:], dim=-1)
        H = self.dense_filter(latent)
        H = Decoder.modified_sigmoid(H)
        c = c.permute(0, 2, 1)
        return dict(f0=batch['f0'], a=a, c=c, H=H)

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)
        a = a.mul(2.0)
        a.add_(1e-07)
        return a


class Z_Encoder(nn.Module):

    def __init__(self, n_fft, hop_length, sample_rate=16000, n_mels=128, n_mfcc=30, gru_units=512, z_units=16, bidirectional=False):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=True, melkwargs=dict(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0))
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=gru_units, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units, z_units)

    def forward(self, batch):
        x = batch['audio']
        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x


class Encoder(nn.Module):
    """
    Encoder. 

    contains: Z_encoder, loudness extractor

    Constructor arguments:
        use_z : Bool, if True, Encoder will produce z as output.
        sample_rate=16000,
        z_units=16,
        n_fft=2048,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        bidirectional=False

    input(dict(audio, f0)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, frame)
        audio : raw audio w/ shape(B, time)

    output : a dict object which contains key-values below

        loudness : torch.tensor w/ shape(B, frame)
        f0 : same as input
        z : (optional) residual information. torch.tensor w/ shape(B, frame, z_units)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hop_length = int(config.sample_rate * config.frame_resolution)
        self.loudness_extractor = LoudnessExtractor(sr=config.sample_rate, frame_length=self.hop_length)
        if config.use_z:
            self.z_encoder = Z_Encoder(sample_rate=config.sample_rate, n_fft=config.n_fft, hop_length=self.hop_length, n_mels=config.n_mels, n_mfcc=config.n_mfcc, gru_units=config.gru_units, z_units=config.z_units, bidirectional=config.bidirectional)

    def forward(self, batch):
        batch['loudness'] = self.loudness_extractor(batch)
        if self.config.use_z:
            batch['z'] = self.z_encoder(batch)
        if self.config.sample_rate % self.hop_length != 0:
            batch['loudness'] = batch['loudness'][:, :batch['f0'].shape[-1]]
            batch['z'] = batch['z'][:, :batch['f0'].shape[-1]]
        return batch


class AutoEncoder(nn.Module):

    def __init__(self, config):
        """
        encoder_config
                use_z=False, 
                sample_rate=16000,
                z_units=16,
                n_fft=2048,
                hop_length=64,
                n_mels=128,
                n_mfcc=30,
                gru_units=512
        
        decoder_config
                mlp_units=512,
                mlp_layers=3,
                use_z=False,
                z_units=16,
                n_harmonics=101,
                n_freq=65,
                gru_units=512,

        components_config
                sample_rate
                hop_length
        """
        super().__init__()
        self.decoder = Decoder(config)
        self.encoder = Encoder(config)
        hop_length = frame_length = int(config.sample_rate * config.frame_resolution)
        self.harmonic_oscillator = HarmonicOscillator(sr=config.sample_rate, frame_length=hop_length)
        self.filtered_noise = FilteredNoise(frame_length=hop_length)
        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3)
        self.crepe = None
        self.config = config

    def forward(self, batch, add_reverb=True):
        """
        z

        input(dict(f0, z(optional), l)) : a dict object which contains key-values below
                f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
                z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
                loudness : torch.tensor w/ shape(B, time)
        """
        batch = self.encoder(batch)
        latent = self.decoder(batch)
        harmonic = self.harmonic_oscillator(latent)
        noise = self.filtered_noise(latent)
        audio = dict(harmonic=harmonic, noise=noise, audio_synth=harmonic + noise[:, :harmonic.shape[-1]])
        if self.config.use_reverb and add_reverb:
            audio['audio_reverb'] = self.reverb(audio)
        audio['a'] = latent['a']
        audio['c'] = latent['c']
        return audio

    def get_f0(self, x, sample_rate=16000, f0_threshold=0.5):
        """
        input:
            x = torch.tensor((1), wave sample)
        
        output:
            f0 : (n_frames, ). fundamental frequencies
        """
        if self.crepe is None:
            self.crepe = CREPE(self.config.crepe)
            for param in self.parameters():
                self.device = param.device
                break
            self.crepe = self.crepe
        self.eval()
        with torch.no_grad():
            time, f0, confidence, activation = self.crepe.predict(x, sr=sample_rate, viterbi=True, step_size=int(self.config.frame_resolution * 1000), batch_size=32)
            f0 = f0.float()
            f0[confidence < f0_threshold] = 0.0
            f0 = f0[:-1]
        return f0

    def reconstruction(self, x, sample_rate=16000, add_reverb=True, f0_threshold=0.5, f0=None):
        """
        input:
            x = torch.tensor((1), wave sample)
            f0 (if exists) = (num_frames, )

        output(dict):
            f0 : (n_frames, ). fundamental frequencies
            a : (n_frames, ). amplitudes
            c : (n_harmonics, n_frames). harmonic constants
            sig : (n_samples)
            audio_reverb : (n_samples + reverb, ). reconstructed signal
        """
        self.eval()
        with torch.no_grad():
            if f0 is None:
                f0 = self.get_f0(x, sample_rate=sample_rate, f0_threshold=f0_threshold)
            batch = dict(f0=f0.unsqueeze(0), audio=x)
            recon = self.forward(batch, add_reverb=add_reverb)
            for k, v in recon.items():
                recon[k] = v[0]
            recon['f0'] = f0
            return recon


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBlock,
     lambda: ([], {'f': 4, 'w': 4, 's': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (MLP,
     lambda: ([], {'n_input': 4, 'n_units': 4, 'n_layer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSSLoss,
     lambda: ([], {'n_ffts': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSSLoss,
     lambda: ([], {'n_fft': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sweetcocoa_ddsp_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

