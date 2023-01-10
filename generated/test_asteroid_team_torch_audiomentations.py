import sys
_module = sys.modules[__name__]
del sys
demo = _module
measure_convolve_execution_time = _module
perf_benchmark = _module
plot = _module
setup = _module
tests = _module
test_audio = _module
test_background_noise = _module
test_band_pass_filter = _module
test_band_stop_filter = _module
test_base_class = _module
test_colored_noise = _module
test_compose = _module
test_config = _module
test_convolution = _module
test_differentiable = _module
test_file_utils = _module
test_gain = _module
test_high_pass_filter = _module
test_impulse_response = _module
test_low_pass_filter = _module
test_mel_utils = _module
test_mix = _module
test_one_of = _module
test_padding = _module
test_peak_normalization = _module
test_pitch_shift = _module
test_polarity_inversion = _module
test_random_crop = _module
test_shift = _module
test_shuffle_channels = _module
test_some_of = _module
test_spliceout = _module
test_time_inversion = _module
utils = _module
torch_audiomentations = _module
augmentations = _module
background_noise = _module
band_pass_filter = _module
band_stop_filter = _module
colored_noise = _module
gain = _module
high_pass_filter = _module
identity = _module
impulse_response = _module
low_pass_filter = _module
mix = _module
padding = _module
peak_normalization = _module
pitch_shift = _module
polarity_inversion = _module
random_crop = _module
shift = _module
shuffle_channels = _module
splice_out = _module
time_inversion = _module
core = _module
composition = _module
transforms_interface = _module
config = _module
convolution = _module
dsp = _module
fft = _module
file = _module
io = _module
mel_scale = _module
multichannel = _module
object_dict = _module

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


import random


import time


import numpy as np


import torch


from scipy.io import wavfile


from scipy.signal import convolve as scipy_convolve


import matplotlib.pyplot as plt


import pandas as pd


import re


import uuid


from scipy.io.wavfile import write


import types


from numpy.testing import assert_almost_equal


from numpy.testing import assert_array_equal


from torchaudio.transforms import Vol


from copy import deepcopy


from torch.optim import SGD


from numpy.testing import assert_equal


from typing import Union


from typing import List


from typing import Optional


from torch import Tensor


from math import ceil


from torch.nn.utils.rnn import pad_sequence


import typing


from random import choices


import warnings


import logging


from torch.nn.functional import pad


from typing import Tuple


import torch.nn


from torch.distributions import Bernoulli


from typing import Any


from typing import Dict


from typing import Text


import torchaudio


import math


class MultichannelAudioNotSupportedException(Exception):
    pass


def is_multichannel(samples) ->bool:
    return samples.shape[1] > 1


class RandomCrop(torch.nn.Module):
    """Crop the audio to predefined length in max_length."""
    supports_multichannel = True

    def __init__(self, max_length: float, sampling_rate: int, max_length_unit: str='seconds'):
        """
        :param max_length: length to which samples are to be cropped.
        :sampling_rate: sampling rate of input samples.
        :max_length_unit: defines the unit of max_length.
            "seconds": Number of seconds
            "samples": Number of audio samples
        """
        super(RandomCrop, self).__init__()
        self.sampling_rate = sampling_rate
        if max_length_unit == 'seconds':
            self.num_samples = int(self.sampling_rate * max_length)
        elif max_length_unit == 'samples':
            self.num_samples = int(max_length)
        else:
            raise ValueError('max_length_unit must be "samples" or "seconds"')

    def forward(self, samples, sampling_rate: typing.Optional[int]=None):
        sample_rate = sampling_rate or self.sampling_rate
        if sample_rate is None:
            raise RuntimeError('sample_rate is required')
        if len(samples) == 0:
            warnings.warn('An empty samples tensor was passed to {}'.format(self.__class__.__name__))
            return samples
        if len(samples.shape) != 3:
            raise RuntimeError('torch-audiomentations expects input tensors to be three-dimensional, with dimension ordering like [batch_size, num_channels, num_samples]. If your audio is mono, you can use a shape like [batch_size, 1, num_samples].')
        if is_multichannel(samples):
            if samples.shape[1] > samples.shape[2]:
                warnings.warn('Multichannel audio must have channels first, not channels last. In other words, the shape must be (batch size, channels, samples), not (batch_size, samples, channels)')
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException('{} only supports mono audio, not multichannel audio'.format(self.__class__.__name__))
        if samples.shape[2] < self.num_samples:
            warnings.warn('audio length less than cropping length')
            return samples
        start_indices = torch.randint(0, samples.shape[2] - self.num_samples, (samples.shape[2],))
        samples_cropped = torch.empty((samples.shape[0], samples.shape[1], self.num_samples), device=samples.device)
        for i, sample in enumerate(samples):
            samples_cropped[i] = sample.unsqueeze(0)[:, :, start_indices[i]:start_indices[i] + self.num_samples]
        return samples_cropped


class BaseCompose(torch.nn.Module):
    """This class can apply a sequence of transforms to waveforms."""

    def __init__(self, transforms: List[torch.nn.Module], shuffle: bool=False, p: float=1.0, p_mode='per_batch', output_type: Optional[str]=None):
        """
        :param transforms: List of waveform transform instances
        :param shuffle: Should the order of transforms be shuffled?
        :param p: The probability of applying the Compose to the given batch.
        :param p_mode: Only "per_batch" is supported at the moment.
        :param output_type: This optional argument can be set to "tensor" or "dict".
        """
        super().__init__()
        self.p = p
        if p_mode != 'per_batch':
            raise ValueError(f'p_mode = "{p_mode}" is not supported')
        self.p_mode = p_mode
        self.shuffle = shuffle
        self.are_parameters_frozen = False
        if output_type is None:
            warnings.warn(f"Transforms now expect an `output_type` argument that currently defaults to 'tensor', will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update your code to something like:\n  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n  >>> augmented_samples = augment(samples).samples", FutureWarning)
            output_type = 'tensor'
        elif output_type == 'tensor':
            warnings.warn(f"`output_type` argument will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update your code to something like:\nyour code to something like:\n  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n  >>> augmented_samples = augment(samples).samples", DeprecationWarning)
        self.output_type = output_type
        self.transforms = torch.nn.ModuleList(transforms)
        for tfm in self.transforms:
            tfm.output_type = 'dict'

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
        for transform in self.transforms:
            transform.unfreeze_parameters()

    @property
    def supported_modes(self) ->set:
        """Return the intersection of supported modes of the transforms in the composition."""
        currently_supported_modes = {'per_batch', 'per_example', 'per_channel'}
        for transform in self.transforms:
            currently_supported_modes = currently_supported_modes.intersection(transform.supported_modes)
        return currently_supported_modes


class ModeNotSupportedException(Exception):
    pass


_ObjectDictBase = typing.Dict[str, typing.Any]


class ObjectDict(_ObjectDictBase):
    """
    Make a dictionary behave like an object, with attribute-style access.

    Here are some examples of how it can be used:

    o = ObjectDict(my_dict)
    # or like this:
    o = ObjectDict(samples=samples, sample_rate=sample_rate)

    # Attribute-style access
    samples = o.samples

    # Dict-style access
    samples = o["samples"]
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class BaseWaveformTransform(torch.nn.Module):
    supported_modes = {'per_batch', 'per_example', 'per_channel'}
    supports_multichannel = True
    requires_sample_rate = True
    supports_target = True
    requires_target = False

    def __init__(self, mode: str='per_example', p: float=0.5, p_mode: Optional[str]=None, sample_rate: Optional[int]=None, target_rate: Optional[int]=None, output_type: Optional[str]=None):
        """

        :param mode:
            mode="per_channel" means each channel gets processed independently.
            mode="per_example" means each (multichannel) audio snippet gets processed
                independently, i.e. all channels in each audio snippet get processed with the
                same parameters.
            mode="per_batch" means all (multichannel) audio snippets in the batch get processed
                with the same parameters.
        :param p: The probability of the transform being applied to a batch/example/channel
            (see mode and p_mode). This number must be in the range [0.0, 1.0].
        :param p_mode: This optional argument can be set to "per_example" or "per_channel" if
            mode is set to "per_batch", or it can be set to "per_channel" if mode is set to
            "per_example". In the latter case, the transform is applied to the randomly selected
            examples, but the channels in those examples will be processed independently, i.e.
            with different parameters. Default value: Same as mode.
        :param sample_rate: sample_rate can be set either here or when
            calling the transform.
        :param target_rate: target_rate can be set either here or when
            calling the transform.
        :param output_type: This optional argument can be set to "tensor" or "dict".

        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.mode = mode
        self._p = p
        self.p_mode = p_mode
        if self.p_mode is None:
            self.p_mode = self.mode
        self.sample_rate = sample_rate
        self.target_rate = target_rate
        if output_type is None:
            warnings.warn(f"Transforms now expect an `output_type` argument that currently defaults to 'tensor', will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update your code to something like:\n  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n  >>> augmented_samples = augment(samples).samples", FutureWarning)
            output_type = 'tensor'
        elif output_type == 'tensor':
            warnings.warn(f"`output_type` argument will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update your code to something like:\nyour code to something like:\n  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n  >>> augmented_samples = augment(samples).samples", DeprecationWarning)
        self.output_type = output_type
        if self.mode not in self.supported_modes:
            raise ModeNotSupportedException('{} does not support mode {}'.format(self.__class__.__name__, self.mode))
        if self.p_mode == 'per_batch':
            assert self.mode in ('per_batch', 'per_example', 'per_channel')
        elif self.p_mode == 'per_example':
            assert self.mode in ('per_example', 'per_channel')
        elif self.p_mode == 'per_channel':
            assert self.mode == 'per_channel'
        else:
            raise Exception('Unknown p_mode {}'.format(self.p_mode))
        self.transform_parameters = {}
        self.are_parameters_frozen = False
        self.bernoulli_distribution = Bernoulli(self._p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        self.bernoulli_distribution = Bernoulli(self._p)

    def forward(self, samples: Tensor=None, sample_rate: Optional[int]=None, targets: Optional[Tensor]=None, target_rate: Optional[int]=None) ->ObjectDict:
        if not self.training:
            output = ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)
            return output.samples if self.output_type == 'tensor' else output
        if not isinstance(samples, Tensor) or len(samples.shape) != 3:
            raise RuntimeError('torch-audiomentations expects three-dimensional input tensors, with dimension ordering like [batch_size, num_channels, num_samples]. If your audio is mono, you can use a shape like [batch_size, 1, num_samples].')
        batch_size, num_channels, num_samples = samples.shape
        if batch_size * num_channels * num_samples == 0:
            warnings.warn('An empty samples tensor was passed to {}'.format(self.__class__.__name__))
            output = ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)
            return output.samples if self.output_type == 'tensor' else output
        if is_multichannel(samples):
            if num_channels > num_samples:
                warnings.warn('Multichannel audio must have channels first, not channels last. In other words, the shape must be (batch size, channels, samples), not (batch_size, samples, channels)')
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException('{} only supports mono audio, not multichannel audio'.format(self.__class__.__name__))
        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None and self.is_sample_rate_required():
            raise RuntimeError('sample_rate is required')
        if targets is None and self.is_target_required():
            raise RuntimeError('targets is required')
        has_targets = targets is not None
        if has_targets and not self.supports_target:
            warnings.warn(f'Targets are not (yet) supported by {self.__class__.__name__}')
        if has_targets:
            if not isinstance(targets, Tensor) or len(targets.shape) != 4:
                raise RuntimeError('torch-audiomentations expects four-dimensional target tensors, with dimension ordering like [batch_size, num_channels, num_frames, num_classes]. If your target is binary, you can use a shape like [batch_size, num_channels, num_frames, 1]. If your target is for the whole channel, you can use a shape like [batch_size, num_channels, 1, num_classes].')
            target_batch_size, target_num_channels, num_frames, num_classes = targets.shape
            if target_batch_size != batch_size:
                raise RuntimeError(f'samples ({batch_size}) and target ({target_batch_size}) batch sizes must be equal.')
            if num_channels != target_num_channels:
                raise RuntimeError(f'samples ({num_channels}) and target ({target_num_channels}) number of channels must be equal.')
            target_rate = target_rate or self.target_rate
            if target_rate is None:
                if num_frames > 1:
                    target_rate = round(sample_rate * num_frames / num_samples)
                    warnings.warn(f'target_rate is required by {self.__class__.__name__}. It has been automatically inferred from targets shape to {target_rate}. If this is incorrect, you can pass it directly.')
                else:
                    target_rate = 0
        if not self.are_parameters_frozen:
            if self.p_mode == 'per_example':
                p_sample_size = batch_size
            elif self.p_mode == 'per_channel':
                p_sample_size = batch_size * num_channels
            elif self.p_mode == 'per_batch':
                p_sample_size = 1
            else:
                raise Exception('Invalid mode')
            self.transform_parameters = {'should_apply': self.bernoulli_distribution.sample(sample_shape=(p_sample_size,))}
        if self.transform_parameters['should_apply'].any():
            cloned_samples = samples.clone()
            if has_targets:
                cloned_targets = targets.clone()
            else:
                cloned_targets = None
                selected_targets = None
            if self.p_mode == 'per_channel':
                cloned_samples = cloned_samples.reshape(batch_size * num_channels, 1, num_samples)
                selected_samples = cloned_samples[self.transform_parameters['should_apply']]
                if has_targets:
                    cloned_targets = cloned_targets.reshape(batch_size * num_channels, 1, num_frames, num_classes)
                    selected_targets = cloned_targets[self.transform_parameters['should_apply']]
                if not self.are_parameters_frozen:
                    self.randomize_parameters(samples=selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                perturbed: ObjectDict = self.apply_transform(samples=selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                cloned_samples[self.transform_parameters['should_apply']] = perturbed.samples
                cloned_samples = cloned_samples.reshape(batch_size, num_channels, num_samples)
                if has_targets:
                    cloned_targets[self.transform_parameters['should_apply']] = perturbed.targets
                    cloned_targets = cloned_targets.reshape(batch_size, num_channels, num_frames, num_classes)
                output = ObjectDict(samples=cloned_samples, sample_rate=perturbed.sample_rate, targets=cloned_targets, target_rate=perturbed.target_rate)
                return output.samples if self.output_type == 'tensor' else output
            elif self.p_mode == 'per_example':
                selected_samples = cloned_samples[self.transform_parameters['should_apply']]
                if has_targets:
                    selected_targets = cloned_targets[self.transform_parameters['should_apply']]
                if self.mode == 'per_example':
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(samples=selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                    perturbed: ObjectDict = self.apply_transform(samples=selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                    cloned_samples[self.transform_parameters['should_apply']] = perturbed.samples
                    if has_targets:
                        cloned_targets[self.transform_parameters['should_apply']] = perturbed.targets
                    output = ObjectDict(samples=cloned_samples, sample_rate=perturbed.sample_rate, targets=cloned_targets, target_rate=perturbed.target_rate)
                    return output.samples if self.output_type == 'tensor' else output
                elif self.mode == 'per_channel':
                    selected_batch_size, selected_num_channels, selected_num_samples = selected_samples.shape
                    assert selected_num_samples == num_samples
                    selected_samples = selected_samples.reshape(selected_batch_size * selected_num_channels, 1, selected_num_samples)
                    if has_targets:
                        selected_targets = selected_targets.reshape(selected_batch_size * selected_num_channels, 1, num_frames, num_classes)
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(samples=selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                    perturbed: ObjectDict = self.apply_transform(selected_samples, sample_rate=sample_rate, targets=selected_targets, target_rate=target_rate)
                    perturbed.samples = perturbed.samples.reshape(selected_batch_size, selected_num_channels, selected_num_samples)
                    cloned_samples[self.transform_parameters['should_apply']] = perturbed.samples
                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(selected_batch_size, selected_num_channels, num_frames, num_classes)
                        cloned_targets[self.transform_parameters['should_apply']] = perturbed.targets
                    output = ObjectDict(samples=cloned_samples, sample_rate=perturbed.sample_rate, targets=cloned_targets, target_rate=perturbed.target_rate)
                    return output.samples if self.output_type == 'tensor' else output
                else:
                    raise Exception('Invalid mode/p_mode combination')
            elif self.p_mode == 'per_batch':
                if self.mode == 'per_batch':
                    cloned_samples = cloned_samples.reshape(1, batch_size * num_channels, num_samples)
                    if has_targets:
                        cloned_targets = cloned_targets.reshape(1, batch_size * num_channels, num_frames, num_classes)
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(samples=cloned_samples, sample_rate=sample_rate, targets=cloned_targets, target_rate=target_rate)
                    perturbed: ObjectDict = self.apply_transform(samples=cloned_samples, sample_rate=sample_rate, targets=cloned_targets, target_rate=target_rate)
                    perturbed.samples = perturbed.samples.reshape(batch_size, num_channels, num_samples)
                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(batch_size, num_channels, num_frames, num_classes)
                    return perturbed.samples if self.output_type == 'tensor' else perturbed
                elif self.mode == 'per_example':
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(samples=cloned_samples, sample_rate=sample_rate, targets=cloned_targets, target_rate=target_rate)
                    perturbed = self.apply_transform(samples=cloned_samples, sample_rate=sample_rate, targets=cloned_targets, target_rate=target_rate)
                    return perturbed.samples if self.output_type == 'tensor' else perturbed
                elif self.mode == 'per_channel':
                    cloned_samples = cloned_samples.reshape(batch_size * num_channels, 1, num_samples)
                    if has_targets:
                        cloned_targets = cloned_targets.reshape(batch_size * num_channels, 1, num_frames, num_classes)
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(samples=cloned_samples, sample_rate=sample_rate, targets=cloned_targets, target_rate=target_rate)
                    perturbed: ObjectDict = self.apply_transform(cloned_samples, sample_rate, targets=cloned_targets, target_rate=target_rate)
                    perturbed.samples = perturbed.samples.reshape(batch_size, num_channels, num_samples)
                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(batch_size, num_channels, num_frames, num_classes)
                    return perturbed.samples if self.output_type == 'tensor' else perturbed
                else:
                    raise Exception('Invalid mode')
            else:
                raise Exception('Invalid p_mode {}'.format(self.p_mode))
        output = ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)
        return output.samples if self.output_type == 'tensor' else output

    def _forward_unimplemented(self, *inputs) ->None:
        pass

    def randomize_parameters(self, samples: Tensor=None, sample_rate: Optional[int]=None, targets: Optional[Tensor]=None, target_rate: Optional[int]=None):
        pass

    def apply_transform(self, samples: Tensor=None, sample_rate: Optional[int]=None, targets: Optional[Tensor]=None, target_rate: Optional[int]=None) ->ObjectDict:
        raise NotImplementedError()

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError()

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    def is_sample_rate_required(self) ->bool:
        return self.requires_sample_rate

    def is_target_required(self) ->bool:
        return self.requires_target


class Compose(BaseCompose):

    def forward(self, samples: Tensor=None, sample_rate: Optional[int]=None, targets: Optional[Tensor]=None, target_rate: Optional[int]=None) ->ObjectDict:
        inputs = ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)
        if random.random() < self.p:
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)
            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)
                else:
                    assert isinstance(tfm, torch.nn.Module)
                    inputs.samples = self.transforms[i](inputs.samples)
        return inputs.samples if self.output_type == 'tensor' else inputs


class SomeOf(BaseCompose):
    """
    SomeOf randomly picks several of the given transforms and applies them.
    The number of transforms to be applied can be chosen as follows:

      - Pick exactly n transforms
        Example: pick exactly 2 of the transforms
                 `SomeOf(2, [transform1, transform2, transform3])`

      - Pick between a minimum and maximum number of transforms
        Example: pick 1 to 3 of the transforms
                 `SomeOf((1, 3), [transform1, transform2, transform3])`

        Example: Pick 2 to all of the transforms
                 `SomeOf((2, None), [transform1, transform2, transform3])`
    """

    def __init__(self, num_transforms: Union[int, Tuple[int, int]], transforms: List[torch.nn.Module], p: float=1.0, p_mode='per_batch', output_type: Optional[str]=None):
        super().__init__(transforms=transforms, p=p, p_mode=p_mode, output_type=output_type)
        self.transform_indexes = []
        self.num_transforms = num_transforms
        self.all_transforms_indexes = list(range(len(self.transforms)))
        if isinstance(num_transforms, tuple):
            self.min_num_transforms = num_transforms[0]
            self.max_num_transforms = num_transforms[1] if num_transforms[1] else len(transforms)
        else:
            self.min_num_transforms = self.max_num_transforms = num_transforms
        assert self.min_num_transforms >= 1, 'min_num_transforms must be >= 1'
        assert self.min_num_transforms <= len(transforms), 'num_transforms must be <= len(transforms)'
        assert self.max_num_transforms <= len(transforms), 'max_num_transforms must be <= len(transforms)'

    def randomize_parameters(self):
        num_transforms_to_apply = random.randint(self.min_num_transforms, self.max_num_transforms)
        self.transform_indexes = sorted(random.sample(self.all_transforms_indexes, num_transforms_to_apply))

    def forward(self, samples: Tensor=None, sample_rate: Optional[int]=None, targets: Optional[Tensor]=None, target_rate: Optional[int]=None) ->ObjectDict:
        inputs = ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)
        if random.random() < self.p:
            if not self.are_parameters_frozen:
                self.randomize_parameters()
            for i in self.transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)
                else:
                    assert isinstance(tfm, torch.nn.Module)
                    inputs.samples = self.transforms[i](inputs.samples)
        return inputs.samples if self.output_type == 'tensor' else inputs


class OneOf(SomeOf):
    """
    OneOf randomly picks one of the given transforms and applies it.
    """

    def __init__(self, transforms: List[torch.nn.Module], p: float=1.0, p_mode='per_batch', output_type: Optional[str]=None):
        super().__init__(num_transforms=1, transforms=transforms, p=p, p_mode=p_mode, output_type=output_type)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseWaveformTransform,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (RandomCrop,
     lambda: ([], {'max_length': 4, 'sampling_rate': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_asteroid_team_torch_audiomentations(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

