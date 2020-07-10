import sys
_module = sys.modules[__name__]
del sys
conf = _module
example_heteromodal = _module
setup = _module
tests = _module
data = _module
inference = _module
test_grid_sampler = _module
test_inference = _module
sampler = _module
test_label_sampler = _module
test_weighted_sampler = _module
test_image = _module
test_images_dataset = _module
test_io = _module
test_queue = _module
test_subject = _module
datasets = _module
test_ixi = _module
test_cli = _module
test_utils = _module
transforms = _module
augmentation = _module
test_oneof = _module
test_random_affine = _module
test_random_elastic_deformation = _module
test_random_flip = _module
test_random_motion = _module
preprocessing = _module
test_crop_pad = _module
test_histogram_standardization = _module
test_resample = _module
test_collate = _module
test_lambda_transform = _module
test_transforms = _module
utils = _module
torchio = _module
cli = _module
dataset = _module
image = _module
aggregator = _module
grid_sampler = _module
io = _module
orientation = _module
queue = _module
label = _module
uniform = _module
weighted = _module
subject = _module
itk_snap = _module
ixi = _module
mni = _module
colin = _module
pediatric = _module
sheep = _module
slicer = _module
external = _module
due = _module
reference = _module
torchio = _module
composition = _module
intensity = _module
random_bias_field = _module
random_blur = _module
random_ghosting = _module
random_motion = _module
random_noise = _module
random_spike = _module
random_swap = _module
random_transform = _module
spatial = _module
random_affine = _module
random_elastic_deformation = _module
random_flip = _module
interpolation = _module
lambda_transform = _module
histogram_standardization = _module
normalization_transform = _module
rescale = _module
z_normalization = _module
bounds_transform = _module
crop = _module
crop_or_pad = _module
pad = _module
resample = _module
to_canonical = _module
transform = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from typing import List


import torch.nn as nn


from torch.utils.data import DataLoader


import torch


import numpy as np


from copy import deepcopy


import copy


import collections


from typing import Dict


from typing import Sequence


from typing import Optional


from typing import Callable


from torch.utils.data import Dataset


import warnings


from typing import Any


from typing import Tuple


from typing import Union


import random


from itertools import islice


from typing import Iterator


from typing import Generator


from torchvision.transforms import Compose as PyTorchCompose


import numbers


from numbers import Number


from abc import ABC


from abc import abstractmethod


from typing import Iterable

