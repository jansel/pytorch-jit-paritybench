import sys
_module = sys.modules[__name__]
del sys
plot_3d_to_2d = _module
plot_colin27 = _module
plot_custom_z_spacing = _module
plot_history = _module
plot_include_exclude = _module
plot_video = _module
conf = _module
print_system = _module
torchio = _module
cli = _module
apply_transform = _module
print_info = _module
constants = _module
data = _module
dataset = _module
image = _module
inference = _module
aggregator = _module
io = _module
queue = _module
sampler = _module
grid = _module
label = _module
sampler = _module
uniform = _module
weighted = _module
subject = _module
datasets = _module
bite = _module
episurg = _module
fpg = _module
itk_snap = _module
ixi = _module
medmnist = _module
mni = _module
colin = _module
icbm = _module
pediatric = _module
sheep = _module
rsna_miccai = _module
rsna_spine_fracture = _module
slicer = _module
visible_human = _module
download = _module
external = _module
due = _module
reference = _module
transforms = _module
augmentation = _module
composition = _module
intensity = _module
random_bias_field = _module
random_blur = _module
random_gamma = _module
random_ghosting = _module
random_labels_to_image = _module
random_motion = _module
random_noise = _module
random_spike = _module
random_swap = _module
random_transform = _module
spatial = _module
random_affine = _module
random_anisotropy = _module
random_elastic_deformation = _module
random_flip = _module
data_parser = _module
fourier = _module
intensity_transform = _module
interpolation = _module
lambda_transform = _module
preprocessing = _module
clamp = _module
histogram_standardization = _module
mask = _module
normalization_transform = _module
rescale = _module
z_normalization = _module
contour = _module
keep_largest_component = _module
label_transform = _module
one_hot = _module
remap_labels = _module
remove_labels = _module
sequential_labels = _module
bounds_transform = _module
copy_affine = _module
crop = _module
crop_or_pad = _module
ensure_shape_multiple = _module
pad = _module
resample = _module
resize = _module
to_canonical = _module
spatial_transform = _module
transform = _module
typing = _module
utils = _module
visualization = _module
tests = _module
test_aggregator = _module
test_grid_sampler = _module
test_inference = _module
test_label_sampler = _module
test_patch_sampler = _module
test_random_sampler = _module
test_uniform_sampler = _module
test_weighted_sampler = _module
test_image = _module
test_io = _module
test_queue = _module
test_subject = _module
test_subjects_dataset = _module
test_ixi = _module
test_medmnist = _module
test_cli = _module
test_utils = _module
test_oneof = _module
test_random_affine = _module
test_random_anisotropy = _module
test_random_bias_field = _module
test_random_blur = _module
test_random_elastic_deformation = _module
test_random_flip = _module
test_random_gamma = _module
test_random_ghosting = _module
test_random_labels_to_image = _module
test_random_motion = _module
test_random_noise = _module
test_random_spike = _module
test_random_swap = _module
test_remap_labels = _module
test_remove_labels = _module
test_sequential_labels = _module
test_clamp = _module
test_contour = _module
test_copy_affine = _module
test_crop = _module
test_crop_pad = _module
test_ensure_shape_multiple = _module
test_histogram_standardization = _module
test_keep_largest = _module
test_mask = _module
test_one_hot = _module
test_pad = _module
test_resample = _module
test_rescale = _module
test_resize = _module
test_to_canonical = _module
test_z_normalization = _module
test_collate = _module
test_invertibility = _module
test_lambda_transform = _module
test_reproducibility = _module
test_transforms = _module
utils = _module
example_heteromodal = _module

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


import matplotlib.animation as animation


import numpy as np


from typing import List


import re


import numpy


import copy


from typing import Callable


from typing import Iterable


from typing import Optional


from typing import Sequence


from torch.utils.data import Dataset


import warnings


from collections import Counter


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Union


from itertools import islice


from typing import Iterator


from torch.utils.data import DataLoader


from typing import Generator


from typing import TYPE_CHECKING


from torch.hub import tqdm


from collections import defaultdict


import scipy.ndimage as ndi


from numbers import Number


from typing import TypeVar


import torch.nn.functional as F


from collections.abc import Iterable


from typing import Sized


import numbers


from abc import ABC


from abc import abstractmethod


from torch.utils.data._utils.collate import default_collate


import random


from random import shuffle


from typing import Set


import logging


import torch.nn as nn

