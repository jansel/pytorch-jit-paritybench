import sys
_module = sys.modules[__name__]
del sys
callcenter = _module
pytorchloader = _module
callcenter_dataset = _module
esc = _module
esc_gen = _module
datasets = _module
esc_dataset = _module
esc_reader = _module
esc_to_tfrecords = _module
esc_utils = _module
gtzan = _module
gtzan_gen = _module
torch_readers = _module
gtzan = _module
gtzan_dataset = _module
librispeech = _module
tfrecord = _module
librispeech_reader = _module
librispeech_to_tfrecords = _module
constants = _module
dataloader_tfrecord = _module
dataset_h5py = _module
dataset_tfrecord = _module
librispeech_gen = _module
misc = _module
basic_dataset = _module
data_loader = _module
transforms = _module
utils = _module
nsynth = _module
nsynth_gen = _module
nsynth_reader = _module
nsynth_utils = _module
dataloader_tfrecord = _module
dataset_h5py = _module

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


import pandas as pd


from torch.utils import data


from torch.utils.data import DataLoader


import torch


import torchvision


from torchvision.transforms import ToPILImage


from torchvision.transforms import Pad


from torchvision.transforms import RandomCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import TenCrop


from torchvision.transforms import Lambda


import tensorflow as tf


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import OneHotEncoder

