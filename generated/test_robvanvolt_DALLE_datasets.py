import sys
_module = sys.modules[__name__]
del sys
cc12m = _module
cc3m = _module
filtered_yfcc100m = _module
wit_clip_class = _module
wit_dtype = _module
wit_image_downloader = _module
wit_url_downloader = _module
openimages_labels = _module
openimages_narrative = _module
wit = _module
wit_clip = _module
wit_old = _module
setup = _module
clip_wit = _module
dataset_sanitycheck = _module
tokenizer_from_wds_or_text = _module
wds_create_legacy = _module
wds_create_shards = _module
wds_from_tfrecords = _module
wds_from_tfrecords_alternative = _module
wds_pytorchread = _module
wds_read = _module

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


import matplotlib.pyplot as plt


import matplotlib.image as mpimg


import torch.nn as nn


import tensorflow as tf

