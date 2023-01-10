import sys
_module = sys.modules[__name__]
del sys
gendocs = _module
download_mnist = _module
mnist_dbm = _module
mnist_dbm_layerwise = _module
mnist_grbm = _module
mnist_gtap = _module
mnist_hopfield = _module
mnist_rbm = _module
mnist_rbm_onehot = _module
mnist_tap = _module
mnist_util = _module
plotting = _module
paysage = _module
backends = _module
common = _module
python_backend = _module
matrix = _module
nonlinearity = _module
rand = _module
typedef = _module
pytorch_backend = _module
matrix = _module
nonlinearity = _module
rand = _module
typedef = _module
read_config = _module
select_backend = _module
batch = _module
hdf = _module
in_memory = _module
shuffle = _module
constraints = _module
factorization = _module
pca = _module
fit = _module
layerwise = _module
methods = _module
sgd = _module
layers = _module
bernoulli_layer = _module
gaussian_layer = _module
layer = _module
onehot_layer = _module
weights = _module
math_utils = _module
nearest_neighbors = _module
online_moments = _module
metrics = _module
generator_metrics = _module
model_assessment = _module
progress_monitor = _module
models = _module
dbm = _module
gradient_util = _module
graph = _module
initialize = _module
state = _module
optimizers = _module
penalties = _module
preprocess = _module
samplers = _module
schedules = _module
setup = _module
test_hdf = _module
test_in_memory = _module
test_shuffle = _module
test_save_read = _module
test_factorization = _module
test_layers = _module
test_math_utils = _module
test_penalties = _module
test_backends = _module
test_derivatives = _module
test_gaussian_1D = _module
test_preprocess = _module
test_rbm = _module
test_samplers = _module
test_tap_machine = _module

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


import numpy


import torch


from typing import Iterable


from typing import Tuple


from typing import Union


from typing import Dict


from typing import List


from numpy import ndarray


from torch import IntTensor as IntTensorCPU


from torch import ShortTensor as ShortTensorCPU


from torch import LongTensor as LongTensorCPU


from torch import ByteTensor as ByteTensorCPU


from torch import FloatTensor as FloatTensorCPU


from torch import DoubleTensor as DoubleTensorCPU


from torch.cuda import IntTensor as IntTensorGPU


from torch.cuda import ShortTensor as ShortTensorGPU


from torch.cuda import LongTensor as LongTensorGPU


from torch.cuda import ByteTensor as ByteTensorGPU


from torch.cuda import FloatTensor as FloatTensorGPU


from torch.cuda import DoubleTensor as DoubleTensorGPU


from numpy import allclose


from scipy import special


import numpy as np

