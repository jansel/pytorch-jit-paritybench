import sys
_module = sys.modules[__name__]
del sys
coreml_ane = _module
dcompile = _module
hwx_parse = _module
struct_recover = _module
new_patch = _module
ane = _module
testconv = _module
ops_ane = _module
console = _module
matmul = _module
cherry = _module
ops_cherry = _module
ops_cuda = _module
ops_llvm = _module
ops_metal = _module
ops_opencl = _module
preprocessing = _module
ops_rawcpu = _module
datasets = _module
imagenet = _module
benchmark_train_efficientnet = _module
efficientnet = _module
mnist_gan = _module
serious_mnist = _module
stable_diffusion = _module
train_efficientnet = _module
train_resnet = _module
transformer = _module
vgg7 = _module
vit = _module
kinne = _module
waifu2x = _module
yolo_nn = _module
yolov3 = _module
augment = _module
amx = _module
gemm = _module
gradcheck = _module
introspection = _module
onnx = _module
thneed = _module
training = _module
utils = _module
resnet = _module
compile = _module
setup = _module
test = _module
external_test_llvm = _module
external_test_opt = _module
graph_batchnorm = _module
test_cl_tiler = _module
test_conv = _module
test_efficientnet = _module
test_gc = _module
test_mnist = _module
test_net_speed = _module
test_nn = _module
test_onnx = _module
test_ops = _module
test_optim = _module
test_shapetracker = _module
test_speed_v_torch = _module
test_tensor = _module
test_train = _module
tinygrad = _module
graph = _module
helpers = _module
lazy = _module
ops_cpu = _module
ops_gpu = _module
ops_torch = _module
mlops = _module
nn = _module
optim = _module
ops = _module
shapetracker = _module
tensor = _module

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


import math


import time


from typing import Tuple


from typing import Union


from typing import Dict


from typing import Any


from typing import List


import numpy as np


from torchvision.utils import make_grid


from torchvision.utils import save_image


import torch


import re


from functools import lru_cache


from collections import namedtuple


import functools


from functools import partial


from enum import Enum


from typing import Type


from typing import NamedTuple

