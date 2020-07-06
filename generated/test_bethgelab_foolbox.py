import sys
_module = sys.modules[__name__]
del sys
conf = _module
evaluate = _module
pytorch_resnet18 = _module
spatial_attack = _module
substituion_model = _module
tensorflow_resnet50 = _module
foolbox_model = _module
foolbox = _module
attacks = _module
additive_noise = _module
base = _module
basic_iterative_method = _module
binarization = _module
blended_noise = _module
blur = _module
boundary_attack = _module
brendel_bethge = _module
carlini_wagner = _module
contrast = _module
contrast_min = _module
dataset_attack = _module
ddn = _module
deepfool = _module
ead = _module
fast_gradient_method = _module
gen_attack = _module
gen_attack_utils = _module
gradient_descent_base = _module
inversion = _module
newtonfool = _module
projected_gradient_descent = _module
saltandpepper = _module
sparse_l1_descent_attack = _module
spatial_attack_transformations = _module
virtual_adversarial_attack = _module
criteria = _module
devutils = _module
distances = _module
gradient_estimators = _module
models = _module
jax = _module
numpy = _module
pytorch = _module
tensorflow = _module
wrappers = _module
plot = _module
tensorboard = _module
types = _module
utils = _module
zoo = _module
common = _module
git_cloner = _module
model_loader = _module
weights_fetcher = _module
setup = _module
conftest = _module
test_attacks = _module
test_attacks_base = _module
test_attacks_raise = _module
test_binarization_attack = _module
test_brendel_bethge_attack = _module
test_criteria = _module
test_dataset_attack = _module
test_devutils = _module
test_distances = _module
test_evaluate = _module
test_fetch_weights = _module
test_gen_attack_utils = _module
test_models = _module
test_plot = _module
test_spatial_attack = _module
test_tensorboard = _module
test_utils = _module
test_zoo = _module

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


import torch


import torch.nn as nn


from typing import Union


from typing import List


from typing import Tuple


from typing import Any


import numpy as np


import math


from typing import cast


import warnings


from typing import Optional


from typing import Callable


from typing import Dict


import functools


import copy

