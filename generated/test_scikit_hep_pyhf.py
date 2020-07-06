import sys
_module = sys.modules[__name__]
del sys
conf = _module
xref = _module
setup = _module
pyhf = _module
cli = _module
infer = _module
patchset = _module
rootio = _module
spec = _module
constraints = _module
contrib = _module
viz = _module
brazil = _module
events = _module
exceptions = _module
calculators = _module
mle = _module
test_statistics = _module
interpolators = _module
code0 = _module
code1 = _module
code2 = _module
code4 = _module
code4p = _module
mixins = _module
modifiers = _module
histosys = _module
lumi = _module
normfactor = _module
normsys = _module
shapefactor = _module
shapesys = _module
staterror = _module
optimize = _module
autodiff = _module
opt_jax = _module
opt_minuit = _module
opt_pytorch = _module
opt_scipy = _module
opt_tflow = _module
parameters = _module
paramsets = _module
paramview = _module
utils = _module
pdf = _module
probability = _module
readxml = _module
simplemodels = _module
tensor = _module
common = _module
jax_backend = _module
numpy_backend = _module
pytorch_backend = _module
tensorflow_backend = _module
version = _module
workspace = _module
writexml = _module
tests = _module
test_benchmark = _module
conftest = _module
test_viz = _module
test_backend_consistency = _module
test_combined_modifiers = _module
test_constraints = _module
test_events = _module
test_examples = _module
test_export = _module
test_import = _module
test_infer = _module
test_init = _module
test_interpolate = _module
test_mixins = _module
test_modifiers = _module
test_notebooks = _module
test_optim = _module
test_paramsets = _module
test_paramviewer = _module
test_patchset = _module
test_pdf = _module
test_probability = _module
test_public_api = _module
test_regression = _module
test_schema = _module
test_scripts = _module
test_tensor = _module
test_tensorviewer = _module
test_toys = _module
test_utils = _module
test_validation = _module
test_workspace = _module
makedata = _module
onoff = _module
run_cls = _module
run_single = _module
make_data = _module

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


import logging


import torch


import torch.autograd

