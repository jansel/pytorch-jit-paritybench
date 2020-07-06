import sys
_module = sys.modules[__name__]
del sys
acquisition_functions = _module
bo = _module
acq = _module
acqmap = _module
acqopt = _module
acquisition = _module
probo = _module
dom = _module
list = _module
real = _module
ds = _module
makept = _module
fn = _module
functionhandler = _module
pp = _module
gp = _module
gp_utils = _module
pp_core = _module
pp_gp_george = _module
pp_gp_my_distmat = _module
pp_gp_stan = _module
pp_gp_stan_distmat = _module
stan = _module
compile_stan = _module
gp_distmat = _module
gp_distmat_fixedsig = _module
gp_hier2 = _module
gp_hier2_matern = _module
gp_hier3 = _module
util = _module
datatransform = _module
print_utils = _module
darts = _module
arch = _module
data = _module
meta_neural_net = _module
metann_runner = _module
nas_algorithms = _module
nas_bench = _module
cell = _module
params = _module
run_experiments_sequential = _module
train_arch_runner = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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

