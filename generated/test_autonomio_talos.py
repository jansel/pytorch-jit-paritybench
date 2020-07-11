import sys
_module = sys.modules[__name__]
del sys
iris = _module
setup = _module
talos = _module
autom8 = _module
automodel = _module
autoparams = _module
autopredict = _module
autoscan = _module
commands = _module
analyze = _module
deploy = _module
evaluate = _module
predict = _module
restore = _module
logging = _module
logging_finish = _module
logging_run = _module
results = _module
metrics = _module
entropy = _module
keras_metrics = _module
model = _module
early_stopper = _module
hidden_layers = _module
ingest_model = _module
network_shape = _module
normalizers = _module
output_layer = _module
ParamSpace = _module
parameters = _module
GamifyMap = _module
reducers = _module
correlation = _module
forrest = _module
gamify = _module
limit_by_metric = _module
local_strategy = _module
reduce_run = _module
reduce_utils = _module
sample_reducer = _module
trees = _module
Scan = _module
scan = _module
scan_addon = _module
scan_finish = _module
scan_prepare = _module
scan_round = _module
scan_run = _module
scan_utils = _module
templates = _module
datasets = _module
models = _module
params = _module
pipelines = _module
utils = _module
best_model = _module
exceptions = _module
experiment_log_callback = _module
generator = _module
gpu_utils = _module
load_model = _module
power_draw_append = _module
power_draw_callback = _module
recover_best_model = _module
rescale_meanzero = _module
sequence_generator = _module
test_utils = _module
torch_history = _module
validation_split = _module
test = _module
test_analyze = _module
test_autom8 = _module
test_latest = _module
test_lr_normalizer = _module
test_predict = _module
test_random_methods = _module
test_reducers = _module
test_rest = _module
test_scan = _module
test_templates = _module
memory_pressure = _module
memory_pressure_check = _module
test_script = _module

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

