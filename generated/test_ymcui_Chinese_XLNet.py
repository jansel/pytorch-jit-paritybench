import sys
_module = sys.modules[__name__]
del sys
src = _module
classifier_utils = _module
cmrc2018_evaluate_drcd = _module
data_utils = _module
function_builder = _module
gpu_utils = _module
model_utils = _module
modeling = _module
prepro_utils = _module
run_classifier = _module
run_cmrc_drcd = _module
squad_utils = _module
summary = _module
tpu_estimator = _module
xlnet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ymcui_Chinese_XLNet(_paritybench_base):
    pass
