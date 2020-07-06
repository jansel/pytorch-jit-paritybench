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

