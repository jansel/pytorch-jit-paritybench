import sys
_module = sys.modules[__name__]
del sys
cnn_layer_visualization = _module
deep_dream = _module
generate_class_specific_samples = _module
generate_regularized_class_specific_samples = _module
grad_times_image = _module
gradcam = _module
guided_backprop = _module
guided_gradcam = _module
integrated_gradients = _module
inverted_representation = _module
layer_activation_with_guided_backprop = _module
misc_functions = _module
scorecam = _module
smooth_grad = _module
vanilla_backprop = _module

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


import torch


from torch.optim import Adam


from torchvision import models


from torch.optim import SGD


from torch.autograd import Variable


from torch.nn import ReLU


import copy


import matplotlib.cm as mpl_color_map


import torch.nn.functional as F

