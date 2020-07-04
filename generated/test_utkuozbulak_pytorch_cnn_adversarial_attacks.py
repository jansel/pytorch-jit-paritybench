import sys
_module = sys.modules[__name__]
del sys
fast_gradient_sign_targeted = _module
fast_gradient_sign_untargeted = _module
gradient_ascent_adv = _module
gradient_ascent_fooling = _module
misc_functions = _module

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


from torch.optim import SGD


from torchvision import models


from torch.nn import functional


import numpy as np


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_utkuozbulak_pytorch_cnn_adversarial_attacks(_paritybench_base):
    pass
