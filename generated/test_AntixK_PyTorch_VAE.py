import sys
_module = sys.modules[__name__]
del sys
experiment = _module
models = _module
base = _module
beta_vae = _module
betatc_vae = _module
cat_vae = _module
cvae = _module
dfcvae = _module
dip_vae = _module
fvae = _module
gamma_vae = _module
hvae = _module
info_vae = _module
iwae = _module
joint_vae = _module
logcosh_vae = _module
lvae = _module
miwae = _module
mssim_vae = _module
swae = _module
twostage_vae = _module
types_ = _module
vampvae = _module
vanilla_vae = _module
vq_vae = _module
wae_mmd = _module
run = _module
bvae = _module
test_betatcvae = _module
test_cat_vae = _module
test_dfc = _module
test_dipvae = _module
test_fvae = _module
test_gvae = _module
test_hvae = _module
test_iwae = _module
test_joint_Vae = _module
test_logcosh = _module
test_lvae = _module
test_miwae = _module
test_mssimvae = _module
test_swae = _module
test_vae = _module
test_vq_vae = _module
test_wae = _module
text_cvae = _module
text_vamp = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch import nn


from abc import abstractmethod


import torch


from torch.nn import functional as F


import math


import numpy as np


from torch.distributions import Gamma


import torch.nn.init as init


import torch.nn.functional as F


from math import floor


from math import pi


from math import log


from torch.distributions import Normal


from math import exp


from torch import distributions as dist


def conv_out_shape(img_size):
    return floor((img_size + 2 - 3) / 2.0) + 1


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_AntixK_PyTorch_VAE(_paritybench_base):
    pass
