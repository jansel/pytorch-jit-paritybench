import sys
_module = sys.modules[__name__]
del sys
ali_bigan_pytorch = _module
ali_bigan_tensorflow = _module
ac_gan_pytorch = _module
ac_gan_tensorflow = _module
began_pytorch = _module
began_tensorflow = _module
bgan_pytorch = _module
bgan_tensorflow = _module
cgan_pytorch = _module
cgan_tensorflow = _module
cogan_pytorch = _module
cogan_tensorflow = _module
discogan_pytorch = _module
discogan_tensorflow = _module
dualgan_pytorch = _module
dualgan_tensorflow = _module
ebgan_pytorch = _module
ebgan_tensorflow = _module
f_gan_pytorch = _module
f_gan_tensorflow = _module
gap_pytorch = _module
gibbsnet_pytorch = _module
wgan_gp_tensorflow = _module
infogan_pytorch = _module
infogan_tensorflow = _module
lsgan_pytorch = _module
lsgan_tensorflow = _module
magan_pytorch = _module
magan_tensorflow = _module
mode_reg_gan_pytorch = _module
mode_reg_gan_tensorflow = _module
softmax_gan_pytorch = _module
softmax_gan_tensorflow = _module
gan_pytorch = _module
gan_tensorflow = _module
wgan_pytorch = _module
wgan_tensorflow = _module
helmholtz = _module
rbm_binary_cd = _module
rbm_binary_pcd = _module
aae_pytorch = _module
aae_tensorflow = _module
avb_pytorch = _module
avb_tensorflow = _module
cvae_pytorch = _module
cvae_tensorflow = _module
dvae_pytorch = _module
dvae_tensorflow = _module
vae_pytorch = _module
vae_tensorflow = _module

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


import torch


import torch.nn


import torch.nn.functional as nn


import torch.autograd as autograd


import torch.optim as optim


import numpy as np


from torch.autograd import Variable


from itertools import *


import copy


import scipy.ndimage.interpolation


from itertools import chain


import random


import tensorflow as tf

