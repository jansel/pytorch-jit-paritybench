import sys
_module = sys.modules[__name__]
del sys
core = _module
agent = _module
agent_single_process = _module
agents = _module
a3c = _module
a3c_single_process = _module
acer = _module
acer_single_process = _module
dqn = _module
empty = _module
env = _module
envs = _module
atari = _module
atari_ram = _module
gym = _module
lab = _module
memories = _module
episode_parameter = _module
episodic = _module
sequential = _module
memory = _module
model = _module
models = _module
a3c_cnn_dis = _module
a3c_mlp_con = _module
acer_cnn_dis = _module
acer_mlp_dis = _module
dqn_cnn = _module
dqn_mlp = _module
empty = _module
main = _module
optims = _module
helpers = _module
sharedAdam = _module
sharedRMSprop = _module
utils = _module
distributions = _module
factory = _module
init_weights = _module
options = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import random


import time


import math


import torch


import torch.optim as optim


from torch.autograd import Variable


import torch.nn.functional as F


import torch.multiprocessing as mp


from torch.autograd import grad


from torch.autograd import backward


import torch.nn as nn


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.logger = args.logger
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        if hasattr(args, 'enable_dueling'):
            self.enable_dueling = args.enable_dueling
            self.dueling_type = args.dueling_type
        if hasattr(args, 'enable_lstm'):
            self.enable_lstm = args.enable_lstm
        self.input_dims = {}
        self.input_dims[0] = args.hist_len
        self.input_dims[1] = args.state_shape
        self.output_dims = args.action_dim

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def print_model(self):
        self.logger.warning('<-----------------------------------> Model')
        self.logger.warning(self)

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()

    def forward(self, input):
        raise NotImplementedError('not implemented in base calss')

