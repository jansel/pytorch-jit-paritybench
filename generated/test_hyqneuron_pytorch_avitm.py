import sys
_module = sys.modules[__name__]
del sys
pytorch_model = _module
pytorch_run = _module
pytorch_visualize = _module
tf_model = _module
tf_run = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


from torch.nn import Parameter


import torch.nn.functional as F


import math


import numpy as np


from torch.autograd import Function


import torch.cuda


import matplotlib.pyplot as plt


from collections import defaultdict


class ProdLDA(nn.Module):

    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        self.en1_fc = nn.Linear(ac.num_input, ac.en1_units)
        self.en2_fc = nn.Linear(ac.en1_units, ac.en2_units)
        self.en2_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(ac.en2_units, ac.num_topic)
        self.mean_bn = nn.BatchNorm1d(ac.num_topic)
        self.logvar_fc = nn.Linear(ac.en2_units, ac.num_topic)
        self.logvar_bn = nn.BatchNorm1d(ac.num_topic)
        self.p_drop = nn.Dropout(0.2)
        self.decoder = nn.Linear(ac.num_topic, ac.num_input)
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)
        prior_mean = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)
        if ac.init_mult != 0:
            self.decoder.weight.data.uniform_(0, ac.init_mult)
        self.logvar_bn.register_parameter('weight', None)
        self.mean_bn.register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)

    def forward(self, input, compute_loss=False, avg_loss=True):
        en1 = F.softplus(self.en1_fc(input))
        en2 = F.softplus(self.en2_fc(en1))
        en2 = self.en2_drop(en2)
        posterior_mean = self.mean_bn(self.mean_fc(en2))
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))
        posterior_var = posterior_logvar.exp()
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_())
        z = posterior_mean + posterior_var.sqrt() * eps
        p = F.softmax(z)
        p = self.p_drop(p)
        recon = F.softmax(self.decoder_bn(self.decoder(p)))
        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        NL = -(input * (recon + 1e-10).log()).sum(1)
        prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic)
        loss = NL + KLD
        if avg:
            return loss.mean()
        else:
            return loss

