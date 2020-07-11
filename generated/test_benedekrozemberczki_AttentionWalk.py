import sys
_module = sys.modules[__name__]
del sys
attentionwalk = _module
main = _module
param_parser = _module
utils = _module

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


import numpy as np


import pandas as pd


class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see the paper.
    """

    def __init__(self, args, shapes):
        """
        Setting up the layer.
        :param args: Arguments object.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        self.args = args
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        half_dim = int(self.args.dimensions / 2)
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], half_dim))
        self.right_factors = torch.nn.Parameter(torch.Tensor(half_dim, self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0], 1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.right_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim=0)
        probs = self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_tensor = weighted_target_tensor * probs
        weighted_tar_mat = torch.sum(weighted_target_tensor, dim=0)
        weighted_tar_mat = weighted_tar_mat.view(self.shapes[1], self.shapes[2])
        estimate = torch.mm(self.left_factors, self.right_factors)
        loss_on_target = -weighted_tar_mat * torch.log(torch.sigmoid(estimate))
        loss_opposite = -adjacency_opposite * torch.log(1 - torch.sigmoid(estimate))
        loss_on_mat = self.args.num_of_walks * weighted_tar_mat.shape[0] * loss_on_target + loss_opposite
        abs_loss_on_mat = torch.abs(loss_on_mat)
        average_loss_on_mat = torch.mean(abs_loss_on_mat)
        norms = torch.mean(torch.abs(self.left_factors)) + torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.args.beta * self.attention.norm(2) ** 2
        loss = average_loss_on_mat + loss_on_regularization + self.args.gamma * norms
        return loss

