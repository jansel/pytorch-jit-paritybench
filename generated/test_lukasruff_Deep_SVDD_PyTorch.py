import sys
_module = sys.modules[__name__]
del sys
src = _module
base = _module
base_dataset = _module
base_net = _module
base_trainer = _module
torchvision_dataset = _module
datasets = _module
cifar10 = _module
main = _module
mnist = _module
preprocessing = _module
deepSVDD = _module
networks = _module
cifar10_LeNet = _module
cifar10_LeNet_elu = _module
mnist_LeNet = _module
optim = _module
ae_trainer = _module
deepSVDD_trainer = _module
utils = _module
collect_results = _module
config = _module
plot_images_grid = _module

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


import logging


import torch.nn as nn


import numpy as np


import torch


import torch.nn.functional as F


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lukasruff_Deep_SVDD_PyTorch(_paritybench_base):
    pass
