import sys
_module = sys.modules[__name__]
del sys
gwnn = _module
gwnn_layer = _module
main = _module
param_parser = _module
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


import time


import torch


class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """

    def __init__(self, in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.
            in_channels, self.out_channels))
        self.diagonal_weight_indices = torch.LongTensor([[node for node in
            range(self.ncount)], [node for node in range(self.ncount)]])
        self.diagonal_weight_indices = self.diagonal_weight_indices.to(self
            .device)
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.
            ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_benedekrozemberczki_GraphWaveletNeuralNetwork(_paritybench_base):
    pass
