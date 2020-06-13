import sys
_module = sys.modules[__name__]
del sys
tf_two_layer_net = _module
two_layer_net_autograd = _module
two_layer_net_custom_function = _module
build_readme = _module
dynamic_net = _module
two_layer_net_module = _module
two_layer_net_nn = _module
two_layer_net_optim = _module
two_layer_net_numpy = _module
two_layer_net_tensor = _module

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


import random


import torch


class DynamicNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
    In the constructor we construct three nn.Linear instances that we will use
    in the forward pass.
    """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
    For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
    and reuse the middle_linear Module that many times to compute hidden layer
    representations.

    Since each forward pass builds a dynamic computation graph, we can use normal
    Python control-flow operators like loops or conditional statements when
    defining the forward pass of the model.

    Here we also see that it is perfectly safe to reuse the same Module many
    times when defining a computational graph. This is a big improvement from Lua
    Torch, where each Module could be used only once.
    """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jcjohnson_pytorch_examples(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DynamicNet(*[], **{'D_in': 4, 'H': 4, 'D_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(TwoLayerNet(*[], **{'D_in': 4, 'H': 4, 'D_out': 4}), [torch.rand([4, 4, 4, 4])], {})

