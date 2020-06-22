import sys
_module = sys.modules[__name__]
del sys
densities = _module
flow = _module
losses = _module
run_experiment = _module
utils = _module
visualization = _module

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


import torch


from torch import nn


from torch.nn import functional as F


from torch.autograd import Variable


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()
        self.transforms = nn.Sequential(*(PlanarFlow(dim) for _ in range(
            flow_length)))
        self.log_jacobians = nn.Sequential(*(PlanarFlowLogDetJacobian(t) for
            t in self.transforms))

    def forward(self, z):
        log_jacobians = []
        for transform, log_jacobian in zip(self.transforms, self.log_jacobians
            ):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)
        zk = z
        return zk, log_jacobians


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


def safe_log(z):
    return torch.log(z + 1e-07)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()
        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())


class FreeEnergyBound(nn.Module):

    def __init__(self, density):
        super().__init__()
        self.density = density

    def forward(self, zk, log_jacobians):
        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - safe_log(self.density(zk))).mean()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ex4sperans_variational_inference_with_normalizing_flows(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(FreeEnergyBound(*[], **{'density': ReLU()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(NormalizingFlow(*[], **{'dim': 4, 'flow_length': 4}), [torch.rand([4, 4])], {})

    def test_002(self):
        self._check(PlanarFlow(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

