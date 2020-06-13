import sys
_module = sys.modules[__name__]
del sys
circle_loss = _module
circle_loss_early = _module
mnist_example = _module

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


from typing import Tuple


import torch


from torch import nn


from torch import Tensor


from torch.optim import SGD


from torch.utils.data import DataLoader


class CircleLoss(nn.Module):

    def __init__(self, m: float, gamma: float) ->None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) ->Tensor:
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.
            logsumexp(logit_p, dim=0))
        return loss


class NormLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int) ->None:
        super(NormLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, inp: Tensor) ->Tensor:
        return nn.functional.linear(nn.functional.normalize(inp), nn.
            functional.normalize(self.weight))


class CircleLossLikeCE(nn.Module):

    def __init__(self, m: float, gamma: float) ->None:
        super(CircleLossLikeCE, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp: Tensor, label: Tensor) ->Tensor:
        a = torch.clamp_min(inp + self.m, min=0).detach()
        src = torch.clamp_min(-inp.gather(dim=1, index=label.unsqueeze(1)) +
            1 + self.m, min=0).detach()
        a.scatter_(dim=1, index=label.unsqueeze(1), src=src)
        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype
            ) * self.m
        src = torch.ones_like(label.unsqueeze(1), dtype=inp.dtype, device=
            inp.device) - self.m
        sigma.scatter_(dim=1, index=label.unsqueeze(1), src=src)
        return self.loss(a * (inp - sigma) * self.gamma, label)


class CircleLossBackward(nn.Module):

    def __init__(self, m: float, gamma: float) ->None:
        super(CircleLossBackward, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, sp: Tensor, sn: Tensor) ->Tensor:
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = torch.log(1 + torch.clamp_max(torch.exp(logit_n).sum() *
            torch.exp(logit_p).sum(), max=1e+38))
        z = -torch.exp(-loss) + 1
        """
        Eq. 10:
        sp.backward(gradient=z * (- ap) * torch.softmax(- logit_p, dim=0) * self.gamma, retain_graph=True)
        I modified it to 
        sp.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma, retain_graph=True)
        """
        sp.backward(gradient=z * -ap * torch.softmax(logit_p, dim=0) * self
            .gamma, retain_graph=True)
        sn.backward(gradient=z * an * torch.softmax(logit_n, dim=0) * self.
            gamma)
        return loss.detach()


class Model(nn.Module):

    def __init__(self) ->None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_channels=1,
            out_channels=8, kernel_size=5), nn.MaxPool2d(kernel_size=2), nn
            .ReLU(), nn.Conv2d(in_channels=8, out_channels=16, kernel_size=
            5), nn.MaxPool2d(kernel_size=2), nn.ReLU(), nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3), nn.MaxPool2d(
            kernel_size=2), nn.ReLU())

    def forward(self, inp: Tensor) ->Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_TinyZeaMays_CircleLoss(_paritybench_base):
    pass
    def test_000(self):
        self._check(CircleLoss(*[], **{'m': 4, 'gamma': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(NormLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Model(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

