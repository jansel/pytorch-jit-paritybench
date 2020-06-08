import sys
_module = sys.modules[__name__]
del sys
nima = _module
api = _module
clean_dataset = _module
cli = _module
common = _module
dataset = _module
emd_loss = _module
inference_model = _module
model = _module
trainer = _module
worker = _module
setup = _module
integration = _module
conftest = _module
test_api = _module
test_cli = _module
unit = _module
test_emd_loss = _module

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


import torch.nn as nn


class EDMLoss(nn.Module):

    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff
            ), 2)))
        return samplewise_emd.mean()


class NIMA(nn.Module):

    def __init__(self, base_model: nn.Module, input_features: int, drop_out:
        float):
        super(NIMA, self).__init__()
        self.base_model = base_model
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=
            drop_out), nn.Linear(input_features, 10), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_truskovskiyk_nima_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(EDMLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
