import sys
_module = sys.modules[__name__]
del sys
main = _module

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


import torch.nn.functional as F


from torch import distributed


from torch import nn


from torch.utils import data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_narumiruna_pytorch_distributed_example(_paritybench_base):
    pass
    def test_000(self):
        self._check(Net(*[], **{}), [torch.rand([784, 784])], {})

