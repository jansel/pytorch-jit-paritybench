import sys
_module = sys.modules[__name__]
del sys
demo_superpoint = _module

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


import numpy as np


import time


import torch


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1
            )
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1,
            padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1,
            padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1,
            padding=1)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1,
            padding=0)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1,
            padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_magicleap_SuperPointPretrainedNetwork(_paritybench_base):
    pass
    def test_000(self):
        self._check(SuperPointNet(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

