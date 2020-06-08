import sys
_module = sys.modules[__name__]
del sys
contrastive = _module
net = _module
test_contrastive = _module
train_mnist = _module

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


import torch.nn


import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3
        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.
            MaxPool2d(2, stride=2), nn.Conv2d(20, 50, kernel_size=5), nn.
            MaxPool2d(2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU(
            inplace=True), nn.Linear(500, 10), nn.Linear(10, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_delijati_pytorch_siamese(_paritybench_base):
    pass
