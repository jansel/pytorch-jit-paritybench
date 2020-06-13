import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
dataset = _module
main = _module
AlexNet = _module
BasicModule = _module
ResNet34 = _module
models = _module
utils = _module
visualize = _module

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


import torch as t


from torch import nn


from torch.nn import functional as F


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, 3,
            stride, 1, bias=False), nn.BatchNorm2d(outchannel), nn.ReLU(
            inplace=True), nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=
            False), nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_chenyuntc_pytorch_best_practice(_paritybench_base):
    pass
    def test_000(self):
        self._check(Flat(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ResidualBlock(*[], **{'inchannel': 4, 'outchannel': 4}), [torch.rand([4, 4, 4, 4])], {})

