import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
filternyt = _module
nyt = _module
main_att = _module
main_mil = _module
BasicModule = _module
PCNN_ATT = _module
PCNN_ONE = _module
models = _module
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


import torch


import numpy as np


import torch.nn as nn


from torch.utils.data import DataLoader


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


class BasicModule(torch.nn.Module):
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
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '.pth'
        torch.save(self.state_dict(), name)
        return name


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ShomyLiu_pytorch_relation_extraction(_paritybench_base):
    pass
