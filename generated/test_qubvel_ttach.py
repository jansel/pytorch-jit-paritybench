import sys
_module = sys.modules[__name__]
del sys
master = _module
setup = _module
test_base = _module
test_transforms = _module
ttach = _module
__version__ = _module
aliases = _module
base = _module
functional = _module
transforms = _module
wrappers = _module

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


from functools import partial


from typing import Optional


from typing import List


from typing import Union


from typing import Tuple


import torch.nn as nn


from typing import Mapping


class Merger:

    def __init__(self, type: str='mean', n: int=1):
        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))
        self.output = None
        self.type = type
        self.n = n

    def append(self, x):
        if self.type == 'tsharpen':
            x = x ** 0.5
        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_qubvel_ttach(_paritybench_base):
    pass
