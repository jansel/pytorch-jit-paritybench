import sys
_module = sys.modules[__name__]
del sys
ptstat = _module
core = _module
dist = _module
bernoulli = _module
categorical = _module
normal = _module
uniform = _module
setup = _module
test_core = _module

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_stepelu_ptstat(_paritybench_base):
    pass
