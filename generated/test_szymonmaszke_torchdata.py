import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
cachers_test = _module
datasets = _module
datasets_test = _module
maps_test = _module
modifiers_test = _module
samplers_test = _module
torchdata_test = _module
utils = _module
torchdata = _module
_base = _module
_dev_utils = _module
cachers = _module
maps = _module
modifiers = _module
samplers = _module

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

class Test_szymonmaszke_torchdata(_paritybench_base):
    pass
