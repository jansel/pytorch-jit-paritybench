import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
test_automata = _module
test_cross = _module
test_derivatives = _module
test_gpu = _module
test_indexing = _module
test_init = _module
test_ops = _module
test_round = _module
test_tensor = _module
test_tools = _module
util = _module
tntorch = _module
anova = _module
autodiff = _module
automata = _module
create = _module
cross = _module
derivatives = _module
logic = _module
metrics = _module
ops = _module
round = _module
tensor = _module
tools = _module

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

class Test_rballester_tntorch(_paritybench_base):
    pass
