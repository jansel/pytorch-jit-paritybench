import sys
_module = sys.modules[__name__]
del sys
imagenet_seq = _module
data = _module
preprocess_sequential = _module
setup = _module
test_lmdb = _module
test_random_read = _module

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

class Test_BayesWatch_sequential_imagenet_dataloader(_paritybench_base):
    pass
