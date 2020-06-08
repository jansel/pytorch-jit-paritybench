import sys
_module = sys.modules[__name__]
del sys
rename_wheel = _module
setup = _module
test = _module
test_fps = _module
test_graclus = _module
test_grid = _module
test_knn = _module
test_nearest = _module
test_radius = _module
test_rw = _module
test_sampler = _module
utils = _module
torch_cluster = _module
fps = _module
graclus = _module
grid = _module
knn = _module
nearest = _module
radius = _module
rw = _module
sampler = _module

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

class Test_rusty1s_pytorch_cluster(_paritybench_base):
    pass
