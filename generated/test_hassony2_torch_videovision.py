import sys
_module = sys.modules[__name__]
del sys
setup = _module
testransforms = _module
torchvideotransforms = _module
functional = _module
stack_transforms = _module
tensor_transforms = _module
utils = _module
images = _module
video_transforms = _module
volume_transforms = _module

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

class Test_hassony2_torch_videovision(_paritybench_base):
    pass
