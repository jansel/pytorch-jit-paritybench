import sys
_module = sys.modules[__name__]
del sys
detection = _module
demo = _module
demo_retinanet = _module
grad_cam = _module
grad_cam_retinanet = _module
interpretability = _module
guided_back_propagation = _module
main = _module

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

class Test_yizt_Grad_CAM_pytorch(_paritybench_base):
    pass
