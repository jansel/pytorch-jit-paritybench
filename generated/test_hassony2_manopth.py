import sys
_module = sys.modules[__name__]
del sys
manopth_demo = _module
manopth_mindemo = _module
mano = _module
webuser = _module
lbs = _module
posemapper = _module
serialization = _module
smpl_handpca_wrapper_HAND_only = _module
verts = _module
manopth = _module
argutils = _module
demo = _module
manolayer = _module
rodrigues_layer = _module
rot6d = _module
rotproj = _module
tensutils = _module
setup = _module
test_demo = _module

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


import numpy as np


import torch


from torch.nn import Module


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hassony2_manopth(_paritybench_base):
    pass
