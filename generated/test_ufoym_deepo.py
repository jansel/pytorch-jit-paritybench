import sys
_module = sys.modules[__name__]
del sys
generator = _module
core = _module
composer = _module
generate = _module
modules = _module
__module__ = _module
boost = _module
caffe = _module
chainer = _module
cntk = _module
darknet = _module
jupyter = _module
jupyterlab = _module
keras = _module
lasagne = _module
mxnet = _module
onnx = _module
opencv = _module
python = _module
pytorch = _module
sonnet = _module
tensorflow = _module
theano = _module
tools = _module
torch = _module

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

class Test_ufoym_deepo(_paritybench_base):
    pass
