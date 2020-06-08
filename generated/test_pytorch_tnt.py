import sys
_module = sys.modules[__name__]
del sys
conf = _module
mnist = _module
mnist_with_meterlogger = _module
mnist_with_visdom = _module
setup = _module
test_datasets = _module
test_meters = _module
test_transforms = _module
torchnet = _module
dataset = _module
batchdataset = _module
concatdataset = _module
listdataset = _module
resampledataset = _module
shuffledataset = _module
splitdataset = _module
tensordataset = _module
transformdataset = _module
engine = _module
logger = _module
meterlogger = _module
visdomlogger = _module
meter = _module
apmeter = _module
aucmeter = _module
averagevaluemeter = _module
classerrormeter = _module
confusionmeter = _module
mapmeter = _module
movingaveragevaluemeter = _module
msemeter = _module
timemeter = _module
transform = _module
utils = _module
multitaskdataloader = _module
resultswriter = _module
table = _module

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


import torch.optim


from torch.autograd import Variable


import torch.nn.functional as F


from torch.nn.init import kaiming_normal


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorch_tnt(_paritybench_base):
    pass
