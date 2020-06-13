import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
pointnet_cls = _module
pointnet_cls_basic = _module
pointnet_seg = _module
transform_nets = _module
pointnet_part_seg = _module
test = _module
train = _module
provider = _module
batch_inference = _module
collect_indoor3d_data = _module
eval_iou_accuracy = _module
gen_indoor3d_h5 = _module
indoor3d_util = _module
model = _module
train_pytorch = _module
data_prep_util = _module
data_utils = _module
eulerangles = _module
model = _module
pc_util = _module
plyfile = _module
tf_util = _module
util_funcs = _module
util_layers = _module

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


import math


import numpy as np


import random


import torch


from torch import nn


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from torch import FloatTensor


from typing import Tuple


from typing import Callable


from typing import Optional


from typing import Union


class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    """

    def __init__(self, N: int, dim: int, *args, **kwargs) ->None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError('Dimensionality %i not supported' % dim)
        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hxdengBerkeley_PointCNN_Pytorch(_paritybench_base):
    pass
