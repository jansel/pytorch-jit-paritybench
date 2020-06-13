import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
learning_curve = _module
model_caffe_to_pytorch = _module
speedtest = _module
summarize_logs = _module
train_fcn16s = _module
train_fcn32s = _module
train_fcn8s = _module
train_fcn8s_atonce = _module
setup = _module
test_fcn32s = _module
torchfcn = _module
datasets = _module
voc = _module
infer = _module
net = _module
solve = _module
nyud_layers = _module
pascalcontext_layers = _module
score = _module
siftflow_layers = _module
surgery = _module
voc_helper = _module
voc_layers = _module
models = _module
fcn16s = _module
fcn32s = _module
fcn8s = _module
vgg = _module
trainer = _module
utils = _module

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


import numpy as np


import torch.nn as nn


import math


from torch.autograd import Variable


import torch.nn.functional as F


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=
        nout, pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,
        decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def fcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.depth, n.label = L.Python(module='nyud_layers', layer=
        'NYUDSegDataLayer', ntop=3, param_str=str(dict(nyud_dir=
        '../data/nyud', split=split, tops=tops, seed=1337)))
    n.data = L.Concat(n.color, n.depth)
    n.conv1_1_bgrd, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score_fr = L.Convolution(n.drop7, num_output=40, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = L.Deconvolution(n.score_fr, convolution_param=dict(
        num_output=40, kernel_size=64, stride=32, bias_term=False), param=[
        dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=
        False, ignore_label=255))
    return n.to_proto()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
        dtype=np.float64)
    weight[(range(in_channels)), (range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_wkentaro_pytorch_fcn(_paritybench_base):
    pass
