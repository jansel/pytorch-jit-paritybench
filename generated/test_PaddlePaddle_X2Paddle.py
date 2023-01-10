import sys
_module = sys.modules[__name__]
del sys
setup = _module
auto_scan_test = _module
onnxbase = _module
test_auto_scan_abs = _module
test_auto_scan_averagepool_10 = _module
test_auto_scan_averagepool_7 = _module
test_auto_scan_compare_ops = _module
test_auto_scan_conv2d = _module
test_auto_scan_elementwise_ops = _module
test_auto_scan_equal = _module
test_auto_scan_hardsigmoid = _module
test_auto_scan_isinf = _module
test_auto_scan_isnan = _module
test_auto_scan_logical_ops = _module
test_auto_scan_mod = _module
test_auto_scan_nonzero = _module
test_auto_scan_reduce_ops = _module
test_auto_scan_sum_7 = _module
test_auto_scan_sum_8 = _module
test_auto_scan_unsqueeze_13 = _module
test_auto_scan_unsqueeze_7 = _module
x2paddle = _module
convert = _module
core = _module
graph = _module
program = _module
util = _module
decoder = _module
caffe_decoder = _module
caffe_pb2 = _module
caffe_shape_inference = _module
onnx_decoder = _module
onnx_shape_inference = _module
pytorch_decoder = _module
tf_decoder = _module
models = _module
op_mapper = _module
caffe2paddle = _module
caffe_custom_layer = _module
detectionoutput = _module
normalize = _module
priorbox = _module
roipooling = _module
select = _module
caffe_op_mapper = _module
onnx2paddle = _module
onnx_custom_layer = _module
nms = _module
one_hot = _module
pad_all_dim2 = _module
pad_all_dim4 = _module
pad_all_dim4_one_input = _module
pad_two_input = _module
roi_align = _module
roi_pooling = _module
onnx_op_mapper = _module
opset10 = _module
opset11 = _module
opset12 = _module
opset13 = _module
opset14 = _module
opset15 = _module
opset7 = _module
opset8 = _module
opset9 = _module
opset_legacy = _module
prim2code = _module
pytorch2paddle = _module
aten = _module
prim = _module
pytorch_custom_layer = _module
gather = _module
instance_norm = _module
pad = _module
pytorch_op_mapper = _module
tf2paddle = _module
tf_op_mapper = _module
optimizer = _module
elimination = _module
transpose_eliminate_pass = _module
transpose_elimination = _module
fusion = _module
adaptive_pool2d_fuse_pass = _module
adaptive_pool2d_fuser = _module
batchnorm2d_fuse_pass = _module
batchnorm2d_fuser = _module
bn_scale_fuse_pass = _module
bn_scale_fuser = _module
constant_fuse_pass = _module
constant_fuser = _module
conv2d_add_fuse_pass = _module
conv2d_add_fuser = _module
dropout_fuse_pass = _module
dropout_fuser = _module
fc_fuse_pass = _module
fc_fuser = _module
if_fuse_pass = _module
if_fuser = _module
interpolate_bilinear_fuse_pass = _module
interpolate_bilinear_fuser = _module
onnx_gelu_fuse_pass = _module
onnx_gelu_fuser = _module
onnx_layernorm_fuse_pass = _module
onnx_layernorm_fuser = _module
prelu_fuse_pass = _module
prelu_fuser = _module
replace_div_to_scale = _module
replace_div_to_scale_pass = _module
reshape_fuse_pass = _module
reshape_fuser = _module
tf_batchnorm_fuse_pass = _module
tf_batchnorm_fuser = _module
trace_fc_fuse_pass = _module
trace_fc_fuser = _module
pass_ = _module
pass_manager = _module
pattern_matcher = _module
pytorch_code_optimizer = _module
hierachical_tree = _module
layer_code_generator = _module
module_graph = _module
parameter_tree = _module
subgraphs_union = _module
paddlenlp = _module
utils = _module
project_convertor = _module
pytorch = _module
api_mapper = _module
learning_rate_scheduler = _module
nn = _module
ops = _module
torchvision = _module
utils = _module
ast_update = _module
convert = _module
dependency_analyzer = _module
mapper = _module
resnet = _module
vgg = _module
torch2paddle = _module
device = _module
distributed = _module
io = _module
layer = _module
nn_functional = _module
nn_init = _module
nn_utils = _module
parambase = _module
tensor = _module
varbase = _module
vision_datasets = _module
vision_transforms = _module
vision_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import logging


import time


import torch


import numpy as np


import copy


import inspect


from collections import OrderedDict


import warnings

