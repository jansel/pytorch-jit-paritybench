import sys
_module = sys.modules[__name__]
del sys
client = _module
mmdnn = _module
conversion = _module
IRToCode = _module
IRToModel = _module
_script = _module
convert = _module
convertToIR = _module
dump_code = _module
extractModel = _module
caffe = _module
caffe_emitter = _module
caffe_pb2 = _module
common_graph = _module
errors = _module
graph = _module
mapper = _module
network = _module
resolver = _module
saver = _module
shape = _module
transformer = _module
utils = _module
writer = _module
cntk = _module
cntk_emitter = _module
cntk_graph = _module
cntk_parser = _module
DataStructure = _module
emitter = _module
parser = _module
IR_graph = _module
IR = _module
graph_pb2 = _module
common = _module
coreml = _module
coreml_emitter = _module
coreml_graph = _module
coreml_parser = _module
coreml_utils = _module
darknet = _module
cfg = _module
darknet_graph = _module
darknet_parser = _module
darknet_utils = _module
prototxt = _module
examples = _module
extract_model = _module
extractor = _module
imagenet_test = _module
test_tfcoreml = _module
keras = _module
mxnet = _module
onnx = _module
paddle = _module
models = _module
alexnet = _module
resnet = _module
vgg = _module
pytorch = _module
tensorflow = _module
inception_resnet_v1 = _module
inception_resnet_v2 = _module
mobilenet = _module
conv_blocks = _module
mobilenet_v2 = _module
mobilenet_v1 = _module
nasnet = _module
nasnet_utils = _module
test_rnn = _module
vis_meta = _module
extra_layers = _module
keras2_emitter = _module
keras2_graph = _module
keras2_parser = _module
mxnet_emitter = _module
mxnet_graph = _module
mxnet_parser = _module
onnx_emitter = _module
onnx_graph = _module
onnx_parser = _module
shape_inference = _module
paddle_graph = _module
paddle_parser = _module
pytorch_emitter = _module
pytorch_graph = _module
pytorch_parser = _module
torch_to_np = _module
rewriter = _module
folder = _module
graph_matcher = _module
rnn_utils = _module
gru_rewriter = _module
lstm_rewriter = _module
tensorflow_emitter = _module
tensorflow_frozenparser = _module
tensorflow_graph = _module
tensorflow_parser = _module
torch = _module
torch_graph = _module
torch_parser = _module
GenerateMdByDataset = _module
GenerateMdFromJson = _module
select_requirements = _module
setup = _module
conversion_imagenet = _module
gen_test = _module
test_caffe = _module
test_caffe_2 = _module
test_caffe_3 = _module
test_caffe_4 = _module
test_cntk = _module
test_cntk_2 = _module
test_coreml = _module
test_coreml_2 = _module
test_darknet = _module
test_keras = _module
test_keras_2 = _module
test_keras_3 = _module
test_keras_4 = _module
test_keras_5 = _module
test_mxnet = _module
test_mxnet_2 = _module
test_mxnet_3 = _module
test_mxnet_4 = _module
test_mxnet_5 = _module
test_paddle = _module
test_pytorch = _module
test_pytorch_2 = _module
test_pytorch_3 = _module
test_pytorch_4 = _module
test_pytorch_5 = _module
test_tensorflow = _module
test_tensorflow_2 = _module
test_tensorflow_3 = _module
test_tensorflow_4 = _module
test_tensorflow_5 = _module
test_tensorflow_6 = _module
test_tensorflow_7 = _module
test_tensorflow_frozen = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_microsoft_MMdnn(_paritybench_base):
    pass
