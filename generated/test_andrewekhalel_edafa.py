import sys
_module = sys.modules[__name__]
del sys
conf = _module
BasePredictor = _module
ClassPredictor = _module
SegPredictor = _module
edafa = _module
exceptions = _module
tests = _module
test_augs = _module
test_logistics = _module
test_multi_output = _module
utils = _module
extract_weights = _module
load_weights = _module
model = _module
deeplab = _module
slim = _module
datasets = _module
build_imagenet_data = _module
cifar10 = _module
dataset_factory = _module
dataset_utils = _module
download_and_convert_cifar10 = _module
download_and_convert_flowers = _module
download_and_convert_mnist = _module
flowers = _module
imagenet = _module
mnist = _module
preprocess_imagenet_validation_data = _module
process_bounding_boxes = _module
deployment = _module
model_deploy = _module
model_deploy_test = _module
download_and_convert_data = _module
eval_image_classifier = _module
export_inference_graph = _module
export_inference_graph_test = _module
nets = _module
alexnet = _module
alexnet_test = _module
cifarnet = _module
cyclegan = _module
cyclegan_test = _module
dcgan = _module
dcgan_test = _module
inception = _module
inception_resnet_v2 = _module
inception_resnet_v2_test = _module
inception_utils = _module
inception_v1 = _module
inception_v1_test = _module
inception_v2 = _module
inception_v2_test = _module
inception_v3 = _module
inception_v3_test = _module
inception_v4 = _module
inception_v4_test = _module
lenet = _module
mobilenet = _module
conv_blocks = _module
mobilenet_v2 = _module
mobilenet_v2_test = _module
mobilenet_v1 = _module
mobilenet_v1_eval = _module
mobilenet_v1_test = _module
mobilenet_v1_train = _module
nasnet = _module
nasnet_test = _module
nasnet_utils = _module
nasnet_utils_test = _module
pnasnet = _module
pnasnet_test = _module
nets_factory = _module
nets_factory_test = _module
overfeat = _module
overfeat_test = _module
pix2pix = _module
pix2pix_test = _module
resnet_utils = _module
resnet_v1 = _module
resnet_v1_test = _module
resnet_v2 = _module
resnet_v2_test = _module
vgg = _module
vgg_test = _module
preprocessing = _module
cifarnet_preprocessing = _module
inception_preprocessing = _module
lenet_preprocessing = _module
preprocessing_factory = _module
vgg_preprocessing = _module
setup = _module
train_image_classifier = _module

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


import collections


import tensorflow as tf

