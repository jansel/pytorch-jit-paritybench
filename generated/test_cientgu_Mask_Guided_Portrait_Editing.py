import sys
_module = sys.modules[__name__]
del sys
aligned_dataset = _module
aligned_dataset_test = _module
aligned_dataset_test_temp = _module
aligned_dataset_train_temp = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
base_model = _module
bninception = _module
models = _module
networks = _module
pix2pixHD_mask_model = _module
ui_model = _module
base_options = _module
test_options = _module
train_options = _module
test_edit = _module
test_edit_free = _module
train = _module
html = _module
image_pool = _module
util = _module
visualizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import random


import torchvision.transforms as transforms


import time


import torch


import torch.utils.data as data


import numpy as np


import torch.utils.data


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import functools


from torch.autograd import Variable


from torchvision import models


from collections import OrderedDict


import scipy.misc


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        None
                except:
                    None
                    for k, v in pretrained_dict.items():
                        None
                        if v.size() == model_dict[k].size():
                            None
                            model_dict[k] = v
                        else:
                            None
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        not_initialized = Set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    None
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.global_pool = nn.AvgPool2d(4, stride=4, padding=0, ceil_mode=True, count_include_pad=True)
        self.last_linear = nn.Linear(4096, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out, inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out, inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out, inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out, inception_3b_pool_proj_bn_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_3x3_bn_out, inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out, inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out, inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out, inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out, inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out, inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out, inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out, inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out, inception_4d_pool_proj_bn_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out, inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out, inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out, inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out, inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class EmbedGlobalBGGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm='batch', embed_nc=128, padding_type='reflect'):
        norm_layer = get_norm_layer(norm_type=norm)
        assert n_blocks >= 0
        super(EmbedGlobalBGGenerator, self).__init__()
        activation = nn.ReLU(True)
        downsample_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i != n_downsampling - 1:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
            else:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        self.downsample_model = nn.Sequential(*downsample_model)
        model = []
        model += [nn.Conv2d(in_channels=ngf * 2 ** n_downsampling + embed_nc, out_channels=ngf * 2 ** n_downsampling, kernel_size=1, padding=0, stride=1, bias=True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        self.model = nn.Sequential(*model)
        bg_encoder = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.bg_encoder = nn.Sequential(*bg_encoder)
        bg_decoder = [nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, padding=0, stride=1, bias=True)]
        bg_decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.bg_decoder = nn.Sequential(*bg_decoder)

    def forward(self, input, type='label_encoder'):
        if type == 'label_encoder':
            return self.downsample_model(input)
        elif type == 'image_G':
            return self.model(input)
        elif type == 'bg_encoder':
            return self.bg_encoder(input)
        elif type == 'bg_decoder':
            return self.bg_decoder(input)
        else:
            None


class EmbedGlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm='batch', embed_nc=128, padding_type='reflect'):
        norm_layer = get_norm_layer(norm_type=norm)
        assert n_blocks >= 0
        super(EmbedGlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        downsample_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i != n_downsampling - 1:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
            else:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        self.downsample_model = nn.Sequential(*downsample_model)
        model = []
        model += [nn.Conv2d(in_channels=ngf * 2 ** n_downsampling + embed_nc, out_channels=ngf * 2 ** n_downsampling, kernel_size=1, padding=0, stride=1, bias=True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, type='label_encoder'):
        if type == 'label_encoder':
            return self.downsample_model(input)
        elif type == 'image_G':
            return self.model(input)
        else:
            None


class DecoderBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if norelu == False:
            layers_list.append(nn.ReLU(True))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten


class DecoderGenerator_mask_skin_image(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_skin_image, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 2 * 2))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64, 3, kernel_size=5, padding=0))
        layers_list.append(nn.Tanh())
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 256
        assert ten.size()[3] == 256
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_mouth_image(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_mouth_image, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 5 * 9))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64, 3, kernel_size=5, padding=0))
        layers_list.append(nn.Tanh())
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 80
        assert ten.size()[3] == 144
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye_image(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_eye_image, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 2 * 3, bias=False))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64, 3, kernel_size=5, padding=0))
        layers_list.append(nn.Tanh())
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 32
        assert ten.size()[3] == 48
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_skin(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_skin, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 2 * 2))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 64
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin, self).__call__(*args, **kwargs)


class DecoderGenerator_mask160(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask160, self).__init__()
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=1))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 18
        assert ten.size()[3] == 18
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask160, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_mouth(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_mouth, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 5 * 9))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 20
        assert ten.size()[3] == 36
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_mask_eye, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512 * 2 * 3, bias=False))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], 512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 8
        assert ten.size()[3] == 12
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye, self).__call__(*args, **kwargs)


class EncoderBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.relu = nn.ReLU(True)

    def forward(self, ten, out=False, t=False):
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten


class EncoderGenerator_mask_mouth(nn.Module):
    """docstring for  EncoderGenerator"""

    def __init__(self, norm_layer):
        super(EncoderGenerator_mask_mouth, self).__init__()
        layers_list = []
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512 * 5 * 9, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512 * 5 * 9, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0], -1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_mouth, self).__call__(*args, **kwargs)


class EncoderGenerator_mask_eye(nn.Module):
    """docstring for  EncoderGenerator"""

    def __init__(self, norm_layer):
        super(EncoderGenerator_mask_eye, self).__init__()
        layers_list = []
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512 * 2 * 3, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512 * 2 * 3, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0], -1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_eye, self).__call__(*args, **kwargs)


class EncoderGenerator_mask_skin(nn.Module):
    """docstring for  EncoderGenerator"""

    def __init__(self, norm_layer):
        super(EncoderGenerator_mask_skin, self).__init__()
        layers_list = []
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512 * 2 * 2, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512 * 2 * 2, out_features=1024), nn.ReLU(True), nn.Linear(in_features=1024, out_features=512))

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0], -1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_skin, self).__call__(*args, **kwargs)


class DecoderGenerator_512_64(nn.Module):

    def __init__(self, norm_layer):
        super(DecoderGenerator_512_64, self).__init__()
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256))
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128))
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        assert ten.size()[1] == 64
        assert ten.size()[2] == 64
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_512_64, self).__call__(*args, **kwargs)


class DecoderGenerator(nn.Module):

    def __init__(self, embed_length, output_size, n_downsample_global, norm_layer):
        output_nc = 2 ** (n_downsample_global + 4)
        None
        None
        None
        super(DecoderGenerator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=embed_length, out_features=8 * 8 * output_nc // 4, bias=False), nn.BatchNorm1d(num_features=8 * 8 * output_nc // 4, momentum=0.9), nn.ReLU(True))
        self.output_nc = output_nc
        self.output_size = output_size
        layers_list = []
        if n_downsample_global == 3:
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 4, channel_out=self.output_nc // 2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 2, channel_out=self.output_nc))
            layers_list.append(DecoderBlock(channel_in=self.output_nc, channel_out=self.output_nc))
        elif n_downsample_global == 4:
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 4, channel_out=self.output_nc // 2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 2, channel_out=self.output_nc))
        elif n_downsample_global == 2:
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 4, channel_out=self.output_nc // 2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc // 2, channel_out=self.output_nc))
            layers_list.append(DecoderBlock(channel_in=self.output_nc, channel_out=self.output_nc))
        else:
            None
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0], -1, 8, 8)
        ten = self.conv(ten)
        assert ten.size()[1] == self.output_nc
        assert ten.size()[2] == self.output_size
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator, self).__call__(*args, **kwargs)


class EncoderVectorGenerator(nn.Module):
    """docstring for  EncoderVectorGenerator"""

    def __init__(self, n_downsample_global, input_nc, embed_length, norm_layer):
        super(EncoderVectorGenerator, self).__init__()
        self.size = input_nc
        layers_list = []
        for i in range(n_downsample_global):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2
        self.conv = nn.Sequential(*layers_list)
        if n_downsample_global == 2:
            feature_size = 16
        elif n_downsample_global == 3:
            feature_size = 8
        else:
            None
        self.fc = nn.Sequential(nn.Linear(in_features=feature_size * feature_size * self.size, out_features=1024, bias=False), nn.BatchNorm1d(num_features=1024, momentum=0.9), nn.ReLU(True), nn.Linear(in_features=1024, out_features=embed_length))

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        output = self.fc(ten)
        return output

    def __call__(self, *args, **kwargs):
        return super(EncoderVectorGenerator, self).__call__(*args, **kwargs)


class EncoderGenerator(nn.Module):
    """docstring for  EncoderGenerator"""

    def __init__(self, n_downsample_global, input_nc, embed_length, norm_layer):
        super(EncoderGenerator, self).__init__()
        self.size = input_nc
        layers_list = []
        for i in range(n_downsample_global):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2
        self.conv = nn.Sequential(*layers_list)
        if n_downsample_global == 2:
            feature_size = 16
        elif n_downsample_global == 3:
            feature_size = 8
        else:
            None
        self.fc = nn.Sequential(nn.Linear(in_features=feature_size * feature_size * self.size, out_features=1024, bias=False))
        self.act = nn.Sequential(nn.BatchNorm1d(num_features=1024, momentum=0.9), nn.ReLU(True))
        self.l_mu = nn.Linear(in_features=1024, out_features=embed_length)
        self.l_var = nn.Linear(in_features=1024, out_features=embed_length)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        ten = self.act(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator, self).__call__(*args, **kwargs)


class EncoderGenerator_256_512(nn.Module):
    """docstring for  EncoderGenerator"""

    def __init__(self, norm_layer):
        super(EncoderGenerator_256_512, self).__init__()
        layers_list = []
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=5, padding=2, stride=2))
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=5, padding=2, stride=2))
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=3, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))
        self.conv = nn.Sequential(*layers_list)
        self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)

    def forward(self, ten):
        ten = self.conv(ten)
        mu = self.c_mu(ten)
        logvar = self.c_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_256_512, self).__call__(*args, **kwargs)


class EncoderResBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderResBlock, self).__init__()
        layers_list1 = []
        layers_list1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.AvgPool2d(2, 2))
        self.conv1 = nn.Sequential(*layers_list1)
        layers_list2 = []
        layers_list2.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=1))
        layers_list2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*layers_list2)

    def forward(self, ten):
        ten1 = self.conv1(ten)
        ten2 = self.conv2(ten)
        return ten1 + ten2


class DecoderResBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderResBlock, self).__init__()
        layers_list1 = []
        layers_list1.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers_list1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        self.conv1 = nn.Sequential(*layers_list1)
        layers_list2 = []
        layers_list2.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=1))
        layers_list2.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv2 = nn.Sequential(*layers_list2)

    def forward(self, ten):
        ten1 = self.conv1(ten)
        ten2 = self.conv2(ten)
        return ten1 + ten2


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


pretrained_settings = {'bninception': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth', 'input_space': 'BGR', 'input_size': [3, 224, 224], 'input_range': [0, 255], 'mean': [104, 117, 128], 'std': [1, 1, 1], 'num_classes': 1000}}}


def bninception(num_classes=1000, pretrained='imagenet'):
    """BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
    """
    model = BNInception(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['bninception'][pretrained]
        assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


class ID_Loss(nn.Module):

    def __init__(self, gpu_ids, load_path):
        super(ID_Loss, self).__init__()
        self.model = bninception.BNInception(num_classes=128)
        assert load_path != ''
        self.load_face_id_model(self.model, load_path)
        self.model = self.model
        self.criterion = nn.L1Loss()

    def load_face_id_model(self, network, load_path):
        assert os.path.isfile(load_path) == True
        try:
            network.load_state_dict(torch.load(load_path))
        except:
            pretrained_dict = torch.load(load_path)
            model_dict = network.state_dict()
            pretrained_dict = {k.replace('module.backbone.', ''): v for k, v in pretrained_dict.items() if k.replace('module.backbone.', '') in model_dict}
            network.load_state_dict(pretrained_dict)

    def forward(self, x, y):
        x_128feature, y_128feature = self.model(x), self.model(y)
        loss = self.criterion(x_128feature, y_128feature.detach())
        return loss


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, layers_num=5):
        h_relu1 = self.slice1(X)
        if layers_num == 1:
            return [h_relu1]
        h_relu2 = self.slice2(h_relu1)
        if layers_num == 2:
            return [h_relu1, h_relu2]
        h_relu3 = self.slice3(h_relu2)
        if layers_num == 3:
            return [h_relu1, h_relu2, h_relu3]
        h_relu4 = self.slice4(h_relu3)
        if layers_num == 4:
            return [h_relu1, h_relu2, h_relu3, h_relu4]
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids, weights=None):
        super(VGGLoss, self).__init__()
        if weights != None:
            self.weights = weights
        else:
            self.weights = [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 8, 1.0 / 8]
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()

    def forward(self, x, y, face_mask, mask_weights):
        assert face_mask.size()[1] == len(mask_weights)
        x_vgg, y_vgg = self.vgg(x, layers_num=len(self.weights)), self.vgg(y, layers_num=len(self.weights))
        mask = []
        mask.append(face_mask.detach())
        downsample = nn.MaxPool2d(2)
        for i in range(len(x_vgg)):
            mask.append(downsample(mask[i]))
            mask[i] = mask[i].detach()
        loss = 0
        for i in range(len(x_vgg)):
            for mask_index in range(len(mask_weights)):
                a = x_vgg[i] * mask[i][:, mask_index:mask_index + 1, :, :]
                loss += self.weights[i] * self.criterion(x_vgg[i] * mask[i][:, mask_index:mask_index + 1, :, :], (y_vgg[i] * mask[i][:, mask_index:mask_index + 1, :, :]).detach()) * mask_weights[mask_index]
        return loss


class MFMLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(MFMLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x_input, y_input):
        loss = 0
        for i in range(len(x_input)):
            x = x_input[i][-2]
            y = y_input[i][-2]
            assert x.dim() == 4
            assert y.dim() == 4
            x_mean = torch.mean(x, 0)
            y_mean = torch.mean(y, 0)
            loss += self.criterion(x_mean, y_mean.detach())
        return loss


def grammatrix(feature):
    assert feature.dim() == 4
    a, b, c, d = feature.size()[0], feature.size()[1], feature.size()[2], feature.size()[3]
    out_tensor = torch.Tensor(a, b, b)
    for batch_index in range(0, a):
        features = feature[batch_index].view(b, c * d)
        G = torch.mm(features, features.t())
        out_tensor[batch_index] = G.clone().div(b * c * d)
    return out_tensor


class GramMatrixLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(GramMatrixLoss, self).__init__()
        self.weights = [1.0, 1.0, 1.0]
        self.vgg = Vgg19()
        self.criterion = nn.MSELoss()

    def forward(self, x, y, label):
        face_mask = (label == 1).type(torch.FloatTensor)
        mask = []
        mask.append(face_mask)
        x_vgg, y_vgg = self.vgg(x, layers_num=len(self.weights)), self.vgg(y, layers_num=len(self.weights))
        downsample = nn.MaxPool2d(2)
        for i in range(len(x_vgg)):
            mask.append(downsample(mask[i]))
            mask[i] = mask[i].detach()
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(grammatrix(x_vgg[i] * mask[i]), grammatrix(y_vgg[i] * mask[i]).detach())
        return loss


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class LocalEnhancer(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        ngf_global = ngf * 2 ** n_local_enhancers
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.Sequential(*model_global)
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True), nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(ngf_global), nn.ReLU(True)]
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        output_prev = self.model(input_downsampled[-1])
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, (0)] + b, indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, (0)] + b, indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]] = mean_feat
        return outputs_mean


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            None
            None
            return res[-2:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        global printlayer_index
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, output_padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias, output_padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias, output_padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        model_output = self.model(x)
        wb, hb = model_output.size()[3], model_output.size()[2]
        wa, ha = x.size()[3], x.size()[2]
        l = int((wb - wa) / 2)
        t = int((hb - ha) / 2)
        model_output = model_output[:, :, t:t + ha, l:l + wa]
        if self.outermost:
            return model_output
        else:
            return torch.cat([x, model_output], 1)


class UnetGenerator(nn.Module):

    def __init__(self, segment_classes, input_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        output_nc = segment_classes
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        softmax = torch.nn.Softmax(dim=1)
        return softmax(self.model(input))


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def softmax2label(featuremap):
    _, label = torch.max(featuremap, dim=1)
    size = label.size()
    label = label.resize_(size[0], 1, size[1], size[2])
    return label


class Pix2PixHD_mask_Model(BaseModel):

    def name(self):
        return 'Pix2PixHD_mask_Model'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_l2_loss):
        flags = True, True, True, use_gan_feat_loss, use_vgg_loss, True, True, use_l2_loss, True, True, True, True

        def loss_filter(kl_loss, l2_mask_image, g_gan, g_gan_feat, g_vgg, d_real, d_fake, l2_image, loss_parsing, g2_gan, d2_real, d2_fake):
            return [l for l, f in zip((kl_loss, l2_mask_image, g_gan, g_gan_feat, g_vgg, d_real, d_fake, l2_image, loss_parsing, g2_gan, d2_real, d2_fake), flags) if f]
        return loss_filter

    def initialize(self, opt):
        assert opt.vae_encoder == True
        self.name = 'Pix2PixHD_mask_Model'
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        self.netG = networks.define_embed_bg_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, 9, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, 256 * 5, gpu_ids=self.gpu_ids)
        self.netP = networks.define_P(opt.label_nc, opt.output_nc, 64, 'unet_128', opt.norm, use_dropout=True, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD2 = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, True, gpu_ids=self.gpu_ids)
        embed_feature_size = opt.longSize // 2 ** opt.n_downsample_global
        self.net_encoder_skin = networks.define_encoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_hair = networks.define_encoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_left_eye = networks.define_encoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_right_eye = networks.define_encoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_mouth = networks.define_encoder_mask(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_skin = networks.define_decoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_hair = networks.define_decoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_left_eye = networks.define_decoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_right_eye = networks.define_decoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_mouth = networks.define_decoder_mask(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_skin_image = networks.define_decoder_mask_image(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_hair_image = networks.define_decoder_mask_image(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_left_eye_image = networks.define_decoder_mask_image(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_right_eye_image = networks.define_decoder_mask_image(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_mouth_image = networks.define_decoder_mask_image(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            None
        weight_list = [0.5, 1, 3, 3, 3, 3, 10, 5, 5, 5, 0.8]
        self.criterionCrossEntropy = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_list))
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.net_encoder_skin, 'encoder_skin', opt.which_epoch, pretrained_path)
            self.load_network(self.net_encoder_hair, 'encoder_hair', opt.which_epoch, pretrained_path)
            self.load_network(self.net_encoder_left_eye, 'encoder_left_eye', opt.which_epoch, pretrained_path)
            self.load_network(self.net_encoder_right_eye, 'encoder_right_eye', opt.which_epoch, pretrained_path)
            self.load_network(self.net_encoder_mouth, 'encoder_mouth', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_skin, 'decoder_skin', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_hair, 'decoder_hair', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_left_eye, 'decoder_left_eye', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_right_eye, 'decoder_right_eye', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_mouth, 'decoder_mouth', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_skin_image, 'decoder_skin_image', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_hair_image, 'decoder_hair_image', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_left_eye_image, 'decoder_left_eye_image', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_right_eye_image, 'decoder_right_eye_image', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_mouth_image, 'decoder_mouth_image', opt.which_epoch, pretrained_path)
            self.load_network(self.netP, 'P', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_l2_loss)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMFM = networks.MFMLoss(self.gpu_ids)
            weight_list = [0.2, 1, 5, 5, 5, 5, 3, 8, 8, 8, 1]
            self.criterionCrossEntropy = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_list))
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids, weights=None)
            self.criterionGM = networks.GramMatrixLoss(self.gpu_ids)
            self.loss_names = self.loss_filter('KL_embed', 'L2_mask_image', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'L2_image', 'ParsingLoss', 'G2_GAN', 'D2_real', 'D2_fake')
            params_decoder = list(self.net_decoder_skin.parameters()) + list(self.net_decoder_hair.parameters()) + list(self.net_decoder_left_eye.parameters()) + list(self.net_decoder_right_eye.parameters()) + list(self.net_decoder_mouth.parameters())
            params_image_decoder = list(self.net_decoder_skin_image.parameters()) + list(self.net_decoder_hair_image.parameters()) + list(self.net_decoder_left_eye_image.parameters()) + list(self.net_decoder_right_eye_image.parameters()) + list(self.net_decoder_mouth_image.parameters())
            params_encoder = list(self.net_encoder_skin.parameters()) + list(self.net_encoder_hair.parameters()) + list(self.net_encoder_left_eye.parameters()) + list(self.net_encoder_right_eye.parameters()) + list(self.net_encoder_mouth.parameters())
            params_together = list(self.netG.parameters()) + params_decoder + params_encoder + params_image_decoder
            self.optimizer_G_together = torch.optim.Adam(params_together, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD2.parameters())
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, image_affine=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        if real_image is not None:
            real_image = Variable(real_image.data)
        if image_affine is not None:
            image_affine = Variable(image_affine.data)
        return input_label, inst_map, real_image, feat_map, image_affine

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate2(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD2.forward(fake_query)
        else:
            return self.netD2.forward(input_concat)

    def forward(self, bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer=False, type='sample_net'):
        if type == 'sample_net':
            return self.forward_sample_net(label, inst, image, feat, image_affine, mask_list, infer)
        elif type == 'vae_net':
            return self.forward_vae_net(bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer)
        else:
            None

    def forward_sample_net(self, label, inst, image, feat, image_affine, infer=False):
        input_label, inst_map, real_image, feat_map, _ = self.encode_input(label, inst, image, feat)
        mask_list = ['1', '4', '5', '6']
        encode_label_feature = self.netG.forward(input_label, type='label_encoder')
        sample_vector1 = Variable(torch.randn(len(input_label), 128, 4, 4), requires_grad=False)
        decode_feature1 = self.net_decoder_mask1(sample_vector1)
        encode_label_feature = encode_label_feature + decode_feature1
        sample_vector4 = Variable(torch.randn(len(input_label), 1024), requires_grad=False)
        decode_feature4 = self.net_decoder_mask4(sample_vector4)
        encode_label_feature[:, :, 23:35, 18:31] = encode_label_feature[:, :, 23:35, 18:31] + decode_feature4
        sample_vector5 = Variable(torch.randn(len(input_label), 1024), requires_grad=False)
        decode_feature5 = self.net_decoder_mask5(sample_vector5)
        encode_label_feature[:, :, 23:35, 33:46] = encode_label_feature[:, :, 23:35, 33:46] + decode_feature5
        sample_vector6 = Variable(torch.randn(len(input_label), 1024), requires_grad=False)
        decode_feature6 = self.net_decoder_mask6(sample_vector6)
        encode_label_feature[:, :, 24:44, 23:41] = encode_label_feature[:, :, 24:44, 23:41] + decode_feature6
        fake_image = self.netG.forward(encode_label_feature, type='image_G')
        mask4_image = fake_image[:, :, 92:140, 72:124]
        mask5_image = fake_image[:, :, 92:140, 132:184]
        mask6_image = fake_image[:, :, 96:176, 92:164]
        mask1 = (label == 1).type(torch.FloatTensor)
        mask1_image = mask1 * fake_image
        reconstruct_mean1, reconstruct_log_var1 = self.net_encoder_mask1(mask1_image)
        loss_l1_vector1 = self.criterionL1(reconstruct_mean1, sample_vector1) * 3
        reconstruct_mean4, reconstruct_log_var4 = self.net_encoder_mask4(mask4_image)
        loss_l1_vector4 = self.criterionL1(reconstruct_mean4, sample_vector4) * 10
        reconstruct_mean5, reconstruct_log_var5 = self.net_encoder_mask5(mask5_image)
        loss_l1_vector5 = self.criterionL1(reconstruct_mean5, sample_vector5) * 10
        reconstruct_mean6, reconstruct_log_var6 = self.net_encoder_mask6(mask6_image)
        loss_l1_vector6 = self.criterionL1(reconstruct_mean6, sample_vector6) * 3
        loss_l1_vector = loss_l1_vector1 + loss_l1_vector4 + loss_l1_vector5 + loss_l1_vector6
        reconstruct_label_feature = self.netP(fake_image)
        reconstruct_label = softmax2label(reconstruct_label_feature)
        pred_fake_pool = self.netD2.forward(torch.cat((input_label, fake_image.detach()), dim=1))
        loss_D2_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.netD2.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D2_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD2.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_sample_GAN = self.criterionGAN(pred_fake, True)
        loss_MFM = self.criterionMFM(pred_fake, pred_real) * self.opt.lambda_feat
        gt_label = torch.squeeze(label.type(torch.LongTensor), 1)
        loss_l1_label = self.criterionCrossEntropy(reconstruct_label_feature, gt_label) * self.opt.lambda_feat
        loss_MFM = loss_MFM.reshape(1)
        loss_D2_fake = loss_D2_fake.reshape(1)
        loss_D2_real = loss_D2_real.reshape(1)
        loss_G_sample_GAN = loss_G_sample_GAN.reshape(1)
        loss_l1_label = loss_l1_label.reshape(1)
        loss_l1_vector = loss_l1_vector.reshape(1)
        zero_tensor = loss_l1_label.clone()
        zero_tensor[0][0] = 0
        return self.loss_filter(zero_tensor, zero_tensor, loss_G_sample_GAN, loss_D2_fake, loss_D2_real, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor, loss_l1_label, loss_l1_vector, zero_tensor, loss_MFM), None if not infer else fake_image, None if not infer else reconstruct_label

    def forward_vae_net(self, bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer=False):
        input_label, inst_map, real_image, feat_map, real_bg_image = self.encode_input(label, inst, bg_image, feat, bg_image)
        mask4_image = torch.zeros(label.size()[0], 3, 32, 48)
        mask5_image = torch.zeros(label.size()[0], 3, 32, 48)
        mask_mouth_image = torch.zeros(label.size()[0], 3, 80, 144)
        mask_mouth = torch.zeros(label.size()[0], 3, 80, 144)
        mask_skin = ((label == 1) + (label == 2) + (label == 3) + (label == 6)).type(torch.FloatTensor)
        mask_skin_image = mask_skin * real_image
        mask_hair = (label == 10).type(torch.FloatTensor)
        mask_hair_image = mask_hair * real_image
        mask_mouth_whole = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
        for batch_index in range(0, label.size()[0]):
            mask4_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][0]) - 16:int(mask_list[batch_index][0]) + 16, int(mask_list[batch_index][1]) - 24:int(mask_list[batch_index][1]) + 24]
            mask5_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][2]) - 16:int(mask_list[batch_index][2]) + 16, int(mask_list[batch_index][3]) - 24:int(mask_list[batch_index][3]) + 24]
            mask_mouth_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
            mask_mouth[batch_index] = mask_mouth_whole[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
        mask_mouth_image = mask_mouth * mask_mouth_image
        encode_label_feature = self.netG.forward(input_label, type='label_encoder')
        bg_feature = self.netG.forward(real_bg_image, type='bg_encoder')
        mask_bg = (label == 0).type(torch.FloatTensor)
        mask_bg_feature = mask_bg * bg_feature
        loss_mask_image = 0
        loss_KL = 0
        mus4, log_variances4 = self.net_encoder_left_eye(mask4_image)
        variances4 = torch.exp(log_variances4 * 0.5)
        random_sample4 = Variable(torch.randn(mus4.size()), requires_grad=True)
        correct_sample4 = random_sample4 * variances4 + mus4
        loss_KL4 = -0.5 * torch.sum(-log_variances4.exp() - torch.pow(mus4, 2) + log_variances4 + 1)
        reconstruce_mask4_image = self.net_decoder_left_eye_image(correct_sample4)
        loss_mask_image += self.criterionL2(reconstruce_mask4_image, mask4_image.detach()) * 10
        loss_KL += loss_KL4
        decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
        mus5, log_variances5 = self.net_encoder_right_eye(mask5_image)
        variances5 = torch.exp(log_variances5 * 0.5)
        random_sample5 = Variable(torch.randn(mus5.size()), requires_grad=True)
        correct_sample5 = random_sample5 * variances5 + mus5
        loss_KL5 = -0.5 * torch.sum(-log_variances5.exp() - torch.pow(mus5, 2) + log_variances5 + 1)
        reconstruce_mask5_image = self.net_decoder_right_eye_image(correct_sample5)
        loss_mask_image += self.criterionL2(reconstruce_mask5_image, mask5_image.detach()) * 10
        loss_KL += loss_KL5
        decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
        mus_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
        variances_skin = torch.exp(log_variances_skin * 0.5)
        random_sample_skin = Variable(torch.randn(mus_skin.size()), requires_grad=True)
        correct_sample_skin = random_sample_skin * variances_skin + mus_skin
        loss_KL_skin = -0.5 * torch.sum(-log_variances_skin.exp() - torch.pow(mus_skin, 2) + log_variances_skin + 1)
        reconstruce_mask_skin_image = self.net_decoder_skin_image(correct_sample_skin)
        reconstruce_mask_skin_image = mask_skin * reconstruce_mask_skin_image
        loss_mask_image += self.criterionL2(reconstruce_mask_skin_image, mask_skin_image.detach()) * 10
        loss_KL += loss_KL_skin
        decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
        mus_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
        variances_hair = torch.exp(log_variances_hair * 0.5)
        random_sample_hair = Variable(torch.randn(mus_hair.size()), requires_grad=True)
        correct_sample_hair = random_sample_hair * variances_hair + mus_hair
        loss_KL_hair = -0.5 * torch.sum(-log_variances_hair.exp() - torch.pow(mus_hair, 2) + log_variances_hair + 1)
        reconstruce_mask_hair_image = self.net_decoder_hair_image(correct_sample_hair)
        reconstruce_mask_hair_image = mask_hair * reconstruce_mask_hair_image
        loss_mask_image += self.criterionL2(reconstruce_mask_hair_image, mask_hair_image.detach()) * 10
        loss_KL += loss_KL_hair
        decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
        mus_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
        variances_mouth = torch.exp(log_variances_mouth * 0.5)
        random_sample_mouth = Variable(torch.randn(mus_mouth.size()), requires_grad=True)
        correct_sample_mouth = random_sample_mouth * variances_mouth + mus_mouth
        loss_KL_mouth = -0.5 * torch.sum(-log_variances_mouth.exp() - torch.pow(mus_mouth, 2) + log_variances_mouth + 1)
        reconstruce_mask_mouth_image = self.net_decoder_mouth_image(correct_sample_mouth)
        reconstruce_mask_mouth_image = mask_mouth * reconstruce_mask_mouth_image
        loss_mask_image += self.criterionL2(reconstruce_mask_mouth_image, mask_mouth_image.detach()) * 10
        loss_KL += loss_KL_mouth
        decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
        left_eye_tensor = torch.zeros(encode_label_feature.size())
        right_eye_tensor = torch.zeros(encode_label_feature.size())
        mouth_tensor = torch.zeros(encode_label_feature.size())
        reorder_left_eye_tensor = torch.zeros(encode_label_feature.size())
        reorder_right_eye_tensor = torch.zeros(encode_label_feature.size())
        reorder_mouth_tensor = torch.zeros(encode_label_feature.size())
        new_order = torch.randperm(label.size()[0])
        reorder_decode_embed_feature4 = decode_embed_feature4[new_order]
        reorder_decode_embed_feature5 = decode_embed_feature5[new_order]
        reorder_decode_embed_feature_mouth = decode_embed_feature_mouth[new_order]
        reorder_decode_embed_feature_skin = decode_embed_feature_skin[new_order]
        reorder_decode_embed_feature_hair = decode_embed_feature_hair[new_order]
        for batch_index in range(0, label.size()[0]):
            try:
                reorder_left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += reorder_decode_embed_feature4[batch_index]
                reorder_right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += reorder_decode_embed_feature5[batch_index]
                reorder_mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += reorder_decode_embed_feature_mouth[batch_index]
            except:
                None
        reconstruct_transfer_face = self.netG.forward(torch.cat((encode_label_feature, reorder_left_eye_tensor, reorder_right_eye_tensor, reorder_decode_embed_feature_skin, reorder_decode_embed_feature_hair, reorder_mouth_tensor), 1), type='image_G')
        reconstruct_transfer_image = self.netG.forward(torch.cat((reconstruct_transfer_face, mask_bg_feature), 1), type='bg_decoder')
        parsing_label_feature = self.netP(reconstruct_transfer_image)
        parsing_label = softmax2label(parsing_label_feature)
        gt_label = torch.squeeze(ori_label.type(torch.LongTensor), 1)
        loss_parsing = self.criterionCrossEntropy(parsing_label_feature, gt_label) * self.opt.lambda_feat
        pred_fake2_pool = self.netD2.forward(torch.cat((input_label, reconstruct_transfer_image.detach()), dim=1))
        loss_D2_fake = self.criterionGAN(pred_fake2_pool, False)
        pred_real2 = self.netD2.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D2_real = self.criterionGAN(pred_real2, True)
        pred_fake2 = self.netD2.forward(torch.cat((input_label, reconstruct_transfer_image), dim=1))
        loss_G2_GAN = self.criterionGAN(pred_fake2, True)
        for batch_index in range(0, label.size()[0]):
            try:
                left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
            except:
                None
        reconstruct_face = self.netG.forward(torch.cat((encode_label_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
        reconstruct_image = self.netG.forward(torch.cat((reconstruct_face, mask_bg_feature), 1), type='bg_decoder')
        mask_left_eye = (label == 4).type(torch.FloatTensor)
        mask_right_eye = (label == 5).type(torch.FloatTensor)
        mask_mouth = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
        loss_L2_image = 0
        for batch_index in range(0, label.size()[0]):
            loss_L2_image += self.criterionL2(mask_left_eye * reconstruct_image, mask_left_eye * real_image) * 10
            loss_L2_image += self.criterionL2(mask_right_eye * reconstruct_image, mask_right_eye * real_image) * 10
            loss_L2_image += self.criterionL2(mask_skin * reconstruct_image, mask_skin * real_image) * 5
            loss_L2_image += self.criterionL2(mask_hair * reconstruct_image, mask_hair * real_image) * 5
            loss_L2_image += self.criterionL2(mask_mouth * reconstruct_image, mask_mouth * real_image) * 10
            loss_L2_image += self.criterionL2(reconstruct_image, real_bg_image) * 10
        pred_fake_pool = self.netD.forward(torch.cat((input_label, reconstruct_image.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.netD.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD.forward(torch.cat((input_label, reconstruct_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        all_mask_tensor = torch.cat((mask_left_eye, mask_right_eye, mask_skin, mask_hair, mask_mouth), 1)
        mask_weight_list = [10, 10, 5, 5, 10]
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.criterionVGG(reconstruct_image, real_image, all_mask_tensor, mask_weights=mask_weight_list) * self.opt.lambda_feat * 3
        return self.loss_filter(loss_KL, loss_mask_image, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_L2_image, loss_parsing, loss_G2_GAN, loss_D2_real, loss_D2_fake), None if not infer else reconstruct_image, None if not infer else reconstruce_mask4_image, None if not infer else reconstruce_mask5_image, None if not infer else reconstruce_mask_skin_image, None if not infer else reconstruce_mask_hair_image, None if not infer else reconstruce_mask_mouth_image, None if not infer else reconstruct_transfer_image, None if not infer else parsing_label

    def inference(self, bg_contentimage, label2, mask2_list, image, label, mask_list):
        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)
        input_label2, _, real_content_image, _, _ = self.encode_input(Variable(label2), None, Variable(bg_contentimage), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()
            mask4_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask5_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask_mouth_image = torch.zeros(label.size()[0], 3, 80, 144)
            mask_mouth = torch.zeros(label.size()[0], 3, 80, 144)
            mask_skin = ((label == 1) + (label == 2) + (label == 3) + (label == 6)).type(torch.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label == 10).type(torch.FloatTensor)
            mask_hair_image = mask_hair * real_image
            mask_mouth_whole = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
            bg_content_feature = self.netG.forward(real_content_image, type='bg_encoder')
            mask_content_bg = (label2 == 0).type(torch.FloatTensor)
            mask_content_bg_feature = mask_content_bg * bg_content_feature
            bg_style_feature = self.netG.forward(real_image, type='bg_encoder')
            mask_style_bg = (label == 0).type(torch.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature
            for batch_index in range(0, label.size()[0]):
                mask4_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][0]) - 16:int(mask_list[batch_index][0]) + 16, int(mask_list[batch_index][1]) - 24:int(mask_list[batch_index][1]) + 24]
                mask5_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][2]) - 16:int(mask_list[batch_index][2]) + 16, int(mask_list[batch_index][3]) - 24:int(mask_list[batch_index][3]) + 24]
                mask_mouth_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
                mask_mouth[batch_index] = mask_mouth_whole[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
            mask_mouth_image = mask_mouth * mask_mouth_image
            encode_label_feature = self.netG.forward(input_label, type='label_encoder')
            encode_label2_feature = self.netG.forward(input_label2, type='label_encoder')
            if self.opt.random_embed == False:
                with torch.no_grad():
                    correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                    decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                    correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                    decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                    correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                    decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                    correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                    decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                    correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                    decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
            else:
                with torch.no_grad():
                    mus4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                    correct_sample4 = Variable(torch.randn(mus4.size()), requires_grad=True)
                    decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                    mus5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                    correct_sample5 = Variable(torch.randn(mus5.size()), requires_grad=True)
                    decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                    mus_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                    correct_sample_skin = Variable(torch.randn(mus_skin.size()), requires_grad=True)
                    decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                    mus_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                    correct_sample_hair = Variable(torch.randn(mus_hair.size()), requires_grad=True)
                    decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                    mus_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                    correct_sample_mouth = Variable(torch.randn(mus_mouth.size()), requires_grad=True)
                    decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
            left_eye_tensor = torch.zeros(encode_label_feature.size())
            right_eye_tensor = torch.zeros(encode_label_feature.size())
            mouth_tensor = torch.zeros(encode_label_feature.size())
            for batch_index in range(0, label.size()[0]):
                try:
                    left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                    mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_style_face = self.netG.forward(torch.cat((encode_label_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
            left_eye_tensor = torch.zeros(encode_label2_feature.size())
            right_eye_tensor = torch.zeros(encode_label2_feature.size())
            mouth_tensor = torch.zeros(encode_label2_feature.size())
            for batch_index in range(0, label.size()[0]):
                try:
                    left_eye_tensor[(batch_index), :, int(mask2_list[batch_index][0] / 4 + 0.5) - 4:int(mask2_list[batch_index][0] / 4 + 0.5) + 4, int(mask2_list[batch_index][1] / 4 + 0.5) - 6:int(mask2_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[(batch_index), :, int(mask2_list[batch_index][2] / 4 + 0.5) - 4:int(mask2_list[batch_index][2] / 4 + 0.5) + 4, int(mask2_list[batch_index][3] / 4 + 0.5) - 6:int(mask2_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                    mouth_tensor[(batch_index), :, int(mask2_list[batch_index][4] / 4 + 0.5) - 10:int(mask2_list[batch_index][4] / 4 + 0.5) + 10, int(mask2_list[batch_index][5] / 4 + 0.5) - 18:int(mask2_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_content_face = self.netG.forward(torch.cat((encode_label2_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face, mask_style_bg_feature), 1), type='bg_decoder')
            reconstruct_fake_content = self.netG.forward(torch.cat((fake_content_face, mask_content_bg_feature), 1), type='bg_decoder')
        return reconstruct_fake_style, reconstruct_fake_content, mask_mouth_image

    def inference_2image(self, bg_image, label, mask_list, ori_label):
        input_label, inst_map, real_image, _, _ = self.encode_input(Variable(label), None, Variable(bg_image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()
            mask4_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask5_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask_mouth_image = torch.zeros(label.size()[0], 3, 80, 144)
            mask_mouth = torch.zeros(label.size()[0], 3, 80, 144)
            mask_skin = ((label == 1) + (label == 2) + (label == 3) + (label == 6)).type(torch.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label == 10).type(torch.FloatTensor)
            mask_hair_image = mask_hair * real_image
            mask_mouth_whole = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
            bg_feature = self.netG.forward(real_image, type='bg_encoder')
            mask_bg = (label == 0).type(torch.FloatTensor)
            mask_bg_feature = mask_bg * bg_feature
            for batch_index in range(0, label.size()[0]):
                mask4_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][0]) - 16:int(mask_list[batch_index][0]) + 16, int(mask_list[batch_index][1]) - 24:int(mask_list[batch_index][1]) + 24]
                mask5_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][2]) - 16:int(mask_list[batch_index][2]) + 16, int(mask_list[batch_index][3]) - 24:int(mask_list[batch_index][3]) + 24]
                mask_mouth_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
                mask_mouth[batch_index] = mask_mouth_whole[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
            mask_mouth_image = mask_mouth * mask_mouth_image
            encode_label_feature = self.netG.forward(input_label, type='label_encoder')
            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                torch.save(decode_embed_feature4, 'a_tesnor.pth')
                torch.save(decode_embed_feature5, 'b_tesnor.pth')
                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
            reorder_left_eye_tensor = torch.zeros(encode_label_feature.size())
            reorder_right_eye_tensor = torch.zeros(encode_label_feature.size())
            reorder_mouth_tensor = torch.zeros(encode_label_feature.size())
            new_order = torch.randperm(label.size()[0])
            if random.random() > 0:
                new_order[0] = 1
                new_order[1] = 0
            reorder_decode_embed_feature4 = decode_embed_feature4[new_order]
            reorder_decode_embed_feature5 = decode_embed_feature5[new_order]
            reorder_decode_embed_feature_mouth = decode_embed_feature_mouth[new_order]
            reorder_decode_embed_feature_skin = decode_embed_feature_skin[new_order]
            reorder_decode_embed_feature_hair = decode_embed_feature_hair[new_order]
            for batch_index in range(0, label.size()[0]):
                try:
                    reorder_left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += reorder_decode_embed_feature4[batch_index]
                    reorder_right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += reorder_decode_embed_feature5[batch_index]
                    reorder_mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += reorder_decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_face = self.netG.forward(torch.cat((encode_label_feature, reorder_left_eye_tensor, reorder_right_eye_tensor, reorder_decode_embed_feature_skin, reorder_decode_embed_feature_hair, reorder_mouth_tensor), 1), type='image_G')
            fake_image = self.netG.forward(torch.cat((fake_face, mask_bg_feature), 1), type='bg_decoder')
        return fake_image

    def inference_multi_embed(self, label, inst, image, mask_list, label2, mask2_list):
        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), Variable(inst), Variable(image), infer=True)
        input_label2, _, _, _, _ = self.encode_input(Variable(label2), Variable(inst), Variable(image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()
            mask4_image = torch.zeros(1, 3, 32, 48)
            mask5_image = torch.zeros(1, 3, 32, 48)
            mask_mouth_image = torch.zeros(1, 3, 80, 144)
            mask_skin = ((label[2:3] == 1) + (label[2:3] == 2) + (label[2:3] == 3) + (label[2:3] == 6)).type(torch.FloatTensor)
            mask_skin_image = mask_skin * real_image[2:3]
            mask_hair = (label[3:4] == 10).type(torch.FloatTensor)
            mask_hair_image = mask_hair * real_image[3:4]
            mask4_image[0] = real_image[(0), :, int(mask_list[0][0]) - 16:int(mask_list[0][0]) + 16, int(mask_list[0][1]) - 24:int(mask_list[0][1]) + 24]
            mask5_image[0] = real_image[(1), :, int(mask_list[1][2]) - 16:int(mask_list[1][2]) + 16, int(mask_list[1][3]) - 24:int(mask_list[1][3]) + 24]
            mask_mouth_image[0] = real_image[(4), :, int(mask_list[4][4]) - 40:int(mask_list[4][4]) + 40, int(mask_list[4][5]) - 72:int(mask_list[4][5]) + 72]
            mask4_image = mask4_image.expand(5, 3, 32, 48)
            mask5_image = mask5_image.expand(5, 3, 32, 48)
            mask_skin_image = mask_skin_image.expand(5, 3, 256, 256)
            mask_hair_image = mask_hair_image.expand(5, 3, 256, 256)
            mask_mouth_image = mask_mouth_image.expand(5, 3, 80, 144)
            encode_label2_feature = self.netG.forward(input_label2, type='label_encoder')
            assert self.opt.random_embed == False
            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
            left_eye_tensor = torch.zeros(encode_label2_feature.size())
            right_eye_tensor = torch.zeros(encode_label2_feature.size())
            mouth_tensor = torch.zeros(encode_label2_feature.size())
            for batch_index in range(0, label.size()[0]):
                try:
                    left_eye_tensor[(batch_index), :, int(mask2_list[batch_index][0] / 4 + 0.5) - 4:int(mask2_list[batch_index][0] / 4 + 0.5) + 4, int(mask2_list[batch_index][1] / 4 + 0.5) - 6:int(mask2_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[(batch_index), :, int(mask2_list[batch_index][2] / 4 + 0.5) - 4:int(mask2_list[batch_index][2] / 4 + 0.5) + 4, int(mask2_list[batch_index][3] / 4 + 0.5) - 6:int(mask2_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                    mouth_tensor[(batch_index), :, int(mask2_list[batch_index][4] / 4 + 0.5) - 10:int(mask2_list[batch_index][4] / 4 + 0.5) + 10, int(mask2_list[batch_index][5] / 4 + 0.5) - 18:int(mask2_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_image2 = self.netG.forward(torch.cat((encode_label2_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
        return fake_image2

    def inference_encode(self, path, image, label, mask_list):
        base_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch))
        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()
            mask4_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask5_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask_mouth_image = torch.zeros(label.size()[0], 3, 80, 144)
            mask_mouth = torch.zeros(label.size()[0], 3, 80, 144)
            mask_skin = ((label == 1) + (label == 2) + (label == 3) + (label == 6)).type(torch.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label == 10).type(torch.FloatTensor)
            mask_hair_image = mask_hair * real_image
            mask_mouth_whole = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
            bg_style_feature = self.netG.forward(real_image, type='bg_encoder')
            mask_style_bg = (label == 0).type(torch.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature
            for batch_index in range(0, label.size()[0]):
                mask4_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][0]) - 16:int(mask_list[batch_index][0]) + 16, int(mask_list[batch_index][1]) - 24:int(mask_list[batch_index][1]) + 24]
                mask5_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][2]) - 16:int(mask_list[batch_index][2]) + 16, int(mask_list[batch_index][3]) - 24:int(mask_list[batch_index][3]) + 24]
                mask_mouth_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
                mask_mouth[batch_index] = mask_mouth_whole[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
            mask_mouth_image = mask_mouth * mask_mouth_image
            encode_label_feature = self.netG.forward(input_label, type='label_encoder')
            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
                None
                image_name = path[0].split('/')[-1].replace('.png', '').replace('.jpg', '')
                save_path = base_path + '/encode_tensor/' + image_name
                None
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                torch.save(decode_embed_feature4, save_path + '/left_eye')
                torch.save(decode_embed_feature5, save_path + '/right_eye')
                torch.save(decode_embed_feature_skin, save_path + '/skin')
                torch.save(decode_embed_feature_hair, save_path + '/hair')
                torch.save(decode_embed_feature_mouth, save_path + '/mouth')
            left_eye_tensor = torch.zeros(encode_label_feature.size())
            right_eye_tensor = torch.zeros(encode_label_feature.size())
            mouth_tensor = torch.zeros(encode_label_feature.size())
            for batch_index in range(0, label.size()[0]):
                try:
                    left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                    mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_style_face = self.netG.forward(torch.cat((encode_label_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face, mask_style_bg_feature), 1), type='bg_decoder')
        return reconstruct_fake_style

    def inference_generate(self, path, image, label, mask_list):
        base_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch))
        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()
            mask4_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask5_image = torch.zeros(label.size()[0], 3, 32, 48)
            mask_mouth_image = torch.zeros(label.size()[0], 3, 80, 144)
            mask_mouth = torch.zeros(label.size()[0], 3, 80, 144)
            mask_skin = ((label == 1) + (label == 2) + (label == 3) + (label == 6)).type(torch.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label == 10).type(torch.FloatTensor)
            mask_hair_image = mask_hair * real_image
            mask_mouth_whole = ((label == 7) + (label == 8) + (label == 9)).type(torch.FloatTensor)
            bg_style_feature = self.netG.forward(real_image, type='bg_encoder')
            mask_style_bg = (label == 0).type(torch.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature
            for batch_index in range(0, label.size()[0]):
                mask4_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][0]) - 16:int(mask_list[batch_index][0]) + 16, int(mask_list[batch_index][1]) - 24:int(mask_list[batch_index][1]) + 24]
                mask5_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][2]) - 16:int(mask_list[batch_index][2]) + 16, int(mask_list[batch_index][3]) - 24:int(mask_list[batch_index][3]) + 24]
                mask_mouth_image[batch_index] = real_image[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
                mask_mouth[batch_index] = mask_mouth_whole[(batch_index), :, int(mask_list[batch_index][4]) - 40:int(mask_list[batch_index][4]) + 40, int(mask_list[batch_index][5]) - 72:int(mask_list[batch_index][5]) + 72]
            mask_mouth_image = mask_mouth * mask_mouth_image
            encode_label_feature = self.netG.forward(input_label, type='label_encoder')
            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)
                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)
                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)
                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)
                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
                image_name = path[0].split('/')[-1].replace('.png', '').replace('.jpg', '')
                save_path = base_path + '/encode_tensor/' + image_name
                assert os.path.exists(save_path) == True
                decode_embed_feature4 = torch.load(save_path + '/left_eye')
                decode_embed_feature5 = torch.load(save_path + '/right_eye')
                decode_embed_feature_skin = torch.load(save_path + '/skin')
                decode_embed_feature_hair = torch.load(save_path + '/hair')
                decode_embed_feature_mouth = torch.load(save_path + '/mouth')
            left_eye_tensor = torch.zeros(encode_label_feature.size())
            right_eye_tensor = torch.zeros(encode_label_feature.size())
            mouth_tensor = torch.zeros(encode_label_feature.size())
            for batch_index in range(0, label.size()[0]):
                try:
                    left_eye_tensor[(batch_index), :, int(mask_list[batch_index][0] / 4 + 0.5) - 4:int(mask_list[batch_index][0] / 4 + 0.5) + 4, int(mask_list[batch_index][1] / 4 + 0.5) - 6:int(mask_list[batch_index][1] / 4 + 0.5) + 6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[(batch_index), :, int(mask_list[batch_index][2] / 4 + 0.5) - 4:int(mask_list[batch_index][2] / 4 + 0.5) + 4, int(mask_list[batch_index][3] / 4 + 0.5) - 6:int(mask_list[batch_index][3] / 4 + 0.5) + 6] += decode_embed_feature5[batch_index]
                    mouth_tensor[(batch_index), :, int(mask_list[batch_index][4] / 4 + 0.5) - 10:int(mask_list[batch_index][4] / 4 + 0.5) + 10, int(mask_list[batch_index][5] / 4 + 0.5) - 18:int(mask_list[batch_index][5] / 4 + 0.5) + 18] += decode_embed_feature_mouth[batch_index]
                except:
                    None
            fake_style_face = self.netG.forward(torch.cat((encode_label_feature, left_eye_tensor, right_eye_tensor, decode_embed_feature_skin, decode_embed_feature_hair, mouth_tensor), 1), type='image_G')
            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face, mask_style_bg_feature), 1), type='bg_decoder')
        return reconstruct_fake_style

    def inference_parsing(self, label, image):
        input_label, _, real_image, _, _ = self.encode_input(Variable(label), None, Variable(image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.netP.eval()
            parsing_label_feature = self.netP(real_image)
            _, g_label = torch.max(parsing_label_feature, dim=1)
        return g_label

    def inference_parsing_filter(self, ori_label, image):
        input_label, _, real_image, _, _ = self.encode_input(Variable(ori_label), None, Variable(image), infer=True)
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            self.netP.eval()
            parsing_label_feature = self.netP(real_image)
            gt_label = torch.squeeze(ori_label.type(torch.LongTensor), 1)
            loss_parsing = self.criterionCrossEntropy(parsing_label_feature, gt_label)
        return loss_parsing

    def sample_features(self, inst):
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[(num // 2), :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_skin, 'encoder_skin', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_hair, 'encoder_hair', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_left_eye, 'encoder_left_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_right_eye, 'encoder_right_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_mouth, 'encoder_mouth', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_skin, 'decoder_skin', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_hair, 'decoder_hair', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_left_eye, 'decoder_left_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_right_eye, 'decoder_right_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_mouth, 'decoder_mouth', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_skin_image, 'decoder_skin_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_hair_image, 'decoder_hair_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_left_eye_image, 'decoder_left_eye_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_right_eye_image, 'decoder_right_eye_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_mouth_image, 'decoder_mouth_image', which_epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_together.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            None
        self.old_lr = lr


class InferenceModel(Pix2PixHD_mask_Model):

    def forward(self, label, inst, image, mask_list, label2, mask2):
        if self.opt.multi_embed_test == True:
            return self.inference_multi_embed(label, inst, image, mask_list, label2, mask2)
        elif self.opt.test_parsing == False:
            return self.inference(label, inst, image, mask_list, label2, mask2)
        else:
            return self.inference_parsing(label, inst, image)


class UIModel(BaseModel):

    def name(self):
        return 'UIModel'

    def initialize(self, opt):
        assert not opt.isTrain
        BaseModel.initialize(self, opt)
        self.use_features = opt.instance_feat or opt.label_feat
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.load_network(self.netG, 'G', opt.which_epoch)
        None

    def toTensor(self, img, normalize=False):
        tensor = torch.from_numpy(np.array(img, np.int32, copy=False))
        tensor = tensor.view(1, img.size[1], img.size[0], len(img.mode))
        tensor = tensor.transpose(1, 2).transpose(1, 3).contiguous()
        if normalize:
            return (tensor.float() / 255.0 - 0.5) / 0.5
        return tensor.float()

    def load_image(self, label_path, inst_path, feat_path):
        opt = self.opt
        label_img = Image.open(label_path)
        if label_path.find('face') != -1:
            label_img = label_img.convert('L')
        ow, oh = label_img.size
        w = opt.loadSize
        h = int(w * oh / ow)
        label_img = label_img.resize((w, h), Image.NEAREST)
        label_map = self.toTensor(label_img)
        self.label_map = label_map
        oneHot_size = 1, opt.label_nc, h, w
        input_label = self.Tensor(torch.Size(oneHot_size)).zero_()
        self.input_label = input_label.scatter_(1, label_map.long(), 1.0)
        if not opt.no_instance:
            inst_img = Image.open(inst_path)
            inst_img = inst_img.resize((w, h), Image.NEAREST)
            self.inst_map = self.toTensor(inst_img)
            self.edge_map = self.get_edges(self.inst_map)
            self.net_input = Variable(torch.cat((self.input_label, self.edge_map), dim=1), volatile=True)
        else:
            self.net_input = Variable(self.input_label, volatile=True)
        self.features_clustered = np.load(feat_path).item()
        self.object_map = self.inst_map if opt.instance_feat else self.label_map
        object_np = self.object_map.cpu().numpy().astype(int)
        self.feat_map = self.Tensor(1, opt.feat_num, h, w).zero_()
        self.cluster_indices = np.zeros(self.opt.label_nc, np.uint8)
        for i in np.unique(object_np):
            label = i if i < 1000 else i // 1000
            if label in self.features_clustered:
                feat = self.features_clustered[label]
                np.random.seed(i + 1)
                cluster_idx = np.random.randint(0, feat.shape[0])
                self.cluster_indices[label] = cluster_idx
                idx = (self.object_map == i).nonzero()
                self.set_features(idx, feat, cluster_idx)
        self.net_input_original = self.net_input.clone()
        self.label_map_original = self.label_map.clone()
        self.feat_map_original = self.feat_map.clone()
        if not opt.no_instance:
            self.inst_map_original = self.inst_map.clone()

    def reset(self):
        self.net_input = self.net_input_prev = self.net_input_original.clone()
        self.label_map = self.label_map_prev = self.label_map_original.clone()
        self.feat_map = self.feat_map_prev = self.feat_map_original.clone()
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev = self.inst_map_original.clone()
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def undo(self):
        self.net_input = self.net_input_prev
        self.label_map = self.label_map_prev
        self.feat_map = self.feat_map_prev
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def change_labels(self, click_src, click_tgt):
        y_src, x_src = click_src[0], click_src[1]
        y_tgt, x_tgt = click_tgt[0], click_tgt[1]
        label_src = int(self.label_map[0, 0, y_src, x_src])
        inst_src = self.inst_map[0, 0, y_src, x_src]
        label_tgt = int(self.label_map[0, 0, y_tgt, x_tgt])
        inst_tgt = self.inst_map[0, 0, y_tgt, x_tgt]
        idx_src = (self.inst_map == inst_src).nonzero()
        if idx_src.shape:
            self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_src, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            if inst_tgt > 1000:
                tgt_indices = (self.inst_map > label_tgt * 1000) & (self.inst_map < (label_tgt + 1) * 1000)
                inst_tgt = self.inst_map[tgt_indices].max() + 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = inst_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            idx_tgt = (self.inst_map == inst_tgt).nonzero()
            if idx_tgt.shape:
                self.copy_features(idx_src, idx_tgt[(0), :])
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_strokes(self, click_src, label_tgt, bw, save):
        size = self.net_input.size()
        h, w = size[2], size[3]
        idx_src = torch.LongTensor(bw ** 2, 4).fill_(0)
        for i in range(bw):
            idx_src[i * bw:(i + 1) * bw, (2)] = min(h - 1, max(0, click_src[0] - bw // 2 + i))
            for j in range(bw):
                idx_src[i * bw + j, 3] = min(w - 1, max(0, click_src[1] - bw // 2 + j))
        idx_src = idx_src
        if idx_src.shape:
            if save:
                self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            for k in range(self.opt.label_nc):
                self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            if self.opt.instance_feat:
                feat = self.features_clustered[label_tgt]
                cluster_idx = self.cluster_indices[label_tgt]
                self.set_features(idx_src, feat, cluster_idx)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_objects(self, click_src, label_tgt, mask, style_id=0):
        y, x = click_src[0], click_src[1]
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
        idx_src = torch.from_numpy(mask).nonzero()
        idx_src[:, (2)] += y
        idx_src[:, (3)] += x
        self.backup_current_state()
        self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        for k in range(self.opt.label_nc):
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
        self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
        self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
        self.set_features(idx_src, self.feat, style_id)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def single_forward(self, net_input, feat_map):
        net_input = torch.cat((net_input, feat_map), dim=1)
        fake_image = self.netG.forward(net_input)
        if fake_image.size()[0] == 1:
            return fake_image.data[0]
        return fake_image.data

    def style_forward(self, click_pt, style_id=-1):
        if click_pt is None:
            self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))
            self.crop = None
            self.mask = None
        else:
            instToChange = int(self.object_map[0, 0, click_pt[0], click_pt[1]])
            self.instToChange = instToChange
            label = instToChange if instToChange < 1000 else instToChange // 1000
            self.feat = self.features_clustered[label]
            self.fake_image = []
            self.mask = self.object_map == instToChange
            idx = self.mask.nonzero()
            self.get_crop_region(idx)
            if idx.size():
                if style_id == -1:
                    min_y, min_x, max_y, max_x = self.crop
                    for cluster_idx in range(self.opt.multiple_output):
                        self.set_features(idx, self.feat, cluster_idx)
                        fake_image = self.single_forward(self.net_input, self.feat_map)
                        fake_image = util.tensor2im(fake_image[:, min_y:max_y, min_x:max_x])
                        self.fake_image.append(fake_image)
                    """### To speed up previewing different style results, either crop or downsample the label maps
                    if instToChange > 1000:
                        (min_y, min_x, max_y, max_x) = self.crop                                                
                        ### crop                                                
                        _, _, h, w = self.net_input.size()
                        offset = 512
                        y_start, x_start = max(0, min_y-offset), max(0, min_x-offset)
                        y_end, x_end = min(h, (max_y + offset)), min(w, (max_x + offset))
                        y_region = slice(y_start, y_start+(y_end-y_start)//16*16)
                        x_region = slice(x_start, x_start+(x_end-x_start)//16*16)
                        net_input = self.net_input[:,:,y_region,x_region]                    
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            fake_image = self.single_forward(net_input, self.feat_map[:,:,y_region,x_region])                            
                            fake_image = util.tensor2im(fake_image[:,min_y-y_start:max_y-y_start,min_x-x_start:max_x-x_start])
                            self.fake_image.append(fake_image)
                    else:
                        ### downsample
                        (min_y, min_x, max_y, max_x) = [crop//2 for crop in self.crop]                    
                        net_input = self.net_input[:,:,::2,::2]                    
                        size = net_input.size()
                        net_input_batch = net_input.expand(self.opt.multiple_output, size[1], size[2], size[3])             
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            feat_map = self.feat_map[:,:,::2,::2]
                            if cluster_idx == 0:
                                feat_map_batch = feat_map
                            else:
                                feat_map_batch = torch.cat((feat_map_batch, feat_map), dim=0)
                        fake_image_batch = self.single_forward(net_input_batch, feat_map_batch)
                        for i in range(self.opt.multiple_output):
                            self.fake_image.append(util.tensor2im(fake_image_batch[i,:,min_y:max_y,min_x:max_x]))"""
                else:
                    self.set_features(idx, self.feat, style_id)
                    self.cluster_indices[label] = style_id
                    self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def backup_current_state(self):
        self.net_input_prev = self.net_input.clone()
        self.label_map_prev = self.label_map.clone()
        self.inst_map_prev = self.inst_map.clone()
        self.feat_map_prev = self.feat_map.clone()

    def get_crop_region(self, idx):
        size = self.net_input.size()
        h, w = size[2], size[3]
        min_y, min_x = idx[:, (2)].min(), idx[:, (3)].min()
        max_y, max_x = idx[:, (2)].max(), idx[:, (3)].max()
        crop_min = 128
        if max_y - min_y < crop_min:
            min_y = max(0, (max_y + min_y) // 2 - crop_min // 2)
            max_y = min(h - 1, min_y + crop_min)
        if max_x - min_x < crop_min:
            min_x = max(0, (max_x + min_x) // 2 - crop_min // 2)
            max_x = min(w - 1, min_x + crop_min)
        self.crop = min_y, min_x, max_y, max_x
        self.mask = self.mask[:, :, min_y:max_y, min_x:max_x]

    def update_features(self, cluster_idx, mask=None, click_pt=None):
        self.feat_map_prev = self.feat_map.clone()
        if mask is not None:
            y, x = click_pt[0], click_pt[1]
            mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
            idx = torch.from_numpy(mask).nonzero()
            idx[:, (2)] += y
            idx[:, (3)] += x
        else:
            idx = (self.object_map == self.instToChange).nonzero()
        self.set_features(idx, self.feat, cluster_idx)

    def set_features(self, idx, feat, cluster_idx):
        for k in range(self.opt.feat_num):
            self.feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]

    def copy_features(self, idx_src, idx_tgt):
        for k in range(self.opt.feat_num):
            val = self.feat_map[idx_tgt[0], idx_tgt[1] + k, idx_tgt[2], idx_tgt[3]]
            self.feat_map[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = val

    def get_current_visuals(self, getLabel=False):
        mask = self.mask
        if self.mask is not None:
            mask = np.transpose(self.mask[0].cpu().float().numpy(), (1, 2, 0)).astype(np.uint8)
        dict_list = [('fake_image', self.fake_image), ('mask', mask)]
        if getLabel:
            label = util.tensor2label(self.net_input.data[0], self.opt.label_nc)
            dict_list += [('label', label)]
        return OrderedDict(dict_list)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNInception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (DecoderBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderGenerator_512_64,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (DecoderGenerator_mask_eye,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([512, 512])], {}),
     True),
    (DecoderGenerator_mask_eye_image,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([512, 512])], {}),
     True),
    (DecoderGenerator_mask_mouth,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([512, 512])], {}),
     True),
    (DecoderGenerator_mask_skin,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([512, 512])], {}),
     True),
    (DecoderResBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EmbedGlobalBGGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EmbedGlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EncoderBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EncoderGenerator_256_512,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EncoderGenerator_mask_skin,
     lambda: ([], {'norm_layer': 1}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (EncoderResBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (LocalEnhancer,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UIModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (UnetGenerator,
     lambda: ([], {'segment_classes': 4, 'input_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_cientgu_Mask_Guided_Portrait_Editing(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

