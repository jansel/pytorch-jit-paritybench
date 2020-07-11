import sys
_module = sys.modules[__name__]
del sys
gen_image_crops_VID = _module
gen_imdb_VID = _module
Config = _module
SiamNet = _module
Tracking_Utils = _module
run_SiamFC = _module
DataAugmentation = _module
SiamNet = _module
Utils = _module
VIDDataset = _module
run_Train_SiamFC = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torchvision.transforms.functional as F


from torch.autograd import Variable


from torch.utils.data.dataset import Dataset


from torch.optim.lr_scheduler import StepLR


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


class Config:

    def __init__(self):
        self.pos_pair_range = 100
        self.num_pairs = 53200.0
        self.val_ratio = 0.1
        self.num_epoch = 50
        self.batch_size = 8
        self.examplar_size = 127
        self.instance_size = 255
        self.sub_mean = 0
        self.train_num_workers = 12
        self.val_num_workers = 8
        self.stride = 8
        self.rPos = 16
        self.rNeg = 0
        self.label_weight_method = 'balanced'
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.step_size = 1
        self.gamma = 0.8685
        self.num_scale = 3
        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_LR = 0.59
        self.response_UP = 16
        self.windowing = 'cosine'
        self.w_influence = 0.176
        self.video = 'Lemming'
        self.visualization = 0
        self.bbox_output = True
        self.bbox_output_path = './tracking_result/'
        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 17
        self.net_base_path = '/home/hfan/Desktop/PyTorch-SiamFC/Train/model/'
        self.seq_base_path = '/home/hfan/Desktop/demo-sequences/'
        self.net = 'SiamFC_50_model.pth'


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()
        self.feat_extraction = nn.Sequential(nn.Conv2d(3, 96, 11, 2), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, groups=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 384, 3, 1, groups=2), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3, 1, groups=2))
        self.adjust = nn.Conv2d(1, 1, 1, 1)
        self._initialize_weight()
        self.config = Config()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        return score

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))
        out = F.conv2d(x, z, groups=batch_size_x)
        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))
        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1
                if tmp_layer_idx < 6:
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    m.weight.data.fill_(0.001)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        return F.binary_cross_entropy_with_logits(prediction, label, weight, size_average=False) / self.config.batch_size

