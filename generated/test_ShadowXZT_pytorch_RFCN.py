import sys
_module = sys.modules[__name__]
del sys
master = _module
faster_rcnn = _module
datasets = _module
coco = _module
ds_utils = _module
factory = _module
imagenet3d = _module
imdb = _module
imdb2 = _module
kitti = _module
kitti_tracking = _module
kittivoc = _module
nissan = _module
nthu = _module
pascal3d = _module
pascal_voc = _module
pascal_voc2 = _module
voc_eval = _module
fast_rcnn = _module
bbox_transform = _module
config = _module
nms_wrapper = _module
network = _module
nms = _module
py_cpu_nms = _module
psroi_pooling = _module
_ext = _module
psroi_pooling = _module
build = _module
functions = _module
psroi_pooling = _module
modules = _module
psroi_pool = _module
pycocotools = _module
cocoeval = _module
mask = _module
roi_data_layer = _module
layer = _module
minibatch = _module
minibatch2 = _module
roidb = _module
roidb2 = _module
roi_pooling = _module
roi_pooling = _module
build = _module
roi_pool = _module
main = _module
roi_pool = _module
roi_pool_py = _module
rpn_msr = _module
anchor_target_layer = _module
generate = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer = _module
setup = _module
utils = _module
blob = _module
boxes_grid = _module
timer = _module
vgg16 = _module
train = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


import numpy as np


from torch.autograd import Function


from torch.nn.modules.module import Module


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):

    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RoIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width))
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))
                    is_empty = hend <= hstart or wend <= wstart
                    if is_empty:
                        outputs[(roi_ind), :, (ph), (pw)] = 0
                    else:
                        data = features[batch_ind]
                        outputs[(roi_ind), :, (ph), (pw)] = torch.max(torch.max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0].view(-1)
        return outputs


psroi_pooling = RoIPool(7, 7, 1 / 16)


class PSRoIPoolingFunction(Function):

    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, self.output_dim, self.pooled_height, self.pooled_width)
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        output = output
        mappingchannel = mappingchannel
        psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim, features, rois, output, mappingchannel)
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.size()
        return output

    def backward(self, grad_output):
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width)
        psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.output_dim, grad_output, self.rois, grad_input, self.mappingchannel)
        return grad_input, None


class PSRoIPool(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSRoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return PSRoIPoolingFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)


class VGG16(nn.Module):

    def __init__(self, bn=False):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn), Conv2d(64, 64, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn), Conv2d(128, 128, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)
        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn), Conv2d(256, 256, 3, same_padding=True, bn=bn), Conv2d(256, 256, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ShadowXZT_pytorch_RFCN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

