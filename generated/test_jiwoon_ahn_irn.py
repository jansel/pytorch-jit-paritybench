import sys
_module = sys.modules[__name__]
del sys
imutils = _module
indexing = _module
pyutils = _module
torchutils = _module
resnet50 = _module
resnet50_cam = _module
resnet50_irn = _module
run_sample = _module
cam_to_ir_label = _module
eval_cam = _module
eval_ins_seg = _module
eval_sem_seg = _module
make_cam = _module
make_cocoann = _module
make_ins_seg_labels = _module
make_sem_seg_labels = _module
train_cam = _module
train_irn = _module
dataloader = _module
make_cls_labels = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn.functional as F


import numpy as np


from torch.utils.data import Subset


import math


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch import multiprocessing


from torch.utils.data import DataLoader


from torch import cuda


from torch.backends import cudnn


from torch.utils.data import Dataset


class FixedBatchNorm(nn.BatchNorm2d):

    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = FixedBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = FixedBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = FixedBatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1, 1, 1)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = FixedBatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.inplanes = 1024

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), FixedBatchNorm(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.mean_shift = Net.MeanShift(2)
        self.fc_edge1 = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.fc_edge2 = nn.Sequential(nn.Conv2d(256, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.fc_edge3 = nn.Sequential(nn.Conv2d(512, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge4 = nn.Sequential(nn.Conv2d(1024, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge5 = nn.Sequential(nn.Conv2d(2048, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)
        self.fc_dp1 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.fc_dp2 = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.fc_dp3 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.ReLU(inplace=True))
        self.fc_dp4 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp5 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp6 = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp7 = nn.Sequential(nn.Conv2d(448, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.ReLU(inplace=True), nn.Conv2d(256, 2, 1, bias=False), self.mean_shift)
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.dp_layers = nn.ModuleList([self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7])


    class MeanShift(nn.Module):

        def __init__(self, num_features):
            super(Net.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[(...), :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[(...), :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[(...), :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[(...), :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[(...), :dp3.size(2), :dp3.size(3)]
        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[(...), :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))
        return edge_out, dp_out

    def trainable_parameters(self):
        return tuple(self.edge_layers.parameters()), tuple(self.dp_layers.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x


class AffinityDisplacementLoss(Net):
    path_indices_prefix = 'path_indices'

    def __init__(self, path_index):
        super(AffinityDisplacementLoss, self).__init__()
        self.path_index = path_index
        self.n_path_lengths = len(path_index.path_indices)
        for i, pi in enumerate(path_index.path_indices):
            self.register_buffer(AffinityDisplacementLoss.path_indices_prefix + str(i), torch.from_numpy(pi))
        self.register_buffer('disp_target', torch.unsqueeze(torch.unsqueeze(torch.from_numpy(path_index.search_dst).transpose(1, 0), 0), -1).float())

    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)
        for i in range(self.n_path_lengths):
            ind = self._buffers[AffinityDisplacementLoss.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat

    def to_pair_displacement(self, disp):
        height, width = disp.size(2), disp.size(3)
        radius_floor = self.path_index.radius_floor
        cropped_height = height - radius_floor
        cropped_width = width - 2 * radius_floor
        disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]
        disp_dst = [disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width] for dy, dx in self.path_index.search_dst]
        disp_dst = torch.stack(disp_dst, 2)
        pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst
        pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1), pair_disp.size(2), -1)
        return pair_disp

    def to_displacement_loss(self, pair_disp):
        return torch.abs(pair_disp - self.disp_target)

    def forward(self, *inputs):
        x, return_loss = inputs
        edge_out, dp_out = super().forward(x)
        if return_loss is False:
            return edge_out, dp_out
        aff = self.to_affinity(torch.sigmoid(edge_out))
        pos_aff_loss = -1 * torch.log(aff + 1e-05)
        neg_aff_loss = -1 * torch.log(1.0 + 1e-05 - aff)
        pair_disp = self.to_pair_displacement(dp_out)
        dp_fg_loss = self.to_displacement_loss(pair_disp)
        dp_bg_loss = torch.abs(pair_disp)
        return pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss


class EdgeDisplacement(Net):

    def __init__(self, crop_size=512, stride=4):
        super(EdgeDisplacement, self).__init__()
        self.crop_size = crop_size
        self.stride = stride

    def forward(self, x):
        feat_size = (x.size(2) - 1) // self.stride + 1, (x.size(3) - 1) // self.stride + 1
        x = F.pad(x, [0, self.crop_size - x.size(3), 0, self.crop_size - x.size(2)])
        edge_out, dp_out = super().forward(x)
        edge_out = edge_out[(...), :feat_size[0], :feat_size[1]]
        dp_out = dp_out[(...), :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)
        dp_out = dp_out[0]
        return edge_out, dp_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FixedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jiwoon_ahn_irn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

