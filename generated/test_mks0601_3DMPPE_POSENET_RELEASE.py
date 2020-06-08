import sys
_module = sys.modules[__name__]
del sys
base = _module
logger = _module
resnet = _module
timer = _module
utils = _module
dir_utils = _module
pose_utils = _module
vis = _module
Human36M = _module
MPII = _module
MSCOCO = _module
MuCo = _module
MuPoTS = _module
dataset = _module
config = _module
model = _module
test = _module
train = _module
coco_img_name = _module
mupots_img_name = _module

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


import abc


from torch.utils.data import DataLoader


import torch.optim


from torch.nn.parallel.data_parallel import DataParallel


import torch


import torch.nn as nn


from torch.nn import functional as F


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
        resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 
            512], 'resnet18'), (34): (BasicBlock, [3, 4, 6, 3], [64, 64, 
            128, 256, 512], 'resnet34'), (50): (Bottleneck, [3, 4, 6, 3], [
            64, 256, 512, 1024, 2048], 'resnet50'), (101): (Bottleneck, [3,
            4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'), (152): (
            Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
            }
        block, layers, channels, name = resnet_spec[resnet_type]
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


_global_config['depth_dim'] = 1


class HeadNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 256
        super(HeadNet, self).__init__()
        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(in_channels=self.inplanes,
            out_channels=joint_num * cfg.depth_dim, kernel_size=1, stride=1,
            padding=0)

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                out_channels=self.outplanes, kernel_size=4, stride=2,
                padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


_global_config['output_shape'] = 4


def soft_argmax(heatmaps, joint_num):
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim * cfg.
        output_shape[0] * cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.
        output_shape[0], cfg.output_shape[1]))
    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, cfg.
        output_shape[1] + 1).type(torch.cuda.FloatTensor), devices=[accu_x.
        device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, cfg.
        output_shape[0] + 1).type(torch.cuda.FloatTensor), devices=[accu_y.
        device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1, cfg.
        depth_dim + 1).type(torch.cuda.FloatTensor), devices=[accu_z.device
        .index])[0]
    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1
    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


class ResPoseNet(nn.Module):

    def __init__(self, backbone, head, joint_num):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)
        hm = self.head(fm)
        coord = soft_argmax(hm, self.joint_num)
        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:, :, (0)] + loss_coord[:, :, (1)] + 
                loss_coord[:, :, (2)] * target_have_depth) / 3.0
            return loss_coord


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mks0601_3DMPPE_POSENET_RELEASE(_paritybench_base):
    pass

    def test_000(self):
        self._check(HeadNet(*[], **{'joint_num': 4}), [torch.rand([4, 2048, 4, 4])], {})
