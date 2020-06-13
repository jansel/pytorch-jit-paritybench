import sys
_module = sys.modules[__name__]
del sys
src = _module
capb_parser = _module
check_duplicated_image = _module
config = _module
kpda_parser = _module
kpdetector = _module
concatenate_results = _module
predict = _module
predict_ensemble = _module
verify_result = _module
lr_scheduler = _module
offline_evaluation = _module
pytorch_utils = _module
stage1 = _module
data_generator = _module
focal_loss = _module
fpn = _module
label_encoder = _module
predict = _module
retinanet = _module
trainval = _module
stage2 = _module
cascade_pyramid_network = _module
cascade_pyramid_network_v10 = _module
cascade_pyramid_network_v11 = _module
cascade_pyramid_network_v12 = _module
cascade_pyramid_network_v14 = _module
cascade_pyramid_network_v2 = _module
cascade_pyramid_network_v5 = _module
cascade_pyramid_network_v6 = _module
cascade_pyramid_network_v7 = _module
cascade_pyramid_network_v8 = _module
keypoint_encoder = _module
predict_ensemble = _module
predict_one = _module
trainval = _module
viserrloss = _module
stage2v13 = _module
cascade_pyramid_network_v13 = _module
trainval = _module
viserrloss_v13 = _module
stage2v15 = _module
cascade_pyramid_network_v15 = _module
nasnet = _module
stage2v2 = _module
cascade_pyramid_network_v2 = _module
trainval = _module
viserrloss_v2 = _module
stage2v3 = _module
cascade_pyramid_network_v3 = _module
trainval = _module
viserrloss_v3 = _module
stage2v4 = _module
cascade_pyramid_network_v4 = _module
inceptionresnetv2 = _module
stage2v9 = _module
cascade_pyramid_network_v9 = _module
senet = _module
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


import numpy as np


import torch


from torch.autograd import Variable


from torch.backends import cudnn


from torch.nn import DataParallel


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


import math


from scipy.ndimage.morphology import binary_dilation


from torch.nn import MSELoss


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


from torch.utils import model_zoo


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, output, labels):
        pt = output * labels + (1 - output) * (1 - labels)
        return torch.mean(-self.alpha * (1 - pt) ** self.gamma * torch.log(
            pt + 1e-07))

    def forward(self, reg_preds, reg_targets, cls_preds, cls_targets):
        """
        :param reg_preds: [batch_size, anchor#, 4]
        :param reg_targets: [batch_size, anchor#, 4]
        :param cls_preds: [batch_size, anchor#, class#]
        :param cls_targets: [batch_size, anchor#]
        :return: 
        """
        batch_size = cls_targets.size(0)
        pos = cls_targets > 0.5
        num_pos = pos.data.long().sum()
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(reg_preds)
            masked_reg_preds = reg_preds[mask].view(-1, 4)
            masked_reg_targets = reg_targets[mask].view(-1, 4)
            regress_loss = F.smooth_l1_loss(masked_reg_preds,
                masked_reg_targets, size_average=True)
        else:
            regress_loss = 0
        neg = (cls_targets > -0.5) & (cls_targets < 0.5)
        num_neg = neg.data.long().sum()
        mask_pos = pos.unsqueeze(2).expand_as(cls_preds)
        mask_neg = neg.unsqueeze(2).expand_as(cls_preds)
        masked_pos_cls_preds = F.sigmoid(cls_preds[mask_pos])
        masked_neg_cls_preds = F.sigmoid(cls_preds[mask_neg])
        classify_loss = 0.5 * self.focal_loss(masked_pos_cls_preds,
            cls_targets[pos]) + 0.5 * self.focal_loss(masked_neg_cls_preds,
            cls_targets[neg])
        loss = classify_loss + regress_loss
        pos_total = num_pos
        neg_total = num_neg
        pos_correct = ((masked_pos_cls_preds > 0.5) * (cls_targets[pos] > 0.5)
            ).data.long().sum()
        neg_correct = ((masked_neg_cls_preds < 0.5) * (cls_targets[neg] < 0.5)
            ).data.long().sum()
        return [loss, classify_loss, regress_loss] + [pos_correct,
            pos_total, neg_correct, neg_total]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, block, num_blocks, pretrained_model=None):
        super(FPN, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7


def FPN101(pretrained=False):
    if pretrained:
        return FPN(Bottleneck, [3, 4, 23, 3], torchvision.models.resnet50(
            pretrained=True))
    return FPN(Bottleneck, [3, 4, 23, 3])


class RetinaNet(nn.Module):

    def __init__(self, config, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN101(pretrained=True)
        if num_classes > 2:
            self.num_classes = num_classes + 1
        else:
            self.num_classes = 1
        self.reg_head = self._make_head(config.anchor_num * 4)
        self.cls_head = self._make_head(config.anchor_num * self.num_classes)
        self.freeze_bn()

    def forward(self, x):
        fms = self.fpn(x)
        reg_preds = []
        cls_preds = []
        for fm in fms:
            reg_pred = self.reg_head(fm)
            cls_pred = self.cls_head(fm)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(x.
                size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.
                size(0), -1, self.num_classes)
            reg_preds.append(reg_pred)
            cls_preds.append(cls_pred)
        return torch.cat(reg_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            conv.weight.data.normal_(0.0, 0.01)
            conv.bias.data.fill_(0)
            layers.append(conv)
            layers.append(nn.ReLU(True))
        final_conv = nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1)
        final_conv.weight.data.normal_(0.0, 0.01)
        final_conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
        layers.append(final_conv)
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


def GlobalNet152(config, pretrained=False):
    if pretrained:
        return GlobalNet(config, Bottleneck, [3, 4, 23, 3], torchvision.
            models.resnet152(pretrained=True))
    return GlobalNet(config, Bottleneck, [3, 4, 23, 3])


class CascadePyramidNet(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNet, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = Bottleneck(256, 64, 1)
        self.toplayer2 = Bottleneck(256, 64, 1)
        self.toplayer3 = nn.Sequential(Bottleneck(256, 64, 1), nn.Conv2d(
            256, config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV10(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV10, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DilatedBottleneck(nn.Module):

    def __init__(self, in_planes, planes, shortcut=False):
        super(DilatedBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=2,
            dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if shortcut:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, num_blocks, pretrained_model):
        super(GlobalNet, self).__init__()
        self.in_planes = 512
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = self._make_layer2(DilatedBottleneck, 256, num_blocks[2])
        self.layer4 = self._make_layer2(DilatedBottleneck, 256, num_blocks[3])
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, num_blocks):
        shortcuts = [True] + [False] * (num_blocks - 1)
        layers = []
        for shortcut in shortcuts:
            layers.append(block(self.in_planes, planes, shortcut))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 2,
            stride=2, padding=2 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 2, stride=2, padding=2 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV11(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV11, self).__init__()
        self.global_net = GlobalNet152(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = F.relu(self.bottleneck3(p3))
        p4 = F.relu(self.bottleneck4(p4))
        p5 = F.relu(self.bottleneck5(p5))
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV12(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV12, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
        self.latlayer2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU())
        self.latlayer3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU())
        self.latlayer4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU())
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer4 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        p4 = self.latlayer2(c4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        p1 = self.toplayer4(p2)
        return p1, p2, p3, p4


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.output = nn.Sequential(Bottleneck(256 * 3, 64, 1), nn.Conv2d(
            256, config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4):
        p3 = F.relu(self.bottleneck3(p3))
        p4 = F.relu(self.bottleneck4(p4))
        return self.output(torch.cat([p2, p3, p4], dim=1))


class CascadePyramidNetV14(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV14, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p1, p2, p3, p4 = self.global_net(x)
        out = self.refine_net(p2, p3, p4)
        return p1, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.subnet = self._make_head(config.num_keypoints)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            conv.weight.data.normal_(0.0, 0.01)
            conv.bias.data.fill_(0)
            layers.append(conv)
            layers.append(nn.ReLU(True))
        final_conv = nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1)
        layers.append(final_conv)
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        o5 = self.subnet(p5)
        o4 = self.subnet(p4)
        o3 = self.subnet(p3)
        o2 = self.subnet(p2)
        return p2, p3, p4, p5, o2, o3, o4, o5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(256, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


def GlobalNet101(config, pretrained=False):
    if pretrained:
        return GlobalNet(config, Bottleneck, [3, 4, 23, 3], torchvision.
            models.resnet50(pretrained=True))
    return GlobalNet(config, Bottleneck, [3, 4, 23, 3])


class CascadePyramidNetV2(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV2, self).__init__()
        self.global_net = GlobalNet101(config, True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5, o2, o3, o4, o5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return (o2, o3, o4, o5), out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer5 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer4 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c0 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c0, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        p1 = self._upsample_add(p2, self.latlayer5(c0))
        p1 = self.toplayer4(p1)
        return p1, p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck1 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck2 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 16, stride=16, padding=16 // 2))
        self.output = nn.Sequential(Bottleneck(256 * 5, 64, 1), nn.Conv2d(
            256, config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p1, p2, p3, p4, p5):
        p1 = self.bottleneck1(p1)
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p1, p2, p3, p4, p5], dim=1))


class CascadePyramidNetV5(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV5, self).__init__()
        self.global_net = GlobalNet152(config, True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p1, p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p1, p2, p3, p4, p5)
        return p1, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(256, 256,
            kernel_size=3, stride=1, padding=1))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.UpsamplingBilinear2d(scale_factor=4), nn.
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.UpsamplingBilinear2d(
            scale_factor=8), nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV6(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV6, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 2,
            stride=2, padding=2 // 2)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 2,
            stride=2, padding=2 // 2)
        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 2,
            stride=2, padding=2 // 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self.upsample1(p5) + self.latlayer2(c4)
        p4 = self.toplayer1(p4)
        p3 = self.upsample2(p4) + self.latlayer3(c3)
        p3 = self.toplayer2(p3)
        p2 = self.upsample3(p3) + self.latlayer4(c2)
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV7(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV7, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(2048, 
            256, kernel_size=1, stride=1, padding=0))
        self.latlayer2 = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(1024, 
            256, kernel_size=1, stride=1, padding=0))
        self.latlayer3 = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(512, 
            256, kernel_size=1, stride=1, padding=0))
        self.latlayer4 = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(256, 
            256, kernel_size=1, stride=1, padding=0))
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(nn.Dropout2d(0.5), Bottleneck(1024, 64,
            1), nn.Conv2d(256, config.num_keypoints, kernel_size=1, stride=
            1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV8(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV8, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class VisErrorLoss(nn.Module):

    def __init__(self):
        super(VisErrorLoss, self).__init__()

    def compute_l1_weighted_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        amplitude = torch.max(hm_targets)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        pos_ids = (hm_targets > amplitude / 10) & (vismap >= 0)
        neg_ids = (hm_targets <= amplitude / 10) & (vismap >= 0)
        diff = (hm_targets - hm_preds).abs()
        pos_loss = (diff * pos_ids.float()).sum(2).sum(0) / (pos_ids.float(
            ).sum(2).sum(0) + epsilon)
        neg_loss = (diff * neg_ids.float()).sum(2).sum(0) / (neg_ids.float(
            ).sum(2).sum(0) + epsilon)
        total_loss = 0.5 * pos_loss + 0.5 * neg_loss
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def compute_l2_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        ids = vismap == 1
        diff = (hm_targets - hm_preds) ** 2
        total_loss = (diff * ids.float()).sum(2).sum(0) / (ids.float().sum(
            2).sum(0) + epsilon)
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def forward(self, hm_targets, hm_preds1, hm_preds2, vismap):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number] 
        :return: 
        """
        loss1 = self.compute_l1_weighted_loss(hm_targets, hm_preds1, vismap)
        loss2 = self.compute_l1_weighted_loss(hm_targets, hm_preds2, vismap,
            ohem=0.5)
        return loss1 + loss2, loss1, loss2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.latlayer4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1, padding=0), nn.ReLU(False))
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.subnet1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1), nn.ReLU(False), nn.Conv2d(256, config.num_keypoints,
            kernel_size=3, stride=1, padding=1))
        self.subnet2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1), nn.ReLU(False), nn.Conv2d(256, config.num_keypoints,
            kernel_size=3, stride=1, padding=1))
        self.subnet3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1), nn.ReLU(False), nn.Conv2d(256, config.num_keypoints,
            kernel_size=3, stride=1, padding=1))
        self.subnet4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1,
            stride=1), nn.ReLU(False), nn.Conv2d(256, config.num_keypoints,
            kernel_size=3, stride=1, padding=1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, conv1x1, y):
        _, _, H, W = y.size()
        return conv1x1(F.upsample(x, size=(H, W), mode='bilinear')) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.toplayer1, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.toplayer2, self.latlayer3(c3))
        p2 = self._upsample_add(p3, self.toplayer3, self.latlayer4(c2))
        o5 = F.upsample(self.subnet1(p5), scale_factor=8, mode='bilinear')
        o4 = F.upsample(self.subnet2(p4), scale_factor=4, mode='bilinear')
        o3 = F.upsample(self.subnet3(p3), scale_factor=2, mode='bilinear')
        o2 = self.subnet4(p2)
        return (p2, p3, p4, p5), (o2, o3, o4, o5)


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=3, stride=1, padding=1))

    def forward(self, p2, p3, p4, p5):
        p3 = F.relu(self.bottleneck3(p3))
        p4 = F.relu(self.bottleneck4(p4))
        p5 = F.relu(self.bottleneck5(p5))
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV13(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV13, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        (p2, p3, p4, p5), (o2, o3, o4, o5) = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return (o2, o3, o4, o5), out


class VisErrorLossV13(nn.Module):

    def __init__(self):
        super(VisErrorLossV13, self).__init__()

    def compute_l1_weighted_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        amplitude = torch.max(hm_targets)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        pos_ids = (hm_targets > amplitude / 10) & (vismap == 1)
        neg_ids = (hm_targets <= amplitude / 10) & (vismap == 1)
        diff = (hm_targets - hm_preds).abs()
        pos_loss = (diff * pos_ids.float()).sum(2).sum(0) / (pos_ids.float(
            ).sum(2).sum(0) + epsilon)
        neg_loss = (diff * neg_ids.float()).sum(2).sum(0) / (neg_ids.float(
            ).sum(2).sum(0) + epsilon)
        total_loss = 0.5 * pos_loss + 0.5 * neg_loss
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def compute_l2_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        ids = vismap == 1
        diff = (hm_targets - hm_preds) ** 2
        total_loss = (diff * ids.float()).sum(2).sum(0) / (ids.float().sum(
            2).sum(0) + epsilon)
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def forward(self, hm_targets, hm_preds1, hm_preds2, vismap):
        """
        :param hm_targets: list of 4 elements, each is [batch size, keypoint number, h, w]
        :param hm_preds1: list of 4 elements, each is [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number] 
        :return: 
        """
        loss1 = 0
        for p in hm_preds1:
            loss1 += self.compute_l1_weighted_loss(hm_targets, p, vismap)
        loss1 /= 4.0
        loss2 = self.compute_l1_weighted_loss(hm_targets, hm_preds2, vismap,
            ohem=0.5)
        return loss1 + loss2, loss1, loss2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


pretrained_settings = {'senet154': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
    'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1
    ], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'se_resnet50': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1
    ], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'se_resnet101': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
    'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1
    ], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'se_resnet152': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
    'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1
    ], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'se_resnext50_32x4d': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'
    , 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0,
    1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'se_resnext101_32x4d': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'
    , 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0,
    1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}}


def nasnetalarge(num_classes=1001, pretrained='imagenet'):
    """NASNetALarge model architecture from the
    `"NASNet" <https://arxiv.org/abs/1707.07012>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['nasnetalarge'][pretrained]
        assert num_classes == settings['num_classes'
            ], 'num_classes should be {}, but is {}'.format(settings[
            'num_classes'], num_classes)
        model = NASNetALarge(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(model.last_linear.in_features, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = NASNetALarge(num_classes=num_classes)
    return model


class GlobalNet(nn.Module):

    def __init__(self, config):
        super(GlobalNet, self).__init__()
        pretrained_model = nasnetalarge(num_classes=1000, pretrained='imagenet'
            )
        self.pm = pretrained_model
        self.latlayer1 = nn.Conv2d(4032, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(2016, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(1008, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer4 = nn.Conv2d(168, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def features(self, input):
        x_conv0 = self.pm.conv0(input)
        x_stem_0 = self.pm.cell_stem_0(x_conv0)
        x_stem_1 = self.pm.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.pm.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.pm.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.pm.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.pm.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.pm.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.pm.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.pm.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.pm.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.pm.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.pm.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.pm.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.pm.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.pm.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.pm.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.pm.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.pm.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.pm.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.pm.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.pm.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.pm.cell_17(x_cell_16, x_cell_15)
        return x_stem_0, x_cell_5, x_cell_11, x_cell_17

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c2, c3, c4, c5 = self.features(x)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV15(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV15, self).__init__()
        self.global_net = GlobalNet(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters,
            self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels,
            kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96,
        penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False)
            )
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters //
            filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters //
            filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left
            =filters // 2, in_channels_right=2 * filters,
            out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_4 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_5 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters,
            out_channels_left=2 * filters, in_channels_right=6 * filters,
            out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=8 * filters,
            out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_10 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_11 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            filters, out_channels_left=4 * filters, in_channels_right=12 *
            filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=16 * filters,
            out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_16 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_17 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.subnet = self._make_head(config.num_keypoints)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU(True))
        final_conv = nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1)
        layers.append(final_conv)
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        o5 = self.subnet(p5)
        o4 = self.subnet(p4)
        o3 = self.subnet(p3)
        o2 = self.subnet(p2)
        return p2, p3, p4, p5, o2, o3, o4, o5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(256, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV2(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV2, self).__init__()
        self.global_net = GlobalNet101(config, True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5, o2, o3, o4, o5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return (o2, o3, o4, o5), out


class VisErrorLossV2(nn.Module):

    def __init__(self):
        super(VisErrorLossV2, self).__init__()

    def compute_l1_weighted_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        amplitude = torch.max(hm_targets)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        pos_ids = (hm_targets > amplitude / 10) & (vismap == 1)
        neg_ids = (hm_targets <= amplitude / 10) & (vismap == 1)
        diff = (hm_targets - hm_preds).abs()
        pos_loss = (diff * pos_ids.float()).sum(2).sum(0) / (pos_ids.float(
            ).sum(2).sum(0) + epsilon)
        neg_loss = (diff * neg_ids.float()).sum(2).sum(0) / (neg_ids.float(
            ).sum(2).sum(0) + epsilon)
        total_loss = 0.5 * pos_loss + 0.5 * neg_loss
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def compute_l2_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        ids = vismap == 1
        diff = (hm_targets - hm_preds) ** 2
        total_loss = (diff * ids.float()).sum(2).sum(0) / (ids.float().sum(
            2).sum(0) + epsilon)
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def forward(self, hm_targets, hm_preds1, hm_preds2, vismap):
        """
        :param hm_targets: list of 4 elements, each is [batch size, keypoint number, h, w]
        :param hm_preds1: list of 4 elements, each is [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number] 
        :return: 
        """
        loss1 = 0
        for t, p in zip(hm_targets, hm_preds1):
            loss1 += self.compute_l1_weighted_loss(t, p, vismap)
            break
        loss2 = self.compute_l1_weighted_loss(hm_targets[0], hm_preds2,
            vismap, ohem=0.5)
        return loss1 + loss2, loss1, loss2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):

    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = Bottleneck(256, 64, 1)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 2,
            stride=2, padding=2 // 2)
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1))
        self.upsample4 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2)
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1))
        self.upsample8 = nn.ConvTranspose2d(256, 256, kernel_size=2 * 8,
            stride=8, padding=8 // 2)
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        u3 = self.upsample2(p3)
        p4 = self.bottleneck4(p4)
        u4 = self.upsample4(p4)
        p5 = self.bottleneck5(p5)
        u5 = self.upsample8(p5)
        output = self.output(torch.cat([p2, u3, u4, u5], dim=1))
        return output, p3, p4, p5


class CascadePyramidNetV3(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV3, self).__init__()
        self.global_net = GlobalNet152(config, True)
        self.refine_net_1 = RefineNet(config)
        self.refine_net_2 = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out1, p3, p4, p5 = self.refine_net_1(p2, p3, p4, p5)
        out2, p3, p4, p5 = self.refine_net_2(out1, p3, p4, p5)
        return p2, (out1, out2)


class VisErrorLossV3(nn.Module):

    def __init__(self):
        super(VisErrorLossV3, self).__init__()

    def compute_l1_weighted_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        amplitude = torch.max(hm_targets)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        pos_ids = (hm_targets > amplitude / 10) & (vismap == 1)
        neg_ids = (hm_targets <= amplitude / 10) & (vismap == 1)
        diff = (hm_targets - hm_preds).abs()
        pos_loss = (diff * pos_ids.float()).sum(2).sum(0) / (pos_ids.float(
            ).sum(2).sum(0) + epsilon)
        neg_loss = (diff * neg_ids.float()).sum(2).sum(0) / (neg_ids.float(
            ).sum(2).sum(0) + epsilon)
        total_loss = 0.5 * pos_loss + 0.5 * neg_loss
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def compute_l2_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        """
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        ids = vismap == 1
        diff = (hm_targets - hm_preds) ** 2
        total_loss = (diff * ids.float()).sum(2).sum(0) / (ids.float().sum(
            2).sum(0) + epsilon)
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def forward(self, hm_targets, hm_preds1, hm_preds2, vismap):
        """
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number] 
        :return: 
        """
        loss1 = self.compute_l1_weighted_loss(hm_targets, hm_preds1, vismap)
        loss2 = self.compute_l1_weighted_loss(hm_targets, hm_preds2[0],
            vismap, ohem=0.5)
        loss3 = self.compute_l1_weighted_loss(hm_targets, hm_preds2[1],
            vismap, ohem=0.3)
        return loss1 + loss2 + loss3, loss1, loss2, loss3


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    """InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'
            ], 'num_classes should be {}, but is {}'.format(settings[
            'num_classes'], num_classes)
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model


class GlobalNet(nn.Module):

    def __init__(self, config):
        super(GlobalNet, self).__init__()
        pretrained_model = inceptionresnetv2(num_classes=1000, pretrained=
            'imagenet')
        self.conv2d_1a = pretrained_model.conv2d_1a
        self.conv2d_2a = pretrained_model.conv2d_2a
        self.conv2d_2b = pretrained_model.conv2d_2b
        self.maxpool_3a = pretrained_model.maxpool_3a
        self.conv2d_3b = pretrained_model.conv2d_3b
        self.conv2d_4a = pretrained_model.conv2d_4a
        self.maxpool_5a = pretrained_model.maxpool_5a
        self.mixed_5b = pretrained_model.mixed_5b
        self.repeat = pretrained_model.repeat
        self.mixed_6a = pretrained_model.mixed_6a
        self.repeat_1 = pretrained_model.repeat_1
        self.mixed_7a = pretrained_model.mixed_7a
        self.repeat_2 = pretrained_model.repeat_2
        self.block8 = pretrained_model.block8
        self.conv2d_7b = pretrained_model.conv2d_7b
        self.latlayer1 = nn.Conv2d(1536, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1088, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        c1 = self.conv2d_2b(x)
        x = self.maxpool_3a(c1)
        x = self.conv2d_3b(x)
        c2 = self.conv2d_4a(x)
        x = self.maxpool_5a(c2)
        x = self.mixed_5b(x)
        c3 = self.repeat(x)
        x = self.mixed_6a(c3)
        c4 = self.repeat_1(x)
        x = self.mixed_7a(c4)
        x = self.repeat_2(x)
        x = self.block8(x)
        c5 = self.conv2d_7b(x)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV4(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV4, self).__init__()
        self.global_net = GlobalNet(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2, padding=1
            )
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2,
            padding=1))
        self.branch2 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2,
            padding=1))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2,
            padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2,
            padding=1))
        self.branch3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1
            )
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1,
            padding=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2, padding=1)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
        downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
            stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=
            False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'
        ], 'num_classes should be {}, but is {}'.format(settings[
        'num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
        dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class GlobalNet(nn.Module):

    def __init__(self, config):
        super(GlobalNet, self).__init__()
        pretrained_model = senet154(num_classes=1000, pretrained='imagenet')
        self.layer0 = pretrained_model.layer0
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3,
            stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5


class RefineNet(nn.Module):

    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1), nn.
            ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=
            2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), nn.ConvTranspose2d(256, 256, kernel_size=2 * 4,
            stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1), Bottleneck
            (256, 64, 1), Bottleneck(256, 64, 1), nn.ConvTranspose2d(256, 
            256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1), nn.Conv2d(256,
            config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))


class CascadePyramidNetV9(nn.Module):

    def __init__(self, config):
        super(CascadePyramidNetV9, self).__init__()
        self.global_net = GlobalNet(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gathierry_FashionAI_KeyPointsDetectionOfApparel(_paritybench_base):
    pass
    def test_000(self):
        self._check(AvgPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Block17(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_003(self):
        self._check(Block35(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_004(self):
        self._check(Block8(*[], **{}), [torch.rand([4, 2080, 64, 64])], {})

    def test_005(self):
        self._check(BranchSeparables(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(BranchSeparablesReduction(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(BranchSeparablesStem(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(CellStem0(*[], **{'stem_filters': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(CellStem1(*[], **{'stem_filters': 4, 'num_filters': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 8, 64, 64])], {})

    def test_010(self):
        self._check(DilatedBottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(MaxPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(Mixed_5b(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_013(self):
        self._check(Mixed_6a(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_014(self):
        self._check(Mixed_7a(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_015(self):
        self._check(NASNetALarge(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_016(self):
        self._check(NormalCell(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(ReductionCell0(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(ReductionCell1(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'dw_kernel': 4, 'dw_stride': 1, 'dw_padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_021(self):
        self._check(VisErrorLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

