import sys
_module = sys.modules[__name__]
del sys
dataset = _module
ctw = _module
data_util = _module
dataload = _module
deploy = _module
make_list = _module
synth_text = _module
total_text = _module
Deteval = _module
Pascal_VOC = _module
Python_scripts = _module
polygon_wrapper = _module
Evaluation_Protocol = _module
demo = _module
eval_textsnake = _module
network = _module
loss = _module
resnet = _module
textnet = _module
vgg = _module
train_textsnake = _module
util = _module
augmentation = _module
config = _module
detection = _module
misc = _module
option = _module
shedule = _module
summary = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models.resnet as resnet


import torch.utils.model_zoo as model_zoo


import time


import torch.backends.cudnn as cudnn


import torch.utils.data as data


from torch.optim import lr_scheduler


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem(self, predict, target, train_mask, negative_ratio=3.0):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()
        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = 0.0
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, tr_mask, tcl_mask, sin_map, cos_map, radii_map, train_mask):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """
        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)
        sin_pred = input[:, (4)].contiguous().view(-1)
        cos_pred = input[:, (5)].contiguous().view(-1)
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale
        radii_pred = input[:, (6)].contiguous().view(-1)
        train_mask = train_mask.view(-1)
        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        radii_map = radii_map.contiguous().view(-1)
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        loss_tcl = 0.0
        tr_train_mask = train_mask * tr_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl = F.cross_entropy(tcl_pred[tr_train_mask], tcl_mask[tr_train_mask].long())
        loss_radii, loss_sin, loss_cos = 0.0, 0.0, 0.0
        tcl_train_mask = train_mask * tcl_mask
        if tcl_train_mask.sum().item() > 0:
            ones = radii_map.new(radii_pred[tcl_mask].size()).fill_(1.0).float()
            loss_radii = F.smooth_l1_loss(radii_pred[tcl_mask] / radii_map[tcl_mask], ones)
            loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
            loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])
        return loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos


class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = resnet.resnet50(pretrained=True)
        self.stage1 = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool)
        self.stage2 = self.net.layer1
        self.stage3 = self.net.layer2
        self.stage4 = self.net.layer3
        self.stage5 = self.net.layer4

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', output_channel=7, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.output_channel = output_channel
        if backbone == 'vgg':
            self.backbone = VGG16(pretrain=self.is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512 + 256, 128)
            self.merge3 = Upsample(256 + 128, 64)
            self.merge2 = Upsample(128 + 64, 32)
            self.merge1 = Upsample(64 + 32, 16)
            self.predict = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.Conv2d(16, self.output_channel, kernel_size=1, stride=1, padding=0))
        elif backbone == 'resnet':
            pass

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)
        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)
        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)
        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)
        up1 = self.merge1(C1, up2)
        output = self.predict(up1)
        return output

    def load_model(self, model_path):
        None
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


model_urls = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


_global_config['D'] = 4


class VGG16(nn.Module):

    def __init__(self, pretrain=True):
        super().__init__()
        net = VGG(make_layers(cfg['D']), init_weights=False)
        if pretrain:
            net.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        self.stage1 = nn.Sequential(*[net.features[layer] for layer in range(0, 5)])
        self.stage2 = nn.Sequential(*[net.features[layer] for layer in range(5, 10)])
        self.stage3 = nn.Sequential(*[net.features[layer] for layer in range(10, 17)])
        self.stage4 = nn.Sequential(*[net.features[layer] for layer in range(17, 24)])
        self.stage5 = nn.Sequential(*[net.features[layer] for layer in range(24, 31)])

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResNet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (VGG,
     lambda: ([], {'features': _mock_layer()}),
     lambda: ([torch.rand([25088, 25088])], {}),
     True),
]

class Test_princewang1994_TextSnake_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

