import sys
_module = sys.modules[__name__]
del sys
CUB_loader = _module
Loss = _module
RACNN = _module
models = _module
vgg = _module
trainer = _module
utils = _module
visual = _module
logger = _module

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


import torch


import torch.nn.functional as F


import torch.nn as nn


from torch.autograd import Variable


import torch.utils.model_zoo as model_zoo


import math


import numpy as np


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.nn.init as init


import torch.utils.data as data


model_urls = {'vgg11':
    'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13':
    'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16':
    'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19':
    'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn':
    'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn':
    'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn':
    'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn':
    'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


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


_global_config['E'] = 4


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


class RACNN(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(RACNN, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False
        self.b1 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        self.b2 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        self.b3 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        self.feature_pool1 = nn.AvgPool2d(kernel_size=28, stride=28)
        self.feature_pool2 = nn.AvgPool2d(kernel_size=14, stride=14)
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apn1 = nn.Sequential(nn.Linear(512 * 14 * 14, 1024), nn.Tanh(),
            nn.Linear(1024, 3), nn.Sigmoid())
        self.apn2 = nn.Sequential(nn.Linear(512 * 14 * 14, 1024), nn.Tanh(),
            nn.Linear(1024, 3), nn.Sigmoid())
        self.crop_resize = AttentionCropLayer()
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(512, num_classes)

    def forward(self, x):
        conv5_4 = self.b1.features[:-1](x)
        pool5 = self.feature_pool1(conv5_4)
        atten1 = self.apn1(self.atten_pool(conv5_4).view(-1, 512 * 14 * 14))
        scaledA_x = self.crop_resize(x, atten1 * 448)
        conv5_4_A = self.b2.features[:-1](scaledA_x)
        pool5_A = self.feature_pool2(conv5_4_A)
        atten2 = self.apn2(conv5_4_A.view(-1, 512 * 14 * 14))
        scaledAA_x = self.crop_resize(scaledA_x, atten2 * 224)
        pool5_AA = self.feature_pool2(self.b3.features[:-1](scaledAA_x))
        pool5 = pool5.view(-1, 512)
        pool5_A = pool5_A.view(-1, 512)
        pool5_AA = pool5_AA.view(-1, 512)
        """#Feature fusion
        scale123 = torch.cat([pool5, pool5_A, pool5_AA], 1)
        scale12 = torch.cat([pool5, pool5_A], 1)
        """
        logits1 = self.classifier1(pool5)
        logits2 = self.classifier2(pool5_A)
        logits3 = self.classifier3(pool5_AA)
        return [logits1, logits2, logits3], [conv5_4, conv5_4_A], [atten1,
            atten2], [scaledA_x, scaledAA_x]


class AttentionCropFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1.0 / (1.0 + torch.exp(-10.0 * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tl = tl if tl > in_size / 3 else in_size / 3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size - tl else in_size - tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size - tl else in_size - tl
            w_off = int(tx - tl) if tx - tl > 0 else 0
            h_off = int(ty - tl) if ty - tl > 0 else 0
            w_end = int(tx + tl) if tx + tl < in_size else in_size
            h_end = int(ty + tl) if ty + tl < in_size else in_size
            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk
            xatt_cropped = xatt[:, w_off:w_end, h_off:h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(224, 224), mode=
                'bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())
        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = in_size / 3 * 2
        short_size = in_size / 3
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = ((x < short_size) + (x >= long_size) + (y < short_size) + (y >=
            long_size) > 0).float() * 2 - 1
        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()
        ret[:, (0)] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, (1)] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, (2)] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jeong_tae_RACNN_pytorch(_paritybench_base):
    pass
