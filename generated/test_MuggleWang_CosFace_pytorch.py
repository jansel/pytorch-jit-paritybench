import sys
_module = sys.modules[__name__]
del sys
dataset = _module
layer = _module
lfw_eval = _module
main = _module
net = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Parameter


import math


import torch.utils.data


import torch.optim


import torch.backends.cudnn as cudnn


def cosine_sim(x1, x2, dim=1, eps=1e-08):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', s=' + str(self.s) + ', m=' + str(self.m) + ')'


class AngleLinear(nn.Module):

    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x **
            2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 *
            x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, input, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.
            iter) ** (-1 * self.power))
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = (-1.0) ** k * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb
            ) + cos_theta
        output *= NormOfFeature.view(-1, 1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', m=' + str(self.m) + ')'


class Block(nn.Module):

    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        return x + self.prelu2(self.conv2(self.prelu1(self.conv1(x))))


class sphere(nn.Module):

    def __init__(self, type=20, is_gray=False):
        super(sphere, self).__init__()
        block = Block
        if type is 20:
            layers = [1, 2, 4, 1]
        elif type is 64:
            layers = [3, 7, 16, 3]
        else:
            raise ValueError('sphere' + str(type) +
                ' IS NOT SUPPORTED! (sphere20 or sphere64)')
        filter_list = [3, 64, 128, 256, 512]
        if is_gray:
            filter_list[0] = 1
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1
            ], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2
            ], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3
            ], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4
            ], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 6, 512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.PReLU(planes))
        for i in range(blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class BlockIR(nn.Module):

    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                planes))

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class LResNet(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet, self).__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride
                =1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride
                =1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1
            ], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2
            ], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3
            ], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4
            ], layers[3], stride=2)
        self.fc = nn.Sequential(nn.BatchNorm1d(filter_list[4] * 7 * 6), nn.
            Dropout(p=0.4), nn.Linear(filter_list[4] * 7 * 6, 512), nn.
            BatchNorm1d(512))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d
                ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MuggleWang_CosFace_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(Block(*[], **{'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BlockIR(*[], **{'inplanes': 4, 'planes': 4, 'stride': 1, 'dim_match': 4}), [torch.rand([4, 4, 4, 4])], {})
