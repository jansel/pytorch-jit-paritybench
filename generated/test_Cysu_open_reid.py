import sys
_module = sys.modules[__name__]
del sys
conf = _module
oim_loss = _module
softmax_loss = _module
triplet_loss = _module
reid = _module
datasets = _module
cuhk01 = _module
cuhk03 = _module
dukemtmc = _module
market1501 = _module
viper = _module
dist_metric = _module
evaluation_metrics = _module
classification = _module
ranking = _module
evaluators = _module
feature_extraction = _module
cnn = _module
database = _module
loss = _module
oim = _module
triplet = _module
metric_learning = _module
euclidean = _module
kissme = _module
models = _module
inception = _module
resnet = _module
trainers = _module
utils = _module
data = _module
dataset = _module
preprocessor = _module
sampler = _module
transforms = _module
logging = _module
meters = _module
osutils = _module
serialization = _module
setup = _module
test_cuhk01 = _module
test_cuhk03 = _module
test_dukemtmc = _module
test_market1501 = _module
test_viper = _module
test_cmc = _module
test_database = _module
test_oim = _module
test_inception = _module
test_preprocessor = _module

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


from torch import nn


from torch import autograd


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn import init


from torch.nn import Parameter


class OIM(autograd.Function):

    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum
                ) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):

    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
        weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
            size_average=self.size_average)
        return loss, inputs


class TripletLoss(nn.Module):

    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
        return loss, prec


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
    bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride
        =stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


class Block(nn.Module):

    def __init__(self, in_planes, out_planes, pool_method, stride):
        super(Block, self).__init__()
        self.branches = nn.ModuleList([nn.Sequential(_make_conv(in_planes,
            out_planes, kernel_size=1, padding=0), _make_conv(out_planes,
            out_planes, stride=stride)), nn.Sequential(_make_conv(in_planes,
            out_planes, kernel_size=1, padding=0), _make_conv(out_planes,
            out_planes), _make_conv(out_planes, out_planes, stride=stride))])
        if pool_method == 'Avg':
            assert stride == 1
            self.branches.append(_make_conv(in_planes, out_planes,
                kernel_size=1, padding=0))
            self.branches.append(nn.Sequential(nn.AvgPool2d(kernel_size=3,
                stride=1, padding=1), _make_conv(in_planes, out_planes,
                kernel_size=1, padding=0)))
        else:
            self.branches.append(nn.MaxPool2d(kernel_size=3, stride=stride,
                padding=1))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1)


class InceptionNet(nn.Module):

    def __init__(self, cut_at_pooling=False, num_features=256, norm=False,
        dropout=0, num_classes=0):
        super(InceptionNet, self).__init__()
        self.cut_at_pooling = cut_at_pooling
        self.conv1 = _make_conv(3, 32)
        self.conv2 = _make_conv(32, 32)
        self.conv3 = _make_conv(32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.in_planes = 32
        self.inception4a = self._make_inception(64, 'Avg', 1)
        self.inception4b = self._make_inception(64, 'Max', 2)
        self.inception5a = self._make_inception(128, 'Avg', 1)
        self.inception5b = self._make_inception(128, 'Max', 2)
        self.inception6a = self._make_inception(256, 'Avg', 1)
        self.inception6b = self._make_inception(256, 'Max', 2)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.has_embedding:
                self.feat = nn.Linear(self.in_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            else:
                self.num_features = self.in_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes
                    )
        self.reset_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception6a(x)
        x = self.inception6b(x)
        if self.cut_at_pooling:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def _make_inception(self, out_planes, pool_method, stride):
        block = Block(self.in_planes, out_planes, pool_method, stride)
        self.in_planes = (out_planes * 4 if pool_method == 'Avg' else 
            out_planes * 2 + self.in_planes)
        return block

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Cysu_open_reid(_paritybench_base):
    pass

    def test_000(self):
        self._check(Block(*[], **{'in_planes': 4, 'out_planes': 4, 'pool_method': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(InceptionNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
