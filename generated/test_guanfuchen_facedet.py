import sys
_module = sys.modules[__name__]
del sys
demo = _module
facedet = _module
dataloader = _module
aflw_face_loader = _module
fddb_face_loader = _module
wider_face_loader = _module
modelloader = _module
cascadeface = _module
facebox = _module
mtcnnface = _module
utils = _module
test = _module
context = _module
test_cascadeface = _module
test_facebox = _module
test_mtcnnface = _module
torch2caffe = _module
train = _module
train_mtcnn = _module
validate = _module

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


from torch.autograd import Variable


from torch import optim


from torch.utils import data


import torch.nn.functional as F


import numpy as np


from torch import nn


import math


import itertools


class LRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True
        ):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1,
                padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Cascade12Net(nn.Module):

    def __init__(self):
        super(Cascade12Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=
            3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(400, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class Cascade12CalNet(nn.Module):

    def __init__(self):
        super(Cascade12CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=
            3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(400, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 45)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class Cascade24Net(nn.Module):

    def __init__(self):
        super(Cascade24Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=
            5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(6400, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class Cascade24CalNet(nn.Module):

    def __init__(self):
        super(Cascade24CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=
            5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(3200, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 45)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class Cascade48Net(nn.Module):

    def __init__(self):
        super(Cascade48Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=
            5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=5e-05, beta=0.75)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =5, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=5e-05, beta=0.75)
        self.fc1 = nn.Linear(5184, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class Cascade48CalNet(nn.Module):

    def __init__(self):
        super(Cascade48CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=
            5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=5e-05, beta=0.75)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =5, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=5e-05, beta=0.75)
        self.fc1 = nn.Linear(5184, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 45)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class CReLU(nn.Module):

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)


class StrideConv(nn.Module):
    """
    StrideConv：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param groups:
        :param bias:
        """
        super(StrideConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class StridePool(nn.Module):
    """
    StridePool：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """

    def __init__(self, kernel_size, stride=None):
        super(StridePool, self).__init__()
        padding = (kernel_size - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


class Inception(nn.Module):
    """
    Inception：输入128，32，32，输出也是128，32，32
    """

    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = StrideConv(in_channels=128, out_channels=32, kernel_size=1
            )
        self.pool1 = StridePool(kernel_size=3, stride=1)
        self.conv2 = StrideConv(in_channels=128, out_channels=32, kernel_size=1
            )
        self.conv3 = StrideConv(in_channels=128, out_channels=24, kernel_size=1
            )
        self.conv4 = StrideConv(in_channels=24, out_channels=32, kernel_size=3)
        self.conv5 = StrideConv(in_channels=128, out_channels=24, kernel_size=1
            )
        self.conv6 = StrideConv(in_channels=24, out_channels=32, kernel_size=3)
        self.conv7 = StrideConv(in_channels=32, out_channels=32, kernel_size=3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x)
        x2 = self.conv2(x2)
        x3 = self.conv3(x)
        x3 = self.conv4(x3)
        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)
        return torch.cat((x1, x2, x3, x4), 1)


class FaceBoxExtractor(nn.Module):

    def __init__(self):
        super(FaceBoxExtractor, self).__init__()
        self.conv1 = StrideConv(in_channels=3, out_channels=24, kernel_size
            =7, stride=4)
        self.crelu1 = CReLU()
        self.pool1 = StridePool(kernel_size=3, stride=2)
        self.conv2 = StrideConv(in_channels=24 * 2, out_channels=64,
            kernel_size=5, stride=2)
        self.crelu2 = CReLU()
        self.pool2 = StridePool(kernel_size=3, stride=2)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        self.conv3_1 = StrideConv(in_channels=128, out_channels=128,
            kernel_size=1, stride=1)
        self.conv3_2 = StrideConv(in_channels=128, out_channels=256,
            kernel_size=3, stride=2)
        self.conv4_1 = StrideConv(in_channels=256, out_channels=128,
            kernel_size=1, stride=1)
        self.conv4_2 = StrideConv(in_channels=128, out_channels=256,
            kernel_size=3, stride=2)

    def forward(self, x):
        xs = []
        x = self.conv1(x)
        x = self.crelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.crelu2(x)
        x = self.pool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x1 = self.inception3(x)
        xs.append(x1)
        x = x1
        x = self.conv3_1(x)
        x2 = self.conv3_2(x)
        xs.append(x2)
        x = x2
        x = self.conv4_1(x)
        x3 = self.conv4_2(x)
        xs.append(x3)
        x = x3
        return xs


class FaceBox(nn.Module):
    steps = 32 / 1024.0, 64 / 1024.0, 128 / 1024.0
    fm_sizes = 32, 16, 8
    aspect_ratios = (1, 2, 4), (1,), (1,)
    box_sizes = 32 / 1024.0, 256 / 1024.0, 512 / 1024.0
    density = (-3, -1, 1, 3), (-1, 1), (0,)

    def __init__(self, num_classes=2):
        super(FaceBox, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 21, 1, 1
        self.in_channels = 128, 256, 256
        self.extractor = FaceBoxExtractor()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers.append(StrideConv(self.in_channels[i], self.
                num_anchors[i] * 4, kernel_size=3))
            self.conf_layers.append(StrideConv(self.in_channels[i], self.
                num_anchors[i] * self.num_classes, kernel_size=3))

    def forward(self, x):
        loc_preds = []
        conf_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)
            loc_preds.append(loc_pred)
            conf_pred = self.conf_layers[i](x)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.view(conf_pred.size(0), -1, 2)
            conf_preds.append(conf_pred)
        loc_preds = torch.cat(loc_preds, 1)
        conf_preds = torch.cat(conf_preds, 1)
        return loc_preds, conf_preds


class FaceBoxLoss(nn.Module):

    def __init__(self, num_classes=2):
        super(FaceBoxLoss, self).__init__()
        self.num_classes = num_classes

    def cross_entropy_loss(self, x, y):
        x = x.detach()
        y = y.detach()
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1, keepdim=True)
            ) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1))

    def hard_negative_mining(self, conf_loss, pos):
        """
        conf_loss [N*21482,]
        pos [N,21482]
        return negative indice
        """
        batch_size, num_boxes = pos.size()
        conf_loss[pos.view(-1, 1)] = 0
        conf_loss = conf_loss.view(batch_size, -1)
        _, idx = conf_loss.sort(1, descending=True)
        _, rank = idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)
        neg = rank < num_neg.expand_as(rank)
        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """
        loc_preds[batch,21842,4]
        loc_targets[batch,21842,4]
        conf_preds[batch,21842,2]
        conf_targets[batch,21842]
        """
        batch_size, num_boxes, _ = loc_preds.size()
        pos = conf_targets > 0
        num_pos = pos.float().sum(1, keepdim=True)
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]), requires_grad=True)
        pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
        pos_loc_preds = loc_preds[pos_mask1].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask1].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets,
            size_average=False)
        conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.
            num_classes), conf_targets.view(-1, 1))
        neg = self.hard_negative_mining(conf_loss, pos)
        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
        mask = (pos_mask + neg_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        preds = conf_preds[mask].view(-1, self.num_classes)
        targets = conf_targets[pos_and_neg]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        N = num_pos.data.sum()
        loc_loss /= N
        conf_loss /= N
        None
        return loc_loss + conf_loss


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=
            3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size
            =3, stride=1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size
            =3, stride=1)
        self.relu3 = nn.PReLU()
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1,
            kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4,
            kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=10,
            kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        cls = F.sigmoid(self.conv4_1(x))
        box = self.conv4_2(x)
        return cls, box


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=
            3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size
            =3, stride=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size
            =2, stride=1)
        self.relu3 = nn.PReLU()
        self.fc4 = nn.Linear(in_features=576, out_features=128)
        self.relu4 = nn.PReLU()
        self.fc5_1 = nn.Linear(in_features=128, out_features=1)
        self.fc5_2 = nn.Linear(in_features=128, out_features=4)
        self.fc5_3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(batch_size, -1)
        x = self.fc4(x)
        x = self.relu4(x)
        cls = F.sigmoid(self.fc5_1(x))
        box = self.fc5_2(x)
        return cls, box


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=
            3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =3, stride=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =3, stride=1)
        self.relu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=2, stride=1)
        self.relu4 = nn.PReLU()
        self.fc5 = nn.Linear(in_features=1152, out_features=256)
        self.relu5 = nn.PReLU()
        self.fc6_1 = nn.Linear(in_features=256, out_features=1)
        self.fc6_2 = nn.Linear(in_features=256, out_features=4)
        self.fc6_3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(batch_size, -1)
        x = self.fc5(x)
        x = self.relu5(x)
        cls = F.sigmoid(self.fc6_1(x))
        box = self.fc6_2(x)
        landmark = self.fc6_3(x)
        return cls, box, landmark


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_guanfuchen_facedet(_paritybench_base):
    pass
    def test_000(self):
        self._check(LRN(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CReLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(StrideConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(StridePool(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Inception(*[], **{}), [torch.rand([4, 128, 64, 64])], {})

    def test_005(self):
        self._check(FaceBoxExtractor(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(FaceBox(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_007(self):
        self._check(PNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

