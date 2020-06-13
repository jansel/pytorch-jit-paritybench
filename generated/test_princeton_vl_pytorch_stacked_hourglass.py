import sys
_module = sys.modules[__name__]
del sys
dp = _module
ref = _module
layers = _module
posenet = _module
loss = _module
pose = _module
test = _module
train = _module
group = _module
img = _module
misc = _module

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


from torch import nn


import torch


import numpy as np


from torch.nn import DataParallel


class Conv(nn.Module):

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False,
        relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
            padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, '{} {}'.format(x.size()[1],
            self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):

    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


Pool = nn.MaxPool2d


class Hourglass(nn.Module):

    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class UnFlatten(nn.Module):

    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):

    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):

    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs
        ):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128), Pool(2, 2), Residual(128, 128), Residual(128,
            inp_dim))
        self.hgs = nn.ModuleList([nn.Sequential(Hourglass(4, inp_dim, bn,
            increase)) for i in range(nstack)])
        self.features = nn.ModuleList([nn.Sequential(Residual(inp_dim,
            inp_dim), Conv(inp_dim, inp_dim, 1, bn=True, relu=True)) for i in
            range(nstack)])
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn
            =False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in
            range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in
            range(nstack - 1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](
                    feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, (
                i)], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = (pred - gt) ** 2
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l


class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """

    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}
        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]
        if not self.training:
            return self.model(imgs, **inps)
        else:
            combined_hm_preds = self.model(imgs, **inps)
            if type(combined_hm_preds) != list and type(combined_hm_preds
                ) != tuple:
                combined_hm_preds = [combined_hm_preds]
            loss = self.calc_loss(**labels, combined_hm_preds=combined_hm_preds
                )
            return list(combined_hm_preds) + list([loss])


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_princeton_vl_pytorch_stacked_hourglass(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Residual(*[], **{'inp_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Merge(*[], **{'x_dim': 4, 'y_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(HeatmapLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

