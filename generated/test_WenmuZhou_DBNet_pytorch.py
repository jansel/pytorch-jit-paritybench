import sys
_module = sys.modules[__name__]
del sys
base = _module
base_dataset = _module
base_trainer = _module
data_loader = _module
dataset = _module
modules = _module
augment = _module
iaa_augment = _module
make_border_map = _module
make_shrink_map = _module
random_crop_data = _module
models = _module
loss = _module
model = _module
basic = _module
basic_loss = _module
resnet = _module
segmentation_body = _module
segmentation_head = _module
shufflenetv2 = _module
post_processing = _module
seg_detector_representer = _module
tools = _module
eval = _module
predict = _module
train = _module
trainer = _module
utils = _module
cal_recall = _module
rrc_evaluation_funcs = _module
script = _module
compute_mean_std = _module
make_trainfile = _module
metrics = _module
ocr_metric = _module
icdar2015 = _module
detection = _module
deteval = _module
icdar2013 = _module
iou = _module
mtwi2018 = _module
quad_metric = _module
schedulers = _module
util = _module

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


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


class DBLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean',
        eps=1e-06):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'
            ], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, (0), :, :]
        threshold_maps = pred[:, (1), :, :]
        binary_maps = pred[:, (2), :, :]
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'],
            batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch[
            'threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps,
            loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch[
                'shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = (self.alpha * loss_shrink_maps + self.beta *
                loss_threshold_maps + loss_binary_maps)
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BalanceCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, negative_ratio=3.0, eps=1e-06):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.
        Tensor, return_origin=False):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(
            positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    """
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    """

    def __init__(self, eps=1e-06):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, (0), :, :]
            gt = gt[:, (0), :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):

    def __init__(self, eps=1e-06):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


BatchNorm2d = nn.BatchNorm2d


def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dcn=None):
        self.dcn = dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn
            )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn
            )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn
            )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn)
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


class FPN(nn.Module):

    def __init__(self, backbone_out_channels, inner_channels=256):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
            inner_channels, kernel_size=1, inplace=inplace)
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels,
            kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels,
            kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels,
            kernel_size=3, padding=1, inplace=inplace)
        self.conv = nn.Sequential(nn.Conv2d(self.conv_out, self.conv_out,
            kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(self.
            conv_out), nn.ReLU(inplace=inplace))
        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear')
        return torch.cat([p2, p3, p4, p5], dim=1)


class FPEM_FFM(nn.Module):

    def __init__(self, backbone_out_channels, inner_channels=128, fpem_repeat=2
        ):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        self.conv_out = inner_channels
        inplace = True
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
            inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
            inner_channels, kernel_size=1, inplace=inplace)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(self.conv_out))
        self.out_channels = self.conv_out * 4

    def forward(self, x):
        c2, c3, c4, c5 = x
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear')
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        return Fy


class FPEM(nn.Module):

    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, padding=1, stride=
            stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class DBHead(nn.Module):

    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(nn.Conv2d(in_channels, in_channels //
            4, 3, padding=1), nn.BatchNorm2d(in_channels // 4), nn.ReLU(
            inplace=True), nn.ConvTranspose2d(in_channels // 4, in_channels //
            4, 2, 2), nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=
            True), nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize.apply(self.weights_init)
        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0001)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias
        =False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels //
            4, 3, padding=1, bias=bias), nn.BatchNorm2d(inner_channels // 4
            ), nn.ReLU(inplace=True), self._init_upsample(inner_channels //
            4, inner_channels // 4, smooth=smooth, bias=bias), nn.
            BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self.
            _init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias
            ), nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=
        False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [nn.Upsample(scale_factor=2, mode='nearest'), nn.
                Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels,
                    kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp,
                kernel_size=3, stride=self.stride, padding=1), nn.
                BatchNorm2d(inp), nn.Conv2d(inp, branch_features,
                kernel_size=1, stride=1, padding=0, bias=False), nn.
                BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else
            branch_features, branch_features, kernel_size=1, stride=1,
            padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.
            ReLU(inplace=True), self.depthwise_conv(branch_features,
            branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features,
            branch_features, kernel_size=1, stride=1, padding=0, bias=False
            ), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias,
            groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels,
            output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(
            output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names,
            stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels,
                    output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels,
            output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(
            output_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.maxpool(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c2, c3, c4, c5


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_WenmuZhou_DBNet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvBnRelu(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BalanceCrossEntropyLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(DiceLoss(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_003(self):
        self._check(MaskL1Loss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(FPN(*[], **{'backbone_out_channels': [4, 4, 4, 4]}), [torch.rand([4, 4, 4, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(FPEM_FFM(*[], **{'backbone_out_channels': [4, 4, 4, 4]}), [torch.rand([4, 4, 4, 64, 64])], {})

    def test_006(self):
        self._check(FPEM(*[], **{}), [torch.rand([4, 128, 4, 4]), torch.rand([4, 128, 4, 4]), torch.rand([4, 128, 64, 64]), torch.rand([4, 128, 4, 4])], {})

    def test_007(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ConvHead(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(DBHead(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

