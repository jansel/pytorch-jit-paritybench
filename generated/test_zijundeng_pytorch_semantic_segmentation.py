import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cityscapes = _module
voc = _module
eval_voc = _module
models = _module
config = _module
duc_hdc = _module
fcn16s = _module
fcn32s = _module
fcn8s = _module
gcn = _module
psp_net = _module
seg_net = _module
u_net = _module
train = _module
utils = _module
joint_transforms = _module
misc = _module
transforms = _module

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


from math import ceil


import numpy as np


from torch.autograd import Variable


class _DenseUpsamplingConvModule(nn.Module):

    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = down_factor ** 2 * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


root = '/media/b3-542/LIBRARY/Datasets/cityscapes'


class ResNetDUCHDC(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
            self.layer3[idx].conv2.padding = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = layer4_group_config[idx
                ], layer4_group_config[idx]
            self.layer4[idx].conv2.padding = layer4_group_config[idx
                ], layer4_group_config[idx]
        self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
        dtype=np.float64)
    weight[(list(range(in_channels))), (list(range(out_channels))), :, :
        ] = filt
    return torch.from_numpy(weight).float()


class FCN32VGG(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))
        features, classifier = list(vgg.features.children()), list(vgg.
            classifier.children())
        features[0].padding = 100, 100
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True
        self.features5 = nn.Sequential(*features)
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(fc6, nn.ReLU(inplace=True), nn.
            Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes,
            num_classes, 64))

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19:19 + x_size[2], 19:19 + x_size[3]].contiguous()


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class GCN(nn.Module):

    def __init__(self, num_classes, input_size, pretrained=True):
        super(GCN, self).__init__()
        self.input_size = input_size
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gcm1 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)
        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self
            .brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6,
            self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        fm0 = self.layer0(x)
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))
        fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)
        fs2 = self.brm6(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)
        fs3 = self.brm7(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)
        fs4 = self.brm8(F.upsample_bilinear(fs3, fm0.size()[2:]))
        out = self.brm9(F.upsample_bilinear(fs4, self.input_size))
        return out


class _PyramidPoolingModule(nn.Module):

    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.
                Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=0.95), nn.ReLU(
                inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNetDeform(nn.Module):

    def __init__(self, num_classes, input_size, pretrained=True, use_aux=True):
        super(PSPNetDeform, self).__init__()
        self.input_size = input_size
        self.use_aux = use_aux
        resnet = models.resnet101()
        if pretrained:
            resnet.load_state_dict(torch.load(res101_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.padding = 1, 1
                m.stride = 1, 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.padding = 1, 1
                m.stride = 1, 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2 = Conv2dDeformable(self.layer3[idx].conv2)
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2 = Conv2dDeformable(self.layer4[idx].conv2)
        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(nn.Conv2d(4096, 512, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(512, momentum=0.95), nn.
            ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(512, num_classes,
            kernel_size=1))
        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)
        initialize_weights(self.ppm, self.final)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, self.input_size, mode='bilinear'), F.upsample(
                aux, self.input_size, mode='bilinear')
        return F.upsample(x, self.input_size, mode='bilinear')


class _DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels / 2
        layers = [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=
            2, stride=2), nn.Conv2d(in_channels, middle_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.
            ReLU(inplace=True)]
        layers += [nn.Conv2d(middle_channels, middle_channels, kernel_size=
            3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace
            =True)] * (num_conv_layers - 2)
        layers += [nn.Conv2d(middle_channels, out_channels, kernel_size=3,
            padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.
            BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(
            out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels, middle_channels,
            kernel_size=3), nn.BatchNorm2d(middle_channels), nn.ReLU(
            inplace=True), nn.Conv2d(middle_channels, middle_channels,
            kernel_size=3), nn.BatchNorm2d(middle_channels), nn.ReLU(
            inplace=True), nn.ConvTranspose2d(middle_channels, out_channels,
            kernel_size=2, stride=2))

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3), nn.
            BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[
            2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:],
            mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:],
            mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:],
            mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True,
        ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.
            log_softmax(inputs), targets)


class Conv2dDeformable(nn.Module):

    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 *
            regular_filter.in_channels, kernel_size=3, padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()
        offset = self.offset_filter(x)
        offset_w, offset_h = torch.split(offset, self.regular_filter.
            in_channels, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(
            x_shape[3]))
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(
            x_shape[3]))
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np
                .linspace(-1, 1, x_shape[2]))
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w
                grid_h = grid_h
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w
        offset_h = offset_h + self.grid_h
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])
            ).unsqueeze(1)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(
            x_shape[3]))
        x = self.regular_filter(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zijundeng_pytorch_semantic_segmentation(_paritybench_base):
    pass
    def test_000(self):
        self._check(_BoundaryRefineModule(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(_DecoderBlock(*[], **{'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(_DenseUpsamplingConvModule(*[], **{'down_factor': 4, 'in_dim': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(_EncoderBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(_PyramidPoolingModule(*[], **{'in_dim': 4, 'reduction_dim': 4, 'setting': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

