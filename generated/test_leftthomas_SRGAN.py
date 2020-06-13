import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
loss = _module
model = _module
pytorch_ssim = _module
test_benchmark = _module
test_image = _module
test_video = _module
train = _module

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


import math


from math import exp


import torch.nn.functional as F


from torch.autograd import Variable


class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        adversarial_loss = torch.mean(1 - out_labels)
        perception_loss = self.mse_loss(self.loss_network(out_images), self
            .loss_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        return (image_loss + 0.001 * adversarial_loss + 0.006 *
            perception_loss + 2e-08 * tv_loss)


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w
            ) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Generator(nn.Module):

    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding
            =4), nn.PReLU())
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,
            padding=1), nn.BatchNorm2d(64))
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), nn.Conv2d(64, 64, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Conv2d(64,
            128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.
            LeakyReLU(0.2), nn.Conv2d(128, 128, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Conv2d(
            128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.
            LeakyReLU(0.2), nn.Conv2d(256, 256, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), nn.Conv2d(
            256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.
            LeakyReLU(0.2), nn.Conv2d(512, 512, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.
            AdaptiveAvgPool2d(1), nn.Conv2d(512, 1024, kernel_size=1), nn.
            LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1))

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):

    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2,
            kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 
        sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0
        )
    window = Variable(_2D_window.expand(channel, 1, window_size,
        window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
        groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
        groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +
        C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type(
            ) == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.
            size_average)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leftthomas_SRGAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(TVLoss(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Generator(*[], **{'scale_factor': 1.0}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_003(self):
        self._check(ResidualBlock(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(SSIM(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

