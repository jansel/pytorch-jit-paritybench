import sys
_module = sys.modules[__name__]
del sys
augmentation = _module
coco = _module
imagematte = _module
spd = _module
videomatte = _module
youtubevis = _module
spd_preprocess = _module
evaluate_hr = _module
evaluate_lr = _module
generate_imagematte_with_background_image = _module
generate_imagematte_with_background_video = _module
generate_videomatte_with_background_image = _module
generate_videomatte_with_background_video = _module
hubconf = _module
inference = _module
inference_speed_test = _module
inference_utils = _module
model = _module
decoder = _module
deep_guided_filter = _module
fast_guided_filter = _module
lraspp = _module
mobilenetv3 = _module
model = _module
resnet = _module
train = _module
train_config = _module
train_loss = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import random


import torch


from torchvision import transforms


from torchvision.transforms import functional as F


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import Optional


from typing import Tuple


from torchvision.transforms.functional import to_pil_image


from torch import Tensor


from torch import nn


from torch.nn import functional as F


from torchvision.models.mobilenetv3 import MobileNetV3


from torchvision.models.mobilenetv3 import InvertedResidualConfig


from torchvision.transforms.functional import normalize


from typing import List


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import Bottleneck


from torch import distributed as dist


from torch import multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.optim import Adam


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.utils.data import ConcatDataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.tensorboard import SummaryWriter


from torchvision.utils import make_grid


from torchvision.transforms.functional import center_crop


class AvgPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)

    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3

    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3

    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class ConvGRU(nn.Module):

    def __init__(self, channels: int, kernel_size: int=3, padding: int=1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding), nn.Sigmoid())
        self.hh = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size, padding=padding), nn.Tanh())

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)), device=x.device, dtype=x.dtype)
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class BottleneckBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)

    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r


class OutputBlock(nn.Module):

    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x

    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x

    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r

    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class RecurrentDecoder(nn.Module):

    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, r1: Optional[Tensor], r2: Optional[Tensor], r3: Optional[Tensor], r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4


class Projection(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return self.conv(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class DeepGuidedFilterRefiner(nn.Module):

    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False, groups=4)
        self.box_filter.weight.data[...] = 1 / 9
        self.conv = nn.Sequential(nn.Conv2d(4 * 2 + hid_channels, hid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(hid_channels), nn.ReLU(True), nn.Conv2d(hid_channels, hid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(hid_channels), nn.ReLU(True), nn.Conv2d(hid_channels, 4, kernel_size=1, bias=True))

    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)
        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x = self.box_filter(base_x * base_x) - mean_x * mean_x
        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        b = mean_y - A * mean_x
        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)
        out = A * fine_x + b
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha

    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(fine_src.flatten(0, 1), base_src.flatten(0, 1), base_fgr.flatten(0, 1), base_pha.flatten(0, 1), base_hid.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha

    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha, base_hid)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid)


class BoxFilter(nn.Module):

    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        kernel_size = 2 * self.r + 1
        kernel_x = torch.full((x.data.shape[1], 1, 1, kernel_size), 1 / kernel_size, device=x.device, dtype=x.dtype)
        kernel_y = torch.full((x.data.shape[1], 1, kernel_size, 1), 1 / kernel_size, device=x.device, dtype=x.dtype)
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.data.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.data.shape[1])
        return x


class FastGuidedFilter(nn.Module):

    def __init__(self, r: int, eps: float=1e-05):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        mean_x = self.boxfilter(lr_x)
        mean_y = self.boxfilter(lr_y)
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        A = F.interpolate(A, hr_x.shape[2:], mode='bilinear', align_corners=False)
        b = F.interpolate(b, hr_x.shape[2:], mode='bilinear', align_corners=False)
        return A * hr_x + b


class FastGuidedFilterRefiner(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)

    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha):
        fine_src_gray = fine_src.mean(1, keepdim=True)
        base_src_gray = base_src.mean(1, keepdim=True)
        fgr, pha = self.guilded_filter(torch.cat([base_src, base_src_gray], dim=1), torch.cat([base_fgr, base_pha], dim=1), torch.cat([fine_src, fine_src_gray], dim=1)).split([3, 1], dim=1)
        return fgr, pha

    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(fine_src.flatten(0, 1), base_src.flatten(0, 1), base_fgr.flatten(0, 1), base_pha.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha

    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha)


class LRASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.aspp2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.Sigmoid())

    def forward_single_frame(self, x):
        return self.aspp1(x) * self.aspp2(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class MobileNetV3LargeEncoder(MobileNetV3):

    def __init__(self, pretrained: bool=False):
        super().__init__(inverted_residual_setting=[InvertedResidualConfig(16, 3, 16, 16, False, 'RE', 1, 1, 1), InvertedResidualConfig(16, 3, 64, 24, False, 'RE', 2, 1, 1), InvertedResidualConfig(24, 3, 72, 24, False, 'RE', 1, 1, 1), InvertedResidualConfig(24, 5, 72, 40, True, 'RE', 2, 1, 1), InvertedResidualConfig(40, 5, 120, 40, True, 'RE', 1, 1, 1), InvertedResidualConfig(40, 5, 120, 40, True, 'RE', 1, 1, 1), InvertedResidualConfig(40, 3, 240, 80, False, 'HS', 2, 1, 1), InvertedResidualConfig(80, 3, 200, 80, False, 'HS', 1, 1, 1), InvertedResidualConfig(80, 3, 184, 80, False, 'HS', 1, 1, 1), InvertedResidualConfig(80, 3, 184, 80, False, 'HS', 1, 1, 1), InvertedResidualConfig(80, 3, 480, 112, True, 'HS', 1, 1, 1), InvertedResidualConfig(112, 3, 672, 112, True, 'HS', 1, 1, 1), InvertedResidualConfig(112, 5, 672, 160, True, 'HS', 2, 2, 1), InvertedResidualConfig(160, 5, 960, 160, True, 'HS', 1, 2, 1), InvertedResidualConfig(160, 5, 960, 160, True, 'HS', 1, 2, 1)], last_channel=1280)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'))
        del self.avgpool
        del self.classifier

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        f4 = x
        return [f1, f2, f3, f4]

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class ResNet50Encoder(ResNet):

    def __init__(self, pretrained: bool=False):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], replace_stride_with_dilation=[False, False, True], norm_layer=None)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth'))
        del self.avgpool
        del self.fc

    def forward_single_frame(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = x
        x = self.layer2(x)
        f3 = x
        x = self.layer3(x)
        x = self.layer4(x)
        f4 = x
        return [f1, f2, f3, f4]

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class MattingNetwork(nn.Module):

    def __init__(self, variant: str='mobilenetv3', refiner: str='deep_guided_filter', pretrained_backbone: bool=False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self, src: Tensor, r1: Optional[Tensor]=None, r2: Optional[Tensor]=None, r3: Optional[Tensor]=None, r4: Optional[Tensor]=None, downsample_ratio: float=1, segmentation_pass: bool=False):
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0.0, 1.0)
            pha = pha.clamp(0.0, 1.0)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])], {}),
     True),
    (BoxFilter,
     lambda: ([], {'r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvGRU,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastGuidedFilter,
     lambda: ([], {'r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LRASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MattingNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (MobileNetV3LargeEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (OutputBlock,
     lambda: ([], {'in_channels': 4, 'src_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Projection,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet50Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UpsamplingBlock,
     lambda: ([], {'in_channels': 4, 'skip_channels': 4, 'src_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])], {}),
     True),
]

class Test_PeterL1n_RobustVideoMatting(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

