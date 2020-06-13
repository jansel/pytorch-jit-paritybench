import sys
_module = sys.modules[__name__]
del sys
demo = _module
models = _module
with_mobilenet = _module
modules = _module
conv = _module
draw = _module
inference_engine_openvino = _module
inference_engine_pytorch = _module
input_reader = _module
legacy_pose_extractor = _module
load_state = _module
one_euro_filter = _module
parse_poses = _module
pose = _module
convert_to_onnx = _module
setup = _module

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


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True,
    dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1,
    stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
        stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True), nn.Conv2d(in_channels, out_channels, 1, 1, 0,
        bias=False), nn.ELU(inplace=True))


class Cpm(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding
            =0, bn=False)
        self.trunk = nn.Sequential(conv_dw_no_bn(out_channels, out_channels
            ), conv_dw_no_bn(out_channels, out_channels), conv_dw_no_bn(
            out_channels, out_channels))
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):

    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(conv(num_channels, num_channels, bn=
            False), conv(num_channels, num_channels, bn=False), conv(
            num_channels, num_channels, bn=False))
        self.heatmaps = nn.Sequential(conv(num_channels, 512, kernel_size=1,
            padding=0, bn=False), conv(512, num_heatmaps, kernel_size=1,
            padding=0, bn=False, relu=False))
        self.pafs = nn.Sequential(conv(num_channels, 512, kernel_size=1,
            padding=0, bn=False), conv(512, num_pafs, kernel_size=1,
            padding=0, bn=False, relu=False))

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1,
            padding=0, bn=False)
        self.trunk = nn.Sequential(conv(out_channels, out_channels), conv(
            out_channels, out_channels, dilation=2, padding=2))

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):

    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(RefinementStageBlock(in_channels,
            out_channels), RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels))
        self.heatmaps = nn.Sequential(conv(out_channels, out_channels,
            kernel_size=1, padding=0, bn=False), conv(out_channels,
            num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False))
        self.pafs = nn.Sequential(conv(out_channels, out_channels,
            kernel_size=1, padding=0, bn=False), conv(out_channels,
            num_pafs, kernel_size=1, padding=0, bn=False, relu=False))

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageLight(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.trunk = nn.Sequential(RefinementStageBlock(in_channels,
            mid_channels), RefinementStageBlock(mid_channels, mid_channels))
        self.feature_maps = nn.Sequential(conv(mid_channels, mid_channels,
            kernel_size=1, padding=0, bn=False), conv(mid_channels,
            out_channels, kernel_size=1, padding=0, bn=False, relu=False))

    def forward(self, x):
        trunk_features = self.trunk(x)
        feature_maps = self.feature_maps(trunk_features)
        return [feature_maps]


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ratio, should_align=False):
        super().__init__()
        self.should_align = should_align
        self.bottleneck = nn.Sequential(conv(in_channels, in_channels //
            ratio, kernel_size=1, padding=0), conv(in_channels // ratio, 
            in_channels // ratio), conv(in_channels // ratio, out_channels,
            kernel_size=1, padding=0))
        if self.should_align:
            self.align = conv(in_channels, out_channels, kernel_size=1,
                padding=0)

    def forward(self, x):
        res = self.bottleneck(x)
        if self.should_align:
            x = self.align(x)
        return x + res


class Pose3D(nn.Module):

    def __init__(self, in_channels, num_2d_heatmaps, ratio=2, out_channels=57):
        super().__init__()
        self.stem = nn.Sequential(ResBlock(in_channels + num_2d_heatmaps,
            in_channels, ratio, should_align=True), ResBlock(in_channels,
            in_channels, ratio), ResBlock(in_channels, in_channels, ratio),
            ResBlock(in_channels, in_channels, ratio), ResBlock(in_channels,
            in_channels, ratio))
        self.prediction = RefinementStageLight(in_channels, in_channels,
            out_channels)

    def forward(self, x, feature_maps_2d):
        stem = self.stem(torch.cat([x, feature_maps_2d], 1))
        feature_maps = self.prediction(stem)
        return feature_maps


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1,
    dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
        stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(
        in_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(
        out_channels), nn.ReLU(inplace=True))


class PoseEstimationWithMobileNet(nn.Module):

    def __init__(self, num_refinement_stages=1, num_channels=128,
        num_heatmaps=19, num_pafs=38, is_convertible_by_mo=False):
        super().__init__()
        self.is_convertible_by_mo = is_convertible_by_mo
        self.model = nn.Sequential(conv(3, 32, stride=2, bias=False),
            conv_dw(32, 64), conv_dw(64, 128, stride=2), conv_dw(128, 128),
            conv_dw(128, 256, stride=2), conv_dw(256, 256), conv_dw(256, 
            512), conv_dw(512, 512, dilation=2, padding=2), conv_dw(512, 
            512), conv_dw(512, 512), conv_dw(512, 512), conv_dw(512, 512))
        self.cpm = Cpm(512, num_channels)
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels +
                num_heatmaps + num_pafs, num_channels, num_heatmaps, num_pafs))
        self.Pose3D = Pose3D(128, num_2d_heatmaps=57)
        if self.is_convertible_by_mo:
            self.fake_conv_heatmaps = nn.Conv2d(num_heatmaps, num_heatmaps,
                kernel_size=1, bias=False)
            self.fake_conv_heatmaps.weight = nn.Parameter(torch.zeros(
                num_heatmaps, num_heatmaps, 1, 1))
            self.fake_conv_pafs = nn.Conv2d(num_pafs, num_pafs, kernel_size
                =1, bias=False)
            self.fake_conv_pafs.weight = nn.Parameter(torch.zeros(num_pafs,
                num_pafs, 1, 1))

    def forward(self, x):
        model_features = self.model(x)
        backbone_features = self.cpm(model_features)
        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(refinement_stage(torch.cat([
                backbone_features, stages_output[-2], stages_output[-1]],
                dim=1)))
        keypoints2d_maps = stages_output[-2]
        paf_maps = stages_output[-1]
        if self.is_convertible_by_mo:
            keypoints2d_maps = stages_output[-2] + self.fake_conv_heatmaps(
                stages_output[-2])
            paf_maps = stages_output[-1] + self.fake_conv_pafs(stages_output
                [-1])
        out = self.Pose3D(backbone_features, torch.cat([stages_output[-2],
            stages_output[-1]], dim=1))
        return out, keypoints2d_maps, paf_maps


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Daniil_Osokin_lightweight_human_pose_estimation_3d_demo_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Cpm(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(InitialStage(*[], **{'num_channels': 4, 'num_heatmaps': 4, 'num_pafs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(RefinementStageBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(RefinementStage(*[], **{'in_channels': 4, 'out_channels': 4, 'num_heatmaps': 4, 'num_pafs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(RefinementStageLight(*[], **{'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(ResBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(Pose3D(*[], **{'in_channels': 4, 'num_2d_heatmaps': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(PoseEstimationWithMobileNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

