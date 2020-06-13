import sys
_module = sys.modules[__name__]
del sys
autozoom = _module
benchmark = _module
common = _module
depthestim = _module
interface = _module

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


import math


import numpy


import random


import re


import scipy


import scipy.io


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(
                num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0
                ], out_channels=intChannels[2], kernel_size=1, stride=1,
                padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters
            =intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=
            intChannels[0], out_channels=intChannels[1], kernel_size=3,
            stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels
            [1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1],
            out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), torch.nn
            .PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.
            Conv2d(in_channels=intChannels[0], out_channels=intChannels[1],
            kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
            num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
            in_channels=intChannels[1], out_channels=intChannels[2],
            kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Semantics(torch.nn.Module):

    def __init__(self):
        super(Semantics, self).__init__()
        moduleVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval(
            )
        self.moduleVgg = torch.nn.Sequential(moduleVgg[0:3], moduleVgg[3:6],
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            moduleVgg[7:10], moduleVgg[10:13], torch.nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True), moduleVgg[14:17],
            moduleVgg[17:20], moduleVgg[20:23], moduleVgg[23:26], torch.nn.
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), moduleVgg[
            27:30], moduleVgg[30:33], moduleVgg[33:36], moduleVgg[36:39],
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

    def forward(self, tenInput):
        tenPreprocessed = tenInput[:, ([2, 1, 0]), :, :]
        tenPreprocessed[:, (0), :, :] = (tenPreprocessed[:, (0), :, :] - 0.485
            ) / 0.229
        tenPreprocessed[:, (1), :, :] = (tenPreprocessed[:, (1), :, :] - 0.456
            ) / 0.224
        tenPreprocessed[:, (2), :, :] = (tenPreprocessed[:, (2), :, :] - 0.406
            ) / 0.225
        return self.moduleVgg(tenPreprocessed)


class Disparity(torch.nn.Module):

    def __init__(self):
        super(Disparity, self).__init__()
        self.moduleImage = torch.nn.Conv2d(in_channels=3, out_channels=32,
            kernel_size=7, stride=2, padding=3)
        self.moduleSemantics = torch.nn.Conv2d(in_channels=512,
            out_channels=512, kernel_size=3, stride=1, padding=1)
        for intRow, intFeatures in [(0, 32), (1, 48), (2, 64), (3, 512), (4,
            512), (5, 512)]:
            self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1',
                Basic('relu-conv-relu-conv', [intFeatures, intFeatures,
                intFeatures]))
            self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2',
                Basic('relu-conv-relu-conv', [intFeatures, intFeatures,
                intFeatures]))
            self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3',
                Basic('relu-conv-relu-conv', [intFeatures, intFeatures,
                intFeatures]))
        for intCol in [0, 1]:
            self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol),
                Downsample([32, 48, 48]))
            self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol),
                Downsample([48, 64, 64]))
            self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol),
                Downsample([64, 512, 512]))
            self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol),
                Downsample([512, 512, 512]))
            self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol),
                Downsample([512, 512, 512]))
        for intCol in [2, 3]:
            self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol),
                Upsample([512, 512, 512]))
            self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol),
                Upsample([512, 512, 512]))
            self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol),
                Upsample([512, 64, 64]))
            self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol),
                Upsample([64, 48, 48]))
            self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol),
                Upsample([48, 32, 32]))
        self.moduleDisparity = Basic('conv-relu-conv', [32, 32, 1])

    def forward(self, tenImage, tenSemantics):
        tenColumn = [None, None, None, None, None, None]
        tenColumn[0] = self.moduleImage(tenImage)
        tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
        tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
        tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2]
            ) + self.moduleSemantics(tenSemantics)
        tenColumn[4] = self._modules['3x0 - 4x0'](tenColumn[3])
        tenColumn[5] = self._modules['4x0 - 5x0'](tenColumn[4])
        intColumn = 1
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(
                intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](
                tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' +
                    str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)
                    ](tenColumn[intRow - 1])
        intColumn = 2
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(
                intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](
                tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn
                    ) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn
                    [intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0,
                        0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1,
                        0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        intColumn = 3
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(
                intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](
                tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn
                    ) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn
                    [intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0,
                        0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1,
                        0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        return torch.nn.functional.threshold(input=self.moduleDisparity(
            tenColumn[0]), threshold=0.0, value=0.0)


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(
                num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0
                ], out_channels=intChannels[2], kernel_size=1, stride=1,
                padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters
            =intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=
            intChannels[0], out_channels=intChannels[1], kernel_size=3,
            stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels
            [1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1],
            out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), torch.nn
            .PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.
            Conv2d(in_channels=intChannels[0], out_channels=intChannels[1],
            kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
            num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
            in_channels=intChannels[1], out_channels=intChannels[2],
            kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Refine(torch.nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.moduleImageOne = Basic('conv-relu-conv', [3, 24, 24])
        self.moduleImageTwo = Downsample([24, 48, 48])
        self.moduleImageThr = Downsample([48, 96, 96])
        self.moduleDisparityOne = Basic('conv-relu-conv', [1, 96, 96])
        self.moduleDisparityTwo = Upsample([192, 96, 96])
        self.moduleDisparityThr = Upsample([144, 48, 48])
        self.moduleDisparityFou = Basic('conv-relu-conv', [72, 24, 24])
        self.moduleRefine = Basic('conv-relu-conv', [24, 24, 1])

    def forward(self, tenImage, tenDisparity):
        tenMean = [tenImage.view(tenImage.shape[0], -1).mean(1, True).view(
            tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.
            shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenStd = [tenImage.view(tenImage.shape[0], -1).std(1, True).view(
            tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.
            shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenImage = tenImage.clone()
        tenImage -= tenMean[0]
        tenImage /= tenStd[0] + 1e-07
        tenDisparity = tenDisparity.clone()
        tenDisparity -= tenMean[1]
        tenDisparity /= tenStd[1] + 1e-07
        tenImageOne = self.moduleImageOne(tenImage)
        tenImageTwo = self.moduleImageTwo(tenImageOne)
        tenImageThr = self.moduleImageThr(tenImageTwo)
        tenUpsample = self.moduleDisparityOne(tenDisparity)
        if tenUpsample.shape != tenImageThr.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample,
                size=(tenImageThr.shape[2], tenImageThr.shape[3]), mode=
                'bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityTwo(torch.cat([tenImageThr,
            tenUpsample], 1))
        tenImageThr = None
        if tenUpsample.shape != tenImageTwo.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample,
                size=(tenImageTwo.shape[2], tenImageTwo.shape[3]), mode=
                'bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityThr(torch.cat([tenImageTwo,
            tenUpsample], 1))
        tenImageTwo = None
        if tenUpsample.shape != tenImageOne.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample,
                size=(tenImageOne.shape[2], tenImageOne.shape[3]), mode=
                'bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityFou(torch.cat([tenImageOne,
            tenUpsample], 1))
        tenImageOne = None
        tenRefine = self.moduleRefine(tenUpsample)
        tenRefine *= tenStd[1] + 1e-07
        tenRefine += tenMean[1]
        return torch.nn.functional.threshold(input=tenRefine, threshold=0.0,
            value=0.0)


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(
                num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(
                in_channels=intChannels[0], out_channels=intChannels[1],
                kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
                num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
                in_channels=intChannels[1], out_channels=intChannels[2],
                kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0
                ], out_channels=intChannels[2], kernel_size=1, stride=1,
                padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters
            =intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=
            intChannels[0], out_channels=intChannels[1], kernel_size=3,
            stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels
            [1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1],
            out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), torch.nn
            .PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.
            Conv2d(in_channels=intChannels[0], out_channels=intChannels[1],
            kernel_size=3, stride=1, padding=1), torch.nn.PReLU(
            num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(
            in_channels=intChannels[1], out_channels=intChannels[2],
            kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


def spatial_filter(tenInput, strType):
    tenOutput = None
    if strType == 'laplacian':
        tenLaplacian = tenInput.new_zeros(tenInput.shape[1], tenInput.shape
            [1], 3, 3)
        for intKernel in range(tenInput.shape[1]):
            tenLaplacian[intKernel, intKernel, 0, 1] = -1.0
            tenLaplacian[intKernel, intKernel, 0, 2] = -1.0
            tenLaplacian[intKernel, intKernel, 1, 1] = 4.0
            tenLaplacian[intKernel, intKernel, 1, 0] = -1.0
            tenLaplacian[intKernel, intKernel, 2, 0] = -1.0
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[1, 1, 1, 1
            ], mode='replicate')
        tenOutput = torch.nn.functional.conv2d(input=tenOutput, weight=
            tenLaplacian)
    elif strType == 'median-3':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[1, 1, 1, 1
            ], mode='reflect')
        tenOutput = tenOutput.unfold(2, 3, 1).unfold(3, 3, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0],
            tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 3 * 3)
        tenOutput = tenOutput.median(-1, False)[0]
    elif strType == 'median-5':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[2, 2, 2, 2
            ], mode='reflect')
        tenOutput = tenOutput.unfold(2, 5, 1).unfold(3, 5, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0],
            tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 5 * 5)
        tenOutput = tenOutput.median(-1, False)[0]
    return tenOutput


def depth_to_points(tenDepth, fltFocal):
    tenHorizontal = torch.linspace(-0.5 * tenDepth.shape[3] + 0.5, 0.5 *
        tenDepth.shape[3] - 0.5, tenDepth.shape[3]).view(1, 1, 1, tenDepth.
        shape[3]).expand(tenDepth.shape[0], -1, tenDepth.shape[2], -1)
    tenHorizontal = tenHorizontal * (1.0 / fltFocal)
    tenHorizontal = tenHorizontal.type_as(tenDepth)
    tenVertical = torch.linspace(-0.5 * tenDepth.shape[2] + 0.5, 0.5 *
        tenDepth.shape[2] - 0.5, tenDepth.shape[2]).view(1, 1, tenDepth.
        shape[2], 1).expand(tenDepth.shape[0], -1, -1, tenDepth.shape[3])
    tenVertical = tenVertical * (1.0 / fltFocal)
    tenVertical = tenVertical.type_as(tenDepth)
    return torch.cat([tenDepth * tenHorizontal, tenDepth * tenVertical,
        tenDepth], 1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sniklaus_3d_ken_burns(_paritybench_base):
    pass
    def test_000(self):
        self._check(Downsample(*[], **{'intChannels': [4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Refine(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 1, 64, 64])], {})

    def test_002(self):
        self._check(Upsample(*[], **{'intChannels': [4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

