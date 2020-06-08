import sys
_module = sys.modules[__name__]
del sys
comparison = _module
correlation = _module
run = _module

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


import re


import numpy


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction].replace('{{intStride}}', str(
        objVariables['intStride']))
    while True:
        objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace(
            '}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for
            intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' +
            str.join('+', strIndex) + ']')
    return strKernel


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1,
            1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.
            shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1,
            tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.
            shape[3])
        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal,
            tenVertical], 1).cuda()
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) /
        2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(
        backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1
        ), mode='bilinear', padding_mode='zeros', align_corners=True)


arguments_strModel = 'default'


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()


        class Features(torch.nn.Module):

            def __init__(self):
                super(Features, self).__init__()
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=7, stride=1,
                    padding=3), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=32,
                    out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=64,
                    out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=64, out_channels=96, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=96,
                    out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=96, out_channels=128, kernel_size=3, stride
                    =2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=128, out_channels=192, kernel_size=3,
                    stride=2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1))

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


        class Matching(torch.nn.Module):

            def __init__(self, intLevel):
                super(Matching, self).__init__()
                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][
                    intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(
                        in_channels=32, out_channels=64, kernel_size=1,
                        stride=1, padding=0), torch.nn.LeakyReLU(inplace=
                        False, negative_slope=0.1))
                if intLevel == 6:
                    self.netUpflow = None
                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2,
                        out_channels=2, kernel_size=4, stride=2, padding=1,
                        bias=False, groups=2)
                if intLevel >= 4:
                    self.netUpcorr = None
                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=
                        49, out_channels=49, kernel_size=4, stride=2,
                        padding=1, bias=False, groups=49)
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=49, out_channels=128, kernel_size=3, stride
                    =1, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=128,
                    out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1), torch.nn.Conv2d(
                    in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5,
                    5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1,
                    1][intLevel]))

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                tenFeaturesSecond, tenFlow):
                tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
                tenFeaturesSecond = self.netFeat(tenFeaturesSecond)
                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                if tenFlow is not None:
                    tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond,
                        tenFlow=tenFlow * self.fltBackwarp)
                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(input=
                        correlation.FunctionCorrelation(tenFirst=
                        tenFeaturesFirst, tenSecond=tenFeaturesSecond,
                        intStride=1), negative_slope=0.1, inplace=False)
                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.
                        leaky_relu(input=correlation.FunctionCorrelation(
                        tenFirst=tenFeaturesFirst, tenSecond=
                        tenFeaturesSecond, intStride=2), negative_slope=0.1,
                        inplace=False))
                return (tenFlow if tenFlow is not None else 0.0
                    ) + self.netMain(tenCorrelation)


        class Subpixel(torch.nn.Module):

            def __init__(self, intLevel):
                super(Subpixel, self).__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][
                    intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(
                        in_channels=32, out_channels=64, kernel_size=1,
                        stride=1, padding=0), torch.nn.LeakyReLU(inplace=
                        False, negative_slope=0.1))
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel],
                    out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1), torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=
                    1, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=32,
                    out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][
                    intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][
                    intLevel]))

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                tenFeaturesSecond, tenFlow):
                tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
                tenFeaturesSecond = self.netFeat(tenFeaturesSecond)
                if tenFlow is not None:
                    tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond,
                        tenFlow=tenFlow * self.fltBackward)
                return (tenFlow if tenFlow is not None else 0.0
                    ) + self.netMain(torch.cat([tenFeaturesFirst,
                    tenFeaturesSecond, tenFlow], 1))


        class Regularization(torch.nn.Module):

            def __init__(self, intLevel):
                super(Regularization, self).__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][
                    intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]
                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(
                        in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel],
                        out_channels=128, kernel_size=1, stride=1, padding=
                        0), torch.nn.LeakyReLU(inplace=False,
                        negative_slope=0.1))
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel],
                    out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1), torch.nn.Conv2d(
                    in_channels=128, out_channels=64, kernel_size=3, stride
                    =1, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=64,
                    out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1), torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=
                    1, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1))
                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(torch.nn.Conv2d(
                        in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 
                        9][intLevel], kernel_size=[0, 0, 7, 5, 5, 3, 3][
                        intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][
                        intLevel]))
                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(torch.nn.Conv2d(
                        in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 
                        9][intLevel], kernel_size=([0, 0, 7, 5, 5, 3, 3][
                        intLevel], 1), stride=1, padding=([0, 0, 3, 2, 2, 1,
                        1][intLevel], 0)), torch.nn.Conv2d(in_channels=[0, 
                        0, 49, 25, 25, 9, 9][intLevel], out_channels=[0, 0,
                        49, 25, 25, 9, 9][intLevel], kernel_size=(1, [0, 0,
                        7, 5, 5, 3, 3][intLevel]), stride=1, padding=(0, [0,
                        0, 3, 2, 2, 1, 1][intLevel])))
                self.netScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25,
                    25, 9, 9][intLevel], out_channels=1, kernel_size=1,
                    stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25,
                    25, 9, 9][intLevel], out_channels=1, kernel_size=1,
                    stride=1, padding=0)

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                tenFeaturesSecond, tenFlow):
                tenDifference = (tenFirst - backwarp(tenInput=tenSecond,
                    tenFlow=tenFlow * self.fltBackward)).pow(2.0).sum(1, True
                    ).sqrt().detach()
                tenDist = self.netDist(self.netMain(torch.cat([
                    tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0],
                    2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1),
                    self.netFeat(tenFeaturesFirst)], 1)))
                tenDist = tenDist.pow(2.0).neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()
                tenDivisor = tenDist.sum(1, True).reciprocal()
                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.
                    unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.
                    intUnfold, stride=1, padding=int((self.intUnfold - 1) /
                    2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.
                    unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.
                    intUnfold, stride=1, padding=int((self.intUnfold - 1) /
                    2)).view_as(tenDist)) * tenDivisor
                return torch.cat([tenScaleX, tenScaleY], 1)
        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(intLevel) for
            intLevel in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for
            intLevel in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList([Regularization(
            intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for
            strKey, tenWeight in torch.load(__file__.replace('run.py', 
            'network-' + arguments_strModel + '.pytorch')).items()})

    def forward(self, tenFirst, tenSecond):
        tenFirst[:, (0), :, :] = tenFirst[:, (0), :, :] - 0.411618
        tenFirst[:, (1), :, :] = tenFirst[:, (1), :, :] - 0.434631
        tenFirst[:, (2), :, :] = tenFirst[:, (2), :, :] - 0.454253
        tenSecond[:, (0), :, :] = tenSecond[:, (0), :, :] - 0.410782
        tenSecond[:, (1), :, :] = tenSecond[:, (1), :, :] - 0.433645
        tenSecond[:, (2), :, :] = tenSecond[:, (2), :, :] - 0.452793
        tenFeaturesFirst = self.netFeatures(tenFirst)
        tenFeaturesSecond = self.netFeatures(tenSecond)
        tenFirst = [tenFirst]
        tenSecond = [tenSecond]
        for intLevel in [1, 2, 3, 4, 5]:
            tenFirst.append(torch.nn.functional.interpolate(input=tenFirst[
                -1], size=(tenFeaturesFirst[intLevel].shape[2],
                tenFeaturesFirst[intLevel].shape[3]), mode='bilinear',
                align_corners=False))
            tenSecond.append(torch.nn.functional.interpolate(input=
                tenSecond[-1], size=(tenFeaturesSecond[intLevel].shape[2],
                tenFeaturesSecond[intLevel].shape[3]), mode='bilinear',
                align_corners=False))
        tenFlow = None
        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenFirst[intLevel],
                tenSecond[intLevel], tenFeaturesFirst[intLevel],
                tenFeaturesSecond[intLevel], tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenFirst[intLevel],
                tenSecond[intLevel], tenFeaturesFirst[intLevel],
                tenFeaturesSecond[intLevel], tenFlow)
            tenFlow = self.netRegularization[intLevel](tenFirst[intLevel],
                tenSecond[intLevel], tenFeaturesFirst[intLevel],
                tenFeaturesSecond[intLevel], tenFlow)
        return tenFlow * 20.0


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sniklaus_pytorch_liteflownet(_paritybench_base):
    pass
