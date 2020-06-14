import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
run = _module
sepconv = _module

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


arguments_strModel = 'lf'


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=intInput,
                out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=
                intOutput, out_channels=intOutput, kernel_size=3, stride=1,
                padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(
                in_channels=intOutput, out_channels=intOutput, kernel_size=
                3, stride=1, padding=1), torch.nn.ReLU(inplace=False))

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,
                mode='bilinear', align_corners=True), torch.nn.Conv2d(
                in_channels=intOutput, out_channels=intOutput, kernel_size=
                3, stride=1, padding=1), torch.nn.ReLU(inplace=False))

        def Subnet():
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,
                out_channels=64, kernel_size=3, stride=1, padding=1), torch
                .nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=64,
                out_channels=64, kernel_size=3, stride=1, padding=1), torch
                .nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=64,
                out_channels=51, kernel_size=3, stride=1, padding=1), torch
                .nn.ReLU(inplace=False), torch.nn.Upsample(scale_factor=2,
                mode='bilinear', align_corners=True), torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1,
                padding=1))
        self.netConv1 = Basic(6, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)
        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)
        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)
        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for
            strKey, tenWeight in torch.load(__file__.replace('run.py', 
            'network-' + arguments_strModel + '.pytorch')).items()})

    def forward(self, tenFirst, tenSecond):
        tenConv1 = self.netConv1(torch.cat([tenFirst, tenSecond], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=
            tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=
            tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=
            tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=
            tenConv4, kernel_size=2, stride=2, count_include_pad=False))
        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.
            avg_pool2d(input=tenConv5, kernel_size=2, stride=2,
            count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))
        tenCombine = tenDeconv2 + tenConv2
        tenFirst = torch.nn.functional.pad(input=tenFirst, pad=[int(math.
            floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 /
            2.0)), int(math.floor(51 / 2.0))], mode='replicate')
        tenSecond = torch.nn.functional.pad(input=tenSecond, pad=[int(math.
            floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 /
            2.0)), int(math.floor(51 / 2.0))], mode='replicate')
        tenDot1 = sepconv.FunctionSepconv(tenInput=tenFirst, tenVertical=
            self.netVertical1(tenCombine), tenHorizontal=self.
            netHorizontal1(tenCombine))
        tenDot2 = sepconv.FunctionSepconv(tenInput=tenSecond, tenVertical=
            self.netVertical2(tenCombine), tenHorizontal=self.
            netHorizontal2(tenCombine))
        return tenDot1 + tenDot2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sniklaus_sepconv_slomo(_paritybench_base):
    pass
