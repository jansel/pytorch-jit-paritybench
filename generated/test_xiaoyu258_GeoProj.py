import sys
_module = sys.modules[__name__]
del sys
dataset_generate = _module
distortion_model = _module
dataloaderNetM = _module
dataloaderNetS = _module
eval = _module
logger = _module
modelNetM = _module
modelNetS = _module
resampling = _module
trainNetM = _module
trainNetS = _module

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


from torch.utils import data


from torchvision import transforms


import scipy.io as spio


import numpy as np


import torch


from torch.autograd import Variable


import torch.nn as nn


import scipy.io as scio


import math


import torch.nn.functional as F


class plainEncoderBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class plainDecoderBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(plainDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class resEncoderBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride), nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class resDecoderBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(resDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        self.downsample = None
        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)
            self.downsample = nn.Sequential(nn.ConvTranspose2d(inChannel, outChannel, kernel_size=1, stride=2, output_padding=1), nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class BasicEncoderBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(BasicEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride), nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class BasicEncoderPlainBlock(nn.Module):

    def __init__(self, inChannel, outChannel, stride):
        super(BasicEncoderPlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderNet(nn.Module):

    def __init__(self, layers):
        super(EncoderNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.en_layer1 = self.make_encoder_layer(BasicEncoderPlainBlock, 64, 64, layers[0], stride=1)
        self.en_layer2 = self.make_encoder_layer(BasicEncoderBlock, 64, 128, layers[1], stride=2)
        self.en_layer3 = self.make_encoder_layer(BasicEncoderBlock, 128, 256, layers[2], stride=2)
        self.en_layer4 = self.make_encoder_layer(BasicEncoderBlock, 256, 512, layers[3], stride=2)
        self.en_layer5 = self.make_encoder_layer(BasicEncoderBlock, 512, 512, layers[4], stride=2)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def make_encoder_layer(self, block, inChannel, outChannel, block_num, stride):
        layers = []
        layers.append(block(inChannel, outChannel, stride=stride))
        for i in range(1, block_num):
            layers.append(block(outChannel, outChannel, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.en_layer1(x)
        x = self.en_layer2(x)
        x = self.en_layer3(x)
        x = self.en_layer4(x)
        x = self.en_layer5(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        para = self.fc(x)
        return para


class DecoderNet(nn.Module):

    def __init__(self, layers):
        super(DecoderNet, self).__init__()
        self.de_layer5 = self.make_decoder_layer(resDecoderBlock, 512, 512, layers[4], stride=2)
        self.de_layer4 = self.make_decoder_layer(resDecoderBlock, 512, 256, layers[3], stride=2)
        self.de_layer3 = self.make_decoder_layer(resDecoderBlock, 256, 128, layers[2], stride=2)
        self.de_layer2 = self.make_decoder_layer(resDecoderBlock, 128, 64, layers[1], stride=2)
        self.de_layer1 = self.make_decoder_layer(plainDecoderBlock, 64, 64, layers[0], stride=1)
        self.conv_end = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_decoder_layer(self, block, inChannel, outChannel, block_num, stride):
        layers = []
        for i in range(0, block_num - 1):
            layers.append(block(inChannel, inChannel, stride=1))
        layers.append(block(inChannel, outChannel, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.de_layer5(x)
        x = self.de_layer4(x)
        x = self.de_layer3(x)
        x = self.de_layer2(x)
        x = self.de_layer1(x)
        x = self.conv_end(x)
        return x


class ClassNet(nn.Module):

    def __init__(self):
        super(ClassNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 4 * 4, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EPELoss(nn.Module):

    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue


H = 256


W = 256


batchSize = 32


x0 = W / 2


y0 = H / 2


class GenerateLenFlow(torch.autograd.Function):

    @staticmethod
    def forward(self, Input):
        self.save_for_backward(Input)
        Input = Input
        flow = torch.Tensor(batchSize, 2, H, W)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            coeff = 1 + Input[s] * -1e-05 * ((i - x0) ** 2 + (j - y0) ** 2)
            flow[s, 0] = (i - x0) / coeff + x0 - i
            flow[s, 1] = (j - y0) / coeff + y0 - j
        return flow

    @staticmethod
    def backward(self, grad_output):
        Input, = self.saved_tensors
        Input = Input
        grad_output = grad_output
        grad_input = Variable(torch.ones(batchSize, 1), requires_grad=False)
        grad_current = Variable(torch.ones(batchSize, 2, H, W), requires_grad=False)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            r2 = (i - x0) ** 2 + (j - y0) ** 2
            temp = (1 + r2 * Input[s] * -1e-05) ** 2
            grad_current[s, 0] = (i - x0) * -1 * r2 * -1e-05 / temp
            grad_current[s, 1] = (j - y0) * -1 * r2 * -1e-05 / temp
        grad = grad_output * grad_current
        for s in range(batchSize):
            grad_input[s, 0] = torch.sum(grad[s, :, :, :])
        return grad_input


ProFactor = 0.1


class GenerateProFlow(torch.autograd.Function):

    @staticmethod
    def forward(self, Input):
        self.save_for_backward(Input)
        Input = Input
        flow = torch.Tensor(batchSize, 2, H, W)
        x4 = Input * ProFactor
        a31 = 0
        a32 = 2 * x4 / (1 - 2 * x4)
        a11 = 1
        a12 = x4 / (1 - 2 * x4)
        a13 = 0
        a21 = 0
        a22 = 0.99 + a32 * 0.995
        a23 = 0.005
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            im = i / (W - 1.0)
            jm = j / (H - 1.0)
            flow[s, 0] = (W - 1.0) * (a11 * im + a12[s] * jm + a13) / (a31 * im + a32[s] * jm + 1) - i
            flow[s, 1] = (H - 1.0) * (a21 * im + a22[s] * jm + a23) / (a31 * im + a32[s] * jm + 1) - j
        return flow

    @staticmethod
    def backward(self, grad_output):
        Input, = self.saved_tensors
        Input = Input
        grad_output = grad_output
        grad_input = Variable(torch.ones(batchSize, 1), requires_grad=False)
        grad_current = Variable(torch.ones(batchSize, 2, H, W), requires_grad=False)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            x = Input[s] * ProFactor
            t00 = -1.0 * ((2 * H - 2) * i + (1 - H) * W + H - 1) * j
            t01 = (4 * j * j + (8 - 8 * H) * j + 4 * H * H - 8 * H + 4) * x * x + ((4 * H - 4) * j - 4 * H * H + 8 * H - 4) * x + H * H - 2 * H + 1
            grad_current[s, 0] = ProFactor * t00 / t01
            t10 = -1.0 * ((99 * H - 99) * j * j + (-99 * H * H + 198 * H - 99) * j)
            t11 = (200 * j * j + (400 - 400 * H) * j + 200 * H * H - 400 * H + 200) * x * x + ((200 * H - 200) * j - 200 * H * H + 400 * H - 200) * x + 50 * H * H - 100 * H + 50
            grad_current[s, 1] = ProFactor * t10 / t11
        grad = grad_output * grad_current
        for s in range(batchSize):
            grad_input[s, 0] = torch.sum(grad[s, :, :, :])
        return grad_input


class GenerateRotFlow(torch.autograd.Function):

    @staticmethod
    def forward(self, Input):
        self.save_for_backward(Input)
        Input = Input
        flow = torch.Tensor(batchSize, 2, H, W)
        sina = torch.sin(Input)
        cosa = torch.cos(Input)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            flow[s, 0] = cosa[s] * i + sina[s] * j + (1 - sina[s] - cosa[s]) * W / 2 - i
            flow[s, 1] = -sina[s] * i + cosa[s] * j + (1 + sina[s] - cosa[s]) * H / 2 - j
        return flow

    @staticmethod
    def backward(self, grad_output):
        Input, = self.saved_tensors
        Input = Input
        grad_output = grad_output
        grad_input = Variable(torch.ones(batchSize, 1), requires_grad=False)
        grad_current = Variable(torch.ones(batchSize, 2, H, W), requires_grad=False)
        sina = torch.sin(Input)
        cosa = torch.cos(Input)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            grad_current[s, 0] = -sina[s] * i + cosa[s] * j + (sina[s] - cosa[s]) * W / 2
            grad_current[s, 1] = -cosa[s] * i - sina[s] * j + (cosa[s] + sina[s]) * H / 2
        grad = grad_output * grad_current
        for s in range(batchSize):
            grad_input[s, 0] = torch.sum(grad[s, :, :, :])
        return grad_input


SheFactor = 1.0 / 5


class GenerateSheFlow(torch.autograd.Function):

    @staticmethod
    def forward(self, Input):
        self.save_for_backward(Input)
        Input = Input
        flow = torch.Tensor(batchSize, 2, H, W)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            flow[s, 0] = (Input[s] * j - Input[s] * W / 2.0) * SheFactor
            flow[s, 1] = 0
        return flow

    @staticmethod
    def backward(self, grad_output):
        Input, = self.saved_tensors
        Input = Input
        grad_output = grad_output
        grad_input = Variable(torch.ones(batchSize, 1), requires_grad=False)
        grad_current = Variable(torch.ones(batchSize, 2, H, W), requires_grad=False)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            grad_current[s, 0] = (j - W / 2.0) * SheFactor
            grad_current[s, 1] = 0
        grad = grad_output * grad_current
        for s in range(batchSize):
            grad_input[s, 0] = torch.sum(grad[s, :, :, :])
        return grad_input


class GenerateWavFlow(torch.autograd.Function):

    @staticmethod
    def forward(self, Input):
        self.save_for_backward(Input)
        Input = Input
        flow = torch.Tensor(batchSize, 2, H, W)
        temp = torch.ones(batchSize, 1)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            flow[s, 0] = Input[s] * torch.sin(math.pi * 4 * j / (W * 2))
            flow[s, 1] = 0
        return flow

    @staticmethod
    def backward(self, grad_output):
        Input, = self.saved_tensors
        Input = Input
        grad_output = grad_output
        grad_input = Variable(torch.ones(batchSize, 1), requires_grad=False)
        grad_current = Variable(torch.ones(batchSize, 2, H, W), requires_grad=False)
        temp = torch.ones(batchSize, 1)
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='xy')
        i = torch.from_numpy(i).float()
        j = torch.from_numpy(j).float()
        for s in range(batchSize):
            grad_current[s, 0] = torch.sin(math.pi * 4 * j / (W * 2))
            grad_current[s, 1] = 0
        grad = grad_output * grad_current
        for s in range(batchSize):
            grad_input[s, 0] = torch.sum(grad[s, :, :, :])
        return grad_input


class ModelNet(nn.Module):

    def __init__(self, types):
        super(ModelNet, self).__init__()
        self.types = types

    def forward(self, x):
        para = x
        if self.types == 'barrel' or self.types == 'pincushion':
            OBJflow = GenerateLenFlow.apply
            flow = OBJflow(para)
        elif self.types == 'rotation':
            OBJflow = GenerateRotFlow.apply
            flow = OBJflow(para)
        elif self.types == 'shear':
            OBJflow = GenerateSheFlow.apply
            flow = OBJflow(para)
        elif self.types == 'projective':
            OBJflow = GenerateProFlow.apply
            flow = OBJflow(para)
        elif self.types == 'wave':
            OBJflow = GenerateWavFlow.apply
            flow = OBJflow(para)
        return flow


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicEncoderBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicEncoderPlainBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderNet,
     lambda: ([], {'layers': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (EPELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (plainDecoderBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (plainEncoderBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resDecoderBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resEncoderBlock,
     lambda: ([], {'inChannel': 4, 'outChannel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xiaoyu258_GeoProj(_paritybench_base):
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

