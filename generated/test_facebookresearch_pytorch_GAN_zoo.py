import sys
_module = sys.modules[__name__]
del sys
datasets = _module
eval = _module
hubconf = _module
DCGAN = _module
UTs = _module
test_ac_criterion = _module
models = _module
base_GAN = _module
attrib_dataset = _module
hd5 = _module
utils = _module
db_stats = _module
build_nn_db = _module
inception = _module
inspirational_generation = _module
laplacian_SWD = _module
metric_plot = _module
nn_metric = _module
visualization = _module
gan_visualizer = _module
GDPP_loss = _module
loss_criterions = _module
ac_criterion = _module
base_loss_criterions = _module
gradient_losses = _module
logistic_loss = _module
loss_texture = _module
metrics = _module
inception_score = _module
laplacian_swd = _module
nn_score = _module
DCGAN_nets = _module
networks = _module
constant_net = _module
custom_layers = _module
mini_batch_stddev_module = _module
progressive_conv_net = _module
styleGAN = _module
progressive_gan = _module
DCGAN_trainer = _module
trainer = _module
gan_trainer = _module
progressive_gan_trainer = _module
standard_configurations = _module
dcgan_config = _module
pgan_config = _module
stylegan_config = _module
styleGAN_trainer = _module
config = _module
image_transform = _module
product_module = _module
utils = _module
save_feature_extractor = _module
train = _module
np_visualizer = _module
visualizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.utils.model_zoo as model_zoo


import torch.optim as optim


from math import exp


from math import log


import torch


from copy import deepcopy


import torch.nn as nn


import torchvision.transforms as Transforms


from torch.utils.data import Dataset


import copy


import numpy as np


import math


import torch.nn.functional as F


from random import randint


import random


import torchvision.models as models


import scipy


import scipy.spatial


import numpy


from collections import OrderedDict


from numpy import prod


import time


import scipy.misc


import torchvision.utils as vutils


def getFeatireSize(x):
    s = x.size()
    out = 1
    for p in s[1:]:
        out *= p
    return out


class IDModule(nn.Module):

    def __init__(self):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x.view(-1, getFeatireSize(x))


class FeatureTransform(nn.Module):
    """
    Concatenation of a resize tranform and a normalization
    """

    def __init__(self, mean=None, std=None, size=224):
        super(FeatureTransform, self).__init__()
        self.size = size
        if mean is None:
            mean = [0.0, 0.0, 0.0]
        if std is None:
            std = [1.0, 1.0, 1.0]
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float).view(1, 3, 1, 1))
        if size is None:
            self.upsamplingModule = None
        else:
            self.upsamplingModule = torch.nn.Upsample((size, size), mode='bilinear')

    def forward(self, x):
        if self.upsamplingModule is not None:
            x = self.upsamplingModule(x)
        x = x - self.mean
        x = x / self.std
        return x


def extractIndexedLayers(sequence, x, indexes, detach):
    index = 0
    output = []
    indexes.sort()
    for iSeq, layer in enumerate(sequence):
        if index >= len(indexes):
            break
        x = layer(x)
        if iSeq == indexes[index]:
            if detach:
                output.append(x.view(x.size(0), x.size(1), -1).detach())
            else:
                output.append(x.view(x.size(0), x.size(1), -1))
            index += 1
    return output


def extractRelUIndexes(sequence, layers):
    layers.sort()
    index = 0
    output = []
    indexRef = 0
    indexScale = 1
    hasCaughtRelUOnLayer = False
    while indexRef < len(layers) and index < len(sequence):
        if isinstance(sequence[index], torch.nn.ReLU):
            if not hasCaughtRelUOnLayer and indexScale == layers[indexRef]:
                hasCaughtRelUOnLayer = True
                output.append(index)
                indexRef += 1
        if isinstance(sequence[index], torch.nn.MaxPool2d) or isinstance(sequence[index], torch.nn.AvgPool2d):
            hasCaughtRelUOnLayer = False
            indexScale += 1
        index += 1
    return output


def loadmodule(package, name, prefix='..'):
    """
    A dirty hack to load a module from a string input

    Args:
        package (string): package name
        name (string): module name

    Returns:
        A pointer to the loaded module
    """
    strCmd = 'from ' + prefix + package + ' import ' + name + ' as module'
    exec(strCmd)
    return eval('module')


class LossTexture(torch.nn.Module):
    """
    An implenetation of style transfer's (http://arxiv.org/abs/1703.06868) like
    loss.
    """

    def __init__(self, device, modelName, scalesOut):
        """
        Args:
            - device (torch.device): torch.device("cpu") or
                                     torch.device("cuda:0")
            - modelName (string): name of the torchvision.models model. For
                                  example vgg19
            - scalesOut (list): index of the scales to extract. In the Style
                                transfer paper it was [1,2,3,4]
        """
        super(LossTexture, self).__init__()
        scalesOut.sort()
        model = loadmodule('torchvision.models', modelName, prefix='')
        self.featuresSeq = model(pretrained=True).features
        self.indexLayers = extractRelUIndexes(self.featuresSeq, scalesOut)
        self.reductionFactor = [(1 / float(2 ** (i - 1))) for i in scalesOut]
        refMean = [(2 * p - 1) for p in [0.485, 0.456, 0.406]]
        refSTD = [(2 * p) for p in [0.229, 0.224, 0.225]]
        self.imgTransform = FeatureTransform(mean=refMean, std=refSTD, size=None)
        self.imgTransform = self.imgTransform

    def getLoss(self, fake, reals, mask=None):
        featuresReals = self.getFeatures(reals, detach=True, prepImg=True, mask=mask).mean(dim=0)
        featuresFakes = self.getFeatures(fake, detach=False, prepImg=True, mask=None).mean(dim=0)
        outLoss = ((featuresReals - featuresFakes) ** 2).mean()
        return outLoss

    def getFeatures(self, image, detach=True, prepImg=True, mask=None):
        if prepImg:
            image = self.imgTransform(image)
        fullSequence = extractIndexedLayers(self.featuresSeq, image, self.indexLayers, detach)
        outFeatures = []
        nFeatures = len(fullSequence)
        for i in range(nFeatures):
            if mask is not None:
                locMask = (1.0 + F.upsample(mask, size=(image.size(2) * self.reductionFactor[i], image.size(3) * self.reductionFactor[i]), mode='bilinear')) * 0.5
                locMask = locMask.view(locMask.size(0), locMask.size(1), -1)
                totVal = locMask.sum(dim=2)
                meanReals = (fullSequence[i] * locMask).sum(dim=2) / totVal
                varReals = (fullSequence[i] * fullSequence[i] * locMask).sum(dim=2) / totVal - meanReals * meanReals
            else:
                meanReals = fullSequence[i].mean(dim=2)
                varReals = (fullSequence[i] * fullSequence[i]).mean(dim=2) - meanReals * meanReals
            outFeatures.append(meanReals)
            outFeatures.append(varReals)
        return torch.cat(outFeatures, dim=1)

    def forward(self, x, mask=None):
        return self.getFeatures(x, detach=False, prepImg=False, mask=mask)

    def saveModel(self, pathOut):
        torch.save(dict(model=self, fullDump=True, mean=self.imgTransform.mean.view(-1).tolist(), std=self.imgTransform.std.view(-1).tolist()), pathOut)


def getLayerNormalizationFactor(x):
    """
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    """
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self, module, equalized=True, lrMul=1.0, initBiasToZero=True):
        """
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """
        super(ConstrainedLayer, self).__init__()
        self.module = module
        self.equalized = equalized
        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedLinear(ConstrainedLayer):

    def __init__(self, nChannelsPrevious, nChannels, bias=True, **kwargs):
        """
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self, nn.Linear(nChannelsPrevious, nChannels, bias=bias), **kwargs)


class AdaIN(nn.Module):

    def __init__(self, dimIn, dimOut, epsilon=1e-08):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = EqualizedLinear(dimIn, 2 * dimOut, equalized=True, initBiasToZero=True)
        self.dimOut = dimOut

    def forward(self, x, y):
        batchSize, nChannel, width, height = x.size()
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        varx = torch.clamp((tmpX * tmpX).mean(dim=2).view(batchSize, nChannel, 1, 1) - mux * mux, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - mux) * varx
        styleY = self.styleModulator(y)
        yA = styleY[:, :self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)
        return yA * x + yB


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self, nChannelsPrevious, nChannels, kernelSize, padding=0, bias=True, **kwargs):
        """
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self, nn.Conv2d(nChannelsPrevious, nChannels, kernelSize, padding=padding, bias=bias), **kwargs)


class MappingLayer(nn.Module):

    def __init__(self, dimIn, dimLatent, nLayers, leakyReluLeak=0.2):
        super(MappingLayer, self).__init__()
        self.FC = nn.ModuleList()
        inDim = dimIn
        for i in range(nLayers):
            self.FC.append(EqualizedLinear(inDim, dimLatent, lrMul=0.01, equalized=True, initBiasToZero=True))
            inDim = dimLatent
        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

    def forward(self, x):
        for layer in self.FC:
            x = self.activation(layer(x))
        return x


class NoiseMultiplier(nn.Module):

    def __init__(self):
        super(NoiseMultiplier, self).__init__()
        self.module = nn.Conv2d(1, 1, 1, bias=False)
        self.module.weight.data.fill_(0)

    def forward(self, x):
        return self.module(x)


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-08):
        return x * ((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt()


def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


class GNet(nn.Module):

    def __init__(self, dimInput=512, dimMapping=512, dimOutput=3, nMappingLayers=8, leakyReluLeak=0.2, generationActivation=None, phiTruncation=0.5, gamma_avg=0.99):
        super(GNet, self).__init__()
        self.dimMapping = dimMapping
        self.mapping = MappingLayer(dimInput, dimMapping, nMappingLayers)
        self.baseScale0 = nn.Parameter(torch.ones(1, dimMapping, 4, 4), requires_grad=True)
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        self.noiseModulators = nn.ModuleList()
        self.depthScales = [dimMapping]
        self.noramlizationLayer = NormalizationLayer()
        self.adain00 = AdaIN(dimMapping, dimMapping)
        self.noiseMod00 = NoiseMultiplier()
        self.adain01 = AdaIN(dimMapping, dimMapping)
        self.noiseMod01 = NoiseMultiplier()
        self.conv0 = EqualizedConv2d(dimMapping, dimMapping, 3, equalized=True, initBiasToZero=True, padding=1)
        self.activation = torch.nn.LeakyReLU(leakyReluLeak)
        self.alpha = 0
        self.generationActivation = generationActivation
        self.dimOutput = dimOutput
        self.phiTruncation = phiTruncation
        self.register_buffer('mean_w', torch.randn(1, dimMapping))
        self.gamma_avg = gamma_avg

    def setNewAlpha(self, alpha):
        """
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be in [0,1]')
        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0is defined")
        self.alpha = alpha

    def addScale(self, dimNewScale):
        lastDim = self.depthScales[-1]
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(lastDim, dimNewScale, 3, padding=1, equalized=True, initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.scaleLayers[-1].append(EqualizedConv2d(dimNewScale, dimNewScale, 3, padding=1, equalized=True, initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.toRGBLayers.append(EqualizedConv2d(dimNewScale, self.dimOutput, 1, equalized=True, initBiasToZero=True))
        self.noiseModulators.append(nn.ModuleList())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.depthScales.append(dimNewScale)

    def forward(self, x):
        batchSize = x.size(0)
        mapping = self.mapping(self.noramlizationLayer(x))
        if self.training:
            self.mean_w = self.gamma_avg * self.mean_w + (1 - self.gamma_avg) * mapping.mean(dim=0, keepdim=True)
        if self.phiTruncation < 1:
            mapping = self.mean_w + self.phiTruncation * (mapping - self.mean_w)
        feature = self.baseScale0.expand(batchSize, -1, 4, 4)
        feature = feature + self.noiseMod00(torch.randn((batchSize, 1, 4, 4), device=x.device))
        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)
        feature = feature + self.noiseMod01(torch.randn((batchSize, 1, 4, 4), device=x.device))
        feature = self.activation(feature)
        feature = self.adain01(feature, mapping)
        for nLayer, group in enumerate(self.scaleLayers):
            noiseMod = self.noiseModulators[nLayer]
            feature = Upscale2d(feature)
            feature = group[0](feature) + noiseMod[0](torch.randn((batchSize, 1, feature.size(2), feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[1](feature, mapping)
            feature = group[2](feature) + noiseMod[1](torch.randn((batchSize, 1, feature.size(2), feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[3](feature, mapping)
            if self.alpha > 0 and nLayer == len(self.scaleLayers) - 2:
                y = self.toRGBLayers[-2](feature)
                y = Upscale2d(y)
        feature = self.toRGBLayers[-1](feature)
        if self.alpha > 0:
            feature = self.alpha * y + (1.0 - self.alpha) * feature
        if self.generationActivation is not None:
            feature = self.generationActivation(feature)
        return feature

    def getOutputSize(self):
        side = 2 ** (2 + len(self.toRGBLayers))
        return side, side


def miniBatchStdDev(x, subGroupSize=4):
    """
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input

    Args:

        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-08)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
    return torch.cat([x, y], dim=1)


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class DNet(nn.Module):

    def __init__(self, depthScale0, initBiasToZero=True, leakyReluLeak=0.2, sizeDecisionLayer=1, miniBatchNormalization=False, dimInput=3, equalizedlR=True):
        """
        Build a discriminator for a progressive GAN model

        Args:

            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()
        self.mergeLayers = nn.ModuleList()
        self.initDecisionLayer(sizeDecisionLayer)
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1, equalized=equalizedlR, initBiasToZero=initBiasToZero))
        dimEntryScale0 = depthScale0
        if miniBatchNormalization:
            dimEntryScale0 += 1
        self.miniBatchNormalization = miniBatchNormalization
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, depthScale0, 3, padding=1, equalized=equalizedlR, initBiasToZero=initBiasToZero))
        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16, depthScale0, equalized=equalizedlR, initBiasToZero=initBiasToZero))
        self.alpha = 0
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

    def addScale(self, depthNewScale):
        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale, 3, padding=1, equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthLastScale, 3, padding=1, equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero))
        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput, depthNewScale, 1, equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        """
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be in [0,1]')
        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0is defined")
        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = EqualizedLinear(self.scalesDepth[0], sizeDecisionLayer, equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero)

    def forward(self, x, getFeature=False):
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leakyRelu(self.fromRGBLayers[-2](y))
        x = self.leakyRelu(self.fromRGBLayers[-1](x))
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2
        for groupLayer in reversed(self.scaleLayers):
            for layer in groupLayer:
                x = self.leakyRelu(layer(x))
            x = nn.AvgPool2d((2, 2))(x)
            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1 - self.alpha) * x
            shift -= 1
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)
        x = self.leakyRelu(self.groupScaleZero[0](x))
        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))
        out = self.decisionLayer(x)
        if not getFeature:
            return out
        return out, x


class ConstantNet(nn.Module):
    """A network that does nothing"""

    def __init__(self, shapeOut=None):
        super(ConstantNet, self).__init__()
        self.shapeOut = shapeOut

    def forward(self, x):
        if self.shapeOut is not None:
            x = x.view(x.size[0], self.shapeOut[0], self.shapeOut[1], self.shapeOut[2])
        return x


class MeanStd(nn.Module):

    def __init__(self):
        super(MeanStd, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        mean_x = torch.mean(x, dim=2)
        var_x = torch.mean(x ** 2, dim=2) - mean_x * mean_x
        return torch.cat([mean_x, var_x], dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaIN,
     lambda: ([], {'dimIn': 4, 'dimOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (ConstantNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DNet,
     lambda: ([], {'depthScale0': 1}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (EqualizedConv2d,
     lambda: ([], {'nChannelsPrevious': 4, 'nChannels': 4, 'kernelSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualizedLinear,
     lambda: ([], {'nChannelsPrevious': 4, 'nChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (IDModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MappingLayer,
     lambda: ([], {'dimIn': 4, 'dimLatent': 4, 'nLayers': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanStd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoiseMultiplier,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (NormalizationLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_facebookresearch_pytorch_GAN_zoo(_paritybench_base):
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

