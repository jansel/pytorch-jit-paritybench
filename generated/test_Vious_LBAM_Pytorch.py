import sys
_module = sys.modules[__name__]
del sys
basicFunction = _module
dataloader = _module
generateMask = _module
InpaintingLoss = _module
ActivationFunction = _module
LBAMModel = _module
discriminator = _module
forwardAttentionLayer = _module
reverseAttentionLayer = _module
weightInitial = _module
pytorch_ssim = _module
test = _module
test_random_batch = _module
train = _module

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


import torch


from torch.utils.data import Dataset


from random import randint


import torch.nn as nn


import numpy as np


from torchvision import transforms


import random


import inspect


import re


import math


import collections


from torchvision.utils import save_image


from torch import nn


from torch import autograd


from torch.nn.parameter import Parameter


from torchvision import models


from math import exp


import torch.nn.functional as F


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


from torchvision import datasets


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Resize


from torchvision.transforms import ToPILImage


from torch.utils.data import DataLoader


import time


import torch.optim as optim


from torchvision import utils


class DiscriminatorDoubleColumn(nn.Module):

    def __init__(self, inputChannels):
        super(DiscriminatorDoubleColumn, self).__init__()
        self.globalConv = nn.Sequential(nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.localConv = nn.Sequential(nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.fusionLayer = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=4), nn.Sigmoid())

    def forward(self, batches, masks):
        globalFt = self.globalConv(batches * masks)
        localFt = self.localConv(batches * (1 - masks))
        concatFt = torch.cat((globalFt, localFt), 1)
        return self.fusionLayer(concatFt).view(batches.size()[0], -1)


def calc_gradient_penalty(netD, real_data, fake_data, masks, cuda, Lambda):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    if cuda:
        alpha = alpha
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
    if cuda:
        interpolates = interpolates
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates, masks)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()) if cuda else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()


def gram_matrix(feat):
    b, ch, h, w = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


class InpaintingLossWithGAN(nn.Module):

    def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(InpaintingLossWithGAN, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = DiscriminatorDoubleColumn(3)
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        """ if (self.numOfGPUs > 1):
            self.discriminator = self.discriminator.cuda()
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=range(self.numOfGPUs)) """
        self.lamda = Lamda
        self.writer = SummaryWriter(logPath)

    def forward(self, input, mask, output, gt, count, epoch):
        self.discriminator.zero_grad()
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        gp = calc_gradient_penalty(self.discriminator, gt, output, mask, self.cudaAvailable, self.lamda)
        D_loss = D_fake - D_real + gp
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        self.writer.add_scalar('LossD/Discrinimator loss', D_loss.item(), count)
        output_comp = mask * input + (1 - mask) * output
        holeLoss = 6 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)
        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('only gray an')
        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])
        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))
        """ if self.numOfGPUs > 1:
            holeLoss = holeLoss.sum() / self.numOfGPUs
            validAreaLoss = validAreaLoss.sum() / self.numOfGPUs
            prcLoss = prcLoss.sum() / self.numOfGPUs
            styleLoss = styleLoss.sum() / self.numOfGPUs """
        self.writer.add_scalar('LossG/Hole loss', holeLoss.item(), count)
        self.writer.add_scalar('LossG/Valid loss', validAreaLoss.item(), count)
        self.writer.add_scalar('LossPrc/Perceptual loss', prcLoss.item(), count)
        self.writer.add_scalar('LossStyle/style loss', styleLoss.item(), count)
        GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake
        self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()


class GaussActivation(nn.Module):

    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()
        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    def forward(self, inputFeatures):
        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)
        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu
        leftValuesActiv = self.a * torch.exp(-self.sigma1 * (inputFeatures - self.mu) ** 2)
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)
        rightValueActiv = 1 + (self.a - 1) * torch.exp(-self.sigma2 * (inputFeatures - self.mu) ** 2)
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)
        output = leftValuesActiv + rightValueActiv
        return output


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.updateFunc = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """
        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class ForwardAttentionLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels, kernelSize, stride, padding, dilation=1, groups=1, bias=False):
        super(ForwardAttentionLayer, self).__init__()
        self.conv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        if inputChannels == 4:
            self.maskConv = nn.Conv2d(3, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        else:
            self.maskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init())
        self.maskConv.apply(weights_init())
        self.activationFuncG_A = GaussActivation(1.1, 2.0, 1.0, 1.0)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputFeatures, inputMasks):
        convFeatures = self.conv(inputFeatures)
        maskFeatures = self.maskConv(inputMasks)
        maskActiv = self.activationFuncG_A(maskFeatures)
        convOut = convFeatures * maskActiv
        maskUpdate = self.updateMask(maskFeatures)
        return convOut, maskUpdate, convFeatures, maskActiv


class ForwardAttention(nn.Module):

    def __init__(self, inputChannels, outputChannels, bn=False, sample='down-4', activ='leaky', convBias=False):
        super(ForwardAttention, self).__init__()
        if sample == 'down-4':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 4, 2, 1, bias=convBias)
        elif sample == 'down-5':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 5, 2, 2, bias=convBias)
        elif sample == 'down-7':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 7, 2, 3, bias=convBias)
        elif sample == 'down-3':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 2, 1, bias=convBias)
        else:
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 1, 1, bias=convBias)
        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, inputFeatures, inputMasks):
        features, maskUpdated, convPreF, maskActiv = self.conv(inputFeatures, inputMasks)
        if hasattr(self, 'bn'):
            features = self.bn(features)
        if hasattr(self, 'activ'):
            features = self.activ(features)
        return features, maskUpdated, convPreF, maskActiv


class ReverseAttention(nn.Module):

    def __init__(self, inputChannels, outputChannels, bn=False, activ='leaky', kernelSize=4, stride=2, padding=1, outPadding=0, dilation=1, groups=1, convBias=False, bnChannels=512):
        super(ReverseAttention, self).__init__()
        self.conv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=kernelSize, stride=stride, padding=padding, output_padding=outPadding, dilation=dilation, groups=groups, bias=convBias)
        self.conv.apply(weights_init())
        if bn:
            self.bn = nn.BatchNorm2d(bnChannels)
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, ecFeaturesSkip, dcFeatures, maskFeaturesForAttention):
        nextDcFeatures = self.conv(dcFeatures)
        concatFeatures = torch.cat((ecFeaturesSkip, nextDcFeatures), 1)
        outputFeatures = concatFeatures * maskFeaturesForAttention
        if hasattr(self, 'bn'):
            outputFeatures = self.bn(outputFeatures)
        if hasattr(self, 'activ'):
            outputFeatures = self.activ(outputFeatures)
        return outputFeatures


class ReverseMaskConv(nn.Module):

    def __init__(self, inputChannels, outputChannels, kernelSize=4, stride=2, padding=1, dilation=1, groups=1, convBias=False):
        super(ReverseMaskConv, self).__init__()
        self.reverseMaskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, groups, bias=convBias)
        self.reverseMaskConv.apply(weights_init())
        self.activationFuncG_A = GaussActivation(1.1, 1.0, 0.5, 0.5)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputMasks):
        maskFeatures = self.reverseMaskConv(inputMasks)
        maskActiv = self.activationFuncG_A(maskFeatures)
        maskUpdate = self.updateMask(maskFeatures)
        return maskActiv, maskUpdate


class LBAMModel(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        super(LBAMModel, self).__init__()
        self.ec1 = ForwardAttention(inputChannels, 64, bn=False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)
        for i in range(5, 8):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))
        self.reverseConv1 = ReverseMaskConv(3, 64)
        self.reverseConv2 = ReverseMaskConv(64, 128)
        self.reverseConv3 = ReverseMaskConv(128, 256)
        self.reverseConv4 = ReverseMaskConv(256, 512)
        self.reverseConv5 = ReverseMaskConv(512, 512)
        self.reverseConv6 = ReverseMaskConv(512, 512)
        self.dc1 = ReverseAttention(512, 512, bnChannels=1024)
        self.dc2 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512 * 2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256 * 2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128 * 2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, inputImgs, masks):
        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, masks)
        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef7, _, _, _ = self.ec7(ef6, mu6)
        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap6, _ = self.reverseConv6(revMu5)
        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef7, concatMap6)
        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)
        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)
        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)
        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)
        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)
        dcFeatures7 = self.dc7(dcFeatures6)
        output = (self.tanh(dcFeatures7) + 1) / 2
        return output


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DiscriminatorDoubleColumn,
     lambda: ([], {'inputChannels': 4}),
     lambda: ([torch.rand([4, 4, 256, 256]), torch.rand([4, 4, 256, 256])], {}),
     True),
    (GaussActivation,
     lambda: ([], {'a': 4, 'mu': 4, 'sigma1': 4, 'sigma2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskUpdate,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReverseMaskConv,
     lambda: ([], {'inputChannels': 4, 'outputChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Vious_LBAM_Pytorch(_paritybench_base):
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

