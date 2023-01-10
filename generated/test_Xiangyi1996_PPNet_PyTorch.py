import sys
_module = sys.modules[__name__]
del sys
config = _module
dataloaders = _module
common = _module
customized = _module
pascal = _module
transforms = _module
AlignLossFunctions = _module
Aspp = _module
FewShotSegPartResnet = _module
FewShotSegPartResnetSem = _module
FewShotSegResnet = _module
ResNetBackbone = _module
SemiFewShotPartGraph = _module
models = _module
getSpFeature = _module
ops = _module
train_fewshot = _module
train_graph = _module
train_part_sem = _module
util = _module
kmeans = _module
metric = _module
sbd_instance_process = _module
superpixel = _module
util = _module
utils = _module
voc_classwise_filenames = _module

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


from torch.utils.data import Dataset


import torch


import numpy as np


import torch.nn.functional as F


import matplotlib.pyplot as plt


from scipy import ndimage


import torchvision.transforms.functional as tr_F


import torch.nn as nn


import torch.nn.init as init


from torch.autograd import Variable


from collections import OrderedDict


import torchvision


import torch.optim


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import MultiStepLR


import torch.backends.cudnn as cudnn


from torchvision.transforms import Compose


import time


import matplotlib


import copy


import logging


from logging.config import dictConfig


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, atrous_rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module('c{}'.format(i), nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True))
        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class KmeansClustering:

    def __init__(self, num_cnt, iters, init='kmeans++'):
        super(KmeansClustering, self).__init__()
        self.num_cnt = num_cnt
        self.iters = iters
        self.init_mode = init

    def InitFunc(self, x):
        N, D = x.shape
        if self.init_mode == 'random':
            random_inds = np.random.choice(N, self.num_cnt)
            init_stat = x[random_inds]
        elif self.init_mode == 'kmeans++':
            random_start = np.random.choice(N, 1)
            init_stat = x[[random_start]]
            x1 = x.unsqueeze(1)
            x_lazy = LazyTensor(x1)
            for c_id in range(self.num_cnt):
                init_lazy = LazyTensor(init_stat.unsqueeze(0))
                dist = ((x_lazy - init_lazy) ** 2).sum(-1)
                select = dist.min(1).view(-1).argmax(dim=0)
                init_stat = torch.cat((init_stat, x1[select]), dim=0)
        else:
            raise NotImplementedError
        return init_stat

    def cluster(self, x, center=None):
        if center is None:
            center = self.InitFunc(x)
        with torch.no_grad():
            x_lazy = LazyTensor(x.unsqueeze(1))
            cl = None
            for iter in range(self.iters):
                c_lazy = LazyTensor(center.unsqueeze(0))
                dist = ((x_lazy - c_lazy) ** 2).sum(-1)
                cl = dist.argmin(dim=1).view(-1)
                if iter < self.iters - 1:
                    for cnt_id in range(self.num_cnt):
                        selected = torch.nonzero(cl == cnt_id).squeeze(-1)
                        if selected.shape[0] != 0:
                            selected = torch.index_select(x, 0, selected)
                            center[cnt_id] = selected.mean(dim=0)
        center_new = torch.zeros_like(center)
        for cnt_id in range(self.num_cnt):
            selected = torch.nonzero(cl == cnt_id).squeeze(-1)
            if selected.shape[0] != 0:
                selected = torch.index_select(x, 0, selected)
                center_new[cnt_id] = selected.mean(dim=0)
        return center_new


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, NL=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.nolinear = NL

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.nolinear:
            out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetSemShare4(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetSemShare4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, 2, 2]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], lastRelu=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_fewshot = self.layer4(x)
        feat_semantic = F.relu(feat_fewshot)
        return feat_fewshot, feat_semantic

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, lastRelu=True, sem=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for b_id in range(1, blocks):
            NL = True
            if not lastRelu and b_id == blocks - 1:
                NL = False
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, NL=NL))
        return nn.Sequential(*layers)


def resnet50Sem(cfg=None, pretrained_path=None, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if cfg['resnet'] == 101:
        model = ResNetSemShare4(Bottleneck, [3, 4, 23, 3])
        None
    else:
        model = ResNetSemShare4(Bottleneck, [3, 4, 6, 3])
        None
    if cfg['resnet'] == 101:
        init_path = f'./FewShotSeg-dataset/cache/resnet101-5d3b4d8f.pth'
    else:
        init_path = './FewShotSeg-dataset/cache/resnet50-19c8e357.pth'
    if pretrained_path is not None:
        init_path = f"{cfg['ckpt_dir']}/best.pth"
    None
    pretrained_weight = torch.load(init_path, map_location='cpu')
    model_weight = model.state_dict()
    for key, weight in model_weight.items():
        if key in pretrained_weight:
            model_weight[key] = pretrained_weight[key]
        if key[:3] == 'sem' and key[3:] in pretrained_weight:
            model_weight[key] = pretrained_weight[key[3:]]
    if pretrained_path is not None:
        None
        for key, weight in model_weight.items():
            if 'module.encoder.' + key in pretrained_weight:
                model_weight[key] = pretrained_weight['module.encoder.' + key]
            if key[:3] == 'sem' and key[3:] in pretrained_weight:
                model_weight[key] = pretrained_weight[key[3:]]
    model.load_state_dict(model_weight)
    return model


class FewShotSegPart(nn.Module):
    """
    Fewshot Segmentation model
    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.GLOBAL_CONST = 0.5
        self.config = cfg
        self.encoder = resnet50Sem(cfg=cfg)
        self.aspp = _ASPP(2048, 16, [6, 12, 18, 24])
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=10, init='random')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        img_fts, feat_semantic = self.encoder(imgs_concat)
        if self.training:
            output_semantic = self.aspp(feat_semantic)
            output_semantic = F.interpolate(output_semantic, size=img_size, mode='bilinear', align_corners=True)
        else:
            output_semantic = None
        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        align_loss = torch.zeros(1)
        outputs = []
        for epi in range(batch_size):
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1) for shot in range(n_shots)] for way in range(n_ways)]
            fg_prototypes, bg_prototype = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)
            prototypes = [bg_prototype] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))
            if self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        return output, output_semantic, align_loss / batch_size

    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler
        return dist

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-05)
        return masked_fts

    def getFeaturesArray(self, fts, mask, upscale=2):
        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:]
        h, w = mask.shape[1:]
        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-05)
        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1 * upscale, w1 * upscale), mode='nearest').view(-1)
        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]] * 0
            else:
                fts = fts[mask1 > 0]
        else:
            fts = F.interpolate(fts, size=(h1 * upscale, w1 * upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1 * w1 * upscale ** 2, c)
            fts = fts[mask_bilinear > 0]
        return fts, masked_fts

    def kmeansPrototype(self, fg_fts, bg_fts):
        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts]
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc)
        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)
        fg_propotypes = [(fg_c + self.GLOBAL_CONST * fg_g) for fg_c, fg_g in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.GLOBAL_CONST * bg_prop_glo
        return fg_propotypes, bg_propotypes

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [(sum(way) / n_shots) for way in fg_fts]
        bg_prototype = sum([(sum(way) / n_shots) for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots, h, w = fore_mask.shape
        pred_mask = pred.argmax(dim=1, keepdim=True)
        binary_masks = [(pred_mask == i) for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.cat(binary_masks, dim=1).float()
        qry_prototypes = self.getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))
        loss = torch.zeros(1)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                loss = loss + F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss

    def getAlignProto(self, qry_fts, pred_mask, skip_ways, image_size):
        """
        qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
        pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H' x W'
        """
        pred_mask_global = pred_mask.unsqueeze(2)
        qry_prototypes_global = torch.sum(qry_fts.unsqueeze(1) * pred_mask_global, dim=(0, 3, 4))
        qry_prototypes_global = qry_prototypes_global / (pred_mask_global.sum((0, 3, 4)) + 1e-05)
        n, c, h, w = qry_fts.shape
        qry_fts_s4 = F.interpolate(input=qry_fts, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
        qry_fts_s0 = F.interpolate(input=qry_fts, size=image_size, mode='bilinear', align_corners=True)
        qry_fts_s4_reshape = qry_fts_s4.permute(0, 2, 3, 1).view(-1, c).contiguous()
        qry_fts_s0_reshape = qry_fts_s0.permute(0, 2, 3, 1).view(-1, c).contiguous()
        pred_mask_s4 = F.interpolate(input=pred_mask, size=(h * 2, w * 2), mode='nearest')
        pred_mask_s0 = F.interpolate(input=pred_mask, size=image_size, mode='nearest')
        num_background = pred_mask_s4[:, 0].sum()
        if num_background == 0:
            bg_prototypes = qry_fts_s4_reshape[[0]] * 0
        else:
            if num_background <= 10:
                bg_pred = qry_fts_s0_reshape[pred_mask_s0[:, 0].view(-1) > 0]
            else:
                bg_pred = qry_fts_s4_reshape[pred_mask_s4[:, 0].view(-1) > 0]
            bg_prototypes = self.kmeans.cluster(bg_pred)
        bg_prototypes += qry_prototypes_global[[0]] * self.GLOBAL_CONST
        pred_mask_s4_ways = pred_mask_s4[:, 1:].permute(1, 0, 2, 3)
        pred_mask_s0_ways = pred_mask_s0[:, 1:].permute(1, 0, 2, 3)
        fg_prototypes = []
        for way_id in range(pred_mask_s0_ways.shape[0]):
            if way_id in skip_ways:
                fg_prototypes.append(None)
                continue
            pred_mask_w_s4, pred_mask_w_s0 = pred_mask_s4_ways[way_id], pred_mask_s0_ways[way_id]
            pred_mask_w_s4 = pred_mask_w_s4.view(-1).contiguous()
            pred_mask_w_s0 = pred_mask_w_s0.view(-1).contiguous()
            num_pos = pred_mask_w_s4.sum()
            if num_pos <= 10:
                qry_fts_w = qry_fts_s0_reshape[pred_mask_w_s0 > 0]
            else:
                qry_fts_w = qry_fts_s4_reshape[pred_mask_w_s4 > 0]
            fg_pro = self.kmeans.cluster(qry_fts_w)
            fg_pro += qry_prototypes_global[[way_id + 1]] * self.GLOBAL_CONST
            fg_prototypes.append(fg_pro)
        prototypes = [bg_prototypes] + fg_prototypes
        return prototypes


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(cfg=cfg))]))
        self.device = torch.device('cuda')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        align_loss = torch.zeros(1)
        outputs = []
        for epi in range(batch_size):
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            prototypes = [bg_prototype] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            if self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        output_semantic = torch.zeros(1)
        return output, output_semantic, align_loss / batch_size

    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-05)
        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [(sum(way) / n_shots) for way in fg_fts]
        bg_prototype = sum([(sum(way) / n_shots) for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        pred_mask = pred.argmax(dim=1, keepdim=True)
        binary_masks = [(pred_mask == i) for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-05)
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                loss = loss + F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, 2, 2]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], lastRelu=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, lastRelu=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for b_id in range(1, blocks):
            NL = True
            if not lastRelu and b_id == blocks - 1:
                NL = False
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, NL=NL))
        return nn.Sequential(*layers)


def numerical_stability_softmax(score, dim):
    max_score, _ = score.max(dim, keepdim=True)
    stable_score = score - max_score
    stable_exp = torch.exp(stable_score)
    stable_prob = stable_exp / stable_exp.sum(dim, keepdim=True)
    return stable_prob


class GraphTransformer(nn.Module):

    def __init__(self, in_channels, out_channels, scale=0.2):
        super(GraphTransformer, self).__init__()
        self.out_channels = out_channels
        Linear = nn.Linear
        self.inner_w1 = Linear(in_channels, out_channels, bias=False)
        self.inner_trans = Linear(in_channels, in_channels, bias=False)
        self.inter_w1 = Linear(in_channels, out_channels, bias=False)
        self.inter_w2 = Linear(in_channels, out_channels, bias=False)

    def forward(self, topk_feats, prototypes):
        """
        topk_feats: [[],[],[],]  ## num_class, num_unlabel
        prototypes: [5*2048, 5*2048, 5*2048] ## num_class
        :return:
        """
        un_prototypes = []
        for feats, protos in zip(topk_feats, prototypes):
            feats_all = []
            for feat in feats:
                if feat is not None:
                    if feat.shape[0] == 1:
                        feats_all.append(feat)
                    else:
                        feat_embed_1 = self.inner_w1(feat)
                        atte = torch.mm(feat_embed_1, feat_embed_1.permute(1, 0)) / self.out_channels ** 0.5
                        atte = numerical_stability_softmax(atte, dim=1)
                        feat = feat + F.relu(self.inner_trans(torch.mm(atte, feat)))
                        feats_all.append(feat)
            if len(feats_all) != 0:
                feats_all = torch.cat(feats_all, dim=0)
                feats_all_embed = self.inter_w1(feats_all)
                protos_embed = self.inter_w2(protos)
                atte = torch.mm(protos_embed, feats_all_embed.permute(1, 0)) / self.out_channels ** 0.5
                atte = numerical_stability_softmax(atte, 1)
                un_protos = torch.mm(atte, feats_all)
                un_prototypes.append(un_protos)
            else:
                fake_un_protos = torch.zeros_like(protos)
                un_prototypes.append(fake_un_protos)
        return un_prototypes


def calDistMax(fts, prototype, scaler=20):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: b*512*53*53
        prototype: prototype of one semantic class
            expect shape: 5*512
        return: b*53*53
    """
    dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler
    return dist


def alignLoss(qry_fts, pred, supp_fts, fore_mask, back_mask, getAlignProto):
    """
    Compute the loss for the prototype alignment branch

    Args:
        qry_fts: embedding features for query images
            expect shape: N x C x H' x W'
        pred: predicted segmentation score
            expect shape: N x (1 + Wa) x H x W
        supp_fts: embedding features for support images
            expect shape: Wa x Sh x C x H' x W'
        fore_mask: foreground masks for support images
            expect shape: way x shot x H x W
        back_mask: background masks for support images
            expect shape: way x shot x H x W
    """
    n_ways, n_shots, h, w = fore_mask.shape
    pred_mask = pred.argmax(dim=1, keepdim=True)
    binary_masks = [(pred_mask == i) for i in range(1 + n_ways)]
    skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
    pred_mask = torch.cat(binary_masks, dim=1).float()
    qry_prototypes = getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))
    loss = 0
    for way in range(n_ways):
        if way in skip_ways:
            continue
        prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
        img_fts = supp_fts[way]
        supp_dist = [calDistMax(img_fts, prototype) for prototype in prototypes]
        supp_pred = torch.stack(supp_dist, dim=1)
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
        supp_label = torch.full_like(fore_mask[way], 255, device=img_fts.device).long()
        supp_label[fore_mask[way] == 1] = 1
        supp_label[back_mask[way] == 1] = 0
        loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_shots / n_ways
    return loss


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def getSpFeats(un_feats, un_segments):
    """
    un_feats: N*2048*53*53
    un_segments: N*1*417*417
    """
    n, c, h, w = un_feats.shape
    un_segments = F.interpolate(un_segments, (h, w), mode='nearest')
    s_feats = []
    for nid in range(n):
        un_seg = un_segments[nid]
        un_feat = un_feats[nid]
        unique = un_seg.unique()
        nid_feats = []
        for uid in unique:
            mask = (un_seg == uid).float()
            feat = (un_feat * mask).sum((1, 2)) / mask.sum()
            nid_feats.append(feat)
        nid_feats = torch.stack(nid_feats, dim=0)
        s_feats.append(nid_feats)
    return s_feats


def select_topk_spfeats(s_feats, prototypes, cfg):
    """
    prototypes: [1*2048, 1*2048, 1*2048]  ## num_class
    s_feats: [N*2048, N*2048, xxx] ## num_unlabel
    """
    prototypes = torch.cat(prototypes, dim=0)
    topk_sp_for_pt = [[] for _ in range(prototypes.shape[0])]
    for feat in s_feats:
        dist = F.cosine_similarity(feat.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        dist_topk_value, dist_topk_indices = torch.topk(dist.permute(1, 0), dim=1, k=cfg['topk'])
        for pid, (p_value, p_topk) in enumerate(zip(dist_topk_value, dist_topk_indices)):
            gt = torch.where(p_value > cfg['p_value_thres'])[0]
            if len(gt) > 0.1:
                p_topk = p_topk[:gt[-1] + 1]
                topk_sp_for_pt[pid].append(torch.index_select(feat, dim=0, index=p_topk))
            else:
                topk_sp_for_pt[pid].append(None)
    return topk_sp_for_pt


class SemiFewShotSegPartGraph(nn.Module):
    """
    Semi-Fewshot Segmentation model by using the lots of unlabel images
    """

    def __init__(self, cfg=None):
        super(SemiFewShotSegPartGraph, self).__init__()
        self.config = cfg
        self.encoder = resnet50Sem(cfg=cfg)
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=20, init='random')
        self.n_unlabel = cfg['task']['n_unlabels']
        self.n_ways = cfg['task']['n_ways']
        self.n_shots = cfg['task']['n_shots']
        self.kmean_cnt = self.config['center']
        self.device = torch.device('cuda')
        self.channel = 2048
        self.global_const = 0.8
        self.pt_lambda = self.config['pt_lambda']
        None
        self.un_bs = cfg['un_bs']
        self.latentGraph = GraphTransformer(in_channels=2048, out_channels=512)
        self.aspp = _ASPP(2048, 16, [6, 12, 18, 24])

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, un_imgs, unlabel_segments=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        img_fts, image_fts_semantic = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]
        if self.training and self.config['model']['sem']:
            output_semantic = self.aspp(image_fts_semantic)
            output_semantic = F.interpolate(output_semantic, size=img_size, mode='bilinear', align_corners=True)
        else:
            output_semantic = None
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        un_imgs = torch.cat([torch.cat(way, dim=0) for way in un_imgs], dim=0)
        un_mask = torch.cat([torch.cat(way, dim=0) for way in unlabel_segments], dim=0).unsqueeze(1)
        with torch.no_grad():
            un_fts, _ = self.encoder(un_imgs)
        un_segments_feats = getSpFeats(un_fts, un_mask)
        align_loss = torch.zeros(1)
        align_loss_cs = torch.zeros(1)
        outputs = []
        for epi in range(batch_size):
            """get the prototypes from support images"""
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1) for shot in range(n_shots)] for way in range(n_ways)]
            fg_prototypes, bg_prototype, fg_fts, bg_fts = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)
            un_segments_topk = select_topk_spfeats(un_segments_feats, [bg_fts] + fg_fts, self.config)
            un_segments_topk = [[p for p in func(un_topk, self.un_bs)] for un_topk in un_segments_topk]
            num_class = len(un_segments_topk)
            num_fragments = len(un_segments_topk[0])
            """update the prototypes from the unlabel images"""
            un_pts = [[] for _ in range(num_class)]
            for uid in range(num_fragments):
                feat_topk = [topk[uid] for topk in un_segments_topk]
                un_prototypes = self.latentGraph(feat_topk, [bg_prototype] + fg_prototypes)
                bg_prototype = bg_prototype * self.pt_lambda + (1 - self.pt_lambda) * un_prototypes[0]
                fg_prototypes = [(fg_pt * self.pt_lambda + (1 - self.pt_lambda) * un_fg_pt) for fg_pt, un_fg_pt in zip(fg_prototypes, un_prototypes[1:])]
                for wid, un_pts_fid in enumerate(un_prototypes):
                    un_pts[wid].append(un_pts_fid)
            un_pts = [(torch.stack(pts, dim=0).mean(0) + self.global_const * torch.stack(pts, dim=0).mean((0, 1))) for pts in un_pts]
            prototypes = [bg_prototype] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype, take_max=True) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))
            if self.training:
                align_loss_epi = alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi], self.getAlignProto)
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        return output, output_semantic, align_loss / batch_size

    def calDist(self, fts, prototype, scaler=20, take_max=False):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: 1*512*53*53
            prototype: prototype of one semantic class
                expect shape: 5*512
            return: 5*53*53
        """
        if take_max:
            dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler
        else:
            dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2) * scaler
        return dist

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-05)
        return masked_fts

    def getFeaturesArray(self, fts, mask, upscale=2):
        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:]
        h, w = mask.shape[1:]
        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-05)
        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1 * upscale, w1 * upscale), mode='nearest').view(-1)
        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]] * 0
            else:
                fts = fts[mask1 > 0]
        else:
            fts = F.interpolate(fts, size=(h1 * upscale, w1 * upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1 * w1 * upscale ** 2, c)
            fts = fts[mask_bilinear > 0]
        return fts, masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [(sum(way) / n_shots) for way in fg_fts]
        bg_prototype = sum([(sum(way) / n_shots) for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def kmeansPrototype(self, fg_fts, bg_fts):
        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts]
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc)
        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)
        fg_propotypes = [(fg_c + self.global_const * fg_g) for fg_c, fg_g in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.global_const * bg_prop_glo
        return fg_propotypes, bg_propotypes, fg_prop_glo, bg_prop_glo

    def getAlignProto(self, qry_fts, pred_mask, skip_ways, image_size):
        """
        qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
        pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H' x W'
        """
        pred_mask_global = pred_mask.unsqueeze(2)
        qry_prototypes_global = torch.sum(qry_fts.unsqueeze(1) * pred_mask_global, dim=(0, 3, 4))
        qry_prototypes_global = qry_prototypes_global / (pred_mask_global.sum((0, 3, 4)) + 1e-05)
        n, c, h, w = qry_fts.shape
        qry_fts_s4 = F.interpolate(input=qry_fts, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
        qry_fts_s0 = F.interpolate(input=qry_fts, size=image_size, mode='bilinear', align_corners=True)
        qry_fts_s4_reshape = qry_fts_s4.permute(0, 2, 3, 1).contiguous().view(-1, c)
        qry_fts_s0_reshape = qry_fts_s0.permute(0, 2, 3, 1).contiguous().view(-1, c)
        pred_mask_s4 = F.interpolate(input=pred_mask, size=(h * 2, w * 2), mode='nearest')
        pred_mask_s0 = F.interpolate(input=pred_mask, size=image_size, mode='nearest')
        num_background = pred_mask_s4[:, 0].sum()
        if num_background == 0:
            bg_prototypes = qry_fts_s4_reshape[[0]] * 0
        else:
            if num_background <= 10:
                bg_pred = qry_fts_s0_reshape[pred_mask_s0[:, 0].view(-1) > 0]
            else:
                bg_pred = qry_fts_s4_reshape[pred_mask_s4[:, 0].view(-1) > 0]
            bg_prototypes = self.kmeans.cluster(bg_pred)
        bg_prototypes += qry_prototypes_global[[0]] * self.global_const
        pred_mask_s4_ways = pred_mask_s4[:, 1:].permute(1, 0, 2, 3)
        pred_mask_s0_ways = pred_mask_s0[:, 1:].permute(1, 0, 2, 3)
        fg_prototypes = []
        for way_id in range(pred_mask_s0_ways.shape[0]):
            if way_id in skip_ways:
                fg_prototypes.append(None)
                continue
            pred_mask_w_s4, pred_mask_w_s0 = pred_mask_s4_ways[way_id], pred_mask_s0_ways[way_id]
            pred_mask_w_s4 = pred_mask_w_s4.view(-1).contiguous()
            pred_mask_w_s0 = pred_mask_w_s0.view(-1).contiguous()
            num_pos = pred_mask_w_s4.sum()
            if num_pos <= 10:
                qry_fts_w = qry_fts_s0_reshape[pred_mask_w_s0 > 0]
            else:
                qry_fts_w = qry_fts_s4_reshape[pred_mask_w_s4 > 0]
            fg_pro = self.kmeans.cluster(qry_fts_w)
            fg_pro += qry_prototypes_global[[way_id + 1]] * self.global_const
            fg_prototypes.append(fg_pro)
        prototypes = [bg_prototypes] + fg_prototypes
        return prototypes


class Linear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fan_avg = (self.in_features + self.out_features) / 2.0
        bound = np.sqrt(3.0 / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ASPP,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'atrous_rates': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Xiangyi1996_PPNet_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

