import sys
_module = sys.modules[__name__]
del sys
stage_1 = _module
stage_2_meta_embedding = _module
ClassAwareSampler = _module
dataloader = _module
ModulatedAttLayer = _module
DiscCentroidsLoss = _module
SoftmaxLoss = _module
main = _module
CosNormClassifier = _module
DotProductClassifier = _module
MetaEmbeddingClassifier = _module
ResNet10Feature = _module
ResNet152Feature = _module
ResNetFeature = _module
run_networks = _module
utils = _module

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


from torch.nn import functional as F


import torch.nn as nn


from torch.autograd.function import Function


import math


from torch.nn.parameter import Parameter


import torch.nn.functional as F


import copy


import torch.optim as optim


import numpy as np


import warnings


class ModulatedAttLayer(nn.Module):

    def __init__(self, in_channels, reduction=2, mode='embedded_gaussian'):
        super(ModulatedAttLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1
            )
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels,
            kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        self.init_weights()

    def init_weights(self):
        msra_list = [self.g, self.theta, self.phi]
        for m in msra_list:
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()

    def embedded_gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x.clone()).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x.clone()).view(batch_size, self.
            inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x.clone()).view(batch_size, self.inter_channels, -1)
        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)
        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)
        x_flatten = x.view(-1, 7 * 7 * self.in_channels)
        spatial_att = self.fc_spatial(x_flatten)
        spatial_att = spatial_att.softmax(dim=1)
        spatial_att = spatial_att.view(-1, 7, 7).unsqueeze(1)
        spatial_att = spatial_att.expand(-1, self.in_channels, -1, -1)
        final = spatial_att * mask + x
        return final, [x, spatial_att, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplemented('The code has not been implemented.')
        return output, feature_maps


class DiscCentroidsLossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())
        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.
            size()).long(), diff)
        grad_centroids = grad_centroids / counts.view(-1, 1)
        return (-grad_output * diff / batch_size, None, grad_centroids /
            batch_size, None)


class DiscCentroidsLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's                             dim: {1}"
                .format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.
            size_average else 1)
        loss_attract = self.disccentroidslossfunc(feat.clone(), label, self
            .centroids.clone(), batch_size_tensor).squeeze()
        distmat = torch.pow(feat.clone(), 2).sum(dim=1, keepdim=True).expand(
            batch_size, self.num_classes) + torch.pow(self.centroids.clone(), 2
            ).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat.clone(), self.centroids.clone().t())
        classes = torch.arange(self.num_classes).long()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size *
            self.num_classes), 0.0, 1000000.0)
        loss = loss_attract + 0.01 * loss_repel
        return loss


class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001
        ):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = norm_x / (1 + norm_x) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, *args):
        x = self.fc(x)
        return x, None


class MetaEmbedding_Classifier(nn.Module):

    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

    def forward(self, x, centroids, *args):
        direct_feature = x.clone()
        batch_size = x.size(0)
        feat_size = x.size(1)
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size,
            -1, -1)
        keys_memory = centroids.clone()
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        reachability = (scale / values_nn[:, (0)]).unsqueeze(1).expand(-1,
            feat_size)
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh()
        x = reachability * (direct_feature + concept_selector * memory_feature)
        infused_feature = concept_selector * memory_feature
        logits = self.cosnorm_classifier(x)
        return logits, [direct_feature, infused_feature]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_modulatedatt=False, use_fc=False,
        dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.use_fc = use_fc
        self.use_dropout = True if dropout else False
        if self.use_fc:
            None
            self.fc_add = nn.Linear(512 * block.expansion, 512)
        if self.use_dropout:
            None
            self.dropout = nn.Dropout(p=dropout)
        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            None
            self.modulatedatt = ModulatedAttLayer(in_channels=512 * block.
                expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_modulatedatt:
            x, feature_maps = self.modulatedatt(x)
        else:
            feature_maps = None
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.use_fc:
            x = F.relu(self.fc_add(x))
        if self.use_dropout:
            x = self.dropout(x)
        return x, feature_maps


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhmiao_OpenLongTailRecognition_OLTR(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(CosNorm_Classifier(*[], **{'in_dims': 4, 'out_dims': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(DotProduct_Classifier(*[], **{}), [torch.rand([2048, 2048])], {})

