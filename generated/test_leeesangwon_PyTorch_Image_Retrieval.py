import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
datasets = _module
inference = _module
losses = _module
main = _module
networks = _module
senet = _module
setup = _module
trainer = _module

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


import torch.nn as nn


import numpy as np


import torch.optim as optim


import torch.nn.functional as F


from collections import OrderedDict


import math


from torch.utils import model_zoo


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)
        if embeddings.is_cuda:
            n_pairs = n_pairs
            n_negatives = n_negatives
        anchors = embeddings[n_pairs[:, (0)]]
        positives = embeddings[n_pairs[:, (1)]]
        negatives = embeddings[n_negatives]
        losses = self.n_pair_loss(anchors, positives, negatives
            ) + self.l2_reg * self.l2_loss(anchors, positives)
        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []
        for label in set(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False
                )
            n_pairs.append([anchor, positive])
        n_pairs = np.array(n_pairs)
        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, (1)], n_pairs[i + 1:, (1)]])
            n_negatives.append(negative)
        n_negatives = np.array(n_negatives)
        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)
        positives = torch.unsqueeze(positives, dim=1)
        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))
        x = torch.sum(torch.exp(x), 2)
        loss = torch.mean(torch.log(1 + x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class SelfAttention(nn.Module):
    """ Self attention Layer
    https://github.com/heykeetae/Self-Attention-GAN"""

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leeesangwon_PyTorch_Image_Retrieval(_paritybench_base):
    pass
    def test_000(self):
        self._check(SelfAttention(*[], **{'in_dim': 64, 'activation': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_001(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

