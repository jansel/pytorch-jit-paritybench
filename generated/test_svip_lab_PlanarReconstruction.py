import sys
_module = sys.modules[__name__]
del sys
bin_mean_shift = _module
RecordReaderAll = _module
data_tools = _module
convert_tfrecords = _module
instance_parameter_loss = _module
main = _module
match_segmentation = _module
models = _module
baseline_same = _module
resnet_scene = _module
modules = _module
predict = _module
utils = _module
disp = _module
loss = _module
metric = _module
misc = _module
write_ply = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import time


import random


from torch.utils import data


import torchvision.transforms as tf


import math


class Bin_Mean_Shift(nn.Module):

    def __init__(self, train_iter=5, test_iter=10, bandwidth=0.5, device='cpu'):
        super(Bin_Mean_Shift, self).__init__()
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.bandwidth = bandwidth / 2.0
        self.anchor_num = 10
        self.sample_num = 3000
        self.device = device

    def generate_seed(self, point, bin_num):
        """
        :param point: tensor of size (K, 2)
        :param bin_num: int
        :return: seed_point
        """

        def get_start_end(a, b, k):
            start = a + (b - a) / ((k + 1) * 2)
            end = b - (b - a) / ((k + 1) * 2)
            return start, end
        min_x, min_y = point.min(dim=0)[0]
        max_x, max_y = point.max(dim=0)[0]
        start_x, end_x = get_start_end(min_x.item(), max_x.item(), bin_num)
        start_y, end_y = get_start_end(min_y.item(), max_y.item(), bin_num)
        x = torch.linspace(start_x, end_x, bin_num).view(bin_num, 1)
        y = torch.linspace(start_y, end_y, bin_num).view(1, bin_num)
        x_repeat = x.repeat(1, bin_num).view(-1, 1)
        y_repeat = y.repeat(bin_num, 1).view(-1, 1)
        return torch.cat((x_repeat, y_repeat), dim=1)

    def filter_seed(self, point, prob, seed_point, bandwidth, min_count=3):
        """
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param min_count:  mini_count within a bandwith of seed point
        :param bandwidth: float
        :return: filtered_seed_points
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)
        thres_matrix = (distance_matrix < bandwidth).type(torch.float32) * prob.t()
        count = thres_matrix.sum(dim=1)
        valid = count > min_count
        return seed_point[valid]

    def cal_distance_matrix(self, point_a, point_b):
        """
        :param point_a: tensor of size (m, 2)
        :param point_b: tensor of size (n, 2)
        :return: distance matrix of size (m, n)
        """
        m, n = point_a.size(0), point_b.size(0)
        a_repeat = point_a.repeat(1, n).view(n * m, 2)
        b_repeat = point_b.repeat(m, 1)
        distance = torch.nn.PairwiseDistance(keepdim=True)(a_repeat, b_repeat)
        return distance.view(m, n)

    def shift(self, point, prob, seed_point, bandwidth):
        """
        shift seed points
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param bandwidth: float
        :return:  shifted points with size (n, 2)
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)
        kernel_matrix = torch.exp(-0.5 / bandwidth ** 2 * distance_matrix ** 2) * (1.0 / (bandwidth * np.sqrt(2 * np.pi)))
        weighted_matrix = kernel_matrix * prob.t()
        normalized_matrix = weighted_matrix / weighted_matrix.sum(dim=1, keepdim=True)
        shifted_point = torch.matmul(normalized_matrix, point)
        return shifted_point

    def label2onehot(self, labels):
        """
        convert a label to one hot vector
        :param labels: tensor with size (n, 1)
        :return: one hot vector tensor with size (n, max_lales+1)
        """
        n = labels.size(0)
        label_num = torch.max(labels).int() + 1
        onehot = torch.zeros((n, label_num))
        onehot.scatter_(1, labels.long(), 1.0)
        return onehot

    def merge_center(self, seed_point, bandwidth=0.25):
        """
        merge close seed points
        :param seed_point: tensor of size (n, 2)
        :param bandwidth: float
        :return: merged center
        """
        n = seed_point.size(0)
        distance_matrix = self.cal_distance_matrix(seed_point, seed_point)
        intensity = (distance_matrix < bandwidth).type(torch.float32).sum(dim=1)
        sorted_intensity, indices = torch.sort(intensity, descending=True)
        is_center = np.ones(n, dtype=np.bool)
        indices = indices.cpu().numpy()
        center = np.zeros(n, dtype=np.uint8)
        labels = np.zeros(n, dtype=np.int32)
        cur_label = 0
        for i in range(n):
            if is_center[i]:
                labels[indices[i]] = cur_label
                center[indices[i]] = 1
                for j in range(i + 1, n):
                    if is_center[j]:
                        if distance_matrix[indices[i], indices[j]] < bandwidth:
                            is_center[j] = 0
                            labels[indices[j]] = cur_label
                cur_label += 1
        one_hot = self.label2onehot(torch.Tensor(labels).view(-1, 1))
        weight = one_hot / one_hot.sum(dim=0, keepdim=True)
        return torch.matmul(weight.t(), seed_point)

    def cluster(self, point, center):
        """
        cluter each point to nearset center
        :param point: tensor with size (K, 2)
        :param center: tensor with size (n, 2)
        :return: clustering results, tensor with size (K, n) and sum to one for each row
        """
        distance_matrix = 1.0 / (self.cal_distance_matrix(point, center) + 0.01)
        segmentation = F.softmax(distance_matrix, dim=1)
        return segmentation

    def bin_shift(self, prob, embedding, param, gt_seg, bandwidth):
        """
        discrete seeding mean shift in training stage
        :param prob: tensor with size (1, h, w) indicate probability of being plane
        :param embedding: tensor with size (2, h, w)
        :param param: tensor with size (3, h, w)
        :param gt_seg: ground truth instance segmentation, used for sampling planar embeddings
        :param bandwidth: float
        :return: segmentation results, tensor with size (h*w, K), K is cluster number, row sum to 1
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                center, tensor with size (K, 2) cluster center in embedding space
                sample_prob, tensor with size (N, 1) sampled probability
                sample_seg, tensor with size (N, 1) sampled ground truth instance segmentation
                sample_params, tensor with size (3, N), sampled params
        """
        c, h, w = embedding.size()
        embedding = embedding.view(c, h * w).t()
        param = param.view(3, h * w)
        prob = prob.view(h * w, 1)
        seg = gt_seg.view(-1)
        rand_index = np.random.choice(np.arange(0, h * w)[seg.cpu().numpy() != 20], self.sample_num)
        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, (rand_index)]
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None
        with torch.no_grad():
            for iter in range(self.train_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None
        center = self.merge_center(seed_point, bandwidth=self.bandwidth)
        segmentation = self.cluster(embedding, center)
        sampled_segmentation = segmentation[rand_index]
        return segmentation, sampled_segmentation, center, sample_prob, seg[rand_index].view(-1, 1), sample_param

    def forward(self, logit, embedding, param, gt_seg):
        batch_size, c, h, w = embedding.size()
        assert c == 2
        segmentations, sample_segmentations, centers, sample_probs, sample_gt_segs, sample_params = [], [], [], [], [], []
        for b in range(batch_size):
            segmentation, sample_segmentation, center, prob, sample_seg, sample_param = self.bin_shift(torch.sigmoid(logit[b]), embedding[b], param[b], gt_seg[b], self.bandwidth)
            segmentations.append(segmentation)
            sample_segmentations.append(sample_segmentation)
            centers.append(center)
            sample_probs.append(prob)
            sample_gt_segs.append(sample_seg)
            sample_params.append(sample_param)
        return segmentations, sample_segmentations, sample_params, centers, sample_probs, sample_gt_segs

    def test_forward(self, prob, embedding, param, mask_threshold):
        """
        :param prob: probability of planar, tensor with size (1, h, w)
        :param embedding: tensor with size (2, h, w)
        :param mask_threshold: threshold of planar region
        :return: clustering results: numpy array with shape (h, w),
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                 sample_params, tensor with size (3, N), sampled params
        """
        c, h, w = embedding.size()
        embedding = embedding.view(c, h * w).t()
        prob = prob.view(h * w, 1)
        param = param.view(3, h * w)
        rand_index = np.random.choice(np.arange(0, h * w)[prob.cpu().numpy().reshape(-1) > mask_threshold], self.sample_num)
        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, (rand_index)]
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)
        with torch.no_grad():
            for iter in range(self.test_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)
        center = self.merge_center(seed_point, bandwidth=self.bandwidth)
        segmentation = self.cluster(embedding, center)
        sampled_segmentation = segmentation[rand_index]
        return segmentation, sampled_segmentation, sample_param


class InstanceParameterLoss(nn.Module):

    def __init__(self, k_inv_dot_xy1):
        super(InstanceParameterLoss, self).__init__()
        self.k_inv_dot_xy1 = k_inv_dot_xy1

    def forward(self, segmentation, sample_segmentation, sample_params, valid_region, gt_depth, return_loss=True):
        """
        calculate loss of parameters
        first we combine sample segmentation with sample params to get K plane parameters
        then we used this parameter to infer plane based Q loss as done in PlaneRecover
        the loss enforce parameter is consistent with ground truth depth

        :param segmentation: tensor with size (h*w, K)
        :param sample_segmentation: tensor with size (N, K)
        :param sample_params: tensor with size (3, N), defined as n / d
        :param valid_region: tensor with size (1, 1, h, w), indicate planar region
        :param gt_depth: tensor with size (1, 1, h, w)
        :param return_loss: bool
        :return: loss
                 inferred depth with size (1, 1, h, w) corresponded to instance parameters
        """
        n = sample_segmentation.size(0)
        _, _, h, w = gt_depth.size()
        assert segmentation.size(1) == sample_segmentation.size(1) and segmentation.size(0) == h * w and sample_params.size(1) == sample_segmentation.size(0)
        if not return_loss:
            sample_segmentation[sample_segmentation < 0.5] = 0.0
        weight_matrix = F.normalize(sample_segmentation, p=1, dim=0)
        instance_param = torch.matmul(sample_params, weight_matrix)
        depth_maps = 1.0 / torch.matmul(instance_param.t(), self.k_inv_dot_xy1)
        _, index = segmentation.max(dim=1)
        inferred_depth = depth_maps.t()[range(h * w), index].view(1, 1, h, w)
        if not return_loss:
            return _, inferred_depth, _, instance_param
        valid_region = (valid_region + (gt_depth != 0.0) == 2).view(-1)
        ray = self.k_inv_dot_xy1[:, (valid_region)]
        segmentation = segmentation[valid_region]
        valid_depth = gt_depth.view(1, -1)[:, (valid_region)]
        valid_inferred_depth = inferred_depth.view(1, -1)[:, (valid_region)]
        Q = valid_depth * ray
        Q_loss = torch.abs(torch.matmul(instance_param.t(), Q) - 1.0)
        weighted_Q_loss = Q_loss * segmentation.t()
        loss = torch.sum(torch.mean(weighted_Q_loss, dim=1))
        abs_distance = torch.mean(torch.abs(valid_inferred_depth - valid_depth))
        return loss, inferred_depth, abs_distance, instance_param


class MatchSegmentation(nn.Module):

    def __init__(self):
        super(MatchSegmentation, self).__init__()

    def forward(self, segmentation, prob, gt_instance, gt_plane_num):
        """
        greedy matching
        match segmentation with ground truth instance 
        :param segmentation: tensor with size (N, K)
        :param prob: tensor with size (N, 1)
        :param gt_instance: tensor with size (21, h, w)
        :param gt_plane_num: int
        :return: a (K, 1) long tensor indicate closest ground truth instance id, start from 0
        """
        n, k = segmentation.size()
        _, h, w = gt_instance.size()
        assert prob.size(0) == n and h * w == n
        gt_instance = gt_instance[:gt_plane_num, :, :].view(1, -1, h * w)
        segmentation = segmentation.t().view(k, 1, h * w)
        gt_instance = gt_instance.type(torch.float32)
        ce_loss = -(gt_instance * torch.log(segmentation + 1e-06) + (1 - gt_instance) * torch.log(1 - segmentation + 1e-06))
        ce_loss = torch.mean(ce_loss, dim=2)
        matching = torch.argmin(ce_loss, dim=1, keepdim=True)
        return matching


class ResNet(nn.Module):

    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class Baseline(nn.Module):

    def __init__(self, cfg):
        super(Baseline, self).__init__()
        orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)
        self.backbone = ResNet(orig_resnet)
        self.relu = nn.ReLU(inplace=True)
        channel = 64
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))
        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.pred_prob = nn.Conv2d(channel, 1, (1, 1), padding=0)
        self.embedding_conv = nn.Conv2d(channel, 2, (1, 1), padding=0)
        self.pred_depth = nn.Conv2d(channel, 1, (1, 1), padding=0)
        self.pred_surface_normal = nn.Conv2d(channel, 3, (1, 1), padding=0)
        self.pred_param = nn.Conv2d(channel, 3, (1, 1), padding=0)

    def top_down(self, x):
        c1, c2, c3, c4, c5 = x
        p5 = self.relu(self.c5_conv(c5))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))
        p0 = self.upsample(p1)
        p0 = self.relu(self.p0_conv(p0))
        return p0, p1, p2, p3, p4, p5

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)
        p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5))
        prob = self.pred_prob(p0)
        embedding = self.embedding_conv(p0)
        depth = self.pred_depth(p0)
        surface_normal = self.pred_surface_normal(p0)
        param = self.pred_param(p0)
        return prob, embedding, depth, surface_normal, param


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_svip_lab_PlanarReconstruction(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

