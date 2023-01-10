import sys
_module = sys.modules[__name__]
del sys
classification = _module
dataset = _module
inference = _module
loss = _module
main = _module
model = _module
reconstruction = _module
svm = _module
utils = _module
visualization = _module

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


import time


import numpy as np


import torch


import torch.optim as optim


from torch.optim.lr_scheduler import CosineAnnealingLR


import sklearn.metrics as metrics


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import itertools


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2


class CrossEntropyLoss(nn.Module):

    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)
        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)
            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')
        return loss


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNN_Cls_Encoder(nn.Module):

    def __init__(self, args):
        super(DGCNN_Cls_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.task = args.task
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.feat_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)
        if self.task == 'classify':
            return feat, x0
        elif self.task == 'reconstruct':
            return feat


class Point_Transform_Net(nn.Module):

    def __init__(self):
        super(Point_Transform_Net, self).__init__()
        self.k = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)
        x = self.transform(x)
        x = x.view(batch_size, 3, 3)
        return x


class DGCNN_Seg_Encoder(nn.Module):

    def __init__(self, args):
        super(DGCNN_Seg_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.transform_net = Point_Transform_Net()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.feat_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.feat_dims, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)
        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)
        return feat


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)
    _, num_dims, _ = pts.size()
    x = pts.transpose(2, 1).contiguous()
    x = x.view(batch_size * num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, -1, num_dims)
    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(2))
    x = x.view(batch_size, num_points, 9).transpose(2, 1)
    x = torch.cat((pts, x), dim=1)
    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    x = x.view(batch_size * num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, -1, num_dims)
    x, _ = torch.max(x, dim=2)
    return x


class FoldNet_Encoder(nn.Module):

    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048
        self.mlp1 = nn.Sequential(nn.Conv1d(12, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.ReLU())
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(nn.Conv1d(1024, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, args.feat_dims, 1))

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        return feat


class FoldNet_Decoder(nn.Module):

    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        self.m = 2025
        self.shape = args.shape
        self.meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        self.sphere = np.load('sphere.npy')
        self.gaussian = np.load('gaussian.npy')
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(nn.Conv1d(args.feat_dims + 2, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, 3, 1))
        else:
            self.folding1 = nn.Sequential(nn.Conv1d(args.feat_dims + 3, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, 3, 1))
        self.folding2 = nn.Sequential(nn.Conv1d(args.feat_dims + 3, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, args.feat_dims, 1), nn.ReLU(), nn.Conv1d(args.feat_dims, 3, 1))

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        if x.get_device() != -1:
            points = points
        cat1 = torch.cat((x, points), dim=1)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        return folding_result2.transpose(1, 2)


class DGCNN_Cls_Classifier(nn.Module):

    def __init__(self, args):
        super(DGCNN_Cls_Classifier, self).__init__()
        if args.dataset == 'modelnet40':
            output_channels = 40
        elif args.dataset == 'modelnet10':
            output_channels = 10
        elif args.dataset == 'shapenetcorev2':
            output_channels = 55
        elif args.dataset == 'shapenetpart':
            output_channels = 16
        self.linear1 = nn.Linear(args.feat_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class ReconstructionNet(nn.Module):

    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        self.decoder = FoldNet_Decoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        return self.loss(input, output)


class ClassificationNet(nn.Module):

    def __init__(self, args):
        super(ClassificationNet, self).__init__()
        self.is_eval = args.eval
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        if not self.is_eval:
            self.classifier = DGCNN_Cls_Classifier(args)
        self.loss = CrossEntropyLoss()

    def forward(self, input):
        feature, latent = self.encoder(input)
        if not self.is_eval:
            output = self.classifier(latent)
            return output, feature
        else:
            return feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.classifier.parameters())

    def get_loss(self, input, output):
        return self.loss(input, output)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChamferLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Point_Transform_Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     True),
]

class Test_antao97_UnsupervisedPointCloudReconstruction(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

