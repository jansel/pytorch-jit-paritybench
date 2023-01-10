import sys
_module = sys.modules[__name__]
del sys
datasets = _module
dataset_culane = _module
evaluation = _module
eval_culane_laneatt = _module
eval_culane_official = _module
libs = _module
generator = _module
load_model = _module
prepare = _module
save_model = _module
utils = _module
main = _module
networks = _module
backbone = _module
loss = _module
model = _module
options = _module
args = _module
config = _module
post_processes = _module
post_process = _module
save_prediction = _module
tests = _module
forward = _module
test = _module
trains = _module
train = _module
visualizes = _module
visualize = _module
dataset_tusimple = _module
eval_tusimple = _module
generator = _module
load_model = _module
prepare = _module
save_model = _module
utils = _module
main = _module
backbone = _module
loss = _module
model = _module
config = _module
post_process = _module
forward = _module
test = _module
train = _module
visualize = _module
preprocess = _module
utils = _module
config = _module
dataset_culane = _module
prepare = _module
preprocess = _module
utils = _module
config = _module
dataset_culane = _module
prepare = _module
preprocess = _module
utils = _module
main = _module
config = _module
visualize = _module
preprocess = _module
utils = _module
config = _module
dataset_culane = _module
prepare = _module
preprocess1 = _module
preprocess2 = _module
utils = _module
config = _module
visualize = _module
preprocess = _module
utils = _module
config = _module
dataset_tusimple = _module
prepare = _module
preprocess = _module
utils = _module
config = _module
dataset_tusimple = _module
prepare = _module
preprocess = _module
utils = _module
config = _module
visualize = _module
preprocess = _module
utils = _module
config = _module
dataset_tusimple = _module
prepare = _module
preprocess1 = _module
preprocess2 = _module
utils = _module
config = _module
visualize = _module

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


import torchvision.transforms as transforms


from torch.utils.data import Dataset


import torch


import numpy as np


import random


import torchvision


import torch.nn.modules


import torch.nn as nn


import torchvision.models as models


import torch.nn.functional as F


import math


import time


import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression


from scipy.optimize import curve_fit


from scipy import interpolate


import torch.nn.parallel


import torch.optim


from sklearn.cluster import k_means


from sklearn.cluster import KMeans


class resnet(nn.Module):

    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class SoftmaxFocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll1 = nn.NLLLoss(reduce=True)
        self.nll2 = nn.NLLLoss(reduce=False)

    def forward(self, logits, labels, reduce=True):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.0 - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        if reduce == True:
            loss = self.nll1(log_score, labels)
        else:
            loss = self.nll2(log_score, labels)
        return loss


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


def to_tensor(data):
    return torch.from_numpy(data)


class Loss_Function(nn.Module):

    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.cfg = cfg
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_nce = nn.CrossEntropyLoss()
        self.loss_score = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(gamma=2)
        self.weight = 1
        offset = load_pickle(cfg.dir['pre4'] + 'offset_distribution_{}_{}_{}_{}'.format(cfg.thresd_iou_for_cls, cfg.thresd_iou_upper_for_cls, cfg.thresd_min_offset, cfg.thresd_max_offset))
        self.c_weight = 1 / (np.abs(offset['mean']) / np.abs(offset['mean']).max())
        candidates = load_pickle(cfg.dir['pre3'] + 'lane_candidates_' + str(cfg.n_clusters))
        self.cand_c = to_tensor(candidates['c'])

    def forward(self, out, gt):
        loss_dict = dict()
        l_prob = self.loss_focal(out['prob_logit'], gt['prob'])
        l_offset_tot = torch.FloatTensor([0.0])
        for i in range(self.cfg.top_m):
            l_offset = self.balanced_MSE_loss(out['offset'][:, :, i:i + 1], gt['offset'][:, :, i:i + 1], gt['is_pos_reg']) * 0.1 * self.c_weight[i]
            loss_dict['offset' + str(i + 1)] = l_offset
            l_offset_tot += l_offset
        l_edge = self.loss_mse(out['edge_map'], out['gt_edge_map'])
        l_seg = self.loss_bce(out['seg_map'][:, 0], gt['seg_label'])
        if torch.sum(gt['exist_check']) > 0:
            val = self.loss_focal(out['height_prob_logit'][gt['exist_check'] == 1], gt['height_prob'][gt['exist_check'] == 1], reduce=False) * gt['prob'][gt['exist_check'] == 1].type(torch.float)
            l_prob_h = torch.mean(torch.sum(val, dim=1) / torch.sum(gt['prob'][gt['exist_check'] == 1].type(torch.float), dim=1), dim=0) * 0.01
        else:
            l_prob_h = torch.FloatTensor([0.0])
        loss_dict['sum'] = l_prob + l_offset_tot + l_seg + l_edge + l_prob_h
        loss_dict['prob'] = l_prob
        loss_dict['prob_h'] = l_prob_h
        loss_dict['edge'] = l_edge
        loss_dict['seg'] = l_seg
        return loss_dict

    def balanced_MSE_loss(self, out, gt, gt_check):
        ep = 1e-06
        neg_mask = (gt_check == 0).unsqueeze(2)
        pos_mask = (gt_check != 0).unsqueeze(2)
        neg_num = torch.sum(neg_mask, dim=(1, 2)) + ep
        pos_num = torch.sum(pos_mask, dim=(1, 2)) + ep
        pos_loss = torch.mean(torch.sum(F.mse_loss(out * pos_mask, gt * pos_mask, reduce=False), dim=(1, 2)) / pos_num)
        neg_loss = torch.mean(torch.sum(F.mse_loss(out * neg_mask, gt * neg_mask, reduce=False), dim=(1, 2)) / neg_num)
        return pos_loss + neg_loss


class conv_bn_relu(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class conv1d_bn_relu(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def l2_normalization(x):
    ep = 1e-06
    out = x / (torch.norm(x, p=2, dim=1, keepdim=True) + ep)
    return out


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.sf = self.cfg.scale_factor
        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_c = to_tensor(candidates['c'])
        self.cand_mask = load_pickle(self.cfg.dir['pre3'] + 'candidate_mask_' + str(self.cfg.n_clusters))
        self.cand_iou = load_pickle(self.cfg.dir['pre3'] + 'candidate_iou_map_' + str(self.cfg.n_clusters))
        self.cand_iou_upper = load_pickle(self.cfg.dir['pre3'] + 'candidate_iou_upper_map_' + str(self.cfg.n_clusters))
        self.cand_iou = to_tensor(self.cand_iou)
        self.cand_iou_upper = to_tensor(self.cand_iou_upper)
        self.cand_area = dict()
        sf = cfg.scale_factor[0]
        self.cand_mask[sf], self.cand_area[sf] = self.get_lane_mask_area(self.cand_mask[sf])
        self.n_cand = self.cand_mask[sf].shape[2]
        self.c_feat = 512
        self.c_feat2 = 64
        self.c_sq = 32
        self.c_sq2 = 128
        self.encoder = resnet(layers=self.cfg.backbone, pretrained=True)
        backbone = self.cfg.backbone
        self.feat_squeeze1 = torch.nn.Sequential(conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1), conv_bn_relu(128, 128, 3, padding=1), conv_bn_relu(128, 128, 3, padding=1), conv_bn_relu(128, self.c_feat2, 3, padding=1))
        self.feat_squeeze2 = torch.nn.Sequential(conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1), conv_bn_relu(128, 128, 3, padding=1), conv_bn_relu(128, self.c_feat2, 3, padding=1))
        self.feat_squeeze3 = torch.nn.Sequential(conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1), conv_bn_relu(128, self.c_feat2, 3, padding=1))
        self.feat_combine = torch.nn.Sequential(conv_bn_relu(self.c_feat2 * 3, 256, 3, padding=2, dilation=2), conv_bn_relu(256, 128, 3, padding=2, dilation=2), conv_bn_relu(128, 128, 3, padding=4, dilation=4), torch.nn.Conv2d(128, self.c_sq, 1))
        self.decoder = torch.nn.Sequential(conv_bn_relu(self.c_feat2 * 3, self.c_feat2 * 2, 3, padding=2, dilation=2), conv_bn_relu(self.c_feat2 * 2, 128, 3, padding=2, dilation=2), conv_bn_relu(128, 128, 3, padding=2, dilation=2), conv_bn_relu(128, 128, 3, padding=4, dilation=4), torch.nn.Conv2d(128, 1, 1), nn.Sigmoid())
        self.classification1 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, 2, kernel_size=1, bias=False))
        self.classification2 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, self.cfg.height_class.shape[0], kernel_size=1, bias=False))
        self.regression1 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False))
        self.regression2 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False))
        self.regression3 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False))
        self.regression4 = nn.Sequential(nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_sq), nn.ReLU(inplace=True), nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False))
        self.w1 = nn.Sequential(nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_feat2 * 3), nn.ReLU(inplace=True), nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False))
        self.w2 = nn.Sequential(nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False), nn.BatchNorm1d(self.c_feat2 * 3), nn.ReLU(inplace=True), nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False))

    def get_lane_mask_area(self, mask):
        n, h, w = mask.shape
        area = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            area[i] = mask[i].nonzero().shape[0]
        return mask.view(1, 1, n, h, w), area

    def lane_pooling(self, feat_map, idx, sf):
        b, c, h, w = feat_map.shape
        _, n = idx.shape
        mask = self.cand_mask[sf][:, :, idx].view(b, 1, n, h, w)
        area = self.cand_area[sf][idx].view(b, 1, n, 1, 1)
        line_feat = torch.sum(mask * feat_map.view(b, c, 1, h, w), dim=(3, 4), keepdim=True) / area
        return line_feat[:, :, :, 0, 0]

    def extract_lane_feat(self, feat_map, sf):
        b, c, h, w = feat_map.shape
        line_feat = torch.sum(self.cand_mask[sf][:, :, :] * feat_map.view(b, c, 1, h, w), dim=(3, 4)) / self.cand_area[sf][:].view(1, 1, -1)
        return line_feat

    def selection_and_removal(self, prob, batch_idx):
        idx_max = torch.sort(prob, descending=True, dim=1)[1][0, 0]
        cluster_idx = (self.cand_iou[idx_max] >= self.thresd_nms_iou).nonzero()[:, 0]
        if prob[0][idx_max] >= self.thresd_score:
            self.visit_mask[batch_idx, :, cluster_idx] = 0
            self.center_mask[batch_idx, :, idx_max] = 0
        return prob[0][idx_max], idx_max

    def forward_for_encoding(self, img):
        feat1, feat2, feat3 = self.encoder(img)
        self.feat = dict()
        self.feat[self.sf[0]] = feat1
        self.feat[self.sf[1]] = feat2
        self.feat[self.sf[2]] = feat3

    def forward_for_decoding(self):
        out = self.decoder(self.x_concat)
        return {'seg_map': out}

    def forward_for_squeeze(self):
        x1 = self.feat_squeeze1(self.feat[self.sf[0]])
        x2 = self.feat_squeeze2(self.feat[self.sf[1]])
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.feat_squeeze3(self.feat[self.sf[2]])
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')
        self.x_concat = torch.cat([x1, x2, x3], dim=1)
        self.sq_feat = self.feat_combine(self.x_concat)

    def forward_for_lane_feat_extraction(self):
        self.l_feat = self.extract_lane_feat(self.sq_feat, self.sf[0])

    def forward_for_lane_component_prediction(self):
        out1 = self.classification1(self.l_feat)
        out2 = self.classification2(self.l_feat)
        offset = list()
        offset.append(self.regression1(self.l_feat))
        offset.append(self.regression2(self.l_feat))
        offset.append(self.regression3(self.l_feat))
        offset.append(self.regression4(self.l_feat))
        offset = torch.cat(offset, dim=1)
        return {'prob': F.softmax(out1, dim=1)[:, 1:, :], 'prob_logit': out1, 'height_prob': F.softmax(out2, dim=1), 'height_prob_logit': out2, 'offset': offset.permute(0, 2, 1)}

    def forward_for_matching(self, idx):
        _, d = self.cand_c.shape
        batch_l_feat = self.lane_pooling(self.x_concat, idx, self.sf[0])
        out = self.correlation(batch_l_feat)
        for i in range(len(idx)):
            out[i].fill_diagonal_(0)
        return {'edge_map': out}

    def correlation(self, x):
        x1 = l2_normalization(self.w1(x))
        x2 = l2_normalization(self.w2(x))
        corr = torch.matmul(x1.permute(0, 2, 1), x2)
        return corr


class vgg16_bn(nn.Module):

    def __init__(self, pretrained=False):
        super(vgg16_bn, self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        self.model1 = torch.nn.Sequential(*model[:33])
        self.model2 = torch.nn.Sequential(*model[34:44])

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(out1)
        return out1, out2


class vgg16(nn.Module):

    def __init__(self, pretrained=True):
        super(vgg16, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3']
        last_layer = 'relu4_3'
        last_layer_idx = vgg_feature_layers.index(last_layer)
        self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx + 1])
        self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx + 1:-1])

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(out1)
        return out1, out2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (conv1d_bn_relu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (conv_bn_relu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (vgg16_bn,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_dongkwonjin_Eigenlanes(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

