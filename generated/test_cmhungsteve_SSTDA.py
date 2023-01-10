import sys
_module = sys.modules[__name__]
del sys
batch_gen = _module
centroid = _module
eval = _module
loss = _module
main = _module
model = _module
predict = _module
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


import numpy as np


import random


import torch.nn as nn


import time


from torch.autograd import Function


import torch.nn.functional as F


import copy


import math


from itertools import permutations


from itertools import combinations


from torch import optim


class Centroid(nn.Module):

    def __init__(self, num_f_maps, num_classes):
        super(Centroid, self).__init__()
        self.dim_feat = num_f_maps
        self.num_classes = num_classes
        self.register_buffer('centroid_s', torch.zeros(num_classes, num_f_maps))
        self.register_buffer('centroid_t', torch.zeros(num_classes, num_f_maps))

    def update_centroids(self, feat_s, feat_t, y_s, y_t, method_centroid, ratio_ma):
        label_source = y_s.detach()
        if method_centroid == 'prob_hard':
            label_target = torch.max(y_t, 1)[1].detach()
        centroid_source = torch.zeros(self.num_classes, self.dim_feat, device=feat_s.device)
        centroid_target = torch.zeros(self.num_classes, self.dim_feat, device=feat_t.device)
        for i in range(self.num_classes):
            feat_source_select = feat_s[label_source == i]
            feat_target_select = feat_t[label_target == i]
            centroid_source_current = feat_source_select.mean(0) if feat_source_select.size(0) > 0 else torch.zeros_like(feat_s[0])
            centroid_target_current = feat_target_select.mean(0) if feat_target_select.size(0) > 0 else torch.zeros_like(feat_t[0])
            centroid_source[i] = ratio_ma * self.centroid_s[i] + (1 - ratio_ma) * centroid_source_current
            centroid_target[i] = ratio_ma * self.centroid_t[i] + (1 - ratio_ma) * centroid_target_current
        return centroid_source, centroid_target

    def forward(self):
        return self.centroid_s, self.centroid_t


class GradRevLayer(Function):

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class AdvDomainClsBase(nn.Module):

    def __init__(self, in_feat, hidden_size, type_adv, args):
        super(AdvDomainClsBase, self).__init__()
        self.num_f_maps = args.num_f_maps
        self.DA_adv_video = args.DA_adv_video
        self.pair_ssl = args.pair_ssl
        self.type_adv = type_adv
        if self.type_adv == 'video' and self.DA_adv_video == 'rev_grad_ssl_2':
            self.fc_pair = nn.Linear(self.num_f_maps * 2, self.num_f_maps)
        self.fc1 = nn.Linear(in_feat, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_data, beta):
        feat = GradRevLayer.apply(input_data, beta)
        if self.type_adv == 'video' and self.DA_adv_video == 'rev_grad_ssl_2':
            num_seg = int(input_data.size(-1) / self.num_f_maps)
            feat = feat.reshape(-1, num_seg, self.num_f_maps)
            id_pair = torch.tensor(list(combinations(range(num_seg), 2))).long()
            if self.pair_ssl == 'adjacent':
                id_pair = torch.tensor([(i, i + 1) for i in range(num_seg - 1)])
            if input_data.get_device() >= 0:
                id_pair = id_pair
            feat = feat[:, id_pair, :]
            feat = feat.reshape(-1, self.num_f_maps * 2)
            feat = self.fc_pair(feat)
            feat = feat.reshape(-1, id_pair.size(0) * self.num_f_maps)
        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        return feat


class DilatedResidualLayer(nn.Module):

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageModel(nn.Module):

    def __init__(self, num_layers, num_f_maps, dim_in, num_classes, DA_ens):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim_in, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        if DA_ens != 'none':
            self.conv_out_2 = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out_feat = self.conv_1x1(x)
        for layer in self.layers:
            out_feat = layer(out_feat)
        return out_feat


class MultiStageModel(nn.Module):

    def __init__(self, args, num_classes):
        super(MultiStageModel, self).__init__()
        num_stages = args.num_stages
        num_layers = args.num_layers
        num_f_maps = args.num_f_maps
        dim_in = args.features_dim
        method_centroid = args.method_centroid
        self.use_target = args.use_target
        self.multi_adv = args.multi_adv
        self.DA_adv_video = args.DA_adv_video
        self.ps_lb = args.ps_lb
        self.use_attn = args.use_attn
        self.num_seg = args.num_seg
        self.pair_ssl = args.pair_ssl
        self.DA_ens = args.DA_ens
        self.SS_video = args.SS_video
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim_in, num_classes, self.DA_ens)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, self.DA_ens)) for s in range(num_stages - 1)])
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps, num_f_maps, 'frame', args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]
        if 'rev_grad' in self.DA_adv_video and self.use_target != 'none':
            num_domain_class = 2
            num_concat = 1
            if 'rev_grad_ssl' in self.DA_adv_video:
                num_domain_class = int(math.factorial(self.num_seg * 2) / math.factorial(self.num_seg) ** 2)
                num_concat = self.num_seg * 2
                if self.DA_adv_video == 'rev_grad_ssl_2':
                    if self.pair_ssl == 'all':
                        num_concat = int(math.factorial(self.num_seg * 2) / (2 * math.factorial(self.num_seg * 2 - 2)))
                    elif self.pair_ssl == 'adjacent':
                        num_concat = self.num_seg * 2 - 1
            self.ad_net_video_base = nn.ModuleList()
            self.ad_net_video_base += [AdvDomainClsBase(num_f_maps * num_concat, num_f_maps, 'video', args)]
            self.ad_net_video_cls = nn.ModuleList()
            self.ad_net_video_cls += [nn.Linear(num_f_maps, num_domain_class)]
        if self.SS_video == 'VCOP':
            num_order_pair = int(self.num_seg * (self.num_seg - 1) / 2)
            num_order_class = math.factorial(self.num_seg)
            self.video_order_base = nn.Sequential(nn.Linear(num_f_maps * 2, num_f_maps), nn.ReLU(), nn.Dropout())
            self.video_order_cls = nn.Linear(num_f_maps * num_order_pair, num_order_class)
        if self.multi_adv[1] == 'Y':
            for i in range(1, num_classes):
                self.ad_net_cls += [nn.Linear(num_f_maps, 2)]
            if self.multi_adv[0] == 'Y':
                for i in range(1, num_classes):
                    self.ad_net_base += [AdvDomainClsBase(num_f_maps, num_f_maps, 'frame', args)]
        if method_centroid != 'none':
            self.centroids = nn.ModuleList()
            for s in range(num_stages):
                self.centroids += [Centroid(num_f_maps, num_classes)]

    def forward(self, x_s, x_t, mask_s, mask_t, beta, reverse):
        pred_source, prob_source, feat_source, feat_source_video, pred_d_source, pred_d_source_video, label_d_source, label_d_video_source, pred_source_2, prob_source_2 = self.forward_domain(x_s, mask_s, 0, beta, reverse)
        pred_target, prob_target, feat_target, feat_target_video, pred_d_target, pred_d_target_video, label_d_target, label_d_video_target, pred_target_2, prob_target_2 = self.forward_domain(x_t, mask_t, 1, beta, reverse)
        pred_d = torch.cat((pred_d_source, pred_d_target), 0)
        label_d = torch.cat((label_d_source, label_d_target), 0).long()
        pred_d_video = torch.cat((pred_d_source_video, pred_d_target_video), 0)
        label_d_video = torch.cat((label_d_video_source, label_d_video_target), 0).long()
        if 'rev_grad_ssl' in self.DA_adv_video and self.use_target != 'none':
            label_d_all = ([0] + [1]) * self.num_seg
            list_label_d = torch.tensor(list(set(permutations(label_d_all))))
            if label_d_video.get_device() >= 0:
                list_label_d = list_label_d
            feat_video = torch.cat((feat_source_video, feat_target_video), -1)
            num_batch = feat_video.size(0)
            pred_d_video_ssl_single, label_d_video_ssl_single = self.predict_domain_video_ssl(feat_video[0], list_label_d, beta[1])
            pred_d_video_ssl = pred_d_video_ssl_single.unsqueeze(0)
            label_d_video_ssl = label_d_video_ssl_single.unsqueeze(0)
            for i in range(1, num_batch):
                pred_d_video_ssl_single, label_d_video_ssl_single = self.predict_domain_video_ssl(feat_video[i], list_label_d, beta[1])
                pred_d_video_ssl = torch.cat((pred_d_video_ssl, pred_d_video_ssl_single.unsqueeze(0)), dim=0)
                label_d_video_ssl = torch.cat((label_d_video_ssl, label_d_video_ssl_single.unsqueeze(0)), dim=0)
            pred_d_video = pred_d_video_ssl
            label_d_video = label_d_video_ssl
        if self.SS_video == 'VCOP' and self.use_target != 'none':
            label_order_all = list(range(self.num_seg))
            list_label_order = torch.tensor(list(permutations(label_order_all)))
            if label_d_video.get_device() >= 0:
                list_label_order = list_label_order
            feat_video = torch.cat((feat_source_video, feat_target_video), 0)
            num_batch = feat_video.size(0)
            pred_order_video_ssl_single, label_order_video_ssl_single = self.predict_order_video_ssl(feat_video[0], list_label_order)
            pred_order_video_ssl = pred_order_video_ssl_single.unsqueeze(0)
            label_order_video_ssl = label_order_video_ssl_single.unsqueeze(0)
            for i in range(1, num_batch):
                pred_order_video_ssl_single, label_order_video_ssl_single = self.predict_order_video_ssl(feat_video[i], list_label_order)
                pred_order_video_ssl = torch.cat((pred_order_video_ssl, pred_order_video_ssl_single.unsqueeze(0)), dim=0)
                label_order_video_ssl = torch.cat((label_order_video_ssl, label_order_video_ssl_single.unsqueeze(0)), dim=0)
            pred_d_video = pred_order_video_ssl
            label_d_video = label_order_video_ssl
        return pred_source, prob_source, feat_source, pred_target, prob_target, feat_target, pred_d, pred_d_video, label_d, label_d_video, pred_source_2, prob_source_2, pred_target_2, prob_target_2

    def forward_domain(self, x, mask, domain_GT, beta, reverse):
        out_feat = self.stage1(x)
        if reverse:
            out_feat = GradRevLayer.apply(out_feat, beta[0])
        out = self.stage1.conv_out(out_feat)
        out_2 = out.clone()
        if self.DA_ens != 'none':
            out_2 = self.stage1.conv_out_2(out_feat)
        prob = F.softmax(out, dim=1)
        prob_2 = F.softmax(out_2, dim=1)
        out_d, out_d_video, lb_d, lb_d_video, out_feat_video = self.forward_stage(out_feat, prob, beta, mask, domain_GT)
        outputs_feat = out_feat.unsqueeze(1)
        outputs_feat_video = out_feat_video.unsqueeze(1)
        outputs = out.unsqueeze(1)
        probs = prob.unsqueeze(1)
        outputs_2 = out_2.unsqueeze(1)
        probs_2 = prob_2.unsqueeze(1)
        outputs_d = out_d.unsqueeze(1)
        outputs_d_video = out_d_video.unsqueeze(1)
        labels_d = lb_d.unsqueeze(1)
        labels_d_video = lb_d_video.unsqueeze(1)
        for s in self.stages:
            out_feat = s(prob)
            if reverse:
                out_feat = GradRevLayer.apply(out_feat, beta[0])
            out = s.conv_out(out_feat)
            out_2 = out.clone()
            if self.DA_ens != 'none':
                out_2 = s.conv_out_2(out_feat)
            prob = F.softmax(out, dim=1)
            prob_2 = F.softmax(out_2, dim=1)
            out_d, out_d_video, lb_d, lb_d_video, out_feat_video = self.forward_stage(out_feat, prob, beta, mask, domain_GT)
            outputs_feat = torch.cat((outputs_feat, out_feat.unsqueeze(1)), dim=1)
            outputs_feat_video = torch.cat((outputs_feat_video, out_feat_video.unsqueeze(1)), dim=1)
            outputs = torch.cat((outputs, out.unsqueeze(1)), dim=1)
            probs = torch.cat((probs, prob.unsqueeze(1)), dim=1)
            outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(1)), dim=1)
            probs_2 = torch.cat((probs_2, prob_2.unsqueeze(1)), dim=1)
            outputs_d = torch.cat((outputs_d, out_d.unsqueeze(1)), dim=1)
            outputs_d_video = torch.cat((outputs_d_video, out_d_video.unsqueeze(1)), dim=1)
            labels_d = torch.cat((labels_d, lb_d.unsqueeze(1)), dim=1)
            labels_d_video = torch.cat((labels_d_video, lb_d_video.unsqueeze(1)), dim=1)
        return outputs, probs, outputs_feat, outputs_feat_video, outputs_d, outputs_d_video, labels_d, labels_d_video, outputs_2, probs_2

    def forward_stage(self, out_feat, prob, beta, mask, domain_GT):
        out_d = self.predict_domain_frame(out_feat, beta[0])
        out_feat_video = out_feat
        if self.use_attn == 'domain_attn' and self.use_target != 'none':
            out_feat_video = self.apply_attn_feat_frame(out_feat_video, out_d, prob, 'ver2')
        out_feat_video = self.aggregate_frames(out_feat_video, mask)
        out_d_video = out_d[:, :, :, :self.num_seg].mean(1)
        if self.DA_adv_video == 'rev_grad' and self.use_target != 'none':
            out_d_video = self.predict_domain_video(out_feat_video, beta[1])
        out_d, out_d_video, lb_d, lb_d_video = self.select_masked(out_d, mask, out_d_video, domain_GT)
        return out_d, out_d_video, lb_d, lb_d_video, out_feat_video

    def predict_domain_frame(self, feat, beta_value):
        dim_feat = feat.size(1)
        num_frame = feat.size(2)
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)
        out = self.ad_net_cls[0](self.ad_net_base[0](feat, beta_value))
        out = out.reshape(-1, num_frame, 2).transpose(1, 2)
        out = out.unsqueeze(1)
        if self.multi_adv[1] == 'Y':
            for i in range(1, len(self.ad_net_cls)):
                id_base = i if self.multi_adv[0] == 'Y' else 0
                out_single_class = self.ad_net_cls[i](self.ad_net_base[id_base](feat, beta_value))
                out_single_class = out_single_class.reshape(-1, num_frame, 2).transpose(1, 2)
                out = torch.cat((out, out_single_class.unsqueeze(1)), dim=1)
        return out

    def get_domain_attn(self, pred_domain, type_attn):
        dim_pred = 2
        softmax = nn.Softmax(dim=dim_pred)
        logsoftmax = nn.LogSoftmax(dim=dim_pred)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), dim_pred)
        if type_attn == 'ver1':
            weights = entropy
        elif type_attn == 'ver2':
            weights = 1 - entropy
        return weights

    def apply_attn_feat_frame(self, feat, pred_domain, prob, type_attn):
        weights_attn = self.get_domain_attn(pred_domain, type_attn)
        if self.multi_adv[1] == 'Y':
            classweight = prob.detach()
            classweight_hard = classweight == classweight.max(dim=1, keepdim=True)[0]
            classweight_hard = classweight_hard.float()
            if self.ps_lb == 'soft':
                weights_attn *= classweight
            elif self.ps_lb == 'hard':
                weights_attn *= classweight_hard
        weights_attn = weights_attn.unsqueeze(2).repeat(1, 1, feat.size(1), 1)
        feat_expand = feat.unsqueeze(1).repeat(1, weights_attn.size(1), 1, 1)
        feat_attn = ((weights_attn + 1) * feat_expand).sum(1)
        return feat_attn

    def aggregate_frames(self, out_feat, mask):
        dim_feat = out_feat.size(1)
        num_batch = out_feat.size(0)
        num_total_frame = mask[:, 0, :].sum(-1)
        num_frame_seg = (num_total_frame / self.num_seg).int()
        num_frame_new = self.num_seg * num_frame_seg
        out_feat_video_batch = out_feat[0, :, :num_frame_new[0]].reshape(dim_feat, self.num_seg, num_frame_seg[0])
        out_feat_video_batch = out_feat_video_batch.sum(-1) / num_frame_seg[0]
        out_feat_video = out_feat_video_batch.unsqueeze(0)
        for b in range(1, num_batch):
            out_feat_video_batch = out_feat[b, :, :num_frame_new[b]].reshape(dim_feat, self.num_seg, num_frame_seg[b])
            out_feat_video_batch = out_feat_video_batch.sum(-1) / num_frame_seg[b].float()
            out_feat_video = torch.cat((out_feat_video, out_feat_video_batch.unsqueeze(0)), dim=0)
        return out_feat_video

    def predict_domain_video(self, feat, beta_value):
        dim_feat = feat.size(1)
        num_seg = feat.size(2)
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)
        out = self.ad_net_video_cls[0](self.ad_net_video_base[0](feat, beta_value))
        out = out.reshape(-1, num_seg, 2).transpose(1, 2)
        return out

    def select_masked(self, out_d, mask, out_d_video, domain_GT):
        num_class_domain = out_d.size(1)
        out_d = out_d.transpose(2, 3).transpose(1, 2).reshape(-1, num_class_domain, 2)
        mask_frame = mask[:, 0, :].reshape(-1)
        mask_frame = mask_frame > 0
        out_d = out_d[mask_frame]
        lb_d = torch.full_like(out_d[:, :, 0], domain_GT)
        out_d_video = out_d_video.transpose(1, 2).reshape(-1, 2)
        lb_d_video = torch.full_like(out_d_video[:, 0], domain_GT)
        return out_d, out_d_video, lb_d, lb_d_video

    def predict_domain_video_ssl(self, feat, list_label_d_seg, beta_value):
        num_stage = feat.size(0)
        dim_feat = feat.size(1)
        num_seg = feat.size(2)
        id_new = torch.randperm(num_seg)
        feat = feat[:, :, id_new]
        label_d_seg = (id_new >= num_seg / 2).long()
        if feat.get_device() >= 0:
            label_d_seg = label_d_seg
        label_d_onehot = (list_label_d_seg == label_d_seg).sum(-1) == num_seg
        label_d = label_d_onehot.nonzero()
        label_d = label_d.reshape(-1).repeat(num_stage)
        feat = feat.transpose(1, 2).reshape(-1, num_seg * dim_feat)
        out = self.ad_net_video_cls[0](self.ad_net_video_base[0](feat, beta_value))
        return out, label_d

    def predict_order_video_ssl(self, feat, list_label_order_seg):
        num_stage = feat.size(0)
        num_seg = feat.size(2)
        id_new = torch.randperm(num_seg)
        feat = feat[:, :, id_new]
        label_order_seg = id_new.long()
        if feat.get_device() >= 0:
            label_order_seg = label_order_seg
        label_order_onehot = (list_label_order_seg == label_order_seg).sum(-1) == num_seg
        label_order = label_order_onehot.nonzero()
        label_order = label_order.reshape(-1).repeat(num_stage)
        feat = feat.transpose(1, 2).transpose(0, 1)
        feat_pair = []
        for i in range(num_seg):
            for j in range(i + 1, num_seg):
                feat_pair.append(torch.cat((feat[i], feat[j]), -1))
        feat_pair = [self.video_order_base(i) for i in feat_pair]
        feat_concat = torch.cat(feat_pair, dim=1)
        out = self.video_order_cls(feat_concat)
        return out, label_order


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdvDomainClsBase,
     lambda: ([], {'in_feat': 4, 'hidden_size': 4, 'type_adv': 4, 'args': _mock_config(num_f_maps=4, DA_adv_video=4, pair_ssl=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Centroid,
     lambda: ([], {'num_f_maps': 4, 'num_classes': 4}),
     lambda: ([], {}),
     True),
    (DilatedResidualLayer,
     lambda: ([], {'dilation': 1, 'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (SingleStageModel,
     lambda: ([], {'num_layers': 1, 'num_f_maps': 4, 'dim_in': 4, 'num_classes': 4, 'DA_ens': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_cmhungsteve_SSTDA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

