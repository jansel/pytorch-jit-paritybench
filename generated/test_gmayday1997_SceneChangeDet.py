import sys
_module = sys.modules[__name__]
del sys
code = _module
CD2014config = _module
CMUconfig = _module
GSVconfig = _module
TSUNAMIconfig = _module
cfgs = _module
CD2014 = _module
CMU = _module
GSV = _module
TSUNAMI = _module
dataset = _module
layer = _module
function = _module
loss = _module
model = _module
deeplab_msc_coco = _module
siameseNet = _module
deeplab_v2 = _module
fcn32s_tiny = _module
vgg1024 = _module
train = _module
utils = _module
metric = _module
plot_contrast_sensitive = _module
transforms = _module
tsne_visual = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import torch.nn.init as init


import torch.utils.data as Data


from torch.nn import functional as F


class FeatureCorrelation(nn.Module):

    def __init__(self, scale):
        super(FeatureCorrelation, self).__init__()
        self.scale = scale

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = self.scale * feature_mul.view(b, h, w, h * w
            ).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class l2normalization(nn.Module):

    def __init__(self, scale):
        super(l2normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        """out = scale * x / sqrt(\\sum x_i^2)"""
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt(
            ).expand_as(x)


class l1normalization(nn.Module):

    def __init__(self, scale):
        super(l1normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        return self.scale * x * x.pow(1).sum(dim).clamp(min=1e-12).rsqrt(
            ).expand_as(x)


class scale_feature(nn.Module):

    def __init__(self, scale):
        super(scale_feature, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


class Mahalanobis_Distance(nn.Module):

    def __init__(self):
        super(Mahalanobis_Distance, self).__init__()

    def cal_con(self):
        pass

    def cal_invert_matrix(self):
        pass

    def forward(self, x1, x2):
        dis_abs = x1 - x2


class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self, hingethresh=0.0, margin=2.0):
        super(ConstractiveThresholdHingeLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin

    def forward(self, out_vec_t0, out_vec_t1, label):
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        similar_pair = torch.clamp(distance - self.threshold, min=0.0)
        dissimilar_pair = torch.clamp(self.margin - distance, min=0.0)
        constractive_thresh_loss = torch.sum((1 - label) * torch.pow(
            similar_pair, 2) + label * torch.pow(dissimilar_pair, 2))
        return constractive_thresh_loss


class ConstractiveLoss(nn.Module):

    def __init__(self, margin=2.0, dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self, out_vec_t0, out_vec_t1):
        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
            distance = 1 - 2 * similarity / np.pi
        return distance

    def forward(self, out_vec_t0, out_vec_t1, label):
        distance = self.various_distance(out_vec_t0, out_vec_t1)
        constractive_loss = torch.sum((1 - label) * torch.pow(distance, 2) +
            label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return constractive_loss


class ConstractiveMaskLoss(nn.Module):

    def __init__(self, thresh_flag=False, hinge_thresh=0.0, dist_flag='l2'):
        super(ConstractiveMaskLoss, self).__init__()
        if thresh_flag:
            self.sample_constractive_loss = ConstractiveThresholdHingeLoss(
                margin=2.0, hingethresh=hinge_thresh)
        else:
            self.sample_constractive_loss = ConstractiveLoss(margin=2.0,
                dist_flag=dist_flag)

    def forward(self, out_t0, out_t1, ground_truth):
        n, c, h, w = out_t0.data.shape
        out_t0_rz = torch.transpose(out_t0.view(c, h * w), 1, 0)
        out_t1_rz = torch.transpose(out_t1.view(c, h * w), 1, 0)
        gt_tensor = torch.from_numpy(np.array(ground_truth.data.cpu().numpy
            (), np.float32))
        gt_rz = Variable(torch.transpose(gt_tensor.view(1, h * w), 1, 0))
        loss = self.sample_constractive_loss(out_t0_rz, out_t1_rz, gt_rz)
        return loss


class LogDetDivergence(nn.Module):

    def __init__(self, model, param_name, dim=512):
        super(LogDetDivergence, self).__init__()
        self.param_name = param_name
        self.param_dict = dict(model.named_parameters())
        self.dim = dim
        self.identity_matrix = Variable(torch.from_numpy(np.identity(self.
            dim)).float())

    def select_param(self):
        for layer_name, layer_param in self.param_dict.items():
            if self.param_name in layer_name:
                if 'weight' in layer_name:
                    return layer_param

    def forward(self):
        constrainted_matrix = self.select_param()
        matrix_ = torch.squeeze(torch.squeeze(constrainted_matrix, dim=2),
            dim=2)
        matrix_t = torch.t(matrix_)
        matrixs = torch.mm(matrix_t, matrix_)
        trace_ = torch.trace(torch.mm(matrixs, torch.inverse(matrixs)))
        log_det = torch.logdet(matrixs)
        maha_loss = trace_ - log_det
        return maha_loss


class Mahalanobis_Constraint(nn.Module):

    def __init__(self, model, param_name, dim=512):
        super(Mahalanobis_Constraint, self).__init__()
        self.param_name = param_name
        self.param_dict = dict(model.named_parameters())
        self.dim = dim
        self.identity_matrix = Variable(torch.from_numpy(np.identity(self.
            dim)).float())

    def select_param(self):
        for layer_name, layer_param in self.param_dict.items():
            if self.param_name in layer_name:
                if 'weight' in layer_name:
                    return layer_param

    def forward(self):
        constrainted_matrix = self.select_param()
        matrxi_ = torch.squeeze(torch.squeeze(constrainted_matrix, dim=2),
            dim=2)
        matrxi_t = torch.t(matrxi_)
        matrxi_contrainted = (torch.mm(matrxi_t, matrxi_) - self.
            identity_matrix).view(self.dim ** 2)
        regularizer = torch.pow(matrxi_contrainted, 2).sum(dim=0)
        return regularizer


class SampleHistogramLoss(nn.Module):

    def __init__(self, num_steps):
        super(SampleHistogramLoss, self).__init__()
        self.step = 1.0 / (num_steps - 1)
        self.t = torch.range(0, 1, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]

    def forward(self, feat_t0, feat_t1, label):

        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (delta_repeat == self.t - self.step) & inds
            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.step)[indsa
                ] / self.step
            s_repeat_[indsb] = (-s_repeat_ + Variable(self.t) + self.step)[
                indsb] / self.step
            return s_repeat_.sum(1) / size


class BhattacharyyaDistance(nn.Module):

    def __init__(self):
        super(BhattacharyyaDistance, self).__init__()

    def forward(self, hist1, hist2):
        bh_dist = torch.sqrt(hist1 * hist2).sum()
        return bh_dist


class KLCoefficient(nn.Module):

    def __init__(self):
        super(KLCoefficient, self).__init__()

    def forward(self, hist1, hist2):
        kl = F.kl_div(hist1, hist2)
        dist = 1.0 / 1 + kl
        return dist


class HistogramMaskLoss(nn.Module):

    def __init__(self, num_steps, dist_flag='l2'):
        super(HistogramMaskLoss, self).__init__()
        self.step = 1.0 / (num_steps - 1)
        self.t = torch.range(0, 1, self.step).view(-1, 1)
        self.dist_flag = dist_flag
        self.distance = KLCoefficient()

    def various_distance(self, out_vec_t0, out_vec_t1):
        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
            distance = 1 - 2 * similarity / np.pi
        return distance

    def histogram(self):
        pass

    def forward(self, feat_t0, feat_t1, ground_truth):
        n, c, h, w = feat_t0.data.shape
        out_t0_rz = torch.transpose(feat_t0.view(c, h * w), 1, 0)
        out_t1_rz = torch.transpose(feat_t1.view(c, h * w), 1, 0)
        gt_np = ground_truth.view(h * w).data.cpu().numpy()
        pos_inds_np, neg_inds_np = np.squeeze(np.where(gt_np == 0), 1
            ), np.squeeze(np.where(gt_np != 0), 1)
        pos_size, neg_size = pos_inds_np.shape[0], neg_inds_np.shape[0]
        pos_inds, neg_inds = torch.from_numpy(pos_inds_np), torch.from_numpy(
            neg_inds_np)
        distance = torch.squeeze(self.various_distance(out_t0_rz, out_t1_rz
            ), dim=1)
        pos_dist_ls, neg_dist_ls = distance[pos_inds], distance[neg_inds]
        pos_dist_ls_t, neg_dist_ls_t = torch.from_numpy(pos_dist_ls.data.
            cpu().numpy()), torch.from_numpy(neg_dist_ls.data.cpu().numpy())
        hist_pos = Variable(torch.histc(pos_dist_ls_t, bins=100, min=0, max
            =1) / pos_size, requires_grad=True)
        hist_neg = Variable(torch.histc(neg_dist_ls_t, bins=100, min=0, max
            =1) / neg_size, requires_grad=True)
        loss = self.distance(hist_pos, hist_neg)
        return loss


class deeplab(nn.Module):

    def __init__(self):
        super(deeplab, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1, ceil_mode=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
            128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
            stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=
            256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
            stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1, ceil_mode=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=
            512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
            stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=1, padding=1, ceil_mode=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            512, kernel_size=3, dilation=2, stride=1, padding=2), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, dilation=2, stride=1, padding=2), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, dilation=2, stride=1, padding=2), nn.ReLU(
            inplace=True), nn.MaxPool2d(kernel_size=3, stride=1, padding=1,
            ceil_mode=True), nn.AvgPool2d(kernel_size=3, stride=1, padding=
            1, ceil_mode=True))
        self.fc6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            1024, kernel_size=3, dilation=12, padding=12), nn.ReLU(inplace=
            True), nn.Dropout2d())
        self.fc7 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=
            1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d())

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        fc7_features = self.fc7(x)
        return fc7_features


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


class Deeplab_MS_Att_Scale(nn.Module):

    def __init__(self, class_number=21):
        super(Deeplab_MS_Att_Scale, self).__init__()
        self.truck_branch = deeplab()
        self.scale_attention_branch = nn.Sequential(nn.Conv2d(in_channels=
            1024 * 3, out_channels=512, kernel_size=3, padding=1), nn.ReLU(
            inplace=True), nn.Dropout2d(), nn.Conv2d(in_channels=512,
            out_channels=3, kernel_size=1))
        self.fc8 = nn.Conv2d(in_channels=1024, out_channels=class_number,
            kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size * 0.75) + 1, int(
            input_size * 0.75) + 1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size * 0.5) + 1, int(
            input_size * 0.5) + 1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)
            ), mode='bilinear')
        out = []
        x75 = self.interp1(x)
        x50 = self.interp2(x)
        fc7_x = self.truck_branch(x)
        fc7_x75 = self.truck_branch(x75)
        fc7_x50 = self.truck_branch(x50)
        out.append(fc7_x)
        out.append(self.interp3(fc7_x75))
        out.append(self.interp3(fc7_x50))
        out_cat = torch.cat(out, dim=1)
        scale_att_mask = F.softmax(self.scale_attention_branch(out_cat))
        score_x = self.fc8(fc7_x)
        score_x50 = self.interp3(self.fc8(fc7_x50))
        score_x75 = self.interp3(self.fc8(fc7_x75))
        assert score_x.size() == score_x50.size()
        score_att_x = torch.mul(score_x, scale_att_mask[:, (0), :, :].
            expand_as(score_x))
        score_att_x_075 = torch.mul(score_x75, scale_att_mask[:, (1), :, :]
            .expand_as(score_x75))
        score_att_x_050 = torch.mul(score_x50, scale_att_mask[:, (2), :, :]
            .expand_as(score_x50))
        score_final = score_att_x + score_att_x_075 + score_att_x_050
        return score_final, scale_att_mask

    def init_parameters_from_deeplab(self, pretrain_vgg16_1024):
        conv_blocks = [self.truck_branch.conv1, self.truck_branch.conv2,
            self.truck_branch.conv3, self.truck_branch.conv4, self.
            truck_branch.conv5]
        pretrain_conv_blocks = [pretrain_vgg16_1024.truck_branch.conv1,
            pretrain_vgg16_1024.truck_branch.conv2, pretrain_vgg16_1024.
            truck_branch.conv3, pretrain_vgg16_1024.truck_branch.conv4,
            pretrain_vgg16_1024.truck_branch.conv5]
        for idx, (conv_block, pretrain_conv_block) in enumerate(zip(
            conv_blocks, pretrain_conv_blocks)):
            for l1, l2 in zip(pretrain_conv_block, conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        self.truck_branch.fc6[0
            ].weight.data = pretrain_vgg16_1024.truck_branch.fc6[0
            ].weight.data.view(self.truck_branch.fc6[0].weight.size())
        self.truck_branch.fc6[0
            ].bias.data = pretrain_vgg16_1024.truck_branch.fc6[0
            ].bias.data.view(self.truck_branch.fc6[0].bias.size())
        self.truck_branch.fc7[0
            ].weight.data = pretrain_vgg16_1024.truck_branch.fc7[0
            ].weight.data.view(self.truck_branch.fc7[0].weight.size())
        self.truck_branch.fc7[0
            ].bias.data = pretrain_vgg16_1024.truck_branch.fc7[0
            ].bias.data.view(self.truck_branch.fc7[0].bias.size())
        self.scale_attention_branch[0
            ].weight.data = pretrain_vgg16_1024.scale_attention_branch[0
            ].weight.data
        self.scale_attention_branch[0
            ].bias.data = pretrain_vgg16_1024.scale_attention_branch[0
            ].bias.data
        self.scale_attention_branch[3
            ].weight.data = pretrain_vgg16_1024.scale_attention_branch[3
            ].weight.data
        self.scale_attention_branch[3
            ].bias.data = pretrain_vgg16_1024.scale_attention_branch[3
            ].bias.data
        self.fc8.weight.data.normal_(0, 0.01)
        self.fc8.bias.data.fill_(0)


class deeplab_V2(nn.Module):

    def __init__(self):
        super(deeplab_V2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1, ceil_mode=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=128, out_channels=128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1, ceil_mode=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=
            256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=256, out_channels=256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256,
            out_channels=256, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=
            512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=512, out_channels=512, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=512,
            out_channels=512, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=3, stride=1, padding=1,
            ceil_mode=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=
            True), nn.Conv2d(in_channels=512, out_channels=512, kernel_size
            =3, dilation=2, padding=2), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, dilation=2,
            padding=2), nn.ReLU(inplace=True))
        self.fc6_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            1024, kernel_size=3, dilation=6, padding=6), nn.ReLU(inplace=
            True), nn.Dropout2d(p=0.5))
        self.fc7_1 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels
            =1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.fc6_2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            1024, kernel_size=3, dilation=12, padding=12), nn.ReLU(inplace=
            True), nn.Dropout2d(p=0.5))
        self.fc7_2 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels
            =1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.fc6_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            1024, kernel_size=3, dilation=18, padding=18), nn.ReLU(inplace=
            True), nn.Dropout2d(p=0.5))
        self.fc7_3 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels
            =1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.fc6_4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            1024, kernel_size=3, dilation=24, padding=24), nn.ReLU(inplace=
            True), nn.Dropout2d(p=0.5))
        self.fc7_4 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels
            =1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.embedding_layer = nn.Conv2d(in_channels=1024, out_channels=512,
            kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        conv3_feature = self.conv3(x)
        conv4_feature = self.conv4(conv3_feature)
        conv5_feature = self.conv5(conv4_feature)
        fc6_1 = self.fc6_1(conv5_feature)
        fc7_1 = self.fc7_1(fc6_1)
        fc6_2 = self.fc6_2(conv5_feature)
        fc7_2 = self.fc7_2(fc6_2)
        fc6_3 = self.fc6_3(conv5_feature)
        fc7_3 = self.fc7_3(fc6_3)
        fc6_4 = self.fc6_4(conv5_feature)
        fc7_4 = self.fc7_4(fc6_4)
        fc_feature = fc7_1 + fc7_2 + fc7_3 + fc7_4
        embedding_feature = self.embedding_layer(fc_feature)
        return conv5_feature, fc_feature, embedding_feature


def convert_dict_names_for_fucking_faults():
    deeplab_v2_dict_names_mapping = {'conv1.0': 'conv1_1', 'conv1.2':
        'conv1_2', 'conv2.0': 'conv2_1', 'conv2.2': 'conv2_2', 'conv3.0':
        'conv3_1', 'conv3.2': 'conv3_2', 'conv3.4': 'conv3_3', 'conv4.0':
        'conv4_1', 'conv4.2': 'conv4_2', 'conv4.4': 'conv4_3', 'conv5.0':
        'conv5_1', 'conv5.2': 'conv5_2', 'conv5.4': 'conv5_3'}
    return deeplab_v2_dict_names_mapping


class SiameseNet(nn.Module):

    def __init__(self, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.CNN = deeplab_V2()
        if norm_flag == 'l2':
            self.norm = fun.l2normalization(scale=1)
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()
    """
    def forward(self,t0,t1):

        out_t0_embedding = self.CNN(t0)
        out_t1_embedding = self.CNN(t1)
        #out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        #out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding),self.norm(out_t1_embedding)
        return [out_t0_embedding_norm,out_t1_embedding_norm]
    """

    def forward(self, t0, t1):
        out_t0_conv5, out_t0_fc7, out_t0_embedding = self.CNN(t0)
        out_t1_conv5, out_t1_fc7, out_t1_embedding = self.CNN(t1)
        out_t0_conv5_norm, out_t1_conv5_norm = self.norm(out_t0_conv5
            ), self.norm(out_t1_conv5)
        out_t0_fc7_norm, out_t1_fc7_norm = self.norm(out_t0_fc7), self.norm(
            out_t1_fc7)
        out_t0_embedding_norm, out_t1_embedding_norm = self.norm(
            out_t0_embedding), self.norm(out_t1_embedding)
        return [out_t0_conv5_norm, out_t1_conv5_norm], [out_t0_fc7_norm,
            out_t1_fc7_norm], [out_t0_embedding_norm, out_t1_embedding_norm]
    """
    def forward(self,t0,t1):

        out_t0_conv4,out_t0_conv5,out_t0_fc7,out_t0_embedding = self.CNN(t0)
        out_t1_conv4,out_t1_conv5,out_t1_fc7,out_t1_embedding = self.CNN(t1)
        out_t0_conv4_norm,out_t1_conv4_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding),self.norm(out_t1_embedding)
        return [out_t0_conv4_norm,out_t1_conv4_norm],[out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm],[out_t0_embedding_norm,out_t1_embedding_norm]
    """
    """
    def forward(self,t0,t1):

        out_t0_conv4,out_t0_conv5,out_t0_fc7 = self.CNN(t0)
        out_t1_conv4,out_t1_conv5,out_t1_fc7 = self.CNN(t1)
        out_t0_conv4_norm,out_t1_conv4_norm = self.norm(out_t0_conv4),self.norm(out_t1_conv4)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        return [out_t0_conv4_norm,out_t1_conv4_norm],[out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm]
    """

    def init_parameters_from_deeplab(self, pretrain_vgg16_1024):
        pretrain_dict_names = convert_dict_names_for_fucking_faults()
        keys = sorted(pretrain_dict_names.keys())
        conv_blocks = [self.CNN.conv1, self.CNN.conv2, self.CNN.conv3, self
            .CNN.conv4, self.CNN.conv5]
        ranges = [[0, 2], [0, 2], [0, 2, 4], [0, 2, 4], [0, 2, 4]]
        for key in keys:
            dic_name = pretrain_dict_names[key]
            base_conv_name, conv_index, sub_index = dic_name[:5], int(dic_name
                [4]), int(dic_name[-1])
            conv_blocks[conv_index - 1][ranges[sub_index - 1][sub_index - 1]
                ].weight.data = pretrain_vgg16_1024[key + '.weight']
            conv_blocks[conv_index - 1][ranges[sub_index - 1][sub_index - 1]
                ].bias.data = pretrain_vgg16_1024[key + '.bias']
        self.CNN.fc6_1[0].weight.data = pretrain_vgg16_1024['fc6_1.0.weight'
            ].view(self.CNN.fc6_1[0].weight.size())
        self.CNN.fc6_1[0].bias.data = pretrain_vgg16_1024['fc6_1.0.bias'].view(
            self.CNN.fc6_1[0].bias.size())
        self.CNN.fc7_1[0].weight.data = pretrain_vgg16_1024['fc7_1.0.weight'
            ].view(self.CNN.fc7_1[0].weight.size())
        self.CNN.fc7_1[0].bias.data = pretrain_vgg16_1024['fc7_1.0.bias'].view(
            self.CNN.fc7_1[0].bias.size())
        self.CNN.fc6_2[0].weight.data = pretrain_vgg16_1024['fc6_2.0.weight'
            ].view(self.CNN.fc6_2[0].weight.size())
        self.CNN.fc6_2[0].bias.data = pretrain_vgg16_1024['fc6_2.0.bias'].view(
            self.CNN.fc6_2[0].bias.size())
        self.CNN.fc7_2[0].weight.data = pretrain_vgg16_1024['fc7_2.0.weight'
            ].view(self.CNN.fc7_2[0].weight.size())
        self.CNN.fc7_2[0].bias.data = pretrain_vgg16_1024['fc7_2.0.bias'].view(
            self.CNN.fc7_2[0].bias.size())
        self.CNN.fc6_3[0].weight.data = pretrain_vgg16_1024['fc6_3.0.weight'
            ].view(self.CNN.fc6_3[0].weight.size())
        self.CNN.fc6_3[0].bias.data = pretrain_vgg16_1024['fc6_3.0.bias'].view(
            self.CNN.fc6_3[0].bias.size())
        self.CNN.fc7_3[0].weight.data = pretrain_vgg16_1024['fc7_3.0.weight'
            ].view(self.CNN.fc7_3[0].weight.size())
        self.CNN.fc7_3[0].bias.data = pretrain_vgg16_1024['fc7_3.0.bias'].view(
            self.CNN.fc7_3[0].bias.size())
        self.CNN.fc6_4[0].weight.data = pretrain_vgg16_1024['fc6_4.0.weight'
            ].view(self.CNN.fc6_4[0].weight.size())
        self.CNN.fc6_4[0].bias.data = pretrain_vgg16_1024['fc6_4.0.bias'].view(
            self.CNN.fc6_4[0].bias.size())
        self.CNN.fc7_4[0].weight.data = pretrain_vgg16_1024['fc7_4.0.weight'
            ].view(self.CNN.fc7_4[0].weight.size())
        self.CNN.fc7_4[0].bias.data = pretrain_vgg16_1024['fc7_4.0.bias'].view(
            self.CNN.fc7_4[0].bias.size())

    def init_parameters(self, pretrain_vgg16_1024):
        conv_blocks = [self.CNN.conv1, self.CNN.conv2, self.CNN.conv3, self
            .CNN.conv4, self.CNN.conv5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16_1024.features.children())
        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]],
                conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        self.CNN.fc6[0].weight.data = pretrain_vgg16_1024.classifier[0
            ].weight.data.view(self.CNN.fc6[0].weight.size())
        self.CNN.fc6[0].bias.data = pretrain_vgg16_1024.classifier[0
            ].bias.data.view(self.CNN.fc6[0].bias.size())
        self.CNN.fc7[0].weight.data = pretrain_vgg16_1024.classifier[3
            ].weight.data.view(self.CNN.fc7[0].weight.size())
        self.CNN.fc7[0].bias.data = pretrain_vgg16_1024.classifier[3
            ].bias.data.view(self.CNN.fc7[0].bias.size())


class fcn32s(nn.Module):

    def __init__(self, distance_flag):
        super(fcn32s, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1, ceil_mode=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
            128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=128, out_channels=128, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1, ceil_mode=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=
            256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=256, out_channels=256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256,
            out_channels=256, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=
            512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=512, out_channels=512, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=512,
            out_channels=512, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=3, stride=1, padding=1,
            ceil_mode=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=
            512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=
            True), nn.Conv2d(in_channels=512, out_channels=512, kernel_size
            =3, dilation=2, padding=2), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, dilation=2,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=1, padding=1, ceil_mode=True))
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, dilation=2, padding=2)
        if distance_flag == 'softmax':
            self.fc8 = nn.Softmax2d()
        if distance_flag == 'l2':
            self.fc8 = fun.l2normalization(scale=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.embedding_layer(x)
        embedding_feature = self.fc8(x)
        return embedding_feature


class SiameseNet(nn.Module):

    def __init__(self, distance_flag='softmax'):
        super(SiameseNet, self).__init__()
        self.CNN = fcn32s(distance_flag)

    def forward(self, t0, t1):
        out_t0_conv5 = self.CNN(t0)
        out_t1_conv5 = self.CNN(t1)
        return [out_t0_conv5, out_t1_conv5]

    def init_parameters(self, pretrain_vgg16):
        conv_blocks = [self.CNN.conv1, self.CNN.conv2, self.CNN.conv3, self
            .CNN.conv4, self.CNN.conv5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16.features.children())
        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]],
                conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        init.kaiming_uniform(self.CNN.embedding_layer.weight.data, mode=
            'fan_in')
        init.constant(self.CNN.embedding_layer.bias.data, 0)
        """'
        self.fc6[0].weight.data = pretrain_vgg16.classifier[0].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16.classifier[0].bias.data.view(self.fc6[0].bias.size())
        self.fc7[0].weight.data = pretrain_vgg16.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16.classifier[3].bias.data.view(self.fc7[0].bias.size())

        ###### random init socore layer parameters ###########
        assert  self.upscore.kernel_size[0] == self.upscore.kernel_size[1]
        initial_weight = get_upsampling_weight(self.upscore.in_channels, self.upscore.out_channels, self.upscore.kernel_size[0])
        self.upscore.weight.data.copy_(initial_weight)
        """


_global_config['D'] = 4


class vgg1024(nn.Module):

    def __init__(self):
        super(vgg1024, self).__init__()
        self.features = self._make_layers(cfg['D'])
        self.classifier = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Conv2d(1024, 1024, 1), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Conv2d(1024, 21, 1))

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M' or v == 'A':
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.AvgPool2d(kernel_size=3, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def init_parameters(self, pretrain_vgg16_1024):
        conv_blocks = [self.conv1, self.conv2, self.conv3, self.conv4, self
            .conv5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16_1024.features.children())
        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]],
                conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        self.fc6[0].weight.data = pretrain_vgg16.classifier[0
            ].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16.classifier[0].bias.data.view(
            self.fc6[0].bias.size())
        self.fc7[0].weight.data = pretrain_vgg16.classifier[3
            ].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16.classifier[3].bias.data.view(
            self.fc7[0].bias.size())


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gmayday1997_SceneChangeDet(_paritybench_base):
    pass
    def test_000(self):
        self._check(FeatureCorrelation(*[], **{'scale': 1.0}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(l2normalization(*[], **{'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(l1normalization(*[], **{'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(scale_feature(*[], **{'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Mahalanobis_Distance(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(ConstractiveThresholdHingeLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ConstractiveLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(SampleHistogramLoss(*[], **{'num_steps': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(BhattacharyyaDistance(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(KLCoefficient(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(deeplab(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_011(self):
        self._check(deeplab_V2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_012(self):
        self._check(SiameseNet(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})

