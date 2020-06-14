import sys
_module = sys.modules[__name__]
del sys
demo = _module
eval = _module
core = _module
config = _module
evaluate = _module
loss = _module
trainer = _module
amass_utils = _module
feature_extractor = _module
img_utils = _module
insta_utils = _module
kp_utils = _module
mpii3d_utils = _module
penn_action_utils = _module
posetrack_utils = _module
threedpw_utils = _module
dataset = _module
amass = _module
dataset_2d = _module
dataset_3d = _module
inference = _module
insta = _module
loaders = _module
mpii3d = _module
penn_action = _module
posetrack = _module
threedpw = _module
models = _module
attention = _module
motion_discriminator = _module
resnet = _module
smpl = _module
spin = _module
vibe = _module
losses = _module
prior = _module
temporal_smplify = _module
utils = _module
demo_utils = _module
eval_utils = _module
geometry = _module
pose_tracker = _module
renderer = _module
smooth_bbox = _module
vis = _module
test_2d_datasets = _module
test_3d_datasets = _module
train = _module

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


from torch import nn


import torch.nn.functional as F


from torch.nn.utils import spectral_norm


import math


import numpy as np


import time


from torch.nn import functional as F


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((
        real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, (0)], norm_quat[:, (1)], norm_quat[:, (2)
        ], norm_quat[:, (3)]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 *
        xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 *
        wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class VIBELoss(nn.Module):

    def __init__(self, e_loss_weight=60.0, e_3d_loss_weight=30.0,
        e_pose_loss_weight=1.0, e_shape_loss_weight=0.001,
        d_motion_loss_weight=1.0, device='cuda'):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight
        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(self, generator_outputs, data_2d, data_3d, data_body_mosh=
        None, data_motion_mosh=None, body_discriminator=None,
        motion_discriminator=None):
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        flatten = lambda x: x.reshape(-1)
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in
            x], 0)
        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
        real_2d = reduce(real_2d)
        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])
        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)
        total_predict_thetas = accumulate_thetas(generator_outputs)
        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)
        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)
        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=
            1.0, gt_weight=1.0) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]
        loss_dict = {'loss_kp_2d': loss_kp_2d, 'loss_kp_3d': loss_kp_3d}
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape,
                real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose
        gen_loss = torch.stack(list(loss_dict.values())).sum()
        end_idx = 75
        start_idx = 6
        pred_motion = total_predict_thetas
        e_motion_disc_loss = self.enc_loss(motion_discriminator(pred_motion
            [:, :, start_idx:end_idx]))
        e_motion_disc_loss = e_motion_disc_loss * self.d_motion_loss_weight
        fake_motion = pred_motion.detach()
        real_motion = data_motion_mosh['theta']
        fake_disc_value = motion_discriminator(fake_motion[:, :, start_idx:
            end_idx])
        real_disc_value = motion_discriminator(real_motion[:, :, start_idx:
            end_idx])
        d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = (self.
            dec_loss(real_disc_value, fake_disc_value))
        d_motion_disc_real = d_motion_disc_real * self.d_motion_loss_weight
        d_motion_disc_fake = d_motion_disc_fake * self.d_motion_loss_weight
        d_motion_disc_loss = d_motion_disc_loss * self.d_motion_loss_weight
        loss_dict['e_m_disc_loss'] = e_motion_disc_loss
        loss_dict['d_m_disc_real'] = d_motion_disc_real
        loss_dict['d_m_disc_fake'] = d_motion_disc_fake
        loss_dict['d_m_disc_loss'] = d_motion_disc_loss
        gen_loss = gen_loss + e_motion_disc_loss
        motion_dis_loss = d_motion_disc_loss
        return gen_loss, motion_dis_loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d,
        openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, (-1)].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d,
            gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, (2), :] + gt_keypoints_3d[:, (3
                ), :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, (None), :]
            pred_pelvis = (pred_keypoints_3d[:, (2), :] + pred_keypoints_3d
                [:, (3), :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, (None), :]
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d
                ).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)
            ).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-
            1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid,
                gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid,
                gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.0).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.0).to(self.device)
        return loss_regr_pose, loss_regr_betas


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)


class SelfAttention(nn.Module):

    def __init__(self, attention_size, batch_first=False, layers=1, dropout
        =0.0, non_linearity='tanh'):
        super(SelfAttention, self).__init__()
        self.batch_first = batch_first
        if non_linearity == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))
        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, scores


class MotionDiscriminator(nn.Module):

    def __init__(self, rnn_size, input_size, num_layers, output_size=2,
        feature_pool='concat', use_spectral_norm=False, attention_size=1024,
        attention_layers=1, attention_dropout=0.5):
        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout
        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers
            )
        linear_size = (self.rnn_size if not feature_pool == 'concat' else 
            self.rnn_size * 2)
        if feature_pool == 'attention':
            self.attention = SelfAttention(attention_size=self.
                attention_size, layers=self.attention_layers, dropout=self.
                attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        outputs, state = self.gru(sequence)
        if self.feature_pool == 'concat':
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(
                batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(
                batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == 'attention':
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
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
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, replace_stride_with_dilation=
        None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
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


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow':
    19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 
    20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 
    26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46,
    'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19,
    'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18,
    'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51,
    'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow',
    'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe',
    'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee',
    'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)',
    'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)',
    'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye',
    'Left Ear', 'Right Ear']


VIBE_DATA_DIR = 'data/vibe_data'


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


H36M_TO_J14 = H36M_TO_J17[:14]


class TemporalEncoder(nn.Module):

    def __init__(self, n_layers=1, hidden_size=2048, add_linear=False,
        bidirectional=False, use_residual=True):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size,
            bidirectional=bidirectional, num_layers=n_layers)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2)
        return y


class SMPLifyAnglePrior(nn.Module):

    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32 if 
            dtype == torch.float32 else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        """ Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        """
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, (angle_prior_idxs)] * self.angle_prior_signs
            ).pow(2)


DEFAULT_DTYPE = torch.float32


class L2Prior(nn.Module):

    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior', num_gaussians=6, dtype=
        DEFAULT_DTYPE, epsilon=1e-16, use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()
        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            None
            sys.exit(-1)
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)
        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            None
            sys.exit(-1)
        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')
        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            None
            sys.exit(-1)
        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))
        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=
            dtype))
        sqrdets = np.array([np.sqrt(np.linalg.det(c)) for c in gmm['covars']])
        const = (2 * np.pi) ** (69 / 2.0)
        nll_weights = np.asarray(gmm['weights'] / (const * (sqrdets /
            sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)
        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)
        self.register_buffer('pi_term', torch.log(torch.tensor(2 * np.pi,
            dtype=dtype)))
        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon) for
            cov in covs]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        """ Returns the mean of the mixture """
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means
        prec_diff_prod = torch.einsum('mij,bmj->bmi', [self.precisions,
            diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)
        curr_loglikelihood = 0.5 * diff_prec_quadratic - torch.log(self.
            nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        """ Create graph operation for negative log-likelihood calculation
        """
        likelihoods = []
        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean
            curr_loglikelihood = torch.einsum('bj,ji->bi', [diff_from_mean,
                prec])
            curr_loglikelihood = torch.einsum('bi,bi->b', [
                curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term + self.random_var_dim *
                self.pi_term)
            likelihoods.append(curr_loglikelihood)
        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, (min_idx)]
        weight_component = -torch.log(weight_component)
        return weight_component + log_likelihoods[:, (min_idx)]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mkocabas_VIBE(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(L2Prior(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MotionDiscriminator(*[], **{'rnn_size': 4, 'input_size': 4, 'num_layers': 1}), [torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(SelfAttention(*[], **{'attention_size': 4}), [torch.rand([4, 4, 4, 4])], {})

