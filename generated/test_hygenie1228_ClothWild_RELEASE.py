import sys
_module = sys.modules[__name__]
del sys
base = _module
logger = _module
layer = _module
loss = _module
module = _module
resnet = _module
timer = _module
SMPL = _module
SMPLicit = _module
SMPLicit_options = _module
network = _module
smplicit_core_test = _module
util_smpl = _module
utils = _module
sdf = _module
dir = _module
human_models = _module
postprocessing = _module
preprocessing = _module
transforms = _module
vis = _module
DeepFashion2 = _module
MSCOCO = _module
PW3D = _module
dataset = _module
demo = _module
config = _module
model = _module
test = _module
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


import math


import time


import abc


import numpy as np


from torch.utils.data import DataLoader


import torch.optim


import torchvision.transforms as transforms


from collections import OrderedDict


from torch.nn.parallel.data_parallel import DataParallel


import torch


import torch.nn as nn


from torch.nn import functional as F


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import model_urls


import copy


import torchvision


import torch.nn.functional as F


from torch.autograd import Variable


import random


from torch.utils.data.dataset import Dataset


import torch.backends.cudnn as cudnn


class ClothClsLoss(nn.Module):

    def __init__(self):
        super(ClothClsLoss, self).__init__()
        self.dp_parts = {'head': [22, 23], 'upperbody': [0, 1, 14, 15, 16, 17, 18, 19, 20, 21], 'lowerbody': [6, 7, 8, 9, 10, 11, 12, 13], 'foot': [4, 5]}
        self.part_clothes = {'head': ['hair'], 'upperbody': ['uppercloth', 'coat'], 'lowerbody': ['pants', 'skirts'], 'foot': ['shoes']}
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, out, patch_idx, cloth_idx):
        valid = torch.zeros_like(out)
        index_gt = torch.zeros_like(out)
        for part in self.dp_parts.keys():
            valid_one_part = torch.zeros((out.shape[0],))
            for part_idx in self.dp_parts[part]:
                valid_one_part += (patch_idx == part_idx).any(1)
            for cloth in self.part_clothes[part]:
                if cloth in cfg.cloth_types:
                    valid[valid_one_part > 0, cfg.cloth_types.index(cloth)] = 1
        for idx in range(len(cfg.cloth_types)):
            index_gt[:, idx] += (cloth_idx == idx + 1).any(1)
        loss = self.bce_loss(out, index_gt)
        loss = loss[valid > 0]
        return loss.mean()


class GenderClsLoss(nn.Module):

    def __init__(self):
        super(GenderClsLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, out, gt):
        valid = gt != 0
        gt = F.one_hot(gt.long(), num_classes=3)[:, 1:].float()
        loss = self.bce_loss(out, gt)
        loss = loss[valid]
        return loss.mean()


class SdfDPLoss(nn.Module):

    def __init__(self):
        super(SdfDPLoss, self).__init__()

    def forward(self, sdf, cloth_meshes_unposed, smpl_cloth_idx, smpl_cloth_valid, cloth_idx, sdf_thresh, dist_thresh, v_template):
        batch_size = sdf.shape[0]
        cloth_type = cfg.cloth_types[cloth_idx[0] - 1]
        loss_list = []
        for bid in range(batch_size):
            smpl_mask = smpl_cloth_valid[bid] > 0
            smpl_verts = v_template[bid][smpl_mask[:, None].repeat(1, 3)].view(-1, 3)
            cloth_verts = cloth_meshes_unposed[bid]
            if smpl_verts.shape[0] > 0:
                dists = torch.sqrt(torch.sum((smpl_verts[None, :, :] - cloth_verts[:, None, :]) ** 2, 2))
            else:
                loss_list.append(torch.zeros(1).mean().float())
                continue
            dists[dists < cfg.min_dist_thresh[cloth_type]] = 9999
            dists, query_point_idx = torch.min(dists, 1)
            target_cloth_idx = smpl_cloth_idx[bid][smpl_mask]
            target_cloth_idx = target_cloth_idx[query_point_idx]
            loss_pos = torch.abs(sdf[bid, :]) * (sum([(target_cloth_idx == idx) for idx in cloth_idx]) > 0) * (dists < dist_thresh)
            loss_neg = torch.abs(sdf[bid, :] - sdf_thresh) * (sum([(target_cloth_idx == idx) for idx in cloth_idx]) == 0) * (dists < dist_thresh)
            cloth_exist = (sum([(target_cloth_idx == idx) for idx in cloth_idx]) > 0).sum() > 0
            loss = (loss_pos + loss_neg).mean() * cloth_exist
            loss_list.append(loss)
        loss = torch.stack(loss_list)
        return loss


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='none')

    def forward(self, param, valid):
        zeros = torch.zeros_like(param)
        loss = self.l2_loss(param, zeros) * valid[:, None]
        return loss.mean()


class SdfParseLoss(nn.Module):

    def __init__(self):
        super(SdfParseLoss, self).__init__()

    def forward(self, sdf, cloth_meshes, parse_gt, sdf_thresh, cloth_meshes_unposed, parse_valid, dist_thresh, v_template):
        batch_size = sdf.shape[0]
        inf = 9999
        x, y = cloth_meshes[:, :, 0].long(), cloth_meshes[:, :, 1].long()
        idx = y * cfg.input_img_shape[1] + x
        is_valid = (x >= 0) * (x < cfg.input_img_shape[1]) * (y >= 0) * (y < cfg.input_img_shape[0])
        idx[is_valid == 0] = 0
        min_sdf = sdf * is_valid.float() + inf * (1 - is_valid.float())
        parse_out_min = torch.ones((batch_size, cfg.input_img_shape[0] * cfg.input_img_shape[1])).float() * inf
        max_sdf = sdf * is_valid.float() - inf * (1 - is_valid.float())
        parse_out_max = torch.ones((batch_size, cfg.input_img_shape[0] * cfg.input_img_shape[1])).float() * -inf
        try:
            parse_out_min, _ = scatter_min(min_sdf, idx, 1, parse_out_min)
            parse_out_max, _ = scatter_max(max_sdf, idx, 1, parse_out_max)
        except:
            idx = idx.cpu()
            min_sdf, max_sdf = min_sdf.cpu(), max_sdf.cpu()
            parse_out_min, parse_out_max = parse_out_min.cpu(), parse_out_max.cpu()
            parse_out_min, _ = scatter_min(min_sdf, idx, 1, parse_out_min)
            parse_out_max, _ = scatter_max(max_sdf, idx, 1, parse_out_max)
            parse_out_min, parse_out_max = parse_out_min, parse_out_max
        parse_out_min = parse_out_min.view(batch_size, cfg.input_img_shape[0], cfg.input_img_shape[1])
        parse_out_min[parse_out_min == inf] = 0
        parse_out_max = parse_out_max.view(batch_size, cfg.input_img_shape[0], cfg.input_img_shape[1])
        parse_out_max[parse_out_max == -inf] = sdf_thresh
        loss_pos = torch.abs(parse_out_min) * (parse_gt == 1) * parse_valid
        loss_neg = torch.abs(parse_out_max - sdf_thresh) * (parse_gt == 0) * parse_valid
        loss = loss_pos.mean((1, 2)) + loss_neg.mean((1, 2))
        cloth_exist = (parse_gt == 1).sum((1, 2)) > 0
        loss = loss * cloth_exist
        return loss


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ClothNet(nn.Module):

    def __init__(self):
        super(ClothNet, self).__init__()
        input_feat_dim = 2048
        if 'uppercloth' in cfg.cloth_types:
            self.z_cut_uppercloth = make_linear_layers([input_feat_dim, 6], relu_final=False)
            self.z_style_uppercloth = make_linear_layers([input_feat_dim, 12], relu_final=False)
        if 'coat' in cfg.cloth_types:
            self.z_cut_coat = make_linear_layers([input_feat_dim, 6], relu_final=False)
            self.z_style_coat = make_linear_layers([input_feat_dim, 12], relu_final=False)
        if 'pants' in cfg.cloth_types:
            self.z_cut_pants = make_linear_layers([input_feat_dim, 6], relu_final=False)
            self.z_style_pants = make_linear_layers([input_feat_dim, 12], relu_final=False)
        if 'skirts' in cfg.cloth_types:
            self.z_cut_skirts = make_linear_layers([input_feat_dim, 6], relu_final=False)
            self.z_style_skirts = make_linear_layers([input_feat_dim, 12], relu_final=False)
        if 'hair' in cfg.cloth_types:
            self.z_cut_hair = make_linear_layers([input_feat_dim, 6], relu_final=False)
            self.z_style_hair = make_linear_layers([input_feat_dim, 12], relu_final=False)
        if 'shoes' in cfg.cloth_types:
            self.z_style_shoes = make_linear_layers([input_feat_dim, 4], relu_final=False)
        self.cloth_cls_layer = make_linear_layers([input_feat_dim, len(cfg.cloth_types)], relu_final=False)
        self.gender_cls_layer = make_linear_layers([input_feat_dim, 2], relu_final=False)

    def forward(self, img_feat):
        batch_size = img_feat.shape[0]
        img_feat = img_feat.mean((2, 3))
        z_cuts, z_styles = [], []
        for cloth_type in cfg.cloth_types:
            if cloth_type == 'uppercloth':
                z_cuts.append(self.z_cut_uppercloth(img_feat))
                z_styles.append(self.z_style_uppercloth(img_feat))
            elif cloth_type == 'coat':
                z_cuts.append(self.z_cut_coat(img_feat))
                z_styles.append(self.z_style_coat(img_feat))
            elif cloth_type == 'pants':
                z_cuts.append(self.z_cut_pants(img_feat))
                z_styles.append(self.z_style_pants(img_feat))
            elif cloth_type == 'skirts':
                z_cuts.append(self.z_cut_skirts(img_feat))
                z_styles.append(self.z_style_skirts(img_feat))
            elif cloth_type == 'hair':
                z_cuts.append(self.z_cut_hair(img_feat))
                z_styles.append(self.z_style_hair(img_feat))
            elif cloth_type == 'shoes':
                z_cuts.append(torch.zeros((batch_size, 0)).float())
                z_styles.append(self.z_style_shoes(img_feat))
        scores = self.cloth_cls_layer(img_feat)
        scores = torch.sigmoid(scores)
        genders = self.gender_cls_layer(img_feat)
        genders = F.softmax(genders, dim=-1)
        return genders, scores, z_cuts, z_styles


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
        resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'), (34): (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'), (50): (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'), (101): (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'), (152): (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


class SMPL(object):

    def __init__(self):
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(cfg.human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(cfg.human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.joint_num = 24
        self.joints_name = 'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        self.flip_pairs = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
        self.Astar_pose = torch.zeros(1, self.joint_num * 3)
        self.Astar_pose[0, 5] = 0.04
        self.Astar_pose[0, 8] = -0.04

    def get_custom_template_layer(self, v_template, gender):
        layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False, 'v_template': v_template}
        layer = smplx.create(cfg.human_model_path, 'smpl', gender=gender.upper(), **layer_arg)
        return layer


class Model(nn.Module):

    def __init__(self, backbone, cloth_net, mode):
        super(Model, self).__init__()
        self.backbone = backbone
        self.cloth_net = cloth_net
        self.mode = mode
        self.smpl_layer = [copy.deepcopy(smpl.layer['neutral']), copy.deepcopy(smpl.layer['male']), copy.deepcopy(smpl.layer['female'])]
        self.smplicit_layer = SMPLicit.SMPLicit(cfg.smplicit_path, cfg.cloth_types)
        if mode == 'train':
            self.cloth_cls_loss = ClothClsLoss()
            self.gender_cls_loss = GenderClsLoss()
            self.sdf_dp_loss = SdfDPLoss()
            self.reg_loss = RegLoss()
        self.trainable_modules = [self.backbone, self.cloth_net]

    def forward(self, inputs, targets, meta_info, mode):
        batch_size = inputs['img'].shape[0]
        img_feat = self.backbone(inputs['img'])
        pred_genders, pred_scores, z_cuts, z_styles = self.cloth_net(img_feat)
        smpl_pose = meta_info['smpl_pose']
        smpl_shape = meta_info['smpl_shape']
        cam_trans = meta_info['cam_trans']
        if mode == 'train':
            smpl_gender = targets['gender']
            sdfs, cloth_meshes, cloth_meshes_unposed = self.smplicit_layer(z_cuts, z_styles, smpl_pose, smpl_shape, smpl_gender, do_marching_cube=mode == 'test', valid=torch.ones((len(z_cuts),), dtype=torch.bool), do_smooth=False)
            loss = {}
            loss['cloth_cls'] = cfg.cls_weight * self.cloth_cls_loss(pred_scores, targets['smpl_patch_idx'], targets['smpl_cloth_idx'])
            loss['gender_cls'] = cfg.cls_weight * self.gender_cls_loss(pred_genders, smpl_gender)
            loss['sdf_dp'] = 0.0
            loss['reg'] = 0.0
            z_cut_reg, z_style_reg = 0.0, 0.0
            for i in range(len(cfg.cloth_types)):
                cloth_type = cfg.cloth_types[i]
                if cloth_type == 'uppercloth':
                    target_cloth_idx = i + 1, cfg.cloth_types.index('coat') + 1
                else:
                    target_cloth_idx = i + 1,
                if cloth_type == 'pants' or cloth_type == 'skirts':
                    body_pose = smpl.Astar_pose.float().repeat(batch_size, 1)[:, 3:]
                else:
                    body_pose = torch.zeros((batch_size, (smpl.joint_num - 1) * 3)).float()
                v_template = self.smpl_layer[0](global_orient=torch.zeros((batch_size, 3)).float(), body_pose=body_pose, betas=smpl_shape).vertices
                loss['sdf_dp'] += cfg.dp_weight * self.sdf_dp_loss(sdfs[i], cloth_meshes_unposed[i], targets['smpl_cloth_idx'], meta_info['smpl_cloth_valid'], target_cloth_idx, cfg.sdf_thresh[cloth_type], cfg.dist_thresh[cloth_type], v_template)
                cloth_exist = (sum([(targets['smpl_cloth_idx'] == idx) for idx in target_cloth_idx]) > 0).sum(1) > 0
                if cloth_type != 'shoes':
                    z_cut_reg += cfg.cloth_reg_weight[cloth_type] * self.reg_loss(z_cuts[i], cloth_exist)
                z_style_reg += cfg.cloth_reg_weight[cloth_type] * self.reg_loss(z_styles[i], cloth_exist)
            loss['reg'] = cfg.reg_weight * (z_cut_reg + z_style_reg) / 2.0
            return loss
        else:
            pred_clothes = []
            pred_gender = []
            cloth_meshes = []
            for i in range(batch_size):
                z_cut = []
                z_style = []
                for j in range(len(cfg.cloth_types)):
                    z_cut.append(z_cuts[j][i][None, :])
                    z_style.append(z_styles[j][i][None, :])
                valid_clothes = pred_scores[i] > cfg.cls_threshold
                gender = torch.argmax(pred_genders[i]) + 1
                _, cloth_mesh, _ = self.smplicit_layer(z_cut, z_style, smpl_pose[None, i], smpl_shape[None, i], [gender], True, valid=valid_clothes)
                pred_clothes.append(valid_clothes)
                cloth_meshes.append(cloth_mesh)
                pred_gender.append(gender)
            cloth_meshes = [[i[0] for i in clothmesh] for clothmesh in zip(*cloth_meshes)]
            for i in range(len(cfg.cloth_types)):
                for j in range(batch_size):
                    if cloth_meshes[i][j] is not None:
                        cloth_meshes[i][j].vertices += cam_trans[j].detach().cpu().numpy()
            mesh_cam = self.get_coords(smpl_pose[:, :3], {'shape': smpl_shape, 'pose': smpl_pose[:, 3:]}, cam_trans, pred_gender)
            out = {}
            out['pred_clothes'] = pred_clothes
            out['pred_gender'] = pred_gender
            out['smpl_mesh'] = mesh_cam
            for i, cloth_type in enumerate(cfg.cloth_types):
                out[cloth_type + '_mesh'] = cloth_meshes[i]
            for k, v in targets.items():
                out[f'{k}_target'] = v
            return out

    def get_coords(self, root_pose, params, cam_trans, gender):
        batch_size = root_pose.shape[0]
        if self.mode != 'train':
            mesh_cam = []
            for i in range(batch_size):
                output = self.smpl_layer[gender[i]](betas=params['shape'][None, i], body_pose=params['pose'][None, i], global_orient=root_pose[None, i], transl=cam_trans[None, i])
                mesh_cam.append(output.vertices)
            mesh_cam = torch.cat(mesh_cam, dim=0)
        else:
            output = self.smpl_layer[0](betas=params['shape'], body_pose=params['pose'], global_orient=root_pose, transl=cam_trans)
            mesh_cam = output.vertices
        return mesh_cam


class Options:

    def __init__(self):
        self.upperbody_loadepoch = 11
        self.upperbody_clusters = 'indexs_clusters_tshirt_smpl.npy'
        self.upperbody_num_clusters = 500
        self.upperbody_n_z_cut = 6
        self.upperbody_n_z_style = 12
        self.upperbody_resolution = 128
        self.upperbody_thresh_occupancy = -0.03
        self.coat_thresh_occupancy = -0.08
        self.pants_loadepoch = 60
        self.pants_clusters = 'clusters_lowerbody.npy'
        self.pants_num_clusters = 500
        self.pants_n_z_cut = 6
        self.pants_n_z_style = 12
        self.pants_resolution = 128
        self.pants_thresh_occupancy = -0.02
        self.skirts_loadepoch = 40
        self.skirts_clusters = 'clusters_lowerbody.npy'
        self.skirts_num_clusters = 500
        self.skirts_n_z_cut = 6
        self.skirts_n_z_style = 12
        self.skirts_resolution = 128
        self.skirts_thresh_occupancy = -0.05
        self.hair_loadepoch = 20000
        self.hair_clusters = 'clusters_hairs.npy'
        self.hair_num_clusters = 500
        self.hair_n_z_cut = 6
        self.hair_n_z_style = 12
        self.hair_resolution = 128
        self.hair_thresh_occupancy = -2.0
        self.shoes_loadepoch = 20000
        self.shoes_clusters = 'clusters_shoes.npy'
        self.shoes_n_z_cut = 0
        self.shoes_n_z_style = 4
        self.shoes_resolution = 64
        self.shoes_thresh_occupancy = -0.36
        self.shoes_num_clusters = 100
        self.path_checkpoints = '../../../../data/base_data/smplicit/checkpoints/'
        self.path_cluster_files = '../../../../data/base_data/smplicit/clusters/'
        self.path_SMPL = '../../../../data/base_data/human_models/smpl'
        self.upperbody_b_min = [-0.8, -0.4, -0.3]
        self.upperbody_b_max = [0.8, 0.6, 0.3]
        self.pants_b_min = [-0.3, -1.2, -0.3]
        self.pants_b_max = [0.3, 0.0, 0.3]
        self.skirts_b_min = [-0.3, -1.2, -0.3]
        self.skirts_b_max = [0.3, 0.0, 0.3]
        self.hair_b_min = [-0.35, -0.42, -0.33]
        self.hair_b_max = [0.35, 0.68, 0.37]
        self.shoes_b_min = [-0.1, -1.4, -0.2]
        self.shoes_b_max = [0.25, -0.6, 0.3]


class SMPLicit(nn.Module):

    def __init__(self, root_path, cloth_types):
        super(SMPLicit, self).__init__()
        self._opt = Options()
        uppercloth = Model(osp.join(root_path, self._opt.path_checkpoints, 'upperclothes.pth'), self._opt.upperbody_n_z_cut, self._opt.upperbody_n_z_style, self._opt.upperbody_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.upperbody_clusters), self._opt.upperbody_b_min, self._opt.upperbody_b_max, self._opt.upperbody_resolution, thresh=self._opt.upperbody_thresh_occupancy)
        coat = Model(osp.join(root_path, self._opt.path_checkpoints, 'upperclothes.pth'), self._opt.upperbody_n_z_cut, self._opt.upperbody_n_z_style, self._opt.upperbody_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.upperbody_clusters), self._opt.upperbody_b_min, self._opt.upperbody_b_max, self._opt.upperbody_resolution, thresh=self._opt.coat_thresh_occupancy)
        pants = Model(osp.join(root_path, self._opt.path_checkpoints, 'pants.pth'), self._opt.pants_n_z_cut, self._opt.pants_n_z_style, self._opt.pants_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.pants_clusters), self._opt.pants_b_min, self._opt.pants_b_max, self._opt.pants_resolution, thresh=self._opt.pants_thresh_occupancy)
        skirts = Model(osp.join(root_path, self._opt.path_checkpoints, 'skirts.pth'), self._opt.skirts_n_z_cut, self._opt.skirts_n_z_style, self._opt.skirts_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.skirts_clusters), self._opt.skirts_b_min, self._opt.skirts_b_max, self._opt.skirts_resolution, thresh=self._opt.skirts_thresh_occupancy)
        hair = Model(osp.join(root_path, self._opt.path_checkpoints, 'hair.pth'), self._opt.hair_n_z_cut, self._opt.hair_n_z_style, self._opt.hair_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.hair_clusters), self._opt.hair_b_min, self._opt.hair_b_max, self._opt.hair_resolution, thresh=self._opt.hair_thresh_occupancy)
        shoes = Model(osp.join(root_path, self._opt.path_checkpoints, 'shoes.pth'), self._opt.shoes_n_z_cut, self._opt.shoes_n_z_style, self._opt.shoes_num_clusters, osp.join(root_path, self._opt.path_cluster_files, self._opt.shoes_clusters), self._opt.shoes_b_min, self._opt.shoes_b_max, self._opt.shoes_resolution, thresh=self._opt.shoes_thresh_occupancy)
        self.models = []
        for cloth_type in cloth_types:
            if cloth_type == 'uppercloth':
                self.models.append(uppercloth)
            elif cloth_type == 'coat':
                self.models.append(coat)
            elif cloth_type == 'pants':
                self.models.append(pants)
            elif cloth_type == 'skirts':
                self.models.append(skirts)
            elif cloth_type == 'hair':
                self.models.append(hair)
            elif cloth_type == 'shoes':
                self.models.append(shoes)
            else:
                assert 0, 'Not supported cloth type: ' + cloth_type
        self.cloth_types = cloth_types
        self.SMPL_Layers = [SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_NEUTRAL.pkl'), obj_saveable=True), SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_MALE.pkl'), obj_saveable=True), SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_FEMALE.pkl'), obj_saveable=True)]
        self.SMPL_Layer = None
        self.smpl_faces = self.SMPL_Layers[0].faces
        Astar_pose = torch.zeros(1, 72)
        Astar_pose[0, 5] = 0.04
        Astar_pose[0, 8] = -0.04
        self.register_buffer('Astar_pose', Astar_pose)
        self.step = 1000

    def get_right_shoe(self, sdf, unposed_cloth_mesh, do_marching_cube):
        if not do_marching_cube:
            sdf = torch.cat((sdf, sdf), 1)
            rshoe = torch.stack((-unposed_cloth_mesh[:, :, 0], unposed_cloth_mesh[:, :, 1], unposed_cloth_mesh[:, :, 2]), 2)
            unposed_cloth_mesh = torch.cat((unposed_cloth_mesh, rshoe), 1)
            return sdf, unposed_cloth_mesh
        else:
            rshoe = np.stack((-unposed_cloth_mesh.vertices[:, 0], unposed_cloth_mesh.vertices[:, 1], unposed_cloth_mesh.vertices[:, 2]), 1)
            vertices = np.concatenate((unposed_cloth_mesh.vertices, rshoe))
            faces = np.concatenate((unposed_cloth_mesh.faces, unposed_cloth_mesh.faces[:, ::-1] + len(rshoe)))
            unposed_cloth_mesh = trimesh.Trimesh(vertices, faces)
            return None, unposed_cloth_mesh

    def pose_mesh(self, unposed_cloth_mesh, pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube, smooth=True):
        if not do_marching_cube:
            iters = math.ceil(unposed_cloth_mesh.shape[1] / self.step)
            posed_cloth_mesh = []
            for i in range(iters):
                in_verts = unposed_cloth_mesh[:, i * self.step:(i + 1) * self.step, :]
                out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, unposed_smpl_joint, unposed_smpl_mesh, in_verts)
                posed_cloth_mesh.append(out_verts)
            posed_cloth_mesh = torch.cat(posed_cloth_mesh, 1)
            return posed_cloth_mesh
        else:
            iters = math.ceil(len(unposed_cloth_mesh.vertices) / self.step)
            for i in range(iters):
                in_verts = torch.FloatTensor(unposed_cloth_mesh.vertices[None, i * self.step:(i + 1) * self.step, :])
                out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, unposed_smpl_joint, unposed_smpl_mesh, in_verts)
                unposed_cloth_mesh.vertices[i * self.step:(i + 1) * self.step] = out_verts.cpu().data.numpy()
            posed_cloth_mesh = unposed_cloth_mesh
            if smooth:
                posed_cloth_mesh = trimesh.smoothing.filter_laplacian(posed_cloth_mesh, lamb=0.5)
            return posed_cloth_mesh

    def pose_mesh_lower_body(self, unposed_cloth_mesh, pose, shape, Astar_pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube, smooth=True):
        if not do_marching_cube:
            iters = math.ceil(unposed_cloth_mesh.shape[1] / self.step)
            posed_cloth_mesh = []
            for i in range(iters):
                in_verts = unposed_cloth_mesh[:, i * self.step:(i + 1) * self.step]
                out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, Astar_pose, pose, shape, unposed_smpl_joint, unposed_smpl_mesh)
                posed_cloth_mesh.append(out_verts)
            posed_cloth_mesh = torch.cat(posed_cloth_mesh, 1)
            return posed_cloth_mesh
        else:
            iters = math.ceil(len(unposed_cloth_mesh.vertices) / self.step)
            for i in range(iters):
                in_verts = torch.FloatTensor(unposed_cloth_mesh.vertices[None, i * self.step:(i + 1) * self.step])
                out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, Astar_pose, pose, shape, unposed_smpl_joint, unposed_smpl_mesh)
                unposed_cloth_mesh.vertices[i * self.step:(i + 1) * self.step] = out_verts.cpu().data.numpy()
            posed_cloth_mesh = unposed_cloth_mesh
            if smooth:
                posed_cloth_mesh = trimesh.smoothing.filter_laplacian(posed_cloth_mesh, lamb=0.5)
            return posed_cloth_mesh

    def forward(self, z_cuts, z_styles, pose, shape, gender=[0], do_marching_cube=False, valid=None, do_smooth=True):
        batch_size = pose.shape[0]
        unposed_smpl_joint, unposed_smpl_mesh = [], []
        Astar_smpl_mesh, Astar_smpl_joint = [], []
        for i in range(batch_size):
            SMPL_Layer = self.SMPL_Layers[gender[i]]
            unposed_smpl_joint_i, unposed_smpl_mesh_i = SMPL_Layer.skeleton(shape[None, i], require_body=True)
            Astar_smpl_mesh_i, Astar_smpl_joint_i, _ = SMPL_Layer.forward(beta=shape[None, i], theta=self.Astar_pose.repeat(1, 1), get_skin=True)
            unposed_smpl_joint.append(unposed_smpl_joint_i)
            unposed_smpl_mesh.append(unposed_smpl_mesh_i)
            Astar_smpl_mesh.append(Astar_smpl_mesh_i)
            Astar_smpl_joint.append(Astar_smpl_joint_i)
        unposed_smpl_joint = torch.cat(unposed_smpl_joint)
        unposed_smpl_mesh = torch.cat(unposed_smpl_mesh)
        Astar_smpl_mesh = torch.cat(Astar_smpl_mesh)
        Astar_smpl_joint = torch.cat(Astar_smpl_joint)
        self.SMPL_Layer = self.SMPL_Layers[gender[0]]
        out_sdfs = []
        out_meshes = []
        out_meshes_unposed = []
        for i in range(len(self.models)):
            if ~valid[i]:
                out_sdfs.append([None])
                out_meshes.append([None])
                out_meshes_unposed.append([None])
                continue
            if self.cloth_types[i] in ['uppercloth', 'coat']:
                cloth_type = 'upperbody'
            else:
                cloth_type = self.cloth_types[i]
            resolution = eval(f'self._opt.{cloth_type}_resolution')
            if self.cloth_types[i] == 'coat':
                is_coat = True
            else:
                is_coat = False
            if not do_marching_cube:
                resolution = 21
            if self.cloth_types[i] == 'pants' or self.cloth_types[i] == 'skirts':
                sdf, unposed_cloth_mesh = self.models[i].decode(z_cuts[i], z_styles[i], Astar_smpl_joint, Astar_smpl_mesh, resolution, do_marching_cube, do_smooth)
                if not do_marching_cube:
                    posed_cloth_mesh = self.pose_mesh_lower_body(unposed_cloth_mesh, pose, shape, self.Astar_pose.repeat(batch_size, 1), unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube)
                else:
                    posed_cloth_mesh = []
                    for j in range(len(unposed_cloth_mesh)):
                        if unposed_cloth_mesh[j] is None:
                            posed_cloth_mesh.append(None)
                            continue
                        posed_cloth_mesh.append(self.pose_mesh_lower_body(unposed_cloth_mesh[j], pose[j, None], shape[j, None], self.Astar_pose, unposed_smpl_joint[j, None], unposed_smpl_mesh[j, None], do_marching_cube, do_smooth))
            else:
                sdf, unposed_cloth_mesh = self.models[i].decode(z_cuts[i], z_styles[i], unposed_smpl_joint, unposed_smpl_mesh, resolution, do_marching_cube, do_smooth, is_coat=is_coat)
                if not do_marching_cube:
                    if self.cloth_types[i] == 'shoes':
                        sdf, unposed_cloth_mesh = self.get_right_shoe(sdf, unposed_cloth_mesh, do_marching_cube)
                    posed_cloth_mesh = self.pose_mesh(unposed_cloth_mesh, pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube)
                else:
                    posed_cloth_mesh = []
                    for j in range(len(unposed_cloth_mesh)):
                        if unposed_cloth_mesh[j] is None:
                            posed_cloth_mesh.append(None)
                            continue
                        if self.cloth_types[i] == 'shoes':
                            _, unposed_cloth_mesh[j] = self.get_right_shoe(None, unposed_cloth_mesh[j], do_marching_cube)
                        posed_cloth_mesh.append(self.pose_mesh(unposed_cloth_mesh[j], pose[j, None], unposed_smpl_joint[j, None], unposed_smpl_mesh[j, None], do_marching_cube, do_smooth))
            out_sdfs.append(sdf)
            out_meshes.append(posed_cloth_mesh)
            out_meshes_unposed.append(unposed_cloth_mesh)
        return out_sdfs, out_meshes, out_meshes_unposed


class Network(nn.Module):

    def __init__(self, n_z_style=1, point_pos_size=3, output_dim=1, n_z_cut=12):
        super(Network, self).__init__()
        self.point_pos_size = point_pos_size
        self.fc0_cloth = nn.utils.weight_norm(nn.Linear(n_z_style, 128, bias=True))
        self.fc1_cloth = nn.utils.weight_norm(nn.Linear(128, 128, bias=True))
        self.fc0_query = nn.utils.weight_norm(nn.Conv1d(point_pos_size, 128, kernel_size=1, bias=True))
        self.fc1_query = nn.utils.weight_norm(nn.Conv1d(128, 256, kernel_size=1, bias=True))
        self.fc0 = nn.utils.weight_norm(nn.Conv1d(128 + 256 + n_z_cut, 312, kernel_size=1, bias=True))
        self.fc1 = nn.utils.weight_norm(nn.Conv1d(312, 312, kernel_size=1, bias=True))
        self.fc2 = nn.utils.weight_norm(nn.Conv1d(312, 256, kernel_size=1, bias=True))
        self.fc3 = nn.utils.weight_norm(nn.Conv1d(256, 128, kernel_size=1, bias=True))
        self.fc4 = nn.utils.weight_norm(nn.Conv1d(128, output_dim, kernel_size=1, bias=True))
        self.activation = F.relu

    def forward(self, z_cut, z_style, query):
        batch_size = len(z_style)
        query_num = query.shape[1]
        x_cloth = self.activation(self.fc0_cloth(z_style))
        x_cloth = self.activation(self.fc1_cloth(x_cloth))
        x_cloth = x_cloth.unsqueeze(-1).repeat(1, 1, query_num)
        query = query.reshape(batch_size, query_num, self.point_pos_size).permute(0, 2, 1)
        x_query = self.activation(self.fc0_query(query))
        x_query = self.activation(self.fc1_query(x_query))
        z_cut = z_cut.unsqueeze(-1).repeat(1, 1, query_num)
        _in = torch.cat((x_cloth, x_query, z_cut), 1)
        x = self.fc0(_in)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        if x.shape[1] == 1:
            return x[:, 0]
        else:
            return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RegLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hygenie1228_ClothWild_RELEASE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

