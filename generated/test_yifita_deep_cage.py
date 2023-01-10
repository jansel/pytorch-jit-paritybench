import sys
_module = sys.modules[__name__]
del sys
cage_deformer_3d = _module
common = _module
datasets = _module
deformer_3d = _module
losses = _module
network2 = _module
networks = _module
optimize_cage = _module
option = _module
create_montage = _module
create_scaling_baseline = _module
create_shapenet_testtrain_knn = _module
divide_shapenet_label = _module
evaluation = _module
generate_data_with_known_cage = _module
normalize_humanoids = _module
remesh_shapenet = _module
resample_shapenet = _module
transfer_corres = _module

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


import torch.nn as nn


import torch.nn.parallel


import torch.utils.data


import numpy as np


import itertools


from collections import OrderedDict


import time


from collections import defaultdict


import warnings


import torch.nn.functional as F


from matplotlib.colors import Normalize


from matplotlib import cm


import pandas as pd


from sklearn.neighbors import NearestNeighbors


class FaceNormalLoss(torch.nn.Module):

    def __init__(self, n_faces=100):
        super().__init__()
        self.n_faces = n_faces
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, ref_mesh_V, mesh_V, mesh_F):
        B, F, _ = mesh_F.shape
        face_sample_idx = torch.randint(min(self.n_faces, F), (B, self.n_faces, 1), dtype=torch.int64)
        sampled_F = torch.gather(mesh_F, 1, face_sample_idx.expand(-1, -1, 3))
        ref_normals, _ = compute_face_normals_and_areas(ref_mesh_V, mesh_F)
        normals, _ = compute_face_normals_and_areas(mesh_V, mesh_F)
        cos = self.cos(ref_normals, normals)
        return torch.mean(1 - cos)


class GTNormalLoss(torch.nn.Module):
    """
    compare the PCA normals of two point clouds
    ===
    params:
        NCHW: order of dimensions, default True
        pred: (B,3,N) if NCHW, (B,N,3) otherwise
    """

    def __init__(self, nn_size=10, NCHW=True):
        super().__init__()
        self.nn_size = nn_size
        self.NCHW = NCHW
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, pred, gt_normals):
        pred_normals = batch_normals(pred, nn_size=10, NCHW=self.NCHW)
        cos = self.cos(pred_normals, gt_normals)
        return torch.mean(1 - cos)


class GroundingLoss(torch.nn.Module):

    def __init__(self, up_dim=1):
        super().__init__()
        self.up_dim = up_dim

    def forward(self, source, deformed):
        """
        source:   (B,N,3)
        deformed: (B,N,3)
        """
        eps = 0.01
        ground_level = torch.min(source[:, :, self.up_dim], dim=1)[0]
        ground_point_mask = (source[:, :, self.up_dim] - ground_level.unsqueeze(-1)).abs() < eps
        source_ground_level = torch.masked_select(source[:, :, self.up_dim], ground_point_mask)
        deformed_ground_level = torch.masked_select(deformed[:, :, self.up_dim], ground_point_mask)
        return torch.mean(torch.abs(source_ground_level - deformed_ground_level))


class LabeledChamferDistance(torch.nn.Module):
    """
    Learning to Sample Dovrat et.al
    mean_{xyz1}(nd_{1to2})+eta*max_{xyz1}(nd_{1to2})+(\\gamma+\\delta|xyz1|)mean_{xyz2}(nd_{2to1})
    ===
    :param:
        xyz1: generated points
        xyz2: reference points
    """

    def __init__(self, beta=1.0, gamma=1, delta=0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, xyz1, xyz2, label1=None, label2=None):
        P = xyz1.shape[1]
        if label1 is not None and label2 is not None:
            dist12, dist21, idx12, idx21 = labeled_nndistance(xyz1, xyz2, label1, label2)
        else:
            dist12, dist21, idx12, idx21 = nndistance(xyz1, xyz2)
        loss = torch.mean(dist12, dim=-1) + torch.max(dist12, dim=-1)[0] * self.beta + (self.gamma + self.delta * P) * torch.mean(dist21, dim=-1)
        loss = torch.mean(loss)
        return loss, idx12, idx21


class LocalFeatureLoss(torch.nn.Module):
    """
    penalize point to surface loss
    Given points (B,N,3)
    1. find KNN and the center
    2. fit PCA, get normal
    3. project p-center to normal
    """

    def __init__(self, nn_size=10, metric=torch.nn.MSELoss('mean'), **kwargs):
        super().__init__()
        self.nn_size = nn_size
        self.metric = metric

    def forward(self, xyz1, xyz2, **kwargs):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        B, N, C = xyz1.shape
        grouped_points, idx, _ = group_knn(self.nn_size, xyz1, xyz1, unique=True, NCHW=False)
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        U, S, V = batch_svd(allpoints)
        normals = V[:, :, -1].view(B, N, C).detach()
        ptof1 = dot_product(xyz1 - group_center.squeeze(2), normals, dim=-1)
        grouped_points = torch.gather(xyz2.unsqueeze(1).expand(-1, N, -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        U, S, V = batch_svd(allpoints)
        normals = V[:, :, -1].view(B, N, C).detach()
        ptof2 = dot_product(xyz2 - group_center.squeeze(2), normals, dim=-1)
        loss = self.metric(ptof1.abs(), ptof2.abs())
        bent = ptof2 - ptof1
        bent.masked_fill_(bent < 0, 0.0)
        bent = self.metric(bent, torch.zeros_like(bent))
        loss += 5 * bent
        return loss


class MVCRegularizer(torch.nn.Module):
    """
    penalize MVC with large absolute value and negative values
    alpha * large_weight^2 + beta * (negative_weight)^2
    """

    def __init__(self, alpha=1.0, beta=1.0, threshold=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, weights):
        loss = 0
        if self.alpha > 0:
            large_loss = torch.log(torch.nn.functional.relu(weights.abs() - self.threshold) + 1)
            loss += torch.mean(large_loss) * self.alpha
        if self.beta > 0:
            neg_loss = torch.nn.functional.relu(-weights)
            neg_loss = neg_loss ** 2
            loss += torch.mean(neg_loss) * self.beta
        return loss


class MeshDihedralAngleLoss(torch.nn.Module):
    """
    if vert1 and vert both given, penalize difference of the dihedral angle between vert1 and vert2
    otherwise penalize if dehedral angle < pi/4
    vert1           (B,N,3)
    vert2           (B,N,3)
    edge_points     List(torch.Tensor(E, 4))
    """

    def __init__(self, threshold=np.pi / 6, edge_points=None, reduction='mean'):
        super().__init__()
        self.edge_points = edge_points
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, vert1, vert2=None, edge_points=None):
        if edge_points is None:
            edge_points = self.edge_points
        assert edge_points is not None
        B = vert1.shape[0]
        loss = []
        for b in range(B):
            angles1 = dihedral_angle(vert1[b], edge_points)
            if vert2 is not None:
                angles2 = dihedral_angle(vert2[b], edge_points)
                tmp = self.metric(angles1, angles2)
            else:
                tmp = torch.nn.functional.relu(np.pi / 4 - angles1)
                tmp = tmp * tmp
                tmp = torch.mean(tmp)
            loss.append(tmp)
        loss = torch.stack(loss, dim=0)
        if self.reduction != 'none':
            loss = loss.mean()
        return loss


class MeshSmoothLoss(torch.nn.Module):
    """
    compare laplacian of two meshes with the same connectivity assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    use_cot: cot laplacian is used instead of uniformlaplacian
    consistent_topology: assume face matrix is the same during the entire use
    precompute_L: assume vert1 is always the same
    """

    def __init__(self, metric, use_cot=False, use_norm=False):
        super().__init__()
        if use_cot:
            self.laplacian = CotLaplacian()
        else:
            self.laplacian = UniformLaplacian()
        self.metric = metric

    def forward(self, vert1, face=None):
        lap1 = self.laplacian(vert1, face)
        lap1 = torch.norm(lap1, dim=-1, p=2)
        return lap1.mean()


class SymmetryLoss(torch.nn.Module):
    """
    symmetry loss
    chamfer(mirrored(xyz), xyz)
    ===
    :params:
        sym_plane ("yz"): list of "xy", "yz", "zx"
        NCHW      bool  : point dimension
        xyz             : (B,3,N) or (B,N,3)
    """

    def __init__(self, sym_plane=('yz',), NCHW=True):
        super().__init__()
        self.sym_plane = sym_plane
        assert isinstance(self.sym_plane, tuple) or isinstance(self.sym_plane, list), 'sym_plane must be a list or tuple'
        self.metric = LabeledChamferDistance(beta=0.0, gamma=1.0, delta=0)
        self.register_buffer('base_ones', torch.ones((3,), dtype=torch.float))
        self.NCHW = NCHW
        self.mirror_ops = []
        for p in self.sym_plane:
            if 'x' not in p:
                self.mirror_ops += [lambda xyz: xyz * self.get_mirror_multiplier(0)]
            elif 'y' not in p:
                self.mirror_ops += [lambda xyz: xyz * self.get_mirror_multiplier(1)]
            elif 'z' not in p:
                self.mirror_ops += [lambda xyz: xyz * self.get_mirror_multiplier(2)]
            else:
                raise ValueError

    def get_mirror_multiplier(self, dim_id):
        base_ones = self.base_ones.clone()
        base_ones[dim_id] = -1
        if self.NCHW:
            return base_ones.view((1, 3, 1))
        else:
            return base_ones.view((1, 1, 3))

    def forward(self, xyz):
        loss = 0
        for op in self.mirror_ops:
            m_xyz = op(xyz)
            loss += self.metric(m_xyz.detach(), xyz)[0]
        return loss


class AllLosses(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = defaultdict(float)
        self.labeled_chamfer_loss = LabeledChamferDistance(beta=opt.beta, gamma=opt.gamma, delta=opt.delta)
        self.cage_shortLength_loss = SimpleMeshRepulsionLoss(0.02, reduction='mean', consistent_topology=True)
        self.cage_faceAngle_loss = MeshDihedralAngleLoss(threshold=np.pi / 30)
        self.mvc_reg_loss = MVCRegularizer(threshold=50, beta=1.0, alpha=0.0)
        self.cage_laplacian = MeshLaplacianLoss(torch.nn.L1Loss(reduction='mean'), use_cot=False, use_norm=True, consistent_topology=True, precompute_L=True)
        self.cage_smooth_loss = MeshSmoothLoss(torch.nn.MSELoss(reduction='mean'), use_cot=False, use_norm=True)
        self.grounding_loss = GroundingLoss(up_dim=1 if 'SHAPENET' in opt.dataset else 2)
        if opt.sym_plane is not None:
            self.symmetry_loss = SymmetryLoss(sym_plane=opt.sym_plane, NCHW=False)
        if self.opt.dataset in ('SURREAL', 'FAUST'):
            logger.info('Using GTNormal loss')
            self.shape_normal_loss = GTNormalLoss()
        else:
            logger.info('Using KNN for normal loss')
            self.shape_normal_loss = NormalLoss(reduction='none', nn_size=16)
        self.shape_fnormal_loss = FaceNormalLoss(n_faces=300)
        self.stretch_loss = PointStretchLoss(4 if opt.dim == 3 else 2, reduction='mean')
        self.edge_loss = PointEdgeLengthLoss(4 if opt.dim == 3 else 2, torch.nn.MSELoss(reduction='mean'))
        if self.opt.regular_sampling or not opt.mesh_data:
            logger.info('Using point laplacian loss')
            self.shape_laplacian = PointLaplacianLoss(16, torch.nn.MSELoss(reduction='none'), use_norm=opt.slap_norm)
        else:
            logger.info('Using mesh laplacian loss')
            self.shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(reduction='none'), use_cot=True, use_norm=True, consistent_topology=True, precompute_L=True)
        self.p2f_loss = LocalFeatureLoss(16, torch.nn.MSELoss(reduction='none'))

    def forward(self, all_inputs, all_outputs, progress=1.0):
        self.loss.clear()
        B = all_outputs['new_cage'].shape[0]
        if self.opt.loss == 'LCD':
            loss, idx12, idx21 = self.labeled_chamfer_loss(all_outputs['deformed'], all_inputs['target_shape'], all_inputs['source_label'], all_inputs['target_label'])
            self.idx12 = idx12
            self.idx21 = idx21
            self.loss['LCD'] += loss * opt.loss_weight
            dist = torch.sum((all_outputs['deformed'][self.opt.batch_size * 2:, :, :] - all_inputs['target_shape'][self.opt.batch_size * 2:, :, :]) ** 2, dim=-1)
            self.loss['MSE'] += dist.mean() * opt.loss_weight
        elif self.opt.loss == 'CD':
            loss, idx12, idx21 = self.labeled_chamfer_loss(all_outputs['deformed'], all_inputs['target_shape'])
            self.loss['CD'] = loss
            self.loss['CD'] *= self.opt.loss_weight
            self.idx12 = idx12
            self.idx21 = idx21
            dist = torch.sum((all_outputs['deformed'][self.opt.batch_size * 2:, :, :] - all_inputs['target_shape'][self.opt.batch_size * 2:, :, :]) ** 2, dim=-1)
            self.loss['MSE'] += dist.mean() * self.opt.loss_weight
        elif self.opt.loss == 'MSE':
            dist = torch.sum((all_outputs['deformed'] - all_inputs['target_shape']) ** 2, dim=-1)
            self.loss['MSE'] += dist.mean()
            self.loss['MSE'] += torch.max(dist, dim=1)[0].mean()
            self.loss['MSE'] *= self.opt.loss_weight
        if self.opt.cshape_weight > 0:
            ref_surface = all_inputs['source_shape'] + 0.1 * all_inputs['source_normals']
            loss, _, _ = self.labeled_chamfer_loss(all_outputs['cage'], ref_surface)
            self.loss['CSHAPE'] += loss
            self.loss['CSHAPE'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.cshape_weight, 0)
        if self.opt.gravity_weight > 0:
            cage_shift = torch.mean(all_outputs['cage'], dim=1) - torch.mean(all_inputs['source_shape'], dim=1)
            self.loss['GRAV'] += torch.mean(torch.nn.functional.softshrink(torch.sum(cage_shift ** 2, dim=-1), lambd=0.1))
            self.loss['GRAV'] *= self.opt.gravity_weight
        if self.opt.mvc_weight > 0:
            self.loss['WREG'] += self.mvc_reg_loss(all_outputs['weight']) * self.opt.mvc_weight
        if self.opt.p2f_weight > 0:
            self.loss['P2F'] = torch.mean(self.p2f_loss(all_inputs['source_shape'], all_outputs['deformed']))
            self.loss['P2F'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.p2f_weight, self.opt.p2f_weight / 10)
        if self.opt.slap_weight > 0:
            slap1 = torch.mean(self.shape_laplacian(all_inputs['source_shape'], all_outputs['deformed'], face=all_inputs['source_face']).view(B, -1), dim=-1, keepdim=True)
            if self.opt.blend_style and hasattr(self, 'idx21'):
                slap1 *= 1 - all_inputs['alpha']
                slap2 = torch.mean(self.shape_laplacian(all_outputs['deformed'], all_inputs['target_shape'], idx12=self.idx12).view(B, -1), dim=-1, keepdim=True)
                slap2 *= all_inputs['alpha']
                self.loss['SLAP'] += slap2.mean()
            self.loss['SLAP'] += slap1.mean()
            self.loss['SLAP'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.slap_weight, self.opt.slap_weight / 10)
        if self.opt.snormal_weight > 0:
            snormal1 = torch.mean(self.shape_normal_loss(all_inputs['source_shape'], all_outputs['deformed']), dim=-1, keepdim=True)
            if self.opt.blend_style and hasattr(self, 'idx21'):
                snormal1 *= 1 - all_inputs['alpha']
                snormal2 = torch.mean(self.shape_normal_loss(all_outputs['deformed'], all_inputs['target_shape'], idx12=self.idx12), dim=-1, keepdim=True)
                snormal2 *= all_inputs['alpha']
                self.loss['SNORMAL'] += snormal2.mean()
            self.loss['SNORMAL'] += snormal1.mean()
            self.loss['SNORMAL'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.snormal_weight, self.opt.snormal_weight / 10)
        if self.opt.sym_weight > 0:
            self.loss['SYM'] += self.symmetry_loss(all_outputs['deformed'])
            self.loss['SYM'] += self.symmetry_loss(all_outputs['cage'])
            self.loss['SYM'] *= self.opt.sym_weight
        if self.opt.ground_weight > 0:
            self.loss['GROUND'] += self.grounding_loss(all_inputs['source_shape'], all_outputs['deformed'])
            self.loss['GROUND'] *= self.opt.ground_weight
        if self.opt.cfangle_weight > 0:
            self.loss['CFANGLE'] += self.cage_faceAngle_loss(all_outputs['new_cage'], edge_points=all_inputs['cage_edge_points'])
            self.loss['CFANGLE'] *= self.opt.cfangle_weight
        if self.opt.csmooth_weight > 0:
            self.loss['CSMOOTH'] += self.cage_smooth_loss(all_outputs['new_cage'], face=all_outputs['cage_face'])
            self.loss['CSMOOTH'] *= self.opt.csmooth_weight
        if self.opt.cshort_weight > 0:
            self.loss['CEDGE'] = self.cage_shortLength_loss(all_outputs['cage'], edges=all_inputs['cage_edges'])
            self.loss['CEDGE'] *= self.opt.cshort_weight
        if self.opt.clap_weight > 0:
            self.loss['CLAP'] += self.cage_laplacian(all_outputs['cage'].expand(B, -1, -1).contiguous().detach(), all_outputs['new_cage'].contiguous(), face=all_outputs['cage_face'])
            self.loss['CLAP'] *= self.opt.clap_weight
        if self.opt.sstretch_weight > 0:
            self.loss['SSTRETCH'] += self.stretch_loss(all_outputs['source_shape'], all_outputs['deformed']) * self.opt.sstretch_weight
        if self.opt.sedge_weight > 0:
            self.loss['SEDGE'] += self.edge_loss(all_outputs['source_shape'], all_outputs['deformed'])
            self.loss['SEDGE'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.sedge_weight, self.opt.sedge_weight / 10)
        if self.opt.sfnormal_weight > 0:
            self.loss['SFNORMAL'] += self.shape_fnormal_loss(all_inputs['target_mesh'], all_outputs['deformed_hr'], all_inputs['source_face'].expand(B, -1, -1))
            self.loss['SFNORMAL'] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.sfnormal_weight, self.opt.sfnormal_weight / 10)
        return self.loss


class ExtPointToNearestFaceDistance(torch.nn.Module):
    """
    for every exteror points return the squared distance to the closest face
    """

    def __init__(self, min_dist=0.1, reduction='mean'):
        super().__init__()
        self.min_dist = min_dist
        self.reduction = reduction

    def forward(self, mesh_V, mesh_F, points, exterior_flag, mesh_FN=None):
        """
        mesh_V        (B,N,3)
        mesh_F        (B,F,3)
        mesh_FN       (B,F,3)
        points        (B,P,3)
        exterior_flat (B,P,1)
        """
        if mesh_FN is None:
            mesh_FN, _ = compute_face_normals_and_areas(mesh_V, mesh_F)
            mesh_FN = mesh_FN.detach()
        else:
            mesh_FN = mesh_FN.detach()
        B, F, _ = mesh_F.shape
        _, N, D = mesh_V.shape
        _, P, D = points.shape
        face_points = torch.gather(mesh_V.unsqueeze(1).expand(-1, F, -1, -1), 2, mesh_F.unsqueeze(-1).expand(-1, -1, -1, 3))
        face_center = torch.mean(face_points, dim=-2)
        point_to_face_center = points.unsqueeze(2) - face_center.unsqueeze(1)
        point_to_face_signed_dist = dot_product(point_to_face_center, mesh_FN.unsqueeze(1), dim=-1, keepdim=True) + self.min_dist
        point_to_face_v = point_to_face_signed_dist * mesh_FN.unsqueeze(1)
        point_to_face_sqdist = torch.sum(point_to_face_v * point_to_face_v, dim=-1)
        point_to_face_sqdist.masked_fill_(point_to_face_signed_dist.squeeze(-1) < 0, 10000000000.0)
        point_to_face_sqdist, _ = torch.min(point_to_face_sqdist, dim=-1)
        inside_flag = ~exterior_flag.view(B, P) | torch.all(point_to_face_signed_dist.view(B, P, F) < 0, dim=-1)
        point_to_face_sqdist.masked_fill_(inside_flag, 0)
        if self.reduction == 'mean':
            point_to_face_sqdist = torch.mean(point_to_face_sqdist.view(B, -1), dim=1)
        elif self.reduction == 'max':
            point_to_face_sqdist = torch.max(point_to_face_sqdist.view(B, -1), dim=1)[0]
        elif self.reduction == 'sum':
            point_to_face_sqdist = torch.sum(point_to_face_sqdist.view(B, -1), dim=1)
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError
        point_to_face_sqdist = torch.mean(point_to_face_sqdist, dim=0)
        return point_to_face_sqdist


class ConditionNumberLoss(torch.nn.Module):
    """
    compare ratio of the largest and smallest principal component values
    ===
    params:
        ref_points: (B,N,dim)
        points:     (B,N,dim)
    """

    def __init__(self, ball_size, metric, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.ball_size2 = ball_size * 2
        self.metric = metric
        self.nn_size = 16

    def forward(self, ref_points, points, *args, **kwargs):
        B, N, C = ref_points.shape
        ref_grouped_points, ref_group_idx, ref_group_dist = faiss_knn(self.nn_size, ref_points, ref_points, NCHW=False)
        mask = ref_group_dist < self.ball_size2
        ref_grouped_points.masked_fill_(~mask.unsqueeze(-1), 0.0)
        nball = torch.sum(mask, dim=-1, keepdim=True)
        ref_group_center = torch.sum(ref_grouped_points, dim=2, keepdim=True) / nball.unsqueeze(-1)
        ref_points = ref_grouped_points - ref_group_center
        ref_allpoints = ref_points.view(-1, self.nn_size, C).contiguous()
        U_ref, S_ref, V_ref = batch_svd(ref_allpoints)
        ref_cond = S_ref[:, 0] / (S_ref[:, -1] + S_ref[:, 0])
        ref_cond = ref_cond.view(B, N).contiguous()
        grouped_points = torch.gather(points.unsqueeze(1).expand(-1, N, -1, -1), 2, ref_group_idx.unsqueeze(-1).expand(-1, -1, -1, C))
        grouped_points.masked_fill(~mask.unsqueeze(-1), 0.0)
        group_center = torch.sum(grouped_points, dim=2, keepdim=True) / nball.unsqueeze(-1)
        points = grouped_points - group_center
        allpoints = points.view(-1, self.nn_size, C).contiguous()
        U, S, V = batch_svd(allpoints)
        cond = S[:, 0] / (S[:, -1] + S[:, 0])
        cond = cond.view(B, N).contiguous()
        return self.metric(cond, ref_cond)


class InsideLoss2D(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, cage, shape, shape_normals, epsilon=0.01, interpolate=True):
        """ Penalize polygon cage that is inside the given shape
        Args:
            cage: (B,M,3)
            shape: (B,N,3)
            shape_normals: (B,N,3)
        return:

        """
        B, M, D = cage.shape
        interpolate_n = 10
        cage_p = cage[:, [i for i in range(1, M)] + [0], :]
        t = torch.linspace(0, 1, interpolate_n)
        cage_itp = t.reshape([1, 1, interpolate_n, 1]) * cage_p.unsqueeze(2).expand(-1, -1, interpolate_n, -1) + (1 - t.reshape([1, 1, interpolate_n, 1])) * cage.unsqueeze(2).expand(-1, -1, interpolate_n, -1)
        cage_itp = cage_itp.reshape(B, -1, D)
        nn_point, nn_index, _ = faiss_knn(1, cage_itp, shape, NCHW=False)
        nn_point = nn_point.squeeze(2)
        nn_normal = torch.gather(shape_normals.unsqueeze(1).expand(-1, nn_index.shape[1], -1, -1), 2, nn_index.unsqueeze(-1).expand(-1, -1, -1, shape_normals.shape[-1]))
        nn_normal = nn_normal.squeeze(2)
        dot = dot_product(cage_itp - nn_point - epsilon * nn_normal, nn_normal, dim=-1)
        loss = torch.where(dot < 0, -dot, torch.zeros_like(dot))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'max':
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == 'sum':
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError
        return loss


class InterpolatedCDTriMesh(torch.nn.Module):
    """
    Reconstruction between cage and shape
    mean(shape2cage) + beta*max(shape2cage) + (gamma+delta*|CAGE|*mean(cage2shape))
    """

    def __init__(self, interpolate_n=4, beta=1.0, gamma=1, delta=0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.interpolate_n = 4
        interpolate_n = interpolate_n
        t = torch.linspace(0, 1, interpolate_n)
        sample_weights = torch.meshgrid(t, t)
        sample_weights = torch.stack(sample_weights + (1 - sample_weights[0] - sample_weights[1],), dim=-1).view(-1, 3)
        mask = (sample_weights[:, 2] >= 0).unsqueeze(-1).expand_as(sample_weights)
        self.sample_weights = torch.masked_select(sample_weights, mask).view(-1, 3)
        self.threshold = torch.nn.Hardshrink(0.05)

    def forward(self, cage_v, cage_f, shape, interpolate=True):
        B, M, D = cage_v.shape
        B, F, _ = cage_f.shape
        B, N, _ = shape.shape
        self.sample_weights = self.sample_weights
        cage_face_vertices = torch.gather(cage_v, 1, cage_f.reshape(B, F * 3, 1).expand(-1, -1, cage_v.shape[-1])).reshape(B, F, 1, 3, 3)
        sample_weights = self.sample_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cage_sampled_points = torch.sum(sample_weights * cage_face_vertices, dim=-2).reshape(B, -1, 3)
        cage2shape, shape2cage, _, _ = nndistance(cage_sampled_points, shape)
        shape2cage = self.threshold(shape2cage)
        cage2shape = self.threshold(cage2shape)
        loss = torch.mean(shape2cage, dim=1) * (self.gamma + self.delta * M) + torch.mean(cage2shape, dim=1) + self.beta * torch.max(cage2shape, dim=1)[0]
        loss = torch.mean(loss)
        return loss


class InsideLoss3DTriMesh(torch.nn.Module):
    """Penalize cage inside a triangle mesh
    Args:
        cage_v: (B,M,3)
        cage_f: (B,F,3)
        shape:  (B,N,3)
        shape_f: (B,FF,3)
        shape_fn: (B,FF,3)
    """

    def __init__(self, reduction='mean', interpolate_n=4):
        super().__init__()
        self.reduction = reduction
        interpolate_n = interpolate_n
        t = torch.linspace(0, 1, interpolate_n)
        sample_weights = torch.meshgrid(t, t)
        sample_weights = torch.stack(sample_weights + (1 - sample_weights[0] - sample_weights[1],), dim=-1).view(-1, 3)
        mask = (sample_weights[:, 2] >= 0).unsqueeze(-1).expand_as(sample_weights)
        self.sample_weights = torch.masked_select(sample_weights, mask).view(-1, 3)

    def forward(self, cage_v, cage_f, shape, shape_vn, epsilon=0.01, interpolate=True):
        B, M, D = cage_v.shape
        B, F, _ = cage_f.shape
        B, N, _ = shape.shape
        self.sample_weights = self.sample_weights
        cage_face_vertices = torch.gather(cage_v, 1, cage_f.reshape(B, F * 3, 1).expand(-1, -1, cage_v.shape[-1])).reshape(B, F, 1, 3, 3)
        sample_weights = self.sample_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cage_sampled_points = torch.sum(sample_weights * cage_face_vertices, dim=-2).reshape(B, -1, 3)
        nn_point, nn_index, _ = faiss_knn(1, cage_sampled_points, shape, NCHW=False)
        nn_point = nn_point.squeeze(2)
        nn_normal = torch.gather(shape_vn.unsqueeze(1).expand(-1, nn_index.shape[1], -1, -1), 2, nn_index.unsqueeze(-1).expand(-1, -1, -1, shape_vn.shape[-1]))
        nn_normal = nn_normal.squeeze(2)
        dot = dot_product(cage_sampled_points - nn_point - epsilon * nn_normal, nn_normal, dim=-1)
        loss = torch.where(dot < 0, -dot, torch.zeros_like(dot))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'max':
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == 'sum':
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError
        return loss


class DeformationSharedMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""

    def __init__(self, dim=3, residual=True, normalization='none'):
        super().__init__()
        layer_size = 128
        self.residual = residual
        self.conv1 = Conv1d(dim, layer_size, 1, activation='lrelu', normalization=normalization)
        self.conv2 = Conv1d(layer_size, layer_size, 1, activation='lrelu', normalization=normalization)
        self.conv3 = Conv1d(layer_size, dim, 1, activation=None)

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x += orig
        return x


class MLPDeformer(nn.Module):

    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(Linear(bottleneck_size, 512, activation='lrelu', normalization=normalization), Linear(512, 256, activation='lrelu', normalization=normalization), Linear(256, npoint * dim))

    def forward(self, code, template):
        B, C, N = template.shape
        assert self.npoint == N
        assert self.dim == C
        if code.ndim > 2:
            code = code.view(B, -1)
        x = self.layers(code)
        x = x.reshape(B, C, N)
        if self.residual:
            x += template
        return x


class PointGenCon(nn.Module):

    def __init__(self, bottleneck_size, out_dim, prim_dim, normalization=None, concat_prim=False):
        """
        param:
            cat_prim: keep concatenate atlas coordinate to the features
        """
        super(PointGenCon, self).__init__()
        self.concat_prim = concat_prim
        if concat_prim:
            self.layers = nn.ModuleList([Conv1d(bottleneck_size + prim_dim, bottleneck_size // 2, 1, activation='lrelu', normalization=normalization), Conv1d(bottleneck_size // 2 + prim_dim, bottleneck_size // 4, 1, activation='lrelu', normalization=normalization), Conv1d(bottleneck_size // 4 + prim_dim, out_dim, 1, activation=None, normalization=None)])
        else:
            self.layers = nn.ModuleList([Conv1d(bottleneck_size + prim_dim, bottleneck_size // 2, 1, activation='lrelu', normalization=normalization), Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1, activation='lrelu', normalization=normalization), Conv1d(bottleneck_size // 4, out_dim, 1, activation=None, normalization=None)])

    def forward(self, x, primative):
        if x.ndimension() != primative.ndimension():
            x = x.unsqueeze(-1).expand(-1, -1, primative.shape[-1])
        for i, layer in enumerate(self.layers):
            if self.concat_prim or i == 0:
                x = torch.cat([x, primative], dim=1)
            if i + 1 == len(self.layers):
                xyz = layer(x)
            else:
                x = layer(x)
        return xyz, x


class MultiFoldPointGen(nn.Module):
    """
    :params:
        code (B,C,1) or (B,C)
        primative (B,dim,P)

    :return:
        primative (B,dim,P)
        [point_feat (B,C,P)] decoder's last feature layer before getting the primiative coordinates
    """

    def __init__(self, bottleneck_size, out_dim=3, prim_dim=3, n_fold=3, normalization=None, concat_prim=True, residual=True, return_aux=True):
        super().__init__()
        folds = []
        self.prim_dim = prim_dim
        for i in range(n_fold):
            cur_out_dim = min(bottleneck_size, 64 * (n_fold - i)) if i + 1 < n_fold else 3
            folds += [PointGenCon(bottleneck_size, cur_out_dim, prim_dim, normalization=normalization, concat_prim=concat_prim)]
        self.folds = nn.ModuleList(folds)
        self.return_aux = return_aux
        self.residual = residual
        if self.residual:
            assert prim_dim == out_dim

    def forward(self, code, primative):
        for i, fold in enumerate(self.folds):
            if code.ndimension() != primative.ndimension():
                code_exp = code.unsqueeze(-1).expand(-1, -1, primative.shape[-1])
            else:
                code_exp = code.expand(-1, -1, primative.shape[-1])
            assert primative.shape[1] == self.prim_dim
            xyz, point_feat = fold(code_exp, primative)
        if self.residual:
            xyz = primative + xyz
        if self.return_aux:
            return xyz, point_feat
        return xyz


class PointNet2feat(nn.Module):
    """
    pointcloud (B,3,N)
    return (B,bottleneck_size)
    """

    def __init__(self, dim=3, num_points=2048, num_levels=3, bottleneck_size=512, normalization=None):
        super().__init__()
        assert dim == 3
        self.SA_modules = nn.ModuleList()
        self.postSA_mlp = nn.ModuleList()
        NPOINTS = []
        RADIUS = []
        MLPS = []
        start_radius = 0.2
        start_mlp = 24
        self.l_output = []
        for i in range(num_levels):
            NPOINTS += [num_points // 4]
            num_points = num_points // 4
            RADIUS += [[start_radius]]
            start_radius *= 2
            final_mlp = min(256, start_mlp * 4)
            MLPS += [[[start_mlp, start_mlp * 2, final_mlp]]]
            start_mlp *= 2
            self.l_output.append(start_mlp)
        bottleneck_size_per_SA = bottleneck_size // len(MLPS)
        self.bottleneck_size = bottleneck_size_per_SA * len(MLPS)
        in_channels = 0
        for k in range(len(MLPS)):
            mlps = [([in_channels] + mlp) for mlp in MLPS[k]]
            in_channels = 0
            for idx in range(len(MLPS[k])):
                in_channels += MLPS[k][idx][-1]
            self.SA_modules.append(PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=[32], mlps=mlps, normalization=normalization))
            self.postSA_mlp.append(Conv1d(in_channels, bottleneck_size_per_SA, 1, normalization=normalization, activation='tanh'))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.FloatTensor, return_all=False):
        pointcloud = pointcloud.transpose(1, 2).contiguous()
        li_xyz, li_features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](li_xyz, li_features)
            li_features_post = self.postSA_mlp[i](li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features_post)
        global_code = torch.cat([torch.max(l_feat, dim=-1)[0] for l_feat in l_features], dim=1)
        l_features.append(global_code)
        l_xyz.append(None)
        if return_all:
            return l_features, l_xyz
        else:
            return global_code


class PointNetfeat(nn.Module):

    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation='relu', normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        self.trans = trans
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x, _ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class FixedSourceDeformer(torch.nn.Module):

    def __init__(self, opt, dim, num_points, bottleneck_size, template_vertices=None, template_faces=None, source_vertices=None, source_faces=None, **kwargs):
        super().__init__()
        self.opt = opt
        self.initialized = False
        self.dim = dim
        if opt.pointnet2:
            self.encoder = PointNet2feat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size, normalization=opt.normalization)
            bottleneck_size = self.encoder.bottleneck_size
        else:
            self.encoder = nn.Sequential(PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size, normalization=opt.normalization), Linear(bottleneck_size, bottleneck_size, activation='tanh', normalization=opt.normalization))
        self.initialize_buffers(template_vertices, template_faces, source_vertices, source_faces)
        self.prob = None
        if opt.optimize_template:
            self.template_vertices = nn.Parameter(self.template_vertices)
            logger.info('optimize template cage as parameters')
        if opt.deform_template:
            logger.info('optimize template cage with point fold')
            self.nc_decoder = DeformationSharedMLP(dim, normalization=opt.normalization, residual=opt.c_residual)
        if opt.atlas:
            self.nd_decoder = MultiFoldPointGen(bottleneck_size + dim if opt.use_correspondence else bottleneck_size, dim, n_fold=opt.n_fold, normalization=opt.normalization, concat_prim=opt.concat_prim, return_aux=False, residual=opt.d_residual)
        else:
            self.nd_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size, npoint=self.template_vertices.shape[-1], residual=opt.d_residual, normalization=opt.normalization)

    def initialize_buffers(self, template_vertices=None, template_faces=None, source_vertices=None, source_faces=None):
        if template_vertices is not None:
            assert template_vertices.ndim == 3 and template_vertices.shape[1] == 3
        else:
            template_vertics = torch.zeros((1, self.dim, 1), dtype=torch.float32)
        if template_faces is not None:
            assert template_faces.ndim == 3 and template_faces.shape[2] == 3
        else:
            template_faces = torch.zeros((1, 1, 3), dtype=torch.int64)
        if source_vertices is not None:
            assert source_vertices.ndim == 3 and source_vertices.shape[1] == 3
        else:
            template_vertics = torch.zeros((1, self.dim, 1), dtype=torch.float32)
        if source_faces is not None:
            assert source_faces.ndim == 3 and source_faces.shape[2] == 3
        else:
            source_faces = torch.zeros((1, 1, 3), dtype=torch.int64)
        if not self.initialized:
            self.register_buffer('template_faces', template_faces)
            self.register_buffer('template_vertices', template_vertices)
            self.register_buffer('source_faces', source_faces)
            self.register_buffer('source_vertices', source_vertices)
        else:
            self.template_faces.resize_as_(template_faces).copy_(template_faces)
            self.template_vertices.resize_as_(template_vertices).copy_(template_vertices)
            self.source_faces.resize_as_(source_faces).copy_(source_faces)
            self.source_vertices.resize_as_(source_vertices).copy_(source_vertices)
        self.initialized = True

    def forward(self, target_shape, sample_idx=None, alpha=1.0, cage_only=False):
        """
        source_shape (1,3,M)
        target_shape (B,3,M)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        assert self.initialized
        B, _, N = target_shape.shape
        _, _, P = self.template_vertices.shape
        if sample_idx is not None:
            source_shape = self.source_vertices.expand(B, -1, -1)
            source_shape = torch.gather(source_shape, -1, sample_idx.unsqueeze(1).expand(-1, 3, -1))
        elif self.training and self.source_vertices.shape[-1] != N:
            if self.opt.regular_sampling:
                if self.prob is None:
                    _, farea = compute_face_normals_and_areas(self.source_vertices.transpose(1, 2), self.source_faces)
                    v_area = scatter_add(farea.view(-1, 1).expand(-1, 3).contiguous().view(-1), self.source_faces.view(-1), 0)
                    self.prob = (v_area / torch.sum(v_area)).cpu().numpy()
                random_sample = torch.from_numpy(np.random.choice(self.source_vertices.shape[-1], size=N, p=self.prob).astype(np.int64))
            else:
                random_sample = torch.from_numpy(np.random.choice(self.source_vertices.shape[-1], size=N, replace=False).astype(np.int64))
            source_shape = torch.index_select(self.source_vertices, -1, random_sample)
            source_shape = source_shape.expand(B, -1, -1)
        else:
            source_shape = self.source_vertices.detach()
            source_shape = source_shape.expand(B, -1, -1)
        t_code = self.encoder(target_shape)
        t_code = t_code.unsqueeze(-1)
        template_v = self.template_vertices.view(1, 3, -1)
        if self.opt.deform_template:
            template_v = self.nc_decoder(template_v)
        if self.opt.use_correspondence and self.opt.atlas:
            closest, idx, dist = faiss_knn(3, template_v.expand(B, -1, -1), source_shape, NCHW=True)
            target_xyz = torch.gather(target_shape.unsqueeze(2).expand(-1, -1, P, -1), 3, idx.unsqueeze(1).expand(-1, self.opt.dim, -1, -1))
            target_xyz = torch.median(target_xyz, dim=-1)[0]
            t_code = torch.cat([t_code.expand(-1, -1, P), target_xyz], dim=1).contiguous()
        new_cage = self.nd_decoder(t_code, template_v.view(1, 3, -1).expand(B, -1, -1))
        if not cage_only:
            weights, weights_unnormed = mean_value_coordinates_3D(source_shape.transpose(1, 2), template_v.expand(B, -1, -1).transpose(1, 2), self.template_faces.expand(B, -1, -1), verbose=True)
            mvc = weights.expand(B, -1, -1)
            deformed_shapes = torch.sum(mvc.unsqueeze(-1) * new_cage.transpose(1, 2).unsqueeze(1), dim=2)
        else:
            weights = None
            weights_unnormed = None
            deformed_shapes = None
        cage = template_v.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        return {'source_shape': source_shape.transpose(1, 2), 'cage': cage, 'new_cage': new_cage, 'deformed': deformed_shapes, 'cage_face': self.template_faces, 'weight': weights, 'weight_unnormed': weights_unnormed}


class STN(nn.Module):

    def __init__(self, num_points=2500, dim=3):
        super(STN, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim * dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).expand(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat3DCoded(nn.Module):

    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)
        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_feat = self.bn3(self.conv3(x))
        x, _ = torch.max(point_feat, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2), point_feat


class UnetCageGen(nn.Module):
    """
    Receive sampled feature and location from encoder, for each point in the template,
    find k NN points in l_xyz, concatenate their features to get the code
    Params:
        template   Tensor (B,D,N)
        l_xyz      List(Tensor) of shape (B,N_l,D)
        l_features List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """

    def __init__(self, bottleneck_size, dim=3, knn_k=3, normalization=None, concat_prim=True, n_fold=2, feat_NN=False, **kwargs):
        super().__init__()
        self.decoder = MultiFoldPointGen(bottleneck_size, dim, n_fold=n_fold, normalization=normalization, concat_prim=concat_prim)
        self.feat_NN = feat_NN
        self.knn_k = knn_k

    def interpolate_features(self, query, points, feats, q_normals=None, p_normals=None):
        """
        compute knn point distance and interpolation weight
        :param
            query           (B,M,D)
            points          (B,N,D)
            normals         (B,N,D)
            feats           (B,C,N)
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        B, M, D = query.shape
        feats_t = feats.transpose(1, 2).contiguous()
        grouped_points, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query, points, NCHW=False)
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-08
        weight = torch.exp(-grouped_dist / (h / 2)).detach()
        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW
        pdb.set_trace()
        grouped_feats_ref = torch.gather(feats_t.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1, -1, -1, feats_t.shape[-1]))
        grouped_feats = torch.gather(feats.unsqueeze(2).expand(-1, -1, M, -1), 3, grouped_idx.unsqueeze(1).expand(-1, feats.shape[1], -1, -1))
        None
        weighted_feats = torch.sum(grouped_feats * weight.unsqueeze(1), dim=-1)
        return weighted_feats

    def forward(self, template, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1, 2).contiguous()
        interpolated = []
        for i, xyz_feat in enumerate(zip(l_xyz, l_features)):
            xyz, feat = xyz_feat
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            feat = self.interpolate_features(template_t, xyz, feat, q_normals=None, p_normals=None)
            interpolated += [feat]
        feat_all = torch.cat(interpolated, dim=1)
        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz


class UnetDeformGen(UnetCageGen):
    """
    Params:
        template            Tensor (B,D,N)
        template_features   Tensor (B, sum_l(C_l), N_l) from UnetCageGen
        l_xyz               List(Tensor) of shape (B,N_l,D)
        l_features          List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """

    def interpolate_features(self, query_feats, feats, points):
        """
        find the kNN in feature space, interpolate these feature with exponential weights
        :param
            query_feats (B,C,M)
            feats       (B,C,N)
            points      (B,N,D)
        :return
            weighted_feats (B,C,M)
            weighted_xyz   (B,M,dim)
        """
        B, C, M = query_feats.shape
        query_feats_t = query_feats.transpose(1, 2).contiguous()
        feats_t = feats.transpose(1, 2).contiguous()
        grouped_feats_t, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query_feats_t, feats_t, NCHW=False)
        grouped_feats = grouped_feats_t.permute((0, 3, 1, 2))
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-08
        weight = torch.exp(-grouped_dist / (h / 2)).detach()
        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW
        weighted_feats = torch.sum(grouped_feats * weight.unsqueeze(1), dim=-1)
        grouped_xyz = torch.gather(points.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        weighted_xyz = torch.sum(grouped_xyz * weight.unsqueeze(-1), dim=2)
        return weighted_feats, weighted_xyz

    def forward(self, template, template_features, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1, 2).contiguous()
        interpolated = []
        for i, feat_xyz_feat in enumerate(zip(template_features, l_xyz, l_features)):
            query_feat, xyz, feat = feat_xyz_feat
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            feat, matched_xyz = self.interpolate_features(query_feat, feat, xyz, q_normals=None, p_normals=None)
            interpolated += [feat]
        feat_all = torch.cat(interpolated + [matched_xyz.transpose(1, 2)], dim=1)
        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz


def deform_with_MVC(cage, cage_deformed, cage_face, query, verbose=False):
    """
    cage (B,C,3)
    cage_deformed (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
    deformed = torch.sum(weights.unsqueeze(-1) * cage_deformed.unsqueeze(1), dim=2)
    if verbose:
        return deformed, weights, weights_unnormed
    return deformed


class NetworkFull(nn.Module):

    def __init__(self, opt, dim, bottleneck_size, template_vertices, template_faces, **kargs):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.set_up_template(template_vertices, template_faces)
        if opt.pointnet2:
            self.encoder = PointNet2feat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size)
        else:
            self.encoder = nn.Sequential(PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size), Linear(bottleneck_size, bottleneck_size, activation='lrelu', normalization=opt.normalization))
        if opt.full_net:
            if not opt.atlas:
                self.nc_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size, npoint=self.template_vertices.shape[-1], residual=opt.c_residual, normalization=opt.normalization)
            else:
                self.nc_decoder = MultiFoldPointGen(bottleneck_size, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization, concat_prim=opt.concat_prim, return_aux=False, residual=opt.c_residual)
        self.D_use_C_global_code = opt.c_global
        self.merger = nn.Sequential(Conv1d(bottleneck_size * 2, bottleneck_size * 2, 1, activation='lrelu', normalization=opt.normalization))
        if not opt.atlas:
            self.nd_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size * 2, npoint=self.template_vertices.shape[-1], residual=opt.d_residual, normalization=opt.normalization)
        else:
            self.nd_decoder = MultiFoldPointGen(bottleneck_size * 2, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization, concat_prim=opt.concat_prim, return_aux=False, residual=opt.d_residual)

    def set_up_template(self, template_vertices, template_faces):
        assert template_vertices.ndim == 3 and template_vertices.shape[1] == self.dim
        if self.dim == 3:
            assert template_faces.ndim == 3 and template_faces.shape[2] == 3
        self.register_buffer('template_faces', template_faces)
        self.register_buffer('template_vertices', template_vertices)
        self.template_vertices = nn.Parameter(self.template_vertices, requires_grad=self.opt.optimize_template)
        if self.template_vertices.requires_grad:
            logger.info('Enabled vertex optimization')

    def forward(self, source_shape, target_shape, alpha=1.0):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        init_cage    (B,3,P)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        B, _, N = source_shape.shape
        _, M, N = target_shape.shape
        _, _, P = self.template_vertices.shape
        input_shapes = torch.cat([source_shape, target_shape], dim=0)
        shape_code = self.encoder(input_shapes)
        shape_code.unsqueeze_(-1)
        s_code, t_code = torch.split(shape_code, B, dim=0)
        cage = self.template_vertices.view(1, self.dim, -1).expand(B, -1, -1)
        if self.opt.full_net:
            cage = self.nc_decoder(s_code, cage)
        target_code = torch.cat([s_code, t_code], dim=1)
        target_code = self.merger(target_code)
        new_cage = self.nd_decoder(target_code, cage)
        if self.dim == 3:
            cage = cage.transpose(1, 2).contiguous()
            new_cage = new_cage.transpose(1, 2).contiguous()
            deformed_shapes, weights, weights_unnormed = deform_with_MVC(cage, new_cage, self.template_faces.expand(B, -1, -1), source_shape.transpose(1, 2).contiguous(), verbose=True)
        elif self.dim == 2:
            weights, weights_unnormed = mean_value_coordinates(source_shape, cage, verbose=True)
            deformed_shapes = torch.sum(weights.unsqueeze(1) * new_cage.unsqueeze(-1), dim=2).transpose(1, 2).contiguous()
            cage = cage.transpose(1, 2)
            new_cage = new_cage.transpose(1, 2)
        return {'cage': cage, 'new_cage': new_cage, 'deformed': deformed_shapes, 'cage_face': self.template_faces, 'weight': weights, 'weight_unnormed': weights_unnormed}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GroundingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MVCRegularizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yifita_deep_cage(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

