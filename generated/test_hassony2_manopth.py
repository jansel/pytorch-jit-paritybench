import sys
_module = sys.modules[__name__]
del sys
manopth_demo = _module
manopth_mindemo = _module
mano = _module
webuser = _module
lbs = _module
posemapper = _module
serialization = _module
smpl_handpca_wrapper_HAND_only = _module
verts = _module
manopth = _module
argutils = _module
demo = _module
manolayer = _module
rodrigues_layer = _module
rot6d = _module
rotproj = _module
tensutils = _module
setup = _module
test_demo = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


from torch.nn import Module


from torch.autograd import gradcheck


from torch.autograd import Variable


import warnings


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))


def ready_arguments(fname_or_dict):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        dd = fname_or_dict
    backwards_compatibility_replacements(dd)
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3
    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])
    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if s in dd and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, (0)])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, (1)])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, (2)])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    return dd


def subtract_flat_id(rot_mats):
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], rot_nb)
    results = rot_mats - id_flat
    return results


def th_posemap_axisang(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = rodrigues_layer.batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


class ManoLayer(Module):
    __constants__ = ['use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check', 'side', 'center_idx', 'joint_rot_mode']

    def __init__(self, center_idx=None, flat_hand_mean=True, ncomps=6, side='right', mano_root='mano/models', use_pca=True, root_rot_mode='axisang', joint_rot_mode='axisang', robust_rot=False):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()
        self.center_idx = center_idx
        self.robust_rot = robust_rot
        if root_rot_mode == 'axisang':
            self.rot = 3
        else:
            self.rot = 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45
        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
        smpl_data = ready_arguments(self.mano_path)
        hands_components = smpl_data['hands_components']
        self.smpl_data = smpl_data
        self.register_buffer('th_betas', torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs', torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs', torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer('th_v_template', torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer('th_J_regressor', torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights', torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces', torch.Tensor(smpl_data['f'].astype(np.int32)).long())
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == 'axisang':
            self.register_buffer('th_hands_mean', th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer('th_selected_comps', torch.Tensor(selected_components))
        else:
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer('th_hands_mean_rotmat', th_hands_mean_rotmat)
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self, th_pose_coeffs, th_betas=torch.zeros(1), th_trans=torch.zeros(1), root_palm=torch.Tensor([0]), share_betas=torch.Tensor([0])):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """
        batch_size = th_pose_coeffs.shape[0]
        if self.use_pca or self.joint_rot_mode == 'axisang':
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot + self.ncomps]
            if self.use_pca:
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs
            th_full_pose = torch.cat([th_pose_coeffs[:, :self.rot], self.th_hands_mean + th_full_hand_pose], 1)
            if self.root_rot_mode == 'axisang':
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, 'When not self.use_pca, th_pose_coeffs should have 4 dims, got {}'.format(th_pose_coeffs.dim())
            assert th_pose_coeffs.shape[2:4] == (3, 3), 'When not self.use_pca, th_pose_coeffs have 3x3 matrix for twolast dims, got {}'.format(th_pose_coeffs.shape[2:4])
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, (0)]
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1)
        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
        th_v_posed = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        root_j = th_j[:, (0), :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))
        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, ([(idx - 1) for idx in lev1_idxs])]
        lev2_rots = all_rots[:, ([(idx - 1) for idx in lev2_idxs])]
        lev3_rots = all_rots[:, ([(idx - 1) for idx in lev3_idxs])]
        lev1_j = th_j[:, (lev1_idxs)]
        lev2_j = th_j[:, (lev2_idxs)]
        lev3_j = th_j[:, (lev3_idxs)]
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))
        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, (reorder_idxs)]
        th_results_global = th_results
        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)
        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))
        th_rest_shape_h = torch.cat([th_v_posed.transpose(2, 1), torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device)], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, (3)]
        if self.side == 'right':
            tips = th_verts[:, ([745, 317, 444, 556, 673])]
        else:
            tips = th_verts[:, ([745, 317, 445, 556, 673])]
        if bool(root_palm):
            palm = (th_verts[:, (95)] + th_verts[:, (22)]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)
        th_jtr = th_jtr[:, ([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])]
        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, (self.center_idx)].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)
        th_verts = th_verts * 1000
        th_jtr = th_jtr * 1000
        return th_verts, th_jtr

