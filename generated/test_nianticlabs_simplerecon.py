import sys
_module = sys.modules[__name__]
del sys
generate_test_tuples = _module
generate_train_tuples = _module
ios_logger_preprocessing = _module
precompute_valid_frames = _module
SensorData = _module
download_scannet = _module
reader = _module
arkit_dataset = _module
colmap_dataset = _module
generic_mvs_dataset = _module
scannet_dataset = _module
scanniverse_dataset = _module
seven_scenes_dataset = _module
vdr_dataset = _module
depth_model = _module
losses = _module
cost_volume = _module
layers = _module
networks = _module
options = _module
pc_fusion = _module
test = _module
fusers_helper = _module
keyframe_buffer = _module
mesh_renderer = _module
torch_point_cloud_fusion = _module
tsdf = _module
train = _module
dataset_utils = _module
generic_utils = _module
geometry_utils = _module
metrics_utils = _module
visualization_utils = _module
generate_gt_min_max_cache = _module
load_meshes_and_include_normals = _module
visualize_scene_depth_output = _module
visualize_live_meshing = _module
strip_checkpoint = _module

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


import logging


import numpy as np


import torch


from torchvision import transforms


import functools


import random


from torch.utils.data import Dataset


import re


from scipy.spatial.transform import Rotation as R


import torch.nn.functional as F


from torch import nn


import torch.jit as jit


from torch import Tensor


import torch.nn as nn


from typing import Callable


from typing import Optional


from torchvision import models


from torchvision.ops import FeaturePyramidNetwork


from typing import Tuple


import torch.nn.functional as TF


from torch.utils.data import DataLoader


import torchvision.transforms.functional as TF


import matplotlib.pyplot as plt


import scipy


class MSGradientLoss(nn.Module):

    def __init__(self, num_scales: int=4):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, depth_gt: Tensor, depth_pred: Tensor) ->Tensor:
        depth_pred_pyr = pyrdown(depth_pred, self.num_scales)
        depth_gtn_pyr = pyrdown(depth_gt, self.num_scales)
        grad_loss = torch.tensor(0, dtype=depth_gt.dtype, device=depth_gt.device)
        for depth_pred_down, depth_gtn_down in zip(depth_pred_pyr, depth_gtn_pyr):
            depth_gtn_grad = kornia.filters.spatial_gradient(depth_gtn_down)
            mask_down_b = depth_gtn_grad.isfinite().all(dim=1, keepdim=True)
            depth_pred_grad = kornia.filters.spatial_gradient(depth_pred_down).masked_select(mask_down_b)
            grad_error = torch.abs(depth_pred_grad - depth_gtn_grad.masked_select(mask_down_b))
            grad_loss += torch.mean(grad_error)
        return grad_loss


class ScaleInvariantLoss(jit.ScriptModule):

    def __init__(self, si_lambda: float=0.85):
        super().__init__()
        self.si_lambda = si_lambda

    @jit.script_method
    def forward(self, log_depth_gt: Tensor, log_depth_pred: Tensor) ->Tensor:
        log_diff = log_depth_gt - log_depth_pred
        si_loss = torch.sqrt((log_diff ** 2).mean() - self.si_lambda * log_diff.mean() ** 2)
        return si_loss


class NormalsLoss(nn.Module):

    def forward(self, normals_gt_b3hw: Tensor, normals_pred_b3hw: Tensor) ->Tensor:
        normals_mask_b1hw = torch.logical_and(normals_gt_b3hw.isfinite().all(dim=1, keepdim=True), normals_pred_b3hw.isfinite().all(dim=1, keepdim=True))
        normals_pred_b3hw = normals_pred_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        normals_gt_b3hw = normals_gt_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        with torch.amp.autocast(enabled=False):
            normals_dot_b1hw = 0.5 * (1.0 - torch.einsum('bchw, bchw -> bhw', normals_pred_b3hw, normals_gt_b3hw)).unsqueeze(1)
        normals_loss = normals_dot_b1hw.masked_select(normals_mask_b1hw).mean()
        return normals_loss


@torch.jit.script
def to_homogeneous(input_tensor: Tensor, dim: int=0) ->Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified 
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN


class BackprojectDepth(jit.ScriptModule):
    """
    Layer that projects points from 2D camera to 3D space. The 3D points are 
    represented in homogeneous coordinates.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        xx, yy = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy')
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5
        pix_coords_13N = to_homogeneous(pix_coords_2hw, dim=0).flatten(1).unsqueeze(0)
        self.register_buffer('pix_coords_13N', pix_coords_13N)

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) ->Tensor:
        """ 
        Backprojects spatial points in 2D image space to world space using 
        invK_b44 at the depths defined in depth_b1hw. 
        """
        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N)
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N


class Project3D(jit.ScriptModule):
    """
    Layer that projects 3D points into the 2D camera
    """

    def __init__(self, eps: float=1e-05):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps).view(1, 1, 1))

    @jit.script_method
    def forward(self, points_b4N: Tensor, K_b44: Tensor, cam_T_world_b44: Tensor) ->Tensor:
        """ 
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44
        cam_points_b3N = P_b44[:, :3] @ points_b4N
        depth_b1N = torch.maximum(cam_points_b3N[:, 2:], self.eps)
        pix_coords_b2N = cam_points_b3N[:, :2] / depth_b1N
        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


class MVDepthLoss(nn.Module):

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.backproject = BackprojectDepth(self.height, self.width)
        self.project = Project3D()

    def get_valid_mask(self, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]
        cur_cam_points_b4N = self.backproject(cur_depth_b1hw, cur_invK_b44)
        world_points_b4N = cur_world_T_cam_b44 @ cur_cam_points_b4N
        src_cam_points_b3N = self.project(world_points_b4N, src_K_b44, src_cam_T_world_b44)
        cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pix_coords_b2hw = cam_points_b3hw[:, :2]
        proj_src_depths_b1hw = cam_points_b3hw[:, 2:]
        uv_coords = pix_coords_b2hw.permute(0, 2, 3, 1) / torch.tensor([depth_width, depth_height]).view(1, 1, 1, 2).type_as(pix_coords_b2hw)
        uv_coords = 2 * uv_coords - 1
        src_depth_sampled_b1hw = F.grid_sample(input=src_depth_b1hw, grid=uv_coords, padding_mode='zeros', mode='nearest', align_corners=False)
        valid_mask_b1hw = proj_src_depths_b1hw < 1.05 * src_depth_sampled_b1hw
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, proj_src_depths_b1hw > 0)
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, src_depth_sampled_b1hw > 0)
        return valid_mask_b1hw, src_depth_sampled_b1hw

    def get_error_for_pair(self, depth_pred_b1hw, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]
        valid_mask_b1hw, src_depth_sampled_b1hw = self.get_valid_mask(cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44)
        pred_cam_points_b4N = self.backproject(depth_pred_b1hw, cur_invK_b44)
        pred_world_points_b4N = cur_world_T_cam_b44 @ pred_cam_points_b4N
        src_cam_points_b3N = self.project(pred_world_points_b4N, src_K_b44, src_cam_T_world_b44)
        pred_cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pred_src_depths_b1hw = pred_cam_points_b3hw[:, 2:]
        depth_diff_b1hw = torch.abs(torch.log(src_depth_sampled_b1hw) - torch.log(pred_src_depths_b1hw)).masked_select(valid_mask_b1hw)
        depth_loss = depth_diff_b1hw.nanmean()
        return depth_loss

    def forward(self, depth_pred_b1hw, cur_depth_b1hw, src_depth_bk1hw, cur_invK_b44, src_K_bk44, cur_world_T_cam_b44, src_cam_T_world_bk44):
        src_to_iterate = [torch.unbind(src_depth_bk1hw, dim=1), torch.unbind(src_K_bk44, dim=1), torch.unbind(src_cam_T_world_bk44, dim=1)]
        num_src_frames = src_depth_bk1hw.shape[1]
        loss = 0
        for src_depth_b1hw, src_K_b44, src_cam_T_world_b44 in zip(*src_to_iterate):
            error = self.get_error_for_pair(depth_pred_b1hw, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44)
            loss += error
        return loss / num_src_frames


class CostVolumeManager(nn.Module):
    """
    Class to build a cost volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    collapsing over views by taking a dot product between each source and 
    reference feature, before summing over source views at each pixel location. 
    The final tensor is size batch_size x num_depths x H x  W tensor.
    """

    def __init__(self, matching_height, matching_width, num_depth_bins=64, matching_dim_size=None, num_source_views=None):
        """
        matching_dim_size and num_source_views are not used for the standard 
        cost volume.

        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            matching_dim_size: number of channels per visual feature; the basic 
                dot product cost volume does not need this information at init.
            num_source_views: number of source views; the basic dot product cost 
                volume does not need this information at init.
        """
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.initialise_for_projection()

    def initialise_for_projection(self):
        """
        Set up for backwarping and projection of feature maps

        Args:
            batch_height: height of the current batch of features
            batch_width: width of the current batch of features
        """
        linear_ramp = torch.linspace(0, 1, self.num_depth_bins).view(1, self.num_depth_bins, 1, 1)
        self.register_buffer('linear_ramp_1d11', linear_ramp)
        self.backprojector = BackprojectDepth(height=self.matching_height, width=self.matching_width)
        self.projector = Project3D()

    def get_mask(self, pix_coords_bk2hw):
        """
        Create a mask to ignore features from the edges or outside of source 
        images.
        
        Args:
            pix_coords_bk2hw: sampling locations of source features
            
        Returns:
            mask: a binary mask indicating whether to ignore a pixels
        """
        mask = torch.logical_and(torch.logical_and(pix_coords_bk2hw[:, :, 0] > 2, pix_coords_bk2hw[:, :, 0] < self.matching_width - 2), torch.logical_and(pix_coords_bk2hw[:, :, 1] > 2, pix_coords_bk2hw[:, :, 1] < self.matching_height - 2))
        return mask

    def generate_depth_planes(self, batch_size: int, min_depth: Tensor, max_depth: Tensor) ->Tensor:
        """
        Creates a depth planes tensor of size batch_size x number of depth planes
        x matching height x matching width. Every plane contains the same depths
        and depths will vary with a log scale from min_depth to max_depth.

        Args:
            batch_size: number of these view replications to make for each 
                element in the batch.
            min_depth: minimum depth tensor defining the starting point for 
                depth planes.
            max_depth: maximum depth tensor defining the end point for 
                depth planes.

        Returns:
            depth_planes_bdhw: depth planes tensor.
        """
        linear_ramp_bd11 = self.linear_ramp_1d11.expand(batch_size, self.num_depth_bins, 1, 1)
        log_depth_planes_bd11 = torch.log(min_depth) + torch.log(max_depth / min_depth) * linear_ramp_bd11
        depth_planes_bd11 = torch.exp(log_depth_planes_bd11)
        depth_planes_bdhw = depth_planes_bd11.expand(batch_size, self.num_depth_bins, self.matching_height, self.matching_width)
        return depth_planes_bdhw

    def warp_features(self, src_feats, src_extrinsics, src_Ks, cur_invK, depth_plane_b1hw, batch_size, num_src_frames, num_feat_channels, uv_scale):
        """
        Warps every soruce view feature to the current view at the depth 
        plane defined by depth_plane_b1hw.

        Args:
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            depth_plane_b1hw: depth plane to use for every spatial location. For 
                SimpleRecon, this will be the same value at each location.
            batch_size: the batch size.
            num_src_frames: number of source views.
            num_feat_channels: number of feature channels for feature maps.
            uv_scale: normalization for image space coords before grid_sample.

        Returns:
            world_points_B4N: the world points at every backprojected depth 
                point in depth_plane_b1hw.
            depths: depths for each projected point in every source views.
            src_feat_warped: warped source view for every spatial location at 
                the depth plane.
            mask: depth mask where 1.0 indicated that the point projected to the
                source view is infront of the view.
        """
        world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
        world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, dim=0)
        cam_points_B3N = self.projector(world_points_B4N, src_Ks.view(-1, 4, 4), src_extrinsics.view(-1, 4, 4))
        cam_points_B3hw = cam_points_B3N.view(-1, 3, self.matching_height, self.matching_width)
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        depths = cam_points_B3hw[:, 2:]
        uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1
        src_feat_warped = F.grid_sample(input=src_feats.view(-1, num_feat_channels, self.matching_height, self.matching_width), grid=uv_coords.type_as(src_feats), padding_mode='zeros', mode='bilinear', align_corners=False)
        src_feat_warped = src_feat_warped.view(batch_size, num_src_frames, num_feat_channels, self.matching_height, self.matching_width)
        depths = depths.view(batch_size, num_src_frames, self.matching_height, self.matching_width)
        mask_b = depths > 0
        mask = mask_b.type_as(src_feat_warped)
        return world_points_B4N, depths, src_feat_warped, mask

    def build_cost_volume(self, cur_feats: Tensor, src_feats: Tensor, src_extrinsics: Tensor, src_poses: Tensor, src_Ks: Tensor, cur_invK: Tensor, min_depth: Tensor, max_depth: Tensor, depth_planes_bdhw: Tensor=None, return_mask: bool=False):
        """
        Build the cost volume. Using hypothesised depths, we backwarp src_feats 
        onto cur_feats using known intrinsics and take the dot product. 
        We sum the dot over all src_feats.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """
        del src_poses, return_mask
        batch_size, num_src_frames, num_feat_channels, _, _ = src_feats.shape
        uv_scale = torch.tensor([1 / self.matching_width, 1 / self.matching_height], dtype=src_extrinsics.dtype, device=src_extrinsics.device).view(1, 1, 1, 2)
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, min_depth, max_depth)
        all_dps = []
        for depth_id in range(self.num_depth_bins):
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            _, _, src_feat_warped, mask = self.warp_features(src_feats, src_extrinsics, src_Ks, cur_invK, depth_plane_b1hw, batch_size, num_src_frames, num_feat_channels, uv_scale)
            dot_product_bkhw = torch.sum(src_feat_warped * cur_feats.unsqueeze(1), dim=2) * mask
            dot_product_b1hw = dot_product_bkhw.sum(dim=1, keepdim=True)
            all_dps.append(dot_product_b1hw)
        cost_volume = torch.cat(all_dps, dim=1)
        return cost_volume, depth_planes_bdhw, None

    def indices_to_disparity(self, indices, depth_planes_bdhw):
        """ Convert cost volume indices to 1/depth for visualisation """
        depth = torch.gather(depth_planes_bdhw, dim=1, index=indices.unsqueeze(1)).squeeze(1)
        return depth

    def forward(self, cur_feats, src_feats, src_extrinsics, src_poses, src_Ks, cur_invK, min_depth, max_depth, depth_planes_bdhw=None, return_mask=False):
        """ Runs the cost volume and gets the lowest cost result """
        cost_volume, depth_planes_bdhw, overall_mask_bhw = self.build_cost_volume(cur_feats=cur_feats, src_feats=src_feats, src_extrinsics=src_extrinsics, src_Ks=src_Ks, cur_invK=cur_invK, src_poses=src_poses, min_depth=min_depth, max_depth=max_depth, depth_planes_bdhw=depth_planes_bdhw, return_mask=return_mask)
        with torch.no_grad():
            lowest_cost = self.indices_to_disparity(torch.argmax(cost_volume.detach(), 1), depth_planes_bdhw)
        return cost_volume, lowest_cost, depth_planes_bdhw, overall_mask_bhw


class MLP(nn.Module):

    def __init__(self, channel_list, disable_final_activation=False):
        super(MLP, self).__init__()
        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(nn.Linear(channel_list[layer_index], channel_list[layer_index + 1]))
            layer_list.append(nn.LeakyReLU(inplace=True))
        if disable_final_activation:
            layer_list = layer_list[:-1]
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


def pose_distance(pose_b44):
    """
    DVMVS frame pose distance.
    """
    R = pose_b44[:, :3, :3]
    t = pose_b44[:, :3, 3]
    R_trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    R_measure = torch.sqrt(2 * (1 - torch.minimum(torch.ones_like(R_trace) * 3.0, R_trace) / 3))
    t_measure = torch.norm(t, dim=1)
    combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure


def tensor_bM_to_B(tensor_bMS):
    """Packs an inflated tensor of tupled elements (bMS) into BS. Tuple size 
        is M."""
    num_views = tensor_bMS.shape[1]
    num_batches = tensor_bMS.shape[0]
    tensor_BS = tensor_bMS.view([num_views * num_batches] + list(tensor_bMS.shape[2:]))
    return tensor_BS


def combine_dims(x, dim_begin, dim_end):
    """Views x with the dimensions from dim_begin to dim_end folded."""
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


def get_camera_rays(world_T_cam_b44, world_points_b3N, in_camera_frame, cam_T_world_b44=None, eps=0.0001):
    """
    Computes camera rays for given camera data and points, optionally shifts 
    rays to camera frame.
    """
    if in_camera_frame:
        batch_size = world_points_b3N.shape[0]
        num_points = world_points_b3N.shape[2]
        world_points_b4N = torch.cat([world_points_b3N, torch.ones(batch_size, 1, num_points)], 1)
        camera_points_b3N = torch.matmul(cam_T_world_b44[:, :3, :4], world_points_b4N)
        rays_b3N = camera_points_b3N
    else:
        rays_b3N = world_points_b3N - world_T_cam_b44[:, 0:3, 3][:, :, None].expand(world_points_b3N.shape)
    rays_b3N = torch.nn.functional.normalize(rays_b3N, dim=1)
    return rays_b3N


def tensor_B_to_bM(tensor_BS, batch_size, num_views):
    """Unpacks a flattened tensor of tupled elements (BS) into bMS. Tuple size 
        is M."""
    tensor_bMS = tensor_BS.view([batch_size, num_views] + list(tensor_BS.shape[1:]))
    return tensor_bMS


class FeatureVolumeManager(CostVolumeManager):
    """
    Class to build a feature volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    running an MLP on both visual features and each spatial and depth 
    index's metadata. The final tensor is size 
    batch_size x num_depths x H x  W tensor.

    """

    def __init__(self, matching_height, matching_width, num_depth_bins=64, mlp_channels=[202, 128, 128, 1], matching_dim_size=16, num_source_views=7):
        """
        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            mlp_channels: number of channels at every input/output of the MLP.
                mlp_channels[-1] defines the output size. mlp_channels[0] will 
                be ignored and computed in this initialization function to 
                account for all metadata.
            matching_dim_size: number of channels per visual feature.
            num_source_views: number of source views.
        """
        super().__init__(matching_height, matching_width, num_depth_bins)
        num_visual_channels = matching_dim_size * (1 + num_source_views)
        num_depth_channels = 1 + num_source_views
        num_ray_channels = 3 * (1 + num_source_views)
        num_ray_angle_channels = num_source_views
        num_mask_channels = num_source_views
        num_num_dot_channels = num_source_views
        num_pose_penalty_channels = 3 * num_source_views
        mlp_channels[0] = num_visual_channels + num_depth_channels + num_ray_channels + num_ray_angle_channels + num_mask_channels + num_num_dot_channels + num_pose_penalty_channels
        self.mlp = MLP(channel_list=mlp_channels, disable_final_activation=True)
        None
        None
        None
        None
        None
        None
        None

    def build_cost_volume(self, cur_feats: Tensor, src_feats: Tensor, src_extrinsics: Tensor, src_poses: Tensor, src_Ks: Tensor, cur_invK: Tensor, min_depth: Tensor, max_depth: Tensor, depth_planes_bdhw: Tensor=None, return_mask: bool=False):
        """
        Build the feature volume. Using hypothesised depths, we backwarp 
        src_feats onto cur_feats using known intrinsics and run an MLP on both 
        visual features and each pixel and depth plane's metadata.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """
        batch_size, num_src_frames, num_feat_channels, src_feat_height, src_feat_width = src_feats.shape
        uv_scale = torch.tensor([1 / self.matching_width, 1 / self.matching_height], dtype=src_extrinsics.dtype, device=src_extrinsics.device).view(1, 1, 1, 2)
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, min_depth, max_depth)
        frame_pose_dist_B, r_measure_B, t_measure_B = pose_distance(tensor_bM_to_B(src_poses))
        frame_pose_dist_bkhw = tensor_B_to_bM(frame_pose_dist_B, batch_size=batch_size, num_views=num_src_frames)[:, :, None, None].expand(batch_size, num_src_frames, src_feat_height, src_feat_width)
        r_measure_bkhw = tensor_B_to_bM(r_measure_B, batch_size=batch_size, num_views=num_src_frames)[:, :, None, None].expand(frame_pose_dist_bkhw.shape)
        t_measure_bkhw = tensor_B_to_bM(t_measure_B, batch_size=batch_size, num_views=num_src_frames)[:, :, None, None].expand(frame_pose_dist_bkhw.shape)
        overall_mask_bhw = None
        if return_mask:
            overall_mask_bhw = torch.zeros((batch_size, self.matching_height, self.matching_width), device=src_feats.device, dtype=torch.bool)
        all_dps = []
        for depth_id in range(self.num_depth_bins):
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
            world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, dim=0)
            cam_points_B3N = self.projector(world_points_B4N, src_Ks.view(-1, 4, 4), src_extrinsics.view(-1, 4, 4))
            cam_points_B3hw = cam_points_B3N.view(-1, 3, self.matching_height, self.matching_width)
            pix_coords_B2hw = cam_points_B3hw[:, :2]
            depths = cam_points_B3hw[:, 2:]
            uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1
            src_feat_warped = F.grid_sample(input=src_feats.view(-1, num_feat_channels, self.matching_height, self.matching_width), grid=uv_coords.type_as(src_feats), padding_mode='zeros', mode='bilinear', align_corners=False)
            src_feat_warped = src_feat_warped.view(batch_size, num_src_frames, num_feat_channels, self.matching_height, self.matching_width)
            depths = depths.view(batch_size, num_src_frames, self.matching_height, self.matching_width)
            mask_b = depths > 0
            mask = mask_b.type_as(src_feat_warped)
            if return_mask:
                depth_mask = torch.any(mask_b, dim=1)
                pix_coords_bk2hw = pix_coords_B2hw.view(batch_size, num_src_frames, 2, self.matching_height, self.matching_width)
                bounds_mask = torch.any(self.get_mask(pix_coords_bk2hw), dim=1)
                overall_mask_bhw = torch.logical_and(depth_mask, bounds_mask)
            cur_points_rays_B3hw = F.normalize(world_points_B4N[:, :3, :], dim=1).view(-1, 3, self.matching_height, self.matching_width)
            cur_points_rays_bk3hw = tensor_B_to_bM(cur_points_rays_B3hw, batch_size=batch_size, num_views=num_src_frames)
            src_poses_B44 = tensor_bM_to_B(src_poses)
            src_points_rays_B3hw = get_camera_rays(src_poses_B44, world_points_B4N[:, :3, :], in_camera_frame=False).view(-1, 3, self.matching_height, self.matching_width)
            src_points_rays_bk3hw = tensor_B_to_bM(src_points_rays_B3hw, batch_size=batch_size, num_views=num_src_frames)
            all_rays_bchw = combine_dims(torch.cat([cur_points_rays_bk3hw[:, 0, :, :, :][:, None, :, :, :], src_points_rays_bk3hw], dim=1), 1, 3)
            ray_angle_bkhw = F.cosine_similarity(cur_points_rays_bk3hw, src_points_rays_bk3hw, dim=2, eps=1e-05)
            dot_product_bkhw = torch.sum(src_feat_warped * cur_feats.unsqueeze(1), dim=2) * mask
            combined_visual_features_bchw = combine_dims(torch.cat([src_feat_warped, cur_feats.unsqueeze(1)], dim=1), 1, 3)
            mlp_input_features_bchw = torch.cat([combined_visual_features_bchw, mask, depths, depth_plane_b1hw, dot_product_bkhw, ray_angle_bkhw, all_rays_bchw, frame_pose_dist_bkhw, r_measure_bkhw, t_measure_bkhw], dim=1)
            mlp_input_features_bhwc = mlp_input_features_bchw.permute(0, 2, 3, 1)
            feature_b1hw = self.mlp(mlp_input_features_bhwc).squeeze(-1).unsqueeze(1)
            all_dps.append(feature_b1hw)
        feature_volume = torch.cat(all_dps, dim=1)
        return feature_volume, depth_planes_bdhw, overall_mask_bhw

    def to_fast(self) ->'FastFeatureVolumeManager':
        manager = FastFeatureVolumeManager(self.matching_height, self.matching_width, num_depth_bins=self.num_depth_bins)
        manager.mlp = self.mlp
        return manager


def conv1x1(in_planes: int, out_planes: int, stride: int=1, bias: bool=False) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1, bias: bool=False) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=bias, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=nn.Identity) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        if norm_layer == nn.Identity:
            bias = True
        else:
            bias = False
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes)
        if inplanes == planes * self.expansion and stride == 1:
            self.downsample = None
        else:
            conv = conv1x1 if stride == 1 else conv3x3
            self.downsample = nn.Sequential(conv(inplanes, planes * self.expansion, bias=bias, stride=stride), norm_layer(planes * self.expansion))
        self.stride = stride

    def forward(self, x: Tensor) ->Tensor:
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


class TensorFormatter(nn.Module):
    """Helper to format, apply operation, format back tensor.

    Class to format tensors of shape B x D x C_i x H x W into B*D x C_i x H x W,
    apply an operation, and reshape back into B x D x C_o x H x W.

    Used for multidepth - batching feature extraction on source images"""

    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.depth_chns = None

    def _expand_batch_with_channels(self, x):
        if x.dim() != 5:
            raise ValueError('TensorFormatter expects tensors with 5 dimensions, not {}!'.format(len(x.shape)))
        self.batch_size, self.depth_chns, chns, height, width = x.shape
        x = x.view(self.batch_size * self.depth_chns, chns, height, width)
        return x

    def _reduce_batch_to_channels(self, x):
        if self.batch_size is None or self.depth_chns is None:
            raise ValueError('Cannot  call _reduce_batch_to_channels without first calling_expand_batch_with_channels!')
        _, chns, height, width = x.shape
        x = x.view(self.batch_size, self.depth_chns, chns, height, width)
        return x

    def forward(self, x, apply_func):
        x = self._expand_batch_with_channels(x)
        x = apply_func(x)
        x = self._reduce_batch_to_channels(x)
        return x


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f'conv_{i}', BasicBlock(num_ch_out, num_ch_out))
    return layers


def upsample(x):
    """
    Upsample input tensor by a factor of 2
    """
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class DepthDecoderPP(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])
        self.convs = nn.ModuleDict()
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):
                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0
                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f'diag_conv_{i + 1}{j - 1}'] = BasicBlock(num_ch_in, num_ch_out)
                total_num_ch_in += num_ch_out
                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f'right_conv_{i}{j - 1}'] = BasicBlock(num_ch_in, num_ch_out)
                total_num_ch_in += num_ch_out
                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f'up_conv_{i + 1}{j}'] = BasicBlock(num_ch_in, num_ch_out)
                    total_num_ch_in += num_ch_out
                self.convs[f'in_conv_{i}{j}'] = double_basic_block(total_num_ch_in, num_ch_out)
                self.convs[f'output_{i}'] = nn.Sequential(BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(), nn.Conv2d(num_ch_out, self.num_output_channels, 1))

    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):
                inputs = [self.convs[f'right_conv_{i}{j - 1}'](prev_outputs[i])]
                inputs += [upsample(self.convs[f'diag_conv_{i + 1}{j - 1}'](prev_outputs[i + 1]))]
                if i + j != 4:
                    inputs += [upsample(self.convs[f'up_conv_{i + 1}{j}'](outputs[-1]))]
                output = self.convs[f'in_conv_{i}{j}'](torch.cat(inputs, dim=1))
                outputs += [output]
                depth_outputs[f'log_depth_pred_s{i}_b1hw'] = self.convs[f'output_{i}'](output)
            prev_outputs = outputs[::-1]
        return depth_outputs


class CVEncoder(nn.Module):

    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()
        self.convs = nn.ModuleDict()
        self.num_ch_enc = []
        self.num_blocks = len(num_ch_outs)
        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f'ds_conv_{i}'] = BasicBlock(num_ch_in, num_ch_out, stride=1 if i == 0 else 2)
            self.convs[f'conv_{i}'] = nn.Sequential(BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1), BasicBlock(num_ch_out, num_ch_out, stride=1))
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f'ds_conv_{i}'](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f'conv_{i}'](x)
            outputs.append(x)
        return outputs


class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, num_ch_out, pretrained=True, antialiased=True):
        super().__init__()
        self.num_ch_enc = np.array([64, 64])
        model_source = antialiased_cnns if antialiased else models
        resnets = {(18): model_source.resnet18, (34): model_source.resnet34, (50): model_source.resnet50, (101): model_source.resnet101, (152): model_source.resnet152}
        if num_layers not in resnets:
            raise ValueError('{} is not a valid number of resnet layers'.format(num_layers))
        encoder = resnets[num_layers](pretrained)
        resnet_backbone = [encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, encoder.layer1]
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        self.num_ch_out = num_ch_out
        self.net = nn.Sequential(*resnet_backbone, nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True), nn.Conv2d(128, self.num_ch_out, (3, 3), padding=1, padding_mode='replicate'), nn.InstanceNorm2d(self.num_ch_out))

    def forward(self, input_image):
        return self.net(input_image)


class FPNMatchingEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('mnasnet_100', pretrained=True, features_only=True)
        self.decoder = FeaturePyramidNetwork(self.encoder.feature_info.channels(), out_channels=32)
        self.outconv = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(32, 16, 1), nn.InstanceNorm2d(16))

    def forward(self, x):
        encoder_feats = {f'feat_{i}': f for i, f in enumerate(self.encoder(x))}
        return self.outconv(self.decoder(encoder_feats)['feat_1'])


class NormalGenerator(jit.ScriptModule):

    def __init__(self, height: int, width: int, smoothing_kernel_size: int=5, smoothing_kernel_std: float=2.0):
        """ 
        Estimates normals from depth maps.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.backproject = BackprojectDepth(self.height, self.width)
        self.kernel_size = smoothing_kernel_size
        self.std = smoothing_kernel_std

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) ->Tensor:
        """ 
        First smoothes incoming depth maps with a gaussian blur, backprojects 
        those depth points into world space (see BackprojectDepth), estimates
        the spatial gradient at those points, and finally uses normalized cross 
        correlation to estimate a normal vector at each location.

        """
        depth_smooth_b1hw = kornia.filters.gaussian_blur2d(depth_b1hw, (self.kernel_size, self.kernel_size), (self.std, self.std))
        cam_points_b4N = self.backproject(depth_smooth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)
        gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)
        return F.normalize(torch.cross(gradients_b32hw[:, :, 0], gradients_b32hw[:, :, 1], dim=1), dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CostVolumeManager,
     lambda: ([], {'matching_height': 4, 'matching_width': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'channel_list': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_nianticlabs_simplerecon(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

