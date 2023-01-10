import sys
_module = sys.modules[__name__]
del sys
block_nerf = _module
block_visualize = _module
learning_utils = _module
metrics = _module
waymo_dataset = _module
Madoka = _module
Madoka_long = _module
Otobai = _module
default_forward_facing = _module
default_ubd_inward_facing = _module
sm01_desktop = _module
sm02_multiple_desktop = _module
sm03_meeting = _module
default = _module
block_0 = _module
default_waymo = _module
create_cluster_masks = _module
data_preprocess = _module
fetch_data_from_tf_record = _module
gdown_download_folder = _module
get_one_block_meta_and_visualize = _module
split_block = _module
vis = _module
visualize_cameras = _module
parse_markdown = _module
render_docs_from_csv = _module
eval = _module
eval_block_nerf = _module
lib = _module
dbvgo = _module
dcvgo = _module
dmpigo = _module
dvgo = _module
grid = _module
load_blendedmvs = _module
load_blender = _module
load_co3d = _module
load_data = _module
load_deepvoxels = _module
load_llff = _module
load_nerfpp = _module
load_nsvf = _module
load_tankstemple = _module
masked_adam = _module
utils = _module
mega_nerf = _module
datasets = _module
dataset_utils = _module
filesystem_dataset = _module
memory_dataset = _module
image_metadata = _module
metrics = _module
misc_utils = _module
models = _module
cascade = _module
mega_nerf = _module
mega_nerf_container = _module
model_utils = _module
nerf = _module
opts = _module
ray_utils = _module
rendering = _module
runner = _module
spherical_harmonics = _module
merge_submodules = _module
run = _module
setup = _module
tools = _module
colmap_read_model = _module
colmap_wrapper = _module
pose_utils = _module
imgs2poses = _module
vis_train = _module
vis_volume = _module
train_mega_nerf = _module

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


from torch.optim import SGD


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


import numpy as np


from torchvision import transforms


from torch.utils.data import Dataset


import torch.distributed as dist


from torch.distributed.elastic.multiprocessing.errors import record


import tensorflow as tf


from collections import defaultdict


import random


import copy


import torchvision


import torchvision.transforms as T


import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R


import time


import functools


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.cpp_extension import load


from torch import Tensor


import scipy


import math


import scipy.signal


from typing import List


from typing import Optional


from typing import Tuple


from itertools import cycle


from typing import Dict


from typing import Union


from typing import Type


from torch import nn


from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


from typing import Callable


from torch.cuda.amp import GradScaler


from torch.optim.lr_scheduler import ExponentialLR


from torch.utils.data import DistributedSampler


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms as T


class Alphas2Weights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(alpha, weights, T, alphainv_last, i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


class Raw2Alpha(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1, 1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1, -1).expand(shape).flatten()
    return ray_id, step_id


class DirectBiVoxGO(nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None, mask_cache_world_size=None, fast_color_thres=0, bg_preserve=0.5, density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, bg_use_mlp=True, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, **kwargs):
        super(DirectBiVoxGO, self).__init__()
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectBiVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1, -1, -1]))
        self.register_buffer('xyz_max', torch.Tensor([1, 1, 1]))
        self.fast_color_thres = fast_color_thres
        self.bg_preserve = bg_preserve
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        None
        self._set_grid_resolution(num_voxels)
        self.density_type = density_type
        self.density_config = density_config
        self.density = nn.ModuleList([grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config) for _ in range(2)])
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = nn.ModuleList([grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config) for _ in range(2)])
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = nn.ModuleList([grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config) for _ in range(2)])
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            dim0 += self.k0_dim
            self.rgbnet = nn.ModuleList([nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3)) for _ in range(2)])
            nn.init.constant_(self.rgbnet[0][-1].bias, 0)
            nn.init.constant_(self.rgbnet[1][-1].bias, 0)
            if not bg_use_mlp:
                self.k0[1] = grid.create_grid(k0_type, channels=3, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
                self.rgbnet[1] = None
            None
            None
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = nn.ModuleList([grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max) for _ in range(2)])

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        None
        None
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'num_voxels_base': self.num_voxels_base, 'alpha_init': self.alpha_init, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_world_size': list(self.mask_cache[0].mask.shape), 'fast_color_thres': self.fast_color_thres, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        None
        self.density[0].scale_volume_grid(self.world_size)
        self.density[1].scale_volume_grid(self.world_size)
        self.k0[0].scale_volume_grid(self.world_size)
        self.k0[1].scale_volume_grid(self.world_size)
        if np.prod(list(self.world_size)) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            self_alpha = [F.max_pool3d(self.activate_density(self.density[0].get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0], F.max_pool3d(self.activate_density(self.density[1].get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]]
            self.mask_cache = nn.ModuleList([grid.MaskGrid(path=None, mask=self_alpha[i] > self.fast_color_thres, xyz_min=self.xyz_min, xyz_max=self.xyz_max) for i in range(2)])
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache[0].mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache[0].mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache[0].mask.shape[2])), -1)
        for i in range(2):
            cache_grid_density = self.density[i](cache_grid_xyz)[None, None]
            cache_grid_alpha = self.activate_density(cache_grid_density)
            cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache[i].mask &= cache_grid_alpha > self.fast_color_thres

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density[0].total_variation_add_grad(w, w, w, dense_mode)
        self.density[1].total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0[0].total_variation_add_grad(w, w, w, dense_mode)
        self.k0[1].total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        near = 0
        far = 2 * np.sqrt(3)
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        N_outer = int(np.sqrt(3) / stepdist.item() * (1 - self.bg_preserve)) + 1
        ray_pts_outer = render_utils_cuda.sample_bg_pts_on_rays(rays_o, rays_d, t_max, self.bg_preserve, N_outer)
        return ray_pts, ray_id, step_id, ray_pts_outer

    def _forward(self, ray_pts, viewdirs, interval, N, mask_grid, density_grid, k0_grid, rgbnet=None, ray_id=None, step_id=None, prev_alphainv_last=None):
        if ray_id is None:
            assert len(ray_pts.shape) == 3
            ray_id, step_id = create_full_step_id(ray_pts.shape[:2])
            ray_pts = ray_pts.reshape(-1, 3)
        if prev_alphainv_last is not None:
            mask = prev_alphainv_last > self.fast_color_thres
            ray_id = ray_id.view(N, -1)[mask].view(-1)
            step_id = step_id.view(N, -1)[mask].view(-1)
            ray_pts = ray_pts.view(N, -1, 3)[mask].view(-1, 3)
        mask = mask_grid(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        density = density_grid(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        k0 = k0_grid(ray_pts)
        if rgbnet is None:
            rgb = torch.sigmoid(k0)
        else:
            k0_view = k0
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        return dict(rgb=rgb, alpha=alpha, weights=weights, alphainv_last=alphainv_last, ray_id=ray_id, step_id=step_id)

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        ray_pts, ray_id, step_id, ray_pts_outer = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        fg = self._forward(ray_pts=ray_pts, viewdirs=viewdirs, interval=interval, N=N, mask_grid=self.mask_cache[0], density_grid=self.density[0], k0_grid=self.k0[0], rgbnet=self.rgbnet[0], ray_id=ray_id, step_id=step_id)
        bg = self._forward(ray_pts=ray_pts_outer, viewdirs=viewdirs, interval=interval, N=N, mask_grid=self.mask_cache[1], density_grid=self.density[1], k0_grid=self.k0[1], rgbnet=self.rgbnet[1], prev_alphainv_last=fg['alphainv_last'])
        rgb_marched_fg = segment_coo(src=fg['weights'].unsqueeze(-1) * fg['rgb'], index=fg['ray_id'], out=torch.zeros([N, 3]), reduce='sum')
        rgb_marched_bg = segment_coo(src=bg['weights'].unsqueeze(-1) * bg['rgb'], index=bg['ray_id'], out=torch.zeros([N, 3]), reduce='sum')
        rgb_marched = rgb_marched_fg + fg['alphainv_last'].unsqueeze(-1) * rgb_marched_bg + (fg['alphainv_last'] * bg['alphainv_last']).unsqueeze(-1) * render_kwargs['bg']
        ret_dict.update({'rgb_marched': rgb_marched, 'alphainv_last': torch.cat([fg['alphainv_last'], bg['alphainv_last']]), 'weights': torch.cat([fg['weights'], bg['weights']]), 'raw_alpha': torch.cat([fg['alpha'], bg['alpha']]), 'raw_rgb': torch.cat([fg['rgb'], bg['rgb']]), 'ray_id': torch.cat([fg['ray_id'], bg['ray_id']])})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth_fg = segment_coo(src=fg['weights'] * fg['step_id'], index=fg['ray_id'], out=torch.zeros([N]), reduce='sum')
                depth_bg = segment_coo(src=bg['weights'] * bg['step_id'], index=bg['ray_id'], out=torch.zeros([N]), reduce='sum')
                depth_fg_last = segment_coo(src=fg['step_id'].float(), index=fg['ray_id'], out=torch.zeros([N]), reduce='max')
                depth_bg_last = segment_coo(src=bg['step_id'].float(), index=bg['ray_id'], out=depth_fg_last.clone(), reduce='max')
                depth = depth_fg + fg['alphainv_last'] * (1 + depth_fg_last + depth_bg) + fg['alphainv_last'] * bg['alphainv_last'] * (2 + depth_fg_last + depth_bg_last)
            ret_dict.update({'depth': depth})
        return ret_dict


class DirectContractedVoxGO(nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None, mask_cache_world_size=None, fast_color_thres=0, bg_len=0.2, contracted_norm='inf', density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, **kwargs):
        super(DirectContractedVoxGO, self).__init__()
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectContractedVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1, -1, -1]) - bg_len)
        self.register_buffer('xyz_max', torch.Tensor([1, 1, 1]) + bg_len)
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.bg_len = bg_len
        self.contracted_norm = contracted_norm
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self._set_grid_resolution(num_voxels)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        None
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            None
            None
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        None
        None
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'num_voxels_base': self.num_voxels_base, 'alpha_init': self.alpha_init, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'contracted_norm': self.contracted_norm, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres
        new_p = self.mask_cache.mask.float().mean().item()
        None

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        None
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += ones.grid.grad > 1
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        None
        eps_time = time.time() - eps_time
        None

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        N_inner = int(2 / (2 + 2 * self.bg_len) * self.world_len / stepsize) + 1
        N_outer = N_inner
        b_inner = torch.linspace(0, 2, N_inner + 1)
        b_outer = 2 / torch.linspace(1, 1 / 128, N_outer + 1)
        t = torch.cat([(b_inner[1:] + b_inner[:-1]) * 0.5, (b_outer[1:] + b_outer[:-1]) * 0.5])
        ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        inner_mask = norm <= 1
        ray_pts = torch.where(inner_mask, ray_pts, ray_pts / norm * (1 + self.bg_len - self.bg_len / norm))
        return ray_pts, inner_mask.squeeze(-1), t

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            None
            self.fast_color_thres = self._fast_color_thres[global_step]
        ret_dict = {}
        N = len(rays_o)
        ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])
        mask = inner_mask.clone()
        dist_thres = (2 + 2 * self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:, 1:] - ray_pts[:, :-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[None].repeat(N, 1)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask.flatten()]
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(k0)
        else:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        wsum_mid = segment_coo(src=weights[inner_mask], index=ray_id[inner_mask], out=torch.zeros([N]), reduce='sum')
        s = 1 - 1 / (1 + t)
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'wsum_mid': wsum_mid, 'rgb_marched': rgb_marched, 'raw_density': density, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'step_id': step_id, 'n_max': n_max, 't': t, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([N]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


class DirectMPIGO(torch.nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, mpi_depth=0, mask_cache_path=None, mask_cache_thres=0.001, mask_cache_world_size=None, fast_color_thres=0, density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=0, **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self._set_grid_resolution(num_voxels, mpi_depth)
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.act_shift = grid.DenseGrid(channels=1, world_size=[1, 1, mpi_depth], xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1.0 / mpi_depth - 1e-06)
            p = [1 - g[0]]
            for i in range(1, len(g)):
                p.append((1 - g[:i + 1].sum()) / (1 - g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1 / self.voxel_size_ratio) - 1))
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2 + self.k0_dim
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
        None
        None
        None
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(path=mask_cache_path, mask_cache_thres=mask_cache_thres)
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2])), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256.0 / mpi_depth
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'mpi_depth': self.mpi_depth, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_path': self.mask_cache_path, 'mask_cache_thres': self.mask_cache_thres, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            dens = self.density.get_dense_grid() + self.act_shift.grid
            self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres
        new_p = self.mask_cache.mask.float().mean().item()
        None

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        None
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += ones.grid.grad > 1
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        None
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        None

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        assert near == 0 and far == 1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth - 1) / stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1, 1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1, -1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        vox_emb = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(vox_emb)
        else:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        s = (step_id + 0.5) / N_samples
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'rgb_marched': rgb_marched, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'n_max': N_samples, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([N]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


class DirectVoxGO(torch.nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None, mask_cache_path=None, mask_cache_thres=0.001, mask_cache_world_size=None, fast_color_thres=0, density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        None
        self._set_grid_resolution(num_voxels)
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct, 'rgbnet_full_implicit': rgbnet_full_implicit, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim - 3
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            None
            None
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(path=mask_cache_path, mask_cache_thres=mask_cache_thres)
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2])), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        None
        None
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'num_voxels_base': self.num_voxels_base, 'alpha_init': self.alpha_init, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_path': self.mask_cache_path, 'mask_cache_thres': self.mask_cache_thres, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
        nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1) for co in cam_o.split(100)]).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        None
        far = 1000000000.0
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].flatten(0, -2).split(10000)
            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-06), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += ones.grid.grad > 1
        eps_time = time.time() - eps_time
        None
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Check whether the rays hit the solved coarse geometry or not"""
        far = 1000000000.0
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1000000000.0
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        ray_pts, ray_id, step_id = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(k0)
        else:
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'rgb_marched': rgb_marched, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * step_id, index=ray_id, out=torch.zeros([N]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


class DenseGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        """Add gradients by total variation loss in-place"""
        total_variation_cuda.total_variation_add_grad(self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    xy_feat = F.grid_sample(xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
    feat = torch.cat([xy_feat * z_feat, xz_feat * y_feat, yz_feat * x_feat], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat


def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    xy_feat = F.grid_sample(xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


class TensoRFGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R + R + Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, -1, 3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[..., [0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(self.xy_plane, self.xz_plane, self.yz_plane, self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape, self.channels)
        else:
            out = compute_tensorf_val(self.xy_plane, self.xz_plane, self.yz_plane, self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X, Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X, Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y, Z], mode='bilinear', align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X, 1], mode='bilinear', align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y, 1], mode='bilinear', align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z, 1], mode='bilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        """Add gradients by total variation loss in-place"""
        loss = wx * F.smooth_l1_loss(self.xy_plane[:, :, 1:], self.xy_plane[:, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.xy_plane[:, :, :, 1:], self.xy_plane[:, :, :, :-1], reduction='sum') + wx * F.smooth_l1_loss(self.xz_plane[:, :, 1:], self.xz_plane[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.xz_plane[:, :, :, 1:], self.xz_plane[:, :, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.yz_plane[:, :, 1:], self.yz_plane[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.yz_plane[:, :, :, 1:], self.yz_plane[:, :, :, :-1], reduction='sum') + wx * F.smooth_l1_loss(self.x_vec[:, :, 1:], self.x_vec[:, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.y_vec[:, :, 1:], self.y_vec[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.z_vec[:, :, 1:], self.z_vec[:, :, :-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0, :, :, 0]), torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0, :, :, 0]), torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0, :, :, 0])])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0, :, :, 0]) + torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0, :, :, 0]) + torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0, :, :, 0])
            grid = grid[None, None]
        return grid

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config['n_comp']}"


class MaskGrid(nn.Module):

    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)
        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        """Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'


class Cascade(nn.Module):

    def __init__(self, coarse: nn.Module, fine: nn.Module):
        super(Cascade, self).__init__()
        self.coarse = coarse
        self.fine = fine

    def forward(self, use_coarse: bool, x: torch.Tensor, sigma_only: bool=False, sigma_noise: Optional[torch.Tensor]=None) ->torch.Tensor:
        if use_coarse:
            return self.coarse(x, sigma_only, sigma_noise)
        else:
            return self.fine(x, sigma_only, sigma_noise)


class MegaNeRF(nn.Module):

    def __init__(self, sub_modules: List[nn.Module], centroids: torch.Tensor, boundary_margin: float, xyz_real: bool, cluster_2d: bool, joint_training: bool=False):
        super(MegaNeRF, self).__init__()
        assert boundary_margin >= 1
        self.sub_modules = nn.ModuleList(sub_modules)
        self.register_buffer('centroids', centroids)
        self.boundary_margin = boundary_margin
        self.xyz_real = xyz_real
        self.cluster_dim_start = 1 if cluster_2d else 0
        self.joint_training = joint_training

    def forward(self, x: torch.Tensor, sigma_only: bool=False, sigma_noise: Optional[torch.Tensor]=None) ->torch.Tensor:
        if self.boundary_margin > 1:
            cluster_distances = torch.cdist(x[:, self.cluster_dim_start:3], self.centroids[:, self.cluster_dim_start:])
            inverse_cluster_distances = 1 / (cluster_distances + 1e-08)
            min_cluster_distances = cluster_distances.min(dim=1)[0].unsqueeze(-1).repeat(1, cluster_distances.shape[1])
            inverse_cluster_distances[cluster_distances > self.boundary_margin * min_cluster_distances] = 0
            weights = inverse_cluster_distances / inverse_cluster_distances.sum(dim=-1).unsqueeze(-1)
        else:
            cluster_assignments = torch.cdist(x[:, self.cluster_dim_start:3], self.centroids[:, self.cluster_dim_start:]).argmin(dim=1)
        results = torch.empty(0)
        for i, child in enumerate(self.sub_modules):
            cluster_mask = cluster_assignments == i if self.boundary_margin == 1 else weights[:, i] > 0
            sub_input = x[cluster_mask, 3:] if self.xyz_real else x[cluster_mask]
            if sub_input.shape[0] > 0:
                sub_result = child(sub_input, sigma_only, sigma_noise[cluster_mask] if sigma_noise is not None else None)
                if results.shape[0] == 0:
                    results = torch.zeros(x.shape[0], sub_result.shape[1], device=sub_result.device, dtype=sub_result.dtype)
                if self.boundary_margin == 1:
                    results[cluster_mask] = sub_result
                else:
                    results[cluster_mask] += sub_result * weights[cluster_mask, i].unsqueeze(-1)
            elif self.joint_training:
                sub_result = child(x[:0, 3:] if self.xyz_real else x[:0], sigma_only, sigma_noise[:0] if sigma_noise is not None else None)
                if results.shape[0] == 0:
                    results = torch.zeros(x.shape[0], sub_result.shape[1], device=sub_result.device, dtype=sub_result.dtype)
                results[:0] += 0 * sub_result
        return results


class MegaNeRFContainer(nn.Module):

    def __init__(self, sub_modules: List[nn.Module], bg_sub_modules: List[nn.Module], centroids: torch.Tensor, grid_dim: torch.Tensor, min_position: torch.Tensor, max_position: torch.Tensor, need_viewdir: bool, need_appearance_embedding: bool, cluster_2d: bool):
        super(MegaNeRFContainer, self).__init__()
        for i, sub_module in enumerate(sub_modules):
            setattr(self, 'sub_module_{}'.format(i), sub_module)
        for i, bg_sub_module in enumerate(bg_sub_modules):
            setattr(self, 'bg_sub_module_{}'.format(i), bg_sub_module)
        self.centroids = centroids
        self.grid_dim = grid_dim
        self.min_position = min_position
        self.max_position = max_position
        self.need_viewdir = need_viewdir
        self.need_appearance_embedding = need_appearance_embedding
        self.cluster_2d = cluster_2d


class Embedding(nn.Module):

    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(out, -1)


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int=1, threshold: int=20) ->None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) ->str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class NeRF(nn.Module):

    def __init__(self, pos_xyz_dim: int, pos_dir_dim: int, layers: int, skip_layers: List[int], layer_dim: int, appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int, sigma_activation: nn.Module):
        super(NeRF, self).__init__()
        self.xyz_dim = xyz_dim
        if rgb_dim > 3:
            assert pos_dir_dim == 0
        self.embedding_xyz = Embedding(pos_xyz_dim)
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2
        self.skip_layers = skip_layers
        xyz_encodings = []
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            xyz_encodings.append(layer)
        self.xyz_encodings = nn.ModuleList(xyz_encodings)
        if pos_dir_dim > 0:
            self.embedding_dir = Embedding(pos_dir_dim)
            in_channels_dir = 3 + 3 * pos_dir_dim * 2
        else:
            self.embedding_dir = None
            in_channels_dir = 0
        if appearance_dim > 0:
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None
        if affine_appearance:
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None
        if pos_dir_dim > 0 or appearance_dim > 0 and not affine_appearance:
            self.xyz_encoding_final = nn.Linear(layer_dim, layer_dim)
            self.dir_a_encoding = nn.Sequential(nn.Linear(layer_dim + in_channels_dir + (appearance_dim if not affine_appearance else 0), layer_dim // 2), nn.ReLU(True))
        else:
            self.xyz_encoding_final = None
        self.sigma = nn.Linear(layer_dim, 1)
        self.sigma_activation = sigma_activation
        self.rgb = nn.Linear(layer_dim // 2 if pos_dir_dim > 0 or appearance_dim > 0 and not affine_appearance else layer_dim, rgb_dim)
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()
        else:
            self.rgb_activation = None

    def forward(self, x: torch.Tensor, sigma_only: bool=False, sigma_noise: Optional[torch.Tensor]=None) ->torch.Tensor:
        expected = self.xyz_dim + (0 if sigma_only or self.embedding_dir is None else 3) + (0 if sigma_only or self.embedding_a is None else 1)
        if x.shape[1] != expected:
            raise Exception('Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))
        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)
        sigma = self.sigma(xyz_)
        if sigma_noise is not None:
            sigma += sigma_noise
        sigma = self.sigma_activation(sigma)
        if sigma_only:
            return sigma
        if self.xyz_encoding_final is not None:
            xyz_encoding_final = self.xyz_encoding_final(xyz_)
            dir_a_encoding_input = [xyz_encoding_final]
            if self.embedding_dir is not None:
                dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))
            if self.embedding_a is not None and self.affine is None:
                dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
            dir_a_encoding = self.dir_a_encoding(torch.cat(dir_a_encoding_input, -1))
            rgb = self.rgb(dir_a_encoding)
        else:
            rgb = self.rgb(xyz_)
        if self.affine is not None and self.embedding_a is not None:
            affine_transform = self.affine(self.embedding_a(x[:, -1].long())).view(-1, 3, 4)
            rgb = (affine_transform[:, :, :3] @ rgb.unsqueeze(-1) + affine_transform[:, :, 3:]).squeeze(-1)
        return torch.cat([self.rgb_activation(rgb) if self.rgb_activation is not None else rgb, sigma], -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Embedding,
     lambda: ([], {'num_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShiftedSoftplus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sjtuytc_LargeScaleNeRFPytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

