import sys
_module = sys.modules[__name__]
del sys
activation = _module
gui = _module
network = _module
network_basis = _module
network_hyper = _module
provider = _module
renderer = _module
utils = _module
encoding = _module
ffmlp = _module
backend = _module
ffmlp = _module
setup = _module
freqencoder = _module
backend = _module
freq = _module
setup = _module
gridencoder = _module
backend = _module
grid = _module
setup = _module
loss = _module
main_CCNeRF = _module
main_dnerf = _module
main_nerf = _module
main_sdf = _module
main_tensoRF = _module
clip_utils = _module
gui = _module
network = _module
network_ff = _module
network_tcnn = _module
provider = _module
renderer = _module
utils = _module
raymarching = _module
backend = _module
raymarching = _module
setup = _module
colmap2nerf = _module
hyper2nerf = _module
llff2nerf = _module
tanks2nerf = _module
netowrk = _module
netowrk_ff = _module
network_tcnn = _module
provider = _module
utils = _module
shencoder = _module
backend = _module
setup = _module
sphere_harmonics = _module
network = _module
network_cc = _module
network_cp = _module
utils = _module
test_ffmlp = _module
test_hashencoder = _module
test_hashgrid_grad = _module
test_raymarching = _module
test_shencoder = _module

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


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import math


import numpy as np


from scipy.spatial.transform import Rotation as R


import torch.nn as nn


import torch.nn.functional as F


from scipy.spatial.transform import Slerp


from scipy.spatial.transform import Rotation


from torch.utils.data import DataLoader


from torch.utils.cpp_extension import load


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd.function import once_differentiable


from scipy.spatial.transform import Rotation as Rot


from functools import partial


import random


import torchvision.transforms as T


import torchvision.transforms.functional as TF


import warnings


import pandas as pd


import time


import matplotlib.pyplot as plt


import torch.optim as optim


import torch.distributed as dist


from torch.utils.data import Dataset


from torch.autograd import gradcheck


def custom_meshgrid(*args):
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def sample_pdf(bins, weights, n_samples, det=False):
    weights = weights + 1e-05
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-05, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


class NeRFRenderer(nn.Module):

    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1):
        super().__init__()
        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        self.cuda_ray = cuda_ray
        if cuda_ray:
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3])
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8)
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            step_counter = torch.zeros(16, 2, dtype=torch.int32)
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        aabb = self.aabb_train if self.training else self.aabb_infer
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)
        z_vals = z_vals.expand((N, num_steps))
        z_vals = nears + (fars - nears) * z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])
        density_outputs = self.density(xyzs.reshape(-1, 3))
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1))
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
                z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach()
                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)
            z_vals, z_index = torch.sort(z_vals, dim=1)
            xyzs = torch.cat([xyzs, new_xyzs], dim=1)
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1))
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])
        mask = weights > 0.0001
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3)
        weights_sum = weights.sum(dim=-1)
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
        if self.bg_radius > 0:
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius)
            bg_color = self.background(sph, rays_d.reshape(-1, 3))
        elif bg_color is None:
            bg_color = 1
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        return {'depth': depth, 'image': image, 'weights_sum': weights_sum}

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=0.0001, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if self.bg_radius > 0:
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius)
            bg_color = self.background(sph, rays_d)
        elif bg_color is None:
            bg_color = 1
        results = {}
        if self.training:
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()
            self.local_step += 1
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            sigmas, rgbs = self(xyzs, dirs)
            sigmas = self.density_scale * sigmas
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
                depth = torch.stack(depths, axis=0)
                image = torch.stack(images, axis=0)
            else:
                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
            results['weights_sum'] = weights_sum
        else:
            dtype = torch.float32
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)
            rays_t = nears.clone()
            step = 0
            while step < max_steps:
                n_alive = rays_alive.shape[0]
                if n_alive <= 0:
                    break
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                sigmas, rgbs = self(xyzs, dirs)
                sigmas = self.density_scale * sigmas
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        results['depth'] = depth
        results['image'] = image
        return results

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        if not self.cuda_ray:
            return
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        B = poses.shape[0]
        fx, fy, cx, cy = intrinsic
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        count = torch.zeros_like(self.density_grid)
        poses = poses
        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    indices = raymarching.morton3D(coords).long()
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)
                        head = 0
                        while head < B:
                            tail = min(head + S, B)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]
                            mask_z = cam_xyzs[:, :, 2] > 0
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)
                            count[cas, indices] += mask
                            head += S
        self.density_grid[count == 0] = -1
        None

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        if not self.cuda_ray:
            return
        tmp_grid = -torch.ones_like(self.density_grid)
        if self.iter_density < 16:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            for xs in X:
                for ys in Y:
                    for zs in Z:
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        indices = raymarching.morton3D(coords).long()
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            tmp_grid[cas, indices] = sigmas
        else:
            N = self.grid_size ** 3 // 4
            for cas in range(self.cascade):
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device)
                indices = raymarching.morton3D(coords).long()
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask]
                occ_coords = raymarching.morton3D_invert(occ_indices)
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                cas_xyzs = xyzs * (bound - half_grid_size)
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                tmp_grid[cas, indices] = sigmas
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()
        self.iter_density += 1
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run
        B, N = rays_o.shape[:2]
        device = rays_o.device
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    image[b:b + 1, head:tail] = results_['image']
                    head += max_ray_batch
            results = {}
            results['depth'] = depth
            results['image'] = image
        else:
            results = _run(rays_o, rays_d, **kwargs)
        return results


nvcc_flags = ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']


class _freq_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, degree, output_dim):
        if not inputs.is_cuda:
            inputs = inputs
        inputs = inputs.contiguous()
        B, input_dim = inputs.shape
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)
        _backend.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)
        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims
        grad_inputs = torch.zeros_like(inputs)
        _backend.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)
        return grad_inputs, None, None


freq_encode = _freq_encoder.apply


class FreqEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim + input_dim * 2 * degree

    def __repr__(self):
        return f'FreqEncoder: input_dim={self.input_dim} degree={self.degree} output_dim={self.output_dim}'

    def forward(self, inputs, **kwargs):
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)
        outputs = freq_encode(inputs, self.degree, self.output_dim)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])
        return outputs


def convert_activation(act):
    if act == 'relu':
        return 0
    elif act == 'exponential':
        return 1
    elif act == 'sine':
        return 2
    elif act == 'sigmoid':
        return 3
    elif act == 'squareplus':
        return 4
    elif act == 'softplus':
        return 5
    else:
        return 6


class _ffmlp_forward(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, weights, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference=False, calc_grad_inputs=False):
        B = inputs.shape[0]
        inputs = inputs.contiguous()
        weights = weights.contiguous()
        outputs = torch.empty(B, output_dim, device=inputs.device, dtype=inputs.dtype)
        if not inference:
            forward_buffer = torch.empty(num_layers, B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_forward(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, forward_buffer, outputs)
            ctx.save_for_backward(inputs, weights, outputs, forward_buffer)
            ctx.dims = input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs
        else:
            inference_buffer = torch.empty(B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_inference(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference_buffer, outputs)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        B = grad.shape[0]
        grad = grad.contiguous()
        inputs, weights, outputs, forward_buffer = ctx.saved_tensors
        input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs = ctx.dims
        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=grad.device, dtype=grad.dtype)
        grad_weights = torch.zeros_like(weights)
        backward_buffer = torch.zeros(num_layers, B, hidden_dim, device=grad.device, dtype=grad.dtype)
        _backend.ffmlp_backward(grad, inputs, weights, forward_buffer, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs, backward_buffer, grad_inputs, grad_weights)
        if calc_grad_inputs:
            return grad_inputs, grad_weights, None, None, None, None, None, None, None, None
        else:
            return None, grad_weights, None, None, None, None, None, None, None, None


ffmlp_forward = _ffmlp_forward.apply


class FFMLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = convert_activation(activation)
        self.output_activation = convert_activation('none')
        self.tensorcore_width = 16
        assert hidden_dim in [16, 32, 64, 128, 256], f'FFMLP only support hidden_dim in [16, 32, 64, 128, 256], but got {hidden_dim}'
        assert input_dim > 0 and input_dim % 16 == 0, f'FFMLP input_dim should be 16 * m (m  > 0), but got {input_dim}'
        assert output_dim <= 16, f'FFMLP current only supports output dim <= 16, but got {output_dim}'
        assert num_layers >= 2, f'FFMLP num_layers should be larger than 2 (3 matmuls), but got {num_layers}'
        self.padded_output_dim = int(math.ceil(output_dim / 16)) * 16
        self.num_parameters = hidden_dim * (input_dim + hidden_dim * (num_layers - 1) + self.padded_output_dim)
        self.weights = nn.Parameter(torch.zeros(self.num_parameters))
        self.reset_parameters()
        _backend.allocate_splitk(self.num_layers + 1)

    def cleanup(self):
        _backend.free_splitk()

    def __repr__(self):
        return f'FFMLP: input_dim={self.input_dim} output_dim={self.output_dim} hidden_dim={self.hidden_dim} num_layers={self.num_layers} activation={self.activation}'

    def reset_parameters(self):
        torch.manual_seed(42)
        std = math.sqrt(3 / self.hidden_dim)
        self.weights.data.uniform_(-std, std)

    def forward(self, inputs):
        B, C = inputs.shape
        pad = 128 - B % 128
        if pad > 0:
            inputs = torch.cat([inputs, torch.zeros(pad, C, dtype=inputs.dtype, device=inputs.device)], dim=0)
        outputs = ffmlp_forward(inputs, self.weights, self.input_dim, self.padded_output_dim, self.hidden_dim, self.num_layers, self.activation, self.output_activation, not self.training, inputs.requires_grad)
        if B != outputs.shape[0] or self.padded_output_dim != self.output_dim:
            outputs = outputs[:B, :self.output_dim]
        return outputs


_gridtype_to_id = {'hash': 0, 'tiled': 1}


_interp_to_id = {'linear': 0, 'smoothstep': 1}


class _grid_encode(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        inputs = inputs.contiguous()
        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        S = np.log2(per_level_scale)
        H = base_resolution
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None
        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None
        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)
        if dy_dx is not None:
            grad_inputs = grad_inputs
        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply


def get_encoder(encoding, input_dim=3, multires=6, degree=4, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False, **kwargs):
    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    elif encoding == 'frequency':
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)
    elif encoding == 'sphere_harmonics':
        encoder = SHEncoder(input_dim=input_dim, degree=degree)
    elif encoding == 'hashgrid':
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    elif encoding == 'tiledgrid':
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    elif encoding == 'ash':
        encoder = AshEncoder(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')
    return encoder, encoder.output_dim


class _trunc_exp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


class NeRFNetwork(NeRFRenderer):

    def __init__(self, resolution=[128] * 3, sigma_rank=[96] * 3, color_rank=[288] * 3, color_feat_dim=27, num_layers=3, hidden_dim=128, bound=1, **kwargs):
        super().__init__(bound, **kwargs)
        self.resolution = resolution
        self.sigma_rank = sigma_rank
        self.color_rank = color_rank
        self.color_feat_dim = color_feat_dim
        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]
        self.sigma_vec = self.init_one_svd(self.sigma_rank, self.resolution)
        self.color_vec = self.init_one_svd(self.color_rank, self.resolution)
        self.basis_mat = nn.Linear(self.color_rank[0], self.color_feat_dim, bias=False)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, enc_dim = get_encoder('frequency', input_dim=color_feat_dim, multires=2)
        self.encoder_dir, enc_dim_dir = get_encoder('frequency', input_dim=3, multires=2)
        self.in_dim = enc_dim + enc_dim_dir
        color_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            if l == num_layers - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = nn.ModuleList(color_net)

    def init_one_svd(self, n_component, resolution, scale=0.2):
        vec = []
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            vec.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], resolution[vec_id], 1))))
        return torch.nn.ParameterList(vec)

    def get_sigma_feat(self, x):
        N = x.shape[0]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)
        vec_feat = F.grid_sample(self.sigma_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * F.grid_sample(self.sigma_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * F.grid_sample(self.sigma_vec[2], vec_coord[[2]], align_corners=True).view(-1, N)
        sigma_feat = torch.sum(vec_feat, dim=0)
        return sigma_feat

    def get_color_feat(self, x):
        N = x.shape[0]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)
        vec_feat = F.grid_sample(self.color_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * F.grid_sample(self.color_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * F.grid_sample(self.color_vec[2], vec_coord[[2]], align_corners=True).view(-1, N)
        color_feat = self.basis_mat(vec_feat.T)
        return color_feat

    def forward(self, x, d):
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
        sigma_feat = self.get_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)
        color_feat = self.get_color_feat(x)
        enc_color_feat = self.encoder(color_feat)
        enc_d = self.encoder_dir(d)
        h = torch.cat([enc_color_feat, enc_d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(h)
        return sigma, rgb

    def density(self, x):
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
        sigma_feat = self.get_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)
        return {'sigma': sigma}

    def color(self, x, d, mask=None, **kwargs):
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
        color_feat = self.get_color_feat(x)
        color_feat = self.encoder(color_feat)
        d = self.encoder_dir(d)
        h = torch.cat([color_feat, d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        h = torch.sigmoid(h)
        if mask is not None:
            rgbs[mask] = h
        else:
            rgbs = h
        return rgbs

    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_vec)):
            loss = loss + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss

    @torch.no_grad()
    def upsample_params(self, vec, resolution):
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            vec[i] = torch.nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))

    @torch.no_grad()
    def upsample_model(self, resolution):
        self.upsample_params(self.sigma_vec, resolution)
        self.upsample_params(self.color_vec, resolution)
        self.resolution = resolution

    @torch.no_grad()
    def shrink_model(self):
        half_grid_size = self.bound / self.grid_size
        thresh = min(self.density_thresh, self.mean_density)
        valid_grid = self.density_grid[self.cascade - 1] > thresh
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid))
        valid_pos = (2 * valid_pos / (self.grid_size - 1) - 1) * (self.bound - half_grid_size)
        min_pos = valid_pos.amin(0) - half_grid_size
        max_pos = valid_pos.amax(0) + half_grid_size
        reso = torch.LongTensor(self.resolution)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            self.sigma_vec[i] = nn.Parameter(self.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            self.color_vec[i] = nn.Parameter(self.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0)
        None
        None

    def get_params(self, lr1, lr2):
        return [{'params': self.sigma_vec, 'lr': lr1}, {'params': self.color_vec, 'lr': lr1}, {'params': self.basis_mat.parameters(), 'lr': lr2}, {'params': self.color_net.parameters(), 'lr': lr2}]


class SDFNetwork(nn.Module):

    def __init__(self, encoding='hashgrid', num_layers=3, skips=[], hidden_dim=64, clip_sdf=None):
        super().__init__()
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        assert self.skips == [], 'TCNN does not support concatenating inside, please use skips=[].'
        self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'HashGrid', 'n_levels': 16, 'n_features_per_level': 2, 'log2_hashmap_size': 19, 'base_resolution': 16, 'per_level_scale': 1.3819})
        self.backbone = tcnn.Network(n_input_dims=32, n_output_dims=1, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': hidden_dim, 'n_hidden_layers': num_layers - 1})

    def forward(self, x):
        x = (x + 1) / 2
        x = self.encoder(x)
        h = self.backbone(x)
        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)
        return h


class _sh_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        inputs = inputs.contiguous()
        B, input_dim = inputs.shape
        output_dim = degree ** 2
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)
        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None
        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)
        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, dy_dx = ctx.saved_tensors
        if dy_dx is not None:
            grad = grad.contiguous()
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None


sh_encode = _sh_encoder.apply


class SHEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = degree ** 2
        assert self.input_dim == 3, 'SH encoder only support input dim == 3'
        assert self.degree > 0 and self.degree <= 8, 'SH encoder only supports degree in [1, 8]'

    def __repr__(self):
        return f'SHEncoder: input_dim={self.input_dim} degree={self.degree}'

    def forward(self, inputs, size=1):
        inputs = inputs / size
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)
        outputs = sh_encode(inputs, self.degree, inputs.requires_grad)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])
        return outputs


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for i in range(num_layers - 1):
            self.net.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.net.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(42)
        for p in self.parameters():
            std = math.sqrt(3 / self.hidden_dim)
            p.data.uniform_(-std, std)

    def forward(self, x):
        for i in range(self.num_layers + 1):
            x = self.net[i](x)
            if i != self.num_layers:
                x = self.activation(x)
        return x


class SHEncoder_torch(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5
        self.output_dim = degree ** 2
        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
        self.C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
        self.C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]

    def forward(self, input, **kwargs):
        result = torch.empty((*input.shape[:-1], self.output_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)
        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (3.0 * zz - 1)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'hidden_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ashawkey_torch_ngp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

