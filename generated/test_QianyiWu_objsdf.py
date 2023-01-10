import sys
_module = sys.modules[__name__]
del sys
scannet_dataset = _module
scannet_label = _module
scene_dataset = _module
toydesk_dataset = _module
color_mesh = _module
compute_segm_metric = _module
eval = _module
density = _module
embedder = _module
loss = _module
network = _module
ray_sampler = _module
exp_runner = _module
objsdf_train = _module
volsdf_train = _module
general = _module
plots = _module
rend_util = _module

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


from torchvision.transforms import functional as F


import pandas as pd


from re import I


import torch.nn as nn


from torch import nn


import abc


import torchvision


import matplotlib.pyplot as plt


from torch.nn import functional as F


class Density(nn.Module):

    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):

    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


class AbsDensity(Density):

    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):

    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape) * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)


class VolSDFLoss(nn.Module):

    def __init__(self, rgb_loss, eikonal_weight):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).float()
        loss = rgb_loss + self.eikonal_weight * eikonal_loss
        output = {'loss': loss, 'rgb_loss': rgb_loss, 'eikonal_loss': eikonal_loss}
        return output


class ObjSDFLoss(nn.Module):

    def __init__(self, rgb_loss, eikonal_weight, semantic_weight):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.semantic_weight = semantic_weight
        self.semantic_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_semantic_loss(self, semantic_value, semantic_gt):
        semantic_gt = semantic_gt.squeeze()
        semantic_loss = self.semantic_loss(semantic_value, semantic_gt)
        return semantic_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).float()
        if 'semantic_values' in model_outputs:
            semantic_gt = ground_truth['segs'].long()
            semantic_loss = self.get_semantic_loss(model_outputs['semantic_values'], semantic_gt)
        else:
            semantic_loss = torch.tensor(0.0).float()
        loss = rgb_loss + self.eikonal_weight * eikonal_loss + self.semantic_weight * semantic_loss
        output = {'loss': loss, 'rgb_loss': rgb_loss, 'eikonal_loss': eikonal_loss, 'semantic_loss': semantic_loss}
        return output


class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {'include_input': True, 'input_dims': input_dims, 'max_freq_log2': multires - 1, 'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)
    return embed, embedder_obj.out_dim


class ImplicitNetwork(nn.Module):

    def __init__(self, feature_vector_size, sdf_bounding_sphere, d_in, d_out, dims, geometric_init=True, bias=1.0, skip_in=(), weight_norm=True, multires=0, sphere_scale=1.0):
        super().__init__()
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:, :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:, :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingNetwork(nn.Module):

    def __init__(self, feature_vector_size, mode, d_in, d_out, dims, weight_norm=True, multires_view=0):
        super().__init__()
        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x


class RaySampler(metaclass=abc.ABCMeta):

    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):

    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1):
        super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1), self.far * torch.ones(ray_dirs.shape[0], 1)
        else:
            sphere_intersections = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1)
            far = sphere_intersections[:, 1:]
        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        if model.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
        return z_vals


class ErrorBoundSampler(RaySampler):

    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra, eps, beta_iters, max_total_iters, inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=inverse_sphere_bg)
        self.N_samples_extra = N_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny
        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model):
        beta0 = model.density.get_beta().detach()
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = 1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0))) * (dists ** 2.0).sum(-1)
        beta = torch.sqrt(bound)
        total_iters, not_converge = 0, True
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            with torch.no_grad():
                samples_sdf = model.implicit_network.get_sdf_vals(points_flat)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]), samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = 2.0 * torch.sqrt(area_before_sqrt[mask]) / a[mask]
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance
            total_iters += 1
            not_converge = beta.max() > beta0
            if not_converge and total_iters < self.max_total_iters:
                """ Sample more points proportional to the current error bound"""
                N = self.N_samples_eval
                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * dists[:, :-1] ** 2.0 / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1000000.0) - 1.0) * transmittance[:, :-1]
                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            else:
                """ Sample the final sample set to be used in the volume rendering integral """
                N = self.N_samples
                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-05
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            if not_converge and total_iters < self.max_total_iters or not model.training:
                u = torch.linspace(0.0, 1.0, steps=N).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N])
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
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
        z_samples = samples
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1), self.far * torch.ones(ray_dirs.shape[0], 1)
        if self.inverse_sphere_bg:
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]
        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],))
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = z_vals, z_vals_inverse_sphere
        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * dists ** 2.0 / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1000000.0) - 1.0) * torch.exp(-integral_estimation[:, :-1])
        return bound_opacity.max(-1)[0]

    def get_specfic_z_vals(self, ray_dirs, cam_loc, model, idx):
        beta0 = model.density.get_beta().detach()
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = 1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0))) * (dists ** 2.0).sum(-1)
        beta = torch.sqrt(bound)
        total_iters, not_converge = 0, True
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            with torch.no_grad():
                samples_sdf = model.implicit_network.get_specific_sdf_vals(points_flat, idx)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]), samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = 2.0 * torch.sqrt(area_before_sqrt[mask]) / a[mask]
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance
            total_iters += 1
            not_converge = beta.max() > beta0
            if not_converge and total_iters < self.max_total_iters:
                """ Sample more points proportional to the current error bound"""
                N = self.N_samples_eval
                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * dists[:, :-1] ** 2.0 / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1000000.0) - 1.0) * transmittance[:, :-1]
                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            else:
                """ Sample the final sample set to be used in the volume rendering integral """
                N = self.N_samples
                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-05
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            if not_converge and total_iters < self.max_total_iters or not model.training:
                u = torch.linspace(0.0, 1.0, steps=N).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N])
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
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
        z_samples = samples
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1), self.far * torch.ones(ray_dirs.shape[0], 1)
        if self.inverse_sphere_bg:
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]
        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],))
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = z_vals, z_vals_inverse_sphere
        return z_vals, z_samples_eik

    def get_comp_z_vals(self, ray_dirs, cam_loc, model, op_list):
        beta0 = model.density.get_beta().detach()
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = 1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0))) * (dists ** 2.0).sum(-1)
        beta = torch.sqrt(bound)
        total_iters, not_converge = 0, True
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            with torch.no_grad():
                samples_sdf, _ = model.implicit_network.get_compo_sdf_vals(points_flat, op_list)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]), samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = 2.0 * torch.sqrt(area_before_sqrt[mask]) / a[mask]
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance
            total_iters += 1
            not_converge = beta.max() > beta0
            if not_converge and total_iters < self.max_total_iters:
                """ Sample more points proportional to the current error bound"""
                N = self.N_samples_eval
                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * dists[:, :-1] ** 2.0 / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1000000.0) - 1.0) * transmittance[:, :-1]
                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            else:
                """ Sample the final sample set to be used in the volume rendering integral """
                N = self.N_samples
                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-05
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            if not_converge and total_iters < self.max_total_iters or not model.training:
                u = torch.linspace(0.0, 1.0, steps=N).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N])
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
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
        z_samples = samples
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1), self.far * torch.ones(ray_dirs.shape[0], 1)
        if self.inverse_sphere_bg:
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]
        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],))
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = z_vals, z_vals_inverse_sphere
        return z_vals, z_samples_eik

    def get_multi_model_z_vals(self, ray_dirs, cam_loc, model_list, op_list):
        beta0 = model_list[0].density.get_beta().detach()
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model_list[0])
        samples, samples_idx = z_vals, None
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = 1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0))) * (dists ** 2.0).sum(-1)
        beta = torch.sqrt(bound)
        total_iters, not_converge = 0, True
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            sdf_all = []
            with torch.no_grad():
                samples_sdf1 = model_list[0].implicit_network.get_sdf_vals(points_flat)
                samples_sdf2 = model_list[1].implicit_network.get_specific_sdf_vals(points_flat, op_list[1][0][2])
                sdf_all.append(samples_sdf1)
                sdf_all.append(samples_sdf2)
                samples_sdf, _ = torch.min(torch.stack(sdf_all), 0)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]), samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = 2.0 * torch.sqrt(area_before_sqrt[mask]) / a[mask]
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star
            curr_error = self.get_error_bound(beta0, model_list[0], sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model_list[0], sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max
            density = model_list[0].density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance
            total_iters += 1
            not_converge = beta.max() > beta0
            if not_converge and total_iters < self.max_total_iters:
                """ Sample more points proportional to the current error bound"""
                N = self.N_samples_eval
                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * dists[:, :-1] ** 2.0 / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1000000.0) - 1.0) * transmittance[:, :-1]
                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            else:
                """ Sample the final sample set to be used in the volume rendering integral """
                N = self.N_samples
                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-05
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            if not_converge and total_iters < self.max_total_iters or not model_list[0].training:
                u = torch.linspace(0.0, 1.0, steps=N).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N])
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
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
        z_samples = samples
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1), self.far * torch.ones(ray_dirs.shape[0], 1)
        if self.inverse_sphere_bg:
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]
        if self.N_samples_extra > 0:
            if model_list[0].training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],))
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model_list[0])
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = z_vals, z_vals_inverse_sphere
        return z_vals, z_samples_eik


class VolSDFNetwork(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list('bg_color', default=[1.0, 1.0, 1.0])).float()
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def forward(self, input):
        intrinsics = input['intrinsics']
        uv = input['uv']
        pose = input['pose']
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        weights = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        output = {'rgb_values': rgb_values}
        if self.training:
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map
        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance
        return weights


class SemImplicitNetwork(nn.Module):

    def __init__(self, feature_vector_size, sdf_bounding_sphere, d_in, d_out, dims, geometric_init=True, bias=1.0, skip_in=(), weight_norm=True, multires=0, sphere_scale=1.0, sigmoid=20):
        super().__init__()
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        self.sigmoid = sigmoid
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        g = []
        for idx in range(y.shape[1]):
            gradients = torch.autograd.grad(outputs=y[:, idx:idx + 1], inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
            g.append(gradients)
        g = torch.cat(g)
        return g

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:, :self.d_out]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:, :self.d_out]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, output[:, :self.d_out]

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:, :self.d_out]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        return sdf

    def get_specific_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, :self.d_out]
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        sdf = sdf[:, idx:idx + 1]
        return sdf


class SemVolSDFNetwork(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list('bg_color', default=[1.0, 1.0, 1.0])).float()
        self.implicit_network = SemImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        self.num_semantic = conf.get_int('implicit_network.d_out')

    def forward(self, input):
        intrinsics = input['intrinsics']
        uv = input['uv']
        pose = input['pose']
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        output = {'rgb_values': rgb_values, 'semantic_values': semantic_values}
        if self.training:
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map
        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance
        return weights

    def render_via_semantic(self, input, idx):
        intrinsics = input['intrinsics']
        uv = input['uv']
        pose = input['pose']
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik = self.ray_sampler.get_specfic_z_vals(ray_dirs, cam_loc, self, idx)
        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_specific_outputs(points_flat, idx)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights = self.volume_rendering(z_vals, sdf_raw[:, idx:idx + 1])
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        output = {'rgb_values': rgb_values, 'semantic_values': semantic_values}
        occ = torch.ones(rgb.shape)
        occ_value = torch.sum(weights.unsqueeze(-1) * occ, 1)
        output['occ_values'] = occ_value
        if self.training:
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map
        return output


class SemImplicitNetwork_V2(nn.Module):

    def __init__(self, feature_vector_size, sdf_bounding_sphere, d_in, d_out, dims, geometric_init=True, bias=1.0, skip_in=(), weight_norm=True, multires=0, sphere_scale=1.0):
        """
        In this version, we addtionally predict the semantic label just as in Semantic NeRF
        """
        super().__init__()
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + 1 + feature_vector_size]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:, :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        semantic = output[:, 1:self.d_out + 1]
        feature_vectors = output[:, self.d_out + 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:, :self.d_out]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        semantic = torch.softmax(-self.relu(sdf_raw), dim=-1)
        sdf = output[:, idx:idx + 1]
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:, :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

    def get_specific_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class SemVolSDFNetwork_V2(nn.Module):
    """
    This model predict semantic label just like semantic nerf
    """

    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list('bg_color', default=[1.0, 1.0, 1.0])).float()
        self.implicit_network = SemImplicitNetwork_V2(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        self.num_semantic = conf.get_int('implicit_network.d_out')

    def forward(self, input):
        intrinsics = input['intrinsics']
        uv = input['uv']
        pose = input['pose']
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)
        sdf, feature_vectors, gradients, semantic = self.implicit_network.get_outputs(points_flat)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)
            semantic_values = semantic_values + (1 - acc_map[..., None])
        output = {'rgb_values': rgb_values, 'semantic_values': semantic_values}
        if self.training:
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map
        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance
        return weights

    def volume_rendering_with_threshold(self, z_vals, sdf, semantic, threshold):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])
        density = density * (semantic > threshold)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([10000000000.0]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance
        return weights

    def render_via_semantic(self, input, idx):
        intrinsics = input['intrinsics']
        uv = input['uv']
        pose = input['pose']
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik = self.ray_sampler.get_specfic_z_vals(ray_dirs, cam_loc, self, idx)
        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)
        sdf, feature_vectors, gradients, semantic = self.implicit_network.get_outputs(points_flat)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights = self.volume_rendering_with_threshold(z_vals, sdf, semantic[:, :, idx], 20)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        output = {'rgb_values': rgb_values, 'semantic_values': semantic_values}
        occ = torch.ones(rgb.shape)
        occ_value = torch.sum(weights.unsqueeze(-1) * occ, 1)
        output['occ_values'] = occ_value
        if self.training:
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AbsDensity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleDensity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_QianyiWu_objsdf(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

