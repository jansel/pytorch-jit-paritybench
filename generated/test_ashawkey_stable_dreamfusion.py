import sys
_module = sys.modules[__name__]
del sys
activation = _module
encoding = _module
freqencoder = _module
backend = _module
freq = _module
setup = _module
gradio_app = _module
gridencoder = _module
backend = _module
grid = _module
setup = _module
main = _module
clip = _module
gui = _module
network = _module
network_grid = _module
provider = _module
renderer = _module
sd = _module
utils = _module
optimizer = _module
raymarching = _module
backend = _module
raymarching = _module
setup = _module
shencoder = _module
backend = _module
setup = _module
sphere_harmonics = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.cpp_extension import load


import numpy as np


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torchvision.transforms as T


import torchvision.transforms.functional as TF


import math


from scipy.spatial.transform import Rotation as R


import random


from scipy.spatial.transform import Slerp


from scipy.spatial.transform import Rotation


from torch.utils.data import DataLoader


import time


import warnings


import pandas as pd


import matplotlib.pyplot as plt


import torch.optim as optim


import torch.distributed as dist


from torch.utils.data import Dataset


from typing import List


from torch import Tensor


from torch.optim.optimizer import Optimizer


class FreqEncoder_torch(nn.Module):

    def __init__(self, input_dim, max_freq_log2, N_freqs, log_sampling=True, include_input=True, periodic_fns=(torch.sin, torch.cos)):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)
        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, N_freqs)
        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)
        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)
        return out


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


class CLIP(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/16', device=self.device, jit=False)
        self.aug = T.Compose([T.Resize((224, 224)), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def get_text_embeds(self, prompt, negative_prompt):
        text = clip.tokenize(prompt)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        return text_z

    def train_step(self, text_z, pred_rgb):
        pred_rgb = self.aug(pred_rgb)
        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)
        loss = -(image_z * text_z).sum(-1).mean()
        return loss


class ResBlock(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)
        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.dense(x)
        out = self.norm(out)
        if self.skip is not None:
            identity = self.skip(identity)
        out += identity
        out = self.activation(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dense(x)
        out = self.activation(out)
        return out


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


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

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.cuda_ray = opt.cuda_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh
        self.bg_radius = opt.bg_radius
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        if self.cuda_ray:
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

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, S=128):
        if resolution is None:
            resolution = self.grid_size
        if self.cuda_ray:
            density_thresh = min(self.mean_density, self.density_thresh)
        else:
            density_thresh = self.density_thresh
        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = self.density(pts)
                    sigmas[xi * S:xi * S + len(xs), yi * S:yi * S + len(ys), zi * S:zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        v = torch.from_numpy(vertices)
        f = torch.from_numpy(triangles).int()

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            device = v.device
            v_np = v.cpu().numpy()
            f_np = f.cpu().numpy()
            None
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation
            from scipy.ndimage import binary_erosion
            glctx = dr.RasterizeCudaContext()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]
            vt = torch.from_numpy(vt_np.astype(np.float32)).float()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int()
            uv = vt * 2.0 - 1.0
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            sigmas = torch.zeros(h * w, device=device, dtype=torch.float32)
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            if mask.any():
                xyzs = xyzs[mask]
                all_sigmas = []
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_sigmas.append(results_['sigma'].float())
                    all_feats.append(results_['albedo'].float())
                    head += 640000
                sigmas[mask] = torch.cat(all_sigmas, dim=0)
                feats[mask] = torch.cat(all_feats, dim=0)
            sigmas = sigmas.view(h, w, 1)
            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            mask = mask.cpu().numpy()
            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0
            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)
            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')
            None
            with open(obj_file, 'w') as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')
                None
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                None
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')
                None
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n')
            with open(mtl_file, 'w') as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')
        _export(v, f)

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        results = {}
        aabb = self.aabb_train if self.training else self.aabb_infer
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)
        if light_d is None:
            light_d = rays_o[0] + torch.randn(3, device=device, dtype=torch.float)
            light_d = safe_normalize(light_d)
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
                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))
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
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])
        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, 3)
        if normals is not None:
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.sum(-1).mean()
            normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 0.01).view(N, -1, 3)
            loss_smooth = (normals - normals_perturb).abs()
            results['loss_smooth'] = loss_smooth.mean()
        weights_sum = weights.sum(dim=-1)
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
        if self.bg_radius > 0:
            bg_color = self.background(rays_d.reshape(-1, 3))
        elif bg_color is None:
            bg_color = 1
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        mask = (nears < fars).reshape(*prefix)
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        return results

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=0.0001, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)
        if light_d is None:
            light_d = rays_o[0] + torch.randn(3, device=device, dtype=torch.float)
            light_d = safe_normalize(light_d)
        results = {}
        if self.training:
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()
            self.local_step += 1
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
            if normals is not None:
                weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 0.01)
                loss_smooth = (normals - normals_perturb).abs()
                results['loss_smooth'] = loss_smooth.mean()
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
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
        if self.bg_radius > 0:
            bg_color = self.background(rays_d)
        elif bg_color is None:
            bg_color = 1
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)
        mask = (nears < fars).reshape(*prefix)
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        return results

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        if not self.cuda_ray:
            return
        tmp_grid = -torch.ones_like(self.density_grid)
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
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
                        tmp_grid[cas, indices] = sigmas
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
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
            weights_sum = torch.empty((B, N), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    weights_sum[b:b + 1, head:tail] = results_['weights_sum']
                    image[b:b + 1, head:tail] = results_['image']
                    head += max_ray_batch
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum
        else:
            results = _run(rays_o, rays_d, **kwargs)
        return results


class StableDiffusion(nn.Module):

    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()
        self.device = device
        self.sd_version = sd_version
        None
        if hf_key is not None:
            None
            model_key = hf_key
        elif self.sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
        elif self.sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder='vae')
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder='unet')
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod
        None

    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        latents = self.encode_imgs(pred_rgb_512)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        w = 1 - self.alphas[t]
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        latents.backward(gradient=grad, retain_graph=True)
        return 0

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeds = self.get_text_embeds(prompts, negative_prompts)
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        imgs = self.decode_latents(latents)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        return imgs


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FreqEncoder_torch,
     lambda: ([], {'input_dim': 4, 'max_freq_log2': 4, 'N_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_hidden': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ashawkey_stable_dreamfusion(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

