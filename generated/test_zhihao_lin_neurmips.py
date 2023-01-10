import sys
_module = sys.modules[__name__]
del sys
setup = _module
dataset_convert = _module
experts_forward = _module
experts_test = _module
experts_test_fast = _module
experts_train = _module
mnh = _module
dataset = _module
dataset_replica = _module
dataset_tat = _module
harmonic_embedding = _module
implicit_experts = _module
implicit_function = _module
metric = _module
model_experts = _module
model_teacher = _module
plane_geometry = _module
stats = _module
utils = _module
utils_camera = _module
utils_model = _module
utils_vedo = _module
utils_video = _module
teacher_forward = _module
teacher_test = _module
teacher_train = _module

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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch


import torch.nn.functional as F


import numpy as np


import copy


import time


import math


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from typing import List


import torch.nn as nn


from typing import Tuple


class HarmonicEmbedding(torch.nn.Module):

    def __init__(self, n_harmonic_functions: int=6, omega0: float=1.0, logspace: bool=True):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            ```
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i])
            ]
            ```
        where N corresponds to `n_harmonic_functions`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        either powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced  between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega0`
        before evaluating the harmonic functions.
        """
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (n_harmonic_functions - 1), n_harmonic_functions, dtype=torch.float32)
        self.register_buffer('_frequencies', omega0 * frequencies)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class Experts(nn.Module):
    """
    Single MLP layer with multiple experts
    """

    def __init__(self, n_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.empty(n_experts, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_experts):
            _xavier_init(self.weight[i])
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward_forloop(self, inputs, index):
        outputs = torch.zeros(inputs.shape[0], self.bias.shape[1], device=inputs.device)
        for i in range(self.n_experts):
            idx_i = index == i
            in_i = inputs[idx_i]
            out_i = in_i @ self.weight[i] + self.bias[i]
            outputs[idx_i] = out_i
        return outputs

    def forward(self, inputs, index):
        """
        Args
            inputs: (b, in_feat)
            index: (b, ), max < n_experts
        Return 
            outputs: (b, out_feat)
        """
        weight_sample = self.weight[index]
        bias_sample = self.bias[index]
        prod = torch.einsum('...i,...io->...o', inputs, weight_sample)
        outputs = prod + bias_sample
        return outputs


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(self, n_layers: int, input_dim: int, output_dim: int, skip_dim: int, hidden_dim: int, input_skips: List[int]=()):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = output_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x, z):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y


class NerfExperts(torch.nn.Module):

    def __init__(self, n_harmonic_functions_xyz: int=6, n_harmonic_functions_dir: int=4, n_hidden_neurons_xyz: int=256, n_hidden_neurons_dir: int=128, n_layers_xyz: int=8, n_experts: int=100, append_xyz: List[int]=(5,), **kwargs):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
        """
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3
        self.mlp_xyz = MLPWithInputSkips(n_experts, n_layers_xyz, embedding_dim_xyz, n_hidden_neurons_xyz, embedding_dim_xyz, n_hidden_neurons_xyz, input_skips=append_xyz)
        self.intermediate_linear = Experts(n_experts, n_hidden_neurons_xyz, n_hidden_neurons_xyz)
        self.alpha_layer = Experts(n_experts, n_hidden_neurons_xyz, 1)
        self.alpha_layer.bias.data[:] = 0.0
        self.color_layer = nn.ModuleList([Experts(n_experts, n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir), Experts(n_experts, n_hidden_neurons_dir, 3)])

    def _get_colors(self, features: torch.Tensor, directions: torch.Tensor, index: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        rays_embedding = torch.cat((self.harmonic_embedding_dir(directions), directions), dim=-1)
        intermediate_feat = self.intermediate_linear(features, index)
        color_layer_input = torch.cat([intermediate_feat, rays_embedding], dim=-1)
        color = F.relu(self.color_layer[0](color_layer_input, index))
        color = self.color_layer[1](color, index)
        return color

    def forward(self, points, directions, index, **kwargs):
        embeds_xyz = torch.cat((self.harmonic_embedding_xyz(points), points), dim=-1)
        features = self.mlp_xyz(embeds_xyz, embeds_xyz, index)
        alpha = self.alpha_layer(features, index)
        colors = self._get_colors(features, directions, index)
        output = torch.cat([colors, alpha], dim=-1)
        output = torch.sigmoid(output)
        return output


class NeuralRadianceField(torch.nn.Module):

    def __init__(self, n_harmonic_functions_xyz: int=6, n_harmonic_functions_dir: int=4, n_hidden_neurons_xyz: int=256, n_hidden_neurons_dir: int=128, n_layers_xyz: int=8, append_xyz: List[int]=(5,), **kwargs):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
        """
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3
        self.mlp_xyz = MLPWithInputSkips(n_layers_xyz, embedding_dim_xyz, n_hidden_neurons_xyz, embedding_dim_xyz, n_hidden_neurons_xyz, input_skips=append_xyz)
        self.intermediate_linear = torch.nn.Linear(n_hidden_neurons_xyz, n_hidden_neurons_xyz)
        _xavier_init(self.intermediate_linear)
        self.alpha_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _xavier_init(self.alpha_layer)
        self.alpha_layer.bias.data[:] = 0.0
        self.color_layer = torch.nn.Sequential(torch.nn.Linear(n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir), torch.nn.ReLU(True), torch.nn.Linear(n_hidden_neurons_dir, 3))

    def _get_colors(self, features: torch.Tensor, directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        rays_embedding = torch.cat((self.harmonic_embedding_dir(directions), directions), dim=-1)
        color_layer_input = torch.cat((self.intermediate_linear(features), rays_embedding), dim=-1)
        return self.color_layer(color_layer_input)

    def forward(self, points, directions, **kwargs):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        embeds_xyz = torch.cat((self.harmonic_embedding_xyz(points), points), dim=-1)
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        alpha = self.alpha_layer(features)
        colors = self._get_colors(features, directions)
        output = torch.cat([colors, alpha], dim=-1)
        output = torch.sigmoid(output)
        return output


def farthest_point_sample(points, sample_n: int):
    """
    Input:
        points: (point_n, dim)
    Return:
        idx: (sample_n)
        points_sample: (sample_n, dim)
    """
    idx = 0
    sample_set = [idx]
    dist2set = torch.tensor([])
    for i in range(sample_n - 1):
        dist = points - points[idx]
        dist = torch.sum(dist ** 2, dim=1)[:, None]
        dist2set = torch.cat([dist2set, dist], dim=1)
        min_dist, _ = torch.min(dist2set, dim=1)
        _, max_id = torch.max(min_dist, dim=0)
        idx = max_id.item()
        sample_set.append(idx)
    points_sample = points[sample_set]
    sample_set = torch.LongTensor(sample_set)
    return sample_set, points_sample


def get_points_lrf(points, neighbor_num: int, indices, chunk_size: int=200):
    """
    Input:
        points: (point_n, 3)
        indices: (sample_n,) index of partial points -> reduce computation
    Output:
        Local reference frame at each point computed by PCA
        lrf: (point_n, 3, 3) basis are aranged in columns
    """
    samples = points[indices]
    dist = samples.unsqueeze(1) - points.unsqueeze(0)
    dist = torch.sum(dist ** 2, dim=-1)
    dist_n, neighbor_idx = torch.topk(dist, k=neighbor_num, dim=-1, largest=False)
    neighbors = points[neighbor_idx].cpu()
    lrf_list = []
    sample_n = samples.size(0)
    chunk_n = math.ceil(sample_n / chunk_size)
    for i in range(chunk_n):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, sample_n)
        U, S, V_t = torch.pca_lowrank(neighbors[start:end])
        lrf_list.append(V_t)
    lrf = torch.cat(lrf_list, dim=0)
    return lrf


def orthonormal_basis_from_xy(xy):
    """
    compute orthonormal basis from xy vector: (n, 3, 2)
    """
    x, y = xy[:, :, 0], xy[:, :, 1]
    z = torch.cross(x, y, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    xyz = torch.stack([x, y, z], dim=-1)
    return xyz


class PlaneGeometry(nn.Module):

    def __init__(self, n_plane: int):
        super().__init__()
        self.n_plane = n_plane
        self.center = nn.Parameter(torch.FloatTensor(n_plane, 3))
        self.xy = nn.Parameter(torch.FloatTensor(n_plane, 3, 2))
        self.wh = nn.Parameter(torch.FloatTensor(n_plane, 2))
        self.center.data.uniform_(0, 1)
        self.wh.data[:] = 1
        eyes = torch.eye(3)[None].repeat(n_plane, 1, 1)
        self.xy.data = eyes[:, :, :2]
        self.init_with_box = False

    def initialize(self, points, lrf_neighbors: int=50, wh: float=1.0):
        """
        Initialize planes
            -position: FPS points
            -roation: local PCA basis
            -size: specified in args
        """
        sample_idx, center = farthest_point_sample(points, self.n_plane)
        lrf = get_points_lrf(points, neighbor_num=lrf_neighbors, indices=sample_idx)
        self.center.data = center
        self.xy.data = lrf[:, :, :2]
        self.wh.data[:] = wh

    def initialize_with_box(self, points, lrf_neighbors: int, wh: float, box_factor: float=1.5, random_rate: float=0.0):
        device = points.device
        mean = torch.mean(points, dim=0)
        bound_max = torch.max(points - mean, dim=0)[0] * box_factor + mean
        bound_min = torch.min(points - mean, dim=0)[0] * box_factor + mean
        box_len = torch.max(bound_max - bound_min)
        x_max, y_max, z_max = bound_max
        x_min, y_min, z_min = bound_min
        x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
        face_centers = torch.FloatTensor([[x_min, y_mid, z_mid], [x_max, y_mid, z_mid], [x_mid, y_min, z_mid], [x_mid, y_max, z_mid], [x_mid, y_mid, z_min], [x_mid, y_mid, z_max]])
        eye = torch.eye(3)
        face_xy = torch.stack([eye[:, [1, 2]], eye[:, [1, 2]], eye[:, [0, 2]], eye[:, [0, 2]], eye[:, [0, 1]], eye[:, [0, 1]]], dim=0)
        face_n = 6
        sample_n = self.n_plane - face_n
        if random_rate > 0:
            rand_n = int(sample_n * random_rate)
            fps_n = sample_n - rand_n
            rand_idx = torch.randperm(points.size(0))[:rand_n]
            rand_center = points[rand_idx]
            fps_idx, fps_center = farthest_point_sample(points, fps_n)
            sample_idx = torch.cat([rand_idx, fps_idx])
            center = torch.cat([rand_center, fps_center], dim=0)
        else:
            sample_idx, center = farthest_point_sample(points, sample_n)
        lrf = get_points_lrf(points, neighbor_num=lrf_neighbors, indices=sample_idx)
        self.center.data = torch.cat([face_centers, center], dim=0)
        self.xy.data = torch.cat([face_xy, lrf[:, :, :2]], dim=0)
        self.wh.data[:face_n] = box_len
        self.wh.data[face_n:] = wh
        self.init_with_box = True

    def position(self):
        return self.center

    def basis(self):
        basis = orthonormal_basis_from_xy(self.xy)
        return basis

    def size(self):
        return self.wh

    def get_planes_points(self, resolution: int):
        """
        Get the the position of plane points (image pixel) with resolution in H, W
        Args
            resolution 
        Return
            plane points in world coordinate
            (n_plane, res, res, 3) 
        """
        device = self.center.device
        pix_max = 0.5 * (1 - 0.5 / resolution)
        pix_min = -0.5 * (1 - 0.5 / resolution)
        stride = torch.linspace(pix_max, pix_min, resolution, device=device)
        plane_xy = torch.stack(torch.meshgrid(stride, stride), dim=-1)
        plane_xy = torch.flip(plane_xy, dims=[-1])
        planes_xy = plane_xy.view(1, -1, 2).repeat(self.n_plane, 1, 1)
        planes_xy = planes_xy * self.wh.unsqueeze(1)
        basis = self.basis()
        basis_xy = basis[:, :, :-1]
        from_center = torch.bmm(planes_xy, basis_xy.transpose(1, 2))
        planes_points = self.center.unsqueeze(1) + from_center
        planes_points = planes_points.view(self.n_plane, resolution, resolution, 3)
        return planes_points

    def sample_planes_points(self, points_n):
        """
        Sample random points on planes, total number <= points_n
        Return 
            planes_points: (plane_n*sample_n, 3)
            planes_idx: (plane_n*sample_n, )
        """
        device = self.center.device
        sample_n = math.ceil(points_n / self.n_plane)
        sample_uv = torch.rand(sample_n, 2, device=device) - 0.5
        sample_coord = torch.einsum('pd,sd->psd', self.wh, sample_uv)
        basis = self.basis()
        xy = basis[:, :, :2]
        world_coord = torch.einsum('psa,pba->psb', sample_coord, xy)
        planes_points = self.center.unsqueeze(1) + world_coord
        planes_points = planes_points.detach()
        planes_idx = torch.arange(self.n_plane, device=device)
        planes_idx = planes_idx.unsqueeze(1).repeat(1, sample_n)
        planes_points = planes_points.view(-1, 3)
        planes_idx = planes_idx.view(-1)
        return planes_points, planes_idx

    def planes_vertices(self):
        """
        Return
            planes_vertices: (plane_n, 4, 3)
            which are 4 corners of each planes
        """
        center = self.center
        wh = self.wh
        basis = self.basis()
        xy_basis = basis[:, :, :2]
        xy_vec = xy_basis * wh.unsqueeze(1)
        x_vec, y_vec = xy_vec[:, :, 0], xy_vec[:, :, 1]
        planes_vertices = []
        for i_x in [-0.5, 0.5]:
            for i_y in [-0.5, 0.5]:
                vertices = center + i_x * x_vec + i_y * y_vec
                planes_vertices.append(vertices)
        planes_vertices = torch.stack(planes_vertices, dim=1)
        return planes_vertices.detach()

    def forward(self, points):
        """
        Input:
            points: (point_num, 3) xyz
        Return:
            loss: point-plane "closeness"
        """
        xyz = self.basis()
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        diff = points.unsqueeze(0) - self.center.unsqueeze(1)
        dist_x = torch.bmm(diff, x.unsqueeze(-1)).squeeze()
        dist_y = torch.bmm(diff, y.unsqueeze(-1)).squeeze()
        dist_z = torch.bmm(diff, z.unsqueeze(-1)).squeeze()
        dist_x = torch.abs(dist_x) - self.wh[:, 0].unsqueeze(-1) / 2
        dist_x = torch.clamp(dist_x, min=0)
        dist_y = torch.abs(dist_y) - self.wh[:, 1].unsqueeze(-1) / 2
        dist_y = torch.clamp(dist_y, min=0)
        distance = dist_x ** 2 + dist_y ** 2 + dist_z ** 2
        min_dist, min_id = torch.min(distance, dim=0)
        loss_point2plane = torch.mean(min_dist)
        if self.init_with_box:
            face_n = 6
            loss_area = torch.mean(torch.abs(self.wh[face_n:, 0] * self.wh[face_n:, 1]))
        else:
            loss_area = torch.mean(torch.abs(self.wh[:, 0] * self.wh[:, 1]))
        output = {'loss_point2plane': loss_point2plane, 'loss_area': loss_area}
        return output


def check_inside_planes(planes_points, planes_wh):
    """
    Check if points are inside plane
    Args
        planes_points: (plane_n, point_n, 2)
        planes_wh: (plane_n, 2)
    Return
        inside_planes: (plane_n, point_n)
    """
    norm_scale = (2 / planes_wh).unsqueeze(1)
    planes_points = planes_points * norm_scale
    points_x, points_y = planes_points[:, :, 0], planes_points[:, :, 1]
    bound = 1.0
    in_width = torch.logical_and(points_x >= -bound, points_x <= bound)
    in_height = torch.logical_and(points_y >= -bound, points_y <= bound)
    in_planes = torch.logical_and(in_width, in_height)
    return in_planes


def compute_alpha_weight(alpha_sorted, normalize=False):
    """
    compute alpha weight for composite from raw alpha values (sorted)
    Args
        alpha_sored: (plane_n, sample_n)
        plane[0]: nearest, plane[-1]: farthest
    Return 
        alpha_weight: (plane_n, sample_n)
    """
    plane_n, sample_n = alpha_sorted.size()
    alpha_comp = torch.cumprod(1 - alpha_sorted, dim=0)
    alpha_comp = torch.cat([alpha_comp.new_ones(1, sample_n), alpha_comp[:-1, :]], dim=0)
    alpha_weight = alpha_sorted * alpha_comp
    if normalize == True:
        weight_sum = torch.sum(alpha_weight, dim=0, keepdim=True)
        weight_sum[weight_sum == 0] = 1e-05
        alpha_weight /= weight_sum
    return alpha_weight


def get_ndc_grid(image_size: Tuple[int, int]):
    """
    Get the NDC coordinates of every pixel
    This follows the pytorch3d module NDCGridRaysampler(GridRaysampler)
    here the x is along horizontal direction (width), 
        and y is along vertical (height)
    
    Args
        image_size = (height, width)
    Return
        ndc_girds: in shape (height, width, 3), each position is (x, y, 1)
    """
    height, width = image_size
    half_pix_width = 1.0 / width
    half_pix_height = 1.0 / height
    min_x = 1.0 - half_pix_width
    max_x = -1.0 + half_pix_width
    min_y = 1.0 - half_pix_height
    max_y = -1.0 + half_pix_height
    x_grid_coord = torch.linspace(min_x, max_x, width, dtype=torch.float32)
    y_grid_coord = torch.linspace(min_y, max_y, height, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_grid_coord, x_grid_coord)
    xy_grid = torch.stack([xx, yy], dim=-1)
    ndc_grid = torch.cat([xy_grid, torch.ones(height, width, 1)], dim=-1)
    return ndc_grid


def get_camera_center(camera):
    """
    Return
        center: (1, 3)
    """
    center = camera.get_camera_center()
    return center


def get_normalized_direction(camera, points):
    """
    Args
        points: (..., 3) in world coordinate
    Return
        directions: (..., 3) 
    """
    shape = points.size()
    points = points.view(-1, 3)
    camera_center = get_camera_center(camera)
    directions = points - camera_center
    normalized = F.normalize(directions, dim=-1)
    normalized = normalized.view(*shape)
    return normalized


def grid_sample_planes(sample_points, planes_wh, planes_content, mode='bilinear', padding_mode='zeros'):
    """
    Args:
        sample_points: (plane_n, sample_n, 2)
        planes_wh: (plane_n, 2)
        planes_content: (plane_n, dim, h, w)
    Retrun:
        sampled_content: (plane_n, sample_n, dim)
        in_planes: (plane_n, sample_n) True if sample inside plane
    """
    norm_scale = (-2 / planes_wh).unsqueeze(1)
    grid_points = sample_points * norm_scale
    grid_points = grid_points.unsqueeze(1)
    sampled_content = F.grid_sample(planes_content, grid_points, mode=mode, padding_mode=padding_mode, align_corners=False)
    sampled_content = sampled_content.squeeze(2).transpose(1, 2)
    return sampled_content


def oscillate_ndc_grid(ndc_grid):
    """
    oscillate NDC girds within pixel w/h when trainig -> anti-aliasing
    Args & Return:
        ndc_grid: (h, w, 3)
    """
    h, w, _ = ndc_grid.size()
    device = ndc_grid.device
    half_pix_w = 1.0 / w
    half_pix_h = 1.0 / h
    noise_w = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_w
    noise_h = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_h
    ndc_grid[:, :, 0] += noise_w
    ndc_grid[:, :, 1] += noise_h
    return ndc_grid


def get_camera_k(camera):
    """
    k = [
        [fx,  0, 0],
        [0,  fy, 0],
        [px, py, 1],
    ]
    """
    proj_trans = camera.get_projection_transform()
    proj_mat = proj_trans.get_matrix().squeeze()
    k = proj_mat[:3, :3]
    k[-1, -1] = 1
    return k


def camera_ray_directions(camera, ndc_points: torch.Tensor):
    """
    Calculate (x/z, y/z, 1) of each NDC points, under camera coord.
    Args
        ndc_points: (point_n, 3)
    Return
        xy1_points: (point_n, 3)
    """
    device = ndc_points.device
    k = get_camera_k(camera)
    fx, fy = k[0, 0], k[1, 1]
    px, py = k[2, 0], k[2, 1]
    shift = torch.tensor([[-px, -py]])
    scale = torch.tensor([[fx, fy]])
    xy1_points = torch.ones_like(ndc_points)
    xy1_points[:, :2] = (ndc_points[:, :2] + shift) / scale
    return xy1_points


def filter_tiny_values(tensor, eps: float=1e-05):
    tensor_sign = torch.sign(tensor)
    tensor_sign[tensor_sign == 0] = 1
    tensor_abs = torch.clamp(torch.abs(tensor), min=eps)
    tensor_out = tensor_sign * tensor_abs
    return tensor_out


def ray_plane_intersection(planes_frame: torch.Tensor, planes_center: torch.Tensor, camera, ndc_points: torch.Tensor, eps: float=1e-05):
    """
    Calculate ray-plane intersection in world coord. (follow the paper formulation)
    Return
        depth: (plane_n, point_n)
        intersections: (plane_n, point_n, 3)
    """
    normal_planes = planes_frame[:, :, -1]
    center_planes = planes_center
    R_cam = camera.R
    center_cam = get_camera_center(camera)
    xy1_points = camera_ray_directions(camera, ndc_points)
    cam2world = R_cam[0].T
    d = xy1_points @ cam2world
    num = torch.sum((center_planes - center_cam) * normal_planes, dim=-1)
    den = torch.mm(normal_planes, d.T)
    den = filter_tiny_values(den, eps)
    t = num.unsqueeze(-1) / den
    td = t.unsqueeze(-1) * d.unsqueeze(0)
    o = center_cam.unsqueeze(0)
    intersections = o + td
    depth = t
    return depth, intersections


class ModelExperts(nn.Module):

    def __init__(self, n_plane: int, image_size: Tuple[int], n_harmonic_functions_pos: int, n_harmonic_functions_dir: int, n_hidden_neurons_pos: int, n_hidden_neurons_dir: int, n_layers: int, n_train_sample: int, n_infer_sample: int, anti_aliasing: bool, premultiply_alpha: bool, n_bake_sample: int, bake_res: int, filter_thresh: float, white_bg: bool):
        super().__init__()
        self.n_plane = n_plane
        self.plane_geo = PlaneGeometry(n_plane)
        self.image_size = image_size
        self.ndc_grid = get_ndc_grid(image_size)
        self.plane_radiance_field = NerfExperts(n_harmonic_functions_pos, n_harmonic_functions_dir, n_hidden_neurons_pos, n_hidden_neurons_dir, n_layers, n_experts=n_plane)
        self.n_train_sample = n_train_sample
        self.n_infer_sample = n_infer_sample
        self.anti_aliasing = anti_aliasing
        self.premultiply_alpha = premultiply_alpha
        self.planes_alpha = None
        self.n_bake_sample = n_bake_sample
        self.bake_res = bake_res
        self.filter_thresh = filter_thresh
        self.white_bg = white_bg

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def bake_planes_alpha(self):
        resolution = self.bake_res
        planes_points = self.plane_geo.get_planes_points(resolution)
        planes_points = planes_points.view(-1, 3)
        planes_idx = torch.arange(self.n_plane, device=planes_points.device)
        planes_idx = planes_idx.view(-1, 1, 1).repeat(1, resolution, resolution).view(-1)
        points_total_n = resolution ** 2 * self.n_plane
        sample_n = self.n_bake_sample
        chunk_n = math.ceil(points_total_n / sample_n)
        planes_alpha = []
        with torch.no_grad():
            for i in range(chunk_n):
                start = i * sample_n
                end = min((i + 1) * sample_n, points_total_n)
                points = planes_points[start:end, :]
                dirs = torch.zeros_like(points)
                idx = planes_idx[start:end]
                rgba = self.plane_radiance_field(points, dirs, idx)
                rgba = rgba.detach()
                alpha = rgba[..., -1]
                planes_alpha.append(alpha)
        planes_alpha = torch.cat(planes_alpha, dim=0)
        planes_alpha = planes_alpha.view(self.n_plane, 1, resolution, resolution)
        self.planes_alpha = planes_alpha
        torch.cuda.empty_cache()
        None

    def ray_plane_intersect(self, camera, ndc_points):
        """
        Return
            world_points: (plane_n, point_n, 3)
            planes_depth:  (plane_n, point_n)
            hit:          (plane_n, point_n)
        """
        planes_basis = self.plane_geo.basis()
        planes_center = self.plane_geo.position()
        planes_depth, world_points = ray_plane_intersection(planes_basis, planes_center, camera, ndc_points)
        xy_basis = planes_basis[:, :, :2]
        planes_points = torch.bmm(world_points - planes_center.unsqueeze(1), xy_basis)
        in_planes = check_inside_planes(planes_points, self.plane_geo.size())
        hit = torch.logical_and(in_planes, planes_depth > 0)
        return world_points, planes_points, planes_depth, hit

    def sort_depth_index(self, planes_depth):
        """
        sort points along ray with depth to planes 
        Args
            planes_depth: (plane_n, point_n)
        Return
            sort_id_0, sort_id_0
        """
        depth_sorted, sort_id_0 = torch.sort(planes_depth, dim=0, descending=False)
        point_n = planes_depth.size(1)
        sort_id_1 = torch.arange(point_n)[None]
        sort_idx = [sort_id_0, sort_id_1]
        return depth_sorted, sort_idx

    def get_planes_indices(self, hit):
        """
        Return
            planes_idx_full: (plane_n, point_n) -> accending in 1-dim
        """
        plane_n, point_n = hit.shape
        planes_idx_full = torch.arange(plane_n, device=hit.device)
        planes_idx_full = planes_idx_full.unsqueeze(1).repeat(1, point_n)
        return planes_idx_full

    def predict_points_rgba_experts(self, camera, points, planes_idx):
        """
        Args
            camera
            points: (hit_n, 3): in world coord.
            planes_idx: (hit_n) 
        Return
            poins_rgba: (hit_n, 4)
        """
        view_dirs = get_normalized_direction(camera, points)
        points_rgba = self.plane_radiance_field(points, view_dirs, planes_idx)
        return points_rgba

    def alpha_composite(self, rgb, alpha, depth):
        """
        Return
            color: (point_n, 3)
            depth: (point_n)
        """
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        depth = torch.sum(depth * alpha_weight, dim=0)
        color = torch.sum(rgb * alpha_weight.unsqueeze(-1), dim=0)
        if self.white_bg:
            alpha_sum = torch.sum(alpha_weight, dim=0).unsqueeze(-1)
            white = torch.ones_like(color)
            color = color + (1 - alpha_sum) * white
        return color, depth

    def no_hit_output(self, ndc_points):
        if self.white_bg:
            color_bg = torch.ones_like(ndc_points)
        else:
            color_bg = torch.zeros_like(ndc_points)
        point_n = ndc_points.size(0)
        device = ndc_points.device
        dummy_output = {'color': color_bg, 'depth': torch.zeros(point_n, device=device)}
        return dummy_output

    def sample_baked_alpha(self, planes_points, hit):
        """
        Return 
            alpha_sample: (plane_n, point_n)
        """
        alpha_sample = grid_sample_planes(sample_points=planes_points, planes_wh=self.plane_geo.size(), planes_content=self.planes_alpha, mode='nearest', padding_mode='border')
        alpha_sample = alpha_sample.squeeze(-1)
        alpha_sample[hit == False] = 0
        return alpha_sample

    def process_ndc_points(self, camera, ndc_points):
        """
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        """
        world_points, _, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        planes_idx_full = self.get_planes_indices(hit)
        planes_idx = planes_idx_full[hit]
        points = world_points[hit]
        points_rgba = self.predict_points_rgba_experts(camera, points, planes_idx)
        rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        rgba[hit] = points_rgba
        depth, sort_idx = self.sort_depth_index(planes_depth)
        rgba = rgba[sort_idx]
        rgb, alpha = rgba[:, :, :-1], rgba[:, :, -1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
        output = {'color': color, 'depth': depth}
        return output

    def process_ndc_points_with_alpha(self, camera, ndc_points):
        """
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        """
        world_points, planes_points, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        alpha_baked = self.sample_baked_alpha(planes_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        alpha = alpha_baked[sort_idx]
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        contrib = alpha_weight > self.filter_thresh
        hit = hit[sort_idx]
        hit = torch.logical_and(hit, contrib)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        world_points = world_points[sort_idx]
        points = world_points[hit]
        planes_idx_full = self.get_planes_indices(hit)
        planes_idx_sorted = planes_idx_full[sort_idx]
        planes_idx = planes_idx_sorted[hit]
        points_rgba = self.predict_points_rgba_experts(camera, points, planes_idx)
        rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        rgba[hit] = points_rgba
        rgb, alpha = rgba[:, :, :-1], rgba[:, :, -1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
        output = {'color': color, 'depth': depth}
        return output

    def process(self, camera, ndc_points):
        out = None
        if self.planes_alpha is not None:
            out = self.process_ndc_points_with_alpha(camera, ndc_points)
        else:
            out = self.process_ndc_points(camera, ndc_points)
        return out

    def ndc_points_full(self, camera):
        """
        Return:
            NDC points: (img_h*img_w, 3)
        """
        device = camera.device
        self.ndc_grid = self.ndc_grid
        ndc_grid = self.ndc_grid.clone()
        if self.training and self.anti_aliasing:
            ndc_grid = oscillate_ndc_grid(ndc_grid)
        ndc_points = ndc_grid.view(-1, 3)
        return ndc_points

    def forward_train(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        sample_idx = torch.rand(self.n_train_sample)
        sample_idx = (sample_idx * img_pixel_num).long()
        ndc_points = ndc_points_full[sample_idx]
        output = self.process(camera, ndc_points)
        output['sample_idx'] = sample_idx
        return output

    def forward_full_image(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        if self.n_infer_sample > 0:
            sample_num = self.n_infer_sample
        else:
            sample_num = img_pixel_num
        chunk_num = math.ceil(img_pixel_num / sample_num)
        chunk_outputs = []
        for i in range(chunk_num):
            start = i * sample_num
            end = min((i + 1) * sample_num, img_pixel_num)
            ndc_points = ndc_points_full[start:end]
            chunk_out = self.process(camera, ndc_points)
            chunk_outputs.append(chunk_out)
        img_wh = self.image_size
        shapes = {'color': [*img_wh, 3], 'depth': [*img_wh]}
        output = {key: torch.cat([chunk_out[key] for chunk_out in chunk_outputs], dim=0).view(*shape) for key, shape in shapes.items()}
        return output

    def forward(self, camera):
        ndc_points_full = self.ndc_points_full(camera)
        if self.training:
            output = self.forward_train(camera, ndc_points_full)
        else:
            output = self.forward_full_image(camera, ndc_points_full)
        return output


class ModelTeacher(nn.Module):

    def __init__(self, n_plane: int, image_size: Tuple[int], n_harmonic_functions_pos: int, n_harmonic_functions_dir: int, n_hidden_neurons_pos: int, n_hidden_neurons_dir: int, n_layers: int, n_train_sample: int, n_infer_sample: int, anti_aliasing: bool, premultiply_alpha: bool, n_bake_sample: int, bake_res: int, filter_thresh: float, white_bg: bool):
        super().__init__()
        self.n_plane = n_plane
        self.plane_geo = PlaneGeometry(n_plane)
        self.image_size = image_size
        self.ndc_grid = get_ndc_grid(image_size)
        self.radiance_field = NeuralRadianceField(n_harmonic_functions_pos, n_harmonic_functions_dir, n_hidden_neurons_pos, n_hidden_neurons_dir, n_layers)
        self.n_train_sample = n_train_sample
        self.n_infer_sample = n_infer_sample
        self.anti_aliasing = anti_aliasing
        self.premultiply_alpha = premultiply_alpha
        self.planes_alpha = None
        self.n_bake_sample = n_bake_sample
        self.bake_res = bake_res
        self.filter_thresh = filter_thresh
        self.white_bg = white_bg

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def bake_planes_alpha(self):
        resolution = self.bake_res
        planes_points = self.plane_geo.get_planes_points(resolution)
        planes_points = planes_points.view(-1, 3)
        points_total_n = resolution ** 2 * self.n_plane
        sample_n = self.n_bake_sample
        chunk_n = math.ceil(points_total_n / sample_n)
        planes_alpha = []
        with torch.no_grad():
            for i in range(chunk_n):
                start = i * sample_n
                end = min((i + 1) * sample_n, points_total_n)
                points = planes_points[start:end, :]
                dirs = torch.zeros_like(points)
                rgba = self.radiance_field(points, dirs)
                rgba = rgba.detach()
                alpha = rgba[..., -1]
                planes_alpha.append(alpha)
        planes_alpha = torch.cat(planes_alpha, dim=0)
        planes_alpha = planes_alpha.view(self.n_plane, 1, resolution, resolution)
        self.planes_alpha = planes_alpha
        torch.cuda.empty_cache()
        None

    def ray_plane_intersect(self, camera, ndc_points):
        """
        Return
            world_points: (plane_n, point_n, 3)
            planes_points: (plane_n, point_n, 2)
            planes_depth:  (plane_n, point_n)
            hit:          (plane_n, point_n)
        """
        planes_basis = self.plane_geo.basis()
        planes_center = self.plane_geo.position()
        planes_depth, world_points = ray_plane_intersection(planes_basis, planes_center, camera, ndc_points)
        xy_basis = planes_basis[:, :, :2]
        planes_points = torch.bmm(world_points - planes_center.unsqueeze(1), xy_basis)
        in_planes = check_inside_planes(planes_points, self.plane_geo.size())
        hit = torch.logical_and(in_planes, planes_depth > 0)
        return world_points, planes_points, planes_depth, hit

    def sort_depth_index(self, planes_depth):
        """
        sort points along ray with depth to planes 
        Args
            planes_depth: (plane_n, point_n)
        Return
            sort_id_0, sort_id_0
        """
        depth_sorted, sort_id_0 = torch.sort(planes_depth, dim=0, descending=False)
        point_n = planes_depth.size(1)
        sort_id_1 = torch.arange(point_n)[None]
        sort_idx = [sort_id_0, sort_id_1]
        return depth_sorted, sort_idx

    def predict_points_rgba(self, camera, world_points, hit):
        """
        Return
            points_rgba: (plane_n, point_n, 4)
        """
        points = world_points[hit]
        view_dirs = get_normalized_direction(camera, points)
        rgba = self.radiance_field(points, view_dirs)
        points_rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        points_rgba[hit] = rgba
        return points_rgba

    def alpha_composite(self, rgb, alpha, depth):
        """
        Return
            color: (point_n, 3)
            depth: (point_n)
        """
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        depth = torch.sum(depth * alpha_weight, dim=0)
        color = torch.sum(rgb * alpha_weight.unsqueeze(-1), dim=0)
        if self.white_bg:
            alpha_sum = torch.sum(alpha_weight, dim=0).unsqueeze(-1)
            white = torch.ones_like(color)
            color = color + (1 - alpha_sum) * white
        return color, depth

    def no_hit_output(self, ndc_points):
        if self.white_bg:
            color_bg = torch.ones_like(ndc_points)
        else:
            color_bg = torch.zeros_like(ndc_points)
        point_n = ndc_points.size(0)
        device = ndc_points.device
        dummy_output = {'color': color_bg, 'depth': torch.zeros(point_n, device=device)}
        return dummy_output

    def sample_baked_alpha(self, planes_points, hit):
        """
        Return 
            alpha_sample: (plane_n, point_n)
        """
        alpha_sample = grid_sample_planes(sample_points=planes_points, planes_wh=self.plane_geo.size(), planes_content=self.planes_alpha, mode='nearest', padding_mode='border')
        alpha_sample = alpha_sample.squeeze(-1)
        alpha_sample[hit == False] = 0
        return alpha_sample

    def process_ndc_points(self, camera, ndc_points):
        """
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        """
        world_points, _, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        rgba = self.predict_points_rgba(camera, world_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        rgba = rgba[sort_idx]
        rgb, alpha = rgba[:, :, :-1], rgba[:, :, -1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
        output = {'color': color, 'depth': depth}
        return output

    def process_ndc_points_with_alpha(self, camera, ndc_points):
        """
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        """
        world_points, planes_points, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        alpha_baked = self.sample_baked_alpha(planes_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        alpha = alpha_baked[sort_idx]
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        contrib = alpha_weight > self.filter_thresh
        alpha[contrib == False] = 0
        world_points = world_points[sort_idx]
        hit = hit[sort_idx]
        hit = torch.logical_and(hit, contrib)
        rgba = self.predict_points_rgba(camera, world_points, hit)
        rgb, alpha = rgba[:, :, :-1], rgba[:, :, -1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
        output = {'color': color, 'depth': depth}
        return output

    def process(self, camera, ndc_points):
        out = None
        if self.planes_alpha is not None:
            out = self.process_ndc_points_with_alpha(camera, ndc_points)
        else:
            out = self.process_ndc_points(camera, ndc_points)
        return out

    def ndc_points_full(self, camera):
        """
        Return:
            NDC points: (img_h*img_w, 3)
        """
        device = camera.device
        self.ndc_grid = self.ndc_grid
        ndc_grid = self.ndc_grid.clone()
        if self.training and self.anti_aliasing:
            ndc_grid = oscillate_ndc_grid(ndc_grid)
        ndc_points = ndc_grid.view(-1, 3)
        return ndc_points

    def forward_train(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        sample_idx = torch.rand(self.n_train_sample)
        sample_idx = (sample_idx * img_pixel_num).long()
        ndc_points = ndc_points_full[sample_idx]
        output = self.process(camera, ndc_points)
        output['sample_idx'] = sample_idx
        return output

    def forward_full_image(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        if self.n_infer_sample > 0:
            sample_num = self.n_infer_sample
        else:
            sample_num = img_pixel_num
        chunk_num = math.ceil(img_pixel_num / sample_num)
        chunk_outputs = []
        for i in range(chunk_num):
            start = i * sample_num
            end = min((i + 1) * sample_num, img_pixel_num)
            ndc_points = ndc_points_full[start:end]
            chunk_out = self.process(camera, ndc_points)
            chunk_outputs.append(chunk_out)
        img_wh = self.image_size
        shapes = {'color': [*img_wh, 3], 'depth': [*img_wh]}
        output = {key: torch.cat([chunk_out[key] for chunk_out in chunk_outputs], dim=0).view(*shape) for key, shape in shapes.items()}
        return output

    def forward(self, camera):
        ndc_points_full = self.ndc_points_full(camera)
        if self.training:
            output = self.forward_train(camera, ndc_points_full)
        else:
            output = self.forward_full_image(camera, ndc_points_full)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (HarmonicEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPWithInputSkips,
     lambda: ([], {'n_layers': 1, 'input_dim': 4, 'output_dim': 4, 'skip_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_zhihao_lin_neurmips(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

