import sys
_module = sys.modules[__name__]
del sys
example1 = _module
example2 = _module
example3 = _module
example4 = _module
neural_renderer = _module
cuda = _module
get_points_from_angles = _module
lighting = _module
load_obj = _module
look = _module
look_at = _module
mesh = _module
perspective = _module
projection = _module
rasterize = _module
renderer = _module
save_obj = _module
vertices_to_faces = _module
setup = _module
test_get_points_from_angles = _module
test_lighting = _module
test_load_obj = _module
test_look = _module
test_look_at = _module
test_perspective = _module
test_rasterize = _module
test_rasterize_depth = _module
test_rasterize_silhouettes = _module
test_renderer = _module
test_save_obj = _module
test_vertices_to_faces = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.autograd import Function


import math


import numpy


class Model(nn.Module):

    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = nn.Parameter(vertices[(None), :, :])
        self.register_buffer('faces', faces[(None), :, :])
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)
        image_ref = torch.from_numpy(imread(filename_ref).astype(np.float32).mean(-1) / 255.0)[(None), :]
        self.register_buffer('image_ref', image_ref)
        renderer = nr.Renderer(camera_mode='look_at')
        self.renderer = renderer

    def forward(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, 90)
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[(None), :, :]) ** 2)
        return loss


class Model(nn.Module):

    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[(None), :, :])
        self.register_buffer('faces', faces[(None), :, :])
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)
        image_ref = torch.from_numpy(imread(filename_ref).astype('float32') / 255.0).permute(2, 0, 1)[(None), :]
        self.register_buffer('image_ref', image_ref)
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

    def forward(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss


class Model(nn.Module):

    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[(None), :, :])
        self.register_buffer('faces', faces[(None), :, :])
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[(None), :, :]) ** 2)
        return loss


class RasterizeFunction(Function):
    """
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    """

    @staticmethod
    def forward(ctx, faces, textures, image_size, near, far, eps, background_color, return_rgb=False, return_alpha=False, return_depth=False):
        """
        Forward pass
        """
        ctx.image_size = image_size
        ctx.near = near
        ctx.far = far
        ctx.eps = eps
        ctx.background_color = background_color
        ctx.return_rgb = return_rgb
        ctx.return_alpha = return_alpha
        ctx.return_depth = return_depth
        faces = faces.clone()
        ctx.device = faces.device
        ctx.batch_size, ctx.num_faces = faces.shape[:2]
        if ctx.return_rgb:
            textures = textures.contiguous()
            ctx.texture_size = textures.shape[2]
        else:
            textures = torch.cuda.FloatTensor(1).fill_(0)
            ctx.texture_size = None
        face_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(-1)
        weight_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3).fill_(0.0)
        depth_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(ctx.far)
        if ctx.return_rgb:
            rgb_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3).fill_(0)
            sampling_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 8).fill_(0)
            sampling_weight_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 8).fill_(0)
        else:
            rgb_map = torch.cuda.FloatTensor(1).fill_(0)
            sampling_index_map = torch.cuda.FloatTensor(1).fill_(0)
            sampling_weight_map = torch.cuda.FloatTensor(1).fill_(0)
        if ctx.return_alpha:
            alpha_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(0)
        else:
            alpha_map = torch.cuda.FloatTensor(1).fill_(0)
        if ctx.return_depth:
            face_inv_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size, 3, 3).fill_(0)
        else:
            face_inv_map = torch.cuda.FloatTensor(1).fill_(0)
        face_index_map, weight_map, depth_map, face_inv_map = RasterizeFunction.forward_face_index_map(ctx, faces, face_index_map, weight_map, depth_map, face_inv_map)
        rgb_map, sampling_index_map, sampling_weight_map = RasterizeFunction.forward_texture_sampling(ctx, faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map)
        rgb_map = RasterizeFunction.forward_background(ctx, face_index_map, rgb_map)
        alpha_map = RasterizeFunction.forward_alpha_map(ctx, alpha_map, face_index_map)
        ctx.save_for_backward(faces, textures, face_index_map, weight_map, depth_map, rgb_map, alpha_map, face_inv_map, sampling_index_map, sampling_weight_map)
        rgb_r, alpha_r, depth_r = torch.tensor([]), torch.tensor([]), torch.tensor([])
        if ctx.return_rgb:
            rgb_r = rgb_map
        if ctx.return_alpha:
            alpha_r = alpha_map.clone()
        if ctx.return_depth:
            depth_r = depth_map.clone()
        return rgb_r, alpha_r, depth_r

    @staticmethod
    def backward(ctx, grad_rgb_map, grad_alpha_map, grad_depth_map):
        """
        Backward pass
        """
        faces, textures, face_index_map, weight_map, depth_map, rgb_map, alpha_map, face_inv_map, sampling_index_map, sampling_weight_map = ctx.saved_tensors
        grad_faces = torch.zeros_like(faces, dtype=torch.float32)
        if ctx.return_rgb:
            grad_textures = torch.zeros_like(textures, dtype=torch.float32)
        else:
            grad_textures = torch.cuda.FloatTensor(1).fill_(0.0)
        if ctx.return_rgb:
            if grad_rgb_map is not None:
                grad_rgb_map = grad_rgb_map.contiguous()
            else:
                grad_rgb_map = torch.zeros_like(rgb_map)
        else:
            grad_rgb_map = torch.cuda.FloatTensor(1).fill_(0.0)
        if ctx.return_alpha:
            if grad_alpha_map is not None:
                grad_alpha_map = grad_alpha_map.contiguous()
            else:
                grad_alpha_map = torch.zeros_like(alpha_map)
        else:
            grad_alpha_map = torch.cuda.FloatTensor(1).fill_(0.0)
        if ctx.return_depth:
            if grad_depth_map is not None:
                grad_depth_map = grad_depth_map.contiguous()
            else:
                grad_depth_map = torch.zeros_like(ctx.depth_map)
        else:
            grad_depth_map = torch.cuda.FloatTensor(1).fill_(0.0)
        grad_faces = RasterizeFunction.backward_pixel_map(ctx, faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces)
        grad_textures = RasterizeFunction.backward_textures(ctx, face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures)
        grad_faces = RasterizeFunction.backward_depth_map(ctx, faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces)
        if not textures.requires_grad:
            grad_textures = None
        return grad_faces, grad_textures, None, None, None, None, None, None, None, None

    @staticmethod
    def forward_face_index_map(ctx, faces, face_index_map, weight_map, depth_map, face_inv_map):
        faces_inv = torch.zeros_like(faces)
        return rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map, depth_map, face_inv_map, faces_inv, ctx.image_size, ctx.near, ctx.far, ctx.return_rgb, ctx.return_alpha, ctx.return_depth)

    @staticmethod
    def forward_texture_sampling(ctx, faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map):
        if not ctx.return_rgb:
            return rgb_map, sampling_index_map, sampling_weight_map
        else:
            return rasterize_cuda.forward_texture_sampling(faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map, ctx.image_size, ctx.eps)

    @staticmethod
    def forward_alpha_map(ctx, alpha_map, face_index_map):
        if ctx.return_alpha:
            alpha_map[face_index_map >= 0] = 1
        return alpha_map

    @staticmethod
    def forward_background(ctx, face_index_map, rgb_map):
        if ctx.return_rgb:
            background_color = torch.cuda.FloatTensor(ctx.background_color)
            mask = (face_index_map >= 0).float()[:, :, :, (None)]
            if background_color.ndimension() == 1:
                rgb_map = rgb_map * mask + (1 - mask) * background_color[(None), (None), (None), :]
            elif background_color.ndimension() == 2:
                rgb_map = rgb_map * mask + (1 - mask) * background_color[:, (None), (None), :]
        return rgb_map

    @staticmethod
    def backward_pixel_map(ctx, faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces):
        if not ctx.return_rgb and not ctx.return_alpha:
            return grad_faces
        else:
            return rasterize_cuda.backward_pixel_map(faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces, ctx.image_size, ctx.eps, ctx.return_rgb, ctx.return_alpha)

    @staticmethod
    def backward_textures(ctx, face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures):
        if not ctx.return_rgb:
            return grad_textures
        else:
            return rasterize_cuda.backward_textures(face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures, ctx.num_faces)

    @staticmethod
    def backward_depth_map(ctx, faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces):
        if not ctx.return_depth:
            return grad_faces
        else:
            return rasterize_cuda.backward_depth_map(faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces, ctx.image_size)


class Rasterize(nn.Module):
    """
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    """

    def __init__(self, image_size, near, far, eps, background_color, return_rgb=False, return_alpha=False, return_depth=False):
        super(Rasterize, self).__init__()
        self.image_size = image_size
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

    def forward(self, faces, textures):
        if faces.device == 'cpu' or textures is not None and textures.device == 'cpu':
            raise TypeError('Rasterize module supports only cuda Tensors')
        return RasterizeFunction.apply(faces, textures, self.image_size, self.near, self.far, self.eps, self.background_color, self.return_rgb, self.return_alpha, self.return_depth)


class Renderer(nn.Module):

    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0, 0, 0], fill_back=True, camera_mode='projection', K=None, R=None, t=None, dist_coeffs=None, orig_size=1024, perspective=True, viewing_angle=30, camera_direction=[0, 0, 1], near=0.1, far=100, light_intensity_ambient=0.5, light_intensity_directional=0.5, light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1], light_direction=[0, 1, 0]):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1.0 / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')
        self.near = near
        self.far = far
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction
        self.rasterizer_eps = 0.001

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        """
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        """
        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, (list(reversed(range(faces.shape[-1]))))]), dim=1)
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, (list(reversed(range(faces.shape[-1]))))]), dim=1).detach()
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, (list(reversed(range(faces.shape[-1]))))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(faces_lighting, textures, self.light_intensity_ambient, self.light_intensity_directional, self.light_color_ambient, self.light_color_directional, self.light_direction)
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps, self.background_color)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, (list(reversed(range(faces.shape[-1]))))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(faces_lighting, textures, self.light_intensity_ambient, self.light_intensity_directional, self.light_color_ambient, self.light_color_directional, self.light_direction)
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        faces = nr.vertices_to_faces(vertices, faces)
        out = nr.rasterize_rgbad(faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps, self.background_color)
        return out['rgb'], out['depth'], out['alpha']

