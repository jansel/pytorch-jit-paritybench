import sys
_module = sys.modules[__name__]
del sys
conftest = _module
conf = _module
main = _module
main = _module
main = _module
kornia = _module
augmentation = _module
augmentation = _module
functional = _module
random_generator = _module
types = _module
utils = _module
color = _module
adjust = _module
core = _module
gray = _module
histogram = _module
hls = _module
hsv = _module
luv = _module
normalize = _module
rgb = _module
xyz = _module
ycbcr = _module
yuv = _module
zca = _module
constants = _module
contrib = _module
extract_patches = _module
max_blur_pool = _module
feature = _module
affine_shape = _module
hardnet = _module
laf = _module
nms = _module
orientation = _module
responses = _module
scale_space_detector = _module
siftdesc = _module
sosnet = _module
filters = _module
blur = _module
filter = _module
gaussian = _module
kernels = _module
laplacian = _module
median = _module
motion = _module
sobel = _module
geometry = _module
camera = _module
perspective = _module
pinhole = _module
conversions = _module
depth = _module
dsnt = _module
epipolar = _module
essential = _module
fundamental = _module
metrics = _module
numeric = _module
projection = _module
scene = _module
triangulation = _module
linalg = _module
spatial_soft_argmax = _module
transform = _module
affwarp = _module
crop = _module
flips = _module
imgwarp = _module
pyramid = _module
warp = _module
depth_warper = _module
homography_warper = _module
jit = _module
losses = _module
depth_smooth = _module
dice = _module
divergence = _module
focal = _module
psnr = _module
ssim = _module
total_variation = _module
tversky = _module
testing = _module
grid = _module
image = _module
confusion_matrix = _module
mean_iou = _module
one_hot = _module
pointcloud_io = _module
setup = _module
test = _module
test_augmentation = _module
test_functional = _module
test_perspective = _module
test_random_generator = _module
test_transformation_matrix = _module
test_adjust = _module
test_core = _module
test_gray = _module
test_histogram = _module
test_hls = _module
test_hsv = _module
test_luv = _module
test_normalize = _module
test_rgb = _module
test_xyz = _module
test_ycbcr = _module
test_yuv = _module
test_zca = _module
test_affine_shape_estimator = _module
test_hardnet = _module
test_laf = _module
test_local_features_orientation = _module
test_nms = _module
test_responces_local_features = _module
test_scale_space_detector = _module
test_siftdesc = _module
test_sosnet = _module
test_blur = _module
test_filters = _module
test_gaussian = _module
test_median = _module
test_motion = _module
test_sobel = _module
test_common = _module
test_epipolar_metrics = _module
test_essential = _module
test_fundamental = _module
test_numeric = _module
test_projection = _module
test_triangulation = _module
test_conversions = _module
test_depth = _module
test_dsnt = _module
test_linalg = _module
test_perspective = _module
test_pinhole = _module
test_spatial_softargmax = _module
test_affine = _module
test_crop = _module
test_flip = _module
test_imgwarp = _module
test_pyramid = _module
test_depth_warper = _module
test_homography_warper = _module
test_focal = _module
test_soft_argmax2d = _module
test_warp = _module
test_imgwarp_speed = _module
smoke_test = _module
test_contrib = _module
test_losses = _module
test_grid = _module
test_image = _module
test_metrics = _module
test_one_hot = _module
test_pointcloud_io = _module
color_adjust = _module
color_conversions = _module
data_augmentation = _module
filter_blurring = _module
filter_edges = _module
gaussian_blur = _module
hello_world = _module
total_variation_denoising = _module
warp_affine = _module
warp_perspective = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from itertools import product


from typing import Dict


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from typing import Callable


from typing import Tuple


from typing import Union


from typing import List


from typing import Optional


from typing import cast


from torch.nn.functional import pad


import random


import math


from torch.distributions import Uniform


from functools import reduce


from typing import TypeVar


from enum import Enum


from torch.nn.modules.utils import _pair


from typing import Iterable


import warnings


from torch.nn.functional import mse_loss


from torch.testing import assert_allclose


from torch.autograd import gradcheck


import logging


from time import time


import torchvision


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class InvDepth(nn.Module):

    def __init__(self, height, width, min_depth=0.5, max_depth=25.0):
        super(InvDepth, self).__init__()
        self._min_range = 1.0 / max_depth
        self._max_range = 1.0 / min_depth
        self.w = nn.Parameter(self._init_weights(height, width))

    def _init_weights(self, height, width):
        r1 = self._min_range
        r2 = self._min_range + (self._max_range - self._min_range) * 0.1
        w_init = (r1 - r2) * torch.rand(1, 1, height, width) + r2
        return w_init

    def forward(self):
        return self.w.clamp(min=self._min_range, max=self._max_range)


class MyHomography(nn.Module):

    def __init__(self, init_homo: torch.Tensor) ->None:
        super().__init__()
        self.homo = nn.Parameter(init_homo.clone().detach())

    def forward(self) ->torch.Tensor:
        return torch.unsqueeze(self.homo, dim=0)


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _to_bchw(tensor: torch.Tensor, color_channel_num: Optional[int]=None) ->torch.Tensor:
    """Converts a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)`, :math:`(H, W, C)` or
            :math:`(B, C, H, W)`.
        color_channel_num (Optional[int]): Color channel of the input tensor.
            If None, it will not alter the input channel.

    Returns:
        torch.Tensor: input tensor of the form :math:`(B, H, W, C)`.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(tensor)}')
    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(f'Input size must be a two, three or four dimensional tensor. Got {tensor.shape}')
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    if color_channel_num is not None and color_channel_num != 1:
        channel_list = [0, 1, 2, 3]
        channel_list.insert(1, channel_list.pop(color_channel_num))
        tensor = tensor.permute(*channel_list)
    return tensor


def _transform_input(input: torch.Tensor) ->torch.Tensor:
    """Reshape an input tensor to be (*, C, H, W). Accept either (H, W), (C, H, W) or (*, C, H, W).
    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor
    """
    return _to_bchw(input)


def _infer_batch_shape(input: UnionType) ->torch.Size:
    """Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


class AugmentationBase(nn.Module):
    """AugmentationBase base class for customized augmentation implementations. For any augmentation,
    the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.

    """

    def __init__(self, return_transform: bool=False) ->None:
        super(AugmentationBase, self).__init__()
        self.return_transform = return_transform

    def infer_batch_shape(self, input: UnionType) ->torch.Size:
        return _infer_batch_shape(input)

    def generate_parameters(self, input_shape: torch.Size) ->Dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        raise NotImplementedError

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]]=None, return_transform: Optional[bool]=None) ->UnionType:
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            self._params = self.generate_parameters(batch_shape)
        else:
            self._params = params
        if isinstance(input, tuple):
            output = self.apply_transform(input[0], self._params)
            transformation_matrix = self.compute_transformation(input[0], self._params)
            if return_transform:
                return output, input[1] @ transformation_matrix
            else:
                return output, input[1]
        output = self.apply_transform(input, self._params)
        if return_transform:
            transformation_matrix = self.compute_transformation(input, self._params)
            return output, transformation_matrix
        return output


class RandomHorizontalFlip(AugmentationBase):
    """Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = nn.Sequential(RandomHorizontalFlip(p=1.0, return_transform=True),
        ...                     RandomHorizontalFlip(p=1.0, return_transform=True))
        >>> seq(input)
        (tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 1., 1.]]]]), tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]))

    """

    def __init__(self, p: float=0.5, return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomHorizontalFlip, self).__init__(return_transform)
        self.p: float = p
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})'
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_hflip_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_hflip(input, params)


class RandomVerticalFlip(AugmentationBase):
    """Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Vertically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = RandomVerticalFlip(p=1.0, return_transform=True)
        >>> seq(input)
        (tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]]), tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  3.],
                 [ 0.,  0.,  1.]]]))

    """

    def __init__(self, p: float=0.5, return_transform: bool=False, same_on_batch: bool=False) ->None:
        super(RandomVerticalFlip, self).__init__(return_transform)
        self.p: float = p
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        repr = f'(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})'
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_vflip_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_vflip(input, params)


FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


class ColorJitter(AugmentationBase):
    """Change the brightness, contrast, saturation and hue randomly given tensor image or a batch of tensor images.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])
    """

    def __init__(self, brightness: FloatUnionType=0.0, contrast: FloatUnionType=0.0, saturation: FloatUnionType=0.0, hue: FloatUnionType=0.0, return_transform: bool=False, same_on_batch: bool=False) ->None:
        super(ColorJitter, self).__init__(return_transform)
        self.brightness: FloatUnionType = brightness
        self.contrast: FloatUnionType = contrast
        self.saturation: FloatUnionType = saturation
        self.hue: FloatUnionType = hue
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        repr = f'(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation},            hue={self.hue}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})'
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_color_jitter_generator(batch_shape[0], self.brightness, self.contrast, self.saturation, self.hue, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_color_jitter(input, params)


class RandomGrayscale(AugmentationBase):
    """Random Grayscale transformation according to a probability p value

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.1
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn((1, 3, 3, 3))
        >>> rec_er = RandomGrayscale(p=1.0)
        >>> rec_er(inputs)
        tensor([[[[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]]]])
    """

    def __init__(self, p: float=0.1, return_transform: bool=False, same_on_batch: bool=False) ->None:
        super(RandomGrayscale, self).__init__(return_transform)
        self.p = p
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        repr = f'(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})'
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_grayscale(input, params)


class RandomErasing(AugmentationBase):
    """
    Erases a random selected rectangle for each image in the batch, putting the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [scale[0], scale[1]) and an aspect ratio sampled
    between [ratio[0], ratio[1])

    Args:
        p (float): probability that the random erasing operation will be performed.
        scale (Tuple[float, float]): range of proportion of erased area against input image.
        ratio (Tuple[float, float]): range of aspect ratio of erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> rec_er = RandomErasing(1.0, (.4, .8), (.3, 1/.3))
        >>> rec_er(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])
    """

    def __init__(self, p: float=0.5, scale: Tuple[float, float]=(0.02, 0.33), ratio: Tuple[float, float]=(0.3, 3.3), value: float=0.0, return_transform: bool=False, same_on_batch: bool=False) ->None:
        super(RandomErasing, self).__init__(return_transform)
        self.p = p
        self.scale: Tuple[float, float] = scale
        self.ratio: Tuple[float, float] = ratio
        self.value: float = value
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        repr = f'(scale={self.scale}, ratio={self.ratio}, value={self.value}, '
        f"""return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_rectangles_params_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], p=self.p, scale=self.scale, ratio=self.ratio, value=self.value, same_on_batch=self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_erase_rectangles(input, params)


T = TypeVar('T', bound='Resample')


class Resample(Enum):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2

    @classmethod
    def get(cls, value: Union[str, int, T]) ->T:
        if type(value) == str:
            return cls[value.upper()]
        if type(value) == int:
            return cls(value)
        if type(value) == cls:
            return value
        raise TypeError()


class RandomPerspective(AugmentationBase):
    """Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation
                                 applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[[1., 0., 0.],
        ...                         [0., 1., 0.],
        ...                         [0., 0., 1.]]]])
        >>> aug = RandomPerspective(0.5, 1.0)
        >>> aug(inputs)
        tensor([[[[0.0000, 0.2289, 0.0000],
                  [0.0000, 0.4800, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(self, distortion_scale: float=0.5, p: float=0.5, interpolation: Union[str, int, Resample]=Resample.BILINEAR.name, return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomPerspective, self).__init__(return_transform)
        self.p: float = p
        self.distortion_scale: float = distortion_scale
        self.interpolation: Resample = Resample.get(interpolation)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(distortion_scale={self.distortion_scale}, p={self.p}, interpolation={self.interpolation.name}, '
        f"""return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_perspective_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], self.p, self.distortion_scale, self.interpolation, self.same_on_batch, self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_perspective_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_perspective(input, params)


TupleFloat = Tuple[float, float]


UnionFloat = Union[float, TupleFloat]


class RandomAffine(AugmentationBase):
    """Random affine transformation of the image keeping center invariant.

    Args:
        degrees (float or tuple): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), return_transform=True)
        >>> aug(input)
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
    """

    def __init__(self, degrees: UnionFloat, translate: Optional[TupleFloat]=None, scale: Optional[TupleFloat]=None, shear: Optional[UnionFloat]=None, resample: Union[str, int, Resample]=Resample.BILINEAR.name, return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomAffine, self).__init__(return_transform)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample: Resample = Resample.get(resample)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, '
        f"""resample={self.resample.name}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch}"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_affine_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], self.degrees, self.translate, self.scale, self.shear, self.resample, self.same_on_batch, self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_affine_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_affine(input, params)


class CenterCrop(AugmentationBase):
    """Crops the given torch.Tensor at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 3, 3)
        >>> aug = CenterCrop(2)
        >>> aug(inputs)
        tensor([[[[-0.1425, -1.1266],
                  [-0.0373, -0.6562]]]])
    """

    def __init__(self, size: Union[int, Tuple[int, int]], return_transform: bool=False) ->None:
        super(CenterCrop, self).__init__(return_transform)
        self.size = size

    def __repr__(self) ->str:
        repr = f'(size={self.size}, return_transform={self.return_transform}'
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        if isinstance(self.size, tuple):
            size_param = self.size[0], self.size[1]
        elif isinstance(self.size, int):
            size_param = self.size, self.size
        else:
            raise Exception(f'Invalid size type. Expected (int, tuple(int, int). Got: {type(self.size)}.')
        return rg.center_crop_params_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], size_param)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_crop_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_crop(input, params)


class RandomRotation(AugmentationBase):
    """Rotate a tensor image or a batch of tensor images a random amount of degrees.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will rotate the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        degrees (sequence or float or tensor): range of degrees to select from. If degrees is a number the
        range of degrees to select from will be (-degrees, +degrees)
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.tensor([[1., 0., 0., 2.],
        ...                       [0., 0., 0., 0.],
        ...                       [0., 1., 2., 0.],
        ...                       [0., 0., 1., 2.]])
        >>> seq = RandomRotation(degrees=45.0, return_transform=True)
        >>> seq(input)
        (tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                  [0.0000, 0.0029, 0.0000, 0.0176],
                  [0.0029, 1.0000, 1.9883, 0.0000],
                  [0.0000, 0.0088, 1.0117, 1.9649]]]]), tensor([[[ 1.0000, -0.0059,  0.0088],
                 [ 0.0059,  1.0000, -0.0088],
                 [ 0.0000,  0.0000,  1.0000]]]))
    """

    def __init__(self, degrees: FloatUnionType, interpolation: Union[str, int, Resample]=Resample.BILINEAR.name, return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomRotation, self).__init__(return_transform)
        self.degrees = degrees
        self.interpolation: Resample = Resample.get(interpolation)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(degrees={self.degrees}, interpolation={self.interpolation.name}, '
        f"""return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_rotation_generator(batch_shape[0], self.degrees, self.interpolation, self.same_on_batch, self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_rotate_tranformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_rotation(input, params)


BoarderUnionType = Union[int, Tuple[int, int], Tuple[int, int, int, int]]


class RandomCrop(AugmentationBase):
    """Random Crop on given size.

    Args:
        size (tuple): Desired output size of the crop, like (h, w).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2))
        >>> aug(inputs)
        tensor([[[[-0.6562, -1.0009],
                  [ 0.2223, -0.5507]]]])
    """

    def __init__(self, size: Tuple[int, int], padding: Optional[BoarderUnionType]=None, pad_if_needed: Optional[bool]=False, fill: int=0, padding_mode: str='constant', return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomCrop, self).__init__(return_transform)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(crop_size={self.size}, padding={self.padding}, fill={self.fill}, '
        f"""pad_if_needed={self.pad_if_needed}, padding_mode=${self.padding_mode}, """
        f"""return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), self.size, same_on_batch=self.same_on_batch, align_corners=self.align_corners)

    def precrop_padding(self, input: torch.Tensor) ->torch.Tensor:
        if self.padding is not None:
            if isinstance(self.padding, int):
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, tuple) and len(self.padding) == 2:
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, tuple) and len(self.padding) == 4:
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)
        if self.pad_if_needed and input.shape[-2] < self.size[0]:
            padding = [0, 0, self.size[0] - input.shape[-2], self.size[0] - input.shape[-2]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)
        if self.pad_if_needed and input.shape[-1] < self.size[1]:
            padding = [self.size[1] - input.shape[-1], self.size[1] - input.shape[-1], 0, 0]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)
        return input

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_crop_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_crop(input, params)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]]=None, return_transform: Optional[bool]=None) ->UnionType:
        if type(input) == tuple:
            input = self.precrop_padding(input[0]), input[1]
        else:
            input = self.precrop_padding(input)
        return super().forward(input, params, return_transform)


class RandomResizedCrop(AugmentationBase):
    """Random Crop on given size and resizing the cropped patch to another.

    Args:
        size (Tuple[int, int]): expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        >>> aug(inputs)
        tensor([[[[3.7500, 4.7500, 5.7500],
                  [5.2500, 6.2500, 7.2500],
                  [4.5000, 5.2500, 6.0000]]]])
    """

    def __init__(self, size: Tuple[int, int], scale: Tuple[float, float]=(0.08, 1.0), ratio: Tuple[float, float]=(3.0 / 4.0, 4.0 / 3.0), interpolation: Union[str, int, Resample]=Resample.BILINEAR.name, return_transform: bool=False, same_on_batch: bool=False, align_corners: bool=False) ->None:
        super(RandomResizedCrop, self).__init__(return_transform)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation: Resample = Resample.get(interpolation)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) ->str:
        repr = f'(size={self.size}, resize_to={self.scale}, resize_to={self.ratio}, '
        f"""interpolation={self.interpolation.name}, return_transform={self.return_transform}, """
        f"""same_on_batch={self.same_on_batch})"""
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        target_size = rg.random_crop_size_generator(self.size, self.scale, self.ratio)
        _target_size = int(target_size[0].data.item()), int(target_size[1].data.item())
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), _target_size, resize_to=self.size, same_on_batch=self.same_on_batch, align_corners=self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_crop_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_crop(input, params)


U = TypeVar('U', bound='BorderType')


class BorderType(Enum):
    CONSTANT = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3

    @classmethod
    def get(cls, value: Union[str, int, U]) ->U:
        if type(value) == str:
            return cls[value.upper()]
        if type(value) == int:
            return cls(value)
        if type(value) == cls:
            return value
        raise TypeError()


class RandomMotionBlur(AugmentationBase):
    """Blurs a tensor using the motion filter. Same transformation happens across batches.

    Args:
        kernel_size (int or Tuple[int, int]): motion kernel width and height (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range.
        angle (float or Tuple[float, float]): angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction (float or Tuple[float, float]): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.
        border_type (int, str or kornia.BorderType): the padding mode to be applied before convolving.
            CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> motion_blur = RandomMotionBlur(3, 35., 0.5)
        >>> motion_blur(input)
        tensor([[[[0.2761, 0.5200, 0.3753, 0.2423, 0.2193],
                  [0.3275, 0.5502, 0.5738, 0.5400, 0.3883],
                  [0.2132, 0.3857, 0.3056, 0.2520, 0.1890],
                  [0.3016, 0.6172, 0.6487, 0.4331, 0.2770],
                  [0.3865, 0.6221, 0.5538, 0.4862, 0.4206]]]])
    """

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], angle: Union[float, Tuple[float, float]], direction: Union[float, Tuple[float, float]], border_type: Union[int, str, BorderType]=BorderType.CONSTANT.name, return_transform: bool=False) ->None:
        super(RandomMotionBlur, self).__init__(return_transform)
        self.kernel_size: Union[int, Tuple[int, int]] = kernel_size
        self.angle: Union[float, Tuple[float, float]] = angle
        self.direction: Union[float, Tuple[float, float]] = direction
        self.border_type: BorderType = BorderType.get(border_type)

    def __repr__(self) ->str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}, border_type='{self.border_type.name.lower()}', return_transform={self.return_transform})"

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_motion_blur_generator(1, self.kernel_size, self.angle, self.direction, self.border_type)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_motion_blur(input, params)


class RandomSolarize(AugmentationBase):
    """ Solarize given tensor image or a batch of tensor images randomly.

    Args:
        thresholds (float or tuple): Default value is 0.1.
            If float x, threshold will be generated from (0.5 - x, 0.5 + x).
            If tuple (x, y), threshold will be generated from (x, y).
        additions (float or tuple): Default value is 0.1.
            If float x, addition will be generated from (-x, x).
            If tuple (x, y), addition will be generated from (x, y).
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> solarize = RandomSolarize(0.1, 0.1)
        >>> solarize(input)
        tensor([[[[0.4132, 0.1412, 0.1790, 0.2226, 0.3980],
                  [0.2754, 0.4194, 0.0130, 0.4538, 0.2771],
                  [0.4394, 0.4923, 0.1129, 0.2594, 0.3844],
                  [0.3909, 0.2118, 0.1094, 0.2516, 0.3728],
                  [0.2278, 0.0000, 0.4876, 0.0353, 0.5100]]]])
    """

    def __init__(self, thresholds: FloatUnionType=0.1, additions: FloatUnionType=0.1, same_on_batch: bool=False, return_transform: bool=False) ->None:
        super(RandomSolarize, self).__init__(return_transform)
        self.thresholds = thresholds
        self.additions = additions
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(thresholds={self.thresholds}, additions={self.additions}, same_on_batch={self.same_on_batch}, return_transform={self.return_transform})'

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_solarize_generator(batch_shape[0], self.thresholds, self.additions, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_solarize(input, params)


class RandomPosterize(AugmentationBase):
    """ Posterize given tensor image or a batch of tensor images randomly.

    Args:
        bits (int or tuple): Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).
            Default value is 3.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> posterize = RandomPosterize(3)
        >>> posterize(input)
        tensor([[[[0.4706, 0.7529, 0.0627, 0.1255, 0.2824],
                  [0.6275, 0.4706, 0.8784, 0.4392, 0.6275],
                  [0.3451, 0.3765, 0.0000, 0.1569, 0.2824],
                  [0.5020, 0.6902, 0.7843, 0.1569, 0.2510],
                  [0.6588, 0.9098, 0.3765, 0.8471, 0.4078]]]])
    """

    def __init__(self, bits: Union[int, Tuple[int, int], torch.Tensor]=3, same_on_batch: bool=False, return_transform: bool=False) ->None:
        super(RandomPosterize, self).__init__(return_transform)
        self.bits = bits
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(bits={self.bits}, same_on_batch={self.same_on_batch}, return_transform={self.return_transform})'

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_posterize_generator(batch_shape[0], self.bits, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_posterize(input, params)


class RandomSharpness(AugmentationBase):
    """ Sharpen given tensor image or a batch of tensor images randomly.

    Args:
        sharpness (float or tuple): factor of sharpness strength. Must be above 0. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> sharpness = RandomSharpness(1.)
        >>> sharpness(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.7720, 0.9537, 0.7566, 0.6323],
                  [0.3489, 0.7325, 0.5629, 0.6284, 0.2939],
                  [0.5185, 0.8648, 0.9106, 0.6249, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]],
        <BLANKLINE>
                 [[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.7720, 0.9537, 0.7566, 0.6323],
                  [0.3489, 0.7325, 0.5629, 0.6284, 0.2939],
                  [0.5185, 0.8648, 0.9106, 0.6249, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]],
        <BLANKLINE>
                 [[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.7720, 0.9537, 0.7566, 0.6323],
                  [0.3489, 0.7325, 0.5629, 0.6284, 0.2939],
                  [0.5185, 0.8648, 0.9106, 0.6249, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(self, sharpness: Union[float, Tuple[float, float], torch.Tensor]=0.5, same_on_batch: bool=False, return_transform: bool=False) ->None:
        super(RandomSharpness, self).__init__(return_transform)
        self.sharpness = sharpness
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(sharpness={self.sharpness}, return_transform={self.return_transform})'

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_sharpness_generator(batch_shape[0], self.sharpness, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_sharpness(input, params)


class RandomEqualize(AugmentationBase):
    """ Equalize given tensor image or a batch of tensor images randomly.

    Args:
        p (float): Probability to equalize an image. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> equalize = RandomEqualize(1.)
        >>> equalize(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(self, p: float=0.5, same_on_batch: bool=False, return_transform: bool=False) ->None:
        super(RandomEqualize, self).__init__(return_transform)
        self.p = p
        self.same_on_batch = same_on_batch

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(p={self.p}, return_transform={self.return_transform})'

    def generate_parameters(self, batch_shape: torch.Size) ->Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) ->torch.Tensor:
        return F.apply_equalize(input, params)


def adjust_saturation_raw(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust color saturation of an image. Expecting input to be in hsv format already.

    See :class:`~kornia.color.AdjustSaturation` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(saturation_factor, (float, torch.Tensor)):
        raise TypeError(f'The saturation_factor should be a float number or torch.Tensor.Got {type(saturation_factor)}')
    if isinstance(saturation_factor, float):
        saturation_factor = torch.tensor([saturation_factor])
    saturation_factor = saturation_factor.to(input.device)
    if (saturation_factor < 0).any():
        raise ValueError(f'Saturation factor must be non-negative. Got {saturation_factor}')
    for _ in input.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)
    h, s, v = torch.chunk(input, chunks=3, dim=-3)
    s_out: torch.Tensor = torch.clamp(s * saturation_factor, min=0, max=1)
    out: torch.Tensor = torch.cat([h, s_out, v], dim=-3)
    return out


pi = torch.tensor(3.141592653589793)


def hsv_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Convert an HSV image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): HSV Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    h: torch.Tensor = image[(...), (0), :, :] / (2 * pi)
    s: torch.Tensor = image[(...), (1), :, :]
    v: torch.Tensor = image[(...), (2), :, :]
    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = h * 6 % 6 - hi
    one: torch.Tensor = torch.tensor(1.0)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)
    out: torch.Tensor = torch.stack([hi, hi, hi], dim=-3)
    out[out == 0] = torch.stack((v, t, p), dim=-3)[out == 0]
    out[out == 1] = torch.stack((q, v, p), dim=-3)[out == 1]
    out[out == 2] = torch.stack((p, v, t), dim=-3)[out == 2]
    out[out == 3] = torch.stack((p, q, v), dim=-3)[out == 3]
    out[out == 4] = torch.stack((t, p, v), dim=-3)[out == 4]
    out[out == 5] = torch.stack((v, p, q), dim=-3)[out == 5]
    return out


def rgb_to_hsv(image: torch.Tensor) ->torch.Tensor:
    """Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    r: torch.Tensor = image[(...), (0), :, :]
    g: torch.Tensor = image[(...), (1), :, :]
    b: torch.Tensor = image[(...), (2), :, :]
    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]
    v: torch.Tensor = maxc
    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v
    s[torch.isnan(s)] = 0.0
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac
    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc
    h: torch.Tensor = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0
    h = h / 6.0 % 1.0
    h = 2 * pi * h
    return torch.stack([h, s, v], dim=-3)


def adjust_saturation(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust color saturation of an image.

    See :class:`~kornia.color.AdjustSaturation` for details.
    """
    x_hsv: torch.Tensor = rgb_to_hsv(input)
    x_adjusted: torch.Tensor = adjust_saturation_raw(x_hsv, saturation_factor)
    out: torch.Tensor = hsv_to_rgb(x_adjusted)
    return out


class AdjustSaturation(nn.Module):
    """Adjust color saturation of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\\*, N).
        saturation_factor (float):  How much to adjust the saturation. 0 will give a black
        and white image, 1 will give the original image while 2 will enhance the saturation
        by a factor of 2.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, saturation_factor: Union[float, torch.Tensor]) ->None:
        super(AdjustSaturation, self).__init__()
        self.saturation_factor: Union[float, torch.Tensor] = saturation_factor

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return adjust_saturation(input, self.saturation_factor)


def adjust_hue_raw(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust hue of an image. Expecting input to be in hsv format already.

    See :class:`~kornia.color.AdjustHue` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f'The hue_factor should be a float number or torch.Tensor in the range between [-PI, PI]. Got {type(hue_factor)}')
    if isinstance(hue_factor, float):
        hue_factor = torch.tensor([hue_factor])
    hue_factor = hue_factor.to(input.device)
    if ((hue_factor < -pi) | (hue_factor > pi)).any():
        raise ValueError(f'Hue-factor must be in the range [-PI, PI]. Got {hue_factor}')
    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)
    h, s, v = torch.chunk(input, chunks=3, dim=-3)
    divisor: float = 2 * pi.item()
    h_out: torch.Tensor = torch.fmod(h + hue_factor, divisor)
    out: torch.Tensor = torch.cat([h_out, s, v], dim=-3)
    return out


def adjust_hue(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust hue of an image.

    See :class:`~kornia.color.AdjustHue` for details.
    """
    x_hsv: torch.Tensor = rgb_to_hsv(input)
    x_adjusted: torch.Tensor = adjust_hue_raw(x_hsv, hue_factor)
    out: torch.Tensor = hsv_to_rgb(x_adjusted)
    return out


class AdjustHue(nn.Module):
    """Adjust hue of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\\*, N).
        hue_factor (float): How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, hue_factor: Union[float, torch.Tensor]) ->None:
        super(AdjustHue, self).__init__()
        self.hue_factor: Union[float, torch.Tensor] = hue_factor

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return adjust_hue(input, self.hue_factor)


def adjust_gamma(input: torch.Tensor, gamma: Union[float, torch.Tensor], gain: Union[float, torch.Tensor]=1.0) ->torch.Tensor:
    """Perform gamma correction on an image.

    See :class:`~kornia.color.AdjustGamma` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(gamma, (float, torch.Tensor)):
        raise TypeError(f'The gamma should be a positive float or torch.Tensor. Got {type(gamma)}')
    if not isinstance(gain, (float, torch.Tensor)):
        raise TypeError(f'The gain should be a positive float or torch.Tensor. Got {type(gain)}')
    if isinstance(gamma, float):
        gamma = torch.tensor([gamma])
    if isinstance(gain, float):
        gain = torch.tensor([gain])
    gamma = gamma.to(input.device)
    gain = gain.to(input.device)
    if (gamma < 0.0).any():
        raise ValueError(f'Gamma must be non-negative. Got {gamma}')
    if (gain < 0.0).any():
        raise ValueError(f'Gain must be non-negative. Got {gain}')
    for _ in input.shape[1:]:
        gamma = torch.unsqueeze(gamma, dim=-1)
        gain = torch.unsqueeze(gain, dim=-1)
    x_adjust: torch.Tensor = gain * torch.pow(input, gamma)
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)
    return out


class AdjustGamma(nn.Module):
    """Perform gamma correction on an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\\*, N).
        gamma (float): Non negative real number, same as γ\\gammaγ in the equation.
          gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
          dark regions lighter.
        gain (float, optional): The constant multiplier. Default 1.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, gamma: Union[float, torch.Tensor], gain: Union[float, torch.Tensor]=1.0) ->None:
        super(AdjustGamma, self).__init__()
        self.gamma: Union[float, torch.Tensor] = gamma
        self.gain: Union[float, torch.Tensor] = gain

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return adjust_gamma(input, self.gamma, self.gain)


def adjust_contrast(input: torch.Tensor, contrast_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust Contrast of an image.

    See :class:`~kornia.color.AdjustContrast` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(contrast_factor, (float, torch.Tensor)):
        raise TypeError(f'The factor should be either a float or torch.Tensor. Got {type(contrast_factor)}')
    if isinstance(contrast_factor, float):
        contrast_factor = torch.tensor([contrast_factor])
    contrast_factor = contrast_factor.to(input.device)
    if (contrast_factor < 0).any():
        raise ValueError(f'Contrast factor must be non-negative. Got {contrast_factor}')
    for _ in input.shape[1:]:
        contrast_factor = torch.unsqueeze(contrast_factor, dim=-1)
    x_adjust: torch.Tensor = input * contrast_factor
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)
    return out


class AdjustContrast(nn.Module):
    """Adjust Contrast of an image. This implementation aligns OpenCV, not PIL. Hence,
    the output differs from TorchVision.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image to be adjusted in the shape of (\\*, N).
        contrast_factor (Union[float, torch.Tensor]): Contrast adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, contrast_factor: Union[float, torch.Tensor]) ->None:
        super(AdjustContrast, self).__init__()
        self.contrast_factor: Union[float, torch.Tensor] = contrast_factor

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return adjust_contrast(input, self.contrast_factor)


def adjust_brightness(input: torch.Tensor, brightness_factor: Union[float, torch.Tensor]) ->torch.Tensor:
    """Adjust Brightness of an image.

    See :class:`~kornia.color.AdjustBrightness` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(brightness_factor, (float, torch.Tensor)):
        raise TypeError(f'The factor should be either a float or torch.Tensor. Got {type(brightness_factor)}')
    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor([brightness_factor])
    brightness_factor = brightness_factor.to(input.device)
    for _ in input.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)
    x_adjust: torch.Tensor = input + brightness_factor
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)
    return out


class AdjustBrightness(nn.Module):
    """Adjust Brightness of an image. This implementation aligns OpenCV, not PIL. Hence,
    the output differs from TorchVision.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Input to be adjusted in the shape of (\\*, N).
        brightness_factor (Union[float, torch.Tensor]): Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, brightness_factor: Union[float, torch.Tensor]) ->None:
        super(AdjustBrightness, self).__init__()
        self.brightness_factor: Union[float, torch.Tensor] = brightness_factor

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return adjust_brightness(input, self.brightness_factor)


def add_weighted(src1: torch.Tensor, alpha: float, src2: torch.Tensor, beta: float, gamma: float) ->torch.Tensor:
    """Blend two Tensors.

    See :class:`~kornia.color.AddWeighted` for details.
    """
    if not isinstance(src1, torch.Tensor):
        raise TypeError('src1 should be a tensor. Got {}'.format(type(src1)))
    if not isinstance(src2, torch.Tensor):
        raise TypeError('src2 should be a tensor. Got {}'.format(type(src2)))
    if not isinstance(alpha, float):
        raise TypeError('alpha should be a float. Got {}'.format(type(alpha)))
    if not isinstance(beta, float):
        raise TypeError('beta should be a float. Got {}'.format(type(beta)))
    if not isinstance(gamma, float):
        raise TypeError('gamma should be a float. Got {}'.format(type(gamma)))
    return src1 * alpha + src2 * beta + gamma


class AddWeighted(nn.Module):
    """Calculates the weighted sum of two Tensors.

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1 (torch.Tensor): Tensor.
        alpha (float): weight of the src1 elements.
        src2 (torch.Tensor): Tensor of same size and channel number as src1.
        beta (float): weight of the src2 elements.
        gamma (float): scalar added to each sum.

    Returns:
        torch.Tensor: Weighted Tensor.
    """

    def __init__(self, alpha: float, beta: float, gamma: float) ->None:
        super(AddWeighted, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, src1: torch.Tensor, src2: torch.Tensor) ->torch.Tensor:
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)


class RgbToGrayscale(nn.Module):
    """convert RGB image to grayscale version of image.

    the image data is assumed to be in the range of (0, 1).

    args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    returns:
        torch.Tensor: grayscale version of the image.

    shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = kornia.color.RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) ->None:
        super(RgbToGrayscale, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return rgb_to_grayscale(input)


def bgr_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Convert a BGR image to RGB.

    See :class:`~kornia.color.BgrToRgb` for details.

    Args:
        image (torch.Tensor): BGR Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W).Got {}'.format(image.shape))
    out: torch.Tensor = image.flip(-3)
    return out


def bgr_to_grayscale(input: torch.Tensor) ->torch.Tensor:
    """Convert a BGR image to grayscale.

    See :class:`~kornia.color.BgrToGrayscale` for details.

    Args:
        input (torch.Tensor): BGR image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(input.shape))
    input_rgb = bgr_to_rgb(input)
    gray: torch.Tensor = rgb_to_grayscale(input_rgb)
    return gray


class BgrToGrayscale(nn.Module):
    """convert BGR image to grayscale version of image.

    the image data is assumed to be in the range of (0, 1).

    args:
        input (torch.Tensor): BGR image to be converted to grayscale.

    returns:
        torch.Tensor: grayscale version of the image.

    shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = kornia.color.BgrToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) ->None:
        super(BgrToGrayscale, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return bgr_to_grayscale(input)


def hls_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Convert an HLS image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): HLS Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    h: torch.Tensor = image[(...), (0), :, :] * 360 / (2 * pi)
    l: torch.Tensor = image[(...), (1), :, :]
    s: torch.Tensor = image[(...), (2), :, :]
    kr = (0 + h / 30) % 12
    kg = (8 + h / 30) % 12
    kb = (4 + h / 30) % 12
    a = s * torch.min(l, torch.tensor(1.0) - l)
    ones_k = torch.ones_like(kr)
    fr: torch.Tensor = l - a * torch.max(torch.min(torch.min(kr - torch.tensor(3.0), torch.tensor(9.0) - kr), ones_k), -1 * ones_k)
    fg: torch.Tensor = l - a * torch.max(torch.min(torch.min(kg - torch.tensor(3.0), torch.tensor(9.0) - kg), ones_k), -1 * ones_k)
    fb: torch.Tensor = l - a * torch.max(torch.min(torch.min(kb - torch.tensor(3.0), torch.tensor(9.0) - kb), ones_k), -1 * ones_k)
    out: torch.Tensor = torch.stack([fr, fg, fb], dim=-3)
    return out


class HlsToRgb(nn.Module):
    """Convert image from HLS to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): HLS image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return hls_to_rgb(image)


def rgb_to_hls(image: torch.Tensor) ->torch.Tensor:
    """Convert an RGB image to HLS
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HLS.


    Returns:
        torch.Tensor: HLS version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    r: torch.Tensor = image[(...), (0), :, :]
    g: torch.Tensor = image[(...), (1), :, :]
    b: torch.Tensor = image[(...), (2), :, :]
    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]
    imax: torch.Tensor = image.max(-3)[1]
    l: torch.Tensor = (maxc + minc) / 2
    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = torch.where(l < 0.5, deltac / (maxc + minc), deltac / (torch.tensor(2.0) - (maxc + minc)))
    hi: torch.Tensor = torch.zeros_like(deltac)
    hi[imax == 0] = ((g - b) / deltac % 6)[imax == 0]
    hi[imax == 1] = ((b - r) / deltac + 2)[imax == 1]
    hi[imax == 2] = ((r - g) / deltac + 4)[imax == 2]
    h: torch.Tensor = 2.0 * pi * (60.0 * hi) / 360.0
    image_hls: torch.Tensor = torch.stack([h, l, s], dim=-3)
    image_hls[torch.isnan(image_hls)] = 0.0
    return image_hls


class RgbToHls(nn.Module):
    """Convert image from RGB to HLS
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HLS.

    returns:
        torch.tensor: HLS version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = kornia.color.RgbToHls()
        >>> output = hls(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_hls(image)


class HsvToRgb(nn.Module):
    """Convert image from HSV to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): HSV image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(HsvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return hsv_to_rgb(image)


class RgbToHsv(nn.Module):
    """Convert image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HSV.

    returns:
        torch.tensor: HSV version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = kornia.color.RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(RgbToHsv, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_hsv(image)


def rgb_to_xyz(image: torch.Tensor) ->torch.Tensor:
    """Converts a RGB image to XYZ.

    See :class:`~kornia.color.RgbToXyz` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ.

    Returns:
        torch.Tensor: XYZ version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    r: torch.Tensor = image[(...), (0), :, :]
    g: torch.Tensor = image[(...), (1), :, :]
    b: torch.Tensor = image[(...), (2), :, :]
    x: torch.Tensor = 0.412453 * r + 0.35758 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.71516 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b
    out: torch.Tensor = torch.stack((x, y, z), -3)
    return out


def rgb_to_luv(image: torch.Tensor, eps: float=1e-12) ->torch.Tensor:
    """Converts a RGB image to Luv.

    See :class:`~kornia.color.RgbToLuv` for details.

    Args:
        image (torch.Tensor): RGB image
        eps (float): for numerically stability when dividing. Default: 1e-8.

    Returns:
        torch.Tensor : Luv image
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    r: torch.Tensor = image[(...), (0), :, :]
    g: torch.Tensor = image[(...), (1), :, :]
    b: torch.Tensor = image[(...), (2), :, :]
    rs: torch.Tensor = torch.where(r > 0.04045, torch.pow((r + 0.055) / 1.055, 2.4), r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045, torch.pow((g + 0.055) / 1.055, 2.4), g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045, torch.pow((b + 0.055) / 1.055, 2.4), b / 12.92)
    image_s = torch.stack((rs, gs, bs), dim=-3)
    xyz_im: torch.Tensor = rgb_to_xyz(image_s)
    x: torch.Tensor = xyz_im[(...), (0), :, :]
    y: torch.Tensor = xyz_im[(...), (1), :, :]
    z: torch.Tensor = xyz_im[(...), (2), :, :]
    L: torch.Tensor = torch.where(torch.gt(y, 0.008856), 116.0 * torch.pow(y, 1.0 / 3.0) - 16.0, 903.3 * y)
    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = 4 * xyz_ref_white[0] / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = 9 * xyz_ref_white[1] / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    u_p: torch.Tensor = 4 * x / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = 9 * y / (x + 15 * y + 3 * z + eps)
    u: torch.Tensor = 13 * L * (u_p - u_w)
    v: torch.Tensor = 13 * L * (v_p - v_w)
    out = torch.stack((L, u, v), dim=-3)
    return out


class RgbToLuv(nn.Module):
    """Converts an image from RGB to Luv

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 illuminant and Observer 2.

    args:
        image (torch.Tensor): RGB image to be converted to Luv.

    returns:
        torch.Tensor: Luv version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> luv = kornia.color.RgbToLuv()
        >>> output = luv(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) ->None:
        super(RgbToLuv, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_luv(image)


def xyz_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Converts a XYZ image to RGB.

    See :class:`~kornia.color.XyzToRgb` for details.

    Args:
        image (torch.Tensor): XYZ Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    x: torch.Tensor = image[(...), (0), :, :]
    y: torch.Tensor = image[(...), (1), :, :]
    z: torch.Tensor = image[(...), (2), :, :]
    r: torch.Tensor = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g: torch.Tensor = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b: torch.Tensor = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z
    out: torch.Tensor = torch.stack((r, g, b), dim=-3)
    return out


def luv_to_rgb(image: torch.Tensor, eps: float=1e-12) ->torch.Tensor:
    """Converts a Luv image to RGB.

    See :class:`~kornia.color.LuvToRgb` for details.

    Args:
        image (torch.Tensor): Luv image
        eps (float): for numerically stability when dividing. Default: 1e-8.

    Returns:
        torch.Tensor : RGB image
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    L: torch.Tensor = image[(...), (0), :, :]
    u: torch.Tensor = image[(...), (1), :, :]
    v: torch.Tensor = image[(...), (2), :, :]
    y: torch.Tensor = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)
    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = 4 * xyz_ref_white[0] / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = 9 * xyz_ref_white[1] / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    a: torch.Tensor = u_w + u / (13 * L + eps)
    d: torch.Tensor = v_w + v / (13 * L + eps)
    c: torch.Tensor = 3 * y * (5 * d - 3)
    z: torch.Tensor = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x: torch.Tensor = -(c / (d + eps) + 3.0 * z)
    xyz_im: torch.Tensor = torch.stack((x, y, z), -3)
    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)
    rs: torch.Tensor = rgbs_im[(...), (0), :, :]
    gs: torch.Tensor = rgbs_im[(...), (1), :, :]
    bs: torch.Tensor = rgbs_im[(...), (2), :, :]
    r: torch.Tensor = torch.where(rs > 0.0031308, 1.055 * torch.pow(rs, 1 / 2.4) - 0.055, 12.92 * rs)
    g: torch.Tensor = torch.where(gs > 0.0031308, 1.055 * torch.pow(gs, 1 / 2.4) - 0.055, 12.92 * gs)
    b: torch.Tensor = torch.where(bs > 0.0031308, 1.055 * torch.pow(bs, 1 / 2.4) - 0.055, 12.92 * bs)
    rgb_im: torch.Tensor = torch.stack((r, g, b), dim=-3)
    return rgb_im


class LuvToRgb(nn.Module):
    """Converts an image from Luv to RGB

    args:
        image (torch.Tensor): Luv image to be converted to RGB.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.LuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) ->None:
        super(LuvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return luv_to_rgb(image)


def normalize(data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) ->torch.Tensor:
    """Normalise the image with channel-wise mean and standard deviation.

    See :class:`~kornia.color.Normalize` for details.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    Returns:
        torch.Tensor: The normalised image tensor.

    """
    if isinstance(mean, float):
        mean = torch.tensor([mean])
    if isinstance(std, float):
        std = torch.tensor([std])
    if not torch.is_tensor(data):
        raise TypeError('data should be a tensor. Got {}'.format(type(data)))
    if not torch.is_tensor(mean):
        raise TypeError('mean should be a tensor or a float. Got {}'.format(type(mean)))
    if not torch.is_tensor(std):
        raise TypeError('std should be a tensor or float. Got {}'.format(type(std)))
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError('mean length and number of channels do not match')
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError('std length and number of channels do not match')
    if mean.shape:
        mean = mean[(...), :, (None), (None)]
    if std.shape:
        std = std[(...), :, (None), (None)]
    out: torch.Tensor = (data - mean) / std
    return out


class Normalize(nn.Module):
    """Normalize a tensor image or a batch of tensor images with mean and standard deviation.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) ->None:
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Normalises an input tensor by the mean and standard deviation.

        Args:
            input: image tensor of size (*, H, W).

        Returns:
            normalised tensor with same size as input (*, H, W).

        """
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + repr


def denormalize(data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) ->torch.Tensor:
    """Denormalize the image given channel-wise mean and standard deviation.

    See :class:`~kornia.color.Normalize` for details.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    Returns:
        torch.Tensor: The normalised image tensor.

    """
    if isinstance(mean, float):
        mean = torch.tensor([mean])
    if isinstance(std, float):
        std = torch.tensor([std])
    if not torch.is_tensor(data):
        raise TypeError('data should be a tensor. Got {}'.format(type(data)))
    if not torch.is_tensor(mean):
        raise TypeError('mean should be a tensor or a float. Got {}'.format(type(mean)))
    if not torch.is_tensor(std):
        raise TypeError('std should be a tensor or float. Got {}'.format(type(std)))
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError('mean length and number of channels do not match')
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError('std length and number of channels do not match')
    if mean.shape:
        mean = mean[(...), :, (None), (None)]
    if std.shape:
        std = std[(...), :, (None), (None)]
    out: torch.Tensor = data * std + mean
    return out


class Denormalize(nn.Module):
    """Denormalize a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will denormalize each channel of the input ``torch.Tensor``
    i.e. ``input[channel] = (input[channel] * std[channel]) + mean[channel]``

    Args:
        mean (torch.Tensor or float): Mean for each channel.
        std (torch.Tensor or float): Standard deviations for each channel.

    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) ->None:
        super(Denormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Denormalises an input tensor by the mean and standard deviation.

        Args:
            input: image tensor of size (*, H, W).

        Returns:
            normalised tensor with same size as input (*, H, W).

        """
        return denormalize(input, self.mean, self.std)

    def __repr__(self):
        repr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + repr


class BgrToRgb(nn.Module):
    """Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(BgrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return bgr_to_rgb(image)


def rgb_to_bgr(image: torch.Tensor) ->torch.Tensor:
    """Convert a RGB image to BGR.

    See :class:`~kornia.color.RgbToBgr` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to BGR.

    Returns:
        torch.Tensor: BGR version of the image.
    """
    return bgr_to_rgb(image)


class RgbToBgr(nn.Module):
    """Convert image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: BGR version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = kornia.color.RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(RgbToBgr, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_bgr(image)


def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) ->torch.Tensor:
    """Convert image from RGB to RGBA.

    See :class:`~kornia.color.RgbToRgba` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA.
        alpha_val (float, torch.Tensor): A float number for the alpha value.

    Returns:
        torch.Tensor: RGBA version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W).Got {image.shape}')
    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f'alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}')
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)
    a: torch.Tensor = cast(torch.Tensor, alpha_val)
    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))
    return torch.cat([r, g, b, a], dim=-3)


class RgbToRgba(nn.Module):
    """Convert image from RGB to RGBA.

    Add an alpha channel to existing RGB image.

    Args:
        alpha_val (float, torch.Tensor): A float number for the alpha value.

    Returns:
        torch.Tensor: RGBA version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = kornia.color.RgbToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) ->None:
        super(RgbToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_rgba(image, self.alpha_val)


class BgrToRgba(nn.Module):
    """Convert image from BGR to RGBA.

    Add an alpha channel to existing BGR image.

    Args:
        alpha_val (float, torch.Tensor): A float number for the alpha value.

    Returns:
        torch.Tensor: RGBA version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = kornia.color.BgrToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) ->None:
        super(BgrToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_rgba(image, self.alpha_val)


def rgba_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Convert image from RGBA to RGB.

    See :class:`~kornia.color.RgbaToRgb` for details.

    Args:
        image (torch.Tensor): RGBA Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W).Got {image.shape}')
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)
    a_one = torch.tensor(1.0) - a
    r_new: torch.Tensor = a_one * r + a * r
    g_new: torch.Tensor = a_one * g + a * g
    b_new: torch.Tensor = a_one * b + a * b
    return torch.cat([r, g, b], dim=-3)


class RgbaToRgb(nn.Module):
    """Convert image from RGBA to RGB.

    Remove an alpha channel from RGB image.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = kornia.color.RgbaToRgb()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def __init__(self) ->None:
        super(RgbaToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgba_to_rgb(image)


def rgba_to_bgr(image: torch.Tensor) ->torch.Tensor:
    """Convert image from RGBA to BGR.

    See :class:`~kornia.color.RgbaToBgr` for details.

    Args:
        image (torch.Tensor): RGBA Image to be converted to BGR.

    Returns:
        torch.Tensor: BGR version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W).Got {image.shape}')
    x_rgb: torch.Tensor = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


class RgbaToBgr(nn.Module):
    """Convert image from RGBA to BGR.

    Remove an alpha channel from BGR image.

    returns:
        torch.Tensor: BGR version of the image.

    shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = kornia.color.RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def __init__(self) ->None:
        super(RgbaToBgr, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgba_to_bgr(image)


class RgbToXyz(nn.Module):
    """Converts an image from RGB to XYZ

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to XYZ.

    returns:
        torch.Tensor: XYZ version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = kornia.color.RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def __init__(self) ->None:
        super(RgbToXyz, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_xyz(image)


class XyzToRgb(nn.Module):
    """Converts an image from XYZ to RGB

    args:
        image (torch.Tensor): XYZ image to be converted to RGB.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def __init__(self) ->None:
        super(XyzToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return xyz_to_rgb(image)


def ycbcr_to_rgb(image: torch.Tensor) ->torch.Tensor:
    """Convert an YCbCr image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    y: torch.Tensor = image[(...), (0), :, :]
    cb: torch.Tensor = image[(...), (1), :, :]
    cr: torch.Tensor = image[(...), (2), :, :]
    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta
    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack((r, g, b), -3)


class YcbcrToRgb(nn.Module):
    """Convert image from YCbCr to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): YCbCr image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return ycbcr_to_rgb(image)


def rgb_to_ycbcr(image: torch.Tensor) ->torch.Tensor:
    """Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))
    r: torch.Tensor = image[(...), (0), :, :]
    g: torch.Tensor = image[(...), (1), :, :]
    b: torch.Tensor = image[(...), (2), :, :]
    delta = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack((y, cb, cr), -3)


class RgbToYcbcr(nn.Module):
    """Convert image from RGB to YCbCr
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to YCbCr.

    returns:
        torch.tensor: YCbCr version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = kornia.color.RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    def __init__(self) ->None:
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_ycbcr(image)


def rgb_to_yuv(input: torch.Tensor) ->torch.Tensor:
    """Convert an RGB image to YUV
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to YUV.
    Returns:
        torch.Tensor: YUV version of the image.
    See :class:`~kornia.color.RgbToYuv` for details."""
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {type(input)}')
    if not (len(input.shape) == 3 or len(input.shape) == 4):
        raise ValueError(f'Input size must have a shape of (*, 3, H, W) or (3, H, W). Got {input.shape}')
    if input.shape[-3] != 3:
        raise ValueError(f'Expected input to have 3 channels, got {input.shape[-3]}')
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.1 * b
    yuv_img: torch.Tensor = torch.cat((y, u, v), -3)
    return yuv_img


class RgbToYuv(nn.Module):
    """Convert image from RGB to YUV
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to YUV.
    returns:
        torch.tensor: YUV version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = kornia.color.RgbToYuv()
        >>> output = yuv(input)  # 2x3x4x5
    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def __init__(self) ->None:
        super(RgbToYuv, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return rgb_to_yuv(input)


def yuv_to_rgb(input: torch.Tensor) ->torch.Tensor:
    """Convert an YUV image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): YUV Image to be converted to RGB.
    Returns:
        torch.Tensor: RGB version of the image.
    See :class:`~kornia.color.YuvToRgb` for details."""
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {type(input)}')
    if not (len(input.shape) == 3 or len(input.shape) == 4):
        raise ValueError(f'Input size must have a shape of (*, 3, H, W) or (3, H, W). Got {input.shape}')
    if input.shape[-3] != 3:
        raise ValueError(f'Expected input to have 3 channels, got {input.shape[-3]}')
    y, u, v = torch.chunk(input, chunks=3, dim=-3)
    r: torch.Tensor = y + 1.14 * v
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u
    rgb_img: torch.Tensor = torch.cat((r, g, b), -3)
    return rgb_img


class YuvToRgb(nn.Module):
    """Convert image from YUV to RGB
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): YUV image to be converted to RGB.
    returns:
        torch.tensor: RGB version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) ->None:
        super(YuvToRgb, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return yuv_to_rgb(input)


def linear_transform(inp: torch.Tensor, transform_matrix: torch.Tensor, mean_vector: torch.Tensor, dim: int=0) ->torch.Tensor:
    """

    Given a transformation matrix and a mean vector, this function will flatten
    the input tensor along the given dimension and subtract the mean vector
    from it. Then the dot product with the transformation matrix will be computed
    and then the resulting tensor is reshaped to the original input shape.

    .. math::

        \\mathbf{X}_{T} = (\\mathbf{X - \\mu})(T)

    args:
        inp (torch.Tensor): Input data :math:`X`
        transform_matrix (torch.Tensor): Transform matrix :math:`T`
        mean_vector (torch.Tensor): mean vector :math:`\\mu`
        dim (int): Batch dimension. Default = 0

    shapes:
        - inp: :math:`(D_0,...,D_{\\text{dim}},...,D_N)` is a batch of N-D tensors.
        - transform_matrix: :math:`(\\Pi_{d=0,d\\neq \\text{dim}}^N D_d, \\Pi_{d=0,d\\neq \\text{dim}}^N D_d)`
        - mean_vector: :math:`(1, \\Pi_{d=0,d\\neq \\text{dim}}^N D_d)`

    returns:
        torch.Tensor : Transformed data

    Example:
        >>> # Example where dim = 3
        >>> inp = torch.ones((10,3,4,5))
        >>> transform_mat = torch.ones((10*3*4,10*3*4))
        >>> mean = 2*torch.ones((1,10*3*4))
        >>> out = kornia.color.linear_transform(inp, transform_mat, mean, 3)
        >>> print(out) # Should a be (10,3,4,5) tensor of -120s
        >>> # Example where dim = 0
        >>> inp = torch.ones((10,2))
        >>> transform_mat = torch.ones((2,2))
        >>> mean = torch.zeros((1,2))
        >>> out = kornia.color.linear_transform(inp, transform_mat, mean)
        >>> print(out) # Should a be (10,3,4,5) tensor of 2s


    """
    inp_size = inp.size()
    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError('Dimension out of range (expected to be in range of [{},{}], but got {}'.format(-len(inp_size), len(inp_size) - 1, dim))
    if dim < 0:
        dim = len(inp_size) + dim
    feat_dims = torch.cat([torch.arange(0, dim), torch.arange(dim + 1, len(inp_size))])
    perm = torch.cat([torch.tensor([dim]), feat_dims])
    perm_inv = torch.argsort(perm)
    new_order: List[int] = perm.tolist()
    inv_order: List[int] = perm_inv.tolist()
    N = inp_size[dim]
    feature_sizes = torch.tensor(inp_size[0:dim] + inp_size[dim + 1:])
    num_features: int = int(torch.prod(feature_sizes).item())
    inp_permute = inp.permute(new_order)
    inp_flat = inp_permute.reshape((-1, num_features))
    inp_center = inp_flat - mean_vector
    inp_transformed = inp_center.mm(transform_matrix)
    inp_transformed = inp_transformed.reshape(inp_permute.size())
    inp_transformed = inp_transformed.permute(inv_order)
    return inp_transformed


def zca_mean(inp: torch.Tensor, dim: int=0, unbiased: bool=True, eps: float=1e-06, return_inverse: bool=False) ->Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """

    Computes the ZCA whitening matrix and mean vector. The output can be used with
    :py:meth:`~kornia.color.linear_transform`

    See :class:`~kornia.color.ZCAWhitening` for details.


    args:
        inp (torch.Tensor) : input data tensor
        dim (int): Specifies the dimension that serves as the samples dimension. Default = 0
        unbiased (bool): Whether to use the unbiased estimate of the covariance matrix. Default = True
        eps (float) : a small number used for numerical stability. Default = 0
        return_inverse (bool): Whether to return the inverse ZCA transform.

    shapes:
        - inp: :math:`(D_0,...,D_{\\text{dim}},...,D_N)` is a batch of N-D tensors.
        - transform_matrix: :math:`(\\Pi_{d=0,d\\neq \\text{dim}}^N D_d, \\Pi_{d=0,d\\neq \\text{dim}}^N D_d)`
        - mean_vector: :math:`(1, \\Pi_{d=0,d\\neq \\text{dim}}^N D_d)`
        - inv_transform: same shape as the transform matrix

    returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A tuple containing the ZCA matrix and the mean vector. If return_inverse is set to True,
        then it returns the inverse ZCA matrix, otherwise it returns None.

    Examples:
        >>> from kornia.color import zca_mean
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> transform_matrix, mean_vector,_ = zca_mean(x) # Returns transformation matrix and data mean
        >>> x = torch.rand(3,20,2,2)
        >>> transform_matrix, mean_vector, inv_transform = zca_mean(x, dim = 1, return_inverse = True)
        >>> # transform_matrix.size() equals (12,12) and the mean vector.size equal (1,12)

    """
    if not isinstance(inp, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(inp)))
    if not isinstance(eps, float):
        raise TypeError(f'eps type is not a float. Got{type(eps)}')
    if not isinstance(unbiased, bool):
        raise TypeError(f'unbiased type is not bool. Got{type(unbiased)}')
    if not isinstance(dim, int):
        raise TypeError("Argument 'dim' must be of type int. Got {}".format(type(dim)))
    if not isinstance(return_inverse, bool):
        raise TypeError('Argument return_inverse must be of type bool {}'.format(type(return_inverse)))
    inp_size = inp.size()
    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError('Dimension out of range (expected to be in range of [{},{}], but got {}'.format(-len(inp_size), len(inp_size) - 1, dim))
    if dim < 0:
        dim = len(inp_size) + dim
    feat_dims = torch.cat([torch.arange(0, dim), torch.arange(dim + 1, len(inp_size))])
    new_order: List[int] = torch.cat([torch.tensor([dim]), feat_dims]).tolist()
    inp_permute = inp.permute(new_order)
    N = inp_size[dim]
    feature_sizes = torch.tensor(inp_size[0:dim] + inp_size[dim + 1:])
    num_features: int = int(torch.prod(feature_sizes).item())
    mean: torch.Tensor = torch.mean(inp_permute, dim=0, keepdim=True)
    mean = mean.reshape((1, num_features))
    inp_center_flat: torch.Tensor = inp_permute.reshape((N, num_features)) - mean
    cov = inp_center_flat.t().mm(inp_center_flat)
    if unbiased:
        cov = cov / float(N - 1)
    else:
        cov = cov / float(N)
    U, S, _ = torch.svd(cov)
    S = S.reshape(-1, 1)
    S_inv_root: torch.Tensor = torch.rsqrt(S + eps)
    T: torch.Tensor = U.mm(S_inv_root * U.t())
    T_inv: Optional[torch.Tensor] = None
    if return_inverse:
        T_inv = U.mm(torch.sqrt(S) * U.t())
    return T, mean, T_inv


class ZCAWhitening(nn.Module):
    """

    Computes the ZCA whitening matrix transform and the mean vector and applies the transform
    to the data. The data tensor is flattened, and the mean :math:`\\mathbf{\\mu}`
    and covariance matrix :math:`\\mathbf{\\Sigma}` are computed from
    the flattened data :math:`\\mathbf{X} \\in \\mathbb{R}^{N \\times D}`, where
    :math:`N` is the sample size and :math:`D` is flattened dimensionality
    (e.g. for a tensor with size 5x3x2x2 :math:`N = 5` and :math:`D = 12`). The ZCA whitening
    transform is given by:

    .. math::

        \\mathbf{X}_{\\text{zca}} = (\\mathbf{X - \\mu})(US^{-\\frac{1}{2}}U^T)^T

    where :math:`U` are the eigenvectors of :math:`\\Sigma` and :math:`S` contain the correpsonding
    eigenvalues of :math:`\\Sigma`. After the transform is applied, the output is reshaped to same shape.

    args:

        dim (int): Determines the dimension that represents the samples axis. Default = 0
        eps (float) : a small number used for numerial stablility. Default=1e-6
        unbiased (bool): Whether to use the biased estimate of the covariance matrix. Default=False
        compute_inv (bool): Compute the inverse transform matrix. Default=False
        detach_transforms (bool): Detaches gradient from the ZCA fitting. Default=True

    shape:
        - x: :math:`(D_0,...,D_{\\text{dim}},...,D_N)` is a batch of N-D tensors.
        - x_whiten: :math:`(D_0,...,D_{\\text{dim}},...,D_N)` same shape as input.


    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> zca = kornia.color.ZCAWhitening().fit(x)
        >>> x_whiten = zca(x)
        >>> zca = kornia.color.ZCAWhitening()
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step
        >>> x_whiten = zca(x) # Can run now without the fitting set
        >>> # Enable backprop through ZCA fitting process
        >>> zca = kornia.color.ZCAWhitening(detach_transforms = False)
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step

    Note:

        This implementation uses :py:meth:`~torch.svd` which yields NaNs in the backwards step
        if the sigular values are not unique. See `here <https://pytorch.org/docs/stable/torch.html#torch.svd>`_ for
        more information.

    References:

        [1] `Stanford PCA & ZCA whitening tutorial <http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/>`_
    """

    def __init__(self, dim: int=0, eps: float=1e-06, unbiased: bool=True, detach_transforms: bool=True, compute_inv: bool=False) ->None:
        super(ZCAWhitening, self).__init__()
        self.dim = dim
        self.eps = eps
        self.unbiased = unbiased
        self.detach_transforms = detach_transforms
        self.compute_inv = compute_inv
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """

        Fits ZCA whitening matrices to the data.

        args:

            x (torch.Tensor): Input data

        returns:
            ZCAWhiten: returns a fitted ZCAWhiten object instance.
        """
        T, mean, T_inv = zca_mean(x, self.dim, self.unbiased, self.eps, self.compute_inv)
        self.mean_vector: torch.Tensor = mean
        self.transform_matrix: torch.Tensor = T
        if T_inv is None:
            self.transform_inv: Optional[torch.Tensor] = torch.empty([0])
        else:
            self.transform_inv = T_inv
        if self.detach_transforms:
            self.mean_vector = self.mean_vector.detach()
            self.transform_matrix = self.transform_matrix.detach()
            self.transform_inv = self.transform_inv.detach()
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor, include_fit: bool=False) ->torch.Tensor:
        """

        Applies the whitening transform to the data

        args:

            x (torch.Tensor): Input data
            include_fit (bool): Indicates whether to fit the data as part of the forward pass

        returns:

            torch.Tensor : The transformed data

        """
        if include_fit:
            self.fit(x)
        if not self.fitted:
            raise RuntimeError('Needs to be fitted first before running. Please call fit or set include_fit to True.')
        x_whiten = linear_transform(x, self.transform_matrix, self.mean_vector, self.dim)
        return x_whiten

    def inverse_transform(self, x: torch.Tensor) ->torch.Tensor:
        """

        Applies the inverse transform to the whitened data.

        args:
            x (torch.Tensor): Whitened data

        returns:
            torch.Tensor: original data



        """
        if not self.fitted:
            raise RuntimeError('Needs to be fitted first before running. Please call fit or set include_fit to True.')
        if not self.compute_inv:
            raise RuntimeError('Did not compute inverse ZCA. Please set compute_inv to True')
        mean_inv: torch.Tensor = -self.mean_vector.mm(self.transform_matrix)
        y = linear_transform(x, self.transform_inv, mean_inv)
        return y


class ExtractTensorPatches(nn.Module):
    """Module that extract patches from tensors and stack them.

    Applies a 2D convolution over an input tensor to extract patches and stack
    them in the depth axis of the output tensor. The function applies a
    Depthwise Convolution by applying the same kernel for all the input planes.

    In the simplest case, the output value of the operator with input size
    :math:`(B, C, H, W)` is :math:`(B, N, C, H_{out}, W_{out})`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.

    * :attr:`window_size` is the size of the sliding window and controls the
      shape of the output tensor and defines the shape of the output patch.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`padding` controls the amount of implicit zeros-paddings on both
      sizes at each dimension.

    The parameters :attr:`window_size`, :attr:`stride` and :attr:`padding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    Arguments:
        window_size (Union[int, Tuple[int, int]]): the size of the convolving
          kernel and the output patch size.
        stride (Optional[Union[int, Tuple[int, int]]]): stride of the
          convolution. Default is 1.
        padding (Optional[Union[int, Tuple[int, int]]]): Zero-padding added to
          both side of the input. Default is 0.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, C, H_{out}, W_{out})`

    Returns:
        torch.Tensor: the tensor with the extracted patches.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = kornia.contrib.extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3.0000, 4.0000, 5.0000],
                  [6.0000, 7.0000, 8.0000]]]])
    """

    def __init__(self, window_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]]=1, padding: Optional[Union[int, Tuple[int, int]]]=0) ->None:
        super(ExtractTensorPatches, self).__init__()
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)
        self.kernel: torch.Tensor = self.create_kernel(self.window_size)

    @staticmethod
    def create_kernel(window_size: Tuple[int, int]) ->torch.Tensor:
        """Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
        """
        window_range: int = window_size[0] * window_size[1]
        kernel: torch.Tensor = torch.zeros(window_range, window_range)
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        batch_size, channels, height, width = input.shape
        kernel: torch.Tensor = self.kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(input.device)
        output_tmp: torch.Tensor = F.conv2d(input, kernel, stride=self.stride, padding=self.padding, groups=channels)
        output: torch.Tensor = output_tmp.view(batch_size, channels, self.window_size[0], self.window_size[1], -1)
        return output.permute(0, 4, 1, 2, 3)


def _compute_zero_padding(kernel_size: int) ->int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2


def _get_pyramid_gaussian_kernel() ->torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[[1.0, 4.0, 6.0, 4.0, 1.0], [4.0, 16.0, 24.0, 16.0, 4.0], [6.0, 24.0, 36.0, 24.0, 6.0], [4.0, 16.0, 24.0, 16.0, 4.0], [1.0, 4.0, 6.0, 4.0, 1.0]]]) / 256.0


def compute_padding(kernel_size: Tuple[int, int]) ->List[int]:
    """Computes padding tuple."""
    assert len(kernel_size) == 2, kernel_size
    computed = [(k // 2) for k in kernel_size]
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1], computed[1], computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0], computed[0]]


def normalize_kernel2d(input: torch.Tensor) ->torch.Tensor:
    """Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError('input should be at least 2D tensor. Got {}'.format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / norm.unsqueeze(-1).unsqueeze(-1)


def filter2D(input: torch.Tensor, kernel: torch.Tensor, border_type: str='reflect', normalized: bool=False) ->torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not isinstance(kernel, torch.Tensor):
        raise TypeError('Input kernel type is not a torch.Tensor. Got {}'.format(type(kernel)))
    if not isinstance(border_type, str):
        raise TypeError('Input border_type is not string. Got {}'.format(type(kernel)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
    if not len(kernel.shape) == 3:
        raise ValueError('Invalid kernel shape, we expect 1xHxW. Got: {}'.format(kernel.shape))
    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError('Invalid border_type, we expect the following: {0}.Got: {1}'.format(borders_list, border_type))
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape
    kernel_numel: int = height * width
    if kernel_numel > 81:
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


class PyrDown(nn.Module):
    """Blurs a tensor and downsamples it.

    Args:
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str='reflect', align_corners: bool=False) ->None:
        super(PyrDown, self).__init__()
        self.border_type: str = border_type
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        x_blur: torch.Tensor = filter2D(input, self.kernel, self.border_type)
        out: torch.Tensor = F.interpolate(x_blur, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        return out


def pyrdown(input: torch.Tensor, border_type: str='reflect', align_corners: bool=False) ->torch.Tensor:
    """Blurs a tensor and downsamples it.

    See :class:`~kornia.transform.PyrDown` for details.
    """
    return PyrDown(border_type, align_corners)(input)


class MaxBlurPool2d(nn.Module):
    """Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling..
        ceil_mode (bool): should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> input = torch.rand(1, 4, 4, 8)
        >>> pool = kornia.contrib.MaxBlurPool2d(kernel_size=3)
        >>> output = pool(input)  # 1x4x2x4
    """

    def __init__(self, kernel_size: int, ceil_mode: bool=False) ->None:
        super(MaxBlurPool2d, self).__init__()
        self.ceil_mode: bool = ceil_mode
        self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(self.kernel_size)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        x_max: torch.Tensor = F.max_pool2d(input, kernel_size=self.kernel_size, padding=self.padding, stride=1, ceil_mode=self.ceil_mode)
        x_down: torch.Tensor = pyrdown(x_max)
        return x_down


def get_diff_kernel_3x3() ->torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_diff_kernel2d() ->torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d_2nd_order() ->torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])


def get_sobel_kernel_3x3() ->torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel2d() ->torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def _get_sobel_kernel_5x5_2nd_order_xy() ->torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor([[-1.0, -2.0, 0.0, 2.0, 1.0], [-2.0, -4.0, 0.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 0.0, -4.0, -2.0], [1.0, 2.0, 0.0, -2.0, -1.0]])


def get_sobel_kernel_5x5_2nd_order() ->torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor([[-1.0, 0.0, 2.0, 0.0, -1.0], [-4.0, 0.0, 8.0, 0.0, -4.0], [-6.0, 0.0, 12.0, 0.0, -6.0], [-4.0, 0.0, 8.0, 0.0, -4.0], [-1.0, 0.0, 2.0, 0.0, -1.0]])


def get_sobel_kernel2d_2nd_order() ->torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) ->torch.Tensor:
    """Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError('mode should be either sobel                         or diff. Got {}'.format(mode))
    if order not in [1, 2]:
        raise TypeError('order should be either 1 or 2                         Got {}'.format(order))
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError('')
    return kernel


class SpatialGradient(nn.Module):
    """Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self, mode: str='sobel', order: int=1, normalized: bool=True) ->None:
        super(SpatialGradient, self).__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel2d(mode, order)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(order=' + str(self.order) + ', ' + 'normalized=' + str(self.normalized) + ', ' + 'mode=' + self.mode + ')'

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).detach()
        kernel: torch.Tensor = tmp_kernel.unsqueeze(1).unsqueeze(1)
        kernel_flip: torch.Tensor = kernel.flip(-3)
        spatial_pad = [self.kernel.size(1) // 2, self.kernel.size(1) // 2, self.kernel.size(2) // 2, self.kernel.size(2) // 2]
        out_channels: int = 3 if self.order == 2 else 2
        padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, (None)]
        return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / float(2 * sigma ** 2))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool=False) ->torch.Tensor:
    """Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 and not force_even or kernel_size <= 0:
        raise TypeError('kernel_size must be an odd positive integer. Got {}'.format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool=False) ->torch.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\\text{kernel_size}_x, \\text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError('kernel_size must be a tuple of length two. Got {}'.format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError('sigma must be a tuple of length two. Got {}'.format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class PatchAffineShapeEstimator(nn.Module):
    """Module, which estimates the second moment matrix of the patch gradients in order to determine the
    affine shape of the local feature as in :cite:`baumberg2000`.

    Args:
        patch_size: int, default = 19
        eps: float, for safe division, default is 1e-10"""

    def __init__(self, patch_size: int=19, eps: float=1e-10):
        super(PatchAffineShapeEstimator, self).__init__()
        self.patch_size: int = patch_size
        self.gradient: nn.Module = SpatialGradient('sobel', 1)
        self.eps: float = eps
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting: torch.Tensor = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)
        return

    def __repr__(self):
        return self.__class__.__name__ + '(patch_size=' + str(self.patch_size) + ', ' + 'eps=' + str(self.eps) + ')'

    def forward(self, patch: torch.Tensor) ->torch.Tensor:
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            ellipse_shape: 3d tensor, shape [Bx1x5] """
        if not torch.is_tensor(patch):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(patch)))
        if not len(patch.shape) == 4:
            raise ValueError('Invalid input shape, we expect Bx1xHxW. Got: {}'.format(patch.shape))
        B, CH, W, H = patch.size()
        if W != self.patch_size or H != self.patch_size or CH != 1:
            raise TypeError('input shape should be must be [Bx1x{}x{}]. Got {}'.format(self.patch_size, self.patch_size, patch.size()))
        self.weighting = self.weighting.to(patch.dtype)
        grads: torch.Tensor = self.gradient(patch) * self.weighting
        gx: torch.Tensor = grads[:, :, (0)]
        gy: torch.Tensor = grads[:, :, (1)]
        ellipse_shape = torch.cat([gx.pow(2).mean(dim=2).mean(dim=2, keepdim=True), (gx * gy).mean(dim=2).mean(dim=2, keepdim=True), gy.pow(2).mean(dim=2).mean(dim=2, keepdim=True)], dim=2)
        bad_mask = (ellipse_shape < self.eps).float().sum(dim=2, keepdim=True) >= 2
        circular_shape = torch.tensor([1.0, 0.0, 1.0]).to(ellipse_shape.device).view(1, 1, 3)
        ellipse_shape = ellipse_shape * (1.0 - bad_mask) + circular_shape * bad_mask
        ellipse_shape = ellipse_shape / ellipse_shape.max(dim=2, keepdim=True)[0]
        return ellipse_shape


def ellipse_to_laf(ells: torch.Tensor) ->torch.Tensor:
    """
    Converts ellipse regions to LAF format. Ellipse (a, b, c)
    and upright covariance matrix [a11 a12; 0 a22] are connected
    by inverse matrix square root:
    A = invsqrt([a b; b c])
    See also https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_frame2oell.m

    Args:
        ells: (torch.Tensor): tensor of ellipses in Oxford format [x y a b c].

    Returns:
        LAF: (torch.Tensor) tensor of ellipses in LAF format.

    Shape:
        - Input: :math:`(B, N, 5)`
        - Output:  :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 10, 5)  # BxNx5
        >>> output = kornia.ellipse_to_laf(input)  #  BxNx2x3
    """
    n_dims = len(ells.size())
    if n_dims != 3:
        raise TypeError('ellipse shape should be must be [BxNx5]. Got {}'.format(ells.size()))
    B, N, dim = ells.size()
    if dim != 5:
        raise TypeError('ellipse shape should be must be [BxNx5]. Got {}'.format(ells.size()))
    a11 = ells[(...), 2:3].abs().sqrt()
    a12 = torch.zeros_like(a11)
    a22 = ells[(...), 4:5].abs().sqrt()
    a21 = ells[(...), 3:4] / (a11 + a22).clamp(1e-09)
    A = torch.stack([a11, a12, a21, a22], dim=-1).view(B, N, 2, 2).inverse()
    out = torch.cat([A, ells[(...), :2].view(B, N, 2, 1)], dim=3)
    return out


def raise_error_if_laf_is_not_valid(laf: torch.Tensor) ->None:
    """Auxilary function, which verifies that input is a torch.tensor of [BxNx2x3] shape

    Args:
        laf
    """
    laf_message: str = 'Invalid laf shape, we expect BxNx2x3. Got: {}'.format(laf.shape)
    if not torch.is_tensor(laf):
        raise TypeError('Laf type is not a torch.Tensor. Got {}'.format(type(laf)))
    if len(laf.shape) != 4:
        raise ValueError(laf_message)
    if laf.size(2) != 2 or laf.size(3) != 3:
        raise ValueError(laf_message)
    return


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) ->torch.Tensor:
    """De-normalizes LAFs from scale to image scale.
        >>> B,N,H,W = images.size()
        >>> MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a21*MIN_SIZE x*W]
        [a21*MIN_SIZE a22*MIN_SIZE y*H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    wf = float(w)
    hf = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) * min_size
    coef[0, 0, 0, 2] = wf
    coef[0, 0, 1, 2] = hf
    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img: torch.Tensor, LAF: torch.Tensor, PS: int=32) ->torch.Tensor:
    """Helper function for affine grid generation.

    Args:
        img: (torch.Tensor) images, LAFs are detected in
        LAF: (torch.Tensor).
        PS (int) -- patch size to be extracted

    Returns:
        grid: (torch.Tensor).

    Shape:
        - Input: :math:`(B, CH, H, W)`,  :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, PS, PS)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()
    LAF_renorm = denormalize_laf(LAF, img)
    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3), [B * N, ch, PS, PS], align_corners=False)
    grid[(...), :, (0)] = 2.0 * grid[(...), :, (0)].clone() / float(w) - 1.0
    grid[(...), :, (1)] = 2.0 * grid[(...), :, (1)].clone() / float(h) - 1.0
    return grid


def get_laf_scale(LAF: torch.Tensor) ->torch.Tensor:
    """Returns a scale of the LAFs

    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].

    Returns:
        torch.Tensor: tensor  BxNx1x1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.get_laf_scale(input)  # BxNx1x1
    """
    raise_error_if_laf_is_not_valid(LAF)
    eps = 1e-10
    out = LAF[(...), 0:1, 0:1] * LAF[(...), 1:2, 1:2] - LAF[(...), 1:2, 0:1] * LAF[(...), 0:1, 1:2] + eps
    return out.abs().sqrt()


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor) ->torch.Tensor:
    """Normalizes LAFs to [0,1] scale from pixel scale. See below:
        >>> B,N,H,W = images.size()
        >>> MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes:
        [a11/MIN_SIZE a21/MIN_SIZE x/W]
        [a21/MIN_SIZE a22/MIN_SIZE y/H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    wf: float = float(w)
    hf: float = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) / min_size
    coef[0, 0, 0, 2] = 1.0 / wf
    coef[0, 0, 1, 2] = 1.0 / hf
    return coef.expand_as(LAF) * LAF


def extract_patches_from_pyramid(img: torch.Tensor, laf: torch.Tensor, PS: int=32, normalize_lafs_before_extraction: bool=True) ->torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level

    Args:
        laf: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32
        normalize_lafs_before_extraction (bool):  if True, lafs are normalized to image size, default = True

    Returns:
        patches: (torch.Tensor)  :math:`(B, N, CH, PS,PS)`
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    num, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    pyr_idx = (scale.log2() + 0.5).relu().long()
    cur_img = img
    cur_pyr_level = int(0)
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        num, ch, h, w = cur_img.size()
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).bool().squeeze()
            if scale_mask.float().sum() == 0:
                continue
            scale_mask = scale_mask.bool().view(-1)
            grid = generate_patch_grid_from_normalized_LAF(cur_img[i:i + 1], nlaf[i:i + 1, (scale_mask), :, :], PS)
            patches = F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w), grid, padding_mode='border', align_corners=False)
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.pyrdown(cur_img)
        cur_pyr_level += 1
    return out


def scale_laf(laf: torch.Tensor, scale_coef: Union[float, torch.Tensor]) ->torch.Tensor:
    """
    Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.
    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        laf: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].
        scale_coef: (torch.Tensor): broadcastable tensor or float.


    Returns:
        torch.Tensor: tensor  BxNx2x3 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Input: :math: `(B, N,)` or ()
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = kornia.scale_laf(input, scale)  # BxNx2x3
    """
    if type(scale_coef) is not float and type(scale_coef) is not torch.Tensor:
        raise TypeError('scale_coef should be float or torch.Tensor Got {}'.format(type(scale_coef)))
    raise_error_if_laf_is_not_valid(laf)
    centerless_laf: torch.Tensor = laf[:, :, :2, :2]
    return torch.cat([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


def make_upright(laf: torch.Tensor, eps: float=1e-09) ->torch.Tensor:
    """
    Rectifies the affine matrix, so that it becomes upright

    Args:
        laf: (torch.Tensor): tensor of LAFs.
        eps (float): for safe division, (default 1e-9)

    Returns:
        torch.Tensor: tensor of same shape.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.make_upright(input)  #  BxNx2x3
    """
    raise_error_if_laf_is_not_valid(laf)
    det = get_laf_scale(laf)
    scale = det
    b2a2 = torch.sqrt(laf[(...), 0:1, 1:2] ** 2 + laf[(...), 0:1, 0:1] ** 2) + eps
    laf1_ell = torch.cat([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim=3)
    laf2_ell = torch.cat([(laf[(...), 1:2, 1:2] * laf[(...), 0:1, 1:2] + laf[(...), 1:2, 0:1] * laf[(...), 0:1, 0:1]) / (b2a2 * det), (det / b2a2).contiguous()], dim=3)
    laf_unit_scale = torch.cat([torch.cat([laf1_ell, laf2_ell], dim=2), laf[(...), :, 2:3]], dim=3)
    return scale_laf(laf_unit_scale, scale)


class LAFAffineShapeEstimator(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs :class:`~kornia.feature.PatchAffineShapeEstimator` on patches to estimate LAFs shape.
    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter.

    Args:
            patch_size: int, default = 32"""

    def __init__(self, patch_size: int=32) ->None:
        super(LAFAffineShapeEstimator, self).__init__()
        self.patch_size = patch_size
        self.affine_shape_detector = PatchAffineShapeEstimator(self.patch_size)
        return

    def __repr__(self):
        return self.__class__.__name__ + '(patch_size=' + str(self.patch_size) + ')'

    def forward(self, laf: torch.Tensor, img: torch.Tensor) ->torch.Tensor:
        """
        Args:
            laf: (torch.Tensor) shape [BxNx2x3]
            img: (torch.Tensor) shape [Bx1xHxW]

        Returns:
            laf_out: (torch.Tensor) shape [BxNx2x3]"""
        raise_error_if_laf_is_not_valid(laf)
        img_message: str = 'Invalid img shape, we expect BxCxHxW. Got: {}'.format(img.shape)
        if not torch.is_tensor(img):
            raise TypeError('img type is not a torch.Tensor. Got {}'.format(type(img)))
        if len(img.shape) != 4:
            raise ValueError(img_message)
        if laf.size(0) != img.size(0):
            raise ValueError('Batch size of laf and img should be the same. Got {}, {}'.format(img.size(0), laf.size(0)))
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img, make_upright(laf), PS, True).view(-1, 1, PS, PS)
        ellipse_shape: torch.Tensor = self.affine_shape_detector(patches)
        ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), ellipse_shape], dim=2).view(B, N, 5)
        scale_orig = get_laf_scale(laf)
        laf_out = ellipse_to_laf(ellipses)
        ellipse_scale = get_laf_scale(laf_out)
        laf_out = scale_laf(laf_out, scale_orig / ellipse_scale)
        return laf_out


class HardNet(nn.Module):
    """
    Module, which computes HardNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Working hard to know your neighbor's
    margins: Local descriptor learning loss". See :cite:`HardNet2017` for more details.

    Args:
        pretrained: (bool) Download and set pretrained weights to the model. Default: false.

    Returns:
        torch.Tensor: HardeNet descriptor of the patches.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = kornia.feature.HardNet()
        >>> descs = hardnet(input) # 16x128
    """

    def __init__(self, pretrained: bool=False) ->None:
        super(HardNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.Dropout(0.3), nn.Conv2d(128, 128, kernel_size=8, bias=False), nn.BatchNorm2d(128, affine=False))
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls['liberty_aug'], map_location=lambda storage, loc: storage)
            self.load_state_dict(pretrained_dict['state_dict'], strict=True)

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float=1e-06) ->torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


def _get_nms_kernel2d(kx: int, ky: int) ->torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)


class NonMaximaSuppression2d(nn.Module):
    """Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = _get_nms_kernel2d(*kernel_size)

    @staticmethod
    def _compute_zero_padding2d(kernel_size: Tuple[int, int]) ->Tuple[int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2
        ky, kx = kernel_size
        return pad(ky), pad(kx)

    def forward(self, x: torch.Tensor, mask_only: bool=False) ->torch.Tensor:
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.size()
        max_non_center = F.conv2d(x, self.kernel.repeat(CH, 1, 1, 1), stride=1, padding=self.padding, groups=CH).view(B, CH, -1, H, W).max(dim=2)[0]
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * mask


def _get_nms_kernel3d(kd: int, ky: int, kx: int) ->torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = kd * ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, kd, ky, kx)


class NonMaximaSuppression3d(nn.Module):
    """Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]):
        super(NonMaximaSuppression3d, self).__init__()
        self.kernel_size: Tuple[int, int, int] = kernel_size
        self.padding: Tuple[int, int, int] = self._compute_zero_padding3d(kernel_size)
        self.kernel = _get_nms_kernel3d(*kernel_size)

    @staticmethod
    def _compute_zero_padding3d(kernel_size: Tuple[int, int, int]) ->Tuple[int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 3, kernel_size

        def pad(x):
            return (x - 1) // 2
        kd, ky, kx = kernel_size
        return pad(kd), pad(ky), pad(kx)

    def forward(self, x: torch.Tensor, mask_only: bool=False) ->torch.Tensor:
        assert len(x.shape) == 5, x.shape
        B, CH, D, H, W = x.size()
        max_non_center = F.conv3d(x, self.kernel.repeat(CH, 1, 1, 1, 1), stride=1, padding=self.padding, groups=CH).view(B, CH, -1, D, H, W).max(dim=2, keepdim=False)[0]
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * mask


class PassLAF(nn.Module):
    """Dummy module to use instead of local feature orientation or affine shape estimator"""

    def forward(self, laf: torch.Tensor, img: torch.Tensor) ->torch.Tensor:
        """
        Args:
            laf: torch.Tensor: 4d tensor
            img (torch.Tensor): the input image tensor

        Return:
            torch.Tensor: unchanged laf from the input."""
        return laf


class PatchDominantGradientOrientation(nn.Module):
    """Module, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36
            eps: float, for safe division, and arctan, default is 1e-8"""

    def __init__(self, patch_size: int=32, num_angular_bins: int=36, eps: float=1e-08):
        super(PatchDominantGradientOrientation, self).__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.gradient = SpatialGradient('sobel', 1)
        self.eps = eps
        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode='circular')
        with torch.no_grad():
            self.angular_smooth.weight[:] = torch.tensor([[[0.33, 0.34, 0.33]]])
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)
        return

    def __repr__(self):
        return self.__class__.__name__ + '(patch_size=' + str(self.patch_size) + ', ' + 'num_ang_bins=' + str(self.num_ang_bins) + ', ' + 'eps=' + str(self.eps) + ')'

    def forward(self, patch: torch.Tensor) ->torch.Tensor:
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            patch: (torch.Tensor) shape [Bx1] """
        if not torch.is_tensor(patch):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(patch)))
        if not len(patch.shape) == 4:
            raise ValueError('Invalid input shape, we expect Bx1xHxW. Got: {}'.format(patch.shape))
        B, CH, W, H = patch.size()
        if W != self.patch_size or H != self.patch_size or CH != 1:
            raise TypeError('input shape should be must be [Bx1x{}x{}]. Got {}'.format(self.patch_size, self.patch_size, patch.size()))
        self.weighting = self.weighting.to(patch.dtype)
        self.angular_smooth = self.angular_smooth.to(patch.dtype)
        grads: torch.Tensor = self.gradient(patch)
        gx: torch.Tensor = grads[:, :, (0)]
        gy: torch.Tensor = grads[:, :, (1)]
        mag: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: torch.Tensor = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        o_big = float(self.num_ang_bins) * (ori + 1.0 * pi) / (2.0 * pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i) * wo0_big + (bo1_big == i) * wo1_big, (1, 1)))
        ang_bins = torch.cat(ang_bins, 1).view(-1, 1, self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)
        angle = -(2.0 * pi * indices / float(self.num_ang_bins) - pi)
        return angle


def deg2rad(tensor: torch.Tensor) ->torch.Tensor:
    """Function that converts angles from degrees to radians.

    Args:
        tensor (torch.Tensor): Tensor of arbitrary shape.

    Returns:
        torch.Tensor: tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = kornia.deg2rad(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return tensor * pi.type(tensor.dtype) / 180.0


def angle_to_rotation_matrix(angle: torch.Tensor) ->torch.Tensor:
    """Create a rotation matrix out of angles in degrees.
    Args:
        angle: (torch.Tensor): tensor of angles in degrees, any shape.

    Returns:
        torch.Tensor: tensor of *x2x2 rotation matrices.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def rad2deg(tensor: torch.Tensor) ->torch.Tensor:
    """Function that converts angles from radians to degrees.

    Args:
        tensor (torch.Tensor): Tensor of arbitrary shape.

    Returns:
        torch.Tensor: Tensor with same shape as input.

    Example:
        >>> input = kornia.pi * torch.rand(1, 3, 3)
        >>> output = kornia.rad2deg(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return 180.0 * tensor / pi.type(tensor.dtype)


class LAFOrienter(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs :class:`~kornia.feature.PatchDominantGradientOrientation`
    on patches and then rotates the LAFs by the estimated angles

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36"""

    def __init__(self, patch_size: int=32, num_angular_bins: int=36):
        super(LAFOrienter, self).__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.angle_detector = PatchDominantGradientOrientation(self.patch_size, self.num_ang_bins)
        return

    def __repr__(self):
        return self.__class__.__name__ + '(patch_size=' + str(self.patch_size) + ', ' + 'num_ang_bins=' + str(self.num_ang_bins) + ')'

    def forward(self, laf: torch.Tensor, img: torch.Tensor) ->torch.Tensor:
        """
        Args:
            laf: (torch.Tensor), shape [BxNx2x3]
            img: (torch.Tensor), shape [Bx1xHxW]

        Returns:
            laf_out: (torch.Tensor), shape [BxNx2x3] """
        raise_error_if_laf_is_not_valid(laf)
        img_message: str = 'Invalid img shape, we expect BxCxHxW. Got: {}'.format(img.shape)
        if not torch.is_tensor(img):
            raise TypeError('img type is not a torch.Tensor. Got {}'.format(type(img)))
        if len(img.shape) != 4:
            raise ValueError(img_message)
        if laf.size(0) != img.size(0):
            raise ValueError('Batch size of laf and img should be the same. Got {}, {}'.format(img.size(0), laf.size(0)))
        B, N = laf.shape[:2]
        patches: torch.Tensor = extract_patches_from_pyramid(img, laf, self.patch_size).view(-1, 1, self.patch_size, self.patch_size)
        angles_radians: torch.Tensor = self.angle_detector(patches).view(B, N)
        rotmat: torch.Tensor = angle_to_rotation_matrix(rad2deg(angles_radians)).view(B * N, 2, 2)
        laf_out: torch.Tensor = torch.cat([torch.bmm(make_upright(laf).view(B * N, 2, 3)[:, :2, :2], rotmat), laf.view(B * N, 2, 3)[:, :2, 2:]], dim=2).view(B, N, 2, 3)
        return laf_out


class GaussianBlur2d(nn.Module):
    """Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str='reflect') ->None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.kernel: torch.Tensor = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), dim=0)
        assert border_type in ['constant', 'reflect', 'replicate', 'circular']
        self.border_type = border_type

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(kernel_size=' + str(self.kernel_size) + ', ' + 'sigma=' + str(self.sigma) + ', ' + 'border_type=' + self.border_type + ')'

    def forward(self, x: torch.Tensor):
        return kornia.filter2D(x, self.kernel, self.border_type)


def gaussian_blur2d(input: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str='reflect') ->torch.Tensor:
    """Function that blurs a tensor using a Gaussian filter.

    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur2d(kernel_size, sigma, border_type)(input)


def spatial_gradient(input: torch.Tensor, mode: str='sobel', order: int=1, normalized: bool=True) ->torch.Tensor:
    """Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return SpatialGradient(mode, order, normalized)(input)


def harris_response(input: torch.Tensor, k: Union[torch.Tensor, float]=0.04, grads_mode: str='sobel', sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Computes the Harris cornerness function. Function does not do
    any normalization or nms.The response map is computed according the following formulation:

    .. math::
        R = max(0, det(M) - k \\cdot trace(M)^2)

    where:

    .. math::
        M = \\sum_{(x,y) \\in W}
        \\begin{bmatrix}
            I^{2}_x & I_x I_y \\\\
            I_x I_y & I^{2}_y \\\\
        \\end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        input: torch.Tensor: 4d tensor
        k (torch.Tensor): the Harris detector free parameter.
        grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid
        sigmas (optional, torch.Tensor): coefficients to be multiplied by multichannel response. \\n
                                         Should be shape of (B)
                                         It is necessary for performing non-maxima-suppression
                                         across different scale pyramid levels.\\
                                         See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_

    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = harris_response(input, 0.04)
        tensor([[[[0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012],
          [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
          [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
          [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
          [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
    if sigmas is not None:
        if not torch.is_tensor(sigmas):
            raise TypeError('sigmas type is not a torch.Tensor. Got {}'.format(type(sigmas)))
        if not len(sigmas.shape) == 1 or sigmas.size(0) != input.size(0):
            raise ValueError('Invalid sigmas shape, we expect B == input.size(0). Got: {}'.format(sigmas.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, (0)]
    dy: torch.Tensor = gradients[:, :, (1)]

    def g(x):
        return gaussian_blur2d(x, (7, 7), (1.0, 1.0))
    dx2: torch.Tensor = g(dx ** 2)
    dy2: torch.Tensor = g(dy ** 2)
    dxy: torch.Tensor = g(dx * dy)
    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2
    scores: torch.Tensor = det_m - k * trace_m ** 2
    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)
    return scores


class CornerHarris(nn.Module):
    """nn.Module that calculates Harris corners
    See :func:`~kornia.feature.harris_response` for details.
    """

    def __init__(self, k: Union[float, torch.Tensor], grads_mode='sobel') ->None:
        super(CornerHarris, self).__init__()
        if type(k) is float:
            self.register_buffer('k', torch.tensor(k))
        else:
            self.register_buffer('k', k)
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(k=' + str(self.k) + ', ' + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
        return harris_response(input, self.k, self.grads_mode, sigmas)


def gftt_response(input: torch.Tensor, grads_mode: str='sobel', sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Computes the Shi-Tomasi cornerness function. Function does not do any normalization or nms.
    The response map is computed according the following formulation:

    .. math::
        R = min(eig(M))

    where:

    .. math::
        M = \\sum_{(x,y) \\in W}
        \\begin{bmatrix}
            I^{2}_x & I_x I_y \\\\
            I_x I_y & I^{2}_y \\\\
        \\end{bmatrix}

    Args:
        input (torch.Tensor): 4d tensor
        grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid
        sigmas (optional, torch.Tensor): coefficients to be multiplied by multichannel response. \\n
                                         Should be shape of (B)
                                         It is necessary for performing non-maxima-suppression
                                         across different scale pyramid levels.\\
                                         See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_

    Return:
        torch.Tensor: the response map per channel.

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]])  # 1x1x7x7
        >>> # compute the response map
        >>> output = gftt_response(input)
        tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
          [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
          [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
          [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
          [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode)
    dx: torch.Tensor = gradients[:, :, (0)]
    dy: torch.Tensor = gradients[:, :, (1)]

    def g(x):
        return gaussian_blur2d(x, (7, 7), (1.0, 1.0))
    dx2: torch.Tensor = g(dx ** 2)
    dy2: torch.Tensor = g(dy ** 2)
    dxy: torch.Tensor = g(dx * dy)
    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2
    e1: torch.Tensor = 0.5 * (trace_m + torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))
    e2: torch.Tensor = 0.5 * (trace_m - torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))
    scores: torch.Tensor = torch.min(e1, e2)
    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)
    return scores


class CornerGFTT(nn.Module):
    """nn.Module that calculates Shi-Tomasi corners
    See :func:`~kornia.feature.gfft_response` for details.
    """

    def __init__(self, grads_mode='sobel') ->None:
        super(CornerGFTT, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
        return gftt_response(input, self.grads_mode, sigmas)


def hessian_response(input: torch.Tensor, grads_mode: str='sobel', sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Computes the absolute of determinant of the Hessian matrix. Function does not do any normalization or nms.
    The response map is computed according the following formulation:

    .. math::
        R = det(H)

    where:

    .. math::
        M = \\sum_{(x,y) \\in W}
        \\begin{bmatrix}
            I_{xx} & I_{xy} \\\\
            I_{xy} & I_{yy} \\\\
        \\end{bmatrix}

    Args:
        input: torch.Tensor: 4d tensor
        grads_mode (string): can be 'sobel' for standalone use or 'diff' for use on Gaussian pyramid
        sigmas (optional, torch.Tensor): coefficients to be multiplied by multichannel response. \\n
                                         Should be shape of (B)
                                         It is necessary for performing non-maxima-suppression
                                         across different scale pyramid levels.\\
                                         See `vlfeat <https://github.com/vlfeat/vlfeat/blob/master/vl/covdet.c#L874>`_

    Return:
         torch.Tensor: the response map per channel.

    Shape:
       - Input: :math:`(B, C, H, W)`
       - Output: :math:`(B, C, H, W)`

    Examples:
         >>> input = torch.tensor([[[
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 1., 1., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
         ]]])  # 1x1x7x7
         >>> # compute the response map
         >>> output = hessian_response(input)
         tensor([[[[0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155],
           [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
           [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.0194, 0.0339, 0.0497, 0.0000, 0.0497, 0.0339, 0.0194],
           [0.0334, 0.0575, 0.0339, 0.0000, 0.0339, 0.0575, 0.0334],
           [0.0155, 0.0334, 0.0194, 0.0000, 0.0194, 0.0334, 0.0155]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
    if sigmas is not None:
        if not torch.is_tensor(sigmas):
            raise TypeError('sigmas type is not a torch.Tensor. Got {}'.format(type(sigmas)))
        if not len(sigmas.shape) == 1 or sigmas.size(0) != input.size(0):
            raise ValueError('Invalid sigmas shape, we expect B == input.size(0). Got: {}'.format(sigmas.shape))
    gradients: torch.Tensor = spatial_gradient(input, grads_mode, 2)
    dxx: torch.Tensor = gradients[:, :, (0)]
    dxy: torch.Tensor = gradients[:, :, (1)]
    dyy: torch.Tensor = gradients[:, :, (2)]
    scores: torch.Tensor = dxx * dyy - dxy ** 2
    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)
    return scores


class BlobHessian(nn.Module):
    """nn.Module that calculates Hessian blobs
    See :func:`~kornia.feature.hessian_response` for details.
    """

    def __init__(self, grads_mode='sobel') ->None:
        super(BlobHessian, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor, sigmas: Optional[torch.Tensor]=None) ->torch.Tensor:
        return hessian_response(input, self.grads_mode, sigmas)


def _get_center_kernel3d(d: int, h: int, w: int, device: torch.device=torch.device('cpu')) ->torch.Tensor:
    """Helper function, which generates a kernel to return center coordinates,
       when applied with F.conv2d to 3d coordinates grid.

    Args:
         d (int): kernel depth.
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [3x3xdxhxw]
    """
    center_kernel = torch.zeros(3, 3, d, h, w, device=device)
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = h // 2 + 1
    else:
        h_i1 = h // 2 - 1
        h_i2 = h // 2 + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = w // 2 + 1
    else:
        w_i1 = w // 2 - 1
        w_i2 = w // 2 + 1
    if d % 2 != 0:
        d_i1 = d // 2
        d_i2 = d // 2 + 1
    else:
        d_i1 = d // 2 - 1
        d_i2 = d // 2 + 1
    center_num = float((h_i2 - h_i1) * (w_i2 - w_i1) * (d_i2 - d_i1))
    center_kernel[(0, 1, 2), (0, 1, 2), d_i1:d_i2, h_i1:h_i2, w_i1:w_i2] = 1.0 / center_num
    return center_kernel


def create_meshgrid(height: int, width: int, normalized_coordinates: bool=True, device: Optional[torch.device]=torch.device('cpu')) ->torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)


def _get_window_grid_kernel3d(d: int, h: int, w: int, device: torch.device=torch.device('cpu')) ->torch.Tensor:
    """Helper function, which generates a kernel to return coordinates,
       residual to window center.

    Args:
         d (int): kernel depth.
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [3x1xdxhxw]
    """
    grid2d = create_meshgrid(h, w, True, device=device)
    if d > 1:
        z = torch.linspace(-1, 1, d, device=device).view(d, 1, 1, 1)
    else:
        z = torch.zeros(1, 1, 1, 1, device=device)
    grid3d = torch.cat([z.repeat(1, h, w, 1).contiguous(), grid2d.repeat(d, 1, 1, 1)], dim=3)
    conv_kernel = grid3d.permute(3, 0, 1, 2).unsqueeze(1)
    return conv_kernel


def create_meshgrid3d(depth: int, height: int, width: int, normalized_coordinates: bool=True, device: Optional[torch.device]=torch.device('cpu')) ->torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        depth (int): the image depth (channels).
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    zs: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
        zs = torch.linspace(-1, 1, depth, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
        zs = torch.linspace(0, depth - 1, depth, device=device, dtype=torch.float)
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([zs, xs, ys])).transpose(1, 2)
    return base_grid.unsqueeze(0).permute(0, 3, 4, 2, 1)


def normalize_pixel_coordinates3d(pixel_coordinates: torch.Tensor, depth: int, height: int, width: int, eps: float=1e-08) ->torch.Tensor:
    """Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates (torch.Tensor): the grid with pixel coordinates.
          Shape can be :math:`(*, 3)`.
        depth (int): the maximum depth in the z-axis.
        height (int): the maximum height in the y-axis.
        width (int): the maximum width in the x-axis.
        eps (float): safe division by zero. (default 1e-8).

    Return:
        torch.Tensor: the normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError('Input pixel_coordinates must be of shape (*, 3). Got {}'.format(pixel_coordinates.shape))
    dhw: torch.Tensor = torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)]).to(pixel_coordinates.device)
    factor: torch.Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)
    return factor * pixel_coordinates - 1


def conv_soft_argmax3d(input: torch.Tensor, kernel_size: Tuple[int, int, int]=(3, 3, 3), stride: Tuple[int, int, int]=(1, 1, 1), padding: Tuple[int, int, int]=(1, 1, 1), temperature: Union[torch.Tensor, float]=torch.tensor(1.0), normalized_coordinates: bool=False, eps: float=1e-08, output_value: bool=True, strict_maxima_bonus: float=0.0) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Function that computes the convolutional spatial Soft-Argmax 3D over the windows
    of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
    themselves.
    On each window, the function computed is:

    .. math::
             ijk(X) = \\frac{\\sum{(i,j,k)} * exp(x / T)  \\in X} {\\sum{exp(x / T)  \\in X}}

    .. math::
             val(X) = \\frac{\\sum{x * exp(x / T)  \\in X}} {\\sum{exp(x / T)  \\in X}}

    where T is temperature.

    Args:
        kernel_size (Tuple[int,int,int]):  size of the window
        stride (Tuple[int,int,int]): stride of the window.
        padding (Tuple[int,int,int]): input zero padding
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the coordinates normalized in the range of [-1, 1]. Otherwise,
                                       it will return the coordinates in the range of the input shape. Default is False.
        eps (float): small value to avoid zero division. Default is 1e-8.
        output_value (bool): if True, val is outputed, if False, only ij
        strict_maxima_bonus (float): pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
                                     This is needed for mimic behavior of strict NMS in classic local features
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

         .. math::
             D_{out} = \\left\\lfloor\\frac{D_{in}  + 2 \\times \\text{padding}[0] -
             (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

         .. math::
             H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[1] -
             (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

         .. math::
             W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[2] -
             (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3, 3, 3), (1, 2, 2), (0, 1, 1))
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 5:
        raise ValueError('Invalid input shape, we expect BxCxDxHxW. Got: {}'.format(input.shape))
    if temperature <= 0:
        raise ValueError('Temperature should be positive float or tensor. Got: {}'.format(temperature))
    b, c, d, h, w = input.shape
    kx, ky, kz = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, d, h, w)
    center_kernel: torch.Tensor = _get_center_kernel3d(kx, ky, kz, device)
    window_kernel: torch.Tensor = _get_window_grid_kernel3d(kx, ky, kz, device)
    x_max = F.adaptive_max_pool3d(input, (1, 1, 1))
    x_exp = ((input - x_max.detach()) / temperature).exp()
    pool_coef: float = float(kx * ky * kz)
    den = pool_coef * F.avg_pool3d(x_exp.view_as(input), kernel_size, stride=stride, padding=padding) + eps
    grid_global: torch.Tensor = create_meshgrid3d(d, h, w, False, device=device).permute(0, 4, 1, 2, 3)
    grid_global_pooled = F.conv3d(grid_global, center_kernel, stride=stride, padding=padding)
    coords_max: torch.Tensor = F.conv3d(x_exp, window_kernel, stride=stride, padding=padding)
    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates3d(coords_max.permute(0, 2, 3, 4, 1), d, h, w)
        coords_max = coords_max.permute(0, 4, 1, 2, 3)
    coords_max = coords_max.view(b, c, 3, coords_max.size(2), coords_max.size(3), coords_max.size(4))
    if not output_value:
        return coords_max
    x_softmaxpool = pool_coef * F.avg_pool3d(x_exp.view(input.size()) * input, kernel_size, stride=stride, padding=padding) / den
    if strict_maxima_bonus > 0:
        in_levels: int = input.size(2)
        out_levels: int = x_softmaxpool.size(2)
        skip_levels: int = (in_levels - out_levels) // 2
        strict_maxima: torch.Tensor = F.avg_pool3d(kornia.feature.nms3d(input, kernel_size), 1, stride, 0)
        strict_maxima = strict_maxima[:, :, skip_levels:out_levels - skip_levels]
        x_softmaxpool *= 1.0 + strict_maxima_bonus * strict_maxima
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3), x_softmaxpool.size(4))
    return coords_max, x_softmaxpool


class ConvSoftArgmax3d(nn.Module):
    """Module that calculates soft argmax 3d per window.

    See :func:`~kornia.geometry.conv_soft_argmax3d` for details.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]=(3, 3, 3), stride: Tuple[int, int, int]=(1, 1, 1), padding: Tuple[int, int, int]=(1, 1, 1), temperature: Union[torch.Tensor, float]=torch.tensor(1.0), normalized_coordinates: bool=False, eps: float=1e-08, output_value: bool=True, strict_maxima_bonus: float=0.0) ->None:
        super(ConvSoftArgmax3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(' + 'kernel_size=' + str(self.kernel_size) + ', ' + 'stride=' + str(self.stride) + ', ' + 'padding=' + str(self.padding) + ', ' + 'temperature=' + str(self.temperature) + ', ' + 'normalized_coordinates=' + str(self.normalized_coordinates) + ', ' + 'eps=' + str(self.eps) + ', ' + 'strict_maxima_bonus=' + str(self.strict_maxima_bonus) + ', ' + 'output_value=' + str(self.output_value) + ')'

    def forward(self, x: torch.Tensor):
        return conv_soft_argmax3d(x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates, self.eps, self.output_value, self.strict_maxima_bonus)


class ScalePyramid(nn.Module):
    """Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur and
    downscaled.
    Arguments:
        n_levels (int): number of the levels in octave.
        init_sigma (float): initial blur level.
        min_size (int): the minimum size of the octave in pixels. Default is 5
        double_image (bool): add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this. Default is False
    Returns:
        Tuple(List(Tensors), List(Tensors), List(Tensors)):
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, NL, C, H, W), (B, NL, C, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples::
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = kornia.ScalePyramid(3, 15)(input)
    """

    def __init__(self, n_levels: int=3, init_sigma: float=1.6, min_size: int=5, double_image: bool=False):
        super(ScalePyramid, self).__init__()
        self.n_levels = n_levels
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(n_levels=' + str(self.n_levels) + ', ' + 'init_sigma=' + str(self.init_sigma) + ', ' + 'min_size=' + str(self.min_size) + ', ' + 'border=' + str(self.border) + ', ' + 'sigma_step=' + str(self.sigma_step) + 'double_image=' + str(self.double_image) + ')'

    def get_kernel_size(self, sigma: float):
        ksize = int(2.0 * 4.0 * sigma + 1.0)
        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def forward(self, x: torch.Tensor) ->Tuple[List, List, List]:
        bs, ch, h, w = x.size()
        pixel_distance = 1.0
        cur_sigma = 0.5
        if self.double_image:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        if self.init_sigma > cur_sigma:
            sigma = math.sqrt(self.init_sigma ** 2 - cur_sigma ** 2)
            cur_sigma = self.init_sigma
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
        else:
            cur_level = x
        sigmas = [cur_sigma * torch.ones(bs, self.n_levels).to(x.device)]
        pixel_dists = [pixel_distance * torch.ones(bs, self.n_levels).to(x.device)]
        pyr = [[cur_level.unsqueeze(1)]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0].squeeze(1)
            for level_idx in range(1, self.n_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step ** 2 - 1.0)
                cur_level = gaussian_blur2d(cur_level, (ksize, ksize), (sigma, sigma))
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level.unsqueeze(1))
                sigmas[-1][:, (level_idx)] = cur_sigma
                pixel_dists[-1][:, (level_idx)] = pixel_distance
            nextOctaveFirstLevel = F.interpolate(pyr[-1][-1].squeeze(1), scale_factor=0.5, mode='bilinear', align_corners=False)
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if min(nextOctaveFirstLevel.size(2), nextOctaveFirstLevel.size(3)) <= self.min_size:
                break
            pyr.append([nextOctaveFirstLevel.unsqueeze(1)])
            sigmas.append(cur_sigma * torch.ones(bs, self.n_levels))
            pixel_dists.append(pixel_distance * torch.ones(bs, self.n_levels))
            oct_idx += 1
        for i in range(len(pyr)):
            pyr[i] = torch.cat(pyr[i], dim=1)
        return pyr, sigmas, pixel_dists


def _create_octave_mask(mask: torch.Tensor, octave_shape: List[int]) ->torch.Tensor:
    """Downsamples a mask based on the given octave shape."""
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode='bilinear', align_corners=False)
    return mask_octave.unsqueeze(1)


def _scale_index_to_scale(max_coords: torch.Tensor, sigmas: torch.Tensor) ->torch.Tensor:
    """Auxilary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d
    to the actual scale, using the sigmas from the ScalePyramid output
    Args:
        max_coords: (torch.Tensor): tensor [BxNx3].
        sigmas: (torch.Tensor): tensor [BxNxD], D >= 1

    Returns:
        torch.Tensor:  tensor [BxNx3].
    """
    B, N, _ = max_coords.shape
    L: int = sigmas.size(1)
    scale_coords = max_coords[:, :, (0)].contiguous().view(-1, 1, 1, 1)
    scale_coords_index = 2.0 * scale_coords / sigmas.size(1) - 1.0
    dummy_x = torch.zeros_like(scale_coords_index)
    scale_grid = torch.cat([scale_coords_index, dummy_x], dim=3)
    scale_val = F.grid_sample(sigmas[0].log2().view(1, 1, 1, -1).expand(scale_grid.size(0), 1, 1, L), scale_grid, align_corners=False)
    out = torch.cat([torch.pow(2.0, scale_val).view(B, N, 1), max_coords[:, :, 1:]], dim=2)
    return out


def laf_to_boundary_points(LAF: torch.Tensor, n_pts: int=50) ->torch.Tensor:
    """
    Converts LAFs to boundary points of the regions + center.
    Used for local features visualization, see visualize_laf function

    Args:
        LAF: (torch.Tensor).
        n_pts: number of points to output

    Returns:
        pts: (torch.Tensor) tensor of boundary points

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    B, N, _, _ = LAF.size()
    pts = torch.cat([torch.sin(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1), torch.cos(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1), torch.ones(n_pts - 1, 1)], dim=1)
    pts = torch.cat([torch.tensor([0, 0, 1.0]).view(1, 3), pts], dim=0).unsqueeze(0).expand(B * N, n_pts, 3)
    pts = pts.to(LAF.device)
    aux = torch.tensor([0, 0, 1.0]).view(1, 1, 3).expand(B * N, 1, 3)
    HLAF = torch.cat([LAF.view(-1, 2, 3), aux.to(LAF.device)], dim=1)
    pts_h = torch.bmm(HLAF, pts.permute(0, 2, 1)).permute(0, 2, 1)
    return kornia.convert_points_from_homogeneous(pts_h.view(B, N, n_pts, 3))


def laf_is_inside_image(laf: torch.Tensor, images: torch.Tensor) ->torch.Tensor:
    """Checks if the LAF is touching or partly outside the image boundary. Returns the mask
    of LAFs, which are fully inside the image, i.e. valid.

    Args:
        laf (torch.Tensor):  :math:`(B, N, 2, 3)`
        images (torch.Tensor): images, lafs are detected in :math:`(B, CH, H, W)`

    Returns:
        mask (torch.Tensor):  :math:`(B, N)`
    """
    raise_error_if_laf_is_not_valid(laf)
    n, ch, h, w = images.size()
    pts: torch.Tensor = laf_to_boundary_points(laf, 12)
    good_lafs_mask: torch.Tensor = (pts[..., 0] >= 0) * (pts[..., 0] <= w) * (pts[..., 1] >= 0) * (pts[..., 1] <= h)
    good_lafs_mask = good_lafs_mask.min(dim=2)[0]
    return good_lafs_mask


class ScaleSpaceDetector(nn.Module):
    """Module for differentiable local feature detection, as close as possible to classical
     local feature detectors like Harris, Hessian-Affine or SIFT (DoG).
     It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
     soft nms function, affine shape estimator and patch orientation estimator.
     Each of those modules could be replaced with learned custom one, as long, as
     they respect output shape.

    Args:
        num_features: (int) Number of features to detect. default = 500. In order to keep everything batchable,
                      output would always have num_features outputed, even for completely homogeneous images.
        mr_size: (float), default 6.0. Multiplier for local feature scale compared to the detection scale.
                    6.0 is matching OpenCV 12.0 convention for SIFT.
        scale_pyr_module: (nn.Module), which generates scale pyramid.
                         See :class:`~kornia.geometry.ScalePyramid` for details. Default is ScalePyramid(3, 1.6, 10)
        resp_module: (nn.Module), which calculates 'cornerness' of the pixel. Default is BlobHessian().
        nms_module: (nn.Module), which outputs per-patch coordinates of the responce maxima.
                    See :class:`~kornia.geometry.ConvSoftArgmax3d` for details.
        ori_module: (nn.Module) for local feature orientation estimation.  Default is :class:`~kornia.feature.PassLAF`,
                    which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module:  (nn.Module) for local feature affine shape estimation. Default is :class:`~kornia.feature.PassLAF`,
                    which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        minima_are_also_good:  (bool) if True, then both response function minima and maxima are detected
                                Useful for symmetric responce functions like DoG or Hessian. Default is False
    """

    def __init__(self, num_features: int=500, mr_size: float=6.0, scale_pyr_module: nn.Module=ScalePyramid(3, 1.6, 10), resp_module: nn.Module=BlobHessian(), nms_module: nn.Module=ConvSoftArgmax3d((3, 3, 3), (1, 1, 1), (1, 1, 1), normalized_coordinates=False, output_value=True), ori_module: nn.Module=PassLAF(), aff_module: nn.Module=PassLAF(), minima_are_also_good: bool=False):
        super(ScaleSpaceDetector, self).__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        self.scale_pyr = scale_pyr_module
        self.resp = resp_module
        self.nms = nms_module
        self.ori = ori_module
        self.aff = aff_module
        self.minima_are_also_good = minima_are_also_good
        return

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ', ' + 'mr_size=' + str(self.mr_size) + ', ' + 'scale_pyr=' + self.scale_pyr.__repr__() + ', ' + 'resp=' + self.resp.__repr__() + ', ' + 'nms=' + self.nms.__repr__() + ', ' + 'ori=' + self.ori.__repr__() + ', ' + 'aff=' + self.aff.__repr__() + ')'

    def detect(self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        dev: torch.device = img.device
        dtype: torch.dtype = img.dtype
        sp, sigmas, pix_dists = self.scale_pyr(img)
        all_responses = []
        all_lafs = []
        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            B, L, CH, H, W = octave.size()
            oct_resp = self.resp(octave.view(B * L, CH, H, W), sigmas_oct.view(-1))
            oct_resp = oct_resp.view_as(octave).permute(0, 2, 1, 3, 4)
            if mask is not None:
                oct_mask: torch.Tensor = _create_octave_mask(mask, oct_resp.shape)
                oct_resp = oct_mask * oct_resp
            coord_max, response_max = self.nms(oct_resp)
            if self.minima_are_also_good:
                coord_min, response_min = self.nms(-oct_resp)
                take_min_mask = response_min > response_max
                response_max = response_min * take_min_mask + (1 - take_min_mask) * response_max
                coord_max = coord_min * take_min_mask.unsqueeze(1) + (1 - take_min_mask.unsqueeze(1)) * coord_max
            responses_flatten = response_max.view(response_max.size(0), -1)
            max_coords_flatten = coord_max.view(response_max.size(0), 3, -1).permute(0, 2, 1)
            if responses_flatten.size(1) > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
                max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
            else:
                resp_flat_best = responses_flatten
                max_coords_best = max_coords_flatten
            B, N = resp_flat_best.size()
            max_coords_best = _scale_index_to_scale(max_coords_best, sigmas_oct)
            rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
            current_lafs = torch.cat([self.mr_size * max_coords_best[:, :, (0)].view(B, N, 1, 1) * rotmat, max_coords_best[:, :, 1:3].view(B, N, 2, 1)], dim=3)
            good_mask = laf_is_inside_image(current_lafs, octave[:, (0)])
            resp_flat_best = resp_flat_best * good_mask
            current_lafs = normalize_laf(current_lafs, octave[:, (0)])
            all_responses.append(resp_flat_best)
            all_lafs.append(current_lafs)
        responses: torch.Tensor = torch.cat(all_responses, dim=1)
        lafs: torch.Tensor = torch.cat(all_lafs, dim=1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))
        return responses, denormalize_laf(lafs, img)

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Three stage local feature detection. First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img (torch.Tensor): image to extract features with shape [BxCxHxW]
            mask (torch.Tensor, optional): a mask with weights where to apply the
            response function. The shae must same as the input image.

        Returns:
            lafs (torch.Tensor): shape [BxNx2x3]. Detected local affine frames.
            responses (torch.Tensor): shape [BxNx1]. Response function values for corresponding lafs"""
        responses, lafs = self.detect(img, self.num_features, mask)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses


def get_sift_bin_ksize_stride_pad(patch_size: int, num_spatial_bins: int) ->Tuple:
    """Returns a tuple with SIFT parameters, given the patch size
    and number of spatial bins.

    Args:
        patch_size: (int)
        num_spatial_bins: (int)

    Returns:
        ksize, stride, pad: ints
    """
    ksize: int = 2 * int(patch_size / (num_spatial_bins + 1))
    stride: int = patch_size // num_spatial_bins
    pad: int = ksize // 4
    return ksize, stride, pad


def get_sift_pooling_kernel(ksize: int=25) ->torch.Tensor:
    """Returns a weighted pooling kernel for SIFT descriptor

    Args:
        ksize: (int): kernel_size

    Returns:
        torch.Tensor: kernel

    Shape:
        Output: :math: `(ksize,ksize)`
    """
    ks_2: float = float(ksize) / 2.0
    xc2: torch.Tensor = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()
    kernel: torch.Tensor = torch.ger(xc2, xc2) / ks_2 ** 2
    return kernel


class SIFTDescriptor(nn.Module):
    """
    Module, which computes SIFT descriptors of given patches

    Args:
        patch_size: (int) Input patch size in pixels (41 is default)
        num_ang_bins: (int) Number of angular bins. (8 is default)
        num_spatial_bins: (int) Number of spatial bins (4 is default)
        clipval: (float) default 0.2
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012)
        is computed

    Returns:
        Tensor: SIFT descriptor of the patches

    Shape:
        - Input: (B, 1, num_spatial_bins, num_spatial_bins)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2)

    Examples::
        >>> input = torch.rand(23, 1, 32, 32)
        >>> SIFT = kornia.SIFTDescriptor(32, 8, 4)
        >>> descs = SIFT(input) # 23x128
    """

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(' + 'num_ang_bins=' + str(self.num_ang_bins) + ', ' + 'num_spatial_bins=' + str(self.num_spatial_bins) + ', ' + 'patch_size=' + str(self.patch_size) + ', ' + 'rootsift=' + str(self.rootsift) + ', ' + 'clipval=' + str(self.clipval) + ')'

    def __init__(self, patch_size: int=41, num_ang_bins: int=8, num_spatial_bins: int=4, rootsift: bool=True, clipval: float=0.2) ->None:
        super(SIFTDescriptor, self).__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.patch_size = patch_size
        ks: int = self.patch_size
        sigma: float = float(ks) / math.sqrt(2.0)
        self.gk = get_gaussian_kernel2d((ks, ks), (sigma, sigma), True)
        self.bin_ksize, self.bin_stride, self.pad = get_sift_bin_ksize_stride_pad(patch_size, num_spatial_bins)
        nw = get_sift_pooling_kernel(ksize=self.bin_ksize).float()
        self.pk = nn.Conv2d(1, 1, kernel_size=(nw.size(0), nw.size(1)), stride=(self.bin_stride, self.bin_stride), padding=(self.pad, self.pad), bias=False)
        self.pk.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))
        return

    def get_pooling_kernel(self) ->torch.Tensor:
        return self.pk.weight.detach()

    def get_weighting_kernel(self) ->torch.Tensor:
        return self.gk.detach()

    def forward(self, input):
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect Bx1xHxW. Got: {}'.format(input.shape))
        B, CH, W, H = input.size()
        if W != self.patch_size or H != self.patch_size or CH != 1:
            raise TypeError('input shape should be must be [Bx1x{}x{}]. Got {}'.format(self.patch_size, self.patch_size, input.size()))
        self.pk = self.pk.to(input.dtype)
        grads: torch.Tensor = spatial_gradient(input, 'diff')
        gx: torch.Tensor = grads[:, :, (0)]
        gy: torch.Tensor = grads[:, :, (1)]
        mag: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: torch.Tensor = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        mag = mag * self.gk.expand_as(mag).type_as(mag)
        o_big: torch.Tensor = float(self.num_ang_bins) * ori / (2.0 * pi)
        bo0_big_: torch.Tensor = torch.floor(o_big)
        wo1_big_: torch.Tensor = o_big - bo0_big_
        bo0_big: torch.Tensor = bo0_big_ % self.num_ang_bins
        bo1_big: torch.Tensor = (bo0_big + 1) % self.num_ang_bins
        wo0_big: torch.Tensor = (1.0 - wo1_big_) * mag
        wo1_big: torch.Tensor = wo1_big_ * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            out = self.pk((bo0_big == i) * wo0_big + (bo1_big == i) * wo1_big)
            ang_bins.append(out)
        ang_bins = torch.cat(ang_bins, dim=1)
        ang_bins = ang_bins.view(B, -1)
        ang_bins = F.normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(F.normalize(ang_bins, p=1) + self.eps)
        return ang_bins


class SOSNet(nn.Module):
    """
    128-dimensional SOSNet model definition for 32x32 patches.
    This is based on the original code from paper
    "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning".
    Args:
        pretrained: (bool) Download and set pretrained weights to the model. Default: false.
    Returns:
        torch.Tensor: SOSNet descriptor of the patches.
    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)
    Examples:
        >>> input = torch.rand(8, 1, 32, 32)
        >>> sosnet = kornia.feature.SOSNet()
        >>> descs = sosnet(input) # 8x128
    """

    def __init__(self, pretrained: bool=False) ->None:
        super(SOSNet, self).__init__()
        self.layers = nn.Sequential(nn.InstanceNorm2d(1, affine=False), nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(128, 128, kernel_size=8, bias=False), nn.BatchNorm2d(128, affine=False))
        self.desc_norm = nn.Sequential(nn.LocalResponseNorm(256, alpha=256, beta=0.5, k=0))
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls['lib'], map_location=lambda storage, loc: storage)
            self.load_state_dict(pretrained_dict['state_dict'], strict=True)
        return

    def forward(self, input: torch.Tensor, eps: float=1e-10) ->torch.Tensor:
        descr = self.desc_norm(self.layers(input) + eps)
        descr = descr.view(descr.size(0), -1)
        return descr


def get_box_kernel2d(kernel_size: Tuple[int, int]) ->torch.Tensor:
    """Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.0) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale * tmp_kernel


class BoxBlur(nn.Module):
    """Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \\frac{1}{\\text{kernel_size}_x * \\text{kernel_size}_y}
        \\begin{bmatrix}
            1 & 1 & 1 & \\cdots & 1 & 1 \\\\
            1 & 1 & 1 & \\cdots & 1 & 1 \\\\
            \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots \\\\
            1 & 1 & 1 & \\cdots & 1 & 1 \\\\
        \\end{bmatrix}

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int], border_type: str='reflect', normalized: bool=True) ->None:
        super(BoxBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.border_type: str = border_type
        self.kernel: torch.Tensor = get_box_kernel2d(kernel_size)
        self.normalized: bool = normalized
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(kernel_size=' + str(self.kernel_size) + ', ' + 'normalized=' + str(self.normalized) + ', ' + 'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):
        return kornia.filter2D(input, self.kernel, self.border_type)


def get_laplacian_kernel2d(kernel_size: int) ->torch.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\\text{kernel_size}_x, \\text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError('ksize must be an odd positive integer. Got {}'.format(kernel_size))
    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


class Laplacian(nn.Module):
    """Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size (int): the size of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = kornia.filters.Laplacian(5)
        >>> output = laplace(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: int, border_type: str='reflect', normalized: bool=True) ->None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized
        self.kernel: torch.Tensor = torch.unsqueeze(get_laplacian_kernel2d(kernel_size), dim=0)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(kernel_size=' + str(self.kernel_size) + ', ' + 'normalized=' + str(self.normalized) + ', ' + 'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):
        return kornia.filter2D(input, self.kernel, self.border_type)


def get_binary_kernel2d(window_size: Tuple[int, int]) ->torch.Tensor:
    """Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


class MedianBlur(nn.Module):
    """Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) ->None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = get_binary_kernel2d(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        b, c, h, w = input.shape
        kernel: torch.Tensor = self.kernel.to(input.device)
        features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=self.padding, stride=1)
        features = features.view(b, c, -1, h, w)
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median


def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) ->torch.Tensor:
    """Calculates an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \\begin{bmatrix}
            \\alpha & \\beta & (1 - \\alpha) \\cdot \\text{x}
            - \\beta \\cdot \\text{y} \\\\
            -\\beta & \\alpha & \\beta \\cdot \\text{x}
            + (1 - \\alpha) \\cdot \\text{y}
        \\end{bmatrix}

    where

    .. math::
        \\alpha = \\text{scale} \\cdot cos(\\text{angle}) \\\\
        \\beta = \\text{scale} \\cdot sin(\\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.

    Returns:
        Tensor: the affine matrix of 2D rotation.

    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> M = kornia.get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError('Input center type is not a torch.Tensor. Got {}'.format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError('Input angle type is not a torch.Tensor. Got {}'.format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError('Input scale type is not a torch.Tensor. Got {}'.format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError('Input center must be a Bx2 tensor. Got {}'.format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError('Input angle must be a B tensor. Got {}'.format(angle.shape))
    if not len(scale.shape) == 1:
        raise ValueError('Input scale must be a B tensor. Got {}'.format(scale.shape))
    if not center.shape[0] == angle.shape[0] == scale.shape[0]:
        raise ValueError('Inputs must have same batch size dimension. Got center {}, angle {} and scale {}'.format(center.shape, angle.shape, scale.shape))
    scaled_rotation: torch.Tensor = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha: torch.Tensor = scaled_rotation[:, (0), (0)]
    beta: torch.Tensor = scaled_rotation[:, (0), (1)]
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]
    batch_size: int = center.shape[0]
    one = torch.tensor(1.0)
    M: torch.Tensor = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[(...), 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y
    return M


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) ->torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(angle)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) ->torch.Tensor:
    """Computes the center of tensor plane."""
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def convert_affinematrix_to_homography(A: torch.Tensor) ->torch.Tensor:
    """Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].

    Examples::

        >>> input = torch.rand(2, 2, 3)  # Bx2x3
        >>> output = kornia.convert_affinematrix_to_homography(input)  # Bx3x3
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError('Input matrix must be a Bx2x3 tensor. Got {}'.format(A.shape))
    H: torch.Tensor = torch.nn.functional.pad(A, [0, 0, 0, 1], 'constant', value=0.0)
    H[..., -1, -1] += 1.0
    return H


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(obj)))


def normal_transform_pixel(height: int, width: int) ->torch.Tensor:
    """Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height (int): image height.
        width (int): image width.

    Returns:
        Tensor: normalized transform.

    Shape:
        Output: :math:`(1, 3, 3)`
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)
    tr_mat = tr_mat.unsqueeze(0)
    return tr_mat


def normalize_homography(dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]) ->torch.Tensor:
    """Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destiantion to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_src (tuple): size of the destination image (height, width).

    Returns:
        Tensor: the normalized homography.

    Shape:
        Output: :math:`(B, 3, 3)`
    """
    check_is_tensor(dst_pix_trans_src_pix)
    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError('Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}'.format(dst_pix_trans_src_pix.shape))
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w)
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def warp_affine(src: torch.Tensor, M: torch.Tensor, dsize: Tuple[int, int], flags: str='bilinear', padding_mode: str='zeros', align_corners: bool=False) ->torch.Tensor:
    """Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \\text{dst}(x, y) = \\text{src} \\left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \\right )

    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (torch.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details

    Returns:
        torch.Tensor: the warped tensor.

    Shape:
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://kornia.readthedocs.io/en/latest/
       tutorials/warp_affine.html>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError('Input src type is not a torch.Tensor. Got {}'.format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError('Input M type is not a torch.Tensor. Got {}'.format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError('Input src must be a BxCxHxW tensor. Got {}'.format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError('Input M must be a Bx2x3 tensor. Got {}'.format(M.shape))
    B, C, H, W = src.size()
    dsize_src = H, W
    out_size = dsize
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, dsize_src, out_size)
    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, out_size[0], out_size[1]], align_corners=align_corners)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode)


def affine(tensor: torch.Tensor, matrix: torch.Tensor, mode: str='bilinear', align_corners: bool=False) ->torch.Tensor:
    """Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.
        mode (str): 'bilinear' | 'nearest'
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        torch.Tensor: The warped image.
    """
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)
    matrix = matrix.expand(tensor.shape[0], -1, -1)
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode, align_corners=align_corners)
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)
    return warped


def rotate(tensor: torch.Tensor, angle: torch.Tensor, center: Union[None, torch.Tensor]=None, mode: str='bilinear', align_corners: bool=False) ->torch.Tensor:
    """Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input tensor type is not a torch.Tensor. Got {}'.format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError('Input angle type is not a torch.Tensor. Got {}'.format(type(angle)))
    if center is not None and not torch.is_tensor(angle):
        raise TypeError('Input center type is not a torch.Tensor. Got {}'.format(type(center)))
    if len(tensor.shape) not in (3, 4):
        raise ValueError('Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {}'.format(tensor.shape))
    if center is None:
        center = _compute_tensor_center(tensor)
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)
    return affine(tensor, rotation_matrix[(...), :2, :3], mode, align_corners)


def get_motion_kernel2d(kernel_size: int, angle: float, direction: float=0.0) ->torch.Tensor:
    """Function that returns motion blur filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.

    Returns:
        torch.Tensor: the motion blur kernel.

    Shape:
        - Output: :math:`(ksize, ksize)`

    Examples::
        >>> kornia.filters.get_motion_kernel2d(5, 0., 0.)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        >>> kornia.filters.get_motion_kernel2d(3, 215., -0.5)
            tensor([[0.0000, 0.0412, 0.0732],
                    [0.1920, 0.3194, 0.0804],
                    [0.2195, 0.0743, 0.0000]])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError('ksize must be an odd integer >= than 3')
    if not isinstance(angle, float):
        raise TypeError('angle must be a float')
    if not isinstance(direction, float):
        raise TypeError('direction must be a float')
    kernel_tuple: Tuple[int, int] = (kernel_size, kernel_size)
    direction = (torch.clamp(torch.tensor(direction), -1.0, 1.0).item() + 1.0) / 2.0
    kernel = torch.zeros(kernel_tuple, dtype=torch.float)
    kernel[(kernel_tuple[0] // 2), :] = torch.linspace(direction, 1.0 - direction, steps=kernel_tuple[0])
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = rotate(kernel, torch.tensor(angle))
    kernel = kernel[0][0]
    kernel /= kernel.sum()
    return kernel


def motion_blur(input: torch.Tensor, kernel_size: int, angle: float, direction: float, border_type: str='constant') ->torch.Tensor:
    """
    Function that blurs a tensor using the motion filter.

    See :class:`~kornia.filters.MotionBlur` for details.
    """
    assert border_type in ['constant', 'reflect', 'replicate', 'circular']
    kernel: torch.Tensor = torch.unsqueeze(get_motion_kernel2d(kernel_size, angle, direction), dim=0)
    return filter2D(input, kernel, border_type)


class MotionBlur(nn.Module):
    """Blurs a tensor using the motion filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
        border_type (str): the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> input = torch.rand(2, 4, 5, 7)
        >>> motion_blur = kornia.filters.MotionBlur(3, 35., 0.5)
        >>> output = motion_blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: int, angle: float, direction: float, border_type: str='constant') ->None:
        super(MotionBlur, self).__init__()
        self.kernel_size = kernel_size
        self.angle: float = angle
        self.direction: float = direction
        self.border_type: str = border_type

    def __repr__(self) ->str:
        return f'{self.__class__.__name__} (kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction})'

    def forward(self, x: torch.Tensor):
        return motion_blur(x, self.kernel_size, self.angle, self.direction, self.border_type)


def get_diff_kernel3d(device=torch.device('cpu'), dtype=torch.float) ->torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: torch.Tensor = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype)
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(device=torch.device('cpu'), dtype=torch.float) ->torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: torch.Tensor = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], [[[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype)
    return kernel.unsqueeze(1)


def get_spatial_gradient_kernel3d(mode: str, order: int, device=torch.device('cpu'), dtype=torch.float) ->torch.Tensor:
    """Function that returns kernel for 1st or 2nd order scale pyramid gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError('mode should be either sobel                         or diff. Got {}'.format(mode))
    if order not in [1, 2]:
        raise TypeError('order should be either 1 or 2                         Got {}'.format(order))
    if mode == 'sobel':
        raise NotImplementedError('Sobel kernel for 3d gradient is not implemented yet')
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel3d(device, dtype)
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)
    else:
        raise NotImplementedError('')
    return kernel


class SpatialGradient3d(nn.Module):
    """Computes the first and second order volume derivative in x, y and d using a diff
    operator.

    Return:
        torch.Tensor: the spatial gradients of the input feature map.

    Shape:
        - Input: :math:`(B, C, D, H, W)`. D, H, W are spatial dimensions, gradient is calculated w.r.t to them.
        - Output: :math:`(B, C, 3, D, H, W)` or :math:`(B, C, 6, D, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self, mode: str='diff', order: int=1) ->None:
        super(SpatialGradient3d, self).__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(order=' + str(self.order) + ', ' + 'mode=' + self.mode + ')'

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 5:
            raise ValueError('Invalid input shape, we expect BxCxDxHxW. Got: {}'.format(input.shape))
        b, c, d, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).detach()
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)
        kernel_flip: torch.Tensor = kernel.flip(-3)
        spatial_pad = [self.kernel.size(2) // 2, self.kernel.size(2) // 2, self.kernel.size(3) // 2, self.kernel.size(3) // 2, self.kernel.size(4) // 2, self.kernel.size(4) // 2]
        out_ch: int = 6 if self.order == 2 else 3
        return F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c).view(b, c, out_ch, d, h, w)


class Sobel(nn.Module):
    """Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.
        eps (float): regularization number to avoid NaN during backprop. Default: 1e-6.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self, normalized: bool=True, eps: float=1e-06) ->None:
        super(Sobel, self).__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(normalized=' + str(self.normalized) + ')'

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        edges: torch.Tensor = spatial_gradient(input, normalized=self.normalized)
        gx: torch.Tensor = edges[:, :, (0)]
        gy: torch.Tensor = edges[:, :, (1)]
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        return magnitude


def _get_center_kernel2d(h: int, w: int, device: torch.device=torch.device('cpu')) ->torch.Tensor:
    """Helper function, which generates a kernel to return center coordinates,
       when applied with F.conv2d to 2d coordinates grid.

    Args:
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [2x2xhxw]
    """
    center_kernel = torch.zeros(2, 2, h, w, device=device)
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = h // 2 + 1
    else:
        h_i1 = h // 2 - 1
        h_i2 = h // 2 + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = w // 2 + 1
    else:
        w_i1 = w // 2 - 1
        w_i2 = w // 2 + 1
    center_kernel[(0, 1), (0, 1), h_i1:h_i2, w_i1:w_i2] = 1.0 / float((h_i2 - h_i1) * (w_i2 - w_i1))
    return center_kernel


def normalize_pixel_coordinates(pixel_coordinates: torch.Tensor, height: int, width: int, eps: float=1e-08) ->torch.Tensor:
    """Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates (torch.Tensor): the grid with pixel coordinates.
          Shape can be :math:`(*, 2)`.
        width (int): the maximum width in the x-axis.
        height (int): the maximum height in the y-axis.
        eps (float): safe division by zero. (default 1e-8).

    Return:
        torch.Tensor: the normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError('Input pixel_coordinates must be of shape (*, 2). Got {}'.format(pixel_coordinates.shape))
    hw: torch.Tensor = torch.stack([torch.tensor(width), torch.tensor(height)]).to(pixel_coordinates.device)
    factor: torch.Tensor = torch.tensor(2.0) / (hw - 1).clamp(eps)
    return factor * pixel_coordinates - 1


def _get_window_grid_kernel2d(h: int, w: int, device: torch.device=torch.device('cpu')) ->torch.Tensor:
    """Helper function, which generates a kernel to with window coordinates,
       residual to window center.

    Args:
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [2x1xhxw]
    """
    window_grid2d = create_meshgrid(h, w, False, device=device)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel


def conv_soft_argmax2d(input: torch.Tensor, kernel_size: Tuple[int, int]=(3, 3), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(1, 1), temperature: Union[torch.Tensor, float]=torch.tensor(1.0), normalized_coordinates: bool=True, eps: float=1e-08, output_value: bool=False) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Function that computes the convolutional spatial Soft-Argmax 2D over the windows
    of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
    themselves. On each window, the function computed is

    .. math::
             ij(X) = \\frac{\\sum{(i,j)} * exp(x / T)  \\in X} {\\sum{exp(x / T)  \\in X}}

    .. math::
             val(X) = \\frac{\\sum{x * exp(x / T)  \\in X}} {\\sum{exp(x / T)  \\in X}}

    where T is temperature.

    Args:
        kernel_size (Tuple[int,int]): the size of the window
        stride  (Tuple[int,int]): the stride of the window.
        padding (Tuple[int,int]): input zero padding
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the coordinates normalized in the range of [-1, 1]. Otherwise,
                                       it will return the coordinates in the range of the input shape. Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.
        output_value (bool): if True, val is outputed, if False, only ij

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, 2, H_{out}, W_{out})`, :math:`(N, C, H_{out}, W_{out})`, where

         .. math::
                  H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] -
                  (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

         .. math::
                  W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] -
                  (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1))
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
    if temperature <= 0:
        raise ValueError('Temperature should be positive float or tensor. Got: {}'.format(temperature))
    b, c, h, w = input.shape
    kx, ky = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, h, w)
    center_kernel: torch.Tensor = _get_center_kernel2d(kx, ky, device)
    window_kernel: torch.Tensor = _get_window_grid_kernel2d(kx, ky, device)
    x_max = F.adaptive_max_pool2d(input, (1, 1))
    x_exp = ((input - x_max.detach()) / temperature).exp()
    pool_coef: float = float(kx * ky)
    den = pool_coef * F.avg_pool2d(x_exp, kernel_size, stride=stride, padding=padding) + eps
    x_softmaxpool = pool_coef * F.avg_pool2d(x_exp * input, kernel_size, stride=stride, padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3))
    grid_global: torch.Tensor = create_meshgrid(h, w, False, device).permute(0, 3, 1, 2)
    grid_global_pooled = F.conv2d(grid_global, center_kernel, stride=stride, padding=padding)
    coords_max: torch.Tensor = F.conv2d(x_exp, window_kernel, stride=stride, padding=padding)
    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates(coords_max.permute(0, 2, 3, 1), h, w)
        coords_max = coords_max.permute(0, 3, 1, 2)
    coords_max = coords_max.view(b, c, 2, coords_max.size(2), coords_max.size(3))
    if output_value:
        return coords_max, x_softmaxpool
    return coords_max


class ConvSoftArgmax2d(nn.Module):
    """Module that calculates soft argmax 2d per window.

    See :func:`~kornia.geometry.conv_soft_argmax2d` for details.
    """

    def __init__(self, kernel_size: Tuple[int, int]=(3, 3), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(1, 1), temperature: Union[torch.Tensor, float]=torch.tensor(1.0), normalized_coordinates: bool=True, eps: float=1e-08, output_value: bool=False) ->None:
        super(ConvSoftArgmax2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(' + 'kernel_size=' + str(self.kernel_size) + ', ' + 'stride=' + str(self.stride) + ', ' + 'padding=' + str(self.padding) + ', ' + 'temperature=' + str(self.temperature) + ', ' + 'normalized_coordinates=' + str(self.normalized_coordinates) + ', ' + 'eps=' + str(self.eps) + ', ' + 'output_value=' + str(self.output_value) + ')'

    def forward(self, x: torch.Tensor):
        return conv_soft_argmax2d(x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates, self.eps, self.output_value)


class SpatialSoftArgmax2d(nn.Module):
    """Module that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.contrib.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature: torch.Tensor=torch.tensor(1.0), normalized_coordinates: bool=True, eps: float=1e-08) ->None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates
        self.eps: float = eps

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(temperature=' + str(self.temperature) + ', ' + 'normalized_coordinates=' + str(self.normalized_coordinates) + ', ' + 'eps=' + str(self.eps) + ')'

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates, self.eps)


def conv_quad_interp3d(input: torch.Tensor, strict_maxima_bonus: float=1.0, eps: float=1e-06):
    """Function that computes the single iteration of quadratic interpolation of of the extremum (max or min) location
    and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT

    Args:
        strict_maxima_bonus (float): pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
                                     This is needed for mimic behavior of strict NMS in classic local features
        eps (float): parameter to control the hessian matrix ill-condition number.
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

         .. math::
             D_{out} = \\left\\lfloor\\frac{D_{in}  + 2 \\times \\text{padding}[0] -
             (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

         .. math::
             H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[1] -
             (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

         .. math::
             W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[2] -
             (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_quad_interp3d(input, 1.0)
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 5:
        raise ValueError('Invalid input shape, we expect BxCxDxHxW. Got: {}'.format(input.shape))
    B, CH, D, H, W = input.shape
    dev: torch.device = input.device
    grid_global: torch.Tensor = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global
    b: torch.Tensor = kornia.filters.spatial_gradient3d(input, order=1, mode='diff')
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A: torch.Tensor = kornia.filters.spatial_gradient3d(input, order=2, mode='diff')
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = A[..., 3]
    dys = A[..., 4]
    dxs = A[..., 5]
    Hes = torch.stack([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss]).view(-1, 3, 3)
    Hes += torch.eye(3, device=Hes.device)[None] * eps
    nms_mask: torch.Tensor = kornia.feature.nms3d(input, (3, 3, 3), True)
    x_solved: torch.Tensor = torch.zeros_like(b)
    x_solved_masked, _ = torch.solve(b[nms_mask.view(-1)], Hes[nms_mask.view(-1)])
    x_solved.masked_scatter_(nms_mask.view(-1, 1, 1), x_solved_masked)
    dx: torch.Tensor = -x_solved
    dx[((dx.abs().max(dim=1, keepdim=True)[0] > 0.7).view(-1)), :, :] = 0
    dy: torch.Tensor = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)
    y_max = input + dy.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max *= 1.0 + strict_maxima_bonus * nms_mask
    dx_res: torch.Tensor = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    coords_max: torch.Tensor = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)
    coords_max = coords_max + dx_res
    return coords_max, y_max


class ConvQuadInterp3d(nn.Module):
    """Module that calculates soft argmax 3d per window
    See :func:`~kornia.geometry.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float=1.0, eps: float=1e-06) ->None:
        super(ConvQuadInterp3d, self).__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps
        return

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(' + 'strict_maxima_bonus=' + str(self.strict_maxima_bonus) + ')'

    def forward(self, x: torch.Tensor):
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)


def resize(input: torch.Tensor, size: Union[int, Tuple[int, int]], interpolation: str='bilinear', align_corners: bool=False) ->torch.Tensor:
    """Resize the input torch.Tensor to the given size.

    See :class:`~kornia.Resize` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input tensor type is not a torch.Tensor. Got {}'.format(type(input)))
    new_size: Tuple[int, int]
    if isinstance(size, int):
        w, h = input.shape[-2:]
        if w <= h and w == size or h <= w and h == size:
            return input
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        new_size = ow, oh
    else:
        new_size = size
    return torch.nn.functional.interpolate(input, size=new_size, mode=interpolation, align_corners=align_corners)


class Resize(nn.Module):
    """Resize the input torch.Tensor to the given size.

    Args:
        size (int, tuple(int, int)): Desired output size. If size is a sequence like (h, w),
        output size will be matched to this. If size is an int, smaller edge of the image will
        be matched to this number. i.e, if height > width, then image will be rescaled
        to (size * height / width, size)
        interpolation (str):  algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' |
        'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The resized tensor.
    """

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str='bilinear', align_corners: bool=False) ->None:
        super(Resize, self).__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return resize(input, self.size, self.interpolation, align_corners=self.align_corners)


class Rotate(nn.Module):
    """Rotate the tensor anti-clockwise about the centre.

    Args:
        angle (torch.Tensor): The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The rotated tensor.
    """

    def __init__(self, angle: torch.Tensor, center: Union[None, torch.Tensor]=None, align_corners: bool=False) ->None:
        super(Rotate, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return rotate(input, self.angle, self.center, align_corners=self.align_corners)


def _compute_translation_matrix(translation: torch.Tensor) ->torch.Tensor:
    """Computes affine matrix for translation."""
    matrix: torch.Tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)
    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[(...), (0), 2:3] += dx
    matrix[(...), (1), 2:3] += dy
    return matrix


def translate(tensor: torch.Tensor, translation: torch.Tensor, align_corners: bool=False) ->torch.Tensor:
    """Translate the tensor in pixel units.

    See :class:`~kornia.Translate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input tensor type is not a torch.Tensor. Got {}'.format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError('Input translation type is not a torch.Tensor. Got {}'.format(type(translation)))
    if len(tensor.shape) not in (3, 4):
        raise ValueError('Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {}'.format(tensor.shape))
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)
    return affine(tensor, translation_matrix[(...), :2, :3], align_corners=align_corners)


class Translate(nn.Module):
    """Translate the tensor in pixel units.

    Args:
        translation (torch.Tensor): tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The translated tensor.
    """

    def __init__(self, translation: torch.Tensor, align_corners: bool=False) ->None:
        super(Translate, self).__init__()
        self.translation: torch.Tensor = translation
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return translate(input, self.translation, self.align_corners)


def _compute_scaling_matrix(scale: torch.Tensor, center: torch.Tensor) ->torch.Tensor:
    """Computes affine matrix for scaling."""
    angle: torch.Tensor = torch.zeros_like(scale)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def scale(tensor: torch.Tensor, scale_factor: torch.Tensor, center: Union[None, torch.Tensor]=None, align_corners: bool=False) ->torch.Tensor:
    """Scales the input image.

    See :class:`~kornia.Scale` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input tensor type is not a torch.Tensor. Got {}'.format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError('Input scale_factor type is not a torch.Tensor. Got {}'.format(type(scale_factor)))
    if center is None:
        center = _compute_tensor_center(tensor)
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0])
    scaling_matrix: torch.Tensor = _compute_scaling_matrix(scale_factor, center)
    return affine(tensor, scaling_matrix[(...), :2, :3], align_corners=align_corners)


class Scale(nn.Module):
    """Scale the tensor by a factor.

    Args:
        scale_factor (torch.Tensor): The scale factor apply. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The scaled tensor.
    """

    def __init__(self, scale_factor: torch.Tensor, center: Union[None, torch.Tensor]=None, align_corners: bool=False) ->None:
        super(Scale, self).__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: Union[None, torch.Tensor] = center
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return scale(input, self.scale_factor, self.center, self.align_corners)


def _compute_shear_matrix(shear: torch.Tensor) ->torch.Tensor:
    """Computes affine matrix for shearing."""
    matrix: torch.Tensor = torch.eye(3, device=shear.device, dtype=shear.dtype)
    matrix = matrix.repeat(shear.shape[0], 1, 1)
    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[(...), (0), 1:2] += shx
    matrix[(...), (1), 0:1] += shy
    return matrix


def shear(tensor: torch.Tensor, shear: torch.Tensor, align_corners: bool=False) ->torch.Tensor:
    """Shear the tensor.

    See :class:`~kornia.Shear` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input tensor type is not a torch.Tensor. Got {}'.format(type(tensor)))
    if not torch.is_tensor(shear):
        raise TypeError('Input shear type is not a torch.Tensor. Got {}'.format(type(shear)))
    if len(tensor.shape) not in (3, 4):
        raise ValueError('Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {}'.format(tensor.shape))
    shear_matrix: torch.Tensor = _compute_shear_matrix(shear)
    return affine(tensor, shear_matrix[(...), :2, :3], align_corners=align_corners)


class Shear(nn.Module):
    """Shear the tensor.

    Args:
        tensor (torch.Tensor): The image tensor to be skewed.
        shear (torch.Tensor): tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The skewed tensor.
    """

    def __init__(self, shear: torch.Tensor, align_corners: bool=False) ->None:
        super(Shear, self).__init__()
        self.shear: torch.Tensor = shear
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return shear(input, self.shear, self.align_corners)


def vflip(input: torch.Tensor) ->torch.Tensor:
    return torch.flip(input, [-2])


class Vflip(nn.Module):
    """Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The vertically flipped image tensor

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> kornia.vflip(input)
        tensor([[[0, 1, 1],
                 [0, 0, 0],
                 [0, 0, 0]]])
    """

    def __init__(self) ->None:
        super(Vflip, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return vflip(input)

    def __repr__(self):
        return self.__class__.__name__


def hflip(input: torch.Tensor) ->torch.Tensor:
    return torch.flip(input, [-1])


class Hflip(nn.Module):
    """Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> kornia.hflip(input)
        tensor([[[0, 0, 0],
                 [0, 0, 0],
                 [1, 1, 0]]])
    """

    def __init__(self) ->None:
        super(Hflip, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return hflip(input)

    def __repr__(self):
        return self.__class__.__name__


def rot180(input: torch.Tensor) ->torch.Tensor:
    return torch.flip(input, [-2, -1])


class Rot180(nn.Module):
    """Rotate a tensor image or a batch of tensor images
        180 degrees. Input must be a tensor of shape (C, H, W)
        or a batch of tensors :math:`(*, C, H, W)`.

        Args:
            input (torch.Tensor): input tensor

        Examples:
            >>> input = torch.tensor([[[
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 1., 1.]]]])
            >>> kornia.rot180(input)
            tensor([[[1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]]])
        """

    def __init__(self) ->None:
        super(Rot180, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return rot180(input)

    def __repr__(self):
        return self.__class__.__name__


class PyrUp(nn.Module):
    """Upsamples a tensor and then blurs it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str='reflect', align_corners: bool=False):
        super(PyrUp, self).__init__()
        self.border_type: str = border_type
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(input.shape))
        b, c, height, width = input.shape
        x_up: torch.Tensor = F.interpolate(input, size=(height * 2, width * 2), mode='bilinear', align_corners=self.align_corners)
        x_blur: torch.Tensor = filter2D(x_up, self.kernel, self.border_type)
        return x_blur


class PinholeCamera:
    """Class that represents a Pinhole Camera model.

    Args:
        intrinsics (torch.Tensor): tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 camera calibration matrix.
        extrinsics (torch.Tensor): tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 rotation-translation matrix.
        height (torch.Tensor): tensor with shape :math:`(B)` containing the
          image height.
        width (torch.Tensor): tensor with shape :math:`(B)` containing the image
          width.

    .. note::
        We assume that the class attributes are in batch form in order to take
        advantage of PyTorch parallelism to boost computing performce.
    """

    def __init__(self, intrinsics: torch.Tensor, extrinsics: torch.Tensor, height: torch.Tensor, width: torch.Tensor) ->None:
        self._check_valid([intrinsics, extrinsics, height, width])
        self._check_valid_params(intrinsics, 'intrinsics')
        self._check_valid_params(extrinsics, 'extrinsics')
        self._check_valid_shape(height, 'height')
        self._check_valid_shape(width, 'width')
        self.height: torch.Tensor = height
        self.width: torch.Tensor = width
        self._intrinsics: torch.Tensor = intrinsics
        self._extrinsics: torch.Tensor = extrinsics

    @staticmethod
    def _check_valid(data_iter: Iterable[torch.Tensor]) ->bool:
        if not all(data.shape[0] for data in data_iter):
            raise ValueError('Arguments shapes must match')
        return True

    @staticmethod
    def _check_valid_params(data: torch.Tensor, data_name: str) ->bool:
        if len(data.shape) not in (3, 4) and data.shape[-2:] != (4, 4):
            raise ValueError('Argument {0} shape must be in the following shape Bx4x4 or BxNx4x4. Got {1}'.format(data_name, data.shape))
        return True

    @staticmethod
    def _check_valid_shape(data: torch.Tensor, data_name: str) ->bool:
        if not len(data.shape) == 1:
            raise ValueError('Argument {0} shape must be in the following shape B. Got {1}'.format(data_name, data.shape))
        return True

    @property
    def intrinsics(self) ->torch.Tensor:
        """The full 4x4 intrinsics matrix.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 4, 4)`
        """
        assert self._check_valid_params(self._intrinsics, 'intrinsics')
        return self._intrinsics

    @property
    def extrinsics(self) ->torch.Tensor:
        """The full 4x4 extrinsics matrix.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 4, 4)`
        """
        assert self._check_valid_params(self._extrinsics, 'extrinsics')
        return self._extrinsics

    @property
    def batch_size(self) ->int:
        """Returns the batch size of the storage.

        Returns:
            int: scalar with the batch size
        """
        return self.intrinsics.shape[0]

    @property
    def fx(self) ->torch.Tensor:
        """Returns the focal lenght in the x-direction.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.intrinsics[..., 0, 0]

    @property
    def fy(self) ->torch.Tensor:
        """Returns the focal lenght in the y-direction.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.intrinsics[..., 1, 1]

    @property
    def cx(self) ->torch.Tensor:
        """Returns the x-coordinate of the principal point.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.intrinsics[..., 0, 2]

    @property
    def cy(self) ->torch.Tensor:
        """Returns the y-coordinate of the principal point.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.intrinsics[..., 1, 2]

    @property
    def tx(self) ->torch.Tensor:
        """Returns the x-coordinate of the translation vector.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.extrinsics[..., 0, -1]

    @tx.setter
    def tx(self, value) ->'PinholeCamera':
        """Set the x-coordinate of the translation vector with the given
        value.
        """
        self.extrinsics[..., 0, -1] = value
        return self

    @property
    def ty(self) ->torch.Tensor:
        """Returns the y-coordinate of the translation vector.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.extrinsics[..., 1, -1]

    @ty.setter
    def ty(self, value) ->'PinholeCamera':
        """Set the y-coordinate of the translation vector with the given
        value.
        """
        self.extrinsics[..., 1, -1] = value
        return self

    @property
    def tz(self) ->torch.Tensor:
        """Returns the z-coordinate of the translation vector.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.extrinsics[..., 2, -1]

    @tz.setter
    def tz(self, value) ->'PinholeCamera':
        """Set the y-coordinate of the translation vector with the given
        value.
        """
        self.extrinsics[..., 2, -1] = value
        return self

    @property
    def rt_matrix(self) ->torch.Tensor:
        """Returns the 3x4 rotation-translation matrix.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 3, 4)`
        """
        return self.extrinsics[(...), :3, :4]

    @property
    def camera_matrix(self) ->torch.Tensor:
        """Returns the 3x3 camera matrix containing the intrinsics.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 3, 3)`
        """
        return self.intrinsics[(...), :3, :3]

    @property
    def rotation_matrix(self) ->torch.Tensor:
        """Returns the 3x3 rotation matrix from the extrinsics.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 3, 3)`
        """
        return self.extrinsics[(...), :3, :3]

    @property
    def translation_vector(self) ->torch.Tensor:
        """Returns the translation vector from the extrinsics.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 3, 1)`
        """
        return self.extrinsics[(...), :3, -1:]

    def clone(self) ->'PinholeCamera':
        """Returns a deep copy of the current object instance."""
        height: torch.Tensor = self.height.clone()
        width: torch.Tensor = self.width.clone()
        intrinsics: torch.Tensor = self.intrinsics.clone()
        extrinsics: torch.Tensor = self.extrinsics.clone()
        return PinholeCamera(intrinsics, extrinsics, height, width)

    def intrinsics_inverse(self) ->torch.Tensor:
        """Returns the inverse of the 4x4 instrisics matrix.

        Returns:
            torch.Tensor: tensor of shape :math:`(B, 4, 4)`
        """
        return self.intrinsics.inverse()

    def scale(self, scale_factor) ->'PinholeCamera':
        """Scales the pinhole model.

        Args:
            scale_factor (torch.Tensor): a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            PinholeCamera: the camera model with scaled parameters.
        """
        intrinsics: torch.Tensor = self.intrinsics.clone()
        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor
        height: torch.Tensor = scale_factor * self.height.clone()
        width: torch.Tensor = scale_factor * self.width.clone()
        return PinholeCamera(intrinsics, self.extrinsics, height, width)

    def scale_(self, scale_factor) ->'PinholeCamera':
        """Scales the pinhole model in-place.

        Args:
            scale_factor (torch.Tensor): a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            PinholeCamera: the camera model with scaled parameters.
        """
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor
        self.height *= scale_factor
        self.width *= scale_factor
        return self

    @classmethod
    def from_parameters(self, fx, fy, cx, cy, height, width, tx, ty, tz, batch_size=1):
        intrinsics = torch.zeros(batch_size, 4, 4)
        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0
        extrinsics = torch.eye(4).repeat(batch_size, 1, 1)
        extrinsics[..., 0, -1] += tx
        extrinsics[..., 1, -1] += ty
        extrinsics[..., 2, -1] += tz
        height_tmp = torch.zeros(batch_size)
        height_tmp[..., 0] += height
        width_tmp = torch.zeros(batch_size)
        width_tmp[..., 0] += width
        return self(intrinsics, extrinsics, height_tmp, width_tmp)


def convert_points_from_homogeneous(points: torch.Tensor, eps: float=1e-08) ->torch.Tensor:
    """Function that converts points from homogeneous to Euclidean space.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    z_vec: torch.Tensor = points[(...), -1:]
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale: torch.Tensor = torch.ones_like(z_vec).masked_scatter_(mask, torch.tensor(1.0) / z_vec[mask])
    return scale * points[(...), :-1]


def convert_points_to_homogeneous(points: torch.Tensor) ->torch.Tensor:
    """Function that converts points from Euclidean to homogeneous space.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    return torch.nn.functional.pad(points, [0, 1], 'constant', 1.0)


def transform_points(trans_01: torch.Tensor, points_1: torch.Tensor) ->torch.Tensor:
    """Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = kornia.transform_points(trans_01, points_1)  # BxNx3
    """
    check_is_tensor(trans_01)
    check_is_tensor(points_1)
    if not trans_01.device == points_1.device:
        raise TypeError('Tensor must be in the same device')
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError('Input batch size must be the same for both tensors or 1')
    if not trans_01.shape[-1] == points_1.shape[-1] + 1:
        raise ValueError('Last input dimensions must differe by one unit')
    points_1_h = convert_points_to_homogeneous(points_1)
    points_0_h = torch.matmul(trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    points_0 = convert_points_from_homogeneous(points_0_h)
    return points_0


def cam2pixel(cam_coords_src: torch.Tensor, dst_proj_src: torch.Tensor, eps: Optional[float]=1e-06) ->torch.Tensor:
    """Transform coordinates in the camera frame to the pixel frame.

    Args:
        cam_coords (torch.Tensor): pixel coordinates defined in the first
          camera coordinates system. Shape must be BxHxWx3.
        dst_proj_src (torch.Tensor): the projection matrix between the
          reference and the non reference camera frame. Shape must be Bx4x4.

    Returns:
        torch.Tensor: array of [-1, 1] coordinates of shape BxHxWx2.
    """
    if not len(cam_coords_src.shape) == 4 and cam_coords_src.shape[3] == 3:
        raise ValueError('Input cam_coords_src has to be in the shape of BxHxWx3. Got {}'.format(cam_coords_src.shape))
    if not len(dst_proj_src.shape) == 3 and dst_proj_src.shape[-2:] == (4, 4):
        raise ValueError('Input dst_proj_src has to be in the shape of Bx4x4. Got {}'.format(dst_proj_src.shape))
    b, h, w, _ = cam_coords_src.shape
    point_coords: torch.Tensor = transform_points(dst_proj_src[:, (None)], cam_coords_src)
    x_coord: torch.Tensor = point_coords[..., 0]
    y_coord: torch.Tensor = point_coords[..., 1]
    z_coord: torch.Tensor = point_coords[..., 2]
    u_coord: torch.Tensor = x_coord / (z_coord + eps)
    v_coord: torch.Tensor = y_coord / (z_coord + eps)
    pixel_coords_dst: torch.Tensor = torch.stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst


def pixel2cam(depth: torch.Tensor, intrinsics_inv: torch.Tensor, pixel_coords: torch.Tensor) ->torch.Tensor:
    """Transform coordinates in the pixel frame to the camera frame.

    Args:
        depth (torch.Tensor): the source depth maps. Shape must be Bx1xHxW.
        intrinsics_inv (torch.Tensor): the inverse intrinsics camera matrix.
          Shape must be Bx4x4.
        pixel_coords (torch.Tensor): the grid with the homogeneous camera
          coordinates. Shape must be BxHxWx3.

    Returns:
        torch.Tensor: array of (u, v, 1) cam coordinates with shape BxHxWx3.
    """
    if not len(depth.shape) == 4 and depth.shape[1] == 1:
        raise ValueError('Input depth has to be in the shape of Bx1xHxW. Got {}'.format(depth.shape))
    if not len(intrinsics_inv.shape) == 3:
        raise ValueError('Input intrinsics_inv has to be in the shape of Bx4x4. Got {}'.format(intrinsics_inv.shape))
    if not len(pixel_coords.shape) == 4 and pixel_coords.shape[3] == 3:
        raise ValueError('Input pixel_coords has to be in the shape of BxHxWx3. Got {}'.format(intrinsics_inv.shape))
    cam_coords: torch.Tensor = transform_points(intrinsics_inv[:, (None)], pixel_coords)
    return cam_coords * depth.permute(0, 2, 3, 1)


def compose_transformations(trans_01: torch.Tensor, trans_12: torch.Tensor) ->torch.Tensor:
    """Functions that composes two homogeneous transformations.

    .. math::

        T_0^{2} = \\begin{bmatrix} R_0^1 R_1^{2} & R_0^{1} t_1^{2} + t_0^{1} \\\\
        \\mathbf{0} & 1\\end{bmatrix}

    Args:
        trans_01 (torch.Tensor): tensor with the homogenous transformation from
          a reference frame 1 respect to a frame 0. The tensor has must have a
          shape of :math:`(B, 4, 4)` or :math:`(4, 4)`.
        trans_12 (torch.Tensor): tensor with the homogenous transformation from
          a reference frame 2 respect to a frame 1. The tensor has must have a
          shape of :math:`(B, 4, 4)` or :math:`(4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`

    Returns:
        torch.Tensor: the transformation between the two frames.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_12 = torch.eye(4)  # 4x4
        >>> trans_02 = kornia.compose_transformations(trans_01, trans_12)  # 4x4

    """
    if not torch.is_tensor(trans_01):
        raise TypeError('Input trans_01 type is not a torch.Tensor. Got {}'.format(type(trans_01)))
    if not torch.is_tensor(trans_12):
        raise TypeError('Input trans_12 type is not a torch.Tensor. Got {}'.format(type(trans_12)))
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError('Input trans_01 must be a of the shape Nx4x4 or 4x4. Got {}'.format(trans_01.shape))
    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError('Input trans_12 must be a of the shape Nx4x4 or 4x4. Got {}'.format(trans_12.shape))
    if not trans_01.dim() == trans_12.dim():
        raise ValueError('Input number of dims must match. Got {} and {}'.format(trans_01.dim(), trans_12.dim()))
    rmat_01: torch.Tensor = trans_01[(...), :3, :3]
    rmat_12: torch.Tensor = trans_12[(...), :3, :3]
    tvec_01: torch.Tensor = trans_01[(...), :3, -1:]
    tvec_12: torch.Tensor = trans_12[(...), :3, -1:]
    rmat_02: torch.Tensor = torch.matmul(rmat_01, rmat_12)
    tvec_02: torch.Tensor = torch.matmul(rmat_01, tvec_12) + tvec_01
    trans_02: torch.Tensor = torch.zeros_like(trans_01)
    trans_02[(...), :3, 0:3] += rmat_02
    trans_02[(...), :3, -1:] += tvec_02
    trans_02[(...), (-1), -1:] += 1.0
    return trans_02


def inverse_transformation(trans_12):
    """Function that inverts a 4x4 homogeneous transformation
    :math:`T_1^{2} = \\begin{bmatrix} R_1 & t_1 \\\\ \\mathbf{0} & 1 \\end{bmatrix}`

    The inverse transformation is computed as follows:

    .. math::

        T_2^{1} = (T_1^{2})^{-1} = \\begin{bmatrix} R_1^T & -R_1^T t_1 \\\\
        \\mathbf{0} & 1\\end{bmatrix}

    Args:
        trans_12 (torch.Tensor): transformation tensor of shape
          :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: tensor with inverted transformations.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = kornia.inverse_transformation(trans_12)  # Nx4x4
    """
    if not torch.is_tensor(trans_12):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(trans_12)))
    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError('Input size must be a Nx4x4 or 4x4. Got {}'.format(trans_12.shape))
    rmat_12: torch.Tensor = trans_12[(...), :3, 0:3]
    tvec_12: torch.Tensor = trans_12[(...), :3, 3:4]
    rmat_21: torch.Tensor = torch.transpose(rmat_12, -1, -2)
    tvec_21: torch.Tensor = torch.matmul(-rmat_21, tvec_12)
    trans_21: torch.Tensor = torch.zeros_like(trans_12)
    trans_21[(...), :3, 0:3] += rmat_21
    trans_21[(...), :3, -1:] += tvec_21
    trans_21[(...), (-1), -1:] += 1.0
    return trans_21


def relative_transformation(trans_01: torch.Tensor, trans_02: torch.Tensor) ->torch.Tensor:
    """Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \\begin{bmatrix} R_1 & t_1 \\\\
    \\mathbf{0} & 1 \\end{bmatrix}` to destination :math:`T_2^{0} =
    \\begin{bmatrix} R_2 & t_2 \\\\ \\mathbf{0} & 1 \\end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \\cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = kornia.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError('Input trans_01 type is not a torch.Tensor. Got {}'.format(type(trans_01)))
    if not torch.is_tensor(trans_02):
        raise TypeError('Input trans_02 type is not a torch.Tensor. Got {}'.format(type(trans_02)))
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError('Input must be a of the shape Nx4x4 or 4x4. Got {}'.format(trans_01.shape))
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError('Input must be a of the shape Nx4x4 or 4x4. Got {}'.format(trans_02.shape))
    if not trans_01.dim() == trans_02.dim():
        raise ValueError('Input number of dims must match. Got {} and {}'.format(trans_01.dim(), trans_02.dim()))
    trans_10: torch.Tensor = inverse_transformation(trans_01)
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


class DepthWarper(nn.Module):
    """Warps a patch by depth.

    .. math::
        P_{src}^{\\{dst\\}} = K_{dst} * T_{src}^{\\{dst\\}}

        I_{src} = \\\\omega(I_{dst}, P_{src}^{\\{dst\\}}, D_{src})

    Args:
        pinholes_dst (PinholeCamera): the pinhole models for the destination
          frame.
        height (int): the height of the image to warp.
        width (int): the width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
           'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool): interpolation flag. Default: True. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    """

    def __init__(self, pinhole_dst: PinholeCamera, height: int, width: int, mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=True):
        super(DepthWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.eps = 1e-06
        self.align_corners: bool = align_corners
        self._pinhole_dst: PinholeCamera = pinhole_dst
        self._pinhole_src: Union[None, PinholeCamera] = None
        self._dst_proj_src: Union[None, torch.Tensor] = None
        self.grid: torch.Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) ->torch.Tensor:
        grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=False)
        return convert_points_to_homogeneous(grid)

    def compute_projection_matrix(self, pinhole_src: PinholeCamera) ->'DepthWarper':
        """Computes the projection matrix from the source to destinaion frame.
        """
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError('Member self._pinhole_dst expected to be of class PinholeCamera. Got {}'.format(type(self._pinhole_dst)))
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError('Argument pinhole_src expected to be of class PinholeCamera. Got {}'.format(type(pinhole_src)))
        dst_trans_src: torch.Tensor = relative_transformation(pinhole_src.extrinsics, self._pinhole_dst.extrinsics)
        dst_proj_src: torch.Tensor = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self

    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        flow = torch.matmul(self._dst_proj_src, point)
        z = 1.0 / flow[:, (2)]
        x = flow[:, (0)] * z
        y = flow[:, (1)] * z
        return torch.cat([x, y], 1)

    def compute_subpixel_step(self) ->torch.Tensor:
        """This computes the required inverse depth step to achieve sub pixel
        accurate sampling of the depth cost volume, per camera.

        Szeliski, Richard, and Daniel Scharstein.
        "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 + delta_d)
        dx = torch.norm(xy_p1 - xy_m1, 2, dim=-1) / 2.0
        dxdd = dx / delta_d
        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src: torch.Tensor) ->torch.Tensor:
        """Computes a grid for warping a given the depth from the reference
        pinhole camera.

        The function `compute_projection_matrix` has to be called beforehand in
        order to have precomputed the relative projection matrices encoding the
        relative pose and the intrinsics between the reference and a non
        reference camera.
        """
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError('Please, call compute_projection_matrix.')
        if len(depth_src.shape) != 4:
            raise ValueError('Input depth_src has to be in the shape of Bx1xHxW. Got {}'.format(depth_src.shape))
        batch_size, _, height, width = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype
        pixel_coords: torch.Tensor = self.grid.to(device).expand(batch_size, -1, -1, -1)
        cam_coords_src: torch.Tensor = pixel2cam(depth_src, self._pinhole_src.intrinsics_inverse(), pixel_coords)
        pixel_coords_src: torch.Tensor = cam2pixel(cam_coords_src, self._dst_proj_src)
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(self, depth_src: torch.Tensor, patch_dst: torch.Tensor) ->torch.Tensor:
        """Warps a tensor from destination frame to reference given the depth
        in the reference frame.

        Args:
            depth_src (torch.Tensor): the depth in the reference frame. The
              tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst (torch.Tensor): the patch in the destination frame. The
              tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            torch.Tensor: the warped patch from destination frame to reference.

        Shape:
            - Output: :math:`(N, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = kornia.PinholeCamera(...)
            >>> pinhole_src = kornia.PinholeCamera(...)
            >>> # create the depth warper, compute the projection matrix
            >>> warper = kornia.DepthWarper(pinhole_dst, height, width)
            >>> warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destionation frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
        """
        return F.grid_sample(patch_dst, self.warp_grid(depth_src), mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)


def warp_grid(grid: torch.Tensor, src_homo_dst: torch.Tensor) ->torch.Tensor:
    """Compute the grid to warp the coordinates grid by the homography/ies.

    Args:
        grid: Unwrapped grid of the shape :math:`(1, N, W, 2)`.
        src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.


    Returns:
        torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
    """
    batch_size: int = src_homo_dst.size(0)
    _, height, width, _ = grid.size()
    grid = grid.expand(batch_size, -1, -1, -1)
    if len(src_homo_dst.shape) == 3:
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)
    flow: torch.Tensor = transform_points(src_homo_dst, grid)
    return flow.view(batch_size, height, width, 2)


def homography_warp(patch_src: torch.Tensor, src_homo_dst: torch.Tensor, dsize: Tuple[int, int], mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, normalized_coordinates: bool=True) ->torch.Tensor:
    """Function that warps image patchs or tensors by homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, H, W)`.
        src_homo_dst (torch.Tensor): The homography or stack of homographies
                                     from destination to source of shape
                                     :math:`(N, 3, 3)`.
        dsize (Tuple[int, int]): The height and width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
        normalized_coordinates (bool): Whether the homography assumes [-1, 1] normalized
                                       coordinates or not.

    Return:
        torch.Tensor: Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = kornia.homography_warp(input, homography, (32, 32))
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError('Patch and homography must be on the same device.                          Got patch.device: {} src_H_dst.device: {}.'.format(patch_src.device, src_homo_dst.device))
    height, width = dsize
    grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)
    warped_grid = warp_grid(grid, src_homo_dst)
    return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


class HomographyWarper(nn.Module):
    """Warp tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\\{dst\\}} * X_{src}

    Args:
        height (int): The height of the destination tensor.
        width (int): The width of the destination tensor.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        normalized_coordinates (bool): wether to use a grid with
          normalized coordinates.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    """
    _warped_grid: Optional[torch.Tensor]

    def __init__(self, height: int, width: int, mode: str='bilinear', padding_mode: str='zeros', normalized_coordinates: bool=True, align_corners: bool=False) ->None:
        super(HomographyWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.normalized_coordinates: bool = normalized_coordinates
        self.align_corners: bool = align_corners
        self.grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)
        self._warped_grid = None

    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) ->None:
        """Compute and store internaly the transformations of the points.

        Useful when the same homography/homographies are reused.

        Args:
            src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.
              The homography assumes normalized coordinates [-1, 1] if
              normalized_coordinates is True.
         """
        self._warped_grid = warp_grid(self.grid, src_homo_dst)

    def forward(self, patch_src: torch.Tensor, src_homo_dst: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Warp a tensor from source into reference frame.

        Args:
            patch_src (torch.Tensor): The tensor to warp.
            src_homo_dst (torch.Tensor, optional): The homography or stack of
              homographies from destination to source. The homography assumes
              normalized coordinates [-1, 1] if normalized_coordinates is True.
              Default: None.

        Return:
            torch.Tensor: Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = kornia.HomographyWarper(32, 32)
            >>> # without precomputing the warp
            >>> output = warper(input, homography)  # NxCxHxW
            >>> # precomputing the warp
            >>> warper.precompute_warp_grid(homography)
            >>> output = warper(input)  # NxCxHxW
        """
        _warped_grid = self._warped_grid
        if src_homo_dst is not None:
            warped_patch = homography_warp(patch_src, src_homo_dst, (self.height, self.width), mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners, normalized_coordinates=self.normalized_coordinates)
        elif _warped_grid is not None:
            if not _warped_grid.device == patch_src.device:
                raise TypeError('Patch and warped grid must be on the same device.                                  Got patch.device: {} warped_grid.device: {}. Wheter                                  recall precompute_warp_grid() with the correct device                                  for the homograhy or change the patch device.'.format(patch_src.device, _warped_grid.device))
            warped_patch = F.grid_sample(patch_src, _warped_grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        else:
            raise RuntimeError('Unknown warping. If homographies are not provided                                 they must be presetted using the method:                                 precompute_warp_grid().')
        return warped_patch


def _gradient_x(img: torch.Tensor) ->torch.Tensor:
    assert len(img.shape) == 4, img.shape
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) ->torch.Tensor:
    assert len(img.shape) == 4, img.shape
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) ->torch.Tensor:
    """Computes image-aware inverse depth smoothness loss.

    See :class:`~kornia.losses.InvDepthSmoothnessLoss` for details.
    """
    if not torch.is_tensor(idepth):
        raise TypeError('Input idepth type is not a torch.Tensor. Got {}'.format(type(idepth)))
    if not torch.is_tensor(image):
        raise TypeError('Input image type is not a torch.Tensor. Got {}'.format(type(image)))
    if not len(idepth.shape) == 4:
        raise ValueError('Invalid idepth shape, we expect BxCxHxW. Got: {}'.format(idepth.shape))
    if not len(image.shape) == 4:
        raise ValueError('Invalid image shape, we expect BxCxHxW. Got: {}'.format(image.shape))
    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError('idepth and image shapes must be the same. Got: {} and {}'.format(idepth.shape, image.shape))
    if not idepth.device == image.device:
        raise ValueError('idepth and image must be in the same device. Got: {} and {}'.format(idepth.device, image.device))
    if not idepth.dtype == image.dtype:
        raise ValueError('idepth and image must be in the same dtype. Got: {} and {}'.format(idepth.dtype, image.dtype))
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class InverseDepthSmoothnessLoss(nn.Module):
    """Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \\text{loss} = \\left | \\partial_x d_{ij} \\right | e^{-\\left \\|
        \\partial_x I_{ij} \\right \\|} + \\left |
        \\partial_y d_{ij} \\right | e^{-\\left \\| \\partial_y I_{ij} \\right \\|}


    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples::

        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = kornia.losses.DepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)
    """

    def __init__(self) ->None:
        super(InverseDepthSmoothnessLoss, self).__init__()

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) ->torch.Tensor:
        return inverse_depth_smoothness_loss(idepth, image)


def one_hot(labels: torch.Tensor, num_classes: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None, eps: Optional[float]=1e-06) ->torch.Tensor:
    """Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError('Input labels type is not a torch.Tensor. Got {}'.format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError('labels must be of the same dtype torch.int64. Got: {}'.format(labels.dtype))
    if num_classes < 1:
        raise ValueError('The number of classes must be bigger than one. Got: {}'.format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float=1e-08) ->torch.Tensor:
    """Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxNxHxW. Got: {}'.format(input.shape))
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError('input and target shapes must be the same. Got: {} and {}'.format(input.shape, input.shape))
    if not input.device == target.device:
        raise ValueError('input and target must be in the same device. Got: {} and {}'.format(input.device, target.device))
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)
    dims = 1, 2, 3
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)
    dice_score = 2.0 * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.0)


class DiceLoss(nn.Module):
    """Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \\text{Dice}(x, class) = \\frac{2 |X| \\cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \\text{loss}(x, class) = 1 - \\text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = kornia.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) ->None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-06

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        return dice_loss(input, target, self.eps)


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float=2.0, reduction: str='none', eps: float=1e-08) ->torch.Tensor:
    """Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) >= 2:
        raise ValueError('Invalid input shape, we expect BxCx*. Got: {}'.format(input.shape))
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(input.size(0), target.size(0)))
    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))
    if not input.device == target.device:
        raise ValueError('input and target must be in the same device. Got: {} and {}'.format(input.device, target.device))
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)
    weight = torch.pow(-input_soft + 1.0, gamma)
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError('Invalid reduction mode: {}'.format(reduction))
    return loss


class FocalLoss(nn.Module):
    """Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma (float): Focusing parameter :math:`\\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float=2.0, reduction: str='none') ->None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-06

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) ->torch.Tensor:
    """Function that computes PSNR

    See :class:`~kornia.losses.PSNRLoss` for details.
    """
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f'Expected 2 torch tensors but got {type(input)} and {type(target)}')
    if input.shape != target.shape:
        raise TypeError(f'Expected tensors of equal shapes, but got {input.shape} and {target.shape}')
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input.device)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


class PSNRLoss(nn.Module):
    """Creates a criterion that calculates the PSNR between 2 images. Given an m x n image, the PSNR is:

    .. math::

        \\text{PSNR} = 10 \\log_{10} \\bigg(\\frac{\\text{MAX}_I^2}{MSE(I,T)}\\bigg)

    where

    .. math::

        \\text{MSE}(I,T) = \\frac{1}{mn}\\sum_{i=0}^{m-1}\\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\\text{MAX}_I=1`).


    Arguments:
        max_val (float): Maximum value of input

    Shape:
        - input: :math:`(*)`
        - approximation: :math:`(*)` same shape as input
        - output: :math:`()` a scalar

    Examples:
        >>> kornia.losses.psnr_loss(torch.ones(1), 1.2*torch.ones(1), 2)
        tensor(20.0000) # 10 * log(4/((1.2-1)**2)) / log(10)

    reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """

    def __init__(self, max_val: float) ->None:
        super(PSNRLoss, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        return psnr_loss(input, target, self.max_val)


class SSIM(nn.Module):
    """Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \\text{SSIM}(x, y) = \\frac{(2\\mu_x\\mu_y+c_1)(2\\sigma_{xy}+c_2)}
      {(\\mu_x^2+\\mu_y^2+c_1)(\\sigma_x^2+\\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\\#\\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \\text{loss}(x, y) = \\frac{1 - \\text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = kornia.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(self, window_size: int, reduction: str='none', max_val: float=1.0) ->None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction
        self.window: torch.Tensor = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)
        self.padding: int = _compute_zero_padding(window_size)
        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) ->torch.Tensor:
        if not torch.is_tensor(img1):
            raise TypeError('Input img1 type is not a torch.Tensor. Got {}'.format(type(img1)))
        if not torch.is_tensor(img2):
            raise TypeError('Input img2 type is not a torch.Tensor. Got {}'.format(type(img2)))
        if not len(img1.shape) == 4:
            raise ValueError('Invalid img1 shape, we expect BxCxHxW. Got: {}'.format(img1.shape))
        if not len(img2.shape) == 4:
            raise ValueError('Invalid img2 shape, we expect BxCxHxW. Got: {}'.format(img2.shape))
        if not img1.shape == img2.shape:
            raise ValueError('img1 and img2 shapes must be the same. Got: {} and {}'.format(img1.shape, img2.shape))
        if not img1.device == img2.device:
            raise ValueError('img1 and img2 must be in the same device. Got: {} and {}'.format(img1.device, img2.device))
        if not img1.dtype == img2.dtype:
            raise ValueError('img1 and img2 must be in the same dtype. Got: {} and {}'.format(img1.dtype, img2.dtype))
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2
        ssim_map = (2.0 * mu1_mu2 + self.C1) * (2.0 * sigma12 + self.C2) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        loss = torch.clamp(-ssim_map + 1.0, min=0, max=1) / 2.0
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss


def total_variation(img: torch.Tensor) ->torch.Tensor:
    """Function that computes Total Variation.

    See :class:`~kornia.losses.TotalVariation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(img)}')
    img_shape = img.shape
    if len(img_shape) == 3 or len(img_shape) == 4:
        pixel_dif1 = img[(...), 1:, :] - img[(...), :-1, :]
        pixel_dif2 = img[(...), :, 1:] - img[(...), :, :-1]
        reduce_axes = -3, -2, -1
    else:
        raise ValueError('Expected input tensor to be of ndim 3 or 4, but got ' + str(len(img_shape)))
    return pixel_dif1.abs().sum(dim=reduce_axes) + pixel_dif2.abs().sum(dim=reduce_axes)


class TotalVariation(nn.Module):
    """Computes the Total Variation according to
    [1] https://en.wikipedia.org/wiki/Total_variation
    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)` where C = number of classes.
        - Output: :math:`(N,)` or :math:`()`
    Examples:
        >>> kornia.losses.total_variation(torch.ones(3,4,4)) # tensor(0.)
        >>> tv = kornia.losses.TotalVariation()
        >>> output = tv(torch.ones(2,3,4,4)) # tensor([0., 0.])
        >>> output.backward()
    """

    def __init__(self) ->None:
        super(TotalVariation, self).__init__()

    def forward(self, img) ->torch.Tensor:
        return total_variation(img)


def tversky_loss(input: torch.Tensor, target: torch.Tensor, alpha: float, beta: float, eps: float=1e-08) ->torch.Tensor:
    """Function that computes Tversky loss.

    See :class:`~kornia.losses.TverskyLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxNxHxW. Got: {}'.format(input.shape))
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError('input and target shapes must be the same. Got: {} and {}'.format(input.shape, input.shape))
    if not input.device == target.device:
        raise ValueError('input and target must be in the same device. Got: {} and {}'.format(input.device, target.device))
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)
    dims = 1, 2, 3
    intersection = torch.sum(input_soft * target_one_hot, dims)
    fps = torch.sum(input_soft * (-target_one_hot + 1.0), dims)
    fns = torch.sum((-input_soft + 1.0) * target_one_hot, dims)
    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)
    return torch.mean(-tversky_loss + 1.0)


class TverskyLoss(nn.Module):
    """Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \\text{S}(P, G, \\alpha; \\beta) =
          \\frac{|PG|}{|PG| + \\alpha |P \\ G| + \\beta |G \\ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\\alpha` and :math:`\\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\\alpha = \\beta = 0.5` => dice coeff
       - :math:`\\alpha = \\beta = 1` => tanimoto coeff
       - :math:`\\alpha + \\beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha: float, beta: float, eps: float=1e-08) ->None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        return tversky_loss(input, target, self.alpha, self.beta, self.eps)


class TVDenoise(torch.nn.Module):

    def __init__(self, noisy_image):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BlobHessian,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BoxBlur,
     lambda: ([], {'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSoftArgmax2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CornerGFTT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ExtractTensorPatches,
     lambda: ([], {'window_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Hflip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvDepth,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([], {}),
     True),
    (InverseDepthSmoothnessLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PSNRLoss,
     lambda: ([], {'max_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PassLAF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PyrDown,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PyrUp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Resize,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RgbaToBgr,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RgbaToRgb,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Rot180,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SOSNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Sobel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialGradient,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TotalVariation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vflip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kornia_kornia(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

